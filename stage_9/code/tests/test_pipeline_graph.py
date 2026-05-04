"""
unit/orchestration/test_pipeline_graph.py — pipeline_graph.py coverage.

Production module is the OUTER LangGraph that watches `runs.status` across
tenants and dispatches to the appropriate stage. Tests:

    * watch node returns pending runs from the SQL it issues
    * dispatch routes by status to the correct downstream agent
    * dispatch isolation: one bad run never stops the rest
    * build_pipeline_graph wires watch → dispatch → END

`run_forever` is excluded — it's an infinite poll loop with sleeps. Its
underlying `graph.invoke({})` is exercised here directly.
"""
from __future__ import annotations

from typing import Any

import pytest

# pipeline_graph imports `langgraph` at module load. Skip the whole file
# (rather than crash collection) when it's not installed.
try:
    import pipeline_graph
    from pipeline_graph import build_pipeline_graph
except ImportError as _exc:
    pytest.skip(
        f"langgraph not installed — {_exc}. Install it (`pip install langgraph`) "
        f"to exercise the outer pipeline graph.",
        allow_module_level=True,
    )

from constants import RunStatus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rows=()) -> None:
        self.rows = list(rows)
        self.executed: list[tuple[str, Any]] = []
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def close(self):
        self.closed = True


class _FakeConn:
    def __init__(self, rows=()) -> None:
        self._rows = rows
        self.last_cursor: _FakeCursor | None = None

    def cursor(self):
        self.last_cursor = _FakeCursor(self._rows)
        return self.last_cursor


# ---------------------------------------------------------------------------
# watch node
# ---------------------------------------------------------------------------

class TestWatchNode:

    def test_returns_zero_pending_when_no_rows(self):
        conn = _FakeConn(rows=[])
        watch = pipeline_graph._make_watch_node(conn)
        out = watch({})
        assert out == {"pending_runs": []}

    def test_returns_rows_as_dicts(self):
        rows = [
            ("t1", "r1", RunStatus.DATA_READY),
            ("t2", "r2", RunStatus.PATTERNS_DISCOVERED),
            ("t3", "r3", RunStatus.FORECASTED),
        ]
        watch = pipeline_graph._make_watch_node(_FakeConn(rows=rows))
        out = watch({})
        assert len(out["pending_runs"]) == 3
        assert out["pending_runs"][0] == {
            "tenant_id": "t1", "run_id": "r1", "status": RunStatus.DATA_READY,
        }

    def test_query_filters_three_target_statuses(self):
        conn = _FakeConn()
        pipeline_graph._make_watch_node(conn)({})
        sql, params = conn.last_cursor.executed[0]
        assert "FROM runs" in sql
        assert "ORDER BY updated_at ASC" in sql
        assert "LIMIT 50" in sql
        # The three statuses we route to stage_8 / 9 / 10
        assert RunStatus.DATA_READY            in params
        assert RunStatus.PATTERNS_DISCOVERED   in params
        assert RunStatus.FORECASTED            in params

    def test_cursor_is_closed_after_query(self):
        conn = _FakeConn(rows=[])
        pipeline_graph._make_watch_node(conn)({})
        assert conn.last_cursor.closed is True


# ---------------------------------------------------------------------------
# dispatch node
# ---------------------------------------------------------------------------

class TestDispatchNode:

    def test_patterns_discovered_calls_stage9_start_run(self, monkeypatch):
        called: list[tuple] = []

        def fake_start_run(tenant_id, run_id, conn):
            called.append((tenant_id, run_id, conn))

        monkeypatch.setattr(pipeline_graph.stage9, "start_run", fake_start_run)

        conn = _FakeConn()
        dispatch = pipeline_graph._make_dispatch_node(conn)
        out = dispatch({"pending_runs": [
            {"tenant_id": "t1", "run_id": "r1",
             "status": RunStatus.PATTERNS_DISCOVERED},
        ]})
        assert called == [("t1", "r1", conn)]
        assert out == {"pending_runs": []}

    def test_data_ready_does_not_call_stage9(self, monkeypatch):
        """DATA_READY routes to stage_8 (not in scope) — must NOT call stage9."""
        calls: list = []
        monkeypatch.setattr(pipeline_graph.stage9, "start_run",
                            lambda *a, **k: calls.append(a))

        dispatch = pipeline_graph._make_dispatch_node(_FakeConn())
        dispatch({"pending_runs": [
            {"tenant_id": "t1", "run_id": "r1", "status": RunStatus.DATA_READY},
        ]})
        assert calls == []

    def test_forecasted_does_not_call_stage9(self, monkeypatch):
        """FORECASTED routes to stage_10 — must NOT trigger stage9."""
        calls: list = []
        monkeypatch.setattr(pipeline_graph.stage9, "start_run",
                            lambda *a, **k: calls.append(a))

        dispatch = pipeline_graph._make_dispatch_node(_FakeConn())
        dispatch({"pending_runs": [
            {"tenant_id": "t1", "run_id": "r1", "status": RunStatus.FORECASTED},
        ]})
        assert calls == []

    def test_one_failing_run_does_not_stop_others(self, monkeypatch):
        """Per-run isolation: stage9.start_run raising on run #1 must NOT
        prevent run #2 from being attempted."""
        attempts: list = []

        def fake_start_run(tenant_id, run_id, conn):
            attempts.append(run_id)
            if run_id == "r1":
                raise RuntimeError("explode")

        monkeypatch.setattr(pipeline_graph.stage9, "start_run", fake_start_run)
        dispatch = pipeline_graph._make_dispatch_node(_FakeConn())
        dispatch({"pending_runs": [
            {"tenant_id": "t1", "run_id": "r1", "status": RunStatus.PATTERNS_DISCOVERED},
            {"tenant_id": "t1", "run_id": "r2", "status": RunStatus.PATTERNS_DISCOVERED},
        ]})
        assert attempts == ["r1", "r2"]

    def test_unknown_status_is_silently_skipped(self, monkeypatch):
        """A status that's not one of the three target values is just ignored."""
        calls: list = []
        monkeypatch.setattr(pipeline_graph.stage9, "start_run",
                            lambda *a, **k: calls.append(a))

        dispatch = pipeline_graph._make_dispatch_node(_FakeConn())
        out = dispatch({"pending_runs": [
            {"tenant_id": "t1", "run_id": "r1", "status": "totally_unknown"},
        ]})
        assert calls == []
        assert out == {"pending_runs": []}

    def test_empty_pending_returns_clean_state(self):
        out = pipeline_graph._make_dispatch_node(_FakeConn())({"pending_runs": []})
        assert out == {"pending_runs": []}

    def test_missing_pending_runs_key_handled(self):
        """state.get with default ensures absence doesn't crash."""
        out = pipeline_graph._make_dispatch_node(_FakeConn())({})
        assert out == {"pending_runs": []}


# ---------------------------------------------------------------------------
# build_pipeline_graph
# ---------------------------------------------------------------------------

class TestBuildPipelineGraph:

    def test_compiles_without_error(self):
        graph = build_pipeline_graph(_FakeConn(rows=[]))
        assert graph is not None

    def test_invoke_runs_watch_then_dispatch(self, monkeypatch):
        """End-to-end smoke: invoking the compiled graph executes both nodes
        in the correct order."""
        invoked: list = []
        monkeypatch.setattr(
            pipeline_graph.stage9, "start_run",
            lambda t, r, c: invoked.append(("dispatch", r)),
        )

        rows = [("t1", "r1", RunStatus.PATTERNS_DISCOVERED)]
        graph = build_pipeline_graph(_FakeConn(rows=rows))
        result = graph.invoke({})
        # dispatch ran with the row that watch produced
        assert invoked == [("dispatch", "r1")]
        # pending_runs is reset at the end
        assert result.get("pending_runs") == []
