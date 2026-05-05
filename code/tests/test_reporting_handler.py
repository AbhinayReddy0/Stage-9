"""
unit/handlers/test_reporting_handler.py — reporting handler coverage.

Reporting handler responsibilities:
    1. Run SelfAssessmentEngine; store result row
    2. Emit model_health signal to Stage 8 + Stage 10
    3. UPDATE stage8.runs.status to 'forecasted' or 'needs_acknowledgment'
    4. Remove RunContext from registry
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_CODE = Path(__file__).resolve().parents[3]
for p in (str(_CODE), str(_CODE / "handlers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from handlers._context import RunContext, store, fetch
from handlers.reporting import reporting_handler, _set_run_status
from infrastructure.constants import Agent, ForecastStatus, RunStatus, SignalType


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


def _make_ctx(run_id, sku_results=()):
    """Build a RunContext with reasonable defaults + register it."""
    ctx = RunContext(tenant_id="t1", run_id=run_id)
    ctx.run_start_time = 100.0
    ctx.execution_mode = "full"
    ctx.sku_results = list(sku_results)
    ctx.pattern_feedback_failures_count = 0
    store(ctx)
    return ctx


def _sku_result(sku_id, status):
    """Minimal SKUResult-compatible object for reporting_handler."""
    r = MagicMock()
    r.sku_id = sku_id
    r.status = status
    r.confidence_final = 0.8
    r.assigned_model = "ses"
    r.processing_tier = "full"
    r.used_fallback = False
    r.pattern_label = "stable"
    r.backtest_mape = 0.05
    return r


@pytest.fixture
def fake_self_assessment():
    """Stub the SelfAssessmentEngine.run output."""
    result = MagicMock()
    result.degradation_detected = False
    result.high_fallback_alert = False
    result.pct_auto_proceed = 1.0
    result.run_duration_seconds = 1.5
    result.total_skus_processed = 1
    result.model_health = {}
    result.recommendations = []
    return result


class TestReportingHandler:

    def test_status_forecasted_when_all_skus_clean(self, fake_self_assessment):
        _make_ctx("run-rep-1", [_sku_result("a", ForecastStatus.FORECASTED)])
        conn = _FakeConn()
        with patch("handlers.reporting.SelfAssessmentEngine") as Eng:
            Eng.return_value.run.return_value = fake_self_assessment
            reporting_handler(tenant_id="t1", run_id="run-rep-1", db=conn)
        # The runs UPDATE happened with FORECASTED
        runs_updates = [
            (sql, p) for sql, p in conn.cur.executed
            if "UPDATE stage8.runs" in sql
        ]
        assert len(runs_updates) == 1
        assert runs_updates[0][1][0] == RunStatus.FORECASTED

    def test_status_needs_ack_when_any_sku_flagged(self, fake_self_assessment):
        _make_ctx("run-rep-2", [
            _sku_result("a", ForecastStatus.FORECASTED),
            _sku_result("b", ForecastStatus.NEEDS_ACKNOWLEDGMENT),
        ])
        conn = _FakeConn()
        with patch("handlers.reporting.SelfAssessmentEngine") as Eng:
            Eng.return_value.run.return_value = fake_self_assessment
            reporting_handler(tenant_id="t1", run_id="run-rep-2", db=conn)
        runs_updates = [
            (sql, p) for sql, p in conn.cur.executed
            if "UPDATE stage8.runs" in sql
        ]
        assert runs_updates[0][1][0] == RunStatus.NEEDS_ACKNOWLEDGMENT

    def test_status_needs_ack_for_watchlist_review(self, fake_self_assessment):
        _make_ctx("run-rep-3", [
            _sku_result("a", ForecastStatus.WATCHLIST_REVIEW),
        ])
        conn = _FakeConn()
        with patch("handlers.reporting.SelfAssessmentEngine") as Eng:
            Eng.return_value.run.return_value = fake_self_assessment
            reporting_handler(tenant_id="t1", run_id="run-rep-3", db=conn)
        runs_updates = [
            (sql, p) for sql, p in conn.cur.executed
            if "UPDATE stage8.runs" in sql
        ]
        assert runs_updates[0][1][0] == RunStatus.NEEDS_ACKNOWLEDGMENT

    def test_emits_model_health_signal_to_stage_8_and_stage_10(self, fake_self_assessment):
        _make_ctx("run-rep-4", [_sku_result("a", ForecastStatus.FORECASTED)])
        conn = _FakeConn()
        with patch("handlers.reporting.SelfAssessmentEngine") as Eng:
            Eng.return_value.run.return_value = fake_self_assessment
            reporting_handler(tenant_id="t1", run_id="run-rep-4", db=conn)
        signal_inserts = [
            (sql, p) for sql, p in conn.cur.executed
            if "cross_agent_signals" in sql
        ]
        assert len(signal_inserts) == 2
        recipients = {p[3] for _, p in signal_inserts}
        assert recipients == {Agent.STAGE_8, Agent.STAGE_10}
        # All have signal_type = MODEL_HEALTH
        for _, p in signal_inserts:
            assert p[4] == SignalType.MODEL_HEALTH

    def test_runcontext_removed_after_run(self, fake_self_assessment):
        _make_ctx("run-rep-5", [_sku_result("a", ForecastStatus.FORECASTED)])
        conn = _FakeConn()
        with patch("handlers.reporting.SelfAssessmentEngine") as Eng:
            Eng.return_value.run.return_value = fake_self_assessment
            reporting_handler(tenant_id="t1", run_id="run-rep-5", db=conn)
        # After reporting, fetch() must raise — context was removed
        with pytest.raises(KeyError):
            fetch("run-rep-5")

    def test_self_assessment_engine_called_with_ctx_data(self, fake_self_assessment):
        _make_ctx("run-rep-6", [_sku_result("a", ForecastStatus.FORECASTED)])
        conn = _FakeConn()
        with patch("handlers.reporting.SelfAssessmentEngine") as Eng:
            Eng.return_value.run.return_value = fake_self_assessment
            reporting_handler(tenant_id="t1", run_id="run-rep-6", db=conn)
        Eng.return_value.run.assert_called_once()
        kwargs = Eng.return_value.run.call_args.kwargs
        assert kwargs["tenant_id"] == "t1"
        assert kwargs["run_id"] == "run-rep-6"
        assert kwargs["execution_mode"] == "full"


class TestSetRunStatus:

    def test_swallow_db_error(self):
        """_set_run_status must NOT propagate exceptions — stage8.runs may
        not exist in dev environments. agent_state_log_s9 is the
        authoritative terminal-state record anyway."""
        bad_conn = MagicMock()
        bad_conn.cursor.side_effect = RuntimeError("stage8.runs missing")
        # Should NOT raise
        _set_run_status(bad_conn, "t1", "r1", RunStatus.FORECASTED)

    def test_commits_on_success(self):
        conn = _FakeConn()
        _set_run_status(conn, "t1", "r1", RunStatus.FORECASTED)
        assert conn.commits == 1

