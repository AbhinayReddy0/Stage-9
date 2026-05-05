"""
unit/handlers/test_preloading_handler.py — preloading handler coverage.

Two units under test:
    * _resolve_execution_mode — pure function, easy to test directly
    * preloading_handler       — orchestrator; verify it stores RunContext
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

_CODE = Path(__file__).resolve().parents[3]
for p in (str(_CODE), str(_CODE / "handlers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from handlers._context import fetch, remove
from handlers.preloading import _resolve_execution_mode, preloading_handler
from infrastructure.constants import ExecutionMode, Param


class _FakeCursor:
    def __init__(self, row=None):
        self._row = row

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, last_complete_at=None):
        self._row = (last_complete_at,) if last_complete_at else None

    def cursor(self):
        return _FakeCursor(self._row)


class _FakeParams:
    def __init__(self, threshold_hours=18.0):
        self._t = threshold_hours

    def get(self, key):
        if key == Param.MICRO_UPDATE_THRESHOLD_HOURS:
            return self._t
        return None


# ---------------------------------------------------------------------------
# _resolve_execution_mode
# ---------------------------------------------------------------------------

class TestResolveExecutionMode:

    def test_first_run_returns_full(self):
        """No prior COMPLETE row → always FULL."""
        conn = _FakeConn(last_complete_at=None)
        mode = _resolve_execution_mode("t1", conn, _FakeParams())
        assert mode == ExecutionMode.FULL

    def test_recent_run_returns_micro_update(self):
        """Last full < threshold hours ago → MICRO_UPDATE."""
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        mode = _resolve_execution_mode("t1", _FakeConn(recent),
                                       _FakeParams(threshold_hours=18.0))
        assert mode == ExecutionMode.MICRO_UPDATE

    def test_old_run_returns_full(self):
        """Last full > threshold hours ago → FULL."""
        old = datetime.now(timezone.utc) - timedelta(hours=24)
        mode = _resolve_execution_mode("t1", _FakeConn(old),
                                       _FakeParams(threshold_hours=18.0))
        assert mode == ExecutionMode.FULL

    def test_threshold_boundary_rounds_to_full(self):
        """At exactly threshold: spec says >= threshold → FULL."""
        boundary = datetime.now(timezone.utc) - timedelta(hours=18, seconds=1)
        mode = _resolve_execution_mode("t1", _FakeConn(boundary),
                                       _FakeParams(threshold_hours=18.0))
        assert mode == ExecutionMode.FULL

    def test_naive_timestamp_treated_as_utc(self):
        """A datetime with tzinfo=None is treated as UTC, not raised."""
        naive_old = (datetime.now(timezone.utc) - timedelta(hours=20)).replace(tzinfo=None)
        mode = _resolve_execution_mode("t1", _FakeConn(naive_old),
                                       _FakeParams(threshold_hours=18.0))
        assert mode == ExecutionMode.FULL

    def test_default_threshold_when_param_missing_or_invalid(self):
        """Param read failure → default 18.0h threshold."""
        bad_params = MagicMock()
        bad_params.get.side_effect = RuntimeError("no such param")
        # Recent run still produces MICRO_UPDATE under the 18h default
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        mode = _resolve_execution_mode("t1", _FakeConn(recent), bad_params)
        assert mode == ExecutionMode.MICRO_UPDATE


# ---------------------------------------------------------------------------
# preloading_handler — wires Preloader + signal consumer + RunContext
# ---------------------------------------------------------------------------

class TestPreloadingHandler:

    def test_runcontext_registered_after_handler(self):
        fake_preloaded = MagicMock()
        fake_preloaded.pattern_ctx = {"sku-1": {}, "sku-2": {}}
        fake_preloader = MagicMock()
        fake_preloader.load.return_value = fake_preloaded
        fake_preloader.params = _FakeParams()

        with patch("handlers.preloading.Preloader", return_value=fake_preloader), \
             patch("handlers.preloading.SignalConsumer", return_value=MagicMock()), \
             patch("handlers.preloading.BatchWriter", return_value=MagicMock()), \
             patch("handlers.preloading._resolve_execution_mode",
                   return_value=ExecutionMode.FULL):
            preloading_handler(tenant_id="t1", run_id="run-pl-1", db=MagicMock())
        try:
            ctx = fetch("run-pl-1")
            assert ctx.tenant_id == "t1"
            assert ctx.execution_mode == ExecutionMode.FULL
            assert sorted(ctx.sku_ids) == ["sku-1", "sku-2"]
        finally:
            remove("run-pl-1")

    def test_handler_uses_resolved_mode(self):
        fake_preloaded = MagicMock()
        fake_preloaded.pattern_ctx = {}
        fake_preloader = MagicMock()
        fake_preloader.load.return_value = fake_preloaded
        fake_preloader.params = _FakeParams()

        with patch("handlers.preloading.Preloader", return_value=fake_preloader), \
             patch("handlers.preloading.SignalConsumer"), \
             patch("handlers.preloading.BatchWriter"), \
             patch("handlers.preloading._resolve_execution_mode",
                   return_value=ExecutionMode.MICRO_UPDATE):
            preloading_handler(tenant_id="t1", run_id="run-pl-2", db=MagicMock())
        try:
            assert fetch("run-pl-2").execution_mode == ExecutionMode.MICRO_UPDATE
        finally:
            remove("run-pl-2")

    def test_empty_preloaded_yields_empty_sku_ids(self):
        fake_preloaded = MagicMock()
        fake_preloaded.pattern_ctx = {}
        fake_preloader = MagicMock()
        fake_preloader.load.return_value = fake_preloaded
        fake_preloader.params = _FakeParams()

        with patch("handlers.preloading.Preloader", return_value=fake_preloader), \
             patch("handlers.preloading.SignalConsumer"), \
             patch("handlers.preloading.BatchWriter"), \
             patch("handlers.preloading._resolve_execution_mode",
                   return_value=ExecutionMode.FULL):
            preloading_handler(tenant_id="t1", run_id="run-pl-3", db=MagicMock())
        try:
            assert fetch("run-pl-3").sku_ids == []
        finally:
            remove("run-pl-3")
