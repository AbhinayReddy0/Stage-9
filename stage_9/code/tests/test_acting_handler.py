"""
unit/handlers/test_acting_handler.py — acting handler coverage.

Acting handler is the biggest of the 6 (~600 lines). It has three execution
tracks — micro_update / cache / dual_pool — and a per-SKU fallback path.
This file tests the public surface that's safe to exercise without spinning
up real subprocess pools or DB:

    * REQUIRED_PRELOAD_KEYS contract
    * _CollectingBatchWriter behavior
    * _per_sku_fallback (P3 isolation) — appends an SKUResult, queues
      pattern_feedback, logs to stage9_sku_execution_log
    * _make_dual_pool_log_failure — wraps fallback + counter
    * _build_demand_df — synthesizes daily index from flat series
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[3]
for p in (str(_CODE), str(_CODE / "handlers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from handlers._context import RunContext, store, remove
from handlers.acting import (
    REQUIRED_PRELOAD_KEYS,
    _CollectingBatchWriter,
    _build_demand_df,
    _per_sku_fallback,
    _make_dual_pool_log_failure,
)
from infrastructure.constants import (
    ForecastStatus, Model, Param, ProcessingTier,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

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


def _make_ctx(run_id):
    """Minimal RunContext for fallback / log-failure tests."""
    ctx = RunContext(tenant_id="t1", run_id=run_id)
    ctx.params = MagicMock()
    ctx.params.get.return_value = 0.30   # CONFIDENCE_FLOOR
    ctx.preloaded = MagicMock()
    ctx.preloaded.pattern_ctx = {
        "sku-1": {"pattern_label": "stable", "stage8_confidence": 0.85},
    }
    ctx.preloaded.sku_tiers = {"sku-1": ProcessingTier.FULL}
    ctx.batch_writer = MagicMock()
    ctx.sku_results = []
    ctx.pattern_feedback_failures_count = 0
    store(ctx)
    return ctx


# ---------------------------------------------------------------------------
# REQUIRED_PRELOAD_KEYS contract
# ---------------------------------------------------------------------------

class TestRequiredPreloadKeysContract:

    def test_is_a_frozen_set(self):
        assert isinstance(REQUIRED_PRELOAD_KEYS, frozenset)

    def test_contains_24_required_keys(self):
        # Sanity check — if the contract grows, the test will fail loudly so
        # the operator notices the contract change.
        assert len(REQUIRED_PRELOAD_KEYS) == 24

    def test_includes_demand_series(self):
        assert "demand_series" in REQUIRED_PRELOAD_KEYS

    def test_includes_thompson_state(self):
        assert "thompson_state" in REQUIRED_PRELOAD_KEYS

    def test_does_not_include_tenant_wide_keys(self):
        """tenant_params / feature_reliability are tenant-wide and stashed via
        worker-globals, not per-SKU. They must NOT appear in the per-SKU
        REQUIRED_PRELOAD_KEYS contract."""
        for tenant_key in ("tenant_params", "feature_reliability"):
            assert tenant_key not in REQUIRED_PRELOAD_KEYS


# ---------------------------------------------------------------------------
# _CollectingBatchWriter
# ---------------------------------------------------------------------------

class TestCollectingBatchWriter:

    def test_queue_accumulates(self):
        bw = _CollectingBatchWriter()
        bw.queue("forecasts", {"sku_id": "x"})
        bw.queue("forecasts", {"sku_id": "y"})
        assert bw.count == 2
        assert len(bw.buffer["forecasts"]) == 2

    def test_drain_returns_and_resets(self):
        bw = _CollectingBatchWriter()
        bw.queue("forecasts", {"sku_id": "x"})
        bw.queue("backtest_decisions", {"sku_id": "y"})
        out = bw.drain()
        assert out == {
            "forecasts":           [{"sku_id": "x"}],
            "backtest_decisions":  [{"sku_id": "y"}],
        }
        assert bw.count == 0
        assert dict(bw.buffer) == {}

    def test_flush_is_noop(self):
        bw = _CollectingBatchWriter()
        bw.queue("forecasts", {"k": 1})
        bw.flush()
        # Buffer still intact — flush() doesn't drain or write to anything
        assert bw.count == 1

    def test_flush_if_needed_is_noop(self):
        bw = _CollectingBatchWriter()
        bw.queue("forecasts", {"k": 1})
        bw.flush_if_needed()
        assert bw.count == 1   # never auto-flushes — batch_size = 10**9

    def test_drain_skips_empty_table_keys(self):
        bw = _CollectingBatchWriter()
        bw.buffer["empty"]   # touch — defaultdict creates an empty list
        bw.queue("forecasts", {"k": 1})
        out = bw.drain()
        assert "empty" not in out
        assert "forecasts" in out


# ---------------------------------------------------------------------------
# _build_demand_df
# ---------------------------------------------------------------------------

class TestBuildDemandDf:

    def test_returns_dataframe_with_date_and_qty_columns(self):
        df = _build_demand_df([1.0, 2.0, 3.0], {})
        assert list(df.columns) == ["date", "qty"]

    def test_length_matches_series(self):
        df = _build_demand_df([1.0] * 30, {})
        assert len(df) == 30

    def test_dates_are_consecutive_daily(self):
        df = _build_demand_df([1.0] * 7, {})
        diffs = df["date"].diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == pd.Timedelta(days=1)

    def test_empty_series_yields_empty_df(self):
        df = _build_demand_df([], {})
        assert len(df) == 0
        assert list(df.columns) == ["date", "qty"]


# ---------------------------------------------------------------------------
# _per_sku_fallback (Principle 3 isolation)
# ---------------------------------------------------------------------------

class TestPerSkuFallback:

    def test_appends_naive_sku_result(self):
        ctx = _make_ctx("run-act-1")
        try:
            with patch("handlers.acting.write_pattern_feedback", return_value=True):
                _per_sku_fallback(ctx, _FakeConn(), "sku-1",
                                  sub_stage="acting", reason="explode")
            assert len(ctx.sku_results) == 1
            r = ctx.sku_results[0]
            assert r.sku_id == "sku-1"
            assert r.status == ForecastStatus.NEEDS_ACKNOWLEDGMENT
            assert r.assigned_model == Model.NAIVE
            assert r.used_fallback is True
            assert r.confidence_final == 0.30   # confidence_floor from mock_params
        finally:
            remove("run-act-1")

    def test_pattern_feedback_failure_increments_counter(self):
        ctx = _make_ctx("run-act-2")
        try:
            with patch("handlers.acting.write_pattern_feedback", return_value=False):
                _per_sku_fallback(ctx, _FakeConn(), "sku-1",
                                  sub_stage="acting", reason="x")
            assert ctx.pattern_feedback_failures_count == 1
        finally:
            remove("run-act-2")

    def test_pattern_feedback_exception_caught_and_counted(self):
        ctx = _make_ctx("run-act-3")
        try:
            with patch("handlers.acting.write_pattern_feedback",
                       side_effect=RuntimeError("db down")):
                # Must NOT raise — fallback path is robust
                _per_sku_fallback(ctx, _FakeConn(), "sku-1",
                                  sub_stage="acting", reason="x")
            assert ctx.pattern_feedback_failures_count == 1
        finally:
            remove("run-act-3")

    def test_writes_to_stage9_sku_execution_log(self):
        ctx = _make_ctx("run-act-4")
        try:
            conn = _FakeConn()
            with patch("handlers.acting.write_pattern_feedback", return_value=True):
                _per_sku_fallback(ctx, conn, "sku-1",
                                  sub_stage="acting",
                                  reason="explode at hp tuning")
            log_inserts = [
                p for sql, p in conn.cur.executed
                if "stage9_sku_execution_log" in sql
            ]
            assert len(log_inserts) == 1
            # row tuple shape: (tenant, run, sku, status, model, code, reason, sub_stage, ms)
            row = log_inserts[0]
            assert row[0] == "t1"
            assert row[2] == "sku-1"
            assert row[3] == "fallback"
            assert row[4] == Model.NAIVE
            assert row[6] == "explode at hp tuning"
            assert row[7] == "acting"
        finally:
            remove("run-act-4")

    def test_long_reason_truncated_to_1000(self):
        ctx = _make_ctx("run-act-5")
        try:
            conn = _FakeConn()
            long_reason = "x" * 5000
            with patch("handlers.acting.write_pattern_feedback", return_value=True):
                _per_sku_fallback(ctx, conn, "sku-1",
                                  sub_stage="acting", reason=long_reason)
            log_row = next(
                p for sql, p in conn.cur.executed
                if "stage9_sku_execution_log" in sql
            )
            assert len(log_row[6]) == 1000
        finally:
            remove("run-act-5")


# ---------------------------------------------------------------------------
# _make_dual_pool_log_failure
# ---------------------------------------------------------------------------

class TestDualPoolLogFailure:

    def test_pattern_feedback_reason_increments_counter(self):
        ctx = _make_ctx("run-act-6")
        try:
            cb = _make_dual_pool_log_failure(ctx, _FakeConn())
            with patch("handlers.acting.write_pattern_feedback", return_value=True):
                cb("t1", "run-act-6", "sku-1",
                   "pattern_feedback retry exhausted")
            assert ctx.pattern_feedback_failures_count == 1
        finally:
            remove("run-act-6")

    def test_non_pattern_feedback_reason_does_not_increment(self):
        ctx = _make_ctx("run-act-7")
        try:
            cb = _make_dual_pool_log_failure(ctx, _FakeConn())
            with patch("handlers.acting.write_pattern_feedback", return_value=True):
                cb("t1", "run-act-7", "sku-1", "model fit error")
            assert ctx.pattern_feedback_failures_count == 0
        finally:
            remove("run-act-7")

    def test_swallows_fallback_exceptions(self):
        ctx = _make_ctx("run-act-8")
        try:
            cb = _make_dual_pool_log_failure(ctx, _FakeConn())
            with patch("handlers.acting._per_sku_fallback",
                       side_effect=RuntimeError("fallback broke")):
                # MUST NOT raise — dual_pool callback is best-effort.
                cb("t1", "run-act-8", "sku-1", "any reason")
        finally:
            remove("run-act-8")
