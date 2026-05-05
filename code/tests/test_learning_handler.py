"""
unit/handlers/test_learning_handler.py — learning handler coverage.

Verifies the post-run flushes:
    1. BatchWriter.flush() called once
    2. Thompson state upserted for every (sku, model, config_hash)
    3. SKU similarity registry written for converged SKUs (mape <= warm_start_max)
    4. forecast_accuracy signal emitted with per-model aggregate MAPE
    5. data_fingerprint_cache upserted from preloaded.new_fingerprints
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_CODE = Path(__file__).resolve().parents[3]
for p in (str(_CODE), str(_CODE / "handlers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from handlers._context import RunContext, store, remove
from handlers.learning import learning_handler


class _RecordingCursor:
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
        self.cur = _RecordingCursor()
        self.commits = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _make_ctx(run_id, **kwargs):
    ctx = RunContext(tenant_id="t1", run_id=run_id)
    ctx.batch_writer = MagicMock()
    ctx.preloaded = MagicMock()
    ctx.preloaded.thompson_state = kwargs.get("thompson_state", {})
    ctx.preloaded.new_fingerprints = kwargs.get("new_fingerprints", {})
    ctx.preloaded.feature_history = kwargs.get("feature_history", {})
    ctx.preloaded.sku_tiers = kwargs.get("sku_tiers", {})
    ctx.sku_results = kwargs.get("sku_results", [])
    ctx.params = MagicMock()
    ctx.params.get.return_value = kwargs.get("warm_start_max_mape", 0.25)
    store(ctx)
    return ctx


def _sku_result(sku_id, mape, model="ses", used_fallback=False, pattern="stable"):
    r = MagicMock()
    r.sku_id = sku_id
    r.assigned_model = model
    r.backtest_mape = mape
    r.used_fallback = used_fallback
    r.pattern_label = pattern
    return r


class TestLearningHandler:

    def test_batch_writer_flushed_first(self):
        _ctx = _make_ctx("run-l-1")
        try:
            learning_handler(tenant_id="t1", run_id="run-l-1", db=_FakeConn())
            _ctx.batch_writer.flush.assert_called_once()
        finally:
            remove("run-l-1")

    def test_thompson_state_upserted_for_each_config(self):
        _make_ctx("run-l-2", thompson_state={
            ("sku-1", "ses"):  {"hash-A": {"alpha": 2.0, "beta": 1.0, "config": {"smoothing_level": 0.3}}},
            ("sku-2", "holt"): {"hash-B": {"alpha": 1.0, "beta": 2.0, "config": {"smoothing_level": 0.4}}},
        })
        try:
            conn = _FakeConn()
            learning_handler(tenant_id="t1", run_id="run-l-2", db=conn)
            thompson_inserts = [
                (sql, p) for sql, p in conn.cur.executed
                if "thompson_sampling_state" in sql
            ]
            assert len(thompson_inserts) == 2
        finally:
            remove("run-l-2")

    def test_no_thompson_writes_when_state_empty(self):
        _make_ctx("run-l-3", thompson_state={})
        try:
            conn = _FakeConn()
            learning_handler(tenant_id="t1", run_id="run-l-3", db=conn)
            assert not any(
                "thompson_sampling_state" in sql for sql, _ in conn.cur.executed
            )
        finally:
            remove("run-l-3")

    def test_similarity_registry_written_for_converged_skus(self):
        """SKUs with backtest_mape <= warm_start_max_mape should land in
        sku_similarity_registry. Above-threshold SKUs are filtered out."""
        _make_ctx("run-l-4",
                  warm_start_max_mape=0.25,
                  sku_results=[
                      _sku_result("sku-good", 0.10),  # below threshold
                      _sku_result("sku-bad",  0.50),  # above threshold
                      _sku_result("sku-edge", 0.25),  # exactly at threshold (<=)
                  ],
                  thompson_state={})
        try:
            conn = _FakeConn()
            with patch("handlers.learning._emit_cross_sku_signal"):
                learning_handler(tenant_id="t1", run_id="run-l-4", db=conn)
            sim_inserts = [
                (sql, p) for sql, p in conn.cur.executed
                if "sku_similarity_registry" in sql
            ]
            # Only sku-good and sku-edge qualify (≤ 0.25)
            assert len(sim_inserts) == 2

        finally:
            remove("run-l-4")

    def test_forecast_accuracy_signal_per_model(self):
        """One forecast_accuracy signal carrying per-model average MAPE."""
        _make_ctx("run-l-5", sku_results=[
            _sku_result("a", 0.10, model="ses"),
            _sku_result("b", 0.20, model="ses"),
            _sku_result("c", 0.05, model="holt"),
        ])
        try:
            conn = _FakeConn()
            learning_handler(tenant_id="t1", run_id="run-l-5", db=conn)
            sig_inserts = [
                (sql, p) for sql, p in conn.cur.executed
                if "cross_agent_signals" in sql and "expires_at" not in sql
            ]
            assert len(sig_inserts) == 1
        finally:
            remove("run-l-5")

    def test_no_accuracy_signal_when_all_fallback(self):
        """If every SKU was a fallback, there's no real MAPE to report —
        the signal must NOT fire."""
        _make_ctx("run-l-6", sku_results=[
            _sku_result("a", 0.30, used_fallback=True),
            _sku_result("b", 0.40, used_fallback=True),
        ])
        try:
            conn = _FakeConn()
            learning_handler(tenant_id="t1", run_id="run-l-6", db=conn)
            sig_inserts = [
                (sql, p) for sql, p in conn.cur.executed
                if "cross_agent_signals" in sql
            ]
            # No forecast_accuracy signal — only any cross_sku_learning ones
            # (which only fire on convergence; we have no converged SKUs here).
            assert sig_inserts == []
        finally:
            remove("run-l-6")

    def test_fingerprint_cache_upserted_per_sku(self):
        _make_ctx("run-l-7", new_fingerprints={
            "sku-a": {"fingerprint": "abc"},
            "sku-b": {"fingerprint": "def"},
        }, sku_tiers={"sku-a": "full", "sku-b": "full"})
        try:
            conn = _FakeConn()
            learning_handler(tenant_id="t1", run_id="run-l-7", db=conn)
            fp_inserts = [
                (sql, p) for sql, p in conn.cur.executed
                if "data_fingerprint_cache" in sql
            ]
            assert len(fp_inserts) == 2
        finally:
            remove("run-l-7")

    def test_total_trials_increments_on_thompson_write(self):
        """The new code passes (state.get('total_trials', 0) + 1) so the
        first write lands as 1, second as 2, etc."""
        _make_ctx("run-l-8", thompson_state={
            ("sku-1", "ses"): {
                "hash": {"alpha": 1.5, "beta": 1.5,
                         "config": {}, "total_trials": 0},
            },
        })
        try:
            conn = _FakeConn()
            learning_handler(tenant_id="t1", run_id="run-l-8", db=conn)
            row = next(p for sql, p in conn.cur.executed if "thompson_sampling_state" in sql)
            # row layout: (tenant, sku, model, hash, jsonb_cfg, alpha, beta, total_trials)
            assert row[7] == 1
        finally:
            remove("run-l-8")

