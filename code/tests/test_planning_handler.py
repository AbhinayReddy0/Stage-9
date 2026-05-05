"""
unit/handlers/test_planning_handler.py — planning handler coverage.

Verifies the per-SKU bookkeeping the handler does:
    * Cache-tier SKUs go to ctx.cache_sku_ids
    * Workable (FULL/PARTIAL) SKUs go to ctx.pipeline_inputs
    * Calibration caches are populated on ctx
    * Demand history is loaded once and stashed on ctx
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
_CODE = Path(__file__).resolve().parents[3]
for p in (str(_CODE), str(_CODE / "handlers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from handlers._context import RunContext, store, remove
from handlers.planning import planning_handler
from infrastructure.constants import ProcessingTier


def _ctx(run_id, sku_tiers):
    """Register a RunContext with the given sku_id → tier map."""
    ctx = RunContext(tenant_id="t1", run_id=run_id)
    ctx.preloaded = MagicMock()
    ctx.preloaded.sku_tiers = sku_tiers
    ctx.preloaded.pattern_ctx = {sid: {} for sid in sku_tiers}
    ctx.params = MagicMock()
    ctx.batch_writer = MagicMock()
    ctx.signal_consumer = MagicMock()
    ctx.sku_ids = list(sku_tiers.keys())
    store(ctx)
    return ctx


def _fake_demand_data(sku_ids):
    """Return a {sku_id: DataFrame} for each SKU with non-empty demand."""
    return {
        sku: pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=120),
            "qty":  [5.0] * 120,
        })
        for sku in sku_ids
    }


class TestPlanningHandler:

    def test_cache_tier_skus_skip_pipeline_input(self):
        """Cache-tier SKUs accumulate in ctx.cache_sku_ids; FULL ones go
        through 9.1 + build_sku_pipeline_input."""
        ctx = _ctx("run-pl-1", {
            "sku-cache":  ProcessingTier.CACHE,
            "sku-full":   ProcessingTier.FULL,
        })
        try:
            with patch("handlers.planning.prefetch_calibrated_windows", return_value={}), \
                 patch("handlers.planning.prefetch_calibration_gaps", return_value={}), \
                 patch("handlers.planning._load_demand_history",
                       return_value=_fake_demand_data(["sku-cache", "sku-full"])), \
                 patch("handlers.planning.run_model_initialisation",
                       return_value=MagicMock(pattern_label="stable",
                                              assigned_model="ses")), \
                 patch("handlers.acting.build_sku_pipeline_input",
                       return_value=MagicMock(sku_id="sku-full")):
                planning_handler(tenant_id="t1", run_id="run-pl-1", db=MagicMock())
            assert ctx.cache_sku_ids == ["sku-cache"]
            assert len(ctx.pipeline_inputs) == 1
        finally:
            remove("run-pl-1")

    def test_no_demand_skus_are_skipped(self):
        """SKUs with empty demand history must NOT be added to either list."""
        ctx = _ctx("run-pl-2", {"sku-empty": ProcessingTier.FULL})
        try:
            with patch("handlers.planning.prefetch_calibrated_windows", return_value={}), \
                 patch("handlers.planning.prefetch_calibration_gaps", return_value={}), \
                 patch("handlers.planning._load_demand_history",
                       return_value={"sku-empty": pd.DataFrame({"date": [], "qty": []})}), \
                 patch("handlers.planning.run_model_initialisation",
                       return_value=MagicMock()):
                planning_handler(tenant_id="t1", run_id="run-pl-2", db=MagicMock())
            assert ctx.pipeline_inputs == []
            assert ctx.cache_sku_ids == []
        finally:
            remove("run-pl-2")

    def test_calibration_caches_are_populated(self):
        ctx = _ctx("run-pl-3", {})
        try:
            cal_windows = {("stable", "ses"): 60}
            cal_gaps    = {("stable", "ses"): 0.10}
            with patch("handlers.planning.prefetch_calibrated_windows",
                       return_value=cal_windows), \
                 patch("handlers.planning.prefetch_calibration_gaps",
                       return_value=cal_gaps), \
                 patch("handlers.planning._load_demand_history", return_value={}):
                planning_handler(tenant_id="t1", run_id="run-pl-3", db=MagicMock())
            assert ctx.calibrated_cache == cal_windows
            assert ctx.calibration_gaps == cal_gaps
        finally:
            remove("run-pl-3")

    def test_demand_data_stashed_on_ctx(self):
        ctx = _ctx("run-pl-4", {})
        try:
            demand = _fake_demand_data(["sku-x"])
            with patch("handlers.planning.prefetch_calibrated_windows", return_value={}), \
                 patch("handlers.planning.prefetch_calibration_gaps", return_value={}), \
                 patch("handlers.planning._load_demand_history", return_value=demand):
                planning_handler(tenant_id="t1", run_id="run-pl-4", db=MagicMock())
            assert ctx.demand_data is demand
        finally:
            remove("run-pl-4")

    def test_91_failure_is_isolated_per_sku(self):
        """When run_model_initialisation raises for one SKU, the run must
        continue (handler logs and proceeds, doesn't propagate)."""
        ctx = _ctx("run-pl-5", {
            "sku-good": ProcessingTier.FULL,
            "sku-bad":  ProcessingTier.FULL,
        })
        try:
            def fake_91(sku_id, **k):
                if sku_id == "sku-bad":
                    raise RuntimeError("9.1 explode")
                return MagicMock(pattern_label="stable", assigned_model="ses")
            with patch("handlers.planning.prefetch_calibrated_windows", return_value={}), \
                 patch("handlers.planning.prefetch_calibration_gaps", return_value={}), \
                 patch("handlers.planning._load_demand_history",
                       return_value=_fake_demand_data(["sku-good", "sku-bad"])), \
                 patch("handlers.planning.run_model_initialisation",
                       side_effect=fake_91), \
                 patch("handlers.acting.build_sku_pipeline_input",
                       return_value=MagicMock(sku_id="sku-good")):
                planning_handler(tenant_id="t1", run_id="run-pl-5", db=MagicMock())
            # Only the good SKU produced a pipeline_input; bad was logged & skipped
            assert len(ctx.pipeline_inputs) == 1
        finally:
            remove("run-pl-5")
