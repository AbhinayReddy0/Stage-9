"""
tests/test_sub_stage_91.py — Sub-Stage 9.1 Acceptance Criteria
===============================================================
24 test cases covering every acceptance criterion from the spec.

Grouped by decision:
    D1 (4)  — Model Assignment
    D2 (4)  — Quantile Selection
    D3 (3)  — Effective Max Horizon
    D4 (4)  — Learning Mode
    D5 (3)  — OOS Adjustment Factor
    D6 (3)  — B2B Mode
    D7 (3)  — Reorder Bias Factor

No database required — all external dependencies replaced with
lightweight test doubles.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from infrastructure.constants import (
    LearningMode,
    LifecycleStage,
    Model,
    Param,
    Pattern,
    TenantMaturity,
)
from pipeline.preloader import PreloadedData
from pipeline.model_initialization import LearningContext, run_model_initialisation


# ===========================================================================
# Test doubles
# ===========================================================================

_DEFAULT_PARAMS: dict[str, Any] = {
    Param.QUANTILE_STABLE:       0.80,
    Param.QUANTILE_TRENDING:     0.80,
    Param.QUANTILE_COLD_START:   0.90,
    Param.QUANTILE_INTERMITTENT: 0.90,
    Param.QUANTILE_SEASONAL:     0.90,
    Param.EXPLOIT_THRESHOLD_NEW:         8,
    Param.EXPLOIT_THRESHOLD_DEVELOPING:  5,
    Param.EXPLOIT_THRESHOLD_ESTABLISHED: 3,
    Param.THOMPSON_EXPLOIT_CONFIDENCE_THRESHOLD: 0.60,
    Param.MIN_SEASONAL_OBS_DAYS: 120,
}


class _MockParams:
    def __init__(self, overrides: dict | None = None) -> None:
        self._values = {**_DEFAULT_PARAMS, **(overrides or {})}

    def get(self, key: str) -> Any:
        return self._values[key]


class _MockSignalConsumer:
    """Returns a fixed list of reorder_outcome payloads."""

    def __init__(self, signals: list[dict] | None = None) -> None:
        self._signals = signals or []

    def peek(self, tenant_id, signal_type, sku_id=None, limit=5) -> list[dict]:
        return self._signals[:limit]


class _CaptureBatchWriter:
    """Records every queue() call so tests can assert rows were written."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def queue(self, table: str, row: dict) -> None:
        self.calls.append((table, row))


class _NullBatchWriter:
    def queue(self, table: str, row: dict) -> None:
        pass


# ===========================================================================
# Builder helpers
# ===========================================================================

_SKU = "sku-test-001"
_TENANT = "tenant-test"
_RUN = "run-test-001"


def _preloaded(
    *,
    pattern_label: str = Pattern.STABLE,
    lifecycle_stage: str | None = None,
    weekend_zero_ratio: float = 0.0,
    on_watchlist: bool = False,
    drift_detected: bool = False,
    obs_days: int = 365,
    criticality_tier: str | None = None,
    service_level_target: float | None = None,
    planned_end_date: date | None = None,
    shelf_life_days: int | None = None,
    oos_record: dict | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    historical_runs: int = 0,
    tenant_maturity: str = TenantMaturity.NEW,
) -> PreloadedData:
    p = PreloadedData(tenant_id=_TENANT)
    p.pattern_ctx[_SKU] = {
        "pattern_label":      pattern_label,
        "lifecycle_stage":    lifecycle_stage,
        "weekend_zero_ratio": weekend_zero_ratio,
        "on_watchlist":       on_watchlist,
        "drift_detected":     drift_detected,
        "obs_days":           obs_days,
    }
    p.sku_metadata[_SKU] = {
        "criticality_tier":    criticality_tier,
        "service_level_target": service_level_target,
        "planned_end_date":    planned_end_date,
        "shelf_life_days":     shelf_life_days,
    }
    if oos_record is not None:
        p.oos_ctx[_SKU] = oos_record
    p.thompson_ctx[_SKU] = {
        "alpha":           alpha,
        "beta":            beta,
        "historical_runs": historical_runs,
    }
    p.signal_context = {"tenant_maturity": tenant_maturity}
    return p


def _run(
    preloaded: PreloadedData,
    *,
    params: _MockParams | None = None,
    consumer: _MockSignalConsumer | None = None,
    bw: _NullBatchWriter | _CaptureBatchWriter | None = None,
) -> LearningContext:
    return run_model_initialisation(
        sku_id=_SKU,
        preloaded=preloaded,
        params=params or _MockParams(),
        batch_writer=bw or _NullBatchWriter(),
        consumer=consumer or _MockSignalConsumer(),
        run_id=_RUN,
    )


# ===========================================================================
# Decision 1 — Model Assignment
# ===========================================================================

def test_d1_cold_start_assigns_naive():
    ctx = _run(_preloaded(pattern_label=Pattern.COLD_START))
    assert ctx.assigned_model == Model.NAIVE


def test_d1_seasonal_assigns_prophet_when_sufficient_history():
    ctx = _run(_preloaded(pattern_label=Pattern.SEASONAL, obs_days=365))
    assert ctx.assigned_model == Model.PROPHET


def test_d1_seasonal_guard_overrides_to_holt_when_insufficient_history():
    ctx = _run(_preloaded(pattern_label=Pattern.SEASONAL, obs_days=90))
    assert ctx.assigned_model == Model.HOLTS_LINEAR
    assert ctx.insufficient_seasonal_history is True


def test_d1_stable_assigns_ses():
    ctx = _run(_preloaded(pattern_label=Pattern.STABLE))
    assert ctx.assigned_model == Model.SES


# ===========================================================================
# Decision 2 — Quantile Selection
# ===========================================================================

def test_d2_criticality_a_uses_099():
    ctx = _run(_preloaded(criticality_tier="A"))
    assert ctx.selected_quantile == pytest.approx(0.99)
    assert ctx.quantile_source == "criticality_a"


def test_d2_pattern_default_when_no_override():
    # No criticality_A → reads quantile from params (stable → 0.80)
    ctx = _run(_preloaded(pattern_label=Pattern.STABLE, criticality_tier=None))
    assert ctx.selected_quantile == pytest.approx(0.80)
    assert ctx.quantile_source == "pattern_default"


# ===========================================================================
# Decision 3 — Effective Max Horizon
# ===========================================================================

def test_d3_planned_end_date_caps_horizon():
    end_date = date.today() + timedelta(days=60)
    ctx = _run(_preloaded(planned_end_date=end_date))
    assert ctx.effective_max_horizon == 60


def test_d3_shelf_life_caps_horizon():
    ctx = _run(_preloaded(shelf_life_days=45))
    assert ctx.effective_max_horizon == 45


def test_d3_floor_at_7_when_very_short():
    # shelf_life of 3 days → min(365,3)=3 → max(7,3)=7
    ctx = _run(_preloaded(shelf_life_days=3))
    assert ctx.effective_max_horizon == 7


# ===========================================================================
# Decision 4 — Learning Mode
# ===========================================================================

def test_d4_exploit_when_all_conditions_met():
    # confidence = 2/(2+1) ≈ 0.667 > 0.60, runs=5 ≥ threshold=3, not flagged
    ctx = _run(
        _preloaded(
            alpha=2.0, beta=1.0, historical_runs=5,
            on_watchlist=False, drift_detected=False,
            tenant_maturity=TenantMaturity.ESTABLISHED,
        )
    )
    assert ctx.learning_mode == LearningMode.EXPLOIT


def test_d4_explore_when_confidence_at_threshold():
    # confidence = 0.6/(0.6+0.4) = 0.60 — NOT > 0.60 → explore
    ctx = _run(
        _preloaded(
            alpha=0.6, beta=0.4, historical_runs=5,
            tenant_maturity=TenantMaturity.ESTABLISHED,
        )
    )
    assert ctx.learning_mode == LearningMode.EXPLORE


def test_d4_explore_when_runs_below_threshold():
    # confidence > 0.60 but runs=2 < threshold=3 → explore
    ctx = _run(
        _preloaded(
            alpha=2.0, beta=1.0, historical_runs=2,
            tenant_maturity=TenantMaturity.ESTABLISHED,
        )
    )
    assert ctx.learning_mode == LearningMode.EXPLORE


def test_d4_explore_when_on_watchlist():
    # all numeric conditions met, but on_watchlist blocks exploit
    ctx = _run(
        _preloaded(
            alpha=2.0, beta=1.0, historical_runs=5,
            on_watchlist=True,
            tenant_maturity=TenantMaturity.ESTABLISHED,
        )
    )
    assert ctx.learning_mode == LearningMode.EXPLORE


# ===========================================================================
# Decision 5 — OOS Adjustment Factor
# ===========================================================================

def test_d5_oos_factor_applied():
    # 1 + 0.4 × 0.8 = 1.32
    ctx = _run(_preloaded(oos_record={"oos_pct": 0.4, "detection_confidence": 0.8}))
    assert ctx.oos_adjustment_factor == pytest.approx(1.32)


def test_d5_oos_factor_capped_at_150():
    # 1 + 1.0 × 0.9 = 1.90 → capped at 1.50
    ctx = _run(_preloaded(oos_record={"oos_pct": 1.0, "detection_confidence": 0.9}))
    assert ctx.oos_adjustment_factor == pytest.approx(1.50)


def test_d5_no_oos_record_returns_neutral():
    ctx = _run(_preloaded())  # no oos_record → factor 1.0
    assert ctx.oos_adjustment_factor == pytest.approx(1.0)


def test_d5_negative_oos_pct_clamped_to_10():
    # Negative oos_pct from bad DB data must not produce factor < 1.0
    ctx = _run(_preloaded(oos_record={"oos_pct": -0.5, "detection_confidence": 1.0}))
    assert ctx.oos_adjustment_factor == pytest.approx(1.0)


# ===========================================================================
# Decision 6 — B2B Mode
# ===========================================================================

def test_d6_b2b_true_above_threshold():
    ctx = _run(_preloaded(weekend_zero_ratio=0.70))
    assert ctx.is_b2b is True


def test_d6_b2b_false_at_threshold():
    # 0.60 is NOT strictly greater than 0.60 → False
    ctx = _run(_preloaded(weekend_zero_ratio=0.60))
    assert ctx.is_b2b is False


def test_d6_b2b_false_below_threshold():
    ctx = _run(_preloaded(weekend_zero_ratio=0.50))
    assert ctx.is_b2b is False


# ===========================================================================
# Decision 7 — Reorder Bias Factor
# ===========================================================================

def test_d7_stockout_only():
    # 2 stockout events ≥ REORDER_STOCKOUT_MIN_EVENTS (2) → 1.10
    signals = [
        {"stockout": True,  "overstock_pct": 0.0},
        {"stockout": True,  "overstock_pct": 0.0},
        {"stockout": False, "overstock_pct": 0.0},
    ]
    ctx = _run(_preloaded(), consumer=_MockSignalConsumer(signals))
    assert ctx.reorder_bias_factor == pytest.approx(1.10)


def test_d7_overstock_only():
    # avg_overstock_pct = 0.40 > 0.30 threshold, no stockouts → 0.92
    signals = [
        {"stockout": False, "overstock_pct": 0.40},
        {"stockout": False, "overstock_pct": 0.40},
    ]
    ctx = _run(_preloaded(), consumer=_MockSignalConsumer(signals))
    assert ctx.reorder_bias_factor == pytest.approx(0.92)


def test_d7_both_conditions_stockout_wins():
    # Both triggered; stockout must win (1.10 not 0.92)
    signals = [
        {"stockout": True,  "overstock_pct": 0.50},
        {"stockout": True,  "overstock_pct": 0.50},
        {"stockout": False, "overstock_pct": 0.50},
    ]
    ctx = _run(_preloaded(), consumer=_MockSignalConsumer(signals))
    assert ctx.reorder_bias_factor == pytest.approx(1.10)


# ===========================================================================
# Cross-cutting — BatchWriter row and context wiring
# ===========================================================================

def test_batchwriter_row_queued_for_every_sku():
    bw = _CaptureBatchWriter()
    _run(_preloaded(), bw=bw)
    assert len(bw.calls) == 1
    table, row = bw.calls[0]
    assert table == "model_initialization_s9"
    assert row["sku_id"] == _SKU
    assert row["tenant_id"] == _TENANT
    assert "assigned_model" in row
    assert "learning_mode" in row


def test_context_fields_wired_correctly():
    ctx = _run(
        _preloaded(
            pattern_label=Pattern.TRENDING,
            lifecycle_stage=LifecycleStage.CLEARANCE,
            shelf_life_days=90,
        )
    )
    assert ctx.sku_id == _SKU
    assert ctx.tenant_id == _TENANT
    assert ctx.run_id == _RUN
    assert ctx.pattern_label == Pattern.TRENDING
    assert ctx.lifecycle_stage == LifecycleStage.CLEARANCE
    assert ctx.effective_max_horizon == 90
    # Downstream fields stay at defaults until sub-stages 9.2/9.3 run
    assert ctx.best_hp == {}
    assert ctx.selected_features == []
    assert ctx.b2b_mode_disabled is False
