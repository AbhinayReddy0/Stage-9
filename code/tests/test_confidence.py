"""
unit/substages/test_confidence.py — coverage for forecasting.confidence

The 5-step compute_confidence() formula is what turns a backtest MAPE into
the confidence_final number Stage 10 reads. Each step is multiplicative and
reasonably easy to test in isolation with a stub TenantParams.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from forecasting.confidence import (
    compute_confidence,
    ForecastContext,
    EXCEPTION_PENALTY_FLAGS,
)


# ---------------------------------------------------------------------------
# TenantParams stub — keyed string lookup matching production .get()
# ---------------------------------------------------------------------------

_PARAM_DEFAULTS = {
    "confidence_base_stable":               0.90,
    "confidence_base_trending":             0.80,
    "confidence_base_seasonal":             0.80,
    "confidence_base_intermittent":         0.60,
    "confidence_base_cold_start":           0.50,
    "confidence_floor":                     0.30,
    "confidence_ceiling":                   0.95,
    "mape_cap_in_confidence":               0.50,
    "exception_penalty":                    0.80,
    "overconfidence_threshold":             0.05,
    "overconfidence_mult":                  0.90,
    "underconfidence_mult":                 1.10,
    "stage8_penalty_threshold":             0.40,
    "stage8_penalty_mult":                  0.85,
    "structural_break_confidence_penalty":  0.15,
    "insufficient_post_break_mult":         0.75,
}


class _Params:
    def __init__(self, **overrides):
        self._d = {**_PARAM_DEFAULTS, **overrides}

    def get(self, key):
        from infrastructure.constants import Param
        if isinstance(key, type(Param.CONFIDENCE_FLOOR)):
            key = key.value if hasattr(key, "value") else str(key)
        return self._d[key]


def _kwargs(**override):
    """Defaults: clean run, stable pattern, mid-range mape, no flags, etc."""
    base = dict(
        pattern_label="stable",
        backtest_mape=0.10,
        exception_flags=[],
        calibration_gap=None,
        stage8_confidence=0.85,
        reorder_bias_factor=1.0,
        ctx=ForecastContext(),
        params=_Params(),
        reasonableness_multiplier=1.0,
    )
    base.update(override)
    return base


# ---------------------------------------------------------------------------
# EXCEPTION_PENALTY_FLAGS
# ---------------------------------------------------------------------------

class TestExceptionPenaltyFlags:

    def test_exact_set_pinned(self):
        """The flag set is contractual — Stage 8 emits these specific names.
        If the set changes, that's a coordinated rollout, not silent drift."""
        assert EXCEPTION_PENALTY_FLAGS == frozenset({
            "stockout", "promo_spike", "unusual_drop", "high_volatility",
            "forecast_unusually_high", "forecast_unusually_low",
        })

    def test_high_mape_excluded(self):
        """high_mape is handled in Step 5 (status determination), not Step 1."""
        assert "high_mape" not in EXCEPTION_PENALTY_FLAGS


# ---------------------------------------------------------------------------
# Step 1 — base × (1 − mape) × exception_penalty
# ---------------------------------------------------------------------------

class TestStep1MapeAndFlags:

    def test_clean_run_returns_base_minus_mape(self):
        base, final = compute_confidence(**_kwargs(backtest_mape=0.10))
        assert base == 0.90
        # Expected: 0.90 × (1 − 0.10) = 0.81 (no other multipliers fire)
        assert final == pytest.approx(0.81, abs=0.001)

    def test_zero_mape_returns_base(self):
        _, final = compute_confidence(**_kwargs(backtest_mape=0.0))
        assert final == pytest.approx(0.90, abs=0.001)

    def test_mape_cap_applied(self):
        """A degenerate MAPE > cap is truncated to cap before subtraction."""
        _, final = compute_confidence(**_kwargs(backtest_mape=2.0))
        # capped at 0.50 → 0.90 × (1 − 0.50) = 0.45
        assert final == pytest.approx(0.45, abs=0.001)

    def test_none_mape_treated_as_cap(self):
        """No backtest available (e.g. cold-start) → use mape_cap."""
        _, final = compute_confidence(**_kwargs(
            pattern_label="cold_start",
            backtest_mape=None,
        ))
        # base for cold_start = 0.50; cap = 0.50; → 0.50 × 0.5 = 0.25, clamped to floor=0.30
        assert final == pytest.approx(0.30, abs=0.001)

    def test_nan_mape_treated_as_cap(self):
        _, final = compute_confidence(**_kwargs(backtest_mape=float("nan")))
        # 0.90 × (1 - 0.50) = 0.45
        assert final == pytest.approx(0.45, abs=0.001)

    def test_exception_flag_applies_penalty(self):
        _, with_flag = compute_confidence(**_kwargs(
            backtest_mape=0.10, exception_flags=["stockout"],
        ))
        _, without   = compute_confidence(**_kwargs(backtest_mape=0.10))
        # Penalty = 0.80
        assert with_flag == pytest.approx(without * 0.80, abs=0.001)

    def test_unrelated_flag_does_not_apply_penalty(self):
        _, no_penalty = compute_confidence(**_kwargs(
            backtest_mape=0.10, exception_flags=["totally_unknown_flag"],
        ))
        _, baseline = compute_confidence(**_kwargs(backtest_mape=0.10))
        assert no_penalty == pytest.approx(baseline, abs=0.001)


# ---------------------------------------------------------------------------
# Step 2 — calibration gap
# ---------------------------------------------------------------------------

class TestStep2CalibrationGap:

    def test_none_gap_no_change(self):
        _, with_none = compute_confidence(**_kwargs(calibration_gap=None))
        _, neutral   = compute_confidence(**_kwargs(calibration_gap=0.0))
        assert with_none == pytest.approx(neutral, abs=0.001)

    def test_overconfidence_when_gap_above_threshold(self):
        """gap > overconfidence_threshold → multiplied by overconfidence_mult."""
        _, with_gap   = compute_confidence(**_kwargs(calibration_gap=0.10))
        _, baseline   = compute_confidence(**_kwargs(calibration_gap=None))
        # ratio should equal overconfidence_mult = 0.90
        assert with_gap / baseline == pytest.approx(0.90, abs=0.005)

    def test_underconfidence_when_gap_below_negative_threshold(self):
        _, with_gap   = compute_confidence(**_kwargs(calibration_gap=-0.10))
        _, baseline   = compute_confidence(**_kwargs(calibration_gap=None))
        # ratio == underconfidence_mult = 1.10, but capped at ceiling=0.95
        # baseline = 0.81; underconfidence × 1.1 = 0.891. No cap.
        assert with_gap / baseline == pytest.approx(1.10, abs=0.005)

    def test_within_threshold_no_calibration_change(self):
        _, with_gap = compute_confidence(**_kwargs(calibration_gap=0.04))
        _, neutral  = compute_confidence(**_kwargs(calibration_gap=0.0))
        # gap=0.04 < threshold=0.05 → no calibration adjustment
        assert with_gap == pytest.approx(neutral, abs=0.001)


# ---------------------------------------------------------------------------
# Step 3 — Stage 8 signal quality
# ---------------------------------------------------------------------------

class TestStep3Stage8Penalty:

    def test_low_stage8_confidence_applies_penalty(self):
        """stage8_confidence < threshold → multiply by stage8_penalty_mult."""
        _, low      = compute_confidence(**_kwargs(stage8_confidence=0.20))
        _, normal   = compute_confidence(**_kwargs(stage8_confidence=0.85))
        # ratio == stage8_penalty_mult = 0.85
        assert low / normal == pytest.approx(0.85, abs=0.005)

    def test_at_threshold_no_penalty(self):
        """Boundary: stage8_confidence == threshold → no penalty (strict <)."""
        _, at_thr   = compute_confidence(**_kwargs(stage8_confidence=0.40))
        _, normal   = compute_confidence(**_kwargs(stage8_confidence=0.85))
        assert at_thr == pytest.approx(normal, abs=0.001)

    def test_none_stage8_confidence_no_penalty(self):
        _, _none = compute_confidence(**_kwargs(stage8_confidence=None))
        _, normal = compute_confidence(**_kwargs(stage8_confidence=0.85))
        assert _none == pytest.approx(normal, abs=0.001)


# ---------------------------------------------------------------------------
# Step 4 — reorder bias factor
# ---------------------------------------------------------------------------

class TestStep4ReorderBias:

    def test_reorder_bias_multiplies_directly(self):
        _, half = compute_confidence(**_kwargs(reorder_bias_factor=0.5))
        _, full = compute_confidence(**_kwargs(reorder_bias_factor=1.0))
        # 0.5 × the full value, but floor=0.30 may clamp on the low side
        assert half / full == pytest.approx(0.5, abs=0.005) or half == 0.30

    def test_reorder_bias_factor_one_is_neutral(self):
        _, with_one = compute_confidence(**_kwargs(reorder_bias_factor=1.0))
        # with reorder_bias=1, no other factors, expect base × (1 − mape) = 0.81
        assert with_one == pytest.approx(0.81, abs=0.001)


# ---------------------------------------------------------------------------
# Step 5 — structural break / insufficient post-break
# ---------------------------------------------------------------------------

class TestStep5StructuralBreak:

    def test_truncated_training_applies_penalty(self):
        """training_data_truncated → × (1 − structural_break_confidence_penalty)."""
        _, truncated = compute_confidence(**_kwargs(
            ctx=ForecastContext(training_data_truncated=True),
        ))
        _, clean     = compute_confidence(**_kwargs(ctx=ForecastContext()))
        # ratio == 1 - 0.15 = 0.85
        assert truncated / clean == pytest.approx(0.85, abs=0.005)

    def test_insufficient_post_break_applies_penalty(self):
        _, short = compute_confidence(**_kwargs(
            ctx=ForecastContext(insufficient_post_break=True),
        ))
        _, clean = compute_confidence(**_kwargs(ctx=ForecastContext()))
        # ratio == insufficient_post_break_mult = 0.75
        assert short / clean == pytest.approx(0.75, abs=0.005)

    def test_truncated_takes_precedence_over_insufficient(self):
        """Both flags true: truncated branch (the elif) means insufficient
        is NOT applied. Confidence loses 15% only."""
        _, both = compute_confidence(**_kwargs(
            ctx=ForecastContext(training_data_truncated=True,
                                insufficient_post_break=True),
        ))
        _, clean = compute_confidence(**_kwargs(ctx=ForecastContext()))
        assert both / clean == pytest.approx(0.85, abs=0.005)


# ---------------------------------------------------------------------------
# Floor / ceiling clamping
# ---------------------------------------------------------------------------

class TestClamping:

    def test_extreme_low_clamped_to_floor(self):
        """All multipliers compounded should never drop below confidence_floor."""
        _, final = compute_confidence(**_kwargs(
            backtest_mape=0.50,
            exception_flags=["stockout"],
            calibration_gap=0.10,
            stage8_confidence=0.10,
            reorder_bias_factor=0.1,
            ctx=ForecastContext(training_data_truncated=True),
        ))
        assert final >= 0.30   # confidence_floor

    def test_extreme_high_clamped_to_ceiling(self):
        """Maxed-out multipliers shouldn't push above confidence_ceiling."""
        _, final = compute_confidence(**_kwargs(
            backtest_mape=0.0,
            calibration_gap=-0.20,    # underconfidence boost
            reorder_bias_factor=2.0,  # 2× boost
        ))
        assert final <= 0.95   # confidence_ceiling


# ---------------------------------------------------------------------------
# Pattern-specific base
# ---------------------------------------------------------------------------

class TestPatternBase:

    def test_stable_pattern_uses_stable_base(self):
        base, _ = compute_confidence(**_kwargs(pattern_label="stable"))
        assert base == pytest.approx(0.90)

    def test_trending_pattern_uses_trending_base(self):
        base, _ = compute_confidence(**_kwargs(pattern_label="trending"))
        assert base == pytest.approx(0.80)

    def test_unknown_pattern_falls_back_to_stable_base(self):
        """Defensive: unrecognised pattern → confidence_base_stable."""
        base, _ = compute_confidence(**_kwargs(pattern_label="never_heard_of_it"))
        assert base == pytest.approx(0.90)
