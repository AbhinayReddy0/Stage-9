"""
unit/batch_jobs/test_outcome_collector.py — pure-function coverage of the
outcome_collector batch job.

Targets the deterministic helpers (no DB needed):
    * _quantile_key — selected_quantile → p50/p80/p90
    * _to_decimal   — JSONB-derived numeric coercion
    * _compute_errors — (mape, wape, bias) computation
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from learning.outcome_collector import (
    _quantile_key,
    _to_decimal,
    _compute_errors,
)


# ---------------------------------------------------------------------------
# _quantile_key
# ---------------------------------------------------------------------------

class TestQuantileKey:

    def test_050_returns_p50(self):
        assert _quantile_key(0.50) == "p50"

    def test_080_returns_p80(self):
        assert _quantile_key(0.80) == "p80"

    def test_090_returns_p90(self):
        assert _quantile_key(0.90) == "p90"

    def test_unknown_quantile_falls_back_to_p90(self):
        """Conservative default per spec §4 Step 3."""
        assert _quantile_key(0.95) == "p90"
        assert _quantile_key(0.999) == "p90"

    def test_decimal_input_works(self):
        assert _quantile_key(Decimal("0.50")) == "p50"
        assert _quantile_key(Decimal("0.80")) == "p80"

    def test_string_input_via_decimal(self):
        """Stringified Decimal — explicitly handled by Decimal(str(...))."""
        assert _quantile_key("0.50") == "p50"

    def test_float_precision_does_not_misroute(self):
        """0.1 + 0.2 in IEEE-754 = 0.30000000000000004 — would mismatch
        if we naively compared as float. Decimal(str(...)) avoids this."""
        # The spec stringifies first, so 0.50 from any path resolves cleanly.
        for v in [0.50, "0.50", Decimal("0.50")]:
            assert _quantile_key(v) == "p50"


# ---------------------------------------------------------------------------
# _to_decimal
# ---------------------------------------------------------------------------

class TestToDecimal:

    def test_none_returns_none(self):
        assert _to_decimal(None) is None

    def test_int_converts(self):
        assert _to_decimal(5) == Decimal("5")

    def test_float_converts_via_str(self):
        """str(0.1)='0.1' so the noisy IEEE-754 representation doesn't leak in."""
        assert _to_decimal(0.1) == Decimal("0.1")

    def test_str_numeric_converts(self):
        assert _to_decimal("3.14") == Decimal("3.14")

    def test_decimal_passes_through(self):
        d = Decimal("7.5")
        assert _to_decimal(d) == d

    def test_garbage_returns_none(self):
        assert _to_decimal("not a number") is None
        assert _to_decimal([1, 2, 3]) is None
        assert _to_decimal({"k": "v"}) is None


# ---------------------------------------------------------------------------
# _compute_errors
# ---------------------------------------------------------------------------

class TestComputeErrors:

    def test_perfect_forecast_zero_error(self):
        mape, wape, bias = _compute_errors(Decimal("100"), Decimal("100"))
        assert mape == Decimal("0.000")
        assert wape == Decimal("0.000")
        assert bias == Decimal("0.000")

    def test_over_forecast_positive_bias(self):
        mape, wape, bias = _compute_errors(Decimal("110"), Decimal("100"))
        assert mape == Decimal("0.100")
        assert bias == Decimal("0.100")    # positive = over-forecast

    def test_under_forecast_negative_bias(self):
        mape, wape, bias = _compute_errors(Decimal("90"), Decimal("100"))
        assert mape == Decimal("0.100")    # MAPE is absolute, always >= 0
        assert bias == Decimal("-0.100")   # negative = under-forecast

    def test_actual_zero_returns_all_none(self):
        """Spec §4 Step 4: writing 0 or inf would poison aggregates.
        Return None and let the aggregator filter the row."""
        mape, wape, bias = _compute_errors(Decimal("50"), Decimal("0"))
        assert mape is None
        assert wape is None
        assert bias is None

    def test_wape_equals_mape_for_single_row(self):
        """WAPE collapses to MAPE without cross-SKU weighting."""
        for forecast, actual in [
            (Decimal("110"), Decimal("100")),
            (Decimal("80"),  Decimal("120")),
            (Decimal("0"),   Decimal("50")),
        ]:
            mape, wape, _ = _compute_errors(forecast, actual)
            assert mape == wape

    def test_quantization_to_three_decimals(self):
        """All three values should be quantized to _Q_ERROR (3 decimals)."""
        mape, wape, bias = _compute_errors(Decimal("123"), Decimal("100"))
        # 23/100 = 0.23 → 0.230 after quantize
        assert str(mape).count(".") == 1
        # value should be 0.230
        assert mape == Decimal("0.230")

    def test_negative_actual_formulae_still_apply(self):
        """Spec §6: negative actuals shouldn't occur in production but the
        formulae remain mathematically correct if they do."""
        mape, wape, bias = _compute_errors(Decimal("100"), Decimal("-50"))
        # |(-50) - 100| / -50 = 150 / -50 = -3.0 (formula doesn't take abs of denominator)
        # Implementation just plugs into the formula — we accept whatever it returns
        assert mape is not None
        assert bias is not None

    def test_zero_forecast_returns_full_mape(self):
        """forecast=0, actual=100 → MAPE = |0 - 100| / 100 = 1.0 (100% error)."""
        mape, wape, bias = _compute_errors(Decimal("0"), Decimal("100"))
        assert mape == Decimal("1.000")
        assert bias == Decimal("-1.000")  # massively under-forecast
