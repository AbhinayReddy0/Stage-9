"""
unit/batch_jobs/test_model_performance_aggregator.py — pure-function coverage.

The aggregator's hot loop is DB-heavy (integration territory), but its
classification + numeric helpers are deterministic and worth pinning:

    * _improvement       — current - prior MAPE
    * _classify_trend    — improving / degrading / stable
    * _to_float          — DB scalar → float, NaN-tolerant
    * _row_to_args       — ModelPerformanceRow → INSERT args tuple
    * _bulk_insert       — fake-cursor fallback path
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from learning.model_performance_aggregator import (
    _improvement,
    _classify_trend,
    _to_float,
    _row_to_args,
    _bulk_insert,
    ModelPerformanceRow,
    TREND_IMPROVING,
    TREND_DEGRADING,
    TREND_STABLE,
)


# ---------------------------------------------------------------------------
# _improvement
# ---------------------------------------------------------------------------

class TestImprovement:

    def test_both_none_returns_none(self):
        assert _improvement(None, None) is None

    def test_current_none_returns_none(self):
        assert _improvement(None, 0.10) is None

    def test_prior_none_returns_none(self):
        assert _improvement(0.10, None) is None

    def test_decrease_returns_negative(self):
        """MAPE went down: current 0.10, prior 0.15 → -0.05 (improving)."""
        assert _improvement(0.10, 0.15) == pytest.approx(-0.05)

    def test_increase_returns_positive(self):
        """MAPE went up: current 0.20, prior 0.15 → 0.05 (degrading)."""
        assert _improvement(0.20, 0.15) == pytest.approx(0.05)

    def test_unchanged_returns_zero(self):
        assert _improvement(0.10, 0.10) == 0.0


# ---------------------------------------------------------------------------
# _classify_trend
# ---------------------------------------------------------------------------

class TestClassifyTrend:

    BAND = 0.01

    def test_none_improvement_yields_stable(self):
        """No prior data — no comparison possible → stable (neutral)."""
        assert _classify_trend(None, self.BAND) == TREND_STABLE

    def test_improvement_below_negative_band_is_improving(self):
        """current - prior < -band → MAPE dropped → 'improving'."""
        assert _classify_trend(-0.05, self.BAND) == TREND_IMPROVING

    def test_improvement_above_positive_band_is_degrading(self):
        assert _classify_trend(0.05, self.BAND) == TREND_DEGRADING

    def test_within_band_is_stable_either_direction(self):
        for delta in (-0.005, 0.0, 0.005):
            assert _classify_trend(delta, self.BAND) == TREND_STABLE

    def test_at_negative_band_boundary_is_stable(self):
        """Boundary inclusive on the stable side: imp == -band → stable."""
        assert _classify_trend(-self.BAND, self.BAND) == TREND_STABLE

    def test_at_positive_band_boundary_is_stable(self):
        assert _classify_trend(self.BAND, self.BAND) == TREND_STABLE

    def test_float_noise_at_boundary_does_not_flip(self):
        """`round(..., 9)` inside _classify_trend prevents IEEE-754 noise from
        flipping the classification on near-boundary deltas like
        0.15 - 0.16 = -0.010000000000000009."""
        noisy_delta = -0.010000000000000009
        # Without rounding, this would be < -0.01 by epsilon → 'improving'.
        # With rounding to 9dp, both sides are -0.01 → 'stable'.
        assert _classify_trend(noisy_delta, 0.01) == TREND_STABLE


# ---------------------------------------------------------------------------
# _to_float
# ---------------------------------------------------------------------------

class TestToFloat:

    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_int_converts(self):
        assert _to_float(5) == 5.0

    def test_float_passes_through(self):
        assert _to_float(3.14) == 3.14

    def test_string_numeric_converts(self):
        assert _to_float("2.5") == 2.5

    def test_nan_returns_none(self):
        """NaN must NOT propagate — surface as missing instead."""
        assert _to_float(float("nan")) is None


# ---------------------------------------------------------------------------
# _row_to_args
# ---------------------------------------------------------------------------

class TestRowToArgs:

    def test_returns_tuple(self):
        row = ModelPerformanceRow(
            tenant_id="t1",
            model_name="ses",
            horizon_days=30,
            avg_mape=0.05,
            median_mape=0.04,
            p90_mape=0.10,
            avg_bias=0.01,
            sample_count=100,
            mape_trend="stable",
            improvement_vs_prior=0.001,
            period_start=None,
            period_end=None,
            prior_avg_mape=None,
        )
        args = _row_to_args(row)
        assert isinstance(args, tuple)

    def test_first_arg_is_tenant_id(self):
        row = ModelPerformanceRow(
            tenant_id="my-tenant", model_name="ses", horizon_days=30,
            avg_mape=0.05, median_mape=0.04, p90_mape=0.10, avg_bias=0.01,
            sample_count=10, mape_trend="stable", improvement_vs_prior=0.0,
            period_start=None, period_end=None, prior_avg_mape=None,
        )
        assert _row_to_args(row)[0] == "my-tenant"


# ---------------------------------------------------------------------------
# _bulk_insert (fake-cursor path)
# ---------------------------------------------------------------------------

class TestBulkInsert:

    def test_fake_cursor_uses_executemany(self):
        """A cursor without `.connection` must trigger the executemany
        fallback (real psycopg2 cursors have .connection; mocks don't)."""
        cur = MagicMock(spec=["execute", "executemany"])
        # spec excludes "connection" attribute — hasattr returns False
        assert not hasattr(cur, "connection")
        _bulk_insert(cur, [("t1", "ses", 30, 0.05, "stable", 0.01, 100)])
        cur.executemany.assert_called_once()

    def test_empty_arg_list_still_calls_through(self):
        """Whether an empty list is a no-op or a 0-row INSERT is up to the
        implementation — either way it must NOT raise."""
        cur = MagicMock(spec=["execute", "executemany"])
        _bulk_insert(cur, [])  # should not raise
