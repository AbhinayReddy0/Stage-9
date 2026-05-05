"""
unit/orchestration/test_tier_router.py — tier_router pure-function coverage.

Targets the deterministic helpers that don't need DB:
    * _compute_level_scale
    * _scale_horizons
    * _get_best_hp_from_thompson
    * _get_model_class
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from forecasting.tier_router import (
    _compute_level_scale,
    _scale_horizons,
    _get_best_hp_from_thompson,
    _get_model_class,
    _SCALE_MIN,
    _SCALE_MAX,
)


# ---------------------------------------------------------------------------
# _compute_level_scale
# ---------------------------------------------------------------------------

class TestComputeLevelScale:

    def test_flat_series_returns_one(self):
        df = pd.DataFrame({"qty": [10.0] * 30})
        assert _compute_level_scale(df) == pytest.approx(1.0, abs=0.05)

    def test_series_shorter_than_two_rows_returns_one(self):
        df = pd.DataFrame({"qty": [10.0]})
        assert _compute_level_scale(df) == 1.0

    def test_all_zero_series_returns_one(self):
        df = pd.DataFrame({"qty": [0.0] * 30})
        assert _compute_level_scale(df) == 1.0

    def test_uptrending_recent_demand_scales_above_one(self):
        """Within the _MICRO_WINDOW slice, demand starts low and climbs.
        SES converges toward the latest values → ratio above 1."""
        # 14-day window: first half low, second half high
        qty = [5.0] * 7 + [15.0] * 7
        df = pd.DataFrame({"qty": qty})
        assert _compute_level_scale(df) > 1.0

    def test_downtrending_recent_demand_scales_below_one(self):
        """Latest values lower than earlier values → SES level below mean."""
        qty = [20.0] * 7 + [5.0] * 7
        df = pd.DataFrame({"qty": qty})
        assert _compute_level_scale(df) < 1.0

    def test_scale_clamped_to_max(self):
        """Spike of 100× on every recent day must NOT exceed _SCALE_MAX."""
        qty = [1.0] * 5 + [1000.0] * 14
        df = pd.DataFrame({"qty": qty})
        assert _compute_level_scale(df) <= _SCALE_MAX + 1e-9

    def test_scale_clamped_to_min(self):
        """Crash of 100× on every recent day must NOT drop below _SCALE_MIN."""
        qty = [1000.0] * 5 + [1.0] * 14
        df = pd.DataFrame({"qty": qty})
        assert _compute_level_scale(df) >= _SCALE_MIN - 1e-9


# ---------------------------------------------------------------------------
# _scale_horizons
# ---------------------------------------------------------------------------

class TestScaleHorizons:

    def _prior(self, value):
        return {f"forecast_{h}d": {"mean": value, "p50": value,
                                   "p80": value * 1.2, "p90": value * 1.4}
                for h in (7, 14, 30, 60, 90, 150, 180, 365)}

    def test_returns_all_horizon_columns(self):
        out = _scale_horizons(self._prior(100.0), 1.5)
        keys = {f"forecast_{h}d" for h in (7, 14, 30, 60, 90, 150, 180, 365)}
        assert keys.issubset(out.keys())

    def test_scale_one_is_identity(self):
        prior = self._prior(100.0)
        out   = _scale_horizons(prior, 1.0)
        for col in prior:
            for k in ("mean", "p50", "p80", "p90"):
                assert out[col][k] == pytest.approx(prior[col][k])

    def test_scaling_multiplies_every_quantile(self):
        out = _scale_horizons(self._prior(100.0), 2.0)
        for col, quantiles in out.items():
            assert quantiles["mean"] == pytest.approx(200.0)
            assert quantiles["p50"]  == pytest.approx(200.0)

    def test_non_dict_quantile_passes_through(self):
        prior = {"forecast_7d":  {"mean": 100, "p50": 100, "p80": 120, "p90": 140},
                 "forecast_14d": None,   # missing/unexpected type
                 "forecast_30d": "weird",
                 "forecast_60d": {"mean": 100, "p50": 100, "p80": 120, "p90": 140},
                 "forecast_90d": {"mean": 100, "p50": 100, "p80": 120, "p90": 140},
                 "forecast_150d": {"mean": 100, "p50": 100, "p80": 120, "p90": 140},
                 "forecast_180d": {"mean": 100, "p50": 100, "p80": 120, "p90": 140},
                 "forecast_365d": {"mean": 100, "p50": 100, "p80": 120, "p90": 140}}
        out = _scale_horizons(prior, 2.0)
        assert out["forecast_14d"] is None
        assert out["forecast_30d"] == "weird"
        assert out["forecast_7d"]["mean"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# _get_best_hp_from_thompson
# ---------------------------------------------------------------------------

class TestGetBestHpFromThompson:

    def test_returns_default_when_state_empty(self):
        default = {"smoothing_level": 0.3}
        preloaded = type("X", (), {"thompson_state": {}})()
        out = _get_best_hp_from_thompson("sku-1", "ses", preloaded, default)
        assert out == default

    def test_returns_highest_alpha_over_alpha_plus_beta(self):
        """The HP with the highest α/(α+β) wins."""
        preloaded = type("X", (), {"thompson_state": {
            ("sku-1", "ses"): {
                "h-strong": {"alpha": 9.0, "beta": 1.0,
                             "config": {"smoothing_level": 0.5}},
                "h-weak":   {"alpha": 1.0, "beta": 9.0,
                             "config": {"smoothing_level": 0.1}},
            }
        }})()
        out = _get_best_hp_from_thompson("sku-1", "ses", preloaded,
                                         default_hp={"smoothing_level": 0.99})
        assert out == {"smoothing_level": 0.5}

    def test_returns_default_when_winner_config_is_empty(self):
        """If the best entry has no `config` key (or empty dict),
        fall back to default to avoid downstream bad-HP crashes."""
        default = {"smoothing_level": 0.3}
        preloaded = type("X", (), {"thompson_state": {
            ("sku-1", "ses"): {"h": {"alpha": 5.0, "beta": 1.0, "config": {}}}
        }})()
        out = _get_best_hp_from_thompson("sku-1", "ses", preloaded, default)
        assert out == default


# ---------------------------------------------------------------------------
# _get_model_class
# ---------------------------------------------------------------------------

class TestGetModelClass:

    def test_returns_class_for_known_model_name(self):
        for name in ("Naive Forecast", "naive_forecast", "naive"):
            cls = _get_model_class(name)
            assert isinstance(cls, type)

    def test_unknown_model_name_falls_back_to_ses(self):
        """The production code defaults unknown names to SES rather than
        raising — guards against schema-name drift in production data."""
        from models.ses import SESModel
        cls = _get_model_class("totally_made_up_model")
        assert cls is SESModel
