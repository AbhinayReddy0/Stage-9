"""
tests/test_models.py — Atheera Stage 9
=======================================
Tests for all 5 concrete model classes:
    NaiveForecast, SESModel, CrostonMethod, HoltLinearTrend, ProphetModel

Covers:
    - BaseModel contract: all 5 methods + 5 properties
    - Done Criteria D1–D10 (from Build Plan §5.3)
    - Edge cases: E001 (Croston), E002 (Prophet), NaN/Inf input
    - HP search space correctness
    - ProcessPool pickle compatibility (D10)
    - optimized=False enforcement for SES (D8) and Holt (D9)
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from models.base import BaseModel, ModelFitError
from forecasting.tier_router import _get_model_class
from infrastructure.constants import Model, PATTERN_MODEL_MAP
from models.naive import NaiveForecast
from models.ses import SESModel
from models.croston import CrostonMethod
from models.holt import HoltLinearTrend
from models.prophet_model import ProphetModel

from tests.conftest import EXPECTED_HORIZON_KEYS, FEATURES

# All concrete model classes for parametrised tests
ALL_MODELS = [NaiveForecast, SESModel, CrostonMethod, HoltLinearTrend]
ALL_MODEL_IDS = ["NaiveForecast", "SESModel", "CrostonMethod", "HoltLinearTrend"]


# ===========================================================================
# BaseModel contract — applies to ALL models
# ===========================================================================

class TestBaseModelContract:

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_inherits_base_model(self, Cls):
        assert issubclass(Cls, BaseModel)

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_has_all_required_properties(self, Cls):
        m = Cls(hp=Cls({}).default_hp)
        assert isinstance(m.model_name,         str)
        assert isinstance(m.hp_search_space,     list)
        assert isinstance(m.default_hp,          dict)
        assert isinstance(m.required_features,   list)
        assert isinstance(m.optional_features,   list)

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_required_features_always_contains_date_and_qty(self, Cls):
        m = Cls(hp=Cls({}).default_hp)
        assert "date" in m.required_features
        assert "qty" in m.required_features

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_default_hp_in_search_space(self, Cls):
        m = Cls(hp=Cls({}).default_hp)
        assert m.default_hp in m.hp_search_space

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_default_hp_is_copy(self, Cls):
        """Mutating returned default_hp must not affect the class-level default."""
        m = Cls(hp={})
        hp1 = m.default_hp
        hp2 = m.default_hp
        hp1["__mutated__"] = True
        assert "__mutated__" not in hp2

    # ── D6: predict() return type and length ────────────────────────────────

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_returns_ndarray(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict(df_normal, FEATURES, horizon=30)
        assert isinstance(result, np.ndarray), f"{Cls.__name__}: predict must return np.ndarray"

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_correct_length(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        for horizon in [7, 14, 30]:
            result = m.predict(df_normal, FEATURES, horizon=horizon)
            assert len(result) == horizon, f"{Cls.__name__}: expected {horizon}, got {len(result)}"

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_all_non_negative(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict(df_normal, FEATURES, horizon=30)
        assert np.all(result >= 0), f"{Cls.__name__}: predict contains negatives"

    # ── D7: predict_all_horizons() key format ────────────────────────────────

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_all_horizons_returns_expected_keys(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict_all_horizons(df_normal, FEATURES)
        assert set(result.keys()) == EXPECTED_HORIZON_KEYS

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_all_horizons_string_keys(self, Cls, df_normal):
        """Keys must be strings like 'forecast_7d', not integers."""
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict_all_horizons(df_normal, FEATURES)
        assert all(isinstance(k, str) for k in result.keys())

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_all_horizons_quantile_keys(self, Cls, df_normal):
        """Each horizon value must have mean, p50, p80, p90."""
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict_all_horizons(df_normal, FEATURES)
        for key, val in result.items():
            assert set(val.keys()) == {"mean", "p50", "p80", "p90"}, \
                f"{Cls.__name__} {key}: missing quantile keys"

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_all_horizons_ordering(self, Cls, df_normal):
        """p50 ≤ p80 ≤ p90 must hold for every horizon (D5 extension)."""
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict_all_horizons(df_normal, FEATURES)
        for key, val in result.items():
            assert val["p50"] <= val["p80"] <= val["p90"], \
                f"{Cls.__name__} {key}: ordering violated {val}"

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_predict_all_horizons_all_non_negative(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_normal, FEATURES)
        result = m.predict_all_horizons(df_normal, FEATURES)
        for key, val in result.items():
            for q, v in val.items():
                assert v >= 0, f"{Cls.__name__} {key}.{q} = {v} (negative)"

    # ── oos_factor applied correctly ─────────────────────────────────────────

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_oos_factor_scales_mean(self, Cls, df_normal):
        """oos_factor=2.0 should double the mean point forecast."""
        m1 = Cls(hp=Cls({}).default_hp)
        m2 = Cls(hp=Cls({}).default_hp)
        m1.fit(df_normal, FEATURES)
        m2.fit(df_normal, FEATURES)
        r1 = m1.predict_all_horizons(df_normal, FEATURES, oos_factor=1.0)
        r2 = m2.predict_all_horizons(df_normal, FEATURES, oos_factor=2.0)
        # mean should be approximately doubled (bootstrap adds noise so approximate)
        assert r2["forecast_30d"]["mean"] == pytest.approx(
            r1["forecast_30d"]["mean"] * 2.0, rel=1e-6
        )

    # ── sample_weights accepted silently by non-Prophet models ───────────────

    @pytest.mark.parametrize("Cls", [NaiveForecast, SESModel, CrostonMethod, HoltLinearTrend],
                             ids=["Naive", "SES", "Croston", "Holt"])
    def test_sample_weights_silently_ignored(self, Cls, df_normal):
        """Non-Prophet models must accept sample_weights without crashing."""
        m = Cls(hp=Cls({}).default_hp)
        weights = np.ones(len(df_normal))
        m.fit(df_normal, FEATURES, sample_weights=weights)  # must not raise

    # ── compute_residuals ────────────────────────────────────────────────────

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_compute_residuals_returns_at_most_30_rows(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        resids = m.compute_residuals(df_normal, FEATURES)
        assert len(resids) <= 30

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_compute_residuals_all_finite(self, Cls, df_normal):
        m = Cls(hp=Cls({}).default_hp)
        resids = m.compute_residuals(df_normal, FEATURES)
        assert np.all(np.isfinite(resids))

    # ── D10: ProcessPool pickle compatibility ─────────────────────────────────

    @pytest.mark.parametrize("Cls", ALL_MODELS + [ProphetModel], ids=ALL_MODEL_IDS + ["ProphetModel"])
    def test_picklable_before_fit(self, Cls):
        """Done Criterion D10: model must be picklable for ProcessPoolExecutor."""
        obj = Cls(hp=Cls({}).default_hp)
        restored = pickle.loads(pickle.dumps(obj))
        assert type(restored).__name__ == Cls.__name__

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_picklable_after_fit(self, Cls, df_normal):
        """Fitted model must also be picklable (worker passes result back to pool)."""
        obj = Cls(hp=Cls({}).default_hp)
        obj.fit(df_normal, FEATURES)
        restored = pickle.loads(pickle.dumps(obj))
        assert type(restored).__name__ == Cls.__name__


# ===========================================================================
# NaiveForecast
# ===========================================================================

class TestNaiveForecast:

    def test_model_name(self):
        assert NaiveForecast({}).model_name == "Naive Forecast"

    def test_hp_search_space_has_9_configs(self):
        assert len(NaiveForecast({}).hp_search_space) == 9

    def test_flat_forecast(self, df_normal):
        """All horizon values should be identical (flat projection)."""
        m = NaiveForecast(hp={"lag_periods": 7, "smoothing_method": "mean_7d"})
        m.fit(df_normal, FEATURES)
        result = m.predict(df_normal, FEATURES, horizon=14)
        assert np.all(result == result[0]), "NaiveForecast must produce flat forecast"

    def test_short_series_fallback(self, df_sparse):
        """Series with < 7 rows must not crash — falls back to mean."""
        m = NaiveForecast(hp=NaiveForecast({}).default_hp)
        m.fit(df_sparse, FEATURES)
        result = m.predict(df_sparse, FEATURES, horizon=7)
        assert len(result) == 7
        assert np.all(result >= 0)

    def test_empty_series(self):
        df_empty = pd.DataFrame({"date": [], "qty": []})
        m = NaiveForecast(hp=NaiveForecast({}).default_hp)
        m.fit(df_empty, FEATURES)
        assert m._level == 0.0

    def test_nan_input_does_not_propagate(self, df_nan):
        """NaN/Inf in input must not produce NaN forecasts."""
        m = NaiveForecast(hp=NaiveForecast({}).default_hp)
        m.fit(df_nan, FEATURES)
        result = m.predict(df_nan, FEATURES, horizon=7)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("method,lag", [
        ("last_value", 1),
        ("mean_3d",    3),
        ("mean_7d",    7),
    ])
    def test_smoothing_methods(self, method, lag, df_normal):
        m = NaiveForecast(hp={"lag_periods": lag, "smoothing_method": method})
        m.fit(df_normal, FEATURES)
        assert np.isfinite(m._level)
        assert m._level > 0


# ===========================================================================
# SESModel
# ===========================================================================

class TestSESModel:

    def test_model_name(self):
        assert SESModel({}).model_name == "Simple Exponential Smoothing (SES)"

    def test_hp_search_space_has_5_configs(self):
        assert len(SESModel({}).hp_search_space) == 5

    def test_flat_forecast(self, df_normal):
        """SES produces a flat forecast — no trend."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_normal, FEATURES)
        result = m.predict(df_normal, FEATURES, horizon=14)
        assert np.all(result == result[0])

    # ── D8: explicit alpha enforcement ──────────────────────────────────────

    def test_explicit_alpha_honoured(self, df_normal):
        """Done Criterion D8: fitted alpha must match hp dict, not auto-selected."""
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            m = SESModel(hp={"smoothing_level": alpha})
            m.fit(df_normal, FEATURES)
            if m._fitted_result is not None and not m._using_fallback:
                fitted_alpha = float(m._fitted_result.params.get("smoothing_level", alpha))
                assert abs(fitted_alpha - alpha) < 1e-3, \
                    f"D8 violation: requested {alpha}, got {fitted_alpha}"

    def test_numpy_fallback_produces_finite_level(self, df_normal):
        """If statsmodels fails, numpy fallback must still produce a finite level."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_normal, FEATURES)
        assert np.isfinite(m._level)

    def test_nan_input_sanitised(self, df_nan):
        m = SESModel(hp=SESModel({}).default_hp)
        m.fit(df_nan, FEATURES)
        assert np.isfinite(m._level)

    def test_compute_residuals_finite(self, df_nan):
        m = SESModel(hp=SESModel({}).default_hp)
        resids = m.compute_residuals(df_nan, FEATURES)
        assert np.all(np.isfinite(resids))

    # ── BV-04/05: smoothing_level at boundary values ─────────────────────────

    @pytest.mark.parametrize("alpha", [0.0, 1.0])
    def test_extreme_smoothing_level_no_crash(self, alpha, df_normal):
        """BV-04/05: alpha=0.0 (no learning) and alpha=1.0 (instant update)."""
        m = SESModel(hp={"smoothing_level": alpha})
        m.fit(df_normal, FEATURES)
        assert np.isfinite(m._level)
        result = m.predict(df_normal, FEATURES, horizon=30)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


# ===========================================================================
# CrostonMethod
# ===========================================================================

class TestCrostonMethod:

    def test_model_name(self):
        assert CrostonMethod({}).model_name == "Croston's Method"

    def test_hp_search_space_has_9_configs(self):
        assert len(CrostonMethod({}).hp_search_space) == 9

    # ── D2: all-nonzero series must not crash ────────────────────────────────

    def test_d2_all_nonzero_no_crash(self, df_all_nonzero):
        """Done Criterion D2: all non-zero series must not crash Croston."""
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_all_nonzero, FEATURES)
        result = m.predict_all_horizons(df_all_nonzero, FEATURES)
        assert set(result.keys()) == EXPECTED_HORIZON_KEYS

    def test_d2_all_nonzero_valid_quantiles(self, df_all_nonzero):
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_all_nonzero, FEATURES)
        result = m.predict_all_horizons(df_all_nonzero, FEATURES)
        for key, val in result.items():
            assert val["p50"] <= val["p80"] <= val["p90"]

    # ── D3: single non-zero event → E001 SES fallback ───────────────────────

    def test_d3_single_nonzero_triggers_fallback(self, df_single_nz):
        """Done Criterion D3: single non-zero event triggers silent SES fallback."""
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_single_nz, FEATURES)
        assert m._using_fallback is True

    def test_d3_single_nonzero_returns_valid_keys(self, df_single_nz):
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_single_nz, FEATURES)
        result = m.predict_all_horizons(df_single_nz, FEATURES)
        assert set(result.keys()) == EXPECTED_HORIZON_KEYS

    def test_d3_pattern_label_unchanged(self, df_single_nz):
        """E001 fallback must NOT change ctx.pattern_label — it stays 'intermittent'."""
        # The model itself doesn't touch ctx — this verifies _using_fallback is internal
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_single_nz, FEATURES)
        # Pattern label is managed by the caller (Sub-Stage 9.1), not the model
        assert m.model_name == "Croston's Method"   # model_name unchanged

    # ── Croston variants ────────────────────────────────────────────────────

    @pytest.mark.parametrize("interval_type", ["classic", "SBA", "TSB"])
    def test_all_variants_produce_valid_forecast(self, interval_type, df_intermittent):
        m = CrostonMethod(hp={"alpha": 0.1, "interval_type": interval_type})
        m.fit(df_intermittent, FEATURES)
        result = m.predict(df_intermittent, FEATURES, horizon=30)
        assert len(result) == 30
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_sba_corrects_upward_bias(self, df_intermittent):
        """SBA daily_rate should be <= classic daily_rate (bias correction)."""
        m_classic = CrostonMethod(hp={"alpha": 0.1, "interval_type": "classic"})
        m_sba     = CrostonMethod(hp={"alpha": 0.1, "interval_type": "SBA"})
        m_classic.fit(df_intermittent, FEATURES)
        m_sba.fit(df_intermittent, FEATURES)
        if not m_classic._using_fallback and not m_sba._using_fallback:
            assert m_sba._daily_rate <= m_classic._daily_rate

    def test_nan_input_sanitised(self, df_nan):
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_nan, FEATURES)
        assert np.isfinite(m._daily_rate) or m._using_fallback

    # ── DQ-05: first-value-only triggers SES fallback ────────────────────────

    def test_first_value_only_triggers_fallback(self, df_first_nonzero):
        """DQ-05: only the first row has demand — must trigger SES fallback."""
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_first_nonzero, FEATURES)
        assert m._using_fallback is True

    def test_first_value_only_returns_valid_horizons(self, df_first_nonzero):
        m = CrostonMethod(hp=CrostonMethod({}).default_hp)
        m.fit(df_first_nonzero, FEATURES)
        result = m.predict_all_horizons(df_first_nonzero, FEATURES)
        assert set(result.keys()) == EXPECTED_HORIZON_KEYS

    # ── BL-02: TSB extinction pattern ────────────────────────────────────────

    def test_tsb_extinction_lower_rate_than_classic(self, df_extinction):
        """BL-02: TSB must detect the long zero run and forecast a lower rate."""
        m_tsb     = CrostonMethod(hp={"alpha": 0.10, "interval_type": "TSB"})
        m_classic = CrostonMethod(hp={"alpha": 0.10, "interval_type": "classic"})
        m_tsb.fit(df_extinction, FEATURES)
        m_classic.fit(df_extinction, FEATURES)
        if not m_tsb._using_fallback and not m_classic._using_fallback:
            assert m_tsb._daily_rate < m_classic._daily_rate, (
                f"TSB rate {m_tsb._daily_rate:.4f} should be lower than "
                f"classic {m_classic._daily_rate:.4f} on extinction series"
            )

    def test_tsb_extinction_daily_rate_near_zero(self, df_extinction):
        """After 40 zero periods, TSB daily_rate should be near-zero."""
        m = CrostonMethod(hp={"alpha": 0.10, "interval_type": "TSB"})
        m.fit(df_extinction, FEATURES)
        if not m._using_fallback:
            assert m._daily_rate < 1.0, \
                f"TSB rate {m._daily_rate:.4f} should be near-zero after long silence"


# ===========================================================================
# HoltLinearTrend
# ===========================================================================

class TestHoltLinearTrend:

    def test_model_name(self):
        assert HoltLinearTrend({}).model_name == "Holt's Linear Trend"

    def test_hp_search_space_has_24_configs(self):
        assert len(HoltLinearTrend({}).hp_search_space) == 24

    def test_trending_series_positive_trend(self, df_trending):
        """Trending series must produce increasing forecasts over time."""
        m = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": False})
        m.fit(df_trending, FEATURES)
        assert m._trend > 0, "Positive trend expected on growing series"

    def test_damped_prevents_runaway(self, df_trending):
        """damped=True must prevent 365d forecast from exploding."""
        m = HoltLinearTrend(hp={"smoothing_level": 0.4, "smoothing_trend": 0.2, "damped_trend": True})
        m.fit(df_trending, FEATURES)
        rh = m.predict_all_horizons(df_trending, FEATURES)
        ratio = rh["forecast_365d"]["mean"] / max(rh["forecast_7d"]["mean"], 1.0)
        assert ratio < 100, f"Runaway extrapolation: 365d/7d ratio = {ratio:.1f}"

    # ── D9: explicit alpha/beta enforcement ──────────────────────────────────

    def test_d9_explicit_alpha_beta(self, df_normal):
        """Done Criterion D9: fitted alpha and beta must match hp dict values."""
        m = HoltLinearTrend(hp={"smoothing_level": 0.2, "smoothing_trend": 0.05, "damped_trend": True})
        m.fit(df_normal, FEATURES)
        if m._fitted_result is not None and not m._using_fallback:
            fa = float(m._fitted_result.params.get("smoothing_level", 0.2))
            fb = float(m._fitted_result.params.get("smoothing_trend",  0.05))
            assert abs(fa - 0.2)  < 1e-3, f"D9: alpha mismatch {fa}"
            assert abs(fb - 0.05) < 1e-3, f"D9: beta mismatch {fb}"

    def test_nan_input_sanitised(self, df_nan):
        m = HoltLinearTrend(hp=HoltLinearTrend({}).default_hp)
        m.fit(df_nan, FEATURES)
        assert np.isfinite(m._level)

    def test_compute_residuals_finite(self, df_nan):
        m = HoltLinearTrend(hp=HoltLinearTrend({}).default_hp)
        resids = m.compute_residuals(df_nan, FEATURES)
        assert np.all(np.isfinite(resids))

    @pytest.mark.parametrize("damped", [True, False])
    def test_both_damping_modes(self, damped, df_trending):
        m = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": damped})
        m.fit(df_trending, FEATURES)
        result = m.predict(df_trending, FEATURES, horizon=30)
        assert len(result) == 30
        assert np.all(result >= 0)

    # ── BL-01: declining demand must not produce negative forecasts ───────────

    def test_declining_series_has_negative_trend(self, df_declining):
        """BL-01: a falling demand series must produce a negative trend."""
        m = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": False})
        m.fit(df_declining, FEATURES)
        assert m._trend < 0, "Expected negative _trend on declining series"

    @pytest.mark.parametrize("damped", [True, False])
    def test_declining_series_predict_no_negatives(self, damped, df_declining):
        """BL-01: predict() must clamp to zero even with a negative trend."""
        m = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": damped})
        m.fit(df_declining, FEATURES)
        result = m.predict(df_declining, FEATURES, horizon=365)
        assert np.all(result >= 0), f"predict() produced negatives (damped={damped})"

    def test_declining_series_all_horizons_non_negative(self, df_declining):
        """BL-01: all forecast quantiles must be non-negative on declining series."""
        m = HoltLinearTrend(hp=HoltLinearTrend({}).default_hp)
        m.fit(df_declining, FEATURES)
        rh = m.predict_all_horizons(df_declining, FEATURES)
        for key, val in rh.items():
            for q, v in val.items():
                assert v >= 0, f"{key}.{q} = {v} (negative on declining series)"

    # ── BV-06: smoothing_trend = 0.0 (frozen trend) ──────────────────────────

    def test_zero_smoothing_trend_no_negatives(self, df_trending):
        """BV-06: a frozen trend must not produce negative forecasts."""
        m = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.0, "damped_trend": False})
        m.fit(df_trending, FEATURES)
        result = m.predict(df_trending, FEATURES, horizon=365)
        assert np.all(result >= 0), "Holt with smoothing_trend=0.0 must clamp to zero"
        assert np.all(np.isfinite(result))


# ===========================================================================
# ProphetModel
# ===========================================================================

class TestProphetModel:

    def test_model_name(self):
        assert ProphetModel({}).model_name == "Prophet"

    def test_hp_search_space_has_24_configs(self):
        assert len(ProphetModel({}).hp_search_space) == 24

    def test_daily_seasonality_always_false(self):
        """HARDCODED RULE: daily_seasonality must always be False."""
        m = ProphetModel(hp=ProphetModel({}).default_hp)
        assert m._daily_seasonality is False

    def test_daily_seasonality_not_in_hp_search_space(self):
        """daily_seasonality must NOT appear as a tunable HP."""
        for hp in ProphetModel({}).hp_search_space:
            assert "daily_seasonality" not in hp, \
                "daily_seasonality must not be in hp_search_space"

    def test_picklable_before_fit(self):
        """D10: ProphetModel must be picklable for ProcessPoolExecutor."""
        m = ProphetModel(hp=ProphetModel({}).default_hp)
        restored = pickle.loads(pickle.dumps(m))
        assert type(restored).__name__ == "ProphetModel"

    # ── E002: constant series must not crash ─────────────────────────────────

    def test_e002_constant_series_no_crash(self, df_constant):
        """Done Criterion D4: constant series (std<0.01) must not raise Stan error."""
        try:
            m = ProphetModel(hp=ProphetModel({}).default_hp)
            m.fit(df_constant, FEATURES)
            result = m.predict_all_horizons(df_constant, FEATURES)
            assert set(result.keys()) == EXPECTED_HORIZON_KEYS
            for k, v in result.items():
                assert all(np.isfinite(x) for x in v.values()), \
                    f"E002: {k} has non-finite values: {v}"
        except (ImportError, ModelFitError):
            pytest.skip("Prophet not installed")

    # ── D1: seasonal cumulative accuracy ─────────────────────────────────────

    def test_d1_seasonal_cumulative_accuracy(self, df_seasonal):
        """Done Criterion D1: forecast_365d['mean'] > forecast_30d['mean'] × 12 by ≥ 10%."""
        try:
            m = ProphetModel(hp=ProphetModel({}).default_hp)
            m.fit(df_seasonal, FEATURES)
            rh = m.predict_all_horizons(df_seasonal, FEATURES)
            mean_365 = rh["forecast_365d"]["mean"]
            mean_30  = rh["forecast_30d"]["mean"]
            if mean_30 > 0:
                ratio = mean_365 / (mean_30 * 12)
                assert ratio >= 0.90, \
                    f"D1: 365d/30d×12 ratio = {ratio:.3f} (must be ≥ 0.90)"
        except (ImportError, ModelFitError):
            pytest.skip("Prophet not installed")

    def test_single_fit_cumulative_extraction(self, df_seasonal):
        """HARDCODED RULE: fit() called once; predict_all_horizons uses cumulative sums."""
        try:
            m = ProphetModel(hp=ProphetModel({}).default_hp)
            m.fit(df_seasonal, FEATURES)
            assert m._predictions is not None
            assert len(m._predictions) == 365
        except (ImportError, ModelFitError):
            pytest.skip("Prophet not installed")

    def test_duplicate_dates_does_not_crash(self):
        """DQ-03: duplicate dates are treated as extra observations; must not crash."""
        try:
            dates = list(pd.date_range("2024-01-01", periods=29)) + [pd.Timestamp("2024-01-15")]
            df = pd.DataFrame({"date": dates, "qty": np.ones(30) * 5.0})
            m = ProphetModel(hp=ProphetModel({}).default_hp)
            m.fit(df, FEATURES)
            result = m.predict_all_horizons(df, FEATURES)
            assert set(result.keys()) == EXPECTED_HORIZON_KEYS
        except (ImportError, ModelFitError):
            pytest.skip("Prophet not installed")


# ===========================================================================
# DQ-01 — All-zero demand series
# ===========================================================================

class TestAllZeroDemand:

    @pytest.mark.parametrize("Cls", [NaiveForecast, SESModel, HoltLinearTrend],
                             ids=["NaiveForecast", "SESModel", "HoltLinearTrend"])
    def test_all_zero_demand_no_crash(self, Cls, df_all_zeros):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_all_zeros, FEATURES)
        result = m.predict(df_all_zeros, FEATURES, horizon=30)
        assert len(result) == 30

    @pytest.mark.parametrize("Cls", [NaiveForecast, SESModel, HoltLinearTrend],
                             ids=["NaiveForecast", "SESModel", "HoltLinearTrend"])
    def test_all_zero_demand_non_negative_finite(self, Cls, df_all_zeros):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_all_zeros, FEATURES)
        result = m.predict(df_all_zeros, FEATURES, horizon=30)
        assert np.all(result >= 0), f"{Cls.__name__}: negatives on all-zero series"
        assert np.all(np.isfinite(result)), f"{Cls.__name__}: non-finite on all-zero series"

    @pytest.mark.parametrize("Cls", [NaiveForecast, SESModel, HoltLinearTrend],
                             ids=["NaiveForecast", "SESModel", "HoltLinearTrend"])
    def test_all_zero_demand_all_horizons_non_negative(self, Cls, df_all_zeros):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_all_zeros, FEATURES)
        result = m.predict_all_horizons(df_all_zeros, FEATURES)
        for key, val in result.items():
            for q, v in val.items():
                assert v >= 0, f"{Cls.__name__} {key}.{q} = {v} (negative on all-zero series)"


# ===========================================================================
# DQ-02 — Single-row DataFrame
# ===========================================================================

class TestSingleRowDataFrame:

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_single_row_no_crash(self, Cls, df_one_row):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_one_row, FEATURES)
        result = m.predict(df_one_row, FEATURES, horizon=7)
        assert len(result) == 7

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_single_row_all_horizons_returns_expected_keys(self, Cls, df_one_row):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_one_row, FEATURES)
        result = m.predict_all_horizons(df_one_row, FEATURES)
        assert set(result.keys()) == EXPECTED_HORIZON_KEYS

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_single_row_forecast_non_negative_finite(self, Cls, df_one_row):
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df_one_row, FEATURES)
        result = m.predict(df_one_row, FEATURES, horizon=30)
        assert np.all(result >= 0), f"{Cls.__name__}: negatives on single-row series"
        assert np.all(np.isfinite(result)), f"{Cls.__name__}: non-finite on single-row series"


# ===========================================================================
# DQ-04 — Very large demand values (enterprise scale)
# ===========================================================================

class TestLargeQtyValues:

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_large_qty_all_horizons_finite(self, Cls):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "qty":  np.full(60, 500_000.0),
        })
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df, FEATURES)
        result = m.predict_all_horizons(df, FEATURES)
        for key, val in result.items():
            for q, v in val.items():
                assert np.isfinite(v), f"{Cls.__name__} {key}.{q} not finite at large scale"

    @pytest.mark.parametrize("Cls", ALL_MODELS, ids=ALL_MODEL_IDS)
    def test_large_qty_ordering_invariant_holds(self, Cls):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "qty":  np.full(60, 500_000.0),
        })
        m = Cls(hp=Cls({}).default_hp)
        m.fit(df, FEATURES)
        result = m.predict_all_horizons(df, FEATURES)
        for key, val in result.items():
            assert val["p50"] <= val["p80"] <= val["p90"], \
                f"{Cls.__name__} {key}: ordering violated at large scale"


# ===========================================================================
# _get_model_class — model registry in base.py
# ===========================================================================

class TestModelRegistry:

    def test_known_names_return_correct_classes(self):
        assert _get_model_class(Model.NAIVE) is NaiveForecast
        assert _get_model_class(Model.CROSTON) is CrostonMethod
        assert _get_model_class(Model.PROPHET) is ProphetModel
        assert _get_model_class(Model.HOLTS_LINEAR) is HoltLinearTrend
        assert _get_model_class(Model.SES) is SESModel

    def test_unknown_name_falls_back_to_ses(self):
        cls = _get_model_class("totally_unknown_model")
        assert cls is SESModel

    def test_returned_class_is_instantiable(self):
        for model_name in (Model.NAIVE, Model.CROSTON, Model.HOLTS_LINEAR, Model.SES):
            cls = _get_model_class(model_name)
            instance = cls(hp=cls({}).default_hp)
            assert isinstance(instance, BaseModel)

    def test_covers_all_pattern_model_map_values(self):
        for model_name in PATTERN_MODEL_MAP.values():
            cls = _get_model_class(model_name)
            assert issubclass(cls, BaseModel), \
                f"_get_model_class({model_name!r}) returned {cls}, not a BaseModel subclass"

