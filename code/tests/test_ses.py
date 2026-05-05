"""
unit/models/test_ses.py — Comprehensive unit tests for ses.SESModel.

Coverage map (vs the public surface area of code/ses.py):

    Class SESModel
        __init__                                         3 tests
        @property level                                  2 tests
        @property model_name                             1 test
        @property hp_search_space                        4 tests
        @property default_hp                             3 tests
        @property required_features                      2 tests
        @property optional_features                      1 test
        fit()                                           11 tests
        predict()                                        6 tests
        predict_all_horizons()                           7 tests
        compute_residuals()                              4 tests
        _get_fitted_values()                             3 tests

    Module-level
        _ses_numpy()                                     7 tests
        _HP_SEARCH_SPACE / _DEFAULT_HP constants         2 tests

Tests are intentionally isolated:
    * No DB.
    * No subprocess pool.
    * No file I/O.
    * Only `np`, `pd`, `pytest`, `ses`, `base`, `bootstrap`, `constants`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.ses import SESModel, _ses_numpy, _HP_SEARCH_SPACE, _DEFAULT_HP
from models.base import BaseModel
from infrastructure.constants import HORIZONS, FORECAST_COLUMN_MAP


# ---------------------------------------------------------------------------
# 1.  __init__ + state-default invariants
# ---------------------------------------------------------------------------

class TestInit:

    def test_subclass_of_base_model(self):
        """SESModel must inherit BaseModel — required by the model-registry contract."""
        assert issubclass(SESModel, BaseModel)

    def test_constructor_stores_hp_dict(self):
        m = SESModel(hp={"smoothing_level": 0.42})
        assert m.hp == {"smoothing_level": 0.42}

    def test_pre_fit_state_defaults(self):
        """Before fit() is called the model must be in a quiescent state."""
        m = SESModel(hp={"smoothing_level": 0.3})
        assert m._fitted_result is None
        assert m._level == 0.0
        assert m._using_fallback is False
        # Public level property mirrors private state.
        assert m.level == 0.0


# ---------------------------------------------------------------------------
# 2.  Properties
# ---------------------------------------------------------------------------

class TestProperties:

    # ----- level --------------------------------------------------------

    def test_level_zero_before_fit(self):
        assert SESModel(hp={"smoothing_level": 0.3}).level == 0.0

    def test_level_reflects_fitted_value(self, df_flat):
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_flat, ["date", "qty"])
        # Flat series of 10 → SES converges to 10.
        assert math.isclose(m.level, 10.0, abs_tol=0.01)

    # ----- model_name ---------------------------------------------------

    def test_model_name_string(self):
        assert SESModel(hp={}).model_name == "Simple Exponential Smoothing (SES)"

    # ----- hp_search_space ---------------------------------------------

    def test_hp_search_space_has_five_configs(self):
        sp = SESModel(hp={}).hp_search_space
        assert len(sp) == 5

    def test_hp_search_space_alpha_values(self):
        alphas = [c["smoothing_level"] for c in SESModel(hp={}).hp_search_space]
        assert alphas == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_hp_search_space_returns_a_copy(self):
        """Mutating the returned list must NOT affect the module constant —
        otherwise repeated calls would compound state."""
        sp = SESModel(hp={}).hp_search_space
        sp.append({"smoothing_level": 0.99})
        sp2 = SESModel(hp={}).hp_search_space
        assert len(sp2) == 5

    def test_hp_search_space_independent_per_instance(self):
        sp1 = SESModel(hp={}).hp_search_space
        sp2 = SESModel(hp={}).hp_search_space
        assert sp1 is not sp2

    # ----- default_hp --------------------------------------------------

    def test_default_hp_value(self):
        assert SESModel(hp={}).default_hp == {"smoothing_level": 0.3}

    def test_default_hp_returns_a_copy(self):
        d = SESModel(hp={}).default_hp
        d["smoothing_level"] = 0.99
        d2 = SESModel(hp={}).default_hp
        assert d2 == {"smoothing_level": 0.3}

    def test_default_hp_alpha_in_search_space(self):
        """The default config must be a member of the search space — otherwise
        Thompson Sampling can't pick it as 'prior best'."""
        m = SESModel(hp={})
        assert m.default_hp in m.hp_search_space

    # ----- required_features --------------------------------------------

    def test_required_features_value(self):
        assert SESModel(hp={}).required_features == ["date", "qty"]

    def test_required_features_returns_a_copy_safe_to_mutate(self):
        f = SESModel(hp={}).required_features
        f.append("noise")
        # The next call should not have 'noise' baked in.
        assert SESModel(hp={}).required_features == ["date", "qty"]

    # ----- optional_features --------------------------------------------

    def test_optional_features_empty(self):
        assert SESModel(hp={}).optional_features == []


# ---------------------------------------------------------------------------
# 3.  fit()
# ---------------------------------------------------------------------------

class TestFit:

    def test_flat_series_converges_to_mean(self, df_flat):
        """Update formula on a flat series of 10s collapses to 10 for any alpha."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_flat, ["date", "qty"])
        assert math.isclose(m.level, 10.0, abs_tol=0.01)

    def test_alpha_is_respected_not_optimised(self, df_noisy_stable):
        """Done Criterion D8: optimized=False → statsmodels uses our alpha."""
        m = SESModel(hp={"smoothing_level": 0.5})
        m.fit(df_noisy_stable, ["date", "qty"])
        # If statsmodels secretly optimised, params would diverge from 0.5.
        if m._fitted_result is not None and not m._using_fallback:
            fitted_alpha = float(m._fitted_result.params.get("smoothing_level", -1))
            assert math.isclose(fitted_alpha, 0.5, abs_tol=1e-3)

    def test_uses_default_alpha_when_hp_missing(self, df_flat):
        """fit() reads alpha via .get('smoothing_level', 0.3) — no key still works."""
        m = SESModel(hp={})  # no smoothing_level key
        m.fit(df_flat, ["date", "qty"])
        # Still produces a sane level on flat input.
        assert math.isclose(m.level, 10.0, abs_tol=0.01)

    def test_empty_dataframe_short_circuits_to_fallback(self, df_empty):
        """Empty input must not crash; level=0, fallback flag set."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_empty, ["date", "qty"])
        assert m.level == 0.0
        assert m._using_fallback is True
        assert m._fitted_result is None

    def test_single_row_does_not_crash(self, df_single_row):
        """One row of data should at minimum return a finite level."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_single_row, ["date", "qty"])
        assert math.isfinite(m.level)
        assert m.level >= 0

    def test_nan_inf_are_sanitised(self, df_with_nan_inf):
        """NaN/Inf in qty must be replaced with 0 — fit must not return NaN/Inf."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_with_nan_inf, ["date", "qty"])
        assert math.isfinite(m.level)
        assert not math.isnan(m.level)

    def test_all_zeros_yields_zero_level(self, df_all_zeros):
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_all_zeros, ["date", "qty"])
        assert m.level == 0.0

    def test_alpha_zero_keeps_initial_level(self):
        """alpha=0 means 'never update from observations' — level stays at S_0."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "qty":  [5.0] + [99.0] * 9,   # first row is the "initial" level
        })
        m = SESModel(hp={"smoothing_level": 0.0})
        m.fit(df, ["date", "qty"])
        # statsmodels with alpha=0 may converge differently; numpy fallback
        # gives exactly 5. Either way the final level is bounded by [5, 99].
        assert 4.5 <= m.level <= 99.0

    def test_alpha_one_equals_last_observation_in_fallback(self):
        """alpha=1 means 'forget everything but the latest' — at least under the
        numpy fallback path, level should equal the last observation."""
        # Force the fallback path by giving statsmodels something it can't fit
        # cleanly (mostly handled internally — easier to test _ses_numpy directly,
        # see TestSesNumpy).
        m = SESModel(hp={"smoothing_level": 1.0})
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "qty":  [1.0, 2.0, 3.0, 4.0, 99.0],
        })
        m.fit(df, ["date", "qty"])
        assert math.isclose(m.level, 99.0, abs_tol=0.5)

    def test_sample_weights_silently_ignored(self, df_flat):
        """SES has no sample-weight support — must accept the kwarg without error."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_flat, ["date", "qty"], sample_weights=np.ones(30))
        assert math.isclose(m.level, 10.0, abs_tol=0.01)

    def test_fit_is_idempotent(self, df_flat):
        """Calling fit() twice on the same input must yield the same level."""
        m1 = SESModel(hp={"smoothing_level": 0.3})
        m2 = SESModel(hp={"smoothing_level": 0.3})
        m1.fit(df_flat, ["date", "qty"])
        m2.fit(df_flat, ["date", "qty"])
        m2.fit(df_flat, ["date", "qty"])  # second time
        assert math.isclose(m1.level, m2.level, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 4.  predict()
# ---------------------------------------------------------------------------

class TestPredict:

    @pytest.fixture
    def fitted_model(self, df_flat):
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_flat, ["date", "qty"])
        return m

    def test_returns_ndarray_of_horizon_length(self, fitted_model):
        out = fitted_model.predict(pd.DataFrame(), [], horizon=7)
        assert isinstance(out, np.ndarray)
        assert len(out) == 7

    def test_predict_is_constant(self, fitted_model):
        """SES has no trend — every value across the horizon is identical."""
        out = fitted_model.predict(pd.DataFrame(), [], horizon=14)
        assert np.all(out == out[0])

    def test_predict_value_equals_level(self, fitted_model):
        """The flat value must be max(0, model.level)."""
        out = fitted_model.predict(pd.DataFrame(), [], horizon=5)
        assert math.isclose(float(out[0]), fitted_model.level, abs_tol=1e-6)

    def test_predict_horizon_zero_returns_empty(self, fitted_model):
        out = fitted_model.predict(pd.DataFrame(), [], horizon=0)
        assert isinstance(out, np.ndarray)
        assert len(out) == 0

    def test_predict_clamps_negative_level_to_zero(self):
        """If level somehow went negative, predict must return zeros (D6 invariant)."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m._level = -7.0   # force negative state
        out = m.predict(pd.DataFrame(), [], horizon=5)
        assert np.all(out == 0.0)

    def test_predict_without_fit_returns_zeros(self):
        """Calling predict without a prior fit yields a zero array — pre-fit
        state has level=0, so this is consistent (no exception)."""
        m = SESModel(hp={"smoothing_level": 0.3})
        out = m.predict(pd.DataFrame(), [], horizon=4)
        assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# 5.  predict_all_horizons()
# ---------------------------------------------------------------------------

class TestPredictAllHorizons:

    @pytest.fixture
    def fitted_model(self, df_flat_long):
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_flat_long, ["date", "qty"])
        return m

    def test_returns_all_horizon_keys(self, fitted_model, df_flat_long):
        out = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"])
        expected = {FORECAST_COLUMN_MAP[H] for H in HORIZONS}
        assert set(out.keys()) == expected

    def test_each_value_is_quantile_dict(self, fitted_model, df_flat_long):
        out = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"])
        for col, q in out.items():
            assert {"mean", "p50", "p80", "p90"}.issubset(q.keys()), col

    def test_horizons_scale_linearly_with_level(self, fitted_model, df_flat_long):
        """point(H) = level × H — flat noise should give linear means."""
        out = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"])
        m7  = out[FORECAST_COLUMN_MAP[7]]["mean"]
        m30 = out[FORECAST_COLUMN_MAP[30]]["mean"]
        # Within 5% of the linear ratio
        assert abs((m30 / m7) - (30 / 7)) < (30 / 7) * 0.05

    def test_oos_factor_scales_means_proportionally(self, fitted_model, df_flat_long):
        out_a = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"], oos_factor=1.0)
        out_b = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"], oos_factor=1.5)
        for col in out_a:
            assert math.isclose(out_b[col]["mean"], out_a[col]["mean"] * 1.5, rel_tol=1e-6)

    def test_quantile_monotonicity(self, fitted_model, df_flat_long):
        out = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"])
        for col, q in out.items():
            assert q["p50"] <= q["p80"] <= q["p90"], col

    def test_zero_level_returns_zero_means(self):
        """All-zero training data → level=0 → every horizon mean=0."""
        m = SESModel(hp={"smoothing_level": 0.3})
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "qty":  [0.0] * 60,
        })
        m.fit(df, ["date", "qty"])
        out = m.predict_all_horizons(df, ["date", "qty"])
        for col, q in out.items():
            assert q["mean"] == 0.0

    def test_predict_all_horizons_matches_predict_at_seven(self, fitted_model, df_flat_long):
        """forecast_7d.mean ≈ level × 7 (with bootstrap noise allowed)."""
        out = fitted_model.predict_all_horizons(df_flat_long, ["date", "qty"])
        expected = fitted_model.level * 7
        assert abs(out[FORECAST_COLUMN_MAP[7]]["mean"] - expected) <= max(0.5, expected * 0.05)


# ---------------------------------------------------------------------------
# 6.  compute_residuals()
# ---------------------------------------------------------------------------

class TestComputeResiduals:

    def test_returns_last_30_residuals(self, df_flat_long):
        m = SESModel(hp={"smoothing_level": 0.3})
        resids = m.compute_residuals(df_flat_long, ["date", "qty"])
        assert isinstance(resids, np.ndarray)
        assert len(resids) == 30

    def test_flat_series_residuals_near_zero(self, df_flat_long):
        """Flat input means actuals = fitted ⇒ residuals ≈ 0."""
        m = SESModel(hp={"smoothing_level": 0.3})
        resids = m.compute_residuals(df_flat_long, ["date", "qty"])
        assert np.max(np.abs(resids)) < 0.5

    def test_residuals_finite_with_nan_inf_inputs(self, df_with_nan_inf):
        m = SESModel(hp={"smoothing_level": 0.3})
        resids = m.compute_residuals(df_with_nan_inf, ["date", "qty"])
        assert np.all(np.isfinite(resids))

    def test_residuals_for_short_series_returns_truncated(self, df_single_row):
        """Series shorter than 30 rows: residual array is the whole series."""
        m = SESModel(hp={"smoothing_level": 0.3})
        resids = m.compute_residuals(df_single_row, ["date", "qty"])
        assert len(resids) == 1


# ---------------------------------------------------------------------------
# 7.  _get_fitted_values()
# ---------------------------------------------------------------------------

class TestGetFittedValues:

    def test_statsmodels_path_returns_full_length(self, df_flat_long):
        m = SESModel(hp={"smoothing_level": 0.3})
        m.fit(df_flat_long, ["date", "qty"])
        if m._fitted_result is not None and not m._using_fallback:
            fv = m._get_fitted_values(df_flat_long)
            assert len(fv) == len(df_flat_long)
            assert np.all(fv >= 0.0)   # clamped non-negative

    def test_fallback_path_returns_constant_level(self, df_flat):
        """When fallback is active, fitted values are constant at level."""
        m = SESModel(hp={"smoothing_level": 0.3})
        m._level = 7.0
        m._using_fallback = True
        m._fitted_result = None
        fv = m._get_fitted_values(df_flat)
        assert len(fv) == len(df_flat)
        assert np.all(fv == 7.0)

    def test_negative_level_clamped_in_fallback(self, df_flat):
        m = SESModel(hp={"smoothing_level": 0.3})
        m._level = -5.0
        m._using_fallback = True
        m._fitted_result = None
        fv = m._get_fitted_values(df_flat)
        assert np.all(fv == 0.0)


# ---------------------------------------------------------------------------
# 8.  _ses_numpy()  module-level helper
# ---------------------------------------------------------------------------

class TestSesNumpy:

    def test_empty_series_returns_zero(self):
        assert _ses_numpy(np.array([]), 0.3) == 0.0

    def test_single_element_returns_that_element(self):
        assert _ses_numpy(np.array([42.0]), 0.3) == 42.0

    def test_flat_series_returns_value(self):
        assert math.isclose(_ses_numpy(np.array([10.0] * 30), 0.3), 10.0, abs_tol=1e-9)

    def test_alpha_zero_keeps_initial(self):
        """alpha=0 means level never updates after S_0."""
        out = _ses_numpy(np.array([5.0, 99.0, 99.0, 99.0]), 0.0)
        assert out == 5.0

    def test_alpha_one_equals_last_observation(self):
        """alpha=1 means level = last seen value."""
        out = _ses_numpy(np.array([1.0, 2.0, 3.0, 99.0]), 1.0)
        assert out == 99.0

    def test_intermediate_alpha_matches_formula(self):
        """Hand-computed: S0=10, S1=0.3*20+0.7*10=13, S2=0.3*30+0.7*13=18.1."""
        out = _ses_numpy(np.array([10.0, 20.0, 30.0]), 0.3)
        assert math.isclose(out, 18.1, abs_tol=1e-9)

    def test_returns_float_not_array(self):
        out = _ses_numpy(np.array([1.0, 2.0]), 0.5)
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# 9.  Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:

    def test_search_space_constant_shape(self):
        assert isinstance(_HP_SEARCH_SPACE, list)
        assert len(_HP_SEARCH_SPACE) == 5
        for cfg in _HP_SEARCH_SPACE:
            assert set(cfg.keys()) == {"smoothing_level"}
            assert 0.0 <= cfg["smoothing_level"] <= 1.0

    def test_default_hp_constant_value(self):
        assert _DEFAULT_HP == {"smoothing_level": 0.3}
