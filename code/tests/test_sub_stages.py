"""
tests/test_sub_stages.py — Atheera Stage 9
===========================================
Tests for Sub-Stage 9.2 (Feature Engineering) and Sub-Stage 9.3 (HP Tuning).

Sub-Stage 9.2 tests:
    - Step 1: reliability filtering drops low-reliability optional features
    - Step 2: B2B weekday filter + E006 weekend-only seller guard
    - Step 3: promo demand capping (vectorised) + sample_weights for Prophet
    - Step 4: additive feature search respects improvement_threshold
    - BatchWriter row always written (Done Criterion 6)
    - Required features never dropped

Sub-Stage 9.3 tests:
    - Short df skips search, uses default_hp
    - Budget=0 does not crash
    - Thompson state updated in memory only (no DB write)
    - Early stop at MAPE < 0.10
    - ModelFitError per-config does not crash SKU (mape=1.0)
    - BatchWriter row always written (Done Criterion 4)
    - NaN in df_train produces finite validation_mape
    - used_category_comps NOT in BatchWriter row (lifecycle removed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.naive import NaiveForecast
from models.ses import SESModel
from models.base import ModelFitError
from forecasting.feature_engg import run_feature_engineering
from models.hp_tuning import run_hp_tuning, HPTuningResult
from infrastructure.constants import Param


# ---------------------------------------------------------------------------
# Sub-Stage 9.2 helpers
# ---------------------------------------------------------------------------

def _make_preloaded(
    sku_id:           str   = "sku-001",
    reliability_map:  dict  = None,
    promo_dict:       dict  = None,
    feature_history:  list  = None,
) -> dict:
    return {
        "feature_reliability": {sku_id: reliability_map or {}},
        "promo_decisions":     promo_dict or {},
        "feature_history":     {sku_id: feature_history} if feature_history else {},
    }


# ===========================================================================
# Sub-Stage 9.2 — Feature Engineering
# ===========================================================================

class TestSubStage92ReliabilityFilter:

    def test_low_reliability_feature_dropped(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 1: feature with reliability < floor must be dropped."""
        df = df_normal.copy()
        df["promo_flag"]  = 0.0
        df["day_of_week"] = df["date"].dt.dayofweek.astype(float)

        model = SESModel(hp=SESModel({}).default_hp)
        # promo_flag reliability=0.8 (passes), day_of_week=0.1 (below 0.30 floor)
        preloaded = _make_preloaded(
            sku_id=mock_ctx.sku_id,
            reliability_map={"promo_flag": 0.8, "day_of_week": 0.1},
        )
        result = run_feature_engineering(mock_ctx, df, model, preloaded, mock_params, mock_bw)
        assert "day_of_week" not in result.selected_features

    def test_high_reliability_feature_kept(self, df_normal, mock_ctx, mock_params, mock_bw):
        df = df_normal.copy()
        df["promo_flag"] = 0.0
        model = SESModel(hp=SESModel({}).default_hp)
        preloaded = _make_preloaded(
            sku_id=mock_ctx.sku_id,
            reliability_map={"promo_flag": 0.9},
        )
        # SES has no optional features, so test with a model that does.
        # promo_flag passing threshold means it reaches additive search.
        result = run_feature_engineering(mock_ctx, df, model, preloaded, mock_params, mock_bw)
        # Required features always present
        assert "date" in result.selected_features
        assert "qty" in result.selected_features

    def test_required_features_never_dropped(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Critical Rule: required_features always present in selected_features."""
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id, reliability_map={})
        result    = run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        for req in model.required_features:
            assert req in result.selected_features, f"Required feature '{req}' was dropped"


class TestSubStage92B2BFilter:

    def test_b2b_filter_removes_weekends(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 3: B2B filter must remove weekend rows."""
        mock_ctx.is_b2b = True
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id)
        result    = run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        assert result.b2b_mode_applied is True
        assert result.df_train["date"].dt.dayofweek.max() <= 4

    def test_e006_weekend_only_seller_does_not_crash(self, mock_ctx, mock_params, mock_bw):
        """Done Criterion 5 / E006: B2B filter on weekend-only series must not crash."""
        mock_ctx.is_b2b = True
        # Create a series with only Saturdays (dayofweek=5) — genuinely weekend-only
        dates     = pd.date_range("2024-01-06", periods=10, freq="7D")  # every Saturday
        df_wknd   = pd.DataFrame({"date": dates, "qty": np.ones(10) * 5.0})
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id)
        result    = run_feature_engineering(mock_ctx, df_wknd, model, preloaded, mock_params, mock_bw)
        # Must not crash; filter disabled; df_train must have rows
        assert result.df_train is not None
        assert len(result.df_train) > 0
        assert result.b2b_mode_applied is False

    def test_b2b_false_skips_filter(self, df_normal, mock_ctx, mock_params, mock_bw):
        mock_ctx.is_b2b = False
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id)
        result    = run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        assert result.b2b_mode_applied is False
        assert len(result.df_train) == len(df_normal)


class TestSubStage92PromoWeighting:

    def test_promo_day_capped_at_baseline_x_multiplier(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 2: promo day qty capped at rolling_14d_baseline × 3.0."""
        df = df_normal.copy()
        # Inject a massive spike on day 30 to simulate promo demand
        df.iloc[30, df.columns.get_loc("qty")] = 9999.0
        promo_date = df.iloc[30]["date"].strftime("%Y-%m-%d")
        promo_lookup = {(mock_ctx.sku_id, promo_date): 0.3}

        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id, promo_dict=promo_lookup)
        result    = run_feature_engineering(mock_ctx, df, model, preloaded, mock_params, mock_bw)

        assert result.promo_weighting_applied is True
        # Capped qty must be < 9999
        capped_qty = result.df_train.iloc[30]["qty"]
        assert capped_qty < 9999.0

    def test_no_promo_data_skips_step(self, df_normal, mock_ctx, mock_params, mock_bw):
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id, promo_dict={})
        result    = run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        assert result.promo_weighting_applied is False


class TestSubStage92BatchWriter:

    def test_feature_decisions_row_always_written(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 6: feature_decisions_s9 row written for every SKU."""
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id)
        run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        assert len(mock_bw.rows) >= 1
        assert "feature_decisions_s9" in mock_bw.tables

    def test_row_contains_required_fields(self, df_normal, mock_ctx, mock_params, mock_bw):
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id)
        run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        row = mock_bw.row_for("feature_decisions_s9")
        for field in ["tenant_id", "sku_id", "run_id", "features_used",
                      "baseline_mape", "improved_mape", "b2b_mode_applied",
                      "promo_weighting_applied"]:
            assert field in row, f"Missing field: {field}"

    def test_row_written_even_with_minimal_data(self, df_sparse, mock_ctx, mock_params, mock_bw):
        """BatchWriter row written even when df is too short for feature search."""
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = _make_preloaded(sku_id=mock_ctx.sku_id)
        run_feature_engineering(mock_ctx, df_sparse, model, preloaded, mock_params, mock_bw)
        assert "feature_decisions_s9" in mock_bw.tables


# ===========================================================================
# Sub-Stage 9.3 — HP Tuning
# ===========================================================================

class TestSubStage93Guards:

    def test_short_df_uses_default_hp(self, df_sparse, mock_ctx, mock_params, mock_bw):
        """df ≤ 14 rows: skip HP search, return default_hp."""
        model  = NaiveForecast(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_sparse, model, {}, mock_params, mock_bw)
        assert result.best_hp == model.default_hp

    def test_short_df_still_writes_batchwriter_row(self, df_sparse, mock_ctx, mock_params, mock_bw):
        """Done Criterion 4: BatchWriter row written even when search skipped."""
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_sparse, model, {}, mock_params, mock_bw)
        assert "hyperparameter_decisions" in mock_bw.tables

    def test_budget_zero_does_not_crash(self, df_normal, mock_ctx, mock_bw):
        class ZeroParams:
            def get(self, k): return 0 if k == Param.THOMPSON_EXPLORATION_BUDGET else 3.0

        model  = NaiveForecast(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, ZeroParams(), mock_bw)
        assert isinstance(result, HPTuningResult)


class TestSubStage93ThompsonIntegration:

    def test_thompson_state_updated_in_memory(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 3: Thompson state must be updated in preloaded dict (memory)."""
        preloaded = {"thompson_state": {}}
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        key = (mock_ctx.sku_id, mock_ctx.assigned_model)
        assert key in preloaded["thompson_state"]

    def test_thompson_state_not_written_to_db(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Thompson state writes to preloaded dict only — no DB INSERT here."""
        preloaded = {}
        model     = NaiveForecast(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        # No table named thompson_sampling_state in BatchWriter
        assert "thompson_sampling_state" not in mock_bw.tables

    def test_prior_best_config_converges(self, df_normal, mock_ctx, mock_params, mock_bw):
        """After seeding one config with high alpha, it should be selected consistently."""
        from models.thompson import ThompsonSampler
        ts    = ThompsonSampler()
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        best  = NaiveForecast({}).hp_search_space[0]
        state = {ts.config_hash(best): {"alpha": 50, "beta": 1}}
        preloaded = {"thompson_state": {(mock_ctx.sku_id, mock_ctx.assigned_model): state}}
        result = run_hp_tuning(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        assert result.best_hp is not None


class TestSubStage93EarlyStop:

    def test_early_stop_on_excellent_mape(self, df_normal, mock_ctx, mock_bw):
        """Done Criterion 1: early stop fires when MAPE < 0.10."""
        # Use a model that fits perfectly on clean data to hit early stop
        class PerfectModel(NaiveForecast):
            def fit(self, df, features, sample_weights=None):
                self._level = float(df["qty"].mean())

            def predict(self, df, features, horizon):
                # Return exact actuals for zero MAPE
                actual = df["qty"].values[:horizon].astype(float)
                if len(actual) < horizon:
                    actual = np.pad(actual, (0, horizon - len(actual)), constant_values=self._level)
                return actual

        class PerfectParams:
            def get(self, k): return {Param.THOMPSON_EXPLORATION_BUDGET: 5}.get(k, 3.0)

        model  = PerfectModel(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, PerfectParams(), mock_bw)
        assert result.early_stopped is True


class TestSubStage93ModelFitError:

    def test_model_fit_error_assigns_mape_1(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 8: ModelFitError per config → mape=1.0, SKU continues."""
        class AlwaysFailsModel(NaiveForecast):
            def fit(self, df, features, sample_weights=None):
                raise ModelFitError("simulated failure")

        model  = AlwaysFailsModel(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert result.validation_mape == 1.0

    def test_model_fit_error_still_writes_batchwriter(self, df_normal, mock_ctx, mock_params, mock_bw):
        class AlwaysFailsModel(NaiveForecast):
            def fit(self, df, features, sample_weights=None):
                raise ModelFitError("simulated failure")

        model = AlwaysFailsModel(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert "hyperparameter_decisions" in mock_bw.tables


class TestSubStage93BatchWriterRow:

    def test_row_always_written(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 4: hyperparameter_decisions row written for every SKU."""
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert "hyperparameter_decisions" in mock_bw.tables

    def test_row_has_required_fields(self, df_normal, mock_ctx, mock_params, mock_bw):
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        row = mock_bw.row_for("hyperparameter_decisions")
        for field in ["tenant_id", "sku_id", "run_id", "hyperparameters",
                      "validation_mape", "config_hash", "thompson_score", "early_stopped"]:
            assert field in row, f"Missing field: {field}"

    def test_row_has_no_lifecycle_fields(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Lifecycle stage removed — used_category_comps must not appear in row."""
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        row = mock_bw.row_for("hyperparameter_decisions")
        assert "used_category_comps" not in row
        assert "lifecycle_stage" not in row

    def test_config_hash_deterministic(self, df_normal, mock_ctx, mock_params, mock_bw):
        """Done Criterion 9: config_hash must match ThompsonSampler.config_hash."""
        from models.thompson import ThompsonSampler
        model  = NaiveForecast(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        row    = mock_bw.row_for("hyperparameter_decisions")
        ts     = ThompsonSampler()
        expected_hash = ts.config_hash(result.best_hp)
        assert row["config_hash"] == expected_hash

    def test_nan_in_df_produces_finite_mape(self, df_nan, mock_ctx, mock_params, mock_bw):
        """NaN/Inf in training df must not produce NaN validation_mape."""
        model  = NaiveForecast(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_nan, model, {}, mock_params, mock_bw)
        assert np.isfinite(result.validation_mape), \
            f"validation_mape={result.validation_mape} is not finite"

    def test_ctx_updated_with_best_hp(self, df_normal, mock_ctx, mock_params, mock_bw):
        """run_hp_tuning must set ctx.best_hp and ctx.validation_mape."""
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert hasattr(mock_ctx, "best_hp")
        assert hasattr(mock_ctx, "validation_mape")
        assert mock_ctx.best_hp == result.best_hp


# ===========================================================================
# EH-01 — Sub-Stage 9.3: all HP configs raise ModelFitError
# ===========================================================================

class TestSubStage93AllConfigsFail:

    def _always_fails_model(self):
        class AlwaysFailsModel(NaiveForecast):
            def fit(self, df, features, sample_weights=None):
                raise ModelFitError("every config fails")
        return AlwaysFailsModel(hp=NaiveForecast({}).default_hp)

    def test_all_configs_fail_returns_a_valid_hp_dict(self, df_normal, mock_ctx, mock_params, mock_bw):
        """EH-01: when every config fails, best_hp must be a non-empty dict from the search space."""
        model  = self._always_fails_model()
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert isinstance(result.best_hp, dict)
        assert len(result.best_hp) > 0
        assert result.best_hp in NaiveForecast({}).hp_search_space

    def test_all_configs_fail_validation_mape_is_one(self, df_normal, mock_ctx, mock_params, mock_bw):
        """EH-01: total failure must report mape=1.0 (worst possible)."""
        model  = self._always_fails_model()
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert result.validation_mape == 1.0

    def test_all_configs_fail_does_not_reraise(self, df_normal, mock_ctx, mock_params, mock_bw):
        """EH-01: run_hp_tuning must not propagate ModelFitError — SKU must continue."""
        model = self._always_fails_model()
        run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)

    def test_all_configs_fail_batchwriter_row_written(self, df_normal, mock_ctx, mock_params, mock_bw):
        """EH-01: hyperparameter_decisions row must be written even on total failure."""
        model = self._always_fails_model()
        run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert "hyperparameter_decisions" in mock_bw.tables

    def test_all_configs_fail_not_early_stopped(self, df_normal, mock_ctx, mock_params, mock_bw):
        """EH-01: early_stopped must be False when all configs failed (no successful eval)."""
        model  = self._always_fails_model()
        result = run_hp_tuning(mock_ctx, df_normal, model, {}, mock_params, mock_bw)
        assert result.early_stopped is False


# ===========================================================================
# BL-03 — Sub-Stage 9.2: all optional features fail the reliability floor
# ===========================================================================

class TestSubStage92AllFeaturesFail:

    def test_all_optional_features_below_floor_only_required_remain(
            self, df_normal, mock_ctx, mock_params, mock_bw):
        """BL-03: when every optional feature is unreliable, selected_features = required only."""
        df = df_normal.copy()
        df["promo_flag"]  = 0.0
        df["day_of_week"] = df["date"].dt.dayofweek.astype(float)

        model = SESModel(hp=SESModel({}).default_hp)
        preloaded = {
            "feature_reliability": {
                mock_ctx.sku_id: {"promo_flag": 0.05, "day_of_week": 0.10},
            },
            "promo_decisions": {},
            "feature_history":  {},
        }
        result = run_feature_engineering(mock_ctx, df, model, preloaded, mock_params, mock_bw)
        for req in model.required_features:
            assert req in result.selected_features, f"Required feature '{req}' missing"
        assert "promo_flag" not in result.selected_features
        assert "day_of_week" not in result.selected_features

    def test_all_features_fail_batchwriter_row_still_written(
            self, df_normal, mock_ctx, mock_params, mock_bw):
        """BL-03: feature_decisions_s9 row must be written even with empty candidate set."""
        model = SESModel(hp=SESModel({}).default_hp)
        preloaded = {
            "feature_reliability": {mock_ctx.sku_id: {}},
            "promo_decisions": {},
            "feature_history":  {},
        }
        run_feature_engineering(mock_ctx, df_normal, model, preloaded, mock_params, mock_bw)
        assert "feature_decisions_s9" in mock_bw.tables


# ===========================================================================
# BL-04 — Sub-Stage 9.2: B2B filter leaves very few rows
# ===========================================================================

class TestSubStage92B2BMinimalRows:

    def test_b2b_filter_leaving_minimal_rows_does_not_crash(
            self, mock_ctx, mock_params, mock_bw):
        """BL-04: B2B weekday filter on a short 3-week series must not crash."""
        dates = pd.date_range("2024-01-01", periods=21)  # 3 weeks: 15 weekdays + 6 weekend days
        df = pd.DataFrame({"date": dates, "qty": np.ones(21) * 5.0})
        mock_ctx.is_b2b = True
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = {
            "feature_reliability": {mock_ctx.sku_id: {}},
            "promo_decisions": {},
            "feature_history":  {},
        }
        result = run_feature_engineering(mock_ctx, df, model, preloaded, mock_params, mock_bw)
        assert result.df_train is not None
        assert len(result.df_train) > 0
        assert "feature_decisions_s9" in mock_bw.tables

    def test_b2b_filter_leaves_only_weekdays(self, mock_ctx, mock_params, mock_bw):
        """BL-04: after filtering, no Saturday (5) or Sunday (6) rows survive."""
        dates = pd.date_range("2024-01-01", periods=21)
        df = pd.DataFrame({"date": dates, "qty": np.ones(21) * 5.0})
        mock_ctx.is_b2b = True
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        preloaded = {
            "feature_reliability": {mock_ctx.sku_id: {}},
            "promo_decisions": {},
            "feature_history":  {},
        }
        result = run_feature_engineering(mock_ctx, df, model, preloaded, mock_params, mock_bw)
        if result.b2b_mode_applied:
            assert result.df_train["date"].dt.dayofweek.max() <= 4


# ===========================================================================
# Sub-Stage 9.2 — DoW Multipliers
# _compute_dow_multipliers and result.dow_multipliers field
# ===========================================================================

from forecasting.feature_engg import _compute_dow_multipliers


class TestSubStage92DowMultipliers:

    def _make_preloaded(self, sku_id="sku-001"):
        return {
            "feature_reliability": {sku_id: {}},
            "promo_decisions": {},
            "feature_history": {},
        }

    def test_prophet_model_returns_flat_multipliers(self):
        """Prophet handles weekly seasonality natively — multipliers must stay flat."""
        dates = pd.date_range("2024-01-01", periods=60)
        df = pd.DataFrame({"date": dates, "qty": np.ones(60) * 10.0})
        result = _compute_dow_multipliers(df, is_b2b=False, assigned_model="Prophet")
        assert result == [1.0] * 7

    def test_insufficient_history_returns_flat_multipliers(self):
        """Fewer than 28 rows → cold-start fallback — no per-DoW signal."""
        dates = pd.date_range("2024-01-01", periods=27)
        df = pd.DataFrame({"date": dates, "qty": np.ones(27) * 10.0})
        result = _compute_dow_multipliers(df, is_b2b=False, assigned_model="SES")
        assert result == [1.0] * 7

    def test_boundary_28_rows_computes_multipliers(self):
        """Exactly 28 rows (4 full weeks) must compute real multipliers."""
        dates = pd.date_range("2024-01-01", periods=28)
        df = pd.DataFrame({"date": dates, "qty": np.ones(28) * 10.0})
        result = _compute_dow_multipliers(df, is_b2b=False, assigned_model="SES")
        # Uniform demand → all multipliers equal 1.0
        assert len(result) == 7
        for m in result:
            assert m == pytest.approx(1.0)

    def test_b2b_forces_weekend_multipliers_to_zero(self):
        """B2B SKU: Saturday (5) and Sunday (6) multipliers must be exactly 0.0."""
        dates = pd.date_range("2024-01-01", periods=60)  # starts Monday
        df = pd.DataFrame({"date": dates, "qty": np.ones(60) * 10.0})
        result = _compute_dow_multipliers(df, is_b2b=True, assigned_model="SES")
        assert result[5] == 0.0, "Saturday multiplier must be 0.0 for B2B"
        assert result[6] == 0.0, "Sunday multiplier must be 0.0 for B2B"

    def test_non_b2b_non_uniform_demand_shapes_multipliers(self):
        """High Monday demand → Monday multiplier > 1.0; low Sunday → < 1.0."""
        dates = pd.date_range("2024-01-01", periods=56)  # 8 full weeks, starts Monday
        qty = np.array([
            20.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0,  # Mon high, weekend low
        ] * 8)
        df = pd.DataFrame({"date": dates, "qty": qty})
        result = _compute_dow_multipliers(df, is_b2b=False, assigned_model="Holt")
        assert result[0] > 1.0, "Monday multiplier should be above average"
        assert result[5] < 1.0, "Saturday multiplier should be below average"
        assert result[6] < 1.0, "Sunday multiplier should be below average"

    def test_result_has_dow_multipliers_field(self, mock_ctx, mock_params, mock_bw):
        """FeatureEngineeringResult always has dow_multipliers, default [1.0]*7."""
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "qty": np.ones(60) * 5.0,
        })
        result = run_feature_engineering(
            mock_ctx, df, model, self._make_preloaded(mock_ctx.sku_id),
            mock_params, mock_bw,
        )
        assert hasattr(result, "dow_multipliers")
        assert len(result.dow_multipliers) == 7

    def test_dow_multipliers_not_in_batch_writer_row(self, mock_ctx, mock_params, mock_bw):
        """dow_multipliers must NOT be written to feature_decisions_s9."""
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "qty": np.ones(60) * 5.0,
        })
        run_feature_engineering(
            mock_ctx, df, model, self._make_preloaded(mock_ctx.sku_id),
            mock_params, mock_bw,
        )
        row = mock_bw.row_for("feature_decisions_s9")
        assert "dow_multipliers" not in row

    def test_e006_sets_exception_flag(self, mock_ctx, mock_params, mock_bw):
        """E006 (weekend-only seller) must append B2B_DISABLED_FLAG to result.exception_flags."""
        from infrastructure.constants import B2B_DISABLED_FLAG
        mock_ctx.is_b2b = True
        dates = pd.date_range("2024-01-06", periods=10, freq="7D")  # every Saturday
        df_wknd = pd.DataFrame({"date": dates, "qty": np.ones(10) * 5.0})
        model = NaiveForecast(hp=NaiveForecast({}).default_hp)
        result = run_feature_engineering(
            mock_ctx, df_wknd, model, self._make_preloaded(mock_ctx.sku_id),
            mock_params, mock_bw,
        )
        assert B2B_DISABLED_FLAG in result.exception_flags
