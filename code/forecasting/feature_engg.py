"""
Sub-Stage 9.2: Feature Engineering

Prepares promo-weighted training data and selects the optimal feature set
for the assigned model. Runs after Sub-Stage 9.1 (Model Initialisation)
and before Sub-Stage 9.3 (HP Tuning).

Four steps run in order. Each step may fail independently — log the failure,
skip that step, continue. A per-step failure NEVER crashes the SKU.

Steps:
    1. Reliability Filtering  — drop optional features below reliability floor
    2. B2B Mode Filter        — weekday-only training data when is_b2b=True
    3. Promo-Weighted Data    — cap/weight promo-day demand (E006 guard)
    4. Additive Feature Search — greedy MAPE-driven feature selection

Output:
    ctx.selected_features  — feature list for Sub-Stage 9.3
    ctx.df_train           — promo-adjusted DataFrame
    ctx.sample_weights     — for NeuralProphet/Prophet only (or None)
    ctx.baseline_mape      — required_features-only MAPE
    BatchWriter queue      — feature_decisions_s9 row per SKU

"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from infrastructure.batch_writer import BatchWriter
from infrastructure.constants import (
    FEATURE_SEARCH_BUDGET,
    FEATURE_SEARCH_EARLY_STOP_MAPE,
    FEATURE_SEARCH_MIN_IMPROVEMENT,
    THOMPSON_VALIDATION_HOLDOUT_DAYS,
    B2B_DISABLED_FLAG,
    PROMO_ROLLING_BASELINE_DAYS,
    Model,
    PROPHET_FAMILY,
)

log = logging.getLogger(__name__)

_ADDITIVE_SEARCH_BUDGET    = FEATURE_SEARCH_BUDGET
_VALIDATION_HOLDOUT_DAYS   = THOMPSON_VALIDATION_HOLDOUT_DAYS

__all__ = ["run_feature_engineering", "FeatureEngineeringResult"]


class FeatureEngineeringResult:
    """
    Carries all outputs of Sub-Stage 9.2 into Sub-Stage 9.3.

    Attributes:
        selected_features:        Feature list that passed reliability + MAPE filters.
        df_train:                 Promo-adjusted, B2B-filtered training DataFrame.
        sample_weights:           numpy array for Prophet, else None.
        baseline_mape:            MAPE using required_features only (Step 4 baseline).
        b2b_mode_applied:         True when weekday filter was applied and not disabled.
        promo_weighting_applied:  True when promo dict had entries for this SKU.
        reliability_map_applied:  Copy of feature_reliability_map used in Step 1.
        improved_mape:            MAPE with final selected_features (post additive search).
        dow_multipliers:          7 floats [Mon..Sun]. Flat [1.0]*7 for Prophet or
                                  insufficient history. Weekend forced to 0.0 for B2B.
                                  NOT written to feature_decisions_s9 — pipeline use only.
        exception_flags:          Edge-case flags raised during feature engineering
                                  (e.g. "b2b_mode_disabled" for E006). Merged into the
                                  SkuForecastInput exception_flags by the worker.
    """

    def __init__(self) -> None:
        self.selected_features:        list[str]                  = []
        self.df_train:                 Optional[pd.DataFrame]     = None
        self.sample_weights:           Optional[np.ndarray]       = None
        self.baseline_mape:            float                      = 1.0
        self.b2b_mode_applied:         bool                       = False
        self.promo_weighting_applied:  bool                       = False
        self.reliability_map_applied:  dict                       = {}
        self.improved_mape:            float                      = 1.0
        self.dow_multipliers:          list[float]                = [1.0] * 7
        self.exception_flags:          list[str]                  = []


_DOW_MIN_OBS = 28   # 4 full weeks — below this, too few per-DoW observations


def _compute_dow_multipliers(
    df_train: pd.DataFrame,
    is_b2b: bool,
    assigned_model: str,
) -> list[float]:
    """
    Compute demand multipliers for each day of week from training data.

    Returns 7 floats indexed [0=Mon .. 6=Sun] where the multipliers
    represent each day's mean demand relative to the overall mean.
    A value of 1.0 means that day is average; 0.0 means no demand.

    Falls back to flat [1.0]*7 when:
      - assigned model is Prophet (handles weekly seasonality natively)
      - fewer than _DOW_MIN_OBS rows in df_train
      - overall mean qty is zero or negative

    B2B SKUs: weekends (5=Sat, 6=Sun) are forced to 0.0. The weekday
    multipliers are computed from df_train which is already weekday-only
    after Step 2's B2B filter, so the overall mean correctly reflects
    weekday-only demand.

    Per-DoW fallback: if a specific day has fewer than 4 observations
    (e.g. a new product with limited weekday coverage), that day uses 1.0.
    """
    if assigned_model in PROPHET_FAMILY:
        return [1.0] * 7

    if len(df_train) < _DOW_MIN_OBS:
        return [1.0] * 7

    overall_mean = float(df_train["qty"].mean())
    if overall_mean <= 0.0:
        return [1.0] * 7

    multipliers: list[float] = []
    for dow in range(7):
        if is_b2b and dow >= 5:
            multipliers.append(0.0)
            continue
        dow_rows = df_train[df_train["date"].dt.dayofweek == dow]["qty"]
        if len(dow_rows) < 4:
            multipliers.append(1.0)
        else:
            multipliers.append(float(dow_rows.mean()) / overall_mean)

    return multipliers


def run_feature_engineering(
    ctx:       Any,                  # LearningContext from Sub-Stage 9.1
    df:        pd.DataFrame,         # full training DataFrame (date, qty, optional cols)
    model:     Any,                  # BaseModel instance (for required/optional features)
    preloaded: dict,                 # bulk-preloaded dicts from PRELOADING state
    params:    Any,                  # TenantParams instance
    batch_writer: BatchWriter,
) -> FeatureEngineeringResult:
    """
    Run all four feature engineering steps for one SKU.

    Per-Step failure isolation: each step is wrapped in try/except. A step
    failure logs the error and continues with the data/features as-is.
    The SKU is never abandoned due to a feature engineering failure.

    Args:
        ctx:          LearningContext with sku_id, tenant_id, run_id, pattern_label,
                      is_b2b, assigned_model.
        df:           Training DataFrame. Must contain 'date' and 'qty' at minimum.
        model:        Assigned model instance (NaiveForecast, SESModel, etc.).
        preloaded:    {'feature_reliability', 'promo_decisions', 'feature_history'}
                      dicts keyed by sku_id.
        params:       TenantParams for reading feature_reliability_floor,
                      max_promo_multiplier, mape_improvement_threshold.
        batch_writer: BatchWriter instance — queues feature_decisions_s9 rows.

    Returns:
        FeatureEngineeringResult with all outputs populated.
    """
    result    = FeatureEngineeringResult()
    sku_id    = ctx.sku_id
    df_work   = df.copy()          # never mutate the caller's DataFrame

    # -------------------------------------------------------------------------
    # Step 1: Reliability Filtering
    # -------------------------------------------------------------------------
    try:
        feature_reliability_map: dict = (
            preloaded.get("feature_reliability", {}).get(sku_id) or {}
        )
        result.reliability_map_applied = dict(feature_reliability_map)

        floor = float(params.get("feature_reliability_floor"))   # starts 0.30

        # Start with required features (never filtered) plus all optional features
        candidate_features: list[str] = list(model.required_features)

        for feature in model.optional_features:
            reliability = float(feature_reliability_map.get(feature, 0.0))
            if reliability < floor:
                log.info(
                    "sub stage 9.2 sku=%s: dropped_feature:%s:reliability:%.2f "
                    "(floor=%.2f)",
                    sku_id, feature, reliability, floor,
                )
            else:
                candidate_features.append(feature)

    except Exception as exc:
        # Step failure — log and proceed with required features only
        log.error(
            "sub stage 9.2 sku=%s: Step 1 (reliability filter) failed: %s "
            "— proceeding with required features only",
            sku_id, exc,
        )
        candidate_features = list(model.required_features)
        result.reliability_map_applied = {}

    # -------------------------------------------------------------------------
    # Step 2: B2B Mode Filter + DoW Multipliers
    # DoW multipliers are computed here because is_b2b and the final
    # df_work are both known at this point. NOT written to batch writer.
    # -------------------------------------------------------------------------
    try:
        is_b2b = getattr(ctx, "is_b2b", False)
        if is_b2b:
            df_weekday = df_work[df_work["date"].dt.dayofweek < 5].copy()

            # E006: weekend-only seller guard — B2B flag was incorrectly set
            if len(df_weekday) == 0:
                df_weekday             = df_work          # revert to all days
                ctx.b2b_mode_disabled  = True
                result.b2b_mode_applied = False
                result.exception_flags.append(B2B_DISABLED_FLAG)
                is_b2b                 = False            # treat as non-B2B for DoW
                log.info(
                    "sub_stage_92 sku=%s E006: B2B filter produced 0 rows "
                    "(weekend-only seller) — filter disabled, flag '%s' set",
                    sku_id, B2B_DISABLED_FLAG,
                )
            else:
                result.b2b_mode_applied = True

            df_work = df_weekday

        result.dow_multipliers = _compute_dow_multipliers(
            df_work,
            is_b2b,
            getattr(ctx, "assigned_model", ""),
        )

    except Exception as exc:
        log.error(
            "sub_stage_9.2 sku=%s: Step 2 (B2B filter) failed: %s "
            "— proceeding without B2B filter",
            sku_id, exc,
        )

    # -------------------------------------------------------------------------
    # Step 3: Promo-Weighted Training Data
    # -------------------------------------------------------------------------
    try:
        promo_lookup: dict = preloaded.get("promo_decisions", {})
        has_promo = any(
            k[0] == sku_id for k in promo_lookup.keys()
        ) if isinstance(promo_lookup, dict) else False

        if has_promo:
            multiplier = float(params.get("max_promo_multiplier"))    # starts 3.0
            uses_sample_weights = _model_uses_sample_weights(ctx.assigned_model)

            if uses_sample_weights:
                # Prophet: build a weight array instead of capping qty.
                # Vectorised: map (sku_id, date_iso) → weight using Series.map,
                # 63x faster than iterrows on 3-year daily series (1ms vs 84ms).
                date_keys = df_work["date"].dt.strftime("%Y-%m-%d").apply(
                    lambda d: promo_lookup.get((sku_id, d), 1.0)
                )
                weights = date_keys.values.astype(float)
                result.sample_weights = weights

            else:
                # All other models: cap qty on promo days at baseline × multiplier.
                # Fully vectorised — no per-row Python loop.
                df_work = df_work.copy()
                rolling_baseline = df_work["qty"].rolling(PROMO_ROLLING_BASELINE_DAYS, min_periods=1).mean()
                # Build promo weight series via vectorised lookup
                promo_weights = df_work["date"].dt.strftime("%Y-%m-%d").apply(
                    lambda d: promo_lookup.get((sku_id, d), 1.0)
                )
                promo_mask = promo_weights < 1.0
                if promo_mask.any():
                    caps = rolling_baseline * multiplier
                    df_work.loc[promo_mask, "qty"] = np.minimum(
                        df_work.loc[promo_mask, "qty"].values,
                        caps[promo_mask].values,
                    )

            result.promo_weighting_applied = True

    except Exception as exc:
        log.error(
            "sub stage 9.2 sku=%s: Step 3 (promo weighting) failed: %s "
            "— proceeding without promo adjustment",
            sku_id, exc,
        )

    result.df_train = df_work

    # -------------------------------------------------------------------------
    # Step 4: Additive Feature Search
    # -------------------------------------------------------------------------
    try:
        improvement_threshold = FEATURE_SEARCH_MIN_IMPROVEMENT

        # Determine starting feature set: use prior best if available
        prior_features: Optional[list[str]] = (
            preloaded.get("feature_history", {}).get(sku_id)
        )
        if prior_features:
            # Validate prior features still exist in candidate set
            starting_features = [f for f in prior_features if f in candidate_features]
            if not starting_features:
                starting_features = list(model.required_features)
        else:
            starting_features = list(model.required_features)

        # Holdout split for MAPE computation
        if len(df_work) <= _VALIDATION_HOLDOUT_DAYS:
            # Not enough data to do a holdout — accept candidate_features as-is
            result.selected_features = candidate_features
            result.baseline_mape     = 1.0
            result.improved_mape     = 1.0
            log.info(
                "sub stage 9.2 sku=%s: too few rows (%d) for feature search "
                "— using candidate_features as-is",
                sku_id, len(df_work),
            )
        else:
            train_split = df_work.iloc[:-_VALIDATION_HOLDOUT_DAYS]
            val_split   = df_work.iloc[-_VALIDATION_HOLDOUT_DAYS:]

            # Baseline MAPE: required_features only
            baseline_mape = _compute_mape(
                model, train_split, val_split, model.required_features
            )
            result.baseline_mape = baseline_mape

            # Track the best feature set and MAPE so far
            best_features = list(starting_features)
            best_mape     = _compute_mape(model, train_split, val_split, best_features)
            configs_tested = 0

            # Optional features not yet in the selected set
            remaining_candidates = [
                f for f in candidate_features
                if f not in best_features
                and f not in model.required_features
            ]

            for feature in remaining_candidates:
                if configs_tested >= _ADDITIVE_SEARCH_BUDGET:
                    break

                test_features = best_features + [feature]
                try:
                    test_mape = _compute_mape(
                        model, train_split, val_split, test_features
                    )
                except Exception as inner_exc:
                    log.debug(
                        "sub stage 9.2 sku=%s: feature '%s' MAPE eval failed: %s",
                        sku_id, feature, inner_exc,
                    )
                    configs_tested += 1
                    continue

                configs_tested += 1

                # Accept feature if MAPE improves by at least improvement_threshold
                if test_mape <= best_mape * (1.0 - improvement_threshold):
                    prev_mape = best_mape
                    best_features = test_features
                    best_mape     = test_mape
                    log.info(
                        "sub stage 9.2 sku=%s: feature '%s' accepted "
                        "(mape %.4f → %.4f, improvement %.2f%%)",
                        sku_id, feature,
                        prev_mape, test_mape,
                        (prev_mape - test_mape) / max(prev_mape, 1e-8) * 100,
                    )

                # Early stop: MAPE is already excellent
                if best_mape < FEATURE_SEARCH_EARLY_STOP_MAPE:
                    log.info(
                        "sub stage 9.2 sku=%s: early stop (mape=%.4f < %.2f)",
                        sku_id, best_mape, FEATURE_SEARCH_EARLY_STOP_MAPE,
                    )
                    break

            result.selected_features = best_features
            result.improved_mape     = best_mape

    except Exception as exc:
        log.error(
            "sub stage 9.2 sku=%s: Step 4 (feature search) failed: %s "
            "— using required_features only",
            sku_id, exc,
        )
        result.selected_features = list(model.required_features)
        result.improved_mape     = result.baseline_mape

    # Always ensure required_features are present (Critical Rule)
    for req in model.required_features:
        if req not in result.selected_features:
            result.selected_features.insert(0, req)

    # -------------------------------------------------------------------------
    # Write feature_decisions_s9 row to BatchWriter
    # Always written — even on partial failure. Done Criterion 6.
    # -------------------------------------------------------------------------
    try:
        batch_writer.queue(
            table="feature_decisions_s9",
            row={
                "tenant_id":                ctx.tenant_id,
                "sku_id":                   sku_id,
                "run_id":                   ctx.run_id,
                "features_used":            result.selected_features,
                "reliability_map_applied":  bool(result.reliability_map_applied),
                "b2b_mode_applied":         result.b2b_mode_applied,
                "promo_weighting_applied":  result.promo_weighting_applied,
                "baseline_mape":            result.baseline_mape,
                "improved_mape":            result.improved_mape,
            },
        )
    except Exception as exc:
        log.error(
            "sub_stage_92 sku=%s: BatchWriter.queue (feature_decisions_s9) failed: %s",
            sku_id, exc,
        )

    return result


# ===========================================================================
# Internal helpers — never called by sub-stages directly
# ===========================================================================

def _model_uses_sample_weights(assigned_model: str) -> bool:
    """
    Return True if the assigned model can use sample_weights (promo weighting).

    Only Prophet support sample weights. All other models
    accept the parameter but ignore it — they get qty-capping instead.
    """
    return assigned_model == Model.PROPHET


def _compute_mape(
    model:         Any,
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    features:      list[str],
) -> float:
    """
    Fit model on train_df, predict for validation window, compute MAPE.

    Returns 1.0 (100% error) if the model raises ModelFitError — treated as
    a failure so this feature combination is penalised in the search.

    Uses numpy-safe division: np.maximum(actual, 1e-8) prevents divide-by-zero
    on days with zero actual demand.
    """
    try:
        model.fit(train_df, features)
        horizon     = len(val_df)
        predictions = model.predict(val_df, features, horizon=horizon)
        actual      = val_df["qty"].values.astype(float)

        mask = actual != 0
        if not mask.any():
            return 1.0
        mape = float(np.mean(
            np.abs(actual[mask] - predictions[mask]) / actual[mask]
        ))
        return min(mape, 99.0)

    except Exception:
        return 1.0   # treat any failure as worst-case MAPE
