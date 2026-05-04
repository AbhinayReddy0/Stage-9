"""
Sub-Stage 9.3: Hyperparameter Tuning via Thompson Sampling

Finds the optimal HP configuration for each SKU's assigned model.
Runs after Sub-Stage 9.2 (Feature Engineering) and before Sub-Stage 9.4
(Backtesting).

All SKUs go through Standard Thompson Sampling regardless of lifecycle stage.
Note: Lifecycle stage dependencies are excluded from this build.

Critical rules:
    - Thompson state updated in MEMORY only — never written to DB here
    - sort_keys=True in all config hashes (deterministic across runs)
    - Prior best config always included in test set (even if below top-N theta)
    - ModelFitError caught per config — mape=1.0 assigned;
    - Thompson penalises hyperparameter_decisions row written via BatchWriter for EVERY SKU

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from models.base import ModelFitError
from infrastructure.batch_writer import BatchWriter
from models.thompson import ThompsonSampler
from infrastructure.constants import (
    THOMPSON_VALIDATION_HOLDOUT_DAYS,
    THOMPSON_EARLY_STOP_MAPE,
    THOMPSON_SUCCESS_IMPROVEMENT,
)

log = logging.getLogger(__name__)

__all__ = ["run_hp_tuning", "HPTuningResult"]

_VALIDATION_HOLDOUT_DAYS = THOMPSON_VALIDATION_HOLDOUT_DAYS
_EARLY_STOP_MAPE = THOMPSON_EARLY_STOP_MAPE
_THOMPSON_SUCCESS_RATIO = 1.0 - THOMPSON_SUCCESS_IMPROVEMENT


@dataclass
class HPTuningResult:
    """All outputs from Sub-Stage 9.3 passed into Sub-Stage 9.4."""

    best_hp: dict = field(default_factory=dict)
    validation_mape: float = 1.0
    training_data_source: str = "real"  # always 'real' — lifecycle paths excluded
    early_stopped: bool = False
    thompson_score: float = 0.5


def run_hp_tuning(
        ctx: Any,  # LearningContext — sku_id, tenant_id, run_id, etc.
        df_train: Any,  # pd.DataFrame — promo-adjusted, from Sub-Stage 9.2
        model: Any,  # BaseModel instance assigned in Sub-Stage 9.1
        preloaded: dict,  # bulk-preloaded dicts (thompson_state, etc.)
        params: Any,  # TenantParams
        batch_writer: BatchWriter,
) -> HPTuningResult:
    """
    Run HP tuning for one SKU and return the best HP configuration.

    All SKUs go through Thompson Sampling regardless of lifecycle stage.

    Args:
        ctx:          LearningContext. Key fields: sku_id, tenant_id, run_id,
                      assigned_model, pattern_label,
                      selected_features, sample_weights, baseline_mape.
        df_train:     Promo-adjusted training DataFrame from Sub-Stage 9.2.
        model:        Assigned model instance (must implement BaseModel contract).
        preloaded:    Contains 'thompson_state' keyed by (sku_id, assigned_model).
        params:       TenantParams for reading thompson_exploration_budget.
        batch_writer: BatchWriter instance — queues hyperparameter_decisions rows.

    Returns:
        HPTuningResult with best_hp, validation_mape, and metadata.
        ctx.best_hp and ctx.validation_mape are also set on the context object.
    """
    # All SKUs run Standard Thompson Sampling path
    result = _run_path2_thompson(ctx, df_train, model, preloaded, params)

    # Write selected best_hp back to ctx for Sub-Stage 9.4
    ctx.best_hp = result.best_hp
    ctx.validation_mape = result.validation_mape

    # -------------------------------------------------------------------------
    # Write hyperparameter_decisions row via BatchWriter
    # ALWAYS written — even when default_hp used (short series).
    # Done Criterion 4: row must exist in BatchWriter queue after Sub-Stage 9.3.
    # -------------------------------------------------------------------------
    sampler = ThompsonSampler()
    try:
        thompson_state = (
            preloaded.get("thompson_state", {})
            .get((ctx.sku_id, ctx.assigned_model), {})
        )
        config_hash = sampler.config_hash(result.best_hp)
        thompson_score = sampler.get_thompson_score(result.best_hp, thompson_state)

        batch_writer.queue(
            table="hyperparameter_decisions",
            row={
                "tenant_id": ctx.tenant_id,
                "sku_id": ctx.sku_id,
                "run_id": ctx.run_id,
                "hyperparameters": result.best_hp,
                "validation_mape": result.validation_mape,
                "config_hash": config_hash,
                "thompson_score": thompson_score,
                "early_stopped": result.early_stopped,
            },
        )
    except Exception as exc:
        log.error(
            "sub_stage_93 sku=%s: BatchWriter.queue (hyperparameter_decisions) "
            "failed: %s",
            ctx.sku_id, exc,
        )

    return result


# ===========================================================================
# Standard Thompson Sampling — runs for all SKUs
# ===========================================================================

def _run_path2_thompson(
        ctx: Any,
        df_train: Any,
        model: Any,
        preloaded: dict,
        params: Any,
) -> HPTuningResult:
    """
    Standard HP tuning via Thompson Sampling. Runs for every SKU.

    PERCEIVE: load Thompson state, compute config hashes, find prior best.
    PLAN:     call ThompsonSampler.select_configs() to pick configs to test.
    ACT:      train + validate each config; catch ModelFitError per config.
    LEARN:    update Thompson state in memory only (NOT written to DB here).
    """
    result = HPTuningResult(training_data_source="real")
    sampler = ThompsonSampler()

    # ------------------------------------------------------------------
    # Guard: insufficient data for a proper HP search
    # ------------------------------------------------------------------
    if len(df_train) < _VALIDATION_HOLDOUT_DAYS:
        log.info(
            "sub_stage_93 sku=%s: only %d rows — skipping HP search, using default_hp",
            ctx.sku_id, len(df_train),
        )
        result.best_hp = dict(model.default_hp)
        result.validation_mape = 1.0
        return result

    # ------------------------------------------------------------------
    # PERCEIVE — load Thompson state for this (sku_id, model) pair
    # ------------------------------------------------------------------
    thompson_state: dict = (
        preloaded.get("thompson_state", {})
        .get((ctx.sku_id, ctx.assigned_model), {})
    )

    prior_best_config = sampler.get_prior_best(
        hp_search_space=model.hp_search_space,
        thompson_state=thompson_state,
        default_hp=model.default_hp,
    )

    # ------------------------------------------------------------------
    # PLAN — select configs to test this run
    # ------------------------------------------------------------------
    budget = int(params.get("thompson_exploration_budget"))  # starts 3; NEVER hardcode

    selected_configs = sampler.select_configs(
        hp_search_space=model.hp_search_space,
        thompson_state=thompson_state,
        budget=budget,
        prior_best_config=prior_best_config,
    )

    # Train / validation split for HP evaluation
    train_split = df_train.iloc[:-_VALIDATION_HOLDOUT_DAYS]
    val_split = df_train.iloc[-_VALIDATION_HOLDOUT_DAYS:]
    selected_features = getattr(ctx, "selected_features", model.required_features)
    sample_weights_ctx = getattr(ctx, "sample_weights", None)

    # ------------------------------------------------------------------
    # ACT — evaluate each selected config
    # ------------------------------------------------------------------
    config_results: list[tuple[dict, float]] = []
    early_stopped = False

    for hp_config in selected_configs:
        try:
            model_instance = type(model)(hp=hp_config)

            model_instance.fit(
                train_split,
                selected_features,
                sample_weights=sample_weights_ctx,
            )
            predictions = model_instance.predict(
                val_split,
                selected_features,
                horizon=_VALIDATION_HOLDOUT_DAYS,
            )
            # Sanitise: NaN/Inf in val_split qty must not propagate into mape.
            actual = np.nan_to_num(
                val_split["qty"].values.astype(float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            mask = actual != 0
            if not mask.any():
                mape = 1.0
            else:
                mape = float(
                    np.mean(np.abs(actual[mask] - predictions[mask]) / actual[mask])
                )
                if not np.isfinite(mape):
                    mape = 1.0
                mape = min(mape, 99.0)

        except ModelFitError as exc:
            # ModelFitError is the ONLY exception that should escape the model layer.
            # Assign worst-case MAPE; Thompson will penalise this config.
            log.info(
                "sub stage 9.3 sku=%s: ModelFitError for config %s: %s "
                "— assigning mape=1.0",
                ctx.sku_id, hp_config, exc,
            )
            mape = 1.0

        config_results.append((hp_config, mape))

        # Early stop: MAPE is already excellent — no need to test more configs
        if mape < _EARLY_STOP_MAPE:
            early_stopped = True
            log.info(
                "sub stage 9.3 sku=%s: early stop (mape=%.4f < %.2f) after %d configs",
                ctx.sku_id, mape, _EARLY_STOP_MAPE, len(config_results),
            )
            break

    # ------------------------------------------------------------------
    # LEARN — update Thompson state in memory only (NOT flushed to DB here)
    # ------------------------------------------------------------------
    baseline_mape = float(getattr(ctx, "baseline_mape", 1.0))

    for hp_config, mape in config_results:
        success = mape <= baseline_mape * _THOMPSON_SUCCESS_RATIO
        thompson_state = sampler.update_state(thompson_state, hp_config, success)

    # Flush updated state back into preloaded dict (in-memory only)
    if "thompson_state" not in preloaded:
        preloaded["thompson_state"] = {}
    preloaded["thompson_state"][(ctx.sku_id, ctx.assigned_model)] = thompson_state

    # Select the best config (lowest validation MAPE across all tested)
    if config_results:
        best_hp, best_mape = min(config_results, key=lambda x: x[1])
    else:
        best_hp = dict(model.default_hp)
        best_mape = 1.0

    result.best_hp = best_hp
    result.validation_mape = best_mape
    result.early_stopped = early_stopped

    return result
