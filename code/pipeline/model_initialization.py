"""
Sub-Stage 9.1: Model Initialisation

Makes 7 decisions for each FULL-tier SKU and outputs a LearningContext
dataclass consumed by Sub-Stages 9.2–9.6.

Decisions (in order):
    1. Model Assignment      — PATTERN_MODEL_MAP; seasonal always → Prophet
    2. Quantile Selection    — criticality_A → 0.99, all others → pattern_default
    3. Effective Max Horizon — min(365, planned_end_date_days, shelf_life_days), floor 7
    4. Learning Mode         — exploit only when all 3 Thompson conditions met
    5. OOS Adjustment Factor — 1 + oos_pct × detection_confidence, capped at 1.50
    6. B2B Mode              — weekend_zero_ratio > 0.60 (strictly greater than)
    7. Reorder Bias Factor   — PEEK last 5 reorder_outcome signals; stockout wins ties

Critical rules:
    - Never raises — all exceptions propagate to caller's process_sku_safely()
    - BatchWriter write is audit-only — wrapped in try/except, never blocks pipeline
    - Decisionn 7 uses PEEK only — processed=TRUE is never set here

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from infrastructure.batch_writer import BatchWriter
from infrastructure.constants import (
    B2B_WEEKEND_ZERO_RATIO_THRESHOLD,
    CRITICALITY_A_QUANTILE,
    CriticalityTier,
    LearningMode,
    Model,
    OOS_ADJUSTMENT_MAX_FACTOR,
    PATTERN_MODEL_MAP,
    PATTERN_QUANTILE_PARAM,
    Param,
    Pattern,
    REORDER_BIAS_FACTOR_DEFAULT,
    REORDER_BIAS_FACTOR_OVERSTOCK,
    REORDER_BIAS_FACTOR_STOCKOUT,
    REORDER_OVERSTOCK_THRESHOLD,
    REORDER_SIGNAL_LOOKBACK,
    REORDER_STOCKOUT_MIN_EVENTS,
    SignalType,
    TenantMaturity,
)
from signals import SignalConsumer
from pipeline.preloader import PreloadedData

log = logging.getLogger(__name__)

__all__ = ["run_model_initialisation", "LearningContext"]

# Maps tenant maturity → exploit threshold param name.
# Defined here (not in constants.py) — used only by the exploit decision.
_MATURITY_EXPLOIT_PARAM: dict[str, str] = {
    TenantMaturity.NEW:         Param.EXPLOIT_THRESHOLD_NEW,
    TenantMaturity.DEVELOPING:  Param.EXPLOIT_THRESHOLD_DEVELOPING,
    TenantMaturity.ESTABLISHED: Param.EXPLOIT_THRESHOLD_ESTABLISHED,
}


@dataclass
class LearningContext:
    """
    Output of Sub-Stage 9.1. Passed into Sub-Stages 9.2–9.6.

    Fields set by 9.1 (all 7 initialisation decisions):
        sku_id, tenant_id, run_id, pattern_label, lifecycle_stage,
        assigned_model, selected_quantile, quantile_source,
        effective_max_horizon, learning_mode, oos_adjustment_factor,
        is_b2b, reorder_bias_factor.

    Fields populated by downstream sub-stages (mutable):
        selected_features   — set by 9.2
        sample_weights      — set by 9.2
        baseline_mape       — set by 9.2
        best_hp             — set by 9.3
        validation_mape     — set by 9.3
        b2b_mode_disabled   — set by 9.2 on E006 (weekend-only seller)
    """

    # Set by Sub-Stage 9.1
    sku_id:                 Any
    tenant_id:              str
    run_id:                 Any
    pattern_label:          str
    lifecycle_stage:        Optional[str]
    assigned_model:         str
    selected_quantile:      float
    quantile_source:        str
    effective_max_horizon:  int
    learning_mode:          str
    oos_adjustment_factor:  float
    is_b2b:                 bool
    reorder_bias_factor:    float

    # Set by Sub-Stage 9.1 — seasonal guard (default False for all non-seasonal SKUs)
    insufficient_seasonal_history: bool = False

    # Populated by downstream sub-stages
    selected_features:  list     = field(default_factory=list)
    sample_weights:     Optional[Any] = None
    baseline_mape:      float    = 1.0
    best_hp:            dict     = field(default_factory=dict)
    validation_mape:    float    = 1.0
    b2b_mode_disabled:  bool     = False


def run_model_initialisation(
    sku_id:       Any,
    preloaded:    PreloadedData,
    params:       Any,            # TenantParams
    batch_writer: BatchWriter,
    consumer:     SignalConsumer,
    run_id:       Any,
) -> LearningContext:
    """
    Run all 7 initialisation decisions for one FULL-tier SKU.

    Args:
        sku_id:       SKU UUID (str or UUID object). Converted to str internally.
        preloaded:    PreloadedData from the PRELOADING handler.
        params:       TenantParams — read-only param snapshot for this tenant.
        batch_writer: BatchWriter — queues the model_initialization_s9 audit row.
        consumer:     SignalConsumer — used for Decision PEEK only.
        run_id:       Run UUID (str or UUID object).

    Returns:
        LearningContext with all 7 decisions populated and downstream fields
        at their default placeholder values (filled in by 9.2–9.3).
    """
    sku_key = str(sku_id)
    pattern_ctx = preloaded.pattern_ctx.get(sku_key, {})

    # ------------------------------------------------------------------
    # Decision 1 — Model Assignment
    # ------------------------------------------------------------------
    pattern_label:   str          = pattern_ctx.get("pattern_label", Pattern.STABLE)
    lifecycle_stage: Optional[str] = pattern_ctx.get("lifecycle_stage")

    if pattern_label == Pattern.SEASONAL:
        assigned_model = Model.PROPHET
    else:
        assigned_model = PATTERN_MODEL_MAP[pattern_label]

    # Seasonal guard: monthly-seasonal SKUs need ≥ min_seasonal_obs_days of history
    # before Prophet can calibrate. Override to Holt until threshold is crossed.
    insufficient_seasonal_history = False
    if pattern_label == Pattern.SEASONAL:
        obs_days = int(pattern_ctx.get("obs_days", 0))
        min_seasonal_obs = int(params.get(Param.MIN_SEASONAL_OBS_DAYS))
        if obs_days < min_seasonal_obs:
            assigned_model = Model.HOLTS_LINEAR
            insufficient_seasonal_history = True
            log.debug(
                "sub_stage_91 sku=%s seasonal guard: obs_days=%d < %d → Holt override",
                sku_key, obs_days, min_seasonal_obs,
            )

    # ------------------------------------------------------------------
    # Decision 2 — Quantile Selection
    # Only criticality_A overrides the pattern default today.
    # sku_slt and lifecycle (clearance) overrides are future additions.
    # ------------------------------------------------------------------
    criticality_tier: Optional[str] = pattern_ctx.get("criticality_tier")

    if criticality_tier == CriticalityTier.A:
        selected_quantile = CRITICALITY_A_QUANTILE
        quantile_source   = "criticality_a"
    else:
        quantile_param    = PATTERN_QUANTILE_PARAM[pattern_label]
        selected_quantile = float(params.get(quantile_param))
        quantile_source   = "pattern_default"

    # ------------------------------------------------------------------
    # Decision 3 — Effective Max Horizon
    # Cap at planned_end_date and shelf_life; floor the result at 7 days.
    # ------------------------------------------------------------------
    planned_end_date = pattern_ctx.get("planned_end_date")   # datetime.date or None
    shelf_life_days  = pattern_ctx.get("shelf_life_days")    # int or None

    max_horizon = 365

    if planned_end_date is not None:
        if isinstance(planned_end_date, date):
            days_remaining = (planned_end_date - date.today()).days
        else:
            days_remaining = int(planned_end_date)
        max_horizon = min(max_horizon, max(0, days_remaining))

    if shelf_life_days is not None:
        max_horizon = min(max_horizon, int(shelf_life_days))

    effective_max_horizon = max(7, max_horizon)

    # ------------------------------------------------------------------
    # Decision 4 — Learning Mode
    # exploit only when ALL THREE conditions hold:
    #   (a) thompson_confidence > 0.60
    #   (b) historical_runs >= exploit_threshold for this tenant's maturity
    #   (c) not on_watchlist AND not drift_detected
    # ------------------------------------------------------------------
    thompson_ctx  = preloaded.thompson_ctx.get(sku_key, {})
    alpha         = float(thompson_ctx.get("alpha", 1.0))
    beta          = float(thompson_ctx.get("beta", 1.0))
    historical_runs = int(thompson_ctx.get("historical_runs", 0))
    on_watchlist  = bool(pattern_ctx.get("on_watchlist", False))
    drift_detected = bool(pattern_ctx.get("drift_detected", False))

    tenant_maturity = preloaded.signal_context.get("tenant_maturity", TenantMaturity.NEW)
    exploit_param   = _MATURITY_EXPLOIT_PARAM.get(tenant_maturity, Param.EXPLOIT_THRESHOLD_NEW)
    exploit_threshold = int(params.get(exploit_param))

    thompson_confidence_threshold = float(
        params.get(Param.THOMPSON_EXPLOIT_CONFIDENCE_THRESHOLD)
    )
    thompson_confidence = alpha / (alpha + beta)
    can_exploit = (
        thompson_confidence > thompson_confidence_threshold
        and historical_runs >= exploit_threshold
        and not on_watchlist
        and not drift_detected
    )
    learning_mode = LearningMode.EXPLOIT if can_exploit else LearningMode.EXPLORE

    # ------------------------------------------------------------------
    # Decision 5 — OOS Adjustment Factor
    # Returns 1.0 when no OOS record exists for this SKU.
    # Intermittent SKUs are always 1.0: their zero-streaks are demand signal,
    # not stockout events — applying OOS uplift inflates every forecast.
    # ------------------------------------------------------------------
    oos_record = preloaded.oos_ctx.get(sku_key)
    if pattern_label == Pattern.INTERMITTENT or oos_record is None:
        oos_adjustment_factor = 1.0
    else:
        oos_pct              = float(oos_record.get("oos_pct", 0.0))
        detection_confidence = float(oos_record.get("detection_confidence", 0.0))
        raw_factor            = 1.0 + oos_pct * detection_confidence
        # Floor at 1.0: negative oos_pct from bad data must not reduce demand.
        oos_adjustment_factor = min(max(raw_factor, 1.0), OOS_ADJUSTMENT_MAX_FACTOR)

    # ------------------------------------------------------------------
    # Decision 6 — B2B Mode
    # Strictly greater than: ratio of exactly 0.60 is NOT B2B.
    # ------------------------------------------------------------------
    weekend_zero_ratio = float(pattern_ctx.get("weekend_zero_ratio", 0.0))
    is_b2b = weekend_zero_ratio > B2B_WEEKEND_ZERO_RATIO_THRESHOLD

    # ------------------------------------------------------------------
    # Decision 7 — Reorder Bias Factor (PEEK — never sets processed=TRUE)
    # Stockout condition takes priority over overstock when both are true.
    # ------------------------------------------------------------------
    reorder_signals = consumer.peek(
        tenant_id=str(preloaded.tenant_id),
        signal_type=SignalType.REORDER_OUTCOME,
        sku_id=sku_key,
        limit=REORDER_SIGNAL_LOOKBACK,
    )

    if not reorder_signals:
        reorder_bias_factor = REORDER_BIAS_FACTOR_DEFAULT
    else:
        # peek() already limits to REORDER_SIGNAL_LOOKBACK — no slice needed.
        stockout_count  = sum(1 for s in reorder_signals if s.get("stockout", False))
        overstock_vals  = [float(s.get("overstock_pct", 0.0)) for s in reorder_signals]
        avg_overstock   = sum(overstock_vals) / len(overstock_vals)

        has_stockout  = stockout_count >= REORDER_STOCKOUT_MIN_EVENTS
        has_overstock = avg_overstock   > REORDER_OVERSTOCK_THRESHOLD

        if has_stockout:                          # stockout always wins ties
            reorder_bias_factor = REORDER_BIAS_FACTOR_STOCKOUT
        elif has_overstock:
            reorder_bias_factor = REORDER_BIAS_FACTOR_OVERSTOCK
        else:
            reorder_bias_factor = REORDER_BIAS_FACTOR_DEFAULT

    # ------------------------------------------------------------------
    # Assemble LearningContext
    # ------------------------------------------------------------------
    ctx = LearningContext(
        sku_id=sku_id,
        tenant_id=str(preloaded.tenant_id),
        run_id=run_id,
        pattern_label=pattern_label,
        lifecycle_stage=lifecycle_stage,
        assigned_model=assigned_model,
        insufficient_seasonal_history=insufficient_seasonal_history,
        selected_quantile=selected_quantile,
        quantile_source=quantile_source,
        effective_max_horizon=effective_max_horizon,
        learning_mode=learning_mode,
        oos_adjustment_factor=oos_adjustment_factor,
        is_b2b=is_b2b,
        reorder_bias_factor=reorder_bias_factor,
    )

    # ------------------------------------------------------------------
    # Write audit row via BatchWriter (best-effort — never raises)
    # model_initialization_s9 row queued for every SKU.
    # ------------------------------------------------------------------
    try:
        batch_writer.queue(
            table="model_initialization_s9",
            row={
                "tenant_id":             ctx.tenant_id,
                "sku_id":                ctx.sku_id,
                "run_id":                ctx.run_id,
                "assigned_model":                  ctx.assigned_model,
                "insufficient_seasonal_history":   ctx.insufficient_seasonal_history,
                "pattern_label":                   ctx.pattern_label,
                "lifecycle_stage":       ctx.lifecycle_stage,
                "selected_quantile":     ctx.selected_quantile,
                "quantile_source":       ctx.quantile_source,
                "effective_max_horizon": ctx.effective_max_horizon,
                "learning_mode":         ctx.learning_mode,
                "oos_adjustment_factor": ctx.oos_adjustment_factor,
                "is_b2b":                ctx.is_b2b,
                "reorder_bias_factor":   ctx.reorder_bias_factor,
            },
        )
    except Exception as exc:
        log.error(
            "sub_stage_91 sku=%s: BatchWriter.queue (model_initialization_s9) failed: %s",
            sku_id, exc,
        )

    return ctx
