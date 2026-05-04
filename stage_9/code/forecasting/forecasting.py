"""
Sub-Stage 9.5 — Forecast Generation and Confidence.

Five ordered steps per the master spec:

  1. generate_horizons        — fit the assigned model and produce a point
     forecast at each of the 8 locked HORIZONS. Prophet/NeuralProphet do
     ONE 365-day fit and read cumulative sums at the boundaries; SES /
     Holt / Naive / Croston produce a level and we linear-scale per
     horizon. NEVER scale Prophet from 30d.
  2. bootstrap_quantiles      — caller-supplied bootstrap_fn turns each
     point + residuals into {mean, p50, p80, p90}. p50 ≤ p80 ≤ p90 is
     enforced after.
  3. reasonableness_check     — daily_30d vs rolling_90d_avg. Out-of-band
     adds 'forecast_unusually_high' / '..._low' and applies a confidence
     multiplier.
  4. compute_confidence       — 5-step formula, all values from
     TenantParams + adaptive_quantile_state. Clamped to
     [confidence_floor, confidence_ceiling].
  5. determine_status         — first match wins:
     watchlist → high_mape → low confidence → 'forecasted'.

Orchestrator run_substage_95 wires them together with per-SKU isolation
(Principle 3), writes the `forecasts` row through the BatchWriter, and
emits one direct cross_agent_signals row per SKU on `signal_conn` (the
same isolation pattern 9.4 uses for pattern_feedback).

Model interface — caller supplies:
    forecast_fn(model_name, train_df, horizons) -> ForecastBundle
    bootstrap_fn(point, residuals, pattern)     -> {mean,p50,p80,p90}
    clearance_adjust_fn(points, lifecycle, discount, params) -> points

Default bootstrap_fn is models.bootstrap.bootstrap_quantiles (log-normal
proxy when residuals < 3, resampling otherwise). Default clearance is a
no-op — wire in the real ClearanceAdjustment from your task tracker when
it exists.
"""

from __future__ import annotations

import logging
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from forecasting.confidence import ForecastContext, EXCEPTION_PENALTY_FLAGS, compute_confidence  # noqa: F401
from infrastructure.constants import (
    CONFIDENCE_BASE_PARAM,
    REASONABLE_HORIZON_DAYS,
    REASONABLENESS_ROLLING_BASELINE_DAYS,
    HORIZONS,
    PROPHET_FAMILY,
)
from models.bootstrap import bootstrap_quantiles
from infrastructure.tenant_params import TenantParams

# Module-level psycopg2 import so _jsonb / signal payload don't re-import per
# call. Falls back gracefully when psycopg2 isn't installed (test environment).
try:
    from psycopg2.extras import Json as _PsycopgJson  # type: ignore
except Exception:  # pragma: no cover
    _PsycopgJson = None

logger = logging.getLogger(__name__)

__all__ = [
    "ForecastContext",
    "ForecastBundle",
    "ForecastResult",
    "SkuForecastInput",
    "CalibrationGapCache",
    "prefetch_calibration_gaps",
    "generate_horizons",
    "bootstrap_quantiles_for_horizons",
    "reasonableness_check",
    "compute_confidence",
    "determine_status",
    "determine_tier",
    "determine_risk_level",
    "emit_forecast_risk_signal",
    "no_clearance_adjust",
    "run_substage_95",
    "run_substage_95_parallel",
]

# ---------------------------------------------------------------------------
# Constants — one block at the top so behavior is easy to audit
# ---------------------------------------------------------------------------

# EXCEPTION_PENALTY_FLAGS and ForecastContext are defined in confidence.py
# and imported above. They remain accessible as forecasting.EXCEPTION_PENALTY_FLAGS
# and forecasting.ForecastContext for any caller that imports them from here.

# Default set of risk levels that trigger a forecast_risk signal. Scaling
# constraint: emitting per SKU at 5M-row scale floods WAL. We default to
# emitting only the actually-actionable risk tiers; pass risk_levels_to_emit
# to override (e.g. {'low','medium','high'} for full emission during debug).
DEFAULT_SIGNAL_RISK_LEVELS = frozenset({"medium", "high"})

# cross_agent_signals direct-write retry policy (mirrors pattern_feedback).
SIGNAL_MAX_RETRIES = 3
SIGNAL_RETRY_DELAY_S = 0.1
SIGNAL_TTL_HOURS = 2160  # 90 days


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

# ForecastContext is defined in confidence.py and imported above.


@dataclass
class ForecastBundle:
    """What forecast_fn returns for one SKU."""
    points_per_horizon: dict[int, float]
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SkuForecastInput:
    """Per-SKU payload to run_substage_95."""
    sku_id: str
    assigned_model: str
    pattern_label: str
    selected_quantile: float
    df: pd.DataFrame
    backtest_mape: float
    exception_flags: list[str]
    stage8_confidence: float
    lifecycle_stage: Optional[str] = None
    discount_series: Optional[np.ndarray] = None
    processing_tier: Optional[str] = None
    is_b2b: Optional[bool] = None
    dow_multipliers: list = field(default_factory=lambda: [1.0] * 7)


@dataclass
class ForecastResult:
    sku_id: str
    forecasts: dict[int, dict[str, float]]
    confidence_base: float
    confidence_final: float
    confidence_tier: str
    status: str
    exception_flags: list[str]
    risk_level: str
    selected_quantile: float


# Type aliases for the injectable callables.
ForecastFn = Callable[[str, pd.DataFrame, list[int]], ForecastBundle]
BootstrapFn = Callable[[float, Optional[np.ndarray], str], dict[str, float]]
ClearanceAdjustFn = Callable[
    [dict[int, float], Optional[str], Optional[np.ndarray], TenantParams],
    dict[int, float],
]

# ---------------------------------------------------------------------------
# Calibration gap cache (analogue of backtest's prefetch_calibrated_windows)
# ---------------------------------------------------------------------------

CalibrationGapCache = dict[tuple[str, str], float]

_SELECT_TENANT_GAPS = (
    "SELECT pattern_label, assigned_model, calibration_gap, horizon_days "
    "FROM stage9.adaptive_quantile_state "
    "WHERE tenant_id = %s AND calibration_gap IS NOT NULL "
    "ORDER BY last_updated DESC NULLS LAST"
)


def prefetch_calibration_gaps(conn, tenant_id: str) -> CalibrationGapCache:
    """
    One bulk SELECT per tenant. Prefer the row at horizon_days=30 (the
    primary horizon Stage 8 cares about); fall back to the largest
    horizon if the 30-day row is missing.
    """
    cache: CalibrationGapCache = {}
    best_score: dict[tuple[str, str], int] = {}
    with conn.cursor() as cur:
        cur.execute(_SELECT_TENANT_GAPS, (tenant_id,))
        for pattern, model, gap, horizon in cur.fetchall():
            if gap is None:
                continue
            # score: 30-day rows beat anything; otherwise larger horizon wins.
            score = 10_000 if horizon == 30 else int(horizon)
            key = (pattern, model)
            if score > best_score.get(key, -1):
                best_score[key] = score
                cache[key] = float(gap)
    return cache


# ---------------------------------------------------------------------------
# Step 1 — Generate 8-horizon point forecasts
# ---------------------------------------------------------------------------

def generate_horizons(
        model_name: str,
        train_df: pd.DataFrame,
        forecast_fn: ForecastFn,
        ctx: ForecastContext,
        *,
        lifecycle_stage: Optional[str] = None,
        discount_series: Optional[np.ndarray] = None,
        params: Optional[TenantParams] = None,
        clearance_adjust_fn: ClearanceAdjustFn = None,
) -> dict[int, float]:
    """
    Run the model and post-process the resulting points:
      * apply OOS adjustment to every horizon
      * cap any horizon > effective_max_horizon to the cap value
      * apply ClearanceAdjustment when lifecycle is 'clearance' or the
        injected clearance gate fires
    """
    if ctx.training_data_truncated and "break_index" in train_df.attrs:
        idx = int(train_df.attrs["break_index"])
        train_df = train_df.iloc[idx:]

    bundle = forecast_fn(model_name, train_df, list(HORIZONS))
    if set(bundle.points_per_horizon) != set(HORIZONS):
        raise ValueError(
            f"forecast_fn must return all {len(HORIZONS)} HORIZONS keys; "
            f"got {sorted(bundle.points_per_horizon)}"
        )

    points = {h: float(bundle.points_per_horizon[h]) for h in HORIZONS}
    points = _apply_oos(points, ctx.oos_adjustment_factor)
    points = _cap_to_effective_horizon(points, ctx.effective_max_horizon)

    if clearance_adjust_fn is not None and clearance_adjust_fn is not no_clearance_adjust:
        points = clearance_adjust_fn(points, lifecycle_stage, discount_series, params)

    # Attach residuals to the result via a shared dict on the caller side —
    # we return points only here, residuals flow through the orchestrator.
    return points


def _apply_oos(points: dict[int, float], factor: float) -> dict[int, float]:
    if factor == 1.0:
        return points
    return {h: float(v) * factor for h, v in points.items()}


def _apply_dow_multipliers(
    points: dict[int, float],
    dow_multipliers: list[float],
    df: pd.DataFrame,
) -> dict[int, float]:
    """
    Reshape flat horizon totals using DoW demand multipliers.

    For each horizon H, the original total assumes a flat daily rate
    (total / H). We replace that with the sum of DoW-shaped daily values
    over the actual forecast window, preserving the weekly total on average
    (multipliers average to ~1.0 across active days).

    Skipped automatically when dow_multipliers is flat [1.0]*7.
    """
    if df.empty or dow_multipliers == [1.0] * 7:
        return points
    start_date = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    adjusted: dict[int, float] = {}
    for h, total in points.items():
        if total == 0.0:
            adjusted[h] = total
            continue
        dow_sum = sum(
            dow_multipliers[(start_date + pd.Timedelta(days=i)).dayofweek]
            for i in range(h)
        )
        adjusted[h] = (total / h) * dow_sum
    return adjusted


def _cap_to_effective_horizon(
        points: dict[int, float], effective_max: int
) -> dict[int, float]:
    """
    For any horizon longer than effective_max (shelf life / planned end),
    Stage 10 should never use the value past that date. We hold the value
    flat at the cap-horizon's level so downstream scaling stays sane.
    """
    if effective_max >= max(HORIZONS):
        return points
    capped: dict[int, float] = {}
    cap_value: Optional[float] = None
    for h in HORIZONS:
        if h <= effective_max:
            cap_value = points[h]
            capped[h] = points[h]
        else:
            capped[h] = cap_value if cap_value is not None else points[h]
    return capped


def no_clearance_adjust(
        points: dict[int, float],
        _lifecycle_stage: Optional[str],
        _discount_series: Optional[np.ndarray],
        _params: Optional[TenantParams],
) -> dict[int, float]:
    return points


# ---------------------------------------------------------------------------
# Step 2 — Bootstrap quantiles per horizon
# ---------------------------------------------------------------------------

def bootstrap_quantiles_for_horizons(
        points: dict[int, float],
        residuals: Optional[np.ndarray],
        pattern: str,
        bootstrap_fn: BootstrapFn,
) -> dict[int, dict[str, float]]:
    """
    Returns {horizon: {mean, p50, p80, p90}} with p50 ≤ p80 ≤ p90 enforced.
    """
    out: dict[int, dict[str, float]] = {}
    for h in HORIZONS:
        q = bootstrap_fn(points[h], residuals, pattern)
        out[h] = _enforce_quantile_monotonicity(q)
    return out


def _enforce_quantile_monotonicity(q: dict[str, float]) -> dict[str, float]:
    p50 = float(q["p50"])
    p80 = max(p50, float(q["p80"]))
    p90 = max(p80, float(q["p90"]))
    return {"mean": float(q["mean"]), "p50": p50, "p80": p80, "p90": p90}


# ---------------------------------------------------------------------------
# Step 3 — Reasonableness check
# ---------------------------------------------------------------------------

_REASONABLENESS_MIN_DAYS = 14  # below this we genuinely can't form a baseline


def reasonableness_check(
        forecast_30d_mean: float,
        df: pd.DataFrame,
        params: TenantParams,
) -> tuple[list[str], float]:
    """
    Compare the 30-day daily-equivalent forecast to the rolling
    historical average. Window is min(90, len(df)) so SKUs with 60 days
    of history still get a check (just over a shorter baseline).

    Returns ([], 1.0) only when there's truly no usable baseline:
      * fewer than _REASONABLENESS_MIN_DAYS rows, OR
      * the window's mean is <= 0 (all-zero or pathological)
    """
    if len(df) < _REASONABLENESS_MIN_DAYS:
        return [], 1.0

    window_days = min(REASONABLENESS_ROLLING_BASELINE_DAYS, len(df))
    rolling_avg = float(df["qty"].iloc[-window_days:].mean())
    if rolling_avg <= 0:
        return [], 1.0

    daily_30d = forecast_30d_mean / REASONABLE_HORIZON_DAYS
    high_band = rolling_avg * params.get("max_forecast_vs_baseline")
    low_band = rolling_avg * params.get("min_forecast_vs_baseline")

    flags: list[str] = []
    multiplier = 1.0
    if daily_30d > high_band:
        flags.append("forecast_unusually_high")
        multiplier *= params.get("forecast_unusually_high_mult")
    if daily_30d < low_band:
        flags.append("forecast_unusually_low")
        multiplier *= params.get("forecast_unusually_low_mult")
    return flags, multiplier


# ---------------------------------------------------------------------------
# Step 4 — Confidence formula
# ---------------------------------------------------------------------------
# compute_confidence() is defined in confidence.py and imported above.
# It remains accessible as forecasting.compute_confidence for any caller
# that imports it from here.


# ---------------------------------------------------------------------------
# Step 5 — Status + tier + risk level
# ---------------------------------------------------------------------------

def determine_status(
        confidence_final: float,
        exception_flags: list[str],
        ctx: ForecastContext,
        params: TenantParams,
) -> str:
    """First match wins."""
    if ctx.on_watchlist:
        return "watchlist_review"
    if "high_mape" in exception_flags:
        return "needs_acknowledgment"
    if confidence_final < params.get("decision_gate_threshold"):
        return "needs_acknowledgment"
    return "forecasted"


def determine_tier(confidence_final: float, params: TenantParams) -> str:
    """The four-tier confidence label written to forecasts.confidence_tier."""
    if confidence_final >= params.get("decision_gate_threshold"):
        return "auto_proceed"
    if confidence_final >= params.get("review_suggested_threshold"):
        return "review_suggested"
    if confidence_final >= params.get("review_required_threshold"):
        return "review_required"
    return "manual_override"


def determine_risk_level(
        confidence_final: float,
        params: TenantParams,
) -> str:
    """Map confidence_final → 'low' / 'medium' / 'high' band."""
    low_min = params.get("risk_low_min")
    med_min = params.get("risk_medium_min")
    if confidence_final >= low_min:
        return "low"
    if confidence_final >= med_min:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Output — direct write to cross_agent_signals (mirrors pattern_feedback)
# ---------------------------------------------------------------------------

_INSERT_FORECAST_RISK_SIGNAL = """
INSERT INTO stage9.cross_agent_signals (
    signal_id, tenant_id, from_agent, to_agent, signal_type,
    sku_id, run_id, payload, confidence, processed,
    created_at, expires_at
) VALUES (
    %s, %s, 'stage9', 'stage10', 'forecast_risk',
    %s, %s, %s, %s, FALSE,
    NOW(), NOW() + make_interval(hours => %s)
)
"""


def _quantile_to_key(selected_quantile: float) -> str:
    """Map selected_quantile float to the JSONB quantile key (p50 / p80 / p90)."""
    from decimal import Decimal as _D
    q = _D(str(selected_quantile))
    if q == _D("0.50"):
        return "p50"
    if q == _D("0.80"):
        return "p80"
    return "p90"


def emit_forecast_risk_signal(
        conn,
        *,
        tenant_id: str,
        sku_id: str,
        run_id: str,
        confidence_final: float,
        confidence_tier: str,
        risk_level: str,
        exception_flags: list[str],
        mape_30d: Optional[float],
        forecast_30d_selected: float,
        selected_quantile: float,
        max_retries: int = SIGNAL_MAX_RETRIES,
        retry_delay_seconds: float = SIGNAL_RETRY_DELAY_S,
) -> bool:
    """
    Direct conn.execute + conn.commit. NOT batched — Stage 10 should be
    able to read the latest risk signal even if Stage 9 crashes immediately
    after. Same retry / dedicated-conn discipline as pattern_feedback.
    """
    payload = _signal_payload(
        risk_level, confidence_final, confidence_tier,
        exception_flags, mape_30d, forecast_30d_selected, selected_quantile,
    )
    args = (
        str(uuid.uuid4()), tenant_id,
        sku_id, run_id, payload, float(confidence_final),
        SIGNAL_TTL_HOURS,
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with conn.cursor() as cur:
                cur.execute(_INSERT_FORECAST_RISK_SIGNAL, args)
            conn.commit()
            return True
        except Exception as e:
            last_err = e
            logger.warning(
                "forecast_risk signal write attempt %d/%d failed sku_id=%s err=%s",
                attempt, max_retries, sku_id, e,
            )
            try:
                conn.rollback()
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)

    logger.error(
        "forecast_risk signal write FAILED after %d attempts sku_id=%s err=%s",
        max_retries, sku_id, last_err,
    )
    return False


def _signal_payload(
        risk_level: str,
        confidence_final: float,
        confidence_tier: str,
        exception_flags: list[str],
        mape_30d: Optional[float],
        forecast_30d_selected: float,
        selected_quantile: float,
):
    payload = {
        "risk_level": risk_level,
        "confidence_final": float(confidence_final),
        "confidence_tier": confidence_tier,
        "exception_flags": list(exception_flags),
        "mape_30d": float(mape_30d) if mape_30d is not None else None,
        "forecast_30d_selected": float(forecast_30d_selected),
        "selected_quantile": float(selected_quantile),
    }
    if _PsycopgJson is None:
        return payload
    return _PsycopgJson(payload)


# ---------------------------------------------------------------------------
# Orchestrator — single process (Principle 3: per-SKU isolation)
# ---------------------------------------------------------------------------

def run_substage_95(
        conn,
        *,
        tenant_id: str,
        run_id: str,
        skus: Iterable[SkuForecastInput],
        params: TenantParams,
        forecast_fn: ForecastFn,
        batch_writer,
        contexts: dict[str, ForecastContext],
        bootstrap_fn: BootstrapFn = bootstrap_quantiles,
        clearance_adjust_fn: ClearanceAdjustFn = None,
        signal_conn=None,
        calibration_gaps: Optional[CalibrationGapCache] = None,
        risk_levels_to_emit: Iterable[str] = DEFAULT_SIGNAL_RISK_LEVELS,
        log_failure_fn: Optional[Callable[[str, str, str, str], None]] = None,
) -> dict[str, ForecastResult]:
    """
    Run Sub-Stage 9.5 for every SKU.

    Per SKU (always — even on failure):
      1-3. Generate horizons → bootstrap quantiles → reasonableness check
      4.   Compute confidence
      5.   Determine status / tier / risk level
      6.   Queue `forecasts` row via BatchWriter (failed SKUs get a fallback
           row with status='needs_acknowledgment' and zeroed quantiles —
           Stage 10 must never see a missing row)
      7.   Emit forecast_risk signal IFF risk_level in risk_levels_to_emit.
           Default emits only {'medium','high'}; pass {'low','medium','high'}
           for full emission.

    `signal_conn` defaults to `conn` and emits a UserWarning — production
    callers MUST pass a dedicated connection so the direct commit doesn't
    touch other in-flight transactions on `conn` (mirrors 9.4's `pf_conn`).

    `log_failure_fn(tenant_id, run_id, sku_id, reason)` records each
    forecast_failed SKU to stage9_sku_execution_log. Defaults to a
    logger.warning.
    """
    if signal_conn is None:
        warnings.warn(
            "signal_conn defaulted to conn — in production pass a dedicated "
            "connection so cross_agent_signals commits don't share transaction "
            "state with batched writes.",
            stacklevel=2,
        )
        signal_conn = conn

    risk_set = frozenset(risk_levels_to_emit)
    log_failure = log_failure_fn or _default_substage_95_log_failure
    clearance_fn = clearance_adjust_fn or no_clearance_adjust

    results: dict[str, ForecastResult] = {}
    for payload in skus:
        ctx = contexts.get(payload.sku_id) or ForecastContext()
        results[payload.sku_id] = _run_one_sku(
            conn,
            signal_conn=signal_conn,
            tenant_id=tenant_id,
            run_id=run_id,
            payload=payload,
            params=params,
            forecast_fn=forecast_fn,
            bootstrap_fn=bootstrap_fn,
            clearance_adjust_fn=clearance_fn,
            batch_writer=batch_writer,
            calibration_gaps=calibration_gaps,
            ctx=ctx,
            risk_set=risk_set,
            log_failure=log_failure,
        )
    return results


def _run_one_sku(
        _conn,
        *,
        signal_conn,
        tenant_id: str,
        run_id: str,
        payload: SkuForecastInput,
        params: TenantParams,
        forecast_fn: ForecastFn,
        bootstrap_fn: BootstrapFn,
        clearance_adjust_fn: ClearanceAdjustFn,
        batch_writer,
        calibration_gaps: Optional[CalibrationGapCache],
        ctx: ForecastContext,
        risk_set: frozenset,
        log_failure: Callable[[str, str, str, str], None],
) -> ForecastResult:
    try:
        bundle = forecast_fn(payload.assigned_model, payload.df, list(HORIZONS))
        if set(bundle.points_per_horizon) != set(HORIZONS):
            raise ValueError(
                f"forecast_fn returned wrong horizon set: "
                f"{sorted(bundle.points_per_horizon)}"
            )

        points = _post_process_points(
            bundle.points_per_horizon, ctx,
            lifecycle_stage=payload.lifecycle_stage,
            discount_series=payload.discount_series,
            params=params,
            clearance_adjust_fn=clearance_adjust_fn,
        )
        if payload.assigned_model not in PROPHET_FAMILY:
            points = _apply_dow_multipliers(
                points, payload.dow_multipliers, payload.df,
            )
        forecasts = bootstrap_quantiles_for_horizons(
            points, bundle.residuals, payload.pattern_label, bootstrap_fn,
        )
        reason_flags, reason_mult = reasonableness_check(
            forecasts[REASONABLE_HORIZON_DAYS]["mean"], payload.df, params,
        )
        all_flags = list(payload.exception_flags) + reason_flags

        gap = (
            calibration_gaps.get((payload.pattern_label, payload.assigned_model))
            if calibration_gaps is not None else None
        )
        confidence_base, confidence_final = compute_confidence(
            pattern_label=payload.pattern_label,
            backtest_mape=payload.backtest_mape,
            exception_flags=all_flags,
            calibration_gap=gap,
            stage8_confidence=payload.stage8_confidence,
            reorder_bias_factor=ctx.reorder_bias_factor,
            ctx=ctx,
            params=params,
            reasonableness_multiplier=reason_mult,
        )

        status = determine_status(confidence_final, all_flags, ctx, params)
        tier = determine_tier(confidence_final, params)
        risk_level = determine_risk_level(confidence_final, params)

    except MemoryError:
        # Don't catch real infra failures — let the run die.
        raise
    except Exception:
        logger.exception(
            "forecast generation failed sku_id=%s model=%s — writing fallback row",
            payload.sku_id, payload.assigned_model,
        )
        log_failure(tenant_id, run_id, payload.sku_id, "forecast_failed")
        # Build a defaulted result. Stage 10 ALWAYS sees a row.
        forecasts = _zeroed_forecasts()
        all_flags = list(payload.exception_flags) + ["forecast_failed"]
        confidence_base = 0.0
        confidence_final = params.get("confidence_floor")
        tier = "manual_override"
        status = "needs_acknowledgment"
        risk_level = "high"

    # Output 1: ALWAYS queue forecasts row (success or fallback).
    batch_writer.queue(
        "forecasts",
        _forecasts_row(
            tenant_id=tenant_id, run_id=run_id, payload=payload,
            forecasts=forecasts, confidence_base=confidence_base,
            confidence_final=confidence_final, tier=tier, status=status,
            exception_flags=all_flags, backtest_mape=payload.backtest_mape,
            context=ctx,
        ),
    )
    batch_writer.flush_if_needed()

    # Output 2: direct cross_agent_signals write — gated by risk level.
    if risk_level in risk_set:
        fc_30d_key = _quantile_to_key(payload.selected_quantile)
        emit_forecast_risk_signal(
            signal_conn,
            tenant_id=tenant_id, sku_id=payload.sku_id, run_id=run_id,
            confidence_final=confidence_final, confidence_tier=tier,
            risk_level=risk_level, exception_flags=all_flags,
            mape_30d=payload.backtest_mape,
            forecast_30d_selected=forecasts[REASONABLE_HORIZON_DAYS].get(fc_30d_key, 0.0),
            selected_quantile=payload.selected_quantile,
        )

    return ForecastResult(
        sku_id=payload.sku_id,
        forecasts=forecasts,
        confidence_base=confidence_base,
        confidence_final=confidence_final,
        confidence_tier=tier,
        status=status,
        exception_flags=all_flags,
        risk_level=risk_level,
        selected_quantile=payload.selected_quantile,
    )


def _post_process_points(
        points_in: dict[int, float],
        ctx: ForecastContext,
        *,
        lifecycle_stage: Optional[str],
        discount_series: Optional[np.ndarray],
        params: Optional[TenantParams],
        clearance_adjust_fn: ClearanceAdjustFn,
) -> dict[int, float]:
    """OOS adjust → cap to effective_max_horizon → optional clearance."""
    points = {h: float(points_in[h]) for h in HORIZONS}
    points = _apply_oos(points, ctx.oos_adjustment_factor)
    points = _cap_to_effective_horizon(points, ctx.effective_max_horizon)
    if clearance_adjust_fn is not None and clearance_adjust_fn is not no_clearance_adjust:
        points = clearance_adjust_fn(points, lifecycle_stage, discount_series, params)
    return points


def _zeroed_forecasts() -> dict[int, dict[str, float]]:
    """Fallback row content when a SKU's forecast generation fails."""
    zero = {"mean": 0.0, "p50": 0.0, "p80": 0.0, "p90": 0.0}
    return {h: dict(zero) for h in HORIZONS}


def _default_substage_95_log_failure(
        tenant_id: str, run_id: str, sku_id: str, reason: str,
) -> None:
    """Stand-in for the stage9_sku_execution_log writer."""
    logger.warning(
        "9.5 fallback tenant=%s run=%s sku=%s reason=%s",
        tenant_id, run_id, sku_id, reason,
    )


def _forecasts_row(
        *,
        tenant_id: str,
        run_id: str,
        payload: SkuForecastInput,
        forecasts: dict[int, dict[str, float]],
        confidence_base: float,
        confidence_final: float,
        tier: str,
        status: str,
        exception_flags: list[str],
        backtest_mape: float,
        context: "ForecastContext",
) -> dict[str, Any]:
    import datetime
    row: dict[str, Any] = {
        "tenant_id": tenant_id,
        "sku_id": payload.sku_id,
        "run_id": run_id,
        "forecast_date": datetime.date.today(),
        "assigned_model": payload.assigned_model,
        "pattern_label": payload.pattern_label,
        "selected_quantile": payload.selected_quantile,
        "confidence_base": confidence_base,
        "confidence_final": confidence_final,
        "confidence_tier": tier,
        "backtest_mape": backtest_mape,
        "exception_flags": _jsonb(exception_flags),
        "status": status,
        "lifecycle_stage": payload.lifecycle_stage,
        "processing_tier": payload.processing_tier,
        "effective_max_horizon": context.effective_max_horizon,
        "oos_adjustment_factor": context.oos_adjustment_factor,
        "reorder_bias_factor": context.reorder_bias_factor,
        "is_b2b": payload.is_b2b,
    }
    for h in HORIZONS:
        row[f"forecast_{h}d"] = _jsonb(forecasts[h])
    return row


def _jsonb(obj):
    """Wrap obj in psycopg2.Json when available; pass through for fakes."""
    if _PsycopgJson is None:
        return obj
    return _PsycopgJson(obj)


# ---------------------------------------------------------------------------
# Parallel orchestrator — process pool
# ---------------------------------------------------------------------------

def run_substage_95_parallel(
        *,
        tenant_id: str,
        run_id: str,
        skus: list[SkuForecastInput],
        contexts: dict[str, ForecastContext],
        params: TenantParams,
        forecast_fn: ForecastFn,
        connect_fn: Callable[[], Any],
        signal_connect_fn: Callable[[], Any],
        bootstrap_fn: BootstrapFn = bootstrap_quantiles,
        clearance_adjust_fn: ClearanceAdjustFn = no_clearance_adjust,
        risk_levels_to_emit: Iterable[str] = DEFAULT_SIGNAL_RISK_LEVELS,
        log_failure_fn: Optional[Callable[[str, str, str, str], None]] = None,
        max_workers: Optional[int] = None,
        executor_factory: Optional[Callable[[int], Any]] = None,
        batch_size: int = 1000,
) -> dict[str, ForecastResult]:
    """
    ProcessPoolExecutor variant. Each worker opens its own conn +
    signal_conn, prefetches calibration_gaps, runs run_substage_95 on
    its slice, and closes both connections in `finally`.
    """
    if not skus:
        return {}

    if executor_factory is None:
        from concurrent.futures import ProcessPoolExecutor as _Pool
        executor_factory = _Pool
    if max_workers is None:
        import os
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    max_workers = max(1, min(max_workers, len(skus)))

    chunks = _partition(skus, max_workers)
    combined: dict[str, ForecastResult] = {}
    with executor_factory(max_workers) as pool:
        futures = [
            pool.submit(
                _worker_run_chunk,
                tenant_id, run_id, chunk, contexts, params,
                forecast_fn, bootstrap_fn, clearance_adjust_fn,
                connect_fn, signal_connect_fn, batch_size,
                tuple(risk_levels_to_emit), log_failure_fn,
            )
            for chunk in chunks
        ]
        for fut in futures:
            combined.update(fut.result())
    return combined


def _partition(items: list, n: int) -> list[list]:
    if n <= 1 or len(items) <= 1:
        return [items]
    size, rem = divmod(len(items), n)
    chunks: list[list] = []
    start = 0
    for i in range(n):
        end = start + size + (1 if i < rem else 0)
        if start < end:
            chunks.append(items[start:end])
        start = end
    return chunks


def _worker_run_chunk(
        tenant_id: str,
        run_id: str,
        chunk: list[SkuForecastInput],
        contexts: dict[str, ForecastContext],
        params: TenantParams,
        forecast_fn: ForecastFn,
        bootstrap_fn: BootstrapFn,
        clearance_adjust_fn: ClearanceAdjustFn,
        connect_fn: Callable[[], Any],
        signal_connect_fn: Callable[[], Any],
        batch_size: int,
        risk_levels_to_emit: tuple,
        log_failure_fn: Optional[Callable[[str, str, str, str], None]],
) -> dict[str, ForecastResult]:
    from infrastructure.batch_writer import BatchWriter

    conn = connect_fn()
    signal_conn = signal_connect_fn()
    try:
        gaps = prefetch_calibration_gaps(conn, tenant_id)
        bw = BatchWriter(conn, batch_size=batch_size)
        results = run_substage_95(
            conn,
            tenant_id=tenant_id,
            run_id=run_id,
            skus=chunk,
            params=params,
            forecast_fn=forecast_fn,
            batch_writer=bw,
            contexts=contexts,
            bootstrap_fn=bootstrap_fn,
            clearance_adjust_fn=clearance_adjust_fn,
            signal_conn=signal_conn,
            calibration_gaps=gaps,
            risk_levels_to_emit=risk_levels_to_emit,
            log_failure_fn=log_failure_fn,
        )
        bw.flush()
        return results
    finally:
        for c in (conn, signal_conn):
            try:
                c.close()
            except Exception:
                pass
