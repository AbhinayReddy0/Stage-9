"""
tier_router.py — Tier-aware SKU execution router for Stage 9.

Dispatches each SKU to the correct sub-stage execution path based on the
processing tier assigned by the Preloader's fingerprint classifier:

  full    — 9.1 → 9.2 → 9.3 → closures  (handed to 9.4 / 9.5)
  partial — 9.1 → 9.2 → closures         (skips 9.3; uses best Thompson HP)
  cache   — load prior forecast → SES micro-level update → direct write
             (bypasses 9.2, 9.3, and 9.4 entirely)

Public API:
    route_sku(sku_id, df, run_ctx, preloaded_dict, db) -> SkuRoutingResult
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtesting.backtesting import BacktestContext, SkuBacktestInput
from infrastructure.constants import (
    MICRO_UPDATE_SES_ALPHA,
    MICRO_UPDATE_SES_WINDOW,
    MICRO_UPDATE_SCALE_MIN,
    MICRO_UPDATE_SCALE_MAX,
    ULTRA_SPARSE_OBS_THRESHOLD,
    Model,
    ProcessingTier,
)
from forecasting.forecasting import ForecastBundle, ForecastContext, SkuForecastInput
from learning.self_assessment import SKUResult
from pipeline.model_initialization import run_model_initialisation
from forecasting.feature_engg import run_feature_engineering
from models.hp_tuning import run_hp_tuning

log = logging.getLogger(__name__)

# Aliases used by tests (constants live in infrastructure.constants)
_SCALE_MIN = MICRO_UPDATE_SCALE_MIN
_SCALE_MAX = MICRO_UPDATE_SCALE_MAX

# ---------------------------------------------------------------------------
# Lazy model registry — deferred to avoid circular imports at module level
# ---------------------------------------------------------------------------

_MODEL_CLASS: dict[str, type] | None = None


def _get_model_class(model_name: str) -> type:
    global _MODEL_CLASS
    if _MODEL_CLASS is None:
        from models.croston import CrostonMethod
        from models.holt import HoltLinearTrend
        from models.naive import NaiveForecast
        from models.prophet_model import ProphetModel
        from models.ses import SESModel
        _MODEL_CLASS = {
            Model.NAIVE: NaiveForecast,
            Model.CROSTON: CrostonMethod,
            Model.PROPHET: ProphetModel,
            Model.HOLTS_LINEAR: HoltLinearTrend,
            Model.SES: SESModel,
        }
    from models.ses import SESModel as _SES
    return _MODEL_CLASS.get(model_name, _SES)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SkuRoutingResult:
    """
    Unified result from route_sku().

    full / partial: lctx, closures, and 9.4/9.5 inputs are populated.
                    sku_result is None.
    cache:          sku_result is pre-built; all pipeline fields are None.
                    Forecast already written to DB via BatchWriter.
    """
    sku_id: str
    tier: str

    # full / partial — fed into 9.4 and 9.5 by acting_handler
    lctx: Any = None
    fit_predict_fn: Any = None
    forecast_fn: Any = None
    backtest_input: Any = None  # SkuBacktestInput
    forecast_input: Any = None  # SkuForecastInput
    backtest_context: Any = None  # BacktestContext
    forecast_context: Any = None  # ForecastContext

    # cache — pre-built result; 9.4 and 9.5 are bypassed entirely
    sku_result: Any = None  # SKUResult


# ---------------------------------------------------------------------------
# SQL — prior forecast read (cache tier)
# ---------------------------------------------------------------------------

_SQL_PRIOR_FORECAST = """
    SELECT DISTINCT ON (sku_id)
        assigned_model,
        pattern_label,
        confidence_final,
        status,
        selected_quantile,
        backtest_mape,
        confidence_base,
        confidence_tier,
        exception_flags,
        lifecycle_stage,
        effective_max_horizon,
        oos_adjustment_factor,
        reorder_bias_factor,
        is_b2b,
        forecast_7d,
        forecast_14d,
        forecast_30d,
        forecast_60d,
        forecast_90d,
        forecast_150d,
        forecast_180d,
        forecast_365d
    FROM stage9.forecasts
    WHERE tenant_id  = %s
      AND sku_id::text = %s
    ORDER BY sku_id, created_at DESC
"""

_HORIZON_COLS: tuple[str, ...] = (
    "forecast_7d", "forecast_14d", "forecast_30d", "forecast_60d",
    "forecast_90d", "forecast_150d", "forecast_180d", "forecast_365d",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_best_hp_from_thompson(
        sku_id: str,
        model_name: str,
        preloaded: Any,
        default_hp: dict,
) -> dict:
    """Return the highest-confidence HP config from thompson_state, or default_hp."""
    state = preloaded.thompson_state.get((sku_id, model_name), {})
    if not state:
        return default_hp
    best_hash = max(
        state,
        key=lambda h: (
                state[h].get("alpha", 1.0)
                / (state[h].get("alpha", 1.0) + state[h].get("beta", 1.0))
        ),
    )
    config = state[best_hash].get("config")
    return config if isinstance(config, dict) and config else default_hp


def _load_prior_forecast(sku_id: str, tenant_id: str, db: Any) -> dict | None:
    """
    Load the most recent forecast row for this SKU from stage9.forecasts.
    Returns None on a cache miss or any DB error.
    """
    try:
        with db.cursor() as cur:
            cur.execute(_SQL_PRIOR_FORECAST, (tenant_id, sku_id))
            row = cur.fetchone()
        if row is None:
            return None
        cols = (
                   "assigned_model", "pattern_label", "confidence_final", "status",
                   "selected_quantile", "backtest_mape", "confidence_base", "confidence_tier",
                   "exception_flags", "lifecycle_stage",
                   "effective_max_horizon", "oos_adjustment_factor", "reorder_bias_factor", "is_b2b",
               ) + _HORIZON_COLS
        return dict(zip(cols, row))
    except Exception as exc:
        log.warning("tier_router prior_forecast_load failed sku=%s: %s", sku_id, exc)
        return None


def _compute_level_scale(df: pd.DataFrame) -> float:
    """
    Run a short SES pass over the last MICRO_UPDATE_SES_WINDOW days of demand and
    return new_level / initial_level, clamped to [MICRO_UPDATE_SCALE_MIN, MICRO_UPDATE_SCALE_MAX].

    initial_level is the mean of all nonzero values in the window so that
    a single outlier day does not set a misleading baseline.
    Returns 1.0 on degenerate inputs (all-zero demand or fewer than 2 rows).
    """
    recent = df["qty"].iloc[-MICRO_UPDATE_SES_WINDOW:].values.astype(float)
    if len(recent) < 2:
        return 1.0

    nonzero = recent[recent > 0]
    initial = float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    if initial == 0.0:
        return 1.0

    level = initial
    for qty in recent:
        level = MICRO_UPDATE_SES_ALPHA * qty + (1.0 - MICRO_UPDATE_SES_ALPHA) * level

    return float(np.clip(level / initial, MICRO_UPDATE_SCALE_MIN, MICRO_UPDATE_SCALE_MAX))


def _scale_horizons(prior: dict, scale: float) -> dict:
    """Multiply every quantile value across all eight horizon columns by scale."""
    result: dict[str, Any] = {}
    for col in _HORIZON_COLS:
        quantiles = prior.get(col)
        if isinstance(quantiles, dict):
            result[col] = {k: round(float(v) * scale, 4) for k, v in quantiles.items()}
        else:
            result[col] = quantiles  # pass through unexpected types unchanged
    return result


# ---------------------------------------------------------------------------
# Fit / forecast closures
# ---------------------------------------------------------------------------

def _make_fit_predict_fn(model_cls: type, best_hp: dict, features: list):
    """fit_predict_fn(df, test_len) -> np.ndarray — used by run_substage_94."""

    def fit_predict(df: pd.DataFrame, test_len: int) -> np.ndarray:
        m = model_cls(hp=best_hp)
        m.fit(df, features)
        return m.predict(df, features, horizon=test_len)

    return fit_predict


def _make_forecast_fn(model_cls: type, best_hp: dict, features: list):
    """forecast_fn(model_name, train_df, horizons) -> ForecastBundle — used by run_substage_95."""

    def forecast_fn(_model_name: str, train_df: pd.DataFrame, horizons: list) -> ForecastBundle:
        m = model_cls(hp=best_hp)
        m.fit(train_df, features)
        residuals = m.compute_residuals(train_df, features)
        points = {
            h: float(np.sum(m.predict(train_df, features, horizon=h)))
            for h in horizons
        }
        return ForecastBundle(points_per_horizon=points, residuals=residuals)

    return forecast_fn


# ---------------------------------------------------------------------------
# Shared pipeline-result builder (full and partial)
# ---------------------------------------------------------------------------

def _build_pipeline_result(
        sku_id: str,
        tier: str,
        lctx: Any,
        model_cls: type,
        train_df: pd.DataFrame,
        preloaded: Any,
) -> SkuRoutingResult:
    """Build closures and 9.4/9.5 inputs from a completed 9.1 LearningContext."""
    features = lctx.selected_features or model_cls(hp={}).required_features
    pattern_ctx = preloaded.pattern_ctx.get(sku_id, {})

    return SkuRoutingResult(
        sku_id=sku_id,
        tier=tier,
        lctx=lctx,
        fit_predict_fn=_make_fit_predict_fn(model_cls, lctx.best_hp, features),
        forecast_fn=_make_forecast_fn(model_cls, lctx.best_hp, features),
        backtest_input=SkuBacktestInput(
            sku_id=sku_id,
            assigned_model=lctx.assigned_model,
            pattern_label=lctx.pattern_label,
            model_hint=None,
            stage8_confidence=None,
            df=train_df,
            obs_days=pattern_ctx.get("obs_days", len(train_df)),
            ultra_sparse=(len(train_df) < ULTRA_SPARSE_OBS_THRESHOLD),
            learning_mode=lctx.learning_mode,
        ),
        forecast_input=SkuForecastInput(
            sku_id=sku_id,
            assigned_model=lctx.assigned_model,
            pattern_label=lctx.pattern_label,
            selected_quantile=lctx.selected_quantile,
            df=train_df,
            backtest_mape=0.0,  # filled after 9.4
            exception_flags=[],  # filled after 9.4
            stage8_confidence=1.0,
            lifecycle_stage=lctx.lifecycle_stage,
            processing_tier=tier,
            is_b2b=lctx.is_b2b,
        ),
        backtest_context=BacktestContext(),
        forecast_context=ForecastContext(
            effective_max_horizon=lctx.effective_max_horizon,
            reorder_bias_factor=lctx.reorder_bias_factor,
            oos_adjustment_factor=lctx.oos_adjustment_factor,
            on_watchlist=pattern_ctx.get("on_watchlist", False),
        ),
    )


# ---------------------------------------------------------------------------
# Tier-specific route functions
# ---------------------------------------------------------------------------

def _route_full(
        sku_id: str,
        df: pd.DataFrame,
        run_ctx: Any,
        preloaded_dict: dict,
) -> SkuRoutingResult:
    """9.1 → 9.2 → 9.3 → closures + 9.4/9.5 inputs."""
    lctx = run_model_initialisation(
        sku_id=sku_id,
        preloaded=run_ctx.preloaded,
        params=run_ctx.params,
        batch_writer=run_ctx.batch_writer,
        consumer=run_ctx.signal_consumer,
        run_id=run_ctx.run_id,
    )
    model_cls = _get_model_class(lctx.assigned_model)
    default_hp = model_cls(hp={}).default_hp

    fe = run_feature_engineering(
        ctx=lctx,
        df=df,
        model=model_cls(hp=default_hp),
        preloaded=preloaded_dict,
        params=run_ctx.params,
        batch_writer=run_ctx.batch_writer,
    )
    hp = run_hp_tuning(
        ctx=lctx,
        df_train=fe.df_train if fe.df_train is not None else df,
        model=model_cls(hp=default_hp),
        preloaded=preloaded_dict,
        params=run_ctx.params,
        batch_writer=run_ctx.batch_writer,
    )
    lctx.best_hp = hp.best_hp
    lctx.validation_mape = hp.validation_mape
    train_df = fe.df_train if fe.df_train is not None else df

    return _build_pipeline_result(
        sku_id, ProcessingTier.FULL, lctx, model_cls, train_df, run_ctx.preloaded,
    )


def _route_partial(
        sku_id: str,
        df: pd.DataFrame,
        run_ctx: Any,
        preloaded_dict: dict,
) -> SkuRoutingResult:
    """9.1 → 9.2 → closures + 9.4/9.5 inputs. 9.3 skipped; best HP from Thompson state."""
    lctx = run_model_initialisation(
        sku_id=sku_id,
        preloaded=run_ctx.preloaded,
        params=run_ctx.params,
        batch_writer=run_ctx.batch_writer,
        consumer=run_ctx.signal_consumer,
        run_id=run_ctx.run_id,
    )
    model_cls = _get_model_class(lctx.assigned_model)
    default_hp = model_cls(hp={}).default_hp

    fe = run_feature_engineering(
        ctx=lctx,
        df=df,
        model=model_cls(hp=default_hp),
        preloaded=preloaded_dict,
        params=run_ctx.params,
        batch_writer=run_ctx.batch_writer,
    )
    lctx.best_hp = _get_best_hp_from_thompson(
        sku_id, lctx.assigned_model, run_ctx.preloaded, default_hp,
    )
    lctx.validation_mape = 1.0  # unknown — no 9.3 ran
    train_df = fe.df_train if fe.df_train is not None else df

    return _build_pipeline_result(
        sku_id, ProcessingTier.PARTIAL, lctx, model_cls, train_df, run_ctx.preloaded,
    )


def route_cache(
        sku_id: str,
        df: pd.DataFrame,
        run_ctx: Any,
        db: Any,
) -> SkuRoutingResult | None:
    """
    Cache path — load the prior forecast, apply SES micro-level update,
    write the adjusted forecast via BatchWriter, return a pre-built SKUResult.

    Returns None if no prior forecast row exists in stage9.forecasts
    (caller promotes the SKU to full tier).
    """
    prior = _load_prior_forecast(sku_id, run_ctx.tenant_id, db)
    if prior is None:
        return None

    scale = _compute_level_scale(df)
    scaled_horizons = _scale_horizons(prior, scale)

    run_ctx.batch_writer.queue("forecasts", {
        "tenant_id": run_ctx.tenant_id,
        "sku_id": sku_id,
        "run_id": run_ctx.run_id,
        "forecast_date": datetime.date.today(),
        "assigned_model": prior["assigned_model"],
        "pattern_label": prior["pattern_label"],
        "confidence_final": prior["confidence_final"],
        "status": prior["status"],
        "selected_quantile": prior["selected_quantile"],
        "backtest_mape": prior["backtest_mape"],
        "confidence_base": prior["confidence_base"],
        "confidence_tier": prior["confidence_tier"],
        "exception_flags": prior["exception_flags"],
        "lifecycle_stage": prior["lifecycle_stage"],
        "processing_tier": ProcessingTier.CACHE,
        "effective_max_horizon": prior["effective_max_horizon"],
        "oos_adjustment_factor": prior["oos_adjustment_factor"],
        "reorder_bias_factor": prior["reorder_bias_factor"],
        "is_b2b": prior["is_b2b"],
        **scaled_horizons,
    })

    pctx = run_ctx.preloaded.pattern_ctx.get(sku_id, {})
    return SkuRoutingResult(
        sku_id=sku_id,
        tier=ProcessingTier.CACHE,
        sku_result=SKUResult(
            sku_id=sku_id,
            status=prior["status"],
            confidence_final=float(prior["confidence_final"] or 0.0),
            processing_tier=ProcessingTier.CACHE,
            assigned_model=prior["assigned_model"] or Model.SES,
            used_fallback=False,
            pattern_label=pctx.get("pattern_label", prior.get("pattern_label", "stable")),
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def route_sku_micro_update(
        sku_id: str,
        df: pd.DataFrame,
        run_ctx: Any,
        db: Any,
) -> SkuRoutingResult | None:
    """
    Micro-update path: SES level correction only, no model retraining.

    Delegates directly to route_cache.  Returns None when no prior forecast
    row exists for this SKU — the caller skips those SKUs silently (new SKUs
    added mid-day are picked up on the next full run).
    """
    return route_cache(sku_id, df, run_ctx, db)


def route_sku(
        sku_id: str,
        df: pd.DataFrame,
        run_ctx: Any,
        preloaded_dict: dict,
        db: Any,
) -> SkuRoutingResult:
    """
    Route one SKU to the correct execution path based on its preloaded tier.

    Automatic fallbacks:
      - SKU not in sku_tiers           → treated as 'full'
      - Cache tier, no prior DB row    → promoted to 'full'

    Raises on unhandled exception — acting_handler wraps this call in
    try/except for per-SKU isolation (Principle 3).
    """
    tier = run_ctx.preloaded.sku_tiers.get(sku_id, ProcessingTier.FULL)

    if tier == ProcessingTier.CACHE:
        result = route_cache(sku_id, df, run_ctx, db)
        if result is not None:
            return result
        log.debug("tier_router cache_miss sku=%s — promoting to full", sku_id)
        tier = ProcessingTier.FULL

    if tier == ProcessingTier.PARTIAL:
        return _route_partial(sku_id, df, run_ctx, preloaded_dict)

    return _route_full(sku_id, df, run_ctx, preloaded_dict)
