"""
ACTING handler — Task 9.

Runs Sub-Stages 9.1 → 9.5 for every SKU in scope. Three execution tracks:

  1. MICRO_UPDATE mode (set by preloading_handler when the last completed
     run was within `micro_update_threshold_hours`)
       → SES level correction only, no model retraining, target < 15 sec.

  2. CACHE tier (set by preloader's fingerprint classifier)
       → Reuse prior forecast, apply SES level correction, queue into the
         shared BatchWriter. Bypasses 9.1 / 9.2 / 9.3 / 9.4 / 9.5 entirely.

  3. FULL + PARTIAL tier
       → Dispatch every SKU to the dual_pool concurrent executor:
            ProcessPool (4 workers, 120s) for Prophet
            ThreadPool (16 workers, 30s)  for Naive / Croston / Holt / SES
         Each worker runs orchestrator.run_one_sku which invokes
         9.2 → 9.3 → 9.4 → 9.5 for one SKU. pattern_feedback and
         cross_agent_signals are written DIRECTLY by the worker (P4 sacred
         + Stage-10 visibility); other rows are returned in `batch_rows`
         and queued by this handler into the run-level BatchWriter post-
         collection (Option B of the BatchWriter strategy).

Isolation (Principle 3): every per-SKU exception path produces a Naive
fallback at confidence_floor with status 'needs_acknowledgment',
pattern_feedback proxy MAPE, and a row in stage9_sku_execution_log. The
run continues — ONE bad SKU never stops the run.

Demand history is loaded in a single bulk SELECT before each track; for
MICRO_UPDATE we use a 30-day window, for CACHE / FULL we use MAX_DEMAND_HISTORY_DAYS
(730 days, aligned with Stage 8's lookback cap).
Stage 8 table: stage8.demand_history (sku_id, sale_date, qty).
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Any

from backtesting.backtesting import (
    BacktestContext,
    SkuBacktestInput,
    run_substage_94,
    write_pattern_feedback,
)
from models.base import get_model_class
from infrastructure.db_utils import DBConnection
from infrastructure.constants import (
    B2B_WEEKEND_ZERO_RATIO_THRESHOLD,
    CriticalityTier,
    ExecutionMode,
    ForecastStatus,
    MAX_DEMAND_HISTORY_DAYS,
    Model,
    Param,
    ProcessingTier,
)
from pipeline.dual_pool import (
    db_config_from_env,
    get_worker_tenant_invariants,
    get_worker_tenant_params,
    run_dual_pool,
    set_worker_globals,
    SkuPipelineInput,
)
from forecasting.feature_engg import run_feature_engineering
from forecasting.forecasting import ForecastBundle, ForecastContext, SkuForecastInput, run_substage_95
from models.hp_tuning import run_hp_tuning
from pipeline.model_initialization import LearningContext
from learning.self_assessment import SKUResult
from infrastructure.tenant_params import TenantParams
from forecasting.tier_router import route_cache, route_sku_micro_update
from handlers._context import RunContext, fetch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demand history bulk-read
# ---------------------------------------------------------------------------

_SQL_DEMAND_HISTORY = f"""
    SELECT sku_id::text, sale_date, qty
      FROM stage8.demand_history
     WHERE tenant_id = %s
       AND sale_date >= CURRENT_DATE - INTERVAL '{MAX_DEMAND_HISTORY_DAYS} days'
     ORDER BY sku_id, sale_date
"""

# Micro-update only needs the SES window (14 days) plus a small buffer.
_SQL_DEMAND_MICRO = """
    SELECT sku_id::text, sale_date, qty
      FROM stage8.demand_history
     WHERE tenant_id = %s
       AND sale_date >= CURRENT_DATE - INTERVAL '30 days'
     ORDER BY sku_id, sale_date
"""

# Per-SKU execution log — written when isolation fallback fires.
_SQL_INSERT_SKU_EXECUTION_LOG = """
    INSERT INTO stage9.stage9_sku_execution_log
        (tenant_id, run_id, sku_id, status, fallback_model,
         error_code, error_message, sub_stage, execution_ms, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
"""


def _load_demand(tenant_id: str, db: DBConnection, *, micro: bool = False) -> dict[str, pd.DataFrame]:
    """One query → {sku_id: DataFrame(date, qty)} for all SKUs.

    micro=True uses a 30-day window (sufficient for SES level correction)
    instead of MAX_DEMAND_HISTORY_DAYS — significantly faster on large catalogs.
    """
    sql = _SQL_DEMAND_MICRO if micro else _SQL_DEMAND_HISTORY
    rows_by_sku: dict[str, list] = {}
    with db.cursor() as cur:
        cur.execute(sql, (tenant_id,))
        for sku_id, sale_date, qty in cur.fetchall():
            rows_by_sku.setdefault(sku_id, []).append(
                {"date": pd.Timestamp(sale_date), "qty": float(qty or 0.0)}
            )
    return {
        sku_id: pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        for sku_id, rows in rows_by_sku.items()
    }


# ---------------------------------------------------------------------------
# Per-SKU isolation fallback
# ---------------------------------------------------------------------------

def _per_sku_fallback(
    ctx: RunContext,
    db: DBConnection,
    sku_id: str,
    *,
    sub_stage: str,
    reason: str,
) -> None:
    """Apply Naive fallback for a single SKU and record the audit trail.

    - Append SKUResult: status='needs_acknowledgment', confidence=floor,
      fallback=True
    - Queue pattern_feedback row with fallback_used=True (proxy MAPE)
    - INSERT one row into stage9_sku_execution_log
    """
    floor = float(ctx.params.get(Param.CONFIDENCE_FLOOR))
    pctx = ctx.preloaded.pattern_ctx.get(sku_id, {}) if ctx.preloaded else {}
    pattern_label = pctx.get("pattern_label", "stable")

    ctx.sku_results.append(SKUResult(
        sku_id=sku_id,
        status=ForecastStatus.NEEDS_ACKNOWLEDGMENT,
        confidence_final=floor,
        processing_tier=(
            ctx.preloaded.sku_tiers.get(sku_id, ProcessingTier.FULL)
            if ctx.preloaded else ProcessingTier.FULL
        ),
        assigned_model=Model.NAIVE,
        used_fallback=True,
        pattern_label=pattern_label,
        backtest_mape=None,
    ))

    # direct DB write with 3-attempt retry, NEVER via BatchWriter.
    # write_pattern_feedback(fallback_used=True) forces proxy MAPE (0.50) and
    # classification_quality='proxy' so Stage 8 sees no gap for this SKU/run.
    # The human-readable `reason` is recorded in stage9_sku_execution_log
    # below — pattern_feedback carries only the learning signal.
    try:
        pf_ok = write_pattern_feedback(
            db,
            tenant_id=ctx.tenant_id,
            sku_id=sku_id,
            run_id=ctx.run_id,
            pattern_label=pattern_label,
            stage8_confidence=pctx.get("stage8_confidence"),
            mape=None,
            wape=None,
            bias=None,
            model_used=Model.NAIVE,
            model_hint=None,
            fallback_used=True,
        )
        if not pf_ok:
            ctx.pattern_feedback_failures_count += 1
    except Exception:
        log.exception("acting_handler pattern_feedback write failed sku=%s", sku_id)
        ctx.pattern_feedback_failures_count += 1

    try:
        with db.cursor() as cur:
            cur.execute(
                _SQL_INSERT_SKU_EXECUTION_LOG,
                (
                    ctx.tenant_id, ctx.run_id, sku_id,
                    "fallback", Model.NAIVE,
                    None, reason[:1000], sub_stage, None,
                ),
            )
        db.commit()
    except Exception:
        log.exception("acting_handler sku_execution_log insert failed sku=%s", sku_id)


def _make_dual_pool_log_failure(ctx: RunContext, db: DBConnection):
    """Build the (tenant_id, run_id, sku_id, reason) callback for dual_pool."""
    def _log_failure(tenant_id: str, run_id: str, sku_id: str, reason: str) -> None:
        log.warning(
            "acting_handler dual_pool sku_failure tenant=%s run=%s sku=%s reason=%s",
            tenant_id, run_id, sku_id, reason,
        )
        if "pattern_feedback" in reason:
            ctx.pattern_feedback_failures_count += 1
        try:
            _per_sku_fallback(ctx, db, sku_id, sub_stage="dual_pool", reason=reason)
        except Exception:
            log.exception("acting_handler _per_sku_fallback failed sku=%s", sku_id)
    return _log_failure


# ---------------------------------------------------------------------------
# Per-SKU pipeline contract
# ---------------------------------------------------------------------------

REQUIRED_PRELOAD_KEYS: frozenset[str] = frozenset({
    "pattern_label",
    "lifecycle_stage",
    "pattern_confidence",
    "on_watchlist",
    "weekend_zero_ratio",
    "parent_style_id",
    "shelf_life_days",
    "planned_end_date",
    "criticality_tier",
    "service_level_target",
    "oos_adjustment_factor",
    "oos_demand_added",
    "demand_series",
    "channel_adjusted",
    "promo_weights",
    "structural_break_alert",
    "thompson_state",
    "selected_quantile",
    "effective_max_horizon",
    "reorder_bias_factor",
    "learning_mode",
    "calibrated_window_days",
    "calibration_gap",
    "tier",
})

_TENANT_WIDE_KEYS: frozenset[str] = frozenset({
    "tenant_params",
    "feature_reliability",
})


# ---------------------------------------------------------------------------
# Collecting BatchWriter — worker-local row accumulation
# ---------------------------------------------------------------------------

class _CollectingBatchWriter:
    """BatchWriter look-alike that accumulates rows in memory.

    Workers call `.queue()` as they would against the real BatchWriter; nothing
    is flushed. `.drain()` at the end of `run_one_sku` packs rows into
    `DualPoolResult.batch_rows`; acting_handler queues those into the
    run-level BatchWriter for cross-SKU batching.

    pattern_feedback and cross_agent_signals are written directly by the
    sub-stages and never pass through here.
    """

    def __init__(self) -> None:
        self.buffer: defaultdict[str, list[dict]] = defaultdict(list)
        self.count: int = 0
        self.batch_size: int = 10 ** 9  # never auto-flushes
        self.conn = None

    def queue(self, table: str, row: dict) -> None:
        self.buffer[table].append(row)
        self.count += 1

    def flush_if_needed(self) -> None:
        return

    def flush(self) -> None:
        return

    def drain(self) -> dict[str, list[dict]]:
        rows = {table: list(rs) for table, rs in self.buffer.items() if rs}
        self.buffer.clear()
        self.count = 0
        return rows


# ---------------------------------------------------------------------------
# Worker helper functions
# ---------------------------------------------------------------------------

def _build_demand_df(demand_series: list[float], promo_weights: dict[str, float]) -> pd.DataFrame:
    """Reconstruct a (date, qty) DataFrame from a flat list of floats.

    Worker has no real dates — synthesizes a daily index ending today.
    promo_weights keys are date strings attached as a 'promo_weight' column.
    """
    n = len(demand_series)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "qty": demand_series})
    if promo_weights:
        df["promo_weight"] = df["date"].dt.strftime("%Y-%m-%d").map(
            lambda d: float(promo_weights.get(d, 0.0))
        )
    return df


def _resolve_thompson_best_hp(thompson_state: list[dict], default_hp: dict) -> dict:
    """Pick the best HP from Beta(α, β) posteriors. Falls back to default."""
    if not thompson_state:
        return default_hp
    try:
        best = max(
            thompson_state,
            key=lambda c: (
                float(c.get("alpha", 1.0))
                / (float(c.get("alpha", 1.0)) + float(c.get("beta", 1.0)))
            ),
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return default_hp
    cfg = best.get("hp_config") or best.get("config")
    if isinstance(cfg, dict) and cfg:
        return cfg
    stat_keys = {"alpha", "beta", "mape", "hp_config", "config"}
    hp = {k: v for k, v in best.items() if k not in stat_keys}
    return hp or default_hp


def _quantile_source(criticality_tier: str | None, on_watchlist: bool) -> str:
    """Mirror Sub-Stage 9.1's quantile-source override chain."""
    if criticality_tier == CriticalityTier.A:
        return "criticality_a"
    if on_watchlist:
        return "watchlist_override"
    return "pattern_default"


def _resolve_tenant_params(pre: dict, tenant_id: str) -> TenantParams:
    """Reconstruct TenantParams from worker globals or preloaded_data."""
    snapshot = get_worker_tenant_params() or pre.get("tenant_params")
    if snapshot is None:
        raise KeyError(
            f"tenant_params not available — neither dual_pool worker globals "
            f"nor preloaded_data carry it (tenant_id={tenant_id})"
        )
    return TenantParams.from_dict(tenant_id, snapshot)


def _resolve_invariant(pre: dict, key: str, default: Any = None) -> Any:
    """Read a tenant-wide invariant — worker globals first, then preloaded_data."""
    invariants = get_worker_tenant_invariants() or {}
    if key in invariants:
        return invariants[key]
    return pre.get(key, default)


def set_test_invariants(
    *,
    tenant_id: str | None = None,
    tenant_params: dict | None = None,
    invariants: dict | None = None,
) -> None:
    """Populate worker globals from tests / single-process callers.

    Production path goes through `dual_pool._init_worker` instead.
    """
    set_worker_globals(tenant_id, tenant_params, invariants)


def _make_fit_predict_fn(model_cls: type, best_hp: dict, features: list):
    """Build the (df, test_len) → ndarray closure used by run_substage_94."""
    def fit_predict(df: pd.DataFrame, test_len: int) -> np.ndarray:
        m = model_cls(hp=best_hp)
        m.fit(df, features)
        return m.predict(df, features, horizon=test_len)
    return fit_predict


def _make_forecast_fn(model_cls: type, best_hp: dict, features: list):
    """Build the (model_name, train_df, horizons) → ForecastBundle closure used by 9.5."""
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


def _silent_log_failure(tenant_id: str, run_id: str, sku_id: str, reason: str) -> None:
    """Per-substage failure callback inside the worker.

    Logs locally; the dual_pool collector's log_failure_fn (built by
    acting_handler) is the canonical path into stage9_sku_execution_log.
    """
    log.warning(
        "run_one_sku worker failure tenant=%s run=%s sku=%s reason=%s",
        tenant_id, run_id, sku_id, reason,
    )


# ---------------------------------------------------------------------------
# Per-SKU pipeline — referenced by dual_pool via pipeline_fn_path
# ---------------------------------------------------------------------------

def run_one_sku(
    sku_input: SkuPipelineInput,
    tenant_id: str,
    run_id: str,
    conn: Any,
) -> dict:
    """Per-SKU pipeline. Pickle-safe top-level function — do not nest or bind
    to a class. `dual_pool._subprocess_worker` re-imports by string path
    `"handlers.acting.run_one_sku"`.

    Runs 9.2 → 9.3 → 9.4 → 9.5 for one SKU and returns the contract dict
    expected by `dual_pool._collect_results`.

    Returns:
        {"sku_id": str, "status": str, "confidence_final": float, "batch_rows": dict}
    """
    sku_id = sku_input.sku_id
    assigned_model = sku_input.assigned_model
    pre = sku_input.preloaded_data

    missing = REQUIRED_PRELOAD_KEYS - pre.keys()
    if missing:
        raise KeyError(
            f"sku_input.preloaded_data missing required keys for sku={sku_id}: "
            f"{sorted(missing)}"
        )

    df = _build_demand_df(pre["demand_series"], pre["promo_weights"])
    params = _resolve_tenant_params(pre, tenant_id)
    feature_reliability = _resolve_invariant(pre, "feature_reliability", {})

    obs_days = len(pre["demand_series"])
    is_b2b = float(pre["weekend_zero_ratio"]) >= B2B_WEEKEND_ZERO_RATIO_THRESHOLD
    quantile_source = _quantile_source(pre["criticality_tier"], pre["on_watchlist"])

    lctx = LearningContext(
        sku_id=sku_id,
        tenant_id=tenant_id,
        run_id=run_id,
        pattern_label=pre["pattern_label"],
        lifecycle_stage=pre["lifecycle_stage"],
        assigned_model=assigned_model,
        selected_quantile=pre["selected_quantile"],
        quantile_source=quantile_source,
        effective_max_horizon=pre["effective_max_horizon"],
        learning_mode=pre["learning_mode"],
        oos_adjustment_factor=pre["oos_adjustment_factor"],
        is_b2b=is_b2b,
        reorder_bias_factor=pre["reorder_bias_factor"],
    )

    feature_engg_dict = {
        "thompson_state":      pre["thompson_state"],
        "feature_reliability": feature_reliability,
        "promo_decisions":     pre["promo_weights"],
        "feature_history":     {},
    }

    bw = _CollectingBatchWriter()
    model_cls = get_model_class(assigned_model)
    default_hp = model_cls(hp={}).default_hp

    # 9.2 feature engineering
    fe = run_feature_engineering(
        ctx=lctx,
        df=df,
        model=model_cls(hp=default_hp),
        preloaded=feature_engg_dict,
        params=params,
        batch_writer=bw,
    )
    train_df = fe.df_train if fe.df_train is not None else df

    # 9.3 HP tuning — FULL only; PARTIAL reuses Thompson posteriors
    if pre["tier"] == ProcessingTier.FULL:
        hp = run_hp_tuning(
            ctx=lctx,
            df_train=train_df,
            model=model_cls(hp=default_hp),
            preloaded=feature_engg_dict,
            params=params,
            batch_writer=bw,
        )
        lctx.best_hp = hp.best_hp
        lctx.validation_mape = hp.validation_mape
    else:
        lctx.best_hp = _resolve_thompson_best_hp(pre["thompson_state"], default_hp)
        lctx.validation_mape = 1.0

    features = lctx.selected_features or model_cls(hp={}).required_features
    fit_predict_fn = _make_fit_predict_fn(model_cls, lctx.best_hp, features)
    forecast_fn = _make_forecast_fn(model_cls, lctx.best_hp, features)

    # 9.4 backtest
    bt_input = SkuBacktestInput(
        sku_id=sku_id,
        assigned_model=assigned_model,
        pattern_label=pre["pattern_label"],
        model_hint=None,
        stage8_confidence=pre["pattern_confidence"],
        df=train_df,
        obs_days=obs_days,
        ultra_sparse=(obs_days < int(params.get(Param.MIN_BACKTEST_WINDOW))),
        learning_mode=lctx.learning_mode,
    )
    bt_ctx = BacktestContext()
    calibrated_cache = {
        (pre["pattern_label"], assigned_model): pre["calibrated_window_days"],
    }

    bt_results = run_substage_94(
        conn,
        tenant_id=tenant_id,
        run_id=run_id,
        skus=[bt_input],
        params=params,
        fit_predict_fn=fit_predict_fn,
        batch_writer=bw,
        pf_conn=conn,
        calibrated_cache=calibrated_cache,
        contexts={sku_id: bt_ctx},
        log_failure_fn=_silent_log_failure,
    )
    bt = bt_results.get(sku_id)

    # 9.5 forecast + confidence + status
    fc_input = SkuForecastInput(
        sku_id=sku_id,
        assigned_model=assigned_model,
        pattern_label=pre["pattern_label"],
        selected_quantile=lctx.selected_quantile,
        df=train_df,
        backtest_mape=(bt.backtest_mape if bt and np.isfinite(bt.backtest_mape) else None),
        exception_flags=(bt.exception_flags if bt else []) + fe.exception_flags,
        stage8_confidence=pre["pattern_confidence"],
        lifecycle_stage=lctx.lifecycle_stage,
        processing_tier=pre["tier"],
        is_b2b=lctx.is_b2b,
        dow_multipliers=fe.dow_multipliers,
    )
    fc_ctx = ForecastContext(
        effective_max_horizon=lctx.effective_max_horizon,
        reorder_bias_factor=lctx.reorder_bias_factor,
        oos_adjustment_factor=lctx.oos_adjustment_factor,
        on_watchlist=pre["on_watchlist"],
        training_data_truncated=(bt.training_data_truncated if bt else False),
        insufficient_post_break=getattr(bt_ctx, "insufficient_post_break", False),
    )
    calibration_gaps = {
        (pre["pattern_label"], assigned_model): (pre["calibration_gap"] or 0.0),
    }

    fc_results = run_substage_95(
        conn,
        tenant_id=tenant_id,
        run_id=run_id,
        skus=[fc_input],
        params=params,
        forecast_fn=forecast_fn,
        batch_writer=bw,
        contexts={sku_id: fc_ctx},
        signal_conn=conn,
        calibration_gaps=calibration_gaps,
        log_failure_fn=_silent_log_failure,
    )

    # 9.6 trigger — module not built yet
    if pre["parent_style_id"] is not None:
        log.debug(
            "run_one_sku 9.6 trigger sku=%s parent_style_id=%s — not yet wired",
            sku_id, pre["parent_style_id"],
        )

    batch_rows = bw.drain()
    updated_thompson_state = feature_engg_dict.get("thompson_state", {})

    fc = fc_results.get(sku_id)
    if fc is None:
        return {
            "sku_id": sku_id,
            "status": "needs_acknowledgment",
            "confidence_final": float(params.get(Param.CONFIDENCE_FLOOR)),
            "batch_rows": batch_rows,
            "backtest_mape": fc_input.backtest_mape,
            "thompson_state": updated_thompson_state,
        }

    return {
        "sku_id": sku_id,
        "status": fc.status,
        "confidence_final": float(fc.confidence_final),
        "batch_rows": batch_rows,
        "backtest_mape": fc_input.backtest_mape,
        "thompson_state": updated_thompson_state,
    }


# ---------------------------------------------------------------------------
# SkuPipelineInput assembler — called by planning_handler
# ---------------------------------------------------------------------------

def build_sku_pipeline_input(
    *,
    sku_id: str,
    lctx: LearningContext,
    df: pd.DataFrame,
    run_ctx: RunContext,
    calibrated_window_days: int,
    calibration_gap: float | None,
    tier: str,
) -> SkuPipelineInput:
    """Assemble a pickle-safe SkuPipelineInput from main-process state."""
    pattern_ctx = run_ctx.preloaded.pattern_ctx.get(sku_id, {})

    channel_record = getattr(run_ctx.preloaded, "channel_splits", {}).get(sku_id, {})
    threshold = float(run_ctx.params.get(Param.CHANNEL_SPLIT_CONFIDENCE_THRESHOLD)) if run_ctx.params else 0.50
    channel_adjusted = bool(channel_record.get("split_confidence", 0.0) >= threshold)

    oos_record = getattr(run_ctx.preloaded, "oos_ctx", {}).get(sku_id, {})

    promo_record = getattr(run_ctx.preloaded, "promo_decisions", {}).get(sku_id, {})
    if isinstance(promo_record, dict) and "weights" in promo_record:
        promo_weights = promo_record["weights"]
    elif isinstance(promo_record, dict):
        promo_weights = promo_record
    else:
        promo_weights = {}

    _portfolio_alerts = getattr(run_ctx.preloaded, "portfolio_alerts", [])
    if isinstance(_portfolio_alerts, dict):
        structural_break_alert = bool(_portfolio_alerts.get(sku_id, False))
    else:
        structural_break_alert = any(
            a.get("alert_type") == "structural_break" for a in _portfolio_alerts
        )

    thompson_record = run_ctx.preloaded.thompson_state.get(
        (sku_id, lctx.assigned_model), {}
    )
    thompson_state = {
        (sku_id, lctx.assigned_model): dict(thompson_record)
        if isinstance(thompson_record, dict) else {}
    }

    qty_col = "qty" if "qty" in df.columns else df.columns[-1]
    demand_series = [float(v) for v in df[qty_col].tolist()]

    return SkuPipelineInput(
        sku_id=sku_id,
        assigned_model=lctx.assigned_model,
        sku_data={},
        preloaded_data={
            "pattern_label":          lctx.pattern_label,
            "lifecycle_stage":        lctx.lifecycle_stage,
            "pattern_confidence":     pattern_ctx.get("pattern_confidence", 1.0),
            "on_watchlist":           pattern_ctx.get("on_watchlist", False),
            "weekend_zero_ratio":     pattern_ctx.get("weekend_zero_ratio", 0.0),
            "parent_style_id":        pattern_ctx.get("parent_style_id"),
            "shelf_life_days":        pattern_ctx.get("shelf_life_days"),
            "planned_end_date":       pattern_ctx.get("planned_end_date"),
            "criticality_tier":       pattern_ctx.get("criticality_tier"),
            "service_level_target":   pattern_ctx.get("service_level_target"),
            "oos_adjustment_factor":  lctx.oos_adjustment_factor,
            "oos_demand_added":       float(oos_record.get("oos_demand_added", 0.0)),
            "demand_series":          demand_series,
            "channel_adjusted":       channel_adjusted,
            "promo_weights":          promo_weights,
            "structural_break_alert": structural_break_alert,
            "thompson_state":         thompson_state,
            "selected_quantile":      lctx.selected_quantile,
            "effective_max_horizon":  lctx.effective_max_horizon,
            "reorder_bias_factor":    lctx.reorder_bias_factor,
            "learning_mode":          lctx.learning_mode,
            "calibrated_window_days": calibrated_window_days,
            "calibration_gap":        calibration_gap,
            "tier":                   tier,
        },
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def acting_handler(*, tenant_id: str, run_id: str, db: DBConnection) -> None:
    log.info("acting_handler starting tenant=%s run=%s", tenant_id, run_id)
    ctx = fetch(run_id)

    # ------------------------------------------------------------------
    # Track 1: MICRO_UPDATE branch — Sync Now within 18 hours of last full.
    # SES level correction only — no model retraining, no 9.4/9.5.
    # Must complete in < 15 seconds for any catalog size.
    # Loads its own 30-day demand window (smaller than the 400-day full
    # window cached on ctx by planning_handler).
    # ------------------------------------------------------------------
    if ctx.execution_mode == ExecutionMode.MICRO_UPDATE:
        demand_data = _load_demand(tenant_id, db, micro=True)
        micro_results: list[SKUResult] = []
        for sku_id in ctx.sku_ids:
            df = demand_data.get(sku_id)
            if df is None or df.empty:
                continue
            try:
                result = route_sku_micro_update(sku_id, df, ctx, db)
                if result is not None:
                    micro_results.append(result.sku_result)
                # SKUs with no prior forecast are skipped — picked up next full.
            except Exception:
                log.exception(
                    "acting_handler micro_update failed sku=%s — skipping", sku_id,
                )

        ctx.batch_writer.flush_if_needed()
        ctx.sku_results = micro_results
        log.info(
            "acting_handler micro_update complete tenant=%s run=%s skus_updated=%d",
            tenant_id, run_id, len(micro_results),
        )
        return

    # ------------------------------------------------------------------
    # FULL pipeline below (FULL + PARTIAL tiers via dual_pool, CACHE in main).
    # `pipeline_inputs` and `cache_sku_ids` were populated by planning_handler.
    # ------------------------------------------------------------------

    # ----- Track 2: CACHE tier — main process ---------------------------
    if ctx.cache_sku_ids:
        # Reuse the demand_data already loaded by planning_handler instead
        # of paying for a second 400-day × N-SKU bulk SELECT (cuts peak
        # memory ~50% and saves ~one bulk SELECT round-trip per run).
        cache_demand = ctx.demand_data or _load_demand(tenant_id, db, micro=False)

        class _CacheCtx:
            tenant_id = ctx.tenant_id
            run_id = ctx.run_id
            preloaded = ctx.preloaded
            batch_writer = ctx.batch_writer

        for sku_id in ctx.cache_sku_ids:
            df = cache_demand.get(sku_id)
            if df is None or df.empty:
                continue
            try:
                result = route_cache(sku_id, df, _CacheCtx(), db)
                if result is not None and result.sku_result is not None:
                    ctx.sku_results.append(result.sku_result)
            except Exception as exc:
                log.exception("acting_handler cache sku=%s failed", sku_id)
                _per_sku_fallback(ctx, db, sku_id, sub_stage="cache", reason=f"cache:{exc}")
        ctx.batch_writer.flush_if_needed()

    # ----- Track 3: FULL + PARTIAL tier — dual_pool ---------------------
    if ctx.pipeline_inputs:
        floor = float(ctx.params.get(Param.CONFIDENCE_FLOOR))
        log_failure_fn = _make_dual_pool_log_failure(ctx, db)

        # Tenant-wide invariants stashed once per worker via _init_worker
        # initargs so they aren't re-pickled per SKU.
        tenant_params_dict = ctx.params.to_dict() if ctx.params else {}
        tenant_invariants = {
            "feature_reliability": ctx.preloaded.feature_reliability,
        }

        stats = run_dual_pool(
            sku_inputs=ctx.pipeline_inputs,
            tenant_id=tenant_id,
            run_id=run_id,
            pipeline_fn_path="handlers.acting.run_one_sku",
            db_config=db_config_from_env(),
            log_failure_fn=log_failure_fn,
            cleanup_conn=db,
            fallback_confidence=floor,
            tenant_params=tenant_params_dict,
            tenant_invariants=tenant_invariants,
        )

        # Drain the per-worker collecting batch_rows into the run-level
        # BatchWriter so cross-SKU batching can amortize INSERT overhead.
        # Pre-compute lookups once, outside the loop, so each iteration is
        # just dict accesses (saves ~10K dict.get().get() chained lookups
        # per run).
        sku_id_to_input = {s.sku_id: s for s in ctx.pipeline_inputs}
        pattern_by_sku = {
            sku_id: ctx.preloaded.pattern_ctx.get(sku_id, {}).get("pattern_label", "stable")
            for sku_id in stats.results
        }
        for sku_id, dp_result in stats.results.items():
            if dp_result.batch_rows:
                for table, rows in dp_result.batch_rows.items():
                    for row in rows:
                        ctx.batch_writer.queue(table, row)

            if dp_result.thompson_state:
                for key, configs in dp_result.thompson_state.items():
                    ctx.preloaded.thompson_state.setdefault(key, {}).update(configs)

            sku_input = sku_id_to_input.get(sku_id)
            tier = ctx.preloaded.sku_tiers.get(sku_id, ProcessingTier.FULL)
            ctx.sku_results.append(SKUResult(
                sku_id=sku_id,
                status=dp_result.status,
                confidence_final=float(dp_result.confidence_final),
                processing_tier=tier,
                assigned_model=(
                    sku_input.assigned_model if sku_input else Model.SES
                ),
                used_fallback=(dp_result.pool == "fallback"),
                pattern_label=pattern_by_sku.get(sku_id, "stable"),
                backtest_mape=dp_result.backtest_mape,
            ))
        # Single flush at end of dual_pool result loop — drops O(N/100)
        # per-SKU flush_if_needed checks. Final flush below catches the rest.
        ctx.batch_writer.flush_if_needed()

        log.info(
            "acting_handler dual_pool tenant=%s process=%d thread=%d "
            "timeouts=%d failures=%d",
            tenant_id,
            stats.process_skus, stats.thread_skus,
            stats.timeouts, stats.failures,
        )

    ctx.batch_writer.flush_if_needed()

    log.info(
        "acting_handler complete tenant=%s run=%s sku_results=%d pf_failures=%d",
        tenant_id, run_id, len(ctx.sku_results), ctx.pattern_feedback_failures_count,
    )
