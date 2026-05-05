"""
Sub-Stage 9.4 — Backtesting and pattern_feedback.

Five ordered steps:

  1. select_backtest_window  — pick the window length, preferring the
     adaptive calibrated value for (tenant, pattern, model), with overrides
     for ultra-sparse / short-history / exploit-mode tenants.
  2. run_backtest            — split train/test, fit caller-supplied model,
     predict, compute MAPE (with zero-day masking), WAPE, and bias.
  3. detect_exceptions       — stockout / promo_spike / unusual_drop /
     high_volatility plus high_mape when backtest_mape > 0.50.
  4. detect_structural_break — ruptures.Pelt(rbf) when portfolio alerts fire.
     Sets ctx.training_data_truncated when post-break length >= 30, else
     ctx.insufficient_post_break (for a steeper confidence penalty in 9.5).
  5. write_pattern_feedback  — Principle 4 sacred write. Direct
     conn.execute + conn.commit (NOT batched), retried 3x on failure,
     proxy MAPE 0.50 for failed SKUs. Must complete before Sub-Stage 9.5.

Orchestrator run_substage_94 glues the five steps together with per-SKU
failure isolation (Principle 3) and writes backtest_decisions through the
shared BatchWriter.

Model interface — the caller supplies:
    fit_predict_fn(train_df: pandas.DataFrame, test_len: int) -> np.ndarray
so the sub-stage stays model-agnostic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from infrastructure.constants import (
    HIGH_MAPE_FLAG_THRESHOLD,
    HIGH_VOLATILITY_CV,
    MIN_POST_BREAK_LEN,
    PATTERN_FEEDBACK_HORIZON_DAYS,
    PATTERN_FEEDBACK_PROXY_MAPE,
    PROMO_SPIKE_RATIO,
    PROMO_SPIKE_Z,
    PROPHET_FAMILY,
    QUALITY_ACCEPTABLE_MAX,
    QUALITY_GOOD_MAX,
    ROLLING_BASELINE_DAYS,
    STOCKOUT_MIN_ZERO_STREAK,
    UNUSUAL_DROP_KEEP_RATIO,
    UNUSUAL_DROP_MIN_STREAK,
    Param,
)
from infrastructure.tenant_params import TenantParams

# Module-level psycopg2 import so _jsonb doesn't re-import per call.
try:
    from psycopg2.extras import Json as _PsycopgJson  # type: ignore
except ImportError:  # pragma: no cover - psycopg2 missing in unit-test env
    _PsycopgJson = None

logger = logging.getLogger(__name__)

# Module-level ruptures check. ruptures is REQUIRED for structural-break
# detection. We don't raise on import failure (Py 3.14 has no wheel yet,
# and unit tests on that interpreter need to load the module), but we DO
# emit logger.critical at module load so production deploys without
# ruptures fail noisily.
try:
    import ruptures as _rpt  # type: ignore

    _RUPTURES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on interpreters w/o wheel
    _rpt = None
    _RUPTURES_AVAILABLE = False
    logger.critical(
        "ruptures library is not installed — structural break detection "
        "will return NO-BREAK for every SKU. Add `ruptures` to "
        "requirements.txt; structural_break_confidence_penalty will not fire."
    )

__all__ = [
    "BacktestContext",
    "BacktestMetrics",
    "BacktestResult",
    "SkuBacktestInput",
    "CalibratedWindowCache",
    "MIN_BACKTEST_OBS_DAYS",
    "prefetch_calibrated_windows",
    "select_backtest_window",
    "run_backtest",
    "detect_exceptions",
    "detect_structural_break",
    "write_pattern_feedback",
    "run_substage_94",
    "run_substage_94_parallel",
]

# thresholds live in stage9.constants — see imports above.
# The two values that remain in this file are implementation details, not
# behavior gates:
PATTERN_FEEDBACK_MAX_RETRIES = 3
PATTERN_FEEDBACK_RETRY_DELAY_S = 0.1  # "100ms per spec"

# Minimum obs_days to run a backtest. Below this threshold we have insufficient
# data for a meaningful train/test split, so backtest is skipped and
# forecasts.backtest_mape is written as NULL.
MIN_BACKTEST_OBS_DAYS = 28

# Default PELT penalty derived from structural_break_sensitivity = 0.30 (seeded default).
# Formula: penalty = max(1, round(3.0 / sensitivity)) so that the seeded value of
# 0.30 → penalty 10 (same as the previous hardcoded value), higher sensitivity → lower
# penalty (more breaks detected), lower sensitivity → higher penalty (fewer breaks).
# _try_backtest computes this per-run from params; this constant is the test-time default.
PELT_PENALTY = 10


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestContext:
    """
    Shared per-SKU context that survives into Sub-Stage 9.5.

    Sub-Stage 9.5 (confidence engine) inspects these flags to decide whether
    to apply the structural_break_confidence_penalty (truncated case) or a
    steeper penalty (insufficient post-break history).
    """
    training_data_truncated: bool = False
    insufficient_post_break: bool = False
    break_index: Optional[int] = None


@dataclass
class BacktestMetrics:
    window_days: int
    mape: float
    wape: float
    bias: float
    actual: np.ndarray = field(repr=False)
    yhat: np.ndarray = field(repr=False)


@dataclass
class BacktestResult:
    sku_id: str
    window_days: int
    backtest_mape: float
    backtest_wape: float
    backtest_bias: float
    exception_flags: list[str]
    structural_break_detected: bool
    break_index: Optional[int]
    training_data_truncated: bool
    fallback_used: bool


@dataclass
class SkuBacktestInput:
    """
    Per-SKU payload handed to run_substage_94.

    Keeping this a dataclass (not 12 positional args) lets the orchestrator
    skip and move on when a single SKU is malformed, without tangling the
    call site.
    """
    sku_id: str
    assigned_model: str
    pattern_label: str
    model_hint: Optional[str]
    stage8_confidence: Optional[float]
    df: pd.DataFrame
    obs_days: int
    ultra_sparse: bool
    learning_mode: str
    portfolio_alerts_list: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Backtest window selection
# ---------------------------------------------------------------------------

_SELECT_CALIBRATED_WINDOW = (
    "SELECT backtest_window_days FROM stage9.adaptive_quantile_state "
    "WHERE tenant_id = %s AND pattern_label = %s AND assigned_model = %s "
    "ORDER BY horizon_days DESC LIMIT 1"
)

_SELECT_TENANT_CALIBRATED_WINDOWS = (
    "SELECT pattern_label, assigned_model, backtest_window_days, horizon_days "
    "FROM stage9.adaptive_quantile_state "
    "WHERE tenant_id = %s AND backtest_window_days IS NOT NULL"
)

CalibratedWindowCache = dict[tuple[str, str], int]


def prefetch_calibrated_windows(conn, tenant_id: str) -> CalibratedWindowCache:
    """
    Fetch every calibrated (pattern, model) window for a tenant in ONE query.

    Returns a dict keyed by (pattern_label, assigned_model). When the table
    has multiple rows per key (one per horizon), the row with the largest
    horizon_days wins — same precedence the per-row lookup used.

    Pass the result as `calibrated_cache` into select_backtest_window to
    avoid the per-SKU N+1 round-trip.
    """
    cache: CalibratedWindowCache = {}
    best_horizon: dict[tuple[str, str], int] = {}
    with conn.cursor() as cur:
        cur.execute(_SELECT_TENANT_CALIBRATED_WINDOWS, (tenant_id,))
        for pattern, model, window_days, horizon in cur.fetchall():
            if window_days is None:
                continue
            key = (pattern, model)
            if horizon > best_horizon.get(key, -1):
                best_horizon[key] = horizon
                cache[key] = int(window_days)
    return cache


def select_backtest_window(
        conn,
        tenant_id: str,
        pattern_label: str,
        assigned_model: str,
        params: TenantParams,
        *,
        obs_days: int,
        ultra_sparse: bool,
        learning_mode: str,
        calibrated_cache: Optional[CalibratedWindowCache] = None,
) -> int:
    """
    Pick the backtest window (Step 1 of Sub-Stage 9.4).

    Precedence:
      1. Calibrated value from adaptive_quantile_state if present.
      2. Otherwise TenantParams.default_backtest_window (60 at seed time).

    Then apply safety overrides in this order (later overrides dominate):
      * ultra_sparse                            → min_backtest_window
      * obs_days < 60                           → max(min_w, obs_days // 3)
      * obs_days >= 180 AND learning_mode='exploit' → max_backtest_window

    The result is finally clamped to [min_w, max_w] and further reduced if
    it would leave no data for training (window >= obs_days).

    `calibrated_cache` (optional) is the dict returned by
    prefetch_calibrated_windows — when provided, no DB round-trip happens
    here. This is the production path; the per-row lookup remains for
    standalone single-SKU use.
    """
    default_w = int(params.get("default_backtest_window"))
    min_w = int(params.get("min_backtest_window"))
    max_w = int(params.get("max_backtest_window"))
    short_threshold = int(params.get("backtest_short_obs_threshold"))
    exploit_threshold = int(params.get("backtest_exploit_obs_threshold"))

    if calibrated_cache is not None:
        calibrated = calibrated_cache.get((pattern_label, assigned_model))
    else:
        calibrated = _lookup_calibrated_window(
            conn, tenant_id, pattern_label, assigned_model
        )
    window = calibrated if calibrated is not None else default_w

    if ultra_sparse:
        window = min_w
    elif obs_days < short_threshold:
        window = max(min_w, obs_days // 3)
    elif obs_days >= exploit_threshold and learning_mode == "exploit":
        window = max_w

    window = max(min_w, min(window, max_w))
    if window >= obs_days:
        window = max(1, obs_days // 3)
    return window


def _lookup_calibrated_window(
        conn: Any, tenant_id: str, pattern_label: str, assigned_model: str,
) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute(
            _SELECT_CALIBRATED_WINDOW,
            (tenant_id, pattern_label, assigned_model),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


# ---------------------------------------------------------------------------
# Step 2 — Run backtest
# ---------------------------------------------------------------------------

FitPredictFn = Callable[[pd.DataFrame, int], np.ndarray]


def run_backtest(
        df: pd.DataFrame,
        window: int,
        fit_predict_fn: FitPredictFn,
) -> BacktestMetrics:
    """
    Split, fit, predict, score (Step 2).

    df must have a 'qty' column of actuals. fit_predict_fn receives the train
    frame and the test length, and returns an array of length `window`.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if window >= len(df):
        raise ValueError(f"window {window} >= df length {len(df)}")

    train = df.iloc[:-window]
    test = df.iloc[-window:]

    # check ensures the model didn't return too much or too little data.
    # If you asked for a 30-day forecast, you must get exactly 30 numbers back.
    yhat = np.asarray(fit_predict_fn(train, window), dtype=float)
    if yhat.shape[0] != window:
        raise ValueError(
            f"fit_predict_fn returned {yhat.shape[0]} values, expected {window}"
        )
    qty_col = "qty" if "qty" in test.columns else "y"
    actual = test[qty_col].to_numpy(dtype=float)

    return BacktestMetrics(
        window_days=window,
        mape=_compute_mape(actual, yhat),
        wape=_compute_wape(actual, yhat),
        bias=_compute_bias(actual, yhat),
        actual=actual,
        yhat=yhat,
    )


def _compute_mape(actual: np.ndarray, yhat: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - yhat[mask]) / actual[mask])))


def _compute_wape(actual: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(np.abs(actual).sum())
    if denom == 0.0:
        return float("nan")
    return float(np.abs(actual - yhat).sum() / denom)


def _compute_bias(actual: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(actual.sum())
    if denom == 0.0:
        return float("nan")
    return float((yhat - actual).sum() / denom)


# ---------------------------------------------------------------------------
# Step 3 — Exception detection
# ---------------------------------------------------------------------------

def detect_exceptions(actual: np.ndarray, backtest_mape: float) -> list[str]:
    """
    Return the exception flags for the backtest test window.

    Order is deterministic (stockout, promo_spike, unusual_drop,
    high_volatility, high_mape) so logs diff cleanly — the 9.5 confidence
    engine reads the set, not the order.
    """
    arr = np.asarray(actual, dtype=float)
    if arr.size == 0:
        return []
    rolling = _rolling_baseline(arr)

    flags: list[str] = []
    if _has_stockout(arr):
        flags.append("stockout")
    if _has_promo_spike(arr, rolling):
        flags.append("promo_spike")
    if _has_unusual_drop(arr, rolling):
        flags.append("unusual_drop")
    if _has_high_volatility(arr):
        flags.append("high_volatility")
    if _is_high_mape(backtest_mape):
        flags.append("high_mape")
    return flags


def _rolling_baseline(arr: np.ndarray) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(window=ROLLING_BASELINE_DAYS, min_periods=1)
        .mean()
        .to_numpy()
    )


def _has_stockout(arr: np.ndarray) -> bool:
    return _longest_run(arr == 0) >= STOCKOUT_MIN_ZERO_STREAK


def _has_promo_spike(arr: np.ndarray, rolling: np.ndarray) -> bool:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rolling > 0, arr / rolling, 0.0)
    if np.any(ratio > PROMO_SPIKE_RATIO):
        return True

    sigma = float(arr.std())
    if sigma == 0.0:
        return False
    z = (arr - float(arr.mean())) / sigma
    return bool(np.any(z > PROMO_SPIKE_Z))


def _has_unusual_drop(arr: np.ndarray, rolling: np.ndarray) -> bool:
    below = (rolling > 0) & (arr < UNUSUAL_DROP_KEEP_RATIO * rolling)
    return _longest_run(below) >= UNUSUAL_DROP_MIN_STREAK


def _has_high_volatility(arr: np.ndarray) -> bool:
    mu = float(arr.mean())
    if mu == 0.0:
        return False
    cv = float(arr.std()) / mu
    return cv >= HIGH_VOLATILITY_CV


def _is_high_mape(mape: Optional[float]) -> bool:
    if mape is None or np.isnan(mape):
        return False
    return mape > HIGH_MAPE_FLAG_THRESHOLD


def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run in a 1-D boolean array."""
    longest = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


# ---------------------------------------------------------------------------
# Step 4 — Structural break detection
# ---------------------------------------------------------------------------

def detect_structural_break(
        series: np.ndarray,
        portfolio_alerts_list: Iterable[Any],
        ctx: BacktestContext,
        *,
        penalty: int = PELT_PENALTY,
) -> tuple[bool, Optional[int]]:
    """
    Step 4. PELT only fires when portfolio alerts are present — the spec
    uses portfolio alerts as the trigger so we don't burn CPU on stable
    SKUs.

    On a detected break, mutates ctx:
      * post-break length >= MIN_POST_BREAK_LEN → training_data_truncated
        (9.5 applies structural_break_confidence_penalty)
      * else                                    → insufficient_post_break
        (9.5 applies a steeper penalty)
    """
    if not portfolio_alerts_list:
        return False, None

    arr = np.asarray(series, dtype=float)
    if arr.size < MIN_POST_BREAK_LEN:
        return False, None

    if not _RUPTURES_AVAILABLE:
        # Module-load already emitted logger.critical. Per-SKU return
        # is silent so we don't flood logs with one-line-per-SKU.
        return False, None

    algo = _rpt.Pelt(model="rbf").fit(arr)
    bkps = algo.predict(pen=penalty)
    # ruptures always appends len(arr) as the final breakpoint, so a
    # result of [len(arr)] (or empty) means no break was found.
    interior = [b for b in bkps if b < arr.size]
    if not interior:
        return False, None

    # Pick the EARLIEST interior breakpoint. This maximises post-break
    # window length, which determines the truncated-vs-insufficient
    # branch below — earlier break = more likely to satisfy the 30-day
    # post_len threshold.
    break_idx = int(interior[0])
    post_len = arr.size - break_idx
    ctx.break_index = break_idx
    if post_len >= MIN_POST_BREAK_LEN:
        ctx.training_data_truncated = True
    else:
        ctx.insufficient_post_break = True
    return True, break_idx


# ---------------------------------------------------------------------------
# Step 5 — pattern_feedback writer (Principle 4 — sacred write)
# ---------------------------------------------------------------------------

_INSERT_PATTERN_FEEDBACK = """
INSERT INTO stage8.pattern_feedback (
    tenant_id, sku_id, run_id,
    pattern_label, stage8_confidence,
    forecast_error_mape, forecast_error_wape, bias,
    model_used, horizon_days, hint_matched,
    classification_quality, fallback_used
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (tenant_id, sku_id, run_id) DO UPDATE SET
    pattern_label         = EXCLUDED.pattern_label,
    stage8_confidence     = EXCLUDED.stage8_confidence,
    forecast_error_mape   = EXCLUDED.forecast_error_mape,
    forecast_error_wape   = EXCLUDED.forecast_error_wape,
    bias                  = EXCLUDED.bias,
    model_used            = EXCLUDED.model_used,
    horizon_days          = EXCLUDED.horizon_days,
    hint_matched          = EXCLUDED.hint_matched,
    classification_quality = EXCLUDED.classification_quality,
    fallback_used         = EXCLUDED.fallback_used
"""


def write_pattern_feedback(
        conn: Any,
        *,
        tenant_id: str,
        sku_id: str,
        run_id: str,
        pattern_label: str,
        stage8_confidence: Optional[float],
        mape: Optional[float],
        wape: Optional[float],
        bias: Optional[float],
        model_used: str,
        model_hint: Optional[str],
        fallback_used: bool = False,
        horizon_days: int = PATTERN_FEEDBACK_HORIZON_DAYS,
        max_retries: int = PATTERN_FEEDBACK_MAX_RETRIES,
        retry_delay_seconds: float = PATTERN_FEEDBACK_RETRY_DELAY_S,
) -> bool:
    """
    Direct-write pattern_feedback row (Step 5, Principle 4).

    DB semantics:
      * direct conn.execute + conn.commit — NEVER batched
      * retry up to max_retries on any DB exception, with retry_delay_seconds
        between attempts (100ms per spec)
      * on failure of all retries, log and return False — the run continues,
        the failure count is later recorded in stage9_self_assessment

    Connection semantics:
      * `conn` should be a DEDICATED connection in production. conn.commit()
        commits the entire transaction on this connection — sharing the
        conn with other writes risks committing them mid-flight or
        rolling them back on a pattern_feedback failure. The orchestrator
        passes a `pf_conn` that is only used here.

    Value semantics:
      * hint_matched is True when model_hint and model_used are the same
        model (case-insensitive)
      * fallback_used=True forces mape=PATTERN_FEEDBACK_PROXY_MAPE (0.50)
        and classification_quality='proxy' — the row still exists so Stage
        8 doesn't see a gap for this SKU/run
      * If mape is NaN, fallback_used is forced to True (NaN is not a
        valid measurement — pollutes Stage 8's learning signal otherwise).
    """
    # NaN/None MAPE is not a real measurement — treat as fallback so
    # quality becomes 'proxy' and Stage 8 can filter the row out. We
    # only force the upgrade when the caller didn't already pass it.
    if not fallback_used and mape is not None and _is_missing_metric(mape):
        fallback_used = True
    hint_matched = _compute_hint_matched(model_hint, model_used)
    if fallback_used:
        mape_to_write: float = PATTERN_FEEDBACK_PROXY_MAPE
        quality = "proxy"
    else:
        mape_to_write = _coerce_metric(mape, default=PATTERN_FEEDBACK_PROXY_MAPE)
        quality = _classify_quality(mape_to_write)

    wape_to_write = _coerce_metric(wape, default=None)
    bias_to_write = _coerce_metric(bias, default=None)

    args = (
        tenant_id, sku_id, run_id,
        pattern_label, stage8_confidence,
        mape_to_write, wape_to_write, bias_to_write,
        model_used, horizon_days, hint_matched,
        quality, fallback_used,
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with conn.cursor() as cur:
                cur.execute(_INSERT_PATTERN_FEEDBACK, args)
            conn.commit()
            return True
        except Exception as e:
            last_err = e
            logger.warning(
                "pattern_feedback write attempt %d/%d failed sku_id=%s err=%s",
                attempt, max_retries, sku_id, e,
            )
            try:
                conn.rollback()
            except Exception:
                logger.debug(
                    "rollback failed during pattern_feedback retry sku_id=%s",
                    sku_id, exc_info=True,
                )
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)

    logger.error(
        "pattern_feedback write FAILED after %d attempts sku_id=%s err=%s",
        max_retries, sku_id, last_err,
    )
    return False


def _compute_hint_matched(
        model_hint: Optional[str], model_used: str
) -> bool:
    if not model_hint:
        return False
    return model_hint.lower() == model_used.lower()


def _classify_quality(mape: Optional[float]) -> str:
    if _is_missing_metric(mape):
        return "poor"
    if mape < QUALITY_GOOD_MAX:
        return "good"
    if mape < QUALITY_ACCEPTABLE_MAX:
        return "acceptable"
    return "poor"


def _is_missing_metric(value: Any) -> bool:
    """A metric is 'missing' when None or NaN — both should trigger the
    same fallback path. Single source of truth for this check."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _coerce_metric(value: Any, *, default: Optional[float]) -> Optional[float]:
    if _is_missing_metric(value):
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(fv):
        return default
    return fv


# ---------------------------------------------------------------------------
# Entry Point — per-SKU run with failure isolation
# ---------------------------------------------------------------------------

def run_substage_94(
        conn: Any,
        *,
        tenant_id: str,
        run_id: str,
        skus: Iterable[SkuBacktestInput],
        params: TenantParams,
        fit_predict_fn: FitPredictFn,
        batch_writer: Any,
        pf_conn: Any = None,
        calibrated_cache: Optional[CalibratedWindowCache] = None,
        contexts: Optional[dict[str, BacktestContext]] = None,
        log_failure_fn: Optional[Callable[[str, str, str, str], None]] = None,
) -> dict[str, BacktestResult]:
    """
    Run Sub-Stage 9.4 for every SKU in `skus` (single-process).

    Per SKU:
      1. Select window (uses `calibrated_cache` if provided — no DB call)
      2. Run backtest via fit_predict_fn
      3. Detect exceptions
      4. Detect structural break (if portfolio alerts fired)
      5. Write pattern_feedback DIRECTLY on `pf_conn` — sacred write,
         even on failure (fallback row with proxy MAPE)
      6. Queue backtest_decisions via BatchWriter (on `conn`)

    Per-SKU isolation (Principle 3): the per-SKU body is wrapped in
    try/except. Anything that escapes — including BatchWriter flushes
    and pattern_feedback retry exhaustion — becomes a `log_failure_fn`
    call and the loop continues to the next SKU.

    `pf_conn` defaults to `conn` for tests; production callers MUST pass
    a dedicated psycopg2 connection so pattern_feedback's commit doesn't
    touch other in-flight transactions on `conn`.

    `log_failure_fn(tenant_id, run_id, sku_id, reason)` records each
    failure to stage9_sku_execution_log. Defaults to a logger.warning
    so unit tests don't need a DB.

    `calibrated_cache` is the dict returned by prefetch_calibrated_windows.
    If None, each SKU issues its own SELECT (the legacy/test path).

    For 5M-SKU runs use run_substage_94_parallel — this function is
    sequential and meant for small batches and tests.
    """
    if pf_conn is None:
        pf_conn = conn  # back-compat for callers that don't isolate yet

    log_failure = log_failure_fn or _default_substage_94_log_failure

    results: dict[str, BacktestResult] = {}
    if contexts is None:
        contexts = {}

    for payload in skus:
        ctx = contexts.setdefault(payload.sku_id, BacktestContext())
        try:
            results[payload.sku_id] = _run_one_sku(
                conn,
                pf_conn=pf_conn,
                tenant_id=tenant_id,
                run_id=run_id,
                payload=payload,
                params=params,
                fit_predict_fn=fit_predict_fn,
                batch_writer=batch_writer,
                calibrated_cache=calibrated_cache,
                ctx=ctx,
                log_failure=log_failure,
            )
        except MemoryError:
            raise
        except Exception as e:
            # Outer-perimeter isolation: a BatchWriter flush failure or
            # any other unforeseen exception lands here. Per Principle 3
            # we never crash the run on one SKU.
            logger.exception(
                "run_substage_94 outer failure sku_id=%s — continuing",
                payload.sku_id,
            )
            log_failure(tenant_id, run_id, payload.sku_id,
                        f"backtest_outer_failure:{e}")
            results[payload.sku_id] = _build_fallback_result(payload, ctx)

    return results


def _default_substage_94_log_failure(
        tenant_id: str, run_id: str, sku_id: str, reason: str,
) -> None:
    """Stand-in for the stage9_sku_execution_log writer."""
    logger.warning(
        "9.4 fallback tenant=%s run=%s sku=%s reason=%s",
        tenant_id, run_id, sku_id, reason,
    )


def _build_fallback_result(
        payload: SkuBacktestInput, ctx: BacktestContext,
) -> BacktestResult:
    return BacktestResult(
        sku_id=payload.sku_id,
        window_days=0,
        backtest_mape=float("nan"),
        backtest_wape=float("nan"),
        backtest_bias=float("nan"),
        exception_flags=[],
        structural_break_detected=False,
        break_index=ctx.break_index,
        training_data_truncated=ctx.training_data_truncated,
        fallback_used=True,
    )


def _run_one_sku(
        conn: Any,
        *,
        pf_conn: Any,
        tenant_id: str,
        run_id: str,
        payload: SkuBacktestInput,
        params: TenantParams,
        fit_predict_fn: FitPredictFn,
        batch_writer: Any,
        calibrated_cache: Optional[CalibratedWindowCache],
        ctx: BacktestContext,
        log_failure: Callable[[str, str, str, str], None],
) -> BacktestResult:
    """
    Steps 1–4 inside one try block (per-SKU isolation, Principle 3),
    then unconditionally write pattern_feedback on its dedicated
    connection (Principle 4) and queue the backtest_decisions row.

    Pattern_feedback's `False` return (3 retries exhausted) triggers a
    `log_failure` call so the audit trail in stage9_sku_execution_log
    records the gap Stage 8 will see.
    """
    window, metrics, flags, break_detected, fallback = _try_backtest(
        conn, tenant_id, payload, params, fit_predict_fn,
        calibrated_cache, ctx,
    )
    if fallback:
        log_failure(tenant_id, run_id, payload.sku_id, "backtest_fit_failed")

    pf_ok = write_pattern_feedback(
        pf_conn,
        tenant_id=tenant_id,
        sku_id=payload.sku_id,
        run_id=run_id,
        pattern_label=payload.pattern_label,
        stage8_confidence=payload.stage8_confidence,
        mape=metrics.mape if metrics else None,
        wape=metrics.wape if metrics else None,
        bias=metrics.bias if metrics else None,
        model_used=payload.assigned_model,
        model_hint=payload.model_hint,
        fallback_used=fallback,
    )
    if not pf_ok:
        # Sacred-write retries exhausted. Record the audit gap explicitly
        # so Stage 8 has a tracking signal even though the row is missing.
        log_failure(
            tenant_id, run_id, payload.sku_id,
            "pattern_feedback_write_exhausted",
        )

    batch_writer.queue(
        "backtest_decisions",
        _backtest_decisions_row(
            tenant_id, run_id, payload.sku_id,
            window, metrics, flags, break_detected, ctx,
        ),
    )
    batch_writer.flush_if_needed()

    return BacktestResult(
        sku_id=payload.sku_id,
        window_days=window,
        backtest_mape=metrics.mape if metrics else float("nan"),
        backtest_wape=metrics.wape if metrics else float("nan"),
        backtest_bias=metrics.bias if metrics else float("nan"),
        exception_flags=flags,
        structural_break_detected=break_detected,
        break_index=ctx.break_index,
        training_data_truncated=ctx.training_data_truncated,
        fallback_used=fallback,
    )


def _try_backtest(
        conn: Any,
        tenant_id: str,
        payload: SkuBacktestInput,
        params: TenantParams,
        fit_predict_fn: FitPredictFn,
        calibrated_cache: Optional[CalibratedWindowCache],
        ctx: BacktestContext,
) -> tuple[int, Optional[BacktestMetrics], list[str], bool, bool]:
    """
    Run Steps 1–4 for one SKU. Any failure becomes a fallback (proxy
    pattern_feedback row written by the caller) — never raises.
    """
    if payload.obs_days < MIN_BACKTEST_OBS_DAYS:
        return 0, None, [], False, False

    try:
        window = select_backtest_window(
            conn,
            tenant_id,
            payload.pattern_label,
            payload.assigned_model,
            params,
            obs_days=payload.obs_days,
            ultra_sparse=payload.ultra_sparse,
            learning_mode=payload.learning_mode,
            calibrated_cache=calibrated_cache,
        )
        metrics = run_backtest(payload.df, window, fit_predict_fn)
        flags = detect_exceptions(metrics.actual, metrics.mape)
        sensitivity = float(params.get(Param.STRUCTURAL_BREAK_SENSITIVITY))
        pelt_pen = max(1, round(3.0 / max(sensitivity, 0.01)))
        qty_col = "qty" if "qty" in payload.df.columns else "y"
        break_detected, _ = detect_structural_break(
            payload.df[qty_col].to_numpy(dtype=float),
            payload.portfolio_alerts_list,
            ctx,
            penalty=pelt_pen,
        )
        return window, metrics, flags, break_detected, False
    except Exception:
        logger.exception(
            "backtest failed sku_id=%s model=%s — writing fallback pattern_feedback",
            payload.sku_id, payload.assigned_model,
        )
        return 0, None, [], False, True


def _clamp_metric(v: Optional[float], lo: float = -99.0, hi: float = 99.0) -> Optional[float]:
    """Clamp a float metric to fit NUMERIC(8,6) — returns None for NaN/inf."""
    import math
    if v is None or math.isnan(v) or math.isinf(v):
        return None
    return max(lo, min(hi, v))


def _backtest_decisions_row(
        tenant_id: str,
        run_id: str,
        sku_id: str,
        window: int,
        metrics: Optional[BacktestMetrics],
        flags: list[str],
        break_detected: bool,
        ctx: BacktestContext,
) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "sku_id": sku_id,
        "run_id": run_id,
        "backtest_window_days": window,
        "backtest_mape": _clamp_metric(metrics.mape, lo=0.0) if metrics else None,
        "backtest_wape": _clamp_metric(metrics.wape, lo=0.0) if metrics else None,
        "backtest_bias": _clamp_metric(metrics.bias) if metrics else None,
        "exception_flags": _jsonb(flags),
        "structural_break_detected": break_detected,
        "break_index": ctx.break_index,
        "training_data_truncated": ctx.training_data_truncated,
    }


def _jsonb(flags: list[str]):
    """
    Adapt the exception_flags list for the JSONB column. Uses the
    module-level _PsycopgJson wrapper when psycopg2 is available; falls
    back to the plain list for unit tests against fake cursors.
    """
    if _PsycopgJson is None:
        return list(flags)
    return _PsycopgJson(list(flags))


# ---------------------------------------------------------------------------
# Parallel entry point — dual-pool (process pool + thread pool)
# ---------------------------------------------------------------------------

def run_substage_94_parallel(
        *,
        tenant_id: str,
        run_id: str,
        skus: list[SkuBacktestInput],
        params: TenantParams,
        fit_predict_fn: FitPredictFn,
        connect_fn: Callable[[], Any],
        pf_connect_fn: Callable[[], Any],
        log_failure_fn: Optional[Callable[[str, str, str, str], None]] = None,
        max_workers: Optional[int] = None,
        executor_factory: Optional[Callable[[int], Any]] = None,
        batch_size: int = 1000,
) -> dict[str, BacktestResult]:
    """
    Process pool variant of run_substage_94 — the production path.

    The caller supplies:
      * `connect_fn`     — zero-arg callable returning a fresh psycopg2
                           connection for the worker's main reads/writes
                           (calibrated cache + backtest_decisions).
      * `pf_connect_fn`  — same, but for the dedicated pattern_feedback
                           connection (kept isolated per P1-4).
      * `fit_predict_fn` — must be importable / picklable so it survives
                           the process boundary. A module-level function
                           or a CloudPickle-friendly closure works.

    `executor_factory` defaults to ProcessPoolExecutor; tests override
    with a synchronous executor or ThreadPoolExecutor to avoid spawning
    real processes.

    Each worker:
      1. Opens its own conn + pf_conn.
      2. Runs prefetch_calibrated_windows once for the tenant.
      3. Calls run_substage_94 on its assigned slice with its own
         BatchWriter (so flushes don't collide across workers).
      4. Closes both connections in a finally block.
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
    combined: dict[str, BacktestResult] = {}
    with executor_factory(max_workers) as pool:
        futures = [
            pool.submit(
                _worker_run_chunk,
                tenant_id, run_id, chunk, params,
                fit_predict_fn, connect_fn, pf_connect_fn, batch_size,
                log_failure_fn,
            )
            for chunk in chunks
        ]
        for fut in futures:
            combined.update(fut.result())
    return combined


def _partition(items: list, n: int) -> list[list]:
    """Split items into n contiguous chunks of near-equal size."""
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
        chunk: list[SkuBacktestInput],
        params: TenantParams,
        fit_predict_fn: FitPredictFn,
        connect_fn: Callable[[], Any],
        pf_connect_fn: Callable[[], Any],
        batch_size: int,
        log_failure_fn: Optional[Callable[[str, str, str, str], None]],
) -> dict[str, BacktestResult]:
    """
    Run inside one worker (process or thread).

    Owns its connections start-to-finish; closes them in a finally so a
    crashing SKU doesn't leak handles. Imported here at module top via
    the BatchWriter import.
    """
    from infrastructure.batch_writer import BatchWriter

    conn = connect_fn()
    pf_conn = pf_connect_fn()
    try:
        cache = prefetch_calibrated_windows(conn, tenant_id)
        bw = BatchWriter(conn, batch_size=batch_size)
        try:
            results = run_substage_94(
                conn,
                tenant_id=tenant_id,
                run_id=run_id,
                skus=chunk,
                params=params,
                fit_predict_fn=fit_predict_fn,
                batch_writer=bw,
                pf_conn=pf_conn,
                calibrated_cache=cache,
                log_failure_fn=log_failure_fn,
            )
            bw.flush()
            return results
        except Exception:
            logger.exception("worker chunk failed tenant_id=%s size=%d",
                             tenant_id, len(chunk))
            raise
    finally:
        for c in (conn, pf_conn):
            try:
                c.close()
            except Exception:
                pass
