"""
stage9_harness.py
=================
Full Stage 9 pipeline against a CSV input file with live DB and learning loop.

Runs the actual Stage 9 model classes + confidence engine and evaluates
forecast accuracy with a walk-forward backtest comparing each horizon
forecast vs actuals. Stage 8 inputs are read from optional CSV columns
or fall back to safe defaults. Requires a live PostgreSQL connection
(set DB_* environment variables before running).

Column auto-detection
---------------------
Required column names are detected automatically from common aliases so
you rarely need --col-xxx flags:

    sku_id        : sku_id, sku, product_id, item_id, product_code
    date          : date, order_date, transaction_date, sale_date, created_at, day
    qty           : qty, quantity, demand, units, sales, units_sold, volume
    pattern_label : pattern_label, pattern, demand_pattern, demand_type

Typical workflow
----------------
If your CSV already contains the Stage 8 columns (oos_pct, detection_confidence,
promo_weight, on_watchlist, confidence_calibrated, weekend_zero_ratio,
criticality_tier), pass it directly:

    python stage9_harness.py --csv your_data.csv [options]

If your CSV only has raw demand, run generate_stage8_inputs.py first:

    python generate_stage8_inputs.py --csv demand.csv --output demand_s8.csv
    python stage9_harness.py --csv demand_s8.csv [options]

Any Stage 8 column that is absent from the CSV falls back to a safe default.

Quick examples
--------------
    # Minimal — columns are auto-detected
    python stage9_harness.py --csv data.csv

    # No pattern_label column — assign one pattern to all SKUs
    python stage9_harness.py --csv data.csv --default-pattern stable

    # Override a specific column name that doesn't match any alias
    python stage9_harness.py --csv data.csv --col-qty ordered_units

    # Different tenant / maturity
    python stage9_harness.py --csv data.csv --tenant-id acme --tenant-maturity established

    # Verbose per-SKU output
    python stage9_harness.py --csv data.csv --verbose

Required CSV columns (auto-detected from aliases above)
--------------------------------------------------------
    sku_id          Unique SKU identifier
    date            Date column — any common format parsed automatically
    qty             Daily demand, float >= 0
    pattern_label   stable | trending | seasonal | intermittent | cold_start
                    (can be absent if --default-pattern is supplied)

Optional CSV columns (Stage 8 inputs — absent columns use safe defaults)
-------------------------------------------------------------------------
    oos_pct                 0.0-1.0   OOS fraction of history (default 0.0)
    detection_confidence    0.0-1.0   Confidence in OOS estimate (default 0.5)
    promo_weight            float     Promo multiplier per day (default 1.0)
    on_watchlist            bool      Stage 8 watchlist flag (default False)
    confidence_calibrated   0.0-1.0   Stage 8 pattern confidence (default 0.80)
    lifecycle_stage         str       introduction|growth|saturation|clearance (default "")
    weekend_zero_ratio      0.0-1.0   Fraction of weekend days with zero demand (default 0.0)
    criticality_tier        str       A or B — tier A overrides quantile to 0.99 (default "")

Flags
-----
    --csv PATH                Input CSV (required)
    --output DIR              Output directory (default: results/)
    --min-train N             Minimum training days before first eval (default: 90)
                              Cold start SKUs always use min_backtest_window (14) regardless.
    --eval-every N            Days between walk-forward eval points (default: 30)
    --verbose, -v             Print per-SKU detail during evaluation
    --col-sku COL             Override auto-detected SKU column name
    --col-date COL            Override auto-detected date column name
    --col-qty COL             Override auto-detected quantity column name
    --col-pattern COL         Override auto-detected pattern column name
    --default-pattern LABEL   Pattern label to assign when column is absent
                              (stable/trending/seasonal/intermittent/cold_start)
    --tenant-id ID            Tenant ID (plain name or UUID; default: harness-tenant)
    --tenant-maturity LEVEL   new | developing | established (default: new)
    --max-history N           Cap demand history to most recent N days per SKU

Output files
------------
    results/summary.csv          MAPE/WAPE/bias per SKU per horizon
    results/pattern_summary.csv  Aggregated by demand pattern
    results/overall_summary.csv  Single-row overall accuracy across all SKUs
    results/forecast_detail.csv  Every forecast vs actual comparison
    results/confidence_log.csv   Confidence scores and exception flags per SKU per eval date
    results/daily_7d.csv         Day-by-day breakdown of 7-day forecasts vs actuals

DB tables written
-----------------
    stage9.forecasts              Primary forecast output (Stage 10 reads this)
    stage9.feature_decisions_s9   Feature engineering audit trail
    stage9.thompson_sampling_state  Thompson sampling state
    stage9.data_fingerprint_cache   SKU fingerprints
    stage9.forecast_outcomes        Walk-forward accuracy for learning loop
    stage9.model_performance_s9     Aggregated model performance
"""

from __future__ import annotations

import argparse
import hashlib
import json as _json
import logging
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Stage 9 imports ───────────────────────────────────────────────────────
from infrastructure.constants import (
    HORIZONS, FORECAST_COLUMN_MAP, PATTERN_MODEL_MAP, Model, Pattern,
    OOS_ADJUSTMENT_MAX_FACTOR, REORDER_BIAS_FACTOR_DEFAULT,
    ULTRA_SPARSE_MAX_OBSERVATION_DAYS,
    B2B_WEEKEND_ZERO_RATIO_THRESHOLD, PATTERN_QUANTILE_PARAM,
    CRITICALITY_A_QUANTILE, CriticalityTier, LearningMode, Param, TenantMaturity,
    PROPHET_FAMILY,
)
from infrastructure.tenant_params import TenantParams
from infrastructure.tenant_params_defaults import TENANT_LEARNING_PARAMS_DEFAULTS
from models.bootstrap import bootstrap_quantiles
from forecasting.forecasting import (
    compute_confidence, determine_status, determine_tier,
    determine_risk_level, ForecastContext, _apply_dow_multipliers,
)
from backtesting.backtesting import (
    select_backtest_window, run_backtest, detect_exceptions,
    detect_structural_break, BacktestContext, MIN_BACKTEST_OBS_DAYS,
)
from forecasting.feature_engg import run_feature_engineering


class _NullBatchWriter:
    """No-op BatchWriter for harness — used in offline (--no-db) mode only."""

    def queue(self, *args, **kwargs): pass

    def flush_if_needed(self): pass

    def flush(self): pass


# ── Add project root to path so Stage 9 modules resolve ──────────────────
PROJECT_ROOT = Path(__file__).parent.parent / "project"
if PROJECT_ROOT.exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    # Fallback: assume script is run from project root
    sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cmdstanpy.*")
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


# ── Model class registry ──────────────────────────────────────────────────
def _load_model_classes() -> dict:
    from models.naive import NaiveForecast
    from models.ses import SESModel
    from models.holt import HoltLinearTrend
    from models.croston import CrostonMethod
    return {
        Model.NAIVE: NaiveForecast,
        Model.SES: SESModel,
        Model.HOLTS_LINEAR: HoltLinearTrend,
        Model.CROSTON: CrostonMethod,
    }


def _load_prophet_class():
    try:
        from models.prophet_model import ProphetModel
        return ProphetModel
    except ImportError:
        return None


# ═════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class SkuContext:
    """Mocked Stage 8 context for one SKU — built from CSV columns."""
    sku_id: str
    pattern_label: str
    obs_days: int = 0
    oos_adjustment_factor: float = 1.0
    on_watchlist: bool = False
    stage8_confidence: float = 0.80
    lifecycle_stage: Optional[str] = None
    is_b2b: bool = False
    criticality_tier: Optional[str] = None
    promo_weights: dict = field(default_factory=dict)  # {date_str: weight}
    # Draft 2 — set True when seasonal SKU has insufficient history for Prophet
    insufficient_seasonal_history: bool = False


@dataclass
class ForecastRecord:
    """One forecast generated at eval_date for one SKU covering all 8 horizons."""
    sku_id: str
    eval_date: str
    pattern_label: str
    assigned_model: str
    selected_quantile: float
    confidence_final: float
    confidence_tier: str
    risk_level: str
    status: str
    horizons: dict  # {horizon_days: {'mean':, 'p50':, 'p80':, 'p90':}}
    backtest_mape: float
    exception_flags: list
    insufficient_seasonal_history: bool = False


@dataclass
class AccuracyRecord:
    """One comparison of a forecast vs actuals for one (sku, eval_date, horizon)."""
    sku_id: str
    eval_date: str
    horizon_days: int
    pattern_label: str
    assigned_model: str
    selected_quantile: float
    forecast_mean: float
    forecast_p80: float
    forecast_p90: float
    actual: float
    mape: float
    wape: float
    bias: float
    mae: float
    rmse: float
    quantile_used: float  # the value at selected_quantile


@dataclass
class DailyRecord:
    """One day within the 7-day horizon forecast — shows how the model handles each weekday."""
    sku_id: str
    eval_date: str
    pattern_label: str
    assigned_model: str
    day_num: int  # 1-7 relative to eval_date
    forecast_date: str
    day_of_week: str  # Monday … Sunday
    is_weekend: bool
    forecast_qty: float  # model daily prediction × oos_adjustment_factor
    actual_qty: float
    error: float  # forecast − actual (signed)
    abs_error: float


# ═════════════════════════════════════════════════════════════════════════
# CSV loader
# ═════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = {"sku_id", "date", "qty", "pattern_label"}

# Common column name aliases used for auto-detection.
# First match found in the CSV wins; canonical name is the dict key.
_COL_ALIASES: dict[str, list[str]] = {
    "sku_id":        ["sku_id", "sku", "product_id", "item_id", "product_code", "item_code"],
    "date":          ["date", "order_date", "transaction_date", "sale_date", "created_at", "day"],
    "qty":           ["qty", "quantity", "demand", "units", "sales", "units_sold", "volume"],
    "pattern_label": ["pattern_label", "pattern", "demand_pattern", "demand_type"],
}

OPTIONAL_DEFAULTS = {
    "oos_pct": 0.0,
    "detection_confidence": 0.5,
    "promo_weight": 1.0,
    "on_watchlist": False,
    "confidence_calibrated": 0.80,
    "lifecycle_stage": "",
    "weekend_zero_ratio": 0.0,
    "criticality_tier": "",
}
VALID_PATTERNS = set(PATTERN_MODEL_MAP.keys())


def _resolve_col(
        canonical: str,
        override: Optional[str],
        actual_cols: list[str],
) -> Optional[str]:
    """Return the source column name that should map to `canonical`.

    Priority: explicit override > alias match > already-canonical name present.
    Returns None if nothing resolves (caller decides whether to error or skip).
    """
    if override:
        return override if override in actual_cols else None
    lower_cols = {c.lower(): c for c in actual_cols}
    for alias in _COL_ALIASES.get(canonical, [canonical]):
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def load_csv(
        path: str,
        col_sku: Optional[str] = None,
        col_date: Optional[str] = None,
        col_qty: Optional[str] = None,
        col_pattern: Optional[str] = None,
        default_pattern: Optional[str] = None,
        max_history_days: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=path)  # type: ignore[call-overload]
    actual_cols = list(df.columns)

    # Auto-detect or explicitly remap each required column
    rename = {}
    detected: list[str] = []
    for canonical, override in [
        ("sku_id", col_sku), ("date", col_date),
        ("qty", col_qty), ("pattern_label", col_pattern),
    ]:
        src = _resolve_col(canonical, override, actual_cols)
        if src and src != canonical:
            rename[src] = canonical
            detected.append(f"{src}->{canonical}")
        elif src is None and canonical != "pattern_label":
            raise ValueError(
                f"Cannot find a column for '{canonical}' in {path}. "
                f"Columns present: {actual_cols}. "
                f"Use --col-{canonical.replace('_','-')} to specify it explicitly."
            )
    if rename:
        df = df.rename(columns=rename)
        print(f"  Column mapping: {', '.join(detected)}")

    # Inject default pattern when column is absent entirely
    if "pattern_label" not in df.columns:
        if default_pattern is None:
            raise ValueError(
                "CSV has no 'pattern_label' column. "
                "Use --default-pattern <label> to set one."
            )
        df["pattern_label"] = default_pattern

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Fill optional columns with defaults
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0).clip(lower=0)

    # Cap to the most recent max_history_days per SKU — mirrors production's
    # MAX_DEMAND_HISTORY_DAYS (730) cutoff in acting/planning handlers.
    if max_history_days is not None:
        cutoff = df["date"].max() - pd.Timedelta(days=max_history_days)
        df = df[df["date"] > cutoff].reset_index(drop=True)
    df["pattern_label"] = df["pattern_label"].str.strip().str.lower()

    # Validate patterns
    bad = set(df["pattern_label"].unique()) - VALID_PATTERNS
    if bad:
        raise ValueError(
            f"Unknown pattern_label values: {bad}. "
            f"Valid: {sorted(VALID_PATTERNS)}"
        )

    df["on_watchlist"] = df["on_watchlist"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df


def build_sku_contexts(df: pd.DataFrame, params: "TenantParams") -> dict[str, SkuContext]:
    """Build one SkuContext per SKU from the first row of each SKU's data."""
    min_seasonal_obs = int(params.get(Param.MIN_SEASONAL_OBS_DAYS))
    contexts = {}
    for sku_id, group in df.groupby("sku_id"):
        first = group.iloc[0]
        pattern = first["pattern_label"]
        obs_days = len(group)

        # Draft 3 — intermittent SKUs: OOS uplift inflates forecasts (zero-streaks
        # are demand signal, not stockout events). Force factor to 1.0.
        if pattern == Pattern.INTERMITTENT:
            oos_factor = 1.0
        else:
            # OOS adjustment factor: 1 + (oos_pct * detection_confidence), capped at 1.50
            oos_pct = float(first.get("oos_pct", 0.0))
            det_conf = float(first.get("detection_confidence", 0.5))
            raw_factor = 1.0 + oos_pct * det_conf
            oos_factor = min(max(raw_factor, 1.0), OOS_ADJUSTMENT_MAX_FACTOR)

        # Draft 2 — seasonal guard: monthly-seasonal SKUs need >= min_seasonal_obs_days
        # of history before Prophet can calibrate. Override to Holt until threshold crossed.
        insufficient_seasonal = (
                pattern == Pattern.SEASONAL and obs_days < min_seasonal_obs
        )

        wzr = float(first.get("weekend_zero_ratio", 0.0))
        is_b2b = wzr > B2B_WEEKEND_ZERO_RATIO_THRESHOLD

        promo_rows = group[group["promo_weight"] < 1.0]
        promo_weights = {
            row["date"].strftime("%Y-%m-%d"): float(row["promo_weight"])
            for _, row in promo_rows.iterrows()
        }

        lifecycle = str(first.get("lifecycle_stage", "")).strip() or None
        criticality = str(first.get("criticality_tier", "")).strip() or None

        contexts[str(sku_id)] = SkuContext(
            sku_id=str(sku_id),
            pattern_label=pattern,
            obs_days=obs_days,
            oos_adjustment_factor=oos_factor,
            on_watchlist=bool(first.get("on_watchlist", False)),
            stage8_confidence=float(first.get("confidence_calibrated", 0.80)),
            lifecycle_stage=lifecycle,
            is_b2b=is_b2b,
            criticality_tier=criticality,
            promo_weights=promo_weights,
            insufficient_seasonal_history=insufficient_seasonal,
        )
    return contexts


# ═════════════════════════════════════════════════════════════════════════
# Model instantiation
# ═════════════════════════════════════════════════════════════════════════

MODEL_CLASSES = {}
PROPHET_CLASS = None


def _get_model_class(model_name: str):
    global MODEL_CLASSES, PROPHET_CLASS
    if not MODEL_CLASSES:
        MODEL_CLASSES = _load_model_classes()
    if model_name == Model.PROPHET:
        if PROPHET_CLASS is None:
            PROPHET_CLASS = _load_prophet_class()
        if PROPHET_CLASS is None:
            # Prophet not installed — fall back to Holt and warn once
            log.warning("Prophet not available; falling back to Holt for seasonal SKUs")
            return MODEL_CLASSES[Model.HOLTS_LINEAR]
        return PROPHET_CLASS
    return MODEL_CLASSES.get(model_name, MODEL_CLASSES[Model.SES])


def _default_hp(model_cls) -> dict:
    """Return default HP for a model class without instantiating with empty dict."""
    try:
        return model_cls(hp={}).default_hp
    except Exception:
        return {}


def _quantile_for_pattern(
        pattern: str,
        params: TenantParams,
        criticality_tier: Optional[str] = None,
) -> float:
    """Criticality A → fixed 0.99 override; all others → pattern default from params."""
    if criticality_tier == CriticalityTier.A:
        return CRITICALITY_A_QUANTILE
    param_name = PATTERN_QUANTILE_PARAM.get(pattern)
    try:
        return float(params.get(param_name))
    except Exception:
        return 0.90


def _fe_ctx(sku_id: str, is_b2b: bool, assigned_model: str,
            tenant_id: str = "harness-tenant", run_id: str = "harness-run") -> SimpleNamespace:
    """Minimal LearningContext substitute for run_feature_engineering."""
    return SimpleNamespace(
        sku_id=sku_id,
        tenant_id=tenant_id,
        run_id=run_id,
        assigned_model=assigned_model,
        is_b2b=is_b2b,
        b2b_mode_disabled=False,
    )


def _fe_preloaded(sku_id: str, promo_weights: dict) -> dict:
    """Minimal preloaded dict for run_feature_engineering.

    promo_weights is {date_str: weight}; production expects {(sku_id, date_str): weight}.
    Steps 1 (reliability) and 4 (feature search) are no-ops with empty dicts.
    """
    return {
        "feature_reliability": {},
        "promo_decisions": {(sku_id, d): w for d, w in promo_weights.items()},
        "feature_history": {},
    }


# ═════════════════════════════════════════════════════════════════════════
# Thompson sampling helpers (Phase 3 — learning loop)
# ═════════════════════════════════════════════════════════════════════════

# Maps tenant maturity string → exploit threshold param name (mirrors model_initialization.py).
_MATURITY_TO_EXPLOIT_PARAM: dict[str, str] = {
    TenantMaturity.NEW: Param.EXPLOIT_THRESHOLD_NEW,
    TenantMaturity.DEVELOPING: Param.EXPLOIT_THRESHOLD_DEVELOPING,
    TenantMaturity.ESTABLISHED: Param.EXPLOIT_THRESHOLD_ESTABLISHED,
}


def _config_hash(config: dict) -> str:
    serialised = _json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]


def _compute_learning_mode(
        sku_id: str,
        model_name: str,
        thompson_state: dict,
        params: "TenantParams",
        exploit_threshold_param: str,
) -> str:
    """Derive explore/exploit from accumulated Thompson evidence for this SKU."""
    key = (sku_id, model_name)
    configs = thompson_state.get(key, {})
    if not configs:
        return LearningMode.EXPLORE
    best_hash = max(
        configs,
        key=lambda h: configs[h].get("alpha", 1.0) / (
                configs[h].get("alpha", 1.0) + configs[h].get("beta", 1.0)
        ),
    )
    s = configs[best_hash]
    alpha = s.get("alpha", 1.0)
    beta = s.get("beta", 1.0)
    total_trials = s.get("total_trials", 0)
    thompson_confidence = alpha / (alpha + beta)
    confidence_threshold = float(params.get(Param.THOMPSON_EXPLOIT_CONFIDENCE_THRESHOLD))
    exploit_threshold = int(params.get(exploit_threshold_param))
    if thompson_confidence > confidence_threshold and total_trials >= exploit_threshold:
        return LearningMode.EXPLOIT
    return LearningMode.EXPLORE


def _update_thompson(
        sku_id: str,
        model_name: str,
        config: dict,
        backtest_mape: float,
        mape_cap: float,
        thompson_state: dict,
) -> None:
    """Update in-memory Thompson alpha/beta after a backtest outcome."""
    cfg_hash = _config_hash(config)
    key = (sku_id, model_name)
    if key not in thompson_state:
        thompson_state[key] = {}
    if cfg_hash not in thompson_state[key]:
        thompson_state[key][cfg_hash] = {
            "alpha": 1.0, "beta": 1.0, "total_trials": 0, "config": config,
        }
    s = thompson_state[key][cfg_hash]
    if backtest_mape <= mape_cap:
        s["alpha"] = s.get("alpha", 1.0) + 1.0
    else:
        s["beta"] = s.get("beta", 1.0) + 1.0
    s["total_trials"] = s.get("total_trials", 0) + 1


# ═════════════════════════════════════════════════════════════════════════
# DB helpers (Phase 1 — optional, enabled via --db flag)
# ═════════════════════════════════════════════════════════════════════════

def _load_thompson_from_db(db: Any, tenant_id: str) -> dict:
    """Load Thompson state from stage9.thompson_sampling_state into harness format."""
    sql = """
        SELECT sku_id::text, assigned_model, config_hash::text,
               alpha_param::float, beta_param::float, total_trials::int, config_json
        FROM stage9.thompson_sampling_state
        WHERE tenant_id = %s
    """
    result: dict = {}
    with db.cursor() as cur:
        cur.execute(sql, (tenant_id,))
        for sku_id, model, cfg_hash, alpha, beta, trials, config in cur.fetchall():
            key = (str(sku_id), str(model))
            if key not in result:
                result[key] = {}
            result[key][str(cfg_hash)] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "total_trials": int(trials),
                "config": config if isinstance(config, dict) else {},
            }
    return result


def _save_thompson_to_db(db: Any, tenant_id: str, thompson_state: dict,
                         sku_uuid_map: dict | None = None) -> None:
    """Upsert in-memory Thompson state back to stage9.thompson_sampling_state."""
    _to_uuid = sku_uuid_map or {}
    sql = """
        INSERT INTO stage9.thompson_sampling_state
            (tenant_id, sku_id, assigned_model, config_hash,
             config_json, alpha_param, beta_param, total_trials, last_updated_at)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, NOW())
        ON CONFLICT (tenant_id, sku_id, assigned_model, config_hash)
        DO UPDATE SET
            alpha_param     = EXCLUDED.alpha_param,
            beta_param      = EXCLUDED.beta_param,
            total_trials    = EXCLUDED.total_trials,
            last_updated_at = NOW()
    """
    with db.cursor() as cur:
        for (sku_id, model), configs in thompson_state.items():
            sku_uuid = _to_uuid.get(sku_id, sku_id)  # translate label -> UUID
            for cfg_hash, s in configs.items():
                cur.execute(sql, (
                    tenant_id, sku_uuid, model, cfg_hash,
                    _json.dumps(s.get("config", {})),
                    s.get("alpha", 1.0),
                    s.get("beta", 1.0),
                    s.get("total_trials", 0),
                ))
    db.commit()
    log.info("harness thompson_state saved tenant=%s configs=%d",
             tenant_id, sum(len(v) for v in thompson_state.values()))


def _load_fingerprints_from_db(db: Any, tenant_id: str) -> dict:
    """Load fingerprint cache from stage9.data_fingerprint_cache."""
    sql = """
        SELECT sku_id::text, fingerprint, pattern_label, demand_total
        FROM stage9.data_fingerprint_cache
        WHERE tenant_id = %s
    """
    result: dict = {}
    with db.cursor() as cur:
        cur.execute(sql, (tenant_id,))
        for sku_id, fingerprint, pattern_label, demand_total in cur.fetchall():
            result[str(sku_id)] = {
                "fingerprint": fingerprint,
                "pattern_label": pattern_label,
                "demand_total": float(demand_total) if demand_total is not None else 0.0,
            }
    return result


def _save_fingerprints_to_db(db: Any, tenant_id: str, fingerprints: dict,
                             sku_uuid_map: dict | None = None) -> None:
    """Upsert computed fingerprints to stage9.data_fingerprint_cache."""
    _to_uuid = sku_uuid_map or {}
    sql = """
        INSERT INTO stage9.data_fingerprint_cache
            (tenant_id, sku_id, fingerprint, tier, pattern_label, demand_total, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (tenant_id, sku_id)
        DO UPDATE SET
            fingerprint   = EXCLUDED.fingerprint,
            tier          = EXCLUDED.tier,
            pattern_label = EXCLUDED.pattern_label,
            demand_total  = EXCLUDED.demand_total,
            updated_at    = NOW()
    """
    with db.cursor() as cur:
        for sku_id, entry in fingerprints.items():
            sku_uuid = _to_uuid.get(sku_id, sku_id)  # translate label -> UUID
            cur.execute(sql, (
                tenant_id, sku_uuid,
                entry["fingerprint"],
                entry.get("tier", "full"),
                entry.get("pattern_label"),
                entry.get("demand_total"),
            ))
    db.commit()
    log.info("harness fingerprint_cache saved tenant=%s skus=%d", tenant_id, len(fingerprints))


# ═════════════════════════════════════════════════════════════════════════
# DB helpers (Phase 2 — learning loop)
# ═════════════════════════════════════════════════════════════════════════

def _seed_and_load_params(db: Any, tenant_id: str, tenant_maturity: str = "new") -> "TenantParams":
    """Seed default tenant params if not present, then load from DB.

    seed_tenant_params signature: (tenant_id, tenant_maturity, overrides_dict=None, conn=None)
    tenant_maturity is required positionally before conn. The original call
    passed db in the tenant_maturity slot, leaving conn=None and triggering
    the explicit guard: "conn is required".
    """
    from infrastructure.seed import seed_tenant_params
    from infrastructure.tenant_params import TenantParams as _TP
    seed_tenant_params(tenant_id, tenant_maturity, conn=db)
    return _TP.load(tenant_id, db)


def _write_forecasts_to_db(
        db: Any,
        tenant_id: str,
        harness_run_id: str,
        forecast_records: list,
        sku_uuid_map: dict | None = None,
) -> int:
    """Insert the latest ForecastRecord per SKU into stage9.forecasts.

    Only the most recent eval_date record per SKU is written so the UNIQUE
    (tenant_id, sku_id, run_id) constraint is always satisfied.
    """
    from psycopg2.extras import Json as _PgJson
    _to_uuid = sku_uuid_map or {}

    # Keep only the latest eval_date record per SKU
    latest: dict[str, "ForecastRecord"] = {}
    for r in forecast_records:
        sku_uuid = _to_uuid.get(r.sku_id, r.sku_id)
        if sku_uuid not in latest or r.eval_date > latest[sku_uuid].eval_date:
            latest[sku_uuid] = r

    sql = """
        INSERT INTO stage9.forecasts
            (tenant_id, sku_id, run_id, forecast_date, assigned_model, pattern_label,
             selected_quantile, confidence_final, confidence_tier, status,
             exception_flags, backtest_mape,
             forecast_7d, forecast_14d, forecast_30d, forecast_60d,
             forecast_90d, forecast_150d, forecast_180d, forecast_365d)
        VALUES (%s,%s,%s,CURRENT_DATE,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (tenant_id, sku_id, run_id) DO UPDATE SET
            forecast_date      = EXCLUDED.forecast_date,
            assigned_model     = EXCLUDED.assigned_model,
            confidence_final   = EXCLUDED.confidence_final,
            confidence_tier    = EXCLUDED.confidence_tier,
            status             = EXCLUDED.status,
            exception_flags    = EXCLUDED.exception_flags,
            backtest_mape      = EXCLUDED.backtest_mape,
            forecast_7d        = EXCLUDED.forecast_7d,
            forecast_14d       = EXCLUDED.forecast_14d,
            forecast_30d       = EXCLUDED.forecast_30d,
            forecast_60d       = EXCLUDED.forecast_60d,
            forecast_90d       = EXCLUDED.forecast_90d,
            forecast_150d      = EXCLUDED.forecast_150d,
            forecast_180d      = EXCLUDED.forecast_180d,
            forecast_365d      = EXCLUDED.forecast_365d
    """

    def _h(horizon_dict):
        """Convert a horizon dict (possibly with numpy floats) to a psycopg2 Json adapter."""
        if horizon_dict is None:
            return None
        return _PgJson({k: float(v) for k, v in horizon_dict.items()})

    import math
    count = 0
    with db.cursor() as cur:
        for sku_uuid, r in latest.items():
            bm = r.backtest_mape if not math.isnan(r.backtest_mape) else None
            h = r.horizons
            cur.execute(sql, (
                tenant_id, sku_uuid, harness_run_id,
                r.assigned_model, r.pattern_label,
                r.selected_quantile,
                round(r.confidence_final, 4),
                r.confidence_tier, r.status,
                _PgJson(r.exception_flags),
                round(bm, 6) if bm is not None else None,
                _h(h.get(7)), _h(h.get(14)),
                _h(h.get(30)), _h(h.get(60)),
                _h(h.get(90)), _h(h.get(150)),
                _h(h.get(180)), _h(h.get(365)),
            ))
            count += 1
    db.commit()
    return count


def _write_forecast_outcomes(
        db: Any,
        tenant_id: str,
        harness_run_id: str,
        accuracy_records: list,
        forecast_records: list,
        sku_uuid_map: dict | None = None,
) -> int:
    """
    Write walk-forward outcomes to stage9.forecast_outcomes.

    Aggregates multiple eval-date comparisons per (sku, horizon) into a single
    row (averaged MAPE/WAPE/bias) so the UNIQUE constraint is respected and the
    ModelPerformanceAggregator sees one clean summary per SKU per horizon.
    outcome_date is set to CURRENT_DATE so rows fall inside the aggregator's
    30-day rolling window.
    """
    import math
    from collections import defaultdict

    _to_uuid = sku_uuid_map or {}  # label -> UUID; identity fallback if not supplied

    def _sku_uuid(label: str) -> str:
        """Translate a display label back to its DB UUID. Falls back to label itself
        if it is already a valid UUID (e.g. callers that bypassed coercion)."""
        return _to_uuid.get(label, label)

    model_by_sku = {_sku_uuid(r.sku_id): r.assigned_model for r in forecast_records}

    buckets: dict = defaultdict(lambda: {
        "forecasts": [], "actuals": [], "mapes": [], "wapes": [], "biases": [],
    })
    for r in accuracy_records:
        if not math.isnan(r.mape):
            key = (_sku_uuid(r.sku_id), r.horizon_days)  # always store by UUID
            buckets[key]["forecasts"].append(r.forecast_mean)
            buckets[key]["actuals"].append(r.actual)
            buckets[key]["mapes"].append(r.mape)
            buckets[key]["wapes"].append(
                r.wape if not math.isnan(r.wape) else 0.0
            )
            buckets[key]["biases"].append(
                r.bias if not math.isnan(r.bias) else 0.0
            )

    sql = """
        INSERT INTO stage9.forecast_outcomes
            (tenant_id, sku_id, run_id, horizon_days, assigned_model,
             forecast_value, actual_value, error_mape, error_wape, bias, outcome_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
        ON CONFLICT (tenant_id, sku_id, run_id, horizon_days) DO NOTHING
    """
    count = 0
    with db.cursor() as cur:
        for (sku_id, horizon_days), v in buckets.items():
            n = len(v["mapes"])
            if n == 0:
                continue
            cur.execute(sql, (
                tenant_id, sku_id, harness_run_id, horizon_days,
                model_by_sku.get(sku_id, "unknown"),
                round(sum(v["forecasts"]) / n, 4),
                round(sum(v["actuals"]) / n, 4),
                round(sum(v["mapes"]) / n, 4),
                round(sum(v["wapes"]) / n, 4),
                round(sum(v["biases"]) / n, 4),
            ))
            count += 1
    db.commit()
    return count


def _run_learning_loop(db: Any, tenant_id: str) -> tuple:
    """Run ModelPerformanceAggregator then LearningParamsUpdater against written outcomes."""
    from datetime import date, timedelta
    from learning.model_performance_aggregator import run_model_performance_aggregator
    from learning.learning_params_updater import LearningParamsUpdater

    agg_stats = run_model_performance_aggregator(
        db, tenant_id=tenant_id,
        as_of=date.today() + timedelta(days=1),  # include today's outcomes
    )
    updater_result = LearningParamsUpdater().run(tenant_id, db)
    return agg_stats, updater_result


def _write_audit_tables(
        db: Any,
        tenant_id: str,
        harness_run_id: str,
        forecast_records: list,
        accuracy_records: list,
        contexts: dict,
        thompson_state: dict,
        exploit_threshold_param: str,
        params: "TenantParams",
        sku_uuid_map: dict | None = None,
) -> None:
    """Write the audit/decision tables that the production pipeline populates but
    the harness previously skipped:

        model_initialization_s9    — one row per SKU (model + quantile selection)
        hyperparameter_decisions   — one row per SKU (HP + Thompson score)
        backtest_decisions         — one row per SKU (MAPE, WAPE, bias, exception flags)
        stage9_sku_execution_log   — one row per SKU (success / failed status)
        adaptive_quantile_state    — one row per (SKU, model, pattern, horizon)
        stage9_self_assessment     — one row per run (aggregate health metrics)
        agent_state_log_s9         — synthetic IDLE → … → COMPLETE state trail
    """
    import math as _math
    from collections import defaultdict as _dd
    from psycopg2.extras import Json as _PgJson

    _to_uuid = sku_uuid_map or {}

    def _sku_uuid(label: str) -> str:
        return _to_uuid.get(label, label)

    # Latest ForecastRecord per SKU (keyed by UUID)
    latest: dict[str, "ForecastRecord"] = {}
    for r in forecast_records:
        su = _sku_uuid(r.sku_id)
        if su not in latest or r.eval_date > latest[su].eval_date:
            latest[su] = r

    skus_with_forecasts = set(latest.keys())

    with db.cursor() as cur:

        # ── 1. model_initialization_s9 ────────────────────────────────
        sql_mi = """
            INSERT INTO stage9.model_initialization_s9
                (tenant_id, sku_id, run_id, assigned_model,
                 insufficient_seasonal_history, pattern_label, lifecycle_stage,
                 selected_quantile, quantile_source, effective_max_horizon,
                 learning_mode, oos_adjustment_factor, is_b2b, reorder_bias_factor)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """
        for su, r in latest.items():
            ctx = contexts.get(su)
            oos_f = ctx.oos_adjustment_factor if ctx else 1.0
            is_b2b = ctx.is_b2b if ctx else False
            lifecycle = ctx.lifecycle_stage if ctx else None
            criticality = ctx.criticality_tier if ctx else None
            q_source = "criticality_a" if criticality == CriticalityTier.A else "pattern_param"
            lm = _compute_learning_mode(
                r.sku_id, r.assigned_model, thompson_state, params, exploit_threshold_param,
            )
            cur.execute(sql_mi, (
                tenant_id, su, harness_run_id,
                r.assigned_model, r.insufficient_seasonal_history,
                r.pattern_label, lifecycle,
                round(r.selected_quantile, 3), q_source, 365,
                lm, round(oos_f, 4), is_b2b, REORDER_BIAS_FACTOR_DEFAULT,
            ))

        # ── 2. hyperparameter_decisions ───────────────────────────────
        sql_hp = """
            INSERT INTO stage9.hyperparameter_decisions
                (tenant_id, sku_id, run_id, hyperparameters, validation_mape,
                 config_hash, thompson_score, early_stopped)
            VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
            ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """
        for su, r in latest.items():
            model_cls = _get_model_class(r.assigned_model)
            hp = _default_hp(model_cls)
            cfg_hash = _config_hash(hp)
            key = (r.sku_id, r.assigned_model)
            s = thompson_state.get(key, {}).get(cfg_hash, {})
            alpha = s.get("alpha", 1.0)
            beta = s.get("beta", 1.0)
            t_score = round(alpha / (alpha + beta), 4)
            bm = r.backtest_mape if not _math.isnan(r.backtest_mape) else None
            cur.execute(sql_hp, (
                tenant_id, su, harness_run_id,
                _PgJson(hp),
                round(bm, 6) if bm is not None else None,
                cfg_hash, t_score, False,
            ))

        # ── 3. backtest_decisions ─────────────────────────────────────
        sql_bd = """
            INSERT INTO stage9.backtest_decisions
                (tenant_id, sku_id, run_id, backtest_mape, backtest_wape,
                 backtest_bias, exception_flags, backtest_window_days,
                 structural_break_detected, break_index, training_data_truncated)
            VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
            ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """
        _acc30: dict = _dd(list)
        for ar in accuracy_records:
            if ar.horizon_days == 30 and not _math.isnan(ar.mape):
                _acc30[_sku_uuid(ar.sku_id)].append(ar)

        for su, r in latest.items():
            bm = r.backtest_mape if not _math.isnan(r.backtest_mape) else None
            wapes = [a.wape for a in _acc30.get(su, []) if not _math.isnan(a.wape)]
            biases = [a.bias for a in _acc30.get(su, []) if not _math.isnan(a.bias)]
            bw = round(sum(wapes) / len(wapes), 6) if wapes else None
            bb = round(sum(biases) / len(biases), 6) if biases else None
            cur.execute(sql_bd, (
                tenant_id, su, harness_run_id,
                round(bm, 6) if bm is not None else None,
                bw, bb,
                _PgJson(r.exception_flags),
                None, False, None, False,
            ))

        # ── 4. stage9_sku_execution_log ───────────────────────────────
        sql_el = """
            INSERT INTO stage9.stage9_sku_execution_log
                (tenant_id, run_id, sku_id, status,
                 fallback_model, error_code, error_message, sub_stage, execution_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        for su in skus_with_forecasts:
            cur.execute(sql_el, (
                tenant_id, harness_run_id, su,
                "success", None, None, None, None, None,
            ))
        all_sku_uuids = {_to_uuid.get(str(sid), str(sid)) for sid in contexts}
        for su in all_sku_uuids - skus_with_forecasts:
            cur.execute(sql_el, (
                tenant_id, harness_run_id, su,
                "failed", None, None, "insufficient_history", None, None,
            ))

        # ── 5. adaptive_quantile_state ────────────────────────────────
        sql_aqs = """
            INSERT INTO stage9.adaptive_quantile_state
                (tenant_id, sku_id, assigned_model, pattern_label, horizon_days,
                 sample_size, target_quantile, actual_coverage)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tenant_id, sku_id, assigned_model, pattern_label, horizon_days)
            DO UPDATE SET
                sample_size     = EXCLUDED.sample_size,
                target_quantile = EXCLUDED.target_quantile,
                actual_coverage = EXCLUDED.actual_coverage,
                last_updated    = NOW()
        """
        _buckets: dict = _dd(list)
        for ar in accuracy_records:
            _buckets[(_sku_uuid(ar.sku_id), ar.assigned_model, ar.pattern_label, ar.horizon_days)].append(ar)

        for (su, model, pattern, horizon), recs in _buckets.items():
            valid = [a for a in recs if not _math.isnan(a.actual)]
            if not valid:
                continue
            covered = sum(1 for a in valid if a.actual <= a.quantile_used)
            cur.execute(sql_aqs, (
                tenant_id, su, model, pattern, horizon,
                len(valid),
                round(valid[0].selected_quantile, 3),
                round(covered / len(valid), 3),
            ))

        # ── 6. stage9_self_assessment ─────────────────────────────────
        sql_sa = """
            INSERT INTO stage9.stage9_self_assessment
                (tenant_id, run_id, avg_mape_this_run,
                 degradation_detected, total_skus_processed,
                 cache_tier_count, partial_tier_count, full_tier_count,
                 fallback_count, pattern_feedback_retry_count, execution_mode)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tenant_id, run_id) DO NOTHING
        """
        mapes_30 = [a.mape for a in accuracy_records
                    if a.horizon_days == 30 and not _math.isnan(a.mape)]
        avg_mape = round(sum(mapes_30) / len(mapes_30), 6) if mapes_30 else None
        cur.execute(sql_sa, (
            tenant_id, harness_run_id,
            avg_mape, False,
            len(contexts), 0, 0, len(skus_with_forecasts), 0, 0,
            "FULL",
        ))

        # ── 7. agent_state_log_s9 ─────────────────────────────────────
        sql_asl = """
            INSERT INTO agent_state_log_s9
                (tenant_id, run_id, from_state, to_state, transitioned_at, reason)
            VALUES (%s,%s,%s,%s,NOW(),NULL)
        """
        for from_s, to_s in [
            ("IDLE", "PRELOADING"), ("PRELOADING", "PERCEIVING"),
            ("PERCEIVING", "PLANNING"), ("PLANNING", "ACTING"),
            ("ACTING", "LEARNING"), ("LEARNING", "REPORTING"),
            ("REPORTING", "COMPLETE"),
        ]:
            cur.execute(sql_asl, (tenant_id, harness_run_id, from_s, to_s))

    db.commit()
    log.info("harness audit_tables written tenant=%s run=%s", tenant_id, harness_run_id)


# ═════════════════════════════════════════════════════════════════════════
# Single-SKU walk-forward evaluation
# ═════════════════════════════════════════════════════════════════════════

def _compute_mape(actual: float, forecast: float) -> float:
    if actual == 0:
        return float("nan")
    return abs(actual - forecast) / abs(actual)


def _compute_wape(actual: float, forecast: float) -> float:
    if actual == 0:
        return float("nan")
    return abs(actual - forecast) / abs(actual)


def _compute_bias(actual: float, forecast: float) -> float:
    if actual == 0:
        return float("nan")
    return (forecast - actual) / abs(actual)


def _compute_mae(actual: float, forecast: float) -> float:
    return abs(actual - forecast)


def _compute_rmse(actual: float, forecast: float) -> float:
    return (actual - forecast) ** 2  # stored as squared error; sqrt taken at aggregation


def evaluate_sku(
        sku_id: str,
        sku_df: pd.DataFrame,
        ctx: SkuContext,
        params: TenantParams,
        min_train_days: int,
        eval_every_days: int,
        verbose: bool,
        thompson_state: dict,
        exploit_threshold_param: str,
        batch_writer=None,
        tenant_id: str = "harness-tenant",
        run_id: str = "harness-run",
        sku_uuid: Optional[str] = None,
) -> tuple[list[ForecastRecord], list[AccuracyRecord], list[DailyRecord]]:
    """
    Walk-forward evaluation for one SKU.

    For each eval_date (spaced eval_every_days apart, starting after
    min_train_days of history), generates 8-horizon forecasts using the
    assigned model, then compares vs actuals once each horizon closes.
    """
    forecast_records: list[ForecastRecord] = []
    accuracy_records: list[AccuracyRecord] = []
    daily_records: list[DailyRecord] = []

    sku_df = sku_df.sort_values("date").reset_index(drop=True)
    all_dates = sku_df["date"].values
    total_days = len(sku_df)

    pattern = ctx.pattern_label

    # Cold start SKUs have limited data by design — use min_backtest_window (14d)
    # as their minimum training window instead of the global min_train_days (90d).
    if pattern == Pattern.COLD_START:
        effective_min_train = int(params.get(Param.MIN_BACKTEST_WINDOW))
    else:
        effective_min_train = min_train_days

    if total_days < effective_min_train + 7:
        if verbose:
            print(f"  SKU {sku_id}: only {total_days} days — skipping (need {effective_min_train + 7})")
        return [], [], []

    base_model_name = PATTERN_MODEL_MAP.get(pattern, Model.SES)
    # Draft 2 context-level guard (whole-SKU: total history too short for any Prophet window)
    if ctx.insufficient_seasonal_history:
        base_model_name = Model.HOLTS_LINEAR
    min_seasonal_obs = int(params.get(Param.MIN_SEASONAL_OBS_DAYS))
    selected_quantile = _quantile_for_pattern(pattern, params, ctx.criticality_tier)

    if verbose:
        guard_note = " [seasonal->Holt guard]" if ctx.insufficient_seasonal_history else ""
        _display_cls = _get_model_class(base_model_name)
        actual_model = _display_cls(hp={}).model_name if base_model_name != Model.PROPHET else "Prophet"
        print(f"  SKU {sku_id}: pattern={pattern}, model={actual_model}{guard_note}, "
              f"quantile={selected_quantile}, days={total_days}")

    # Walk-forward: evaluate at each eval date
    eval_indices = list(range(
        effective_min_train,
        total_days - 7,  # need at least 7 days of future for H1
        eval_every_days,
    ))
    if not eval_indices:
        return [], [], []

    for eval_idx in eval_indices:
        eval_date = pd.Timestamp(all_dates[eval_idx])
        eval_date_str = eval_date.strftime("%Y-%m-%d")

        # Training data up to (but not including) eval_date
        train_raw = sku_df.iloc[:eval_idx].copy()

        # Draft 2 — per-window seasonal guard: eval_idx is the raw training size.
        # A seasonal SKU may have enough total history to pass the context-level
        # check but early walk-forward windows are still too small for Prophet.
        window_insufficient = (
                pattern == Pattern.SEASONAL
                and base_model_name == Model.PROPHET
                and eval_idx < min_seasonal_obs
        )
        model_name = Model.HOLTS_LINEAR if window_insufficient else base_model_name
        model_cls = _get_model_class(model_name)
        hp = _default_hp(model_cls)
        model = model_cls(hp=hp)

        if verbose and window_insufficient:
            print(f"    {eval_date_str}: window={eval_idx}d < {min_seasonal_obs}d "
                  f"-- seasonal->Holt (per-window guard)")

        # ── Feature engineering via production Sub-Stage 9.2 logic ───
        # Each walk-forward window gets a distinct run_id derived from the parent
        # run_id + eval_date so the UNIQUE (tenant_id, sku_id, run_id) constraint
        # on feature_decisions_s9 is satisfied across multiple windows.
        # sku_uuid is the DB UUID for this SKU; sku_id may be a display label.
        _db_sku = sku_uuid or sku_id
        try:
            _run_uuid = uuid.UUID(run_id)
            window_run_id = str(uuid.uuid5(_run_uuid, f"{_db_sku}:{eval_date_str}"))
        except ValueError:
            window_run_id = str(uuid.uuid4())
        fe_result = run_feature_engineering(
            ctx=_fe_ctx(_db_sku, ctx.is_b2b, model_name, tenant_id=tenant_id, run_id=window_run_id),
            df=train_raw[["date", "qty"]].copy(),
            model=model,
            preloaded=_fe_preloaded(sku_id, ctx.promo_weights),
            params=params,
            batch_writer=batch_writer or _NullBatchWriter(),
        )
        train_df = fe_result.df_train

        # Cold-start uses NaiveForecast which needs only 1-2 rows; other models
        # need enough data for a meaningful backtest window.
        min_rows = 2 if pattern == Pattern.COLD_START else 14
        if len(train_df) < min_rows:
            if verbose:
                print(f"    {eval_date_str}: skipping — only {len(train_df)} rows after FE (need {min_rows})")
            continue

        # ── Fit model on prepared training data ───────────────────────
        try:
            model.fit(train_df, ["date", "qty"])
        except Exception as fit_exc:
            if verbose:
                print(f"    {eval_date_str}: model.fit() failed — {fit_exc}")
            continue

        # ── Backtest via production Sub-Stage 9.4 logic ───────────────
        obs_days = len(train_df)
        ultra_sparse = obs_days < ULTRA_SPARSE_MAX_OBSERVATION_DAYS
        bt_ctx = BacktestContext()
        backtest_mape = float("nan")
        exception_flags: list[str] = []

        # Learning mode derived from accumulated Thompson evidence (Phase 3).
        learning_mode = _compute_learning_mode(
            sku_id, model_name, thompson_state, params, exploit_threshold_param,
        )

        if obs_days >= MIN_BACKTEST_OBS_DAYS:
            def _fit_predict(train: "pd.DataFrame", test_len: int) -> "np.ndarray":
                m = model_cls(hp=hp)
                m.fit(train, ["date", "qty"])
                return m.predict(train, ["date", "qty"], horizon=test_len)

            try:
                bt_window = select_backtest_window(
                    None, "harness-tenant", pattern, model_name, params,
                    obs_days=obs_days,
                    ultra_sparse=ultra_sparse,
                    learning_mode=learning_mode,
                    calibrated_cache={},
                )
                bt_metrics = run_backtest(train_df, bt_window, _fit_predict)
                backtest_mape = bt_metrics.mape
                exception_flags = detect_exceptions(bt_metrics.actual, backtest_mape)
                detect_structural_break(
                    train_df["qty"].to_numpy(dtype=float),
                    [],
                    bt_ctx,
                )
                # Update Thompson state so subsequent eval points benefit from
                # accumulated evidence — this is the learning loop that was absent.
                if not np.isnan(backtest_mape):
                    mape_cap = float(params.get(Param.MAPE_CAP_IN_CONFIDENCE))
                    _update_thompson(
                        sku_id, model_name, hp, backtest_mape, mape_cap, thompson_state,
                    )
            except Exception as bt_exc:
                if verbose:
                    print(f"    {eval_date_str}: backtest failed — {bt_exc}")

        # ── Residuals (shared across all horizons) ────────────────────
        try:
            residuals = model.compute_residuals(train_df, ["date", "qty"])
        except Exception as resid_exc:
            if verbose:
                print(f"    {eval_date_str}: compute_residuals failed — {resid_exc}")
            continue

        # ── Bootstrap quantiles per horizon ───────────────────────────
        daily_arr_7d: Optional[np.ndarray] = None
        try:
            # Collect raw OOS-adjusted point totals for all horizons first,
            # then apply DoW multipliers (non-Prophet only) before bootstrapping.
            raw_points: dict[int, float] = {}
            for H in HORIZONS:
                point_arr = model.predict(train_df, ["date", "qty"], horizon=H)
                if H == 7:
                    daily_arr_7d = point_arr * ctx.oos_adjustment_factor
                raw_points[H] = float(point_arr.sum()) * ctx.oos_adjustment_factor

            if model_name not in PROPHET_FAMILY:
                raw_points = _apply_dow_multipliers(
                    raw_points, fe_result.dow_multipliers, train_df,
                )

            forecasts_raw = {}
            for H in HORIZONS:
                col = FORECAST_COLUMN_MAP[H]
                forecasts_raw[col] = bootstrap_quantiles(raw_points[H], residuals, pattern)
        except Exception as fc_exc:
            if verbose:
                print(f"    {eval_date_str}: forecast generation failed — {fc_exc}")
            continue

        # Merge feature engineering edge-case flags (e.g. b2b_mode_disabled for E006)
        # Done before compute_confidence so all flags are visible to status/penalty logic,
        # matching the production path where fe.exception_flags enters SkuForecastInput.
        exception_flags = exception_flags + fe_result.exception_flags

        # ── Confidence score ───────────────────────────────────────────
        fc_ctx = ForecastContext(
            training_data_truncated=bt_ctx.training_data_truncated,
            insufficient_post_break=bt_ctx.insufficient_post_break,
            effective_max_horizon=365,
            reorder_bias_factor=REORDER_BIAS_FACTOR_DEFAULT,
            oos_adjustment_factor=ctx.oos_adjustment_factor,
            on_watchlist=ctx.on_watchlist,
        )
        _, confidence_final = compute_confidence(
            pattern_label=pattern,
            backtest_mape=backtest_mape if not np.isnan(backtest_mape) else 0.50,
            exception_flags=exception_flags,
            calibration_gap=None,
            stage8_confidence=ctx.stage8_confidence,
            reorder_bias_factor=REORDER_BIAS_FACTOR_DEFAULT,
            ctx=fc_ctx,
            params=params,
        )

        status = determine_status(confidence_final, exception_flags, fc_ctx, params)
        tier = determine_tier(confidence_final, params)
        risk = determine_risk_level(confidence_final, params)

        # ── Store forecast record ──────────────────────────────────────
        horizons_dict = {}
        for h in HORIZONS:
            col = FORECAST_COLUMN_MAP[h]
            horizons_dict[h] = forecasts_raw[col]

        fc_rec = ForecastRecord(
            sku_id=sku_id,
            eval_date=eval_date_str,
            pattern_label=pattern,
            assigned_model=model_name,
            selected_quantile=selected_quantile,
            confidence_final=confidence_final,
            confidence_tier=tier,
            risk_level=risk,
            status=status,
            horizons=horizons_dict,
            backtest_mape=backtest_mape,
            exception_flags=exception_flags,
            insufficient_seasonal_history=ctx.insufficient_seasonal_history or window_insufficient,
        )
        forecast_records.append(fc_rec)

        # ── Daily breakdown for H=7 ───────────────────────────────────
        if daily_arr_7d is not None and eval_idx + 7 <= total_days:
            for d in range(7):
                day_row = sku_df.iloc[eval_idx + d]
                fcast_d = float(daily_arr_7d[d])
                actual_d = float(day_row["qty"])
                dow = day_row["date"].strftime("%A")
                daily_records.append(DailyRecord(
                    sku_id=sku_id,
                    eval_date=eval_date_str,
                    pattern_label=pattern,
                    assigned_model=model_name,
                    day_num=d + 1,
                    forecast_date=day_row["date"].strftime("%Y-%m-%d"),
                    day_of_week=dow,
                    is_weekend=dow in ("Saturday", "Sunday"),
                    forecast_qty=round(fcast_d, 4),
                    actual_qty=round(actual_d, 4),
                    error=round(fcast_d - actual_d, 4),
                    abs_error=round(abs(fcast_d - actual_d), 4),
                ))

        # ── Compare vs actuals for each horizon ────────────────────────
        for h in HORIZONS:
            horizon_end_idx = eval_idx + h
            if horizon_end_idx > total_days:
                continue  # horizon hasn't closed yet — skip

            actual_window = sku_df.iloc[eval_idx:horizon_end_idx]["qty"].values
            actual_total = float(actual_window.sum())

            q_key = (
                "p90" if selected_quantile >= 0.90
                else "p80" if selected_quantile >= 0.80
                else "p50"
            )
            fc_quantile = float(horizons_dict[h].get(q_key, horizons_dict[h]["mean"]))
            fc_mean = float(horizons_dict[h]["mean"])
            fc_p80 = float(horizons_dict[h]["p80"])
            fc_p90 = float(horizons_dict[h]["p90"])

            mape = _compute_mape(actual_total, fc_mean)
            wape = _compute_wape(actual_total, fc_mean)
            bias = _compute_bias(actual_total, fc_mean)
            mae = _compute_mae(actual_total, fc_mean)
            rmse = _compute_rmse(actual_total, fc_mean)

            accuracy_records.append(AccuracyRecord(
                sku_id=sku_id,
                eval_date=eval_date_str,
                horizon_days=h,
                pattern_label=pattern,
                assigned_model=model_name,
                selected_quantile=selected_quantile,
                forecast_mean=fc_mean,
                forecast_p80=fc_p80,
                forecast_p90=fc_p90,
                actual=actual_total,
                mape=mape,
                wape=wape,
                bias=bias,
                mae=mae,
                rmse=rmse,
                quantile_used=fc_quantile,
            ))

    return forecast_records, accuracy_records, daily_records


# ═════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════

def build_summary(accuracy_records: list[AccuracyRecord]) -> pd.DataFrame:
    """Per-SKU per-horizon summary (mean MAPE, WAPE, bias across all eval dates)."""
    if not accuracy_records:
        return pd.DataFrame()
    rows = [
        {
            "sku_id": r.sku_id,
            "horizon_days": r.horizon_days,
            "pattern_label": r.pattern_label,
            "assigned_model": r.assigned_model,
            "selected_quantile": r.selected_quantile,
            "mape": r.mape,
            "wape": r.wape,
            "bias": r.bias,
            "mae": r.mae,
            "sq_err": r.rmse,
            "n_evals": 1,
        }
        for r in accuracy_records
        if not np.isnan(r.mape)
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["sku_id", "horizon_days", "pattern_label", "assigned_model", "selected_quantile"])
        .agg(
            mape_mean=("mape", "mean"),
            mape_median=("mape", "median"),
            wape_mean=("wape", "mean"),
            bias_mean=("bias", "mean"),
            mae_mean=("mae", "mean"),
            mse_mean=("sq_err", "mean"),
            n_evals=("n_evals", "sum"),
        )
        .reset_index()
    )
    agg["rmse"] = np.sqrt(agg["mse_mean"])
    agg = agg.drop(columns=["mse_mean"]).round(4)
    return agg.sort_values(["sku_id", "horizon_days"])


def build_pattern_summary(accuracy_records: list[AccuracyRecord]) -> pd.DataFrame:
    """Aggregated MAPE by demand pattern and horizon — the 'both + breakdown' view."""
    if not accuracy_records:
        return pd.DataFrame()
    rows = [
        {
            "pattern_label": r.pattern_label,
            "horizon_days": r.horizon_days,
            "mape": r.mape,
            "wape": r.wape,
            "bias": r.bias,
            "mae": r.mae,
            "sq_err": r.rmse,
        }
        for r in accuracy_records
        if not np.isnan(r.mape)
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["pattern_label", "horizon_days"])
        .agg(
            mape_mean=("mape", "mean"),
            mape_median=("mape", "median"),
            wape_mean=("wape", "mean"),
            bias_mean=("bias", "mean"),
            mae_mean=("mae", "mean"),
            mse_mean=("sq_err", "mean"),
            sku_count=("mape", "count"),
        )
        .reset_index()
    )
    agg["rmse"] = np.sqrt(agg["mse_mean"])
    agg = agg.drop(columns=["mse_mean"]).round(4)
    return agg.sort_values(["pattern_label", "horizon_days"])


def build_overall_summary(accuracy_records: list[AccuracyRecord]) -> pd.DataFrame:
    """Overall MAPE per horizon across all SKUs and patterns."""
    if not accuracy_records:
        return pd.DataFrame()
    rows = [
        {
            "horizon_days": r.horizon_days,
            "mape": r.mape,
            "wape": r.wape,
            "bias": r.bias,
            "mae": r.mae,
            "sq_err": r.rmse,
        }
        for r in accuracy_records
        if not np.isnan(r.mape)
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("horizon_days")
        .agg(
            mape_mean=("mape", "mean"),
            mape_median=("mape", "median"),
            wape_mean=("wape", "mean"),
            bias_mean=("bias", "mean"),
            mae_mean=("mae", "mean"),
            mse_mean=("sq_err", "mean"),
            n_comparisons=("mape", "count"),
        )
        .reset_index()
    )
    agg["rmse"] = np.sqrt(agg["mse_mean"])
    return agg.drop(columns=["mse_mean"]).round(4)


def build_confidence_log(forecast_records: list[ForecastRecord]) -> pd.DataFrame:
    if not forecast_records:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "sku_id": r.sku_id,
            "eval_date": r.eval_date,
            "pattern_label": r.pattern_label,
            "assigned_model": r.assigned_model,
            "confidence_final": round(r.confidence_final, 4),
            "confidence_tier": r.confidence_tier,
            "risk_level": r.risk_level,
            "status": r.status,
            "backtest_mape": round(r.backtest_mape, 4) if not np.isnan(r.backtest_mape) else None,
            "exception_flags": "|".join(r.exception_flags) if r.exception_flags else "",
            "insufficient_seasonal_history": r.insufficient_seasonal_history,
        }
        for r in forecast_records
    ])


def build_daily_detail(daily_records: list[DailyRecord]) -> pd.DataFrame:
    """Day-by-day 7-horizon breakdown — shows how the model handles each day of the week."""
    if not daily_records:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            "sku_id": r.sku_id,
            "eval_date": r.eval_date,
            "pattern_label": r.pattern_label,
            "assigned_model": r.assigned_model,
            "day_num": r.day_num,
            "forecast_date": r.forecast_date,
            "day_of_week": r.day_of_week,
            "is_weekend": r.is_weekend,
            "forecast_qty": r.forecast_qty,
            "actual_qty": r.actual_qty,
            "error": r.error,
            "abs_error": r.abs_error,
        }
        for r in daily_records
    ])
    return df


def build_detail(accuracy_records: list[AccuracyRecord]) -> pd.DataFrame:
    if not accuracy_records:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "sku_id": r.sku_id,
            "eval_date": r.eval_date,
            "horizon_days": r.horizon_days,
            "pattern_label": r.pattern_label,
            "assigned_model": r.assigned_model,
            "forecast_mean": round(r.forecast_mean, 4),
            "forecast_p80": round(r.forecast_p80, 4),
            "forecast_p90": round(r.forecast_p90, 4),
            "actual": round(r.actual, 4),
            "mape": round(r.mape, 4) if not np.isnan(r.mape) else None,
            "wape": round(r.wape, 4) if not np.isnan(r.wape) else None,
            "bias": round(r.bias, 4) if not np.isnan(r.bias) else None,
            "mae_units": round(r.mae, 2),
            "rmse_units": round(np.sqrt(r.rmse), 2),
        }
        for r in accuracy_records
    ])


def print_learning_loop_report(agg_stats: Any, updater_result: dict) -> None:
    print()
    print("-- Learning Loop " + "-" * 48)
    print(f"  ModelPerformanceAggregator: rows_written={agg_stats.rows_written} "
          f"  new_models={agg_stats.new_models or []}")
    if agg_stats.failure_reason:
        print(f"  WARNING: aggregator failure — {agg_stats.failure_reason}")
    status = updater_result.get("status", "unknown")
    updates = updater_result.get("params_updated", 0)
    print(f"  LearningParamsUpdater:      status={status}  params_updated={updates}")


def print_console_report(
        overall: pd.DataFrame,
        pattern_df: pd.DataFrame,
        sku_summary: pd.DataFrame,
        forecast_records: list[ForecastRecord],
) -> None:
    print()
    print("=" * 65)
    print("  STAGE 9 HARNESS -- ACCURACY REPORT")
    print("=" * 65)

    if not overall.empty:
        print()
        print("-- Overall Accuracy by Horizon " + "-" * 34)
        print(
            f"  {'Horizon':>8}  {'MAPE mean':>10}  {'MAE (units)':>12}"
            f"  {'RMSE (units)':>13}  {'Bias mean':>10}  {'N':>6}")
        for _, row in overall.iterrows():
            print(f"  {int(row['horizon_days']):>8}d  "
                  f"{row['mape_mean']:>10.3f}  "
                  f"{row['mae_mean']:>12.1f}  "
                  f"{row['rmse']:>13.1f}  "
                  f"{row['bias_mean']:>10.3f}  "
                  f"{int(row['n_comparisons']):>6}")

    if not pattern_df.empty:
        print()
        print("-- MAPE by Pattern x Horizon " + "-" * 36)
        patterns = pattern_df["pattern_label"].unique()
        for pattern in sorted(patterns):
            pdata = pattern_df[pattern_df["pattern_label"] == pattern]
            print(f"\n  Pattern: {pattern.upper()}")
            print(f"  {'Horizon':>8}  {'MAPE mean':>10}  {'WAPE mean':>10}  {'Bias mean':>10}  {'N':>6}")
            for _, row in pdata.iterrows():
                print(f"  {int(row['horizon_days']):>8}d  "
                      f"{row['mape_mean']:>10.3f}  "
                      f"{row['wape_mean']:>10.3f}  "
                      f"{row['bias_mean']:>10.3f}  "
                      f"{int(row['sku_count']):>6}")

    if not sku_summary.empty:
        print()
        print("-- Per-SKU MAPE at 30-day Horizon " + "-" * 31)
        s30 = sku_summary[sku_summary["horizon_days"] == 30].sort_values("mape_mean")
        if not s30.empty:
            print(f"  {'SKU':>12}  {'Pattern':>14}  {'Model':>30}  {'MAPE':>8}  {'Bias':>8}  {'N':>5}")
            for _, row in s30.iterrows():
                model_short = row["assigned_model"].replace("_", " ").title()[:28]
                print(f"  {str(row['sku_id']):>12}  "
                      f"{row['pattern_label']:>14}  "
                      f"{model_short:>30}  "
                      f"{row['mape_mean']:>8.3f}  "
                      f"{row['bias_mean']:>8.3f}  "
                      f"{int(row['n_evals']):>5}")

    # Confidence tier distribution
    if forecast_records:
        from collections import Counter
        tiers = Counter(r.confidence_tier for r in forecast_records)
        print()
        print("-- Confidence Tier Distribution " + "-" * 33)
        total = sum(tiers.values())
        for tier, count in sorted(tiers.items()):
            pct = 100 * count / total if total > 0 else 0
            print(f"  {tier:>25}: {count:>5}  ({pct:>5.1f}%)")

    print()
    print("=" * 65)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def _run_evaluation_block(
        params, contexts, sku_ids, sku_label_map, df,
        args, thompson_state, exploit_threshold_param,
        batch_writer=None,
        tenant_id: str = "harness-tenant",
        run_id: str = "harness-run",
):
    """Walk-forward evaluation + report building, shared by DB and offline paths."""
    print(f"\nRunning walk-forward evaluation "
          f"(min_train={args.min_train}d, eval_every={args.eval_every}d, "
          f"maturity={args.tenant_maturity}) ...")

    fc_list = []
    acc_list = []
    day_list = []

    for i, sku_id in enumerate(sku_ids, 1):
        print(f"  [{i}/{len(sku_ids)}] {sku_label_map.get(sku_id, sku_id)}", end="")
        if args.verbose:
            print()
        sku_df = df[df["sku_id"] == sku_id].copy()
        ctx = contexts[str(sku_id)]

        _sku_label = sku_label_map.get(sku_id, sku_id)
        fc_recs, acc_recs, day_recs = evaluate_sku(
            sku_id=_sku_label,
            sku_df=sku_df,
            ctx=ctx,
            params=params,
            min_train_days=args.min_train,
            eval_every_days=args.eval_every,
            verbose=args.verbose,
            thompson_state=thompson_state,
            exploit_threshold_param=exploit_threshold_param,
            batch_writer=batch_writer,
            tenant_id=tenant_id,
            run_id=run_id,
            sku_uuid=str(sku_id),
        )
        fc_list.extend(fc_recs)
        acc_list.extend(acc_recs)
        day_list.extend(day_recs)

        if not args.verbose:
            print(f"  -> {len(fc_recs)} forecasts, {len(acc_recs)} comparisons")

    print(f"\nTotal: {len(fc_list)} forecast runs, {len(acc_list)} horizon comparisons")

    sku_summary = build_summary(acc_list)
    pattern_summary = build_pattern_summary(acc_list)
    overall_summary = build_overall_summary(acc_list)
    confidence_log = build_confidence_log(fc_list)
    detail_df = build_detail(acc_list)
    daily_df = build_daily_detail(day_list)

    return fc_list, acc_list, sku_summary, pattern_summary, overall_summary, \
        confidence_log, detail_df, daily_df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 9 full pipeline simulation with live DB and learning loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="results", help="Output directory (default: results/)")
    parser.add_argument("--min-train", type=int, default=90,
                        help="Minimum training days before first eval (default: 90)")
    parser.add_argument("--eval-every", type=int, default=30,
                        help="Days between walk-forward evaluation points (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-SKU progress")
    # Column overrides — only needed when auto-detection fails
    parser.add_argument("--col-sku", default=None, metavar="COL",
                        help="Override auto-detected SKU column name")
    parser.add_argument("--col-date", default=None, metavar="COL",
                        help="Override auto-detected date column name")
    parser.add_argument("--col-qty", default=None, metavar="COL",
                        help="Override auto-detected quantity column name")
    parser.add_argument("--col-pattern", default=None, metavar="COL",
                        help="Override auto-detected pattern label column name")
    parser.add_argument("--default-pattern", default=None, metavar="LABEL",
                        help="Pattern label to assign when column is absent "
                             "(stable/trending/seasonal/intermittent/cold_start)")
    parser.add_argument("--tenant-id", default="harness-tenant", metavar="ID",
                        help="Tenant ID — plain name or UUID (default: harness-tenant)")
    parser.add_argument("--tenant-maturity", default="new",
                        choices=["new", "developing", "established"],
                        help="Tenant maturity for exploit threshold (default: new)")
    parser.add_argument("--max-history", type=int, default=None, metavar="N",
                        help="Cap demand history to the most recent N days per SKU")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\nLoading CSV: {args.csv}")
    try:
        df = load_csv(
            args.csv,
            col_sku=args.col_sku,
            col_date=args.col_date,
            col_qty=args.col_qty,
            col_pattern=args.col_pattern,
            default_pattern=args.default_pattern,
            max_history_days=args.max_history,
        )
    except Exception as csv_exc:
        print(f"ERROR loading CSV: {csv_exc}")
        sys.exit(1)

    # Coerce sku_id to a valid UUID for the same reason as tenant_id: the DB
    # schema declares sku_id UUID. Plain CSV identifiers like "TEST-STB-001"
    # cause psycopg2 DataError on any INSERT. We map each original value to a
    # stable deterministic UUID via uuid5 so the mapping is consistent across
    # runs and the original label is preserved in a separate column for display.
    _SKU_NAMESPACE = uuid.UUID("a2b3c4d5-0000-4000-8000-000000000002")

    def _sku_to_uuid(raw: str) -> str:
        try:
            return str(uuid.UUID(raw))  # already a valid UUID — pass through
        except (ValueError, AttributeError):
            return str(uuid.uuid5(_SKU_NAMESPACE, str(raw)))

    df["sku_id_original"] = df["sku_id"].astype(str)  # keep display label
    df["sku_id"] = df["sku_id"].astype(str).map(_sku_to_uuid)

    sku_ids = df["sku_id"].unique().tolist()
    sku_label_map = dict(zip(df["sku_id"], df["sku_id_original"]))  # UUID  -> original label
    sku_uuid_map = dict(zip(df["sku_id_original"], df["sku_id"]))  # label -> UUID

    print(f"  {len(df)} rows | {len(sku_ids)} SKUs | "
          f"date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Pattern distribution: {dict(df.groupby('pattern_label')['sku_id_original'].nunique())}")

    # The DB schema stores tenant_id as UUID. Coerce any plain string (e.g.
    # "harness-tenant") to a stable, deterministic UUID via uuid5 so the DB
    # never sees an invalid type. A value that is already a valid UUID passes
    # through unchanged, preserving backward-compatibility with callers that
    # supply a real UUID via --tenant-id.
    _HARNESS_NAMESPACE = uuid.UUID("b1d2e3f4-0000-4000-8000-000000000001")
    # Print the resolved UUID so it is visible in run output and can be used
    # directly with --tenant-id on subsequent runs to re-use the same tenant.
    try:
        tenant_id = str(uuid.UUID(args.tenant_id))  # already a valid UUID
    except ValueError:
        tenant_id = str(uuid.uuid5(_HARNESS_NAMESPACE, args.tenant_id))

    # ── DB connection — required for learning loop ─────────────────────
    # pg_conn() is a @contextmanager generator. It must stay inside a single
    # `with` block for the entire duration of the run — the connection is live
    # only while that block is executing. Any attempt to call __enter__ /
    # __exit__ manually risks the generator being GC'd and the connection
    # closed between calls.
    print(f"  Tenant UUID: {tenant_id}  (pass as --tenant-id to reuse)")
    thompson_state: dict = {}
    fingerprint_cache: dict = {}
    harness_run_id = str(uuid.uuid4())

    def _savepoint(conn, sql: str) -> None:
        """Execute a savepoint/rollback/release command on a psycopg2 connection."""
        with conn.cursor() as _cur:
            _cur.execute(sql)

    def _run_with_db(_conn):
        """All DB-dependent work, called from inside `with pg_conn() as db:`.
        
        Each DB operation runs inside its own savepoint so a failure in one
        (e.g. seed_tenant_params on a fresh schema) does not abort the whole
        transaction and block subsequent reads.
        """
        nonlocal thompson_state, fingerprint_cache

        print(f"  DB connected — tenant={tenant_id}  harness_run={harness_run_id[:8]}...")

        # ── Tenant params (savepoint-guarded) ──────────────────────────
        try:
            _savepoint(_conn, "SAVEPOINT sp_params")
            _loaded_params = _seed_and_load_params(_conn, tenant_id, tenant_maturity=args.tenant_maturity)
            _savepoint(_conn, "RELEASE SAVEPOINT sp_params")
            print("  Tenant params loaded from DB")
        except Exception as params_exc:
            try:
                _savepoint(_conn, "ROLLBACK TO SAVEPOINT sp_params")
                _savepoint(_conn, "RELEASE SAVEPOINT sp_params")
            except Exception:
                pass
            print(f"  WARNING: params load failed ({params_exc}) — using defaults")
            _loaded_params = TenantParams(
                tenant_id,
                {name: val for name, val in TENANT_LEARNING_PARAMS_DEFAULTS},
            )

        # ── Load Thompson state and fingerprints (savepoint-guarded) ───
        try:
            _savepoint(_conn, "SAVEPOINT sp_thompson")
            thompson_state = _load_thompson_from_db(_conn, tenant_id)
            fingerprint_cache = _load_fingerprints_from_db(_conn, tenant_id)
            _savepoint(_conn, "RELEASE SAVEPOINT sp_thompson")
            print(f"  DB loaded: {sum(len(v) for v in thompson_state.values())} Thompson configs, "
                  f"{len(fingerprint_cache)} fingerprints")
        except Exception as thompson_exc:
            try:
                _savepoint(_conn, "ROLLBACK TO SAVEPOINT sp_thompson")
                _savepoint(_conn, "RELEASE SAVEPOINT sp_thompson")
            except Exception:
                pass
            print(f"  WARNING: DB load failed ({thompson_exc})")

        return _loaded_params

    def _save_with_db(_conn, _all_forecasts, _all_accuracy):
        """DB saves + learning loop, called from inside the same `with pg_conn()` block.

        No savepoint wraps these steps: each save function calls db.commit()
        internally, which would destroy any savepoint created before it.
        Each step is independently committed; failures are logged and the
        function returns (None, {}) so the caller degrades gracefully.
        """
        save_agg_stats = None
        save_updater_result: dict = {}

        # 1. Save Thompson state
        try:
            _save_thompson_to_db(_conn, tenant_id, thompson_state, sku_uuid_map=sku_uuid_map)
            print(f"\n  Thompson saved: {sum(len(v) for v in thompson_state.values())} configs")
        except Exception as exc:
            print(f"  WARNING: Thompson save failed ({exc})")

        # 2. Save fingerprints
        try:
            from forecasting.fingerprint import compute_fingerprint, classify_tier
            new_fps: dict = {}
            for sid in sku_ids:
                sid_df = df[df["sku_id"] == sid]
                sid_ctx = contexts[str(sid)]
                sales_last_30d = sid_df["qty"].tolist()[-30:]
                oos_pct = float(sid_df["oos_pct"].iloc[0])
                demand_total = float(sum(sales_last_30d))
                fp = compute_fingerprint(
                    str(sid), sales_last_30d, sid_ctx.pattern_label,
                    oos_pct, sid_ctx.lifecycle_stage,
                )
                tier = classify_tier(
                    str(sid), fp, fingerprint_cache,
                    current_pattern_label=sid_ctx.pattern_label,
                    current_demand_total=demand_total,
                )
                new_fps[str(sid)] = {
                    "fingerprint": fp,
                    "tier": tier,
                    "pattern_label": sid_ctx.pattern_label,
                    "demand_total": demand_total,
                }
            _save_fingerprints_to_db(_conn, tenant_id, new_fps, sku_uuid_map=sku_uuid_map)
            print(f"  Fingerprints saved: {len(new_fps)}")
        except Exception as exc:
            print(f"  WARNING: Fingerprint save failed ({exc})")

        # 3. Write forecasts to stage9.forecasts (primary output — Stage 10 reads this)
        try:
            forecasts_written = _write_forecasts_to_db(
                _conn, tenant_id, harness_run_id, _all_forecasts,
                sku_uuid_map=sku_uuid_map,
            )
            print(f"  Forecasts written: {forecasts_written} rows to stage9.forecasts")
        except Exception as exc:
            print(f"  WARNING: Forecasts write failed ({exc})")

        # 4. Write forecast outcomes so learning loop has data
        try:
            outcomes_written = _write_forecast_outcomes(
                _conn, tenant_id, harness_run_id, _all_accuracy, _all_forecasts,
                sku_uuid_map=sku_uuid_map,
            )
            print(f"  Forecast outcomes written: {outcomes_written} rows (run={harness_run_id[:8]}...)")
        except Exception as exc:
            print(f"  WARNING: Forecast outcomes write failed ({exc})")

        # 5. Run learning loop (ModelPerformanceAggregator + LearningParamsUpdater)
        try:
            print("\nRunning learning loop ...")
            save_agg_stats, save_updater_result = _run_learning_loop(_conn, tenant_id)
        except Exception as exc:
            print(f"  WARNING: Learning loop failed ({exc})")

        # 6. Write audit/decision tables
        try:
            _write_audit_tables(
                _conn, tenant_id, harness_run_id,
                _all_forecasts, _all_accuracy,
                contexts, thompson_state, exploit_threshold_param, params,
                sku_uuid_map=sku_uuid_map,
            )
            print("  Audit tables written: model_initialization_s9, "
                  "hyperparameter_decisions, backtest_decisions, "
                  "stage9_sku_execution_log, adaptive_quantile_state, "
                  "stage9_self_assessment, agent_state_log_s9")
        except Exception as exc:
            print(f"  WARNING: Audit tables write failed ({exc})")

        return save_agg_stats, save_updater_result

    # ── DB connection (mandatory) ─────────────────────────────────────────
    agg_stats = None
    updater_result: dict = {}

    from infrastructure.db import pg_conn
    with pg_conn() as db:
        # All DB work — params load, Thompson load, walk-forward, saves,
        # and learning loop — runs inside a single connection.
        params = _run_with_db(db)

        # ── Build Stage 8 contexts (needs params) ─────────────────────
        contexts = build_sku_contexts(df, params)
        guarded = sum(1 for c in contexts.values() if c.insufficient_seasonal_history)
        if guarded:
            print(f"  Seasonal guard: {guarded} SKU(s) insufficient history for Prophet "
                  f"(obs_days < {int(params.get(Param.MIN_SEASONAL_OBS_DAYS))})")

        exploit_threshold_param = _MATURITY_TO_EXPLOIT_PARAM.get(
            args.tenant_maturity, Param.EXPLOIT_THRESHOLD_NEW,
        )

        # ── Walk-forward + reports ─────────────────────────────────────
        from infrastructure.batch_writer import BatchWriter as _BW
        _bw = _BW(db)
        try:
            (all_forecasts, all_accuracy,
             sku_summary, pattern_summary, overall_summary,
             confidence_log, detail_df, daily_df) = _run_evaluation_block(
                params, contexts, sku_ids, sku_label_map, df,
                args, thompson_state, exploit_threshold_param,
                batch_writer=_bw,
                tenant_id=tenant_id,
                run_id=harness_run_id,
            )
            _bw.flush()
        except Exception as bw_exc:
            print(f"  WARNING: BatchWriter flush failed ({bw_exc}) — "
                  f"feature_decisions_s9 rows not written")
            try:
                db.rollback()
            except Exception:
                pass
            (all_forecasts, all_accuracy,
             sku_summary, pattern_summary, overall_summary,
             confidence_log, detail_df, daily_df) = _run_evaluation_block(
                params, contexts, sku_ids, sku_label_map, df,
                args, thompson_state, exploit_threshold_param,
            )

        # ── DB saves + learning loop ───────────────────────────────────
        agg_stats, updater_result = _save_with_db(db, all_forecasts, all_accuracy)

    # ── Console report ─────────────────────────────────────────────────
    print_console_report(overall_summary, pattern_summary, sku_summary, all_forecasts)
    if agg_stats is not None:
        print_learning_loop_report(agg_stats, updater_result)

    # ── Write outputs ──────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "summary.csv": sku_summary,
        "pattern_summary.csv": pattern_summary,
        "overall_summary.csv": overall_summary,
        "forecast_detail.csv": detail_df,
        "confidence_log.csv": confidence_log,
        "daily_7d.csv": daily_df,
    }
    written = []
    for fname, data in files.items():
        if not data.empty:
            path = out_dir / fname
            data.to_csv(path, index=False)
            written.append(str(path))

    if written:
        print(f"Output files written to {args.output}/:")
        for w in written:
            print(f"  {w}")
    print()


# ═════════════════════════════════════════════════════════════════════════
# Quick-run — edit CSV paths here and run:  python tests/stage9_harness.py
# ═════════════════════════════════════════════════════════════════════════

def quick_run():
    """
    Hardcoded entry point — change CSV_PATH (and optionally OUTPUT_DIR)
    then run:  python tests/stage9_harness.py

    All other settings are pinned to sensible defaults. Use the CLI
    (python -m tests.stage9_harness --help) for full control.
    Requires DB_* environment variables to be set.
    """
    import sys as _sys
    import os as _os

    _here = _os.path.dirname(_os.path.abspath(__file__))

    # ── Edit these ─────────────────────────────────────────────────────
    csv_path = _os.path.join(_here, "stage8_inputs", "sku_stable_s8.csv")  # <- change to your CSV
    output_dir = _os.path.join(_here, "..", "results")
    # ── Optional overrides (leave as-is for defaults) ──────────────────
    tenant_id = "harness-tenant"
    tenant_maturity = "new"  # new | developing | established
    min_train = 90
    eval_every = 30
    verbose = False
    # ───────────────────────────────────────────────────────────────────

    argv = [
        "--csv", csv_path,
        "--output", output_dir,
        "--tenant-id", tenant_id,
        "--tenant-maturity", tenant_maturity,
        "--min-train", str(min_train),
        "--eval-every", str(eval_every),
    ]
    if verbose:
        argv.append("--verbose")

    _sys.argv = [_sys.argv[0]] + argv
    main()


if __name__ == "__main__":
    quick_run()
