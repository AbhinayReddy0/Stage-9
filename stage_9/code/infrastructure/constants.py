"""
constants.py — Stage 9 Forecasting Agent
=========================================
Single source of truth for every STRUCTURAL constant in Stage 9.

What belongs here:
  - Fixed string labels (pattern names, model names, status values, table names, etc.)
  - Fixed structural numbers that are hard algorithm decisions, not per-customer thresholds
    (retry counts, pool sizes, timeout limits, algorithm-specific caps)
  - Lookup maps derived purely from other constants in this file

What does NOT belong here:
  - Starting values for tenant_learning_params — those live in the DB seed only
    and are read at runtime via TenantParams.get(param_name)
  - Documentation dicts that no code path reads
  - Derivative mappings where code always does the lookup directly

Every threshold that influences forecasting behaviour at runtime must be
read from tenant_learning_params via TenantParams.get(), never from this file.

Source: Stage 9 Integrated Reference Guide v3.0, April 2026.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# 1. FORECAST HORIZONS
# Used by: Sub-Stage 9.5 (generate all 8 forecasts), Sub-Stage 9.6 (size
#          distribution), preloader.py (FORECAST_COLUMN_MAP lookup),
#          outcome_collector.py (batch job iterates all 8),
#          Stage 10 (nearest-ceiling horizon selection)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed forever. All 8 values exactly as listed.
# 14 is not 15. 150 is always present. Never add, remove, or change any element.

HORIZONS: list[int] = [7, 14, 30, 60, 90, 150, 180, 365]

# Named aliases — use these instead of bare ints when referencing a specific horizon
HORIZON_7D = 7
HORIZON_14D = 14
HORIZON_30D = 30
HORIZON_60D = 60
HORIZON_90D = 90
HORIZON_150D = 150
HORIZON_180D = 180
HORIZON_365D = 365

# Maps horizon int → forecasts table column name (column-per-horizon design)
# Used by: Sub-Stage 9.5 (write), Stage 10 (read), outcome_collector.py (read)
FORECAST_COLUMN_MAP: dict[int, str] = {
    7: "forecast_7d",
    14: "forecast_14d",
    30: "forecast_30d",
    60: "forecast_60d",
    90: "forecast_90d",
    150: "forecast_150d",
    180: "forecast_180d",
    365: "forecast_365d",
}

# Minimum days of history before the 365-day horizon is considered reliable
# Used by: Sub-Stage 9.1 (effective_max_horizon capping)
MIN_HISTORY_FOR_ANNUAL_FORECAST: int = 365


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEMAND PATTERNS
# Used by: Sub-Stage 9.1 (model assignment + quantile param lookup),
#          Sub-Stage 9.4 (pattern_feedback write),
#          Sub-Stage 9.5 (confidence base lookup via PATTERN_BASE_CONFIDENCE_PARAM),
#          preloader.py (PRELOADING — tier classification)
# ══════════════════════════════════════════════════════════════════════════════
# Labels written by Stage 8 into pattern_history.pattern_label.

class Pattern:
    COLD_START = "cold_start"
    INTERMITTENT = "intermittent"
    SEASONAL = "seasonal"
    TRENDING = "trending"
    STABLE = "stable"


ALL_PATTERNS: list[str] = [
    Pattern.COLD_START,
    Pattern.INTERMITTENT,
    Pattern.SEASONAL,
    Pattern.TRENDING,
    Pattern.STABLE,
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL ASSIGNMENTS
# Used by: Sub-Stage 9.1 (assign_model — locked, not configurable),
#          executor.py (PLANNING state — pool routing),
#          models/*.py (model constructors reference these strings),
#          forecasts.assigned_model column (DB value)
# ══════════════════════════════════════════════════════════════════════════════
# Locked to demand pattern. Based on decades of academic research.
# String values must match exactly what is stored in forecasts.assigned_model.

class Model:
    NAIVE = "naive_forecast"  # cold_start
    CROSTON = "croston"  # intermittent
    PROPHET = "prophet"  # seasonal
    HOLTS_LINEAR = "holts_linear_trend"  # trending
    SES = "simple_exponential_smoothing"  # stable


# Primary assignment — pattern → model
# Used by: Sub-Stage 9.1 (model initialisation)
PATTERN_MODEL_MAP: dict[str, str] = {
    Pattern.COLD_START: Model.NAIVE,
    Pattern.INTERMITTENT: Model.CROSTON,
    Pattern.SEASONAL: Model.PROPHET,
    Pattern.TRENDING: Model.HOLTS_LINEAR,
    Pattern.STABLE: Model.SES,
}

# Pool routing — Prophet is CPU-heavy (PyTorch), uses separate OS processes
# Used by: executor.py (PLANNING → ACTING state — splits SKUs into 2 concurrent pools)
PROCESS_POOL_MODELS: frozenset[str] = frozenset([
    Model.PROPHET,
])

THREAD_POOL_MODELS: frozenset[str] = frozenset([
    Model.NAIVE,
    Model.CROSTON,
    Model.HOLTS_LINEAR,
    Model.SES,
])

# Alias used by dual_pool.py, backtesting.py, forecasting.py
PROPHET_FAMILY: frozenset[str] = PROCESS_POOL_MODELS


# ══════════════════════════════════════════════════════════════════════════════
# 4. QUANTILE KEYS
# Used by: Sub-Stage 9.5 (write all 4 into forecast JSONB),
#          Sub-Stage 9.1 (selected_quantile selection),
#          Stage 10 (extract the selected quantile value by key name)
# ══════════════════════════════════════════════════════════════════════════════
# Keys stored inside every forecast JSONB: {"mean": X, "p50": Y, "p80": Z, "p90": W}

class Quantile:
    MEAN = "mean"  # point forecast — display only, never used for ordering
    P50 = "p50"  # 50th percentile
    P80 = "p80"  # 80th percentile — used for trending, stable products
    P90 = "p90"  # 90th percentile — used for seasonal, intermittent, cold_start


ALL_QUANTILE_KEYS: list[str] = [Quantile.MEAN, Quantile.P50, Quantile.P80, Quantile.P90]

# Hard override — criticality_tier = 'A' always forces this quantile regardless of pattern
# Fixed design rule, not in tenant_learning_params
# Used by: Sub-Stage 9.1 (quantile override priority chain — first match wins)
CRITICALITY_A_QUANTILE: float = 0.99


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE TIER & FORECAST STATUS VALUES
# Used by: Sub-Stage 9.5 (write confidence_tier and status to forecasts table),
#          REPORTING state (set run.status),
#          Stage 10 (read confidence_tier and status to determine action)
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceTier:
    AUTO_PROCEED = "auto_proceed"  # confidence >= decision_gate_threshold (param)
    REVIEW_SUGGESTED = "review_suggested"  # review_suggested_threshold <= conf < gate
    REVIEW_REQUIRED = "review_required"  # review_required_threshold <= conf < suggested
    MANUAL_OVERRIDE = "manual_override"  # confidence < review_required_threshold (param)


class ForecastStatus:
    FORECASTED = "forecasted"  # Stage 10 auto-proceeds
    NEEDS_ACKNOWLEDGMENT = "needs_acknowledgment"  # user must Accept / Override / Skip
    WATCHLIST_REVIEW = "watchlist_review"  # Stage 8 flagged — always reviewed


# ══════════════════════════════════════════════════════════════════════════════
# 6. TENANT_LEARNING_PARAMS — PARAMETER NAMES
# Used by: ALL sub-stages via TenantParams.get(Param.PARAM_NAME)
#          Never use bare string literals for param names in application code.
# ══════════════════════════════════════════════════════════════════════════════

class Param:
    # Confidence base per pattern
    # Read by: Sub-Stage 9.5 Step 1 via PATTERN_BASE_CONFIDENCE_PARAM map
    CONFIDENCE_BASE_COLD_START = "confidence_base_cold_start"
    CONFIDENCE_BASE_INTERMITTENT = "confidence_base_intermittent"
    CONFIDENCE_BASE_SEASONAL = "confidence_base_seasonal"
    CONFIDENCE_BASE_TRENDING = "confidence_base_trending"
    CONFIDENCE_BASE_STABLE = "confidence_base_stable"

    # Quantile per pattern — evolve toward service_level_target as calibration accumulates
    # Read by: Sub-Stage 9.1 (selected_quantile — after override chain)
    QUANTILE_COLD_START = "quantile_cold_start"
    QUANTILE_INTERMITTENT = "quantile_intermittent"
    QUANTILE_SEASONAL = "quantile_seasonal"
    QUANTILE_TRENDING = "quantile_trending"
    QUANTILE_STABLE = "quantile_stable"

    # Decision gates — threshold values for confidence tier boundaries
    # Read by: Sub-Stage 9.5 (status determination)
    SERVICE_LEVEL_TARGET = "service_level_target"
    DECISION_GATE_THRESHOLD = "decision_gate_threshold"
    REVIEW_SUGGESTED_THRESHOLD = "review_suggested_threshold"
    REVIEW_REQUIRED_THRESHOLD = "review_required_threshold"

    # Confidence formula multipliers
    # Read by: Sub-Stage 9.5 (Steps 1, 5 — clamp and penalty)
    EXCEPTION_PENALTY = "exception_penalty"
    CONFIDENCE_FLOOR = "confidence_floor"
    CONFIDENCE_CEILING = "confidence_ceiling"
    STRUCTURAL_BREAK_CONFIDENCE_PENALTY = "structural_break_confidence_penalty"
    OVERCONFIDENCE_THRESHOLD = "overconfidence_threshold"

    # Stage 8 uncertainty inheritance thresholds (Tech Context Part 7, Step 3)
    # Read by: Sub-Stage 9.5 (Step 3 — if Stage 8 confidence low → × 0.92 or × 0.90)
    STAGE8_PENALTY_THRESHOLD = "stage8_penalty_threshold"  # composite_confidence below this → × 0.92
    STAGE8_CONFIDENCE_THRESHOLD = "stage8_confidence_threshold"  # pattern_confidence below this → × 0.90

    # Forecast reasonableness bounds (Tech Context Part 5, Sub-Stage 9.5)
    # Read by: Sub-Stage 9.5 (ACT — flag implausible forecasts before writing)
    # daily_forecast > rolling_90d_avg × MAX → flag unusually_high + needs_acknowledgment
    # daily_forecast < rolling_90d_avg × MIN → flag unusually_low
    MAX_FORECAST_VS_BASELINE = "max_forecast_vs_baseline"  # upper bound: 5.0× rolling baseline
    MIN_FORECAST_VS_BASELINE = "min_forecast_vs_baseline"  # lower bound: 0.10× rolling baseline

    # Ordering buffer added on top of forecast
    # Read by: Sub-Stage 9.1 (passed forward to Stage 10 via forecasts table)
    SAFETY_STOCK_FACTOR = "safety_stock_factor"

    # Learning rate — controls how fast params shift toward evidence each nightly update
    # Read by: LearningParamsUpdater batch job
    CALIBRATION_UPDATE_RATE = "calibration_update_rate"

    # Thompson Sampling — HP tuning budget and exploit mode thresholds
    # Read by: Sub-Stage 9.3 (thompson.py)
    THOMPSON_EXPLORATION_BUDGET = "thompson_exploration_budget"
    EXPLOIT_THRESHOLD_NEW = "exploit_threshold_new"
    EXPLOIT_THRESHOLD_DEVELOPING = "exploit_threshold_developing"
    EXPLOIT_THRESHOLD_ESTABLISHED = "exploit_threshold_established"
    THOMPSON_EXPLOIT_CONFIDENCE_THRESHOLD = "thompson_exploit_confidence_threshold"

    # Feature engineering filters
    # Read by: Sub-Stage 9.2
    FEATURE_RELIABILITY_FLOOR = "feature_reliability_floor"
    MAX_PROMO_MULTIPLIER = "max_promo_multiplier"

    # Clearance / markdown pipeline
    # Read by: Sub-Stage 9.1 (trigger check), pipelines/clearance.py
    PRICE_ELASTICITY_CLEARANCE = "price_elasticity_clearance"
    CLEARANCE_MARKDOWN_THRESHOLD = "clearance_markdown_threshold"

    # Backtest window selection
    # Read by: Sub-Stage 9.4
    DEFAULT_BACKTEST_WINDOW = "default_backtest_window"
    MIN_BACKTEST_WINDOW = "min_backtest_window"
    MAX_BACKTEST_WINDOW = "max_backtest_window"
    # obs_days thresholds for window-size overrides (select_backtest_window)
    BACKTEST_SHORT_OBS_THRESHOLD = "backtest_short_obs_threshold"   # default 60
    BACKTEST_EXPLOIT_OBS_THRESHOLD = "backtest_exploit_obs_threshold"  # default 180

    # Size curve learning rate
    # Read by: Sub-Stage 9.6
    SIZE_CURVE_SMOOTHING_ALPHA = "size_curve_smoothing_alpha"

    # Warm-start / similarity registry
    # Read by: SimilarityRegistryUpdater batch job, pipelines/category_comps.py
    WARM_START_MAX_MAPE = "warm_start_max_mape"
    WARM_START_MIN_COMPS = "warm_start_min_comps"

    # Seasonal model guard — minimum obs_days before Prophet is used
    # Read by: Sub-Stage 9.1 (Decision 1 — override seasonal→Holt when insufficient)
    MIN_SEASONAL_OBS_DAYS = "min_seasonal_obs_days"

    # Execution mode decision gate
    # Read by: agent.py (decide FULL vs MICRO_UPDATE on Sync Now)
    MICRO_UPDATE_THRESHOLD_HOURS = "micro_update_threshold_hours"

    # PELT structural break detection sensitivity
    # Read by: Sub-Stage 9.4 (ruptures PELT penalty parameter)
    STRUCTURAL_BREAK_SENSITIVITY = "structural_break_sensitivity"

    # Confidence formula multipliers — read by Sub-Stage 9.5 compute_confidence()
    # and reasonableness_check(). All sourced from tenant_learning_params so they
    # evolve per customer as evidence accumulates.
    MAPE_CAP_IN_CONFIDENCE = "mape_cap_in_confidence"          # cap applied before MAPE term
    OVERCONFIDENCE_MULT = "overconfidence_mult"                # historically over-confident → × 0.90
    UNDERCONFIDENCE_MULT = "underconfidence_mult"              # historically under-confident → × 1.10
    STAGE8_PENALTY_MULT = "stage8_penalty_mult"                # Stage 8 uncertainty penalty
    INSUFFICIENT_POST_BREAK_MULT = "insufficient_post_break_mult"   # steeper break penalty
    FORECAST_UNUSUALLY_HIGH_MULT = "forecast_unusually_high_mult"   # unusually high flag mult
    FORECAST_UNUSUALLY_LOW_MULT = "forecast_unusually_low_mult"     # unusually low flag mult
    RISK_LOW_MIN = "risk_low_min"                              # confidence ≥ this → low risk
    RISK_MEDIUM_MIN = "risk_medium_min"                        # confidence ≥ this → medium risk

    # Model performance aggregator
    MODEL_PERFORMANCE_STABLE_BAND = "model_performance_stable_band"   # MAPE delta band for 'stable' trend

    # Channel split and learning calibration gates (tenant-tunable)
    # Read by: acting.py (channel split decision), LearningParamsUpdater (evidence gates)
    CHANNEL_SPLIT_CONFIDENCE_THRESHOLD = "channel_split_confidence_threshold"
    MIN_LEARNING_EVIDENCE_COUNT = "min_learning_evidence_count"   # min outcomes before param update
    QUANTILE_CALIBRATION_STEP = "quantile_calibration_step"       # binary-search step per nightly cycle


# Convenience maps — pattern → param name for confidence base and quantile
# Used by: Sub-Stage 9.5 (Step 1), Sub-Stage 9.1
PATTERN_BASE_CONFIDENCE_PARAM: dict[str, str] = {
    Pattern.COLD_START: Param.CONFIDENCE_BASE_COLD_START,
    Pattern.INTERMITTENT: Param.CONFIDENCE_BASE_INTERMITTENT,
    Pattern.SEASONAL: Param.CONFIDENCE_BASE_SEASONAL,
    Pattern.TRENDING: Param.CONFIDENCE_BASE_TRENDING,
    Pattern.STABLE: Param.CONFIDENCE_BASE_STABLE,
}

PATTERN_QUANTILE_PARAM: dict[str, str] = {
    Pattern.COLD_START: Param.QUANTILE_COLD_START,
    Pattern.INTERMITTENT: Param.QUANTILE_INTERMITTENT,
    Pattern.SEASONAL: Param.QUANTILE_SEASONAL,
    Pattern.TRENDING: Param.QUANTILE_TRENDING,
    Pattern.STABLE: Param.QUANTILE_STABLE,
}

# Alias used by forecasting.py (original name from stage9 package)
CONFIDENCE_BASE_PARAM: dict[str, str] = PATTERN_BASE_CONFIDENCE_PARAM


# ══════════════════════════════════════════════════════════════════════════════
# 7. AGENT STATES
# Canonical enum: state_machine.AgentState (use enum members in application code).
# These plain-string constants are for DB CHECK constraints and monitoring queries
# that need the raw string values without importing the state machine.
# ══════════════════════════════════════════════════════════════════════════════

ALL_STATES: list[str] = [
    "IDLE", "PRELOADING", "PERCEIVING", "PLANNING", "ACTING",
    "LEARNING", "REPORTING", "COMPLETE", "FAILED",
]

# States from which an unrecoverable exception transitions to FAILED.
# LEARNING and REPORTING are intentionally excluded — VALID_TRANSITIONS has no
# FAILED edge from those states (see state_machine.py for rationale).
FAILABLE_STATES: frozenset[str] = frozenset([
    "PRELOADING", "PERCEIVING", "PLANNING", "ACTING",
])


# ══════════════════════════════════════════════════════════════════════════════
# 8. EXECUTION MODES
# Used by: agent.py (decide full vs micro at run start),
#          forecasts.execution_mode column (stored per run)
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionMode:
    FULL = "full"  # complete pipeline — all sub-stages, all learning
    MICRO_UPDATE = "micro_update"  # lightweight — no model retraining, < 15s


# ══════════════════════════════════════════════════════════════════════════════
# 9. PROCESSING TIERS
# Used by: preloader.py (PRELOADING — classify each SKU by fingerprint comparison),
#          executor.py (ACTING — determines which sub-stages run for each SKU)
# ══════════════════════════════════════════════════════════════════════════════

class ProcessingTier:
    CACHE = "cache"  # fingerprint unchanged — skip all sub-stages (~70% daily)
    PARTIAL = "partial"  # small demand shift < 5% — refit but skip Thompson (~20%)
    FULL = "full"  # significant change or first run — all sub-stages (~10%)


# Threshold: 7-day demand average must shift by this fraction or more to go FULL not PARTIAL
# Used by: preloader.py (tier_classifier)
PARTIAL_TIER_DEMAND_SHIFT_THRESHOLD: float = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# 10. LIFECYCLE STAGES
# Used by: Sub-Stage 9.1 (quantile override + pipeline trigger),
#          Sub-Stage 9.3 (skip HP tuning when introduction → run CategoryComps),
#          pipelines/category_comps.py (trigger condition),
#          pipelines/clearance.py (trigger condition)
# ══════════════════════════════════════════════════════════════════════════════
# Written by Stage 8 into pattern_history.lifecycle_stage.

class LifecycleStage:
    INTRODUCTION = "introduction"  # < INTRODUCTION_MAX_DAYS history → CategoryComps
    GROWTH = "growth"  # weeks 4–12
    SATURATION = "saturation"  # week 12+
    CLEARANCE = "clearance"  # markdown applied → ClearanceAdjustment


# Maximum history days before a product exits introduction phase
# Used by: pipelines/category_comps.py
INTRODUCTION_MAX_DAYS: int = 28


# ══════════════════════════════════════════════════════════════════════════════
# 11. CRITICALITY TIERS
# Used by: Sub-Stage 9.1 (quantile override — first priority in override chain)
# ══════════════════════════════════════════════════════════════════════════════
# Set on canonical_sku.criticality_tier by the customer at onboarding.

class CriticalityTier:
    A = "A"  # critical — quantile overridden to CRITICALITY_A_QUANTILE (0.99)
    B = "B"  # important — standard quantile from params
    C = "C"  # routine — standard quantile (lower buffer acceptable)


# ══════════════════════════════════════════════════════════════════════════════
# 12. CROSS-AGENT SIGNAL TYPES & TTLs
# Used by: cross_agent.py (SignalEmitter.emit sets expires_at, SignalConsumer reads),
#          Sub-Stage 9.4 (emit forecast_accuracy after each backtest),
#          Sub-Stage 9.5 (emit forecast_risk after confidence computed),
#          Sub-Stage 9.1 (consume reorder_outcome + pattern_confidence),
#          LEARNING state (emit cross_sku_learning when SKU converges),
#          REPORTING state (emit model_health broadcast)
# ══════════════════════════════════════════════════════════════════════════════

class SignalType:
    FORECAST_ACCURACY = "forecast_accuracy"  # Stage 9 → Stage 8
    FORECAST_RISK = "forecast_risk"  # Stage 9 → Stage 10
    REORDER_OUTCOME = "reorder_outcome"  # Stage 10 → Stage 9
    CROSS_SKU_LEARNING = "cross_sku_learning"  # Stage 9 → Stage 9
    MODEL_HEALTH = "model_health"  # Stage 9 → all
    PATTERN_CONFIDENCE = "pattern_confidence"  # Stage 8 → Stage 9 (Stage 8.7 writes this)


# TTL in days — signal is ignored after this many days
SIGNAL_TTL_DAYS: dict[str, int] = {
    SignalType.FORECAST_ACCURACY: 90,
    SignalType.FORECAST_RISK: 90,
    SignalType.REORDER_OUTCOME: 90,
    SignalType.CROSS_SKU_LEARNING: 60,
    SignalType.MODEL_HEALTH: 30,
    # PATTERN_CONFIDENCE has no day TTL — consumed once (processed=TRUE) then ignored
}

# pattern_confidence is CONSUMED not peeked — use SignalConsumer.consume() for this type
PATTERN_CONFIDENCE_TTL_RUNS: int = 1


class Agent:
    STAGE_8 = "stage_8"
    STAGE_9 = "stage_9"
    STAGE_10 = "stage_10"
    ALL = "all"  # model_health broadcast target


# ══════════════════════════════════════════════════════════════════════════════
# 13. RUN STATUS VALUES
# Used by: REPORTING state (write run.status after run completes),
#          agent.py (read to detect trigger IN from Stage 8),
#          LangGraph (detects status change to start next stage)
# ══════════════════════════════════════════════════════════════════════════════

class RunStatus:
    DATA_READY = "data_ready"  # Stages 1–7 complete
    PATTERNS_DISCOVERED = "patterns_discovered"  # Stage 8 complete → Stage 9 trigger IN
    FORECASTED = "forecasted"  # Stage 9 complete → Stage 10 trigger IN
    NEEDS_ACKNOWLEDGMENT = "needs_acknowledgment"  # Stage 9: user must act before Stage 10
    REORDERS_READY = "reorders_ready"  # Stage 10 complete


# ══════════════════════════════════════════════════════════════════════════════
# 14. EXCEPTION FLAGS
# Used by: Sub-Stage 9.4 (four detectors — each appends a flag string to exception_flags),
#          Sub-Stage 9.5 (if "high_mape" in flags → needs_acknowledgment status),
#          Stage 10 (if "stockout" in flags → add extra safety buffer)
# ══════════════════════════════════════════════════════════════════════════════

class ExceptionFlag:
    STOCKOUT = "stockout"  # consecutive zero sales + zero inventory
    PROMO_SPIKE = "promo_spike"  # promo_weight > threshold AND demand > 2× rolling avg
    UNUSUAL_DROP = "unusual_drop"  # last 7d < 40% of prior 30d (no promo explanation)
    HIGH_VOLATILITY = "high_volatility"  # coefficient of variation > 1.5
    HIGH_MAPE = "high_mape"  # backtest MAPE above acceptable threshold


# Detector thresholds — fixed algorithm design decisions, not learned per-customer
# Used by: Sub-Stage 9.4 (exception_detection.py)
STOCKOUT_CONSECUTIVE_DAYS: int = 3  # consecutive zero-sale days required to flag stockout
PROMO_DEMAND_MULTIPLIER: float = 2.0  # demand > rolling_avg × this → promo_spike candidate
PROMO_ZSCORE_THRESHOLD: float = 3.0  # OR z-score > 3 → promo_spike (either condition triggers)
UNUSUAL_DROP_THRESHOLD: float = 0.40  # demand must be < 40% of 7d baseline (dropped > 60%)
UNUSUAL_DROP_CONSECUTIVE_PERIODS: int = 3  # must be 3 consecutive periods each below threshold
HIGH_VOLATILITY_CV: float = 1.0  # std / mean >= 1.0 → high_volatility (Master Spec §9.4)

# Aliases used by backtesting.py (different names from original stage9 package)
PROMO_SPIKE_RATIO: float = PROMO_DEMAND_MULTIPLIER
PROMO_SPIKE_Z: float = PROMO_ZSCORE_THRESHOLD
STOCKOUT_MIN_ZERO_STREAK: int = STOCKOUT_CONSECUTIVE_DAYS
UNUSUAL_DROP_KEEP_RATIO: float = UNUSUAL_DROP_THRESHOLD
UNUSUAL_DROP_MIN_STREAK: int = UNUSUAL_DROP_CONSECUTIVE_PERIODS

# Exception detection rolling-baseline window (7d) used by backtesting.py
ROLLING_BASELINE_DAYS: int = 7

# Reasonableness check constants (LOCKED — schema_notes.md / spec).
REASONABLE_HORIZON_DAYS: int = 30         # horizon evaluated by the reasonableness check
REASONABLENESS_ROLLING_BASELINE_DAYS: int = 90  # rolling demand baseline window

# High-MAPE flag threshold — fires when backtest_mape > 0.50 (same as proxy MAPE)
HIGH_MAPE_FLAG_THRESHOLD: float = 0.50

# Micro-update (CACHE tier) SES level-correction parameters
MICRO_UPDATE_SES_ALPHA:  float = 0.30
MICRO_UPDATE_SES_WINDOW: int   = 14
MICRO_UPDATE_SCALE_MIN:  float = 0.50  # never shrink forecast by more than 50%
MICRO_UPDATE_SCALE_MAX:  float = 2.00  # never grow forecast by more than 100%

# Fingerprint tier classification threshold
PARTIAL_TIER_CHANGE_PCT: float = 0.10  # demand within ±10% of cached → partial tier

# Ultra-sparse SKU threshold (below this obs count, backtest uses min_backtest_window)
ULTRA_SPARSE_OBS_THRESHOLD: int = 14

# Feature engineering promo-weighting rolling baseline window
PROMO_ROLLING_BASELINE_DAYS: int = 14

# Model performance aggregator stable band (mirrors tenant param seed value;
# used as function-parameter default for test-friendly callers that skip params)
MODEL_PERFORMANCE_STABLE_BAND: float = 0.02

# Minimum post-break series length to apply truncation branch in 9.4/9.5
MIN_POST_BREAK_LEN: int = 30

# horizon_days written to pattern_feedback (Principle 4 sacred write)
PATTERN_FEEDBACK_HORIZON_DAYS: int = 30


# ══════════════════════════════════════════════════════════════════════════════
# 15. PATTERN FEEDBACK WRITE CONTRACT
# Used by: Sub-Stage 9.4 (write_pattern_feedback — direct conn.execute(), never BatchWriter)
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationQuality:
    GOOD = "good"  # MAPE < 0.15
    ACCEPTABLE = "acceptable"  # MAPE 0.15–0.40
    POOR = "poor"  # MAPE > 0.40
    PROXY = "proxy"  # model failed — MAPE value is synthetic, not real


QUALITY_GOOD_MAPE_CEILING: float = 0.15
QUALITY_ACCEPTABLE_MAPE_CEILING: float = 0.40

# Aliases used by backtesting.py
QUALITY_GOOD_MAX: float = QUALITY_GOOD_MAPE_CEILING
QUALITY_ACCEPTABLE_MAX: float = QUALITY_ACCEPTABLE_MAPE_CEILING

# Written for failed/timeout SKUs — tells Stage 8 this row is not a real measurement
PATTERN_FEEDBACK_PROXY_MAPE: float = 0.50

# Retry configuration — 3 attempts, 100ms apart, before logging failure and continuing
PATTERN_FEEDBACK_MAX_RETRIES: int = 3
PATTERN_FEEDBACK_RETRY_WAIT_S: float = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# 16. THOMPSON SAMPLING
# Used by: Sub-Stage 9.3 (thompson.py — HP config selection and update)
# ══════════════════════════════════════════════════════════════════════════════

THOMPSON_ALPHA_INIT: float = 1.0  # Beta(α, β) starting α for untested configs
THOMPSON_BETA_INIT: float = 1.0  # Beta(α, β) starting β for untested configs
THOMPSON_SUCCESS_IMPROVEMENT: float = 0.02  # config is "success" if MAPE improves ≥ 2%
THOMPSON_EARLY_STOP_MAPE: float = 0.10  # stop HP search if validation MAPE < this
THOMPSON_VALIDATION_HOLDOUT_DAYS: int = 14  # days held out for HP validation

# Feature search config (Sub-Stage 9.2)
# Used by: sub_stage_92.py (additive feature search)
FEATURE_SEARCH_BUDGET: int = 4  # max feature configs to test
FEATURE_SEARCH_MIN_IMPROVEMENT: float = 0.02  # keep feature only if MAPE improves ≥ 2%
FEATURE_SEARCH_EARLY_STOP_MAPE: float = 0.08  # stop feature search early if MAPE < this


# ══════════════════════════════════════════════════════════════════════════════
# 17. CROSTON VARIANTS
# Used by: Sub-Stage 9.3 (Thompson Sampling selects best variant per SKU),
#          models/croston.py
# ══════════════════════════════════════════════════════════════════════════════

class CrostonVariant:
    CLASSIC = "classic"  # original Croston (1972) — slightly biased
    SBA = "sba"  # Syntetos-Boylan Approximation — bias-corrected
    TSB = "tsb"  # Teunter-Syntetos-Babai — handles demand obsolescence


ALL_CROSTON_VARIANTS: list[str] = [CrostonVariant.CLASSIC, CrostonVariant.SBA, CrostonVariant.TSB]

# ══════════════════════════════════════════════════════════════════════════════
# 18. NAIVE FORECAST LOOKBACK OPTIONS
# Used by: Sub-Stage 9.3 (Thompson Sampling selects best window per cold_start SKU),
#          models/naive.py
# ══════════════════════════════════════════════════════════════════════════════

NAIVE_LOOKBACK_OPTIONS: list[int] = [1, 7, 14]  # days — Thompson picks lowest-error window

# ══════════════════════════════════════════════════════════════════════════════
# 19. OOS DEMAND ADJUSTMENT
# Used by: preloader.py (PRELOADING — compute OOS factor before per-SKU loop),
#          sub_stage_92.py (apply OOS uplift to training series)
# ══════════════════════════════════════════════════════════════════════════════
# Formula: factor = min(1 + oos_pct × detection_confidence, OOS_ADJUSTMENT_MAX_FACTOR)
# Dampening by detection_confidence prevents poor inventory data from over-correcting.

OOS_ADJUSTMENT_MAX_FACTOR: float = 1.50  # never more than 50% uplift

# Minimum channel split confidence to use organic demand series instead of total
# Used by: preloader.py (PRELOADING — channel_demand_splits decision)
CHANNEL_SPLIT_MIN_CONFIDENCE: float = 0.50

# ══════════════════════════════════════════════════════════════════════════════
# 20. CLEARANCE ADJUSTMENT PIPELINE
# Used by: Sub-Stage 9.1 (trigger detection), pipelines/clearance.py
# ══════════════════════════════════════════════════════════════════════════════
# Triggers when: lifecycle_stage = 'clearance' OR discount_pct > threshold for N+ days
# Formula: adjusted_demand = baseline × (1 - discount_pct) ^ elasticity (param)

CLEARANCE_CONSECUTIVE_DAYS: int = 5  # days of discounting before pipeline activates
CLEARANCE_BASELINE_WINDOW_DAYS: int = 14  # days before markdown used as demand baseline

# ══════════════════════════════════════════════════════════════════════════════
# 21. CATEGORY COMPS — WARM-START PIPELINE
# Used by: Sub-Stage 9.3 (triggers when lifecycle_stage = introduction),
#          pipelines/category_comps.py
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_COMPS_MIN_COMPS: int = 3  # fewer than this → fall back to Naive
CATEGORY_COMPS_TRAJECTORY_DAYS: int = 60  # comp first-N-days used to build curve
CATEGORY_COMPS_SCALE_UP_PERCENTILE: float = 0.80  # day-3 > comp p80 → scale curve up 15%
CATEGORY_COMPS_SCALE_DOWN_PERCENTILE: float = 0.20  # day-3 < comp p20 → scale curve down 15%
CATEGORY_COMPS_SCALE_FACTOR: float = 0.15  # magnitude of up/down adjustment

TRAINING_SOURCE_CATEGORY_COMPS: str = "category_comps"  # written to hyperparameter_decisions
TRAINING_SOURCE_ACTUAL: str = "actual"

# ══════════════════════════════════════════════════════════════════════════════
# 22. STRUCTURAL BREAK DETECTION
# Used by: Sub-Stage 9.4 (only when portfolio_intelligence has an alert for the SKU)
# ══════════════════════════════════════════════════════════════════════════════

STRUCTURAL_BREAK_PELT_MODEL: str = "rbf"  # ruptures library algorithm
STRUCTURAL_BREAK_MAX_DATE_DELTA_DAYS: int = 14  # detected break must be ±14d of alert_date
STRUCTURAL_BREAK_CONFIDENCE_MULTIPLIER: float = 0.85  # confidence Step 5: × 0.85 when truncated


# ══════════════════════════════════════════════════════════════════════════════
# 23. EDGE CASE IDENTIFIERS & HANDLER CONSTANTS
# Used by: each edge case handler in models/*.py, sub_stage_9*.py, executor.py
#          Identifiers logged to stage9_sku_execution_log for debugging.
# ══════════════════════════════════════════════════════════════════════════════

class EdgeCase:
    E001_CROSTON_ALL_NONZERO = "E001"  # np.diff empty → fall back to SES
    E002_PROPHET_DEGENERATE = "E002"  # zero-variance series → add noise
    E003_PROCESSPOOL_TIMEOUT = "E003"  # worker killed → ensure conn.close() in finally
    E004_BOOTSTRAP_INSUFFICIENT = "E004"  # < 3 residuals → log-normal proxy quantiles
    E005_MODEL_TRAINING_FAILED = "E005"  # model.fit() raises → Naive fallback
    E006_B2B_FILTER_NO_DATA = "E006"  # weekday filter → 0 rows → disable B2B mode
    E007_CATEGORY_COMPS_INSUFFICIENT = "E007"  # < 3 comps found → Naive fallback
    E008_SIZE_CURVE_SUM_NOT_ONE = "E008"  # float drift → normalise size shares


# E001 — Croston: len(non_zero_intervals) must be >= 1 to proceed; 0 → fall back to SES
# Used by: models/croston.py
CROSTON_MIN_INTERVAL_COUNT: int = 1

# E002 — Prophet: inject noise when series has zero variance
# Used by: prophet_model.py
PROPHET_DEGENERATE_STD_THRESHOLD: float = 0.01  # series.std() < this → degenerate
PROPHET_NOISE_SCALE: float = 0.001  # noise = 0.001 × series.mean()
PROPHET_NOISE_ADDED_FLAG: str = "e002_noise_added"  # appended to exception_flags

# E003 — Per-SKU hard timeout limits
# Used by: executor.py (DualPoolExecutor — both pools)
PROCESS_POOL_TIMEOUT_SECONDS: int = 120  # Prophet per-SKU limit
THREAD_POOL_TIMEOUT_SECONDS: int = 30  # all other models per-SKU limit

# E004 — Degenerate bootstrap: need at least 3 residuals for valid quantile generation
# Used by: sub_stage_95.py (quantile generation)
BOOTSTRAP_MIN_RESIDUALS: int = 3

# E005 — Model failure: retry once with default HP before Naive fallback
# Used by: executor.py (process_sku_safely wrapper)
MODEL_FAILURE_RETRY_ATTEMPTS: int = 1

# E006 — B2B filter: weekend_zero_ratio threshold and fallback flag
# Used by: sub_stage_92.py (feature engineering)
B2B_WEEKEND_ZERO_RATIO_THRESHOLD: float = 0.60
B2B_DISABLED_FLAG: str = "b2b_mode_disabled"  # Master Spec §9 E006 exact string

# E007 — CategoryComps insufficient comps fallback flag
# Used by: pipelines/category_comps.py
INSUFFICIENT_COMPS_FLAG: str = "insufficient_comps_for_warm_start"

# E008 — Size curve normalisation tolerance
# Used by: sub_stage_96.py
SIZE_CURVE_SUM_TOLERANCE: float = 1e-6

# ══════════════════════════════════════════════════════════════════════════════
# 24. DUAL-POOL EXECUTION CONFIGURATION
# Used by: executor.py (PLANNING state — configure both pools before ACTING begins)
# ══════════════════════════════════════════════════════════════════════════════

PROCESS_POOL_WORKERS: int = 4   # Prophet — separate OS processes
THREAD_POOL_WORKERS: int = 16   # Naive, Croston, Holt, SES — shared memory, thread-safe

# Per-task and wall-clock timeouts
PROCESS_TIMEOUT: float = 120.0   # seconds per Prophet SKU before the watchdog fires
THREAD_TIMEOUT: float = 30.0     # seconds per lightweight-model SKU
OVERALL_TIMEOUT: float = 3600.0  # 1-hour wall-clock cap for the entire batch

# Submission chunk size — bounds in-flight Future count to N, not total SKU count
DEFAULT_CHUNK_SIZE: int = 10_000

# Subprocess DB connection identity — filtered on by orphan cleanup
APPLICATION_NAME: str = "stage9_subprocess"

# Idle connections older than this are terminated after the pool drains
ORPHAN_IDLE_MINUTES: int = 3

# Fallback confidence used when a worker times out and the caller did not
# supply a floor. Production callers must pass tenant_learning_params.confidence_floor.
DEFAULT_FALLBACK_CONFIDENCE: float = 0.30

# ProcessPoolExecutor tasks-per-child — 1 means each Prophet SKU gets a
# fresh process, so any leaked Stan daemon thread dies with the worker.
DEFAULT_MAX_TASKS_PER_CHILD: int = 1

# ══════════════════════════════════════════════════════════════════════════════
# 25. BATCH WRITER
# Used by: batch_writer.py (ACTING state — accumulates all writes except pattern_feedback)
# ══════════════════════════════════════════════════════════════════════════════

BATCH_WRITER_FLUSH_EVERY: int = 100  # flush accumulated writes every N SKUs processed

# ══════════════════════════════════════════════════════════════════════════════
# 26. DATA FINGERPRINT
# Used by: preloader.py (PRELOADING — compute SHA256 per SKU and compare to cache)
#          ALWAYS use json.dumps(..., sort_keys=True) — never omit sort_keys
# ══════════════════════════════════════════════════════════════════════════════

FINGERPRINT_ALGORITHM: str = "sha256"
FINGERPRINT_HASH_LENGTH: int = 64  # output length in hex characters
FINGERPRINT_LOOKBACK_DAYS: int = 7  # days of demand data included in hash input


# ══════════════════════════════════════════════════════════════════════════════
# 27. DAILY BATCH JOB IDENTIFIERS & SCHEDULE
# Used by: scheduler.py (configure cron), each batch job module
# ══════════════════════════════════════════════════════════════════════════════

class BatchJob:
    OUTCOME_COLLECTOR = "OutcomeCollector"  # 3:00 AM
    MODEL_PERFORMANCE_AGGREGATOR = "ModelPerformanceAggregator"  # 4:00 AM
    LEARNING_PARAMS_UPDATER = "LearningParamsUpdater"  # 4:30 AM
    SIMILARITY_REGISTRY_UPDATER = "SimilarityRegistryUpdater"  # 5:00 AM


BATCH_JOB_SCHEDULE: dict[str, str] = {
    BatchJob.OUTCOME_COLLECTOR: "03:00",
    BatchJob.MODEL_PERFORMANCE_AGGREGATOR: "04:00",
    BatchJob.LEARNING_PARAMS_UPDATER: "04:30",
    BatchJob.SIMILARITY_REGISTRY_UPDATER: "05:00",
}

# Rolling window for ModelPerformanceAggregator
# Used by: model_performance_aggregator.py
MODEL_PERFORMANCE_ROLLING_DAYS: int = 30

# ══════════════════════════════════════════════════════════════════════════════
# 28. REORDER BIAS FACTOR
# Used by: Sub-Stage 9.1 (read reorder_outcome signals → compute bias factor),
#          Sub-Stage 9.5 (Step 4 of confidence formula: × reorder_bias_factor)
# ══════════════════════════════════════════════════════════════════════════════
# Computed from Stage 10 reorder_outcome signals. Corrects for systematic
# under/over-ordering by adjusting Stage 9's effective forecast output.

REORDER_BIAS_FACTOR_DEFAULT: float = 1.00  # neutral — no bias correction
REORDER_BIAS_FACTOR_STOCKOUT: float = 1.10  # 2+ stockouts in last N signals → order 10% more
REORDER_BIAS_FACTOR_OVERSTOCK: float = 0.92  # overstock > 30% in last N signals → order 8% less
REORDER_OVERSTOCK_THRESHOLD: float = 0.30  # avg overstock_pct above this triggers reduction
REORDER_STOCKOUT_MIN_EVENTS: int = 2  # minimum stockout count to trigger increase
REORDER_SIGNAL_LOOKBACK: int = 5  # how many most-recent reorder_outcome signals to examine


# ══════════════════════════════════════════════════════════════════════════════
# 29. DATABASE TABLE NAMES
# Used by: ALL sub-stages, all batch jobs, cross_agent.py, preloader.py —
#          any code that issues a SQL query uses these, never bare string literals
# ══════════════════════════════════════════════════════════════════════════════

class Table:
    # ── Stage 9 owned: learning ──────────────────────────────────────────────
    TENANT_LEARNING_PARAMS = "tenant_learning_params"  # r: all sub-stages via TenantParams
    THOMPSON_SAMPLING_STATE = "thompson_sampling_state"  # r/w: Sub-Stage 9.3
    SKU_SIMILARITY_REGISTRY = "sku_similarity_registry"  # r: Sub-Stage 9.1 warm-start; w: SimilarityRegistryUpdater
    DATA_FINGERPRINT_CACHE = "data_fingerprint_cache"  # r/w: PRELOADING
    ADAPTIVE_QUANTILE_STATE = "adaptive_quantile_state"  # r: Sub-Stage 9.4 window; w: OutcomeCollector
    SIZE_CURVE_REGISTRY = "size_curve_registry"  # r/w: Sub-Stage 9.6
    FORECAST_OUTCOMES = "forecast_outcomes"  # w: OutcomeCollector; r: LearningParamsUpdater

    # ── Stage 9 owned: memory (decision audit trail) ─────────────────────────
    MODEL_INITIALIZATION_S9 = "model_initialization_s9"  # w: Sub-Stage 9.1; r: 9.2–9.6
    FEATURE_DECISIONS_S9 = "feature_decisions_s9"  # w: Sub-Stage 9.2; r: Sub-Stage 9.3
    HYPERPARAMETER_DECISIONS = "hyperparameter_decisions"  # w: Sub-Stage 9.3; r: Sub-Stage 9.3 (prior HP)
    BACKTEST_DECISIONS = "backtest_decisions"  # w: Sub-Stage 9.4; r: self_assessment.py

    # ── Stage 9 owned: primary output ────────────────────────────────────────
    FORECASTS = "forecasts"  # w: Sub-Stages 9.5+9.6; r: Stage 10 (primary)
    PATTERN_FEEDBACK = "pattern_feedback"  # w: Sub-Stage 9.4 (direct only); r: Stage 8

    # ── Stage 9 owned: cross-agent ───────────────────────────────────────────
    CROSS_AGENT_SIGNALS = "cross_agent_signals"  # r/w: Stage 8, 9, 10

    # ── Stage 9 owned: observability ─────────────────────────────────────────
    STAGE9_SELF_ASSESSMENT = "stage9_self_assessment"  # w: REPORTING state; r: dashboard
    MODEL_PERFORMANCE_S9 = "model_performance_s9"  # w: ModelPerformanceAggregator; r: self_assessment
    AGENT_STATE_LOG_S9 = "agent_state_log_s9"  # w: every state transition
    STAGE9_SKU_EXECUTION_LOG = "stage9_sku_execution_log"  # w: every SKU execution

    # ── Stage 8 owned — Stage 9 reads only ───────────────────────────────────
    PATTERN_HISTORY = "pattern_history"  # r: PRELOADING (primary input)
    SIGNAL_CONTEXT = "signal_context"  # r: PERCEIVING
    CHANNEL_DEMAND_SPLITS = "channel_demand_splits"  # r: PRELOADING (organic vs paid)
    OOS_IMPACT_ESTIMATES = "oos_impact_estimates"  # r: PRELOADING (OOS correction)
    FEATURE_DECISIONS_S8 = "feature_decisions"  # r: PRELOADING (reliability map)
    # Note: _S8 suffix is a Python-side disambiguator only; DB table name = "feature_decisions"
    PORTFOLIO_INTELLIGENCE = "portfolio_intelligence_reports"  # r: PRELOADING (break alerts)
    PROMO_DECISIONS = "promo_decisions"  # r: PRELOADING (promo weights)
    TENANT_THRESHOLDS_S8 = "tenant_thresholds"  # r: PRELOADING (confidence bounds)

    # ── Stages 1–7 owned — Stage 9 reads only ────────────────────────────────
    GOLDEN_TABLE = "golden_table"  # r: ACTING (preferred demand series)
    CANONICAL_SKU = "canonical_sku"  # r: PRELOADING (SKU metadata + V3 columns)
    CLEAN_ORDERS = "clean_orders"  # r: fallback when golden_table unavailable
    CLEAN_INVENTORY = "clean_inventory"  # r: Sub-Stage 9.4 (stockout detection)

    # ── Stage 10 owned — referenced for context ───────────────────────────────
    INBOUND_POS = "inbound_pos"  # r: Stage 10 net inventory calculation
    PO_RECOMMENDATIONS = "po_recommendations"  # w: Stage 10 reorder output

    # ── Orchestration ─────────────────────────────────────────────────────────
    RUNS = "runs"  # r/w: run.status transitions that LangGraph monitors
    PIPELINE_LOCKS = "pipeline_locks"  # r/w: agent.py (enforce max 1 concurrent run per tenant)


# ══════════════════════════════════════════════════════════════════════════════
# 30. SELF-ASSESSMENT THRESHOLDS
# Used by: self_assessment.py (REPORTING state — generate health report and
#          recommendations after every run)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed design thresholds — not per-customer, not in tenant_learning_params.

MODEL_DEGRADATION_MAPE_DELTA: float = 0.03  # MAPE > 30d avg + 3pp → degrading model flag
SELF_ASSESSMENT_FALLBACK_RATE_WARN: float = 0.10  # > 10% fallback → upstream data recommendation
SELF_ASSESSMENT_CACHE_HIT_RATE_WARN: float = 0.50  # < 50% cache hits → volatility / fingerprint flag
SELF_ASSESSMENT_SUCCESS_RATE_CRITICAL: float = 0.90  # < 90% success rate → critical alert

# ══════════════════════════════════════════════════════════════════════════════
# 31. NEURAL PROPHET / PROPHET — FIXED CONSTRUCTOR SETTINGS
# Used by: models/neural_prophet.py (set in model constructor — never True)
# ══════════════════════════════════════════════════════════════════════════════

NEURAL_PROPHET_DAILY_SEASONALITY: bool = False  # always False — daily overfits retail demand
PROPHET_DAILY_SEASONALITY: bool = False  # always False
NEURAL_PROPHET_MIN_HISTORY_DAYS: int = 90  # minimum days before NeuralProphet can fit
NEURALPROPHET_FORECAST_DAYS: int = 365  # single forward fit; horizons extracted from it


# ══════════════════════════════════════════════════════════════════════════════
# 32. PRODUCT LIFECYCLE TYPES
# Used by: Sub-Stage 9.1 (lifecycle routing decisions),
#          Sub-Stage 9.6 (seasonal_fashion → sell-through projection active)
# ══════════════════════════════════════════════════════════════════════════════

class ProductLifecycleType:
    NOS = "NOS"  # never out of season — basic replenishment
    SEASONAL_FASHION = "seasonal_fashion"  # has planned_end_date — sell-through applies


# ══════════════════════════════════════════════════════════════════════════════
# 33. CONFIDENCE FORMULA — FIXED STEP MULTIPLIERS
# Used by: Sub-Stage 9.5 (confidence engine Steps 1, 2, 3)
# ══════════════════════════════════════════════════════════════════════════════
# These are fixed algorithm design choices — not per-customer, not in params.

# Step 1: cap MAPE before applying to confidence formula
# Formula: base × (1 − min(MAPE, CONFIDENCE_MAPE_CAP)) × exception_penalty
CONFIDENCE_MAPE_CAP: float = 0.50  # MAPE capped at 0.50 — prevents confidence going negative

# Step 2: calibration adjustment — direction determined by adaptive_quantile_state
CONFIDENCE_CALIBRATION_OVER_PENALTY: float = 0.90  # over-confident historically → × 0.90
CONFIDENCE_CALIBRATION_UNDER_BONUS: float = 1.10  # under-confident historically → × 1.10

# Step 3: Stage 8 signal quality penalty
STAGE8_COMPOSITE_CONFIDENCE_PENALTY_THRESHOLD: float = 0.60  # below → apply multiplier
STAGE8_COMPOSITE_CONFIDENCE_MULTIPLIER: float = 0.92  # × 0.92 applied when Stage 8 confidence is low

# ══════════════════════════════════════════════════════════════════════════════
# 34. FORECAST REASONABLENESS CHECK
# Used by: Sub-Stage 9.5 (ACT phase — flag implausible forecasts before writing)
# ══════════════════════════════════════════════════════════════════════════════
# Upper/lower bounds are per-customer learnable params — read via:
#   params.get(Param.MAX_FORECAST_VS_BASELINE)  # starting 5.0
#   params.get(Param.MIN_FORECAST_VS_BASELINE)  # starting 0.10
# Do NOT use a hardcoded constant — these values live in tenant_learning_params
# so they evolve per customer as accuracy evidence accumulates.

FORECAST_REASONABLENESS_ROLLING_DAYS: int = 90  # fixed rolling window — structural, not learned

# ══════════════════════════════════════════════════════════════════════════════
# 35. SIZE CURVE
# Used by: Sub-Stage 9.6 (only when canonical_sku.parent_style_id is not null)
# ══════════════════════════════════════════════════════════════════════════════

SIZE_DISTRIBUTION_TRIGGER_ATTR: str = "parent_style_id"  # attribute checked to activate 9.6
SELL_THROUGH_TARGET: float = 0.85  # flag markdown need when projected < 85%
SELL_THROUGH_MAX: float = 1.0  # cap: min(1.0, forecast_until_end / inventory)


# ══════════════════════════════════════════════════════════════════════════════
# 37. LEARNING MODE
# Used by: Sub-Stage 9.1 (set learning_mode per SKU based on Thompson state),
#          Sub-Stage 9.3 (explore: test new configs; exploit: use best known config)
# ══════════════════════════════════════════════════════════════════════════════

class LearningMode:
    EXPLORE = "explore"  # early runs — test new HP configurations
    EXPLOIT = "exploit"  # converged runs — use known best configuration


# ══════════════════════════════════════════════════════════════════════════════
# 38. PORTFOLIO INTELLIGENCE ALERT TYPES
# Used by: Sub-Stage 9.4 (checks portfolio_intelligence_reports for these alert
#          types before deciding whether to run PELT structural break detection)
# ══════════════════════════════════════════════════════════════════════════════

class AlertType:
    MARKET_SHIFT = "market_shift"  # demand regime has changed
    CHANNEL_COUNT_CHANGED = "channel_count_changed"  # number of sales channels changed


ALL_ALERT_TYPES: list[str] = [AlertType.MARKET_SHIFT, AlertType.CHANNEL_COUNT_CHANGED]


# ══════════════════════════════════════════════════════════════════════════════
# 39. TENANT MATURITY LEVELS
# Used by: PLANNING state (determine which exploit_threshold param to read),
#          Sub-Stage 9.3 (explore vs exploit decision)
# ══════════════════════════════════════════════════════════════════════════════

class TenantMaturity:
    NEW = "new"  # < exploit_threshold_new (8) runs
    DEVELOPING = "developing"  # < exploit_threshold_developing (5) runs
    ESTABLISHED = "established"  # >= exploit_threshold_established (3) runs


# ══════════════════════════════════════════════════════════════════════════════
# 40. DATA DENSITY LEVELS
# Used by: Sub-Stage 9.4 (select_backtest_window — maps density → window size)
# ══════════════════════════════════════════════════════════════════════════════

class DataDensity:
    ULTRA_SPARSE = "ultra_sparse"  # observation_days < 14 → use min_backtest_window (param)
    SPARSE = "sparse"  # observation_days < 60 → max(min_window, days/divisor)
    NORMAL = "normal"  # standard window from default_backtest_window (param)


ULTRA_SPARSE_MAX_OBSERVATION_DAYS: int = 14  # below this → ULTRA_SPARSE
SPARSE_MAX_OBSERVATION_DAYS: int = 60  # below this → SPARSE
SPARSE_WINDOW_DIVISOR: int = 3  # sparse window = max(min_window, obs_days/3)

# Maximum demand history pulled from stage8.demand_history per run.
# Aligned with Stage 8's own 730-day lookback cap so Stage 9 never works
# with less history than Stage 8 used to derive the pattern label.
# Prophet needs ~2 full years for reliable annual seasonality; 730 days
# satisfies that requirement exactly.
MAX_DEMAND_HISTORY_DAYS: int = 730


# ══════════════════════════════════════════════════════════════════════════════
# 41. RECOMMENDATION STATUS VALUES
# Used by: Sub-Stage 9.1 (reads reorder_outcome.payload to determine what action
#          the user took — informs reorder_bias_factor adjustment direction)
# ══════════════════════════════════════════════════════════════════════════════
# Written by Stage 10/13 into po_recommendations.recommendation_status.

class RecommendationStatus:
    PENDING_ACK = "pending_ack"  # awaiting user action
    ACKNOWLEDGED = "acknowledged"  # user accepted Stage 9's forecast as-is
    OVERRIDDEN = "overridden"  # user provided manual demand value
    SKIPPED = "skipped"  # user excluded this SKU from this cycle


# ══════════════════════════════════════════════════════════════════════════════
# 42. RUN LOCK
# Used by: state_machine.py (acquire/release per-tenant Redis NX lock),
#          run_lock.py (RedisRunLock.acquire)
# ══════════════════════════════════════════════════════════════════════════════

LOCK_TTL_SECONDS: int = 14400  # 4 hours — must exceed max expected run duration
LOCK_KEY_TEMPLATE: str = "stage9_lock_{tenant_id}"  # e.g. "stage9_lock_acme-corp"


# ══════════════════════════════════════════════════════════════════════════════
# 43. MODEL ALGORITHM CONSTANTS
# Fixed design choices in individual model implementations — not learnable.
# Used by: holt.py (phi damping), croston.py (interval floor)
# ══════════════════════════════════════════════════════════════════════════════

HOLT_DAMPING_COEFFICIENT: float = 0.95  # phi: decelerates trend on 180/365-day horizons
CROSTON_INTERVAL_FLOOR: float = 1e-6    # prevents divide-by-zero in interval_smooth denominator


# ══════════════════════════════════════════════════════════════════════════════
# 44. BOOTSTRAP PARAMETERS
# Used by: bootstrap.py (bootstrap_quantiles default arguments)
# ══════════════════════════════════════════════════════════════════════════════

BOOTSTRAP_SAMPLE_COUNT: int = 1000  # resample count — calibrated for stable percentile estimates
BOOTSTRAP_SEED: int = 42            # RNG seed — reproducibility only, not security-sensitive


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRITY CHECK — runs automatically at import time
# Catches accidental mutations and missing coverage before any code runs.
# ══════════════════════════════════════════════════════════════════════════════

def _verify_constants() -> None:
    assert HORIZONS == [7, 14, 30, 60, 90, 150, 180, 365], \
        "HORIZONS must never be modified — all 8 values exactly as specified."
    assert len(HORIZONS) == 8, \
        "HORIZONS must always contain exactly 8 elements."
    assert set(FORECAST_COLUMN_MAP.keys()) == set(HORIZONS), \
        "FORECAST_COLUMN_MAP must have exactly one entry per horizon."
    assert set(PATTERN_MODEL_MAP.keys()) == set(ALL_PATTERNS), \
        "PATTERN_MODEL_MAP must cover all 5 patterns."
    assert set(PATTERN_BASE_CONFIDENCE_PARAM.keys()) == set(ALL_PATTERNS), \
        "PATTERN_BASE_CONFIDENCE_PARAM must cover all 5 patterns."
    assert set(PATTERN_QUANTILE_PARAM.keys()) == set(ALL_PATTERNS), \
        "PATTERN_QUANTILE_PARAM must cover all 5 patterns."
    assert PROCESS_POOL_MODELS & THREAD_POOL_MODELS == frozenset(), \
        "No model may appear in both executor pools."
    assert (PROCESS_POOL_MODELS | THREAD_POOL_MODELS) == (set(PATTERN_MODEL_MAP.values()) | {Model.PROPHET}), \
        "PROCESS_POOL_MODELS ∪ THREAD_POOL_MODELS must cover all model strings exactly."
    assert CONFIDENCE_MAPE_CAP == 0.50, \
        "MAPE cap in confidence Step 1 is always 0.50."
    assert PATTERN_FEEDBACK_PROXY_MAPE == 0.50, \
        "Proxy MAPE for failed SKUs is always 0.50."
    assert PATTERN_FEEDBACK_MAX_RETRIES == 3, \
        "pattern_feedback must retry exactly 3 times."
    assert OOS_ADJUSTMENT_MAX_FACTOR == 1.50, \
        "OOS adjustment capped at 1.50 (50% maximum uplift)."
    assert STRUCTURAL_BREAK_CONFIDENCE_MULTIPLIER == 0.85, \
        "Structural break confidence penalty is × 0.85."
    assert REORDER_BIAS_FACTOR_STOCKOUT == 1.10, \
        "2+ stockouts → reorder_bias_factor = 1.10."
    assert REORDER_BIAS_FACTOR_OVERSTOCK == 0.92, \
        "overstock > 30% → reorder_bias_factor = 0.92."
    assert CATEGORY_COMPS_MIN_COMPS == 3, \
        "CategoryComps requires minimum 3 comparable SKUs."
    assert BATCH_WRITER_FLUSH_EVERY == 100, \
        "BatchWriter flushes every 100 SKUs."
    assert FEATURE_SEARCH_EARLY_STOP_MAPE == 0.08, \
        "Feature search early-stop MAPE is 0.08."
    assert SignalType.PATTERN_CONFIDENCE == "pattern_confidence", \
        "pattern_confidence signal type string must match exactly."
    assert set(ALL_ALERT_TYPES) == {AlertType.MARKET_SHIFT, AlertType.CHANNEL_COUNT_CHANGED}, \
        "ALL_ALERT_TYPES must cover both documented alert types."
    assert HIGH_VOLATILITY_CV == 1.0, \
        "HIGH_VOLATILITY_CV is 1.0 per Master Spec §9.4 — not 1.5."
    assert PROMO_ZSCORE_THRESHOLD == 3.0, \
        "promo_spike z-score threshold is 3.0 per Master Spec §9.4."
    assert UNUSUAL_DROP_CONSECUTIVE_PERIODS == 3, \
        "unusual_drop requires 3 consecutive periods per Master Spec §9.4."
    assert REORDER_SIGNAL_LOOKBACK == 5, \
        "Reorder bias factor computed from last 5 signals per Master Spec §9.1."
    # Reasonableness bounds (max_forecast_vs_baseline, min_forecast_vs_baseline) are
    # learnable params in tenant_learning_params — no constant to assert here.
    assert B2B_DISABLED_FLAG == "b2b_mode_disabled", \
        "B2B disabled flag string is 'b2b_mode_disabled' per Master Spec §9 E006."


_verify_constants()
