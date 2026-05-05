"""
Starting values for all tenant_learning_params rows.

Source of truth:
  - 37 params from STAGE_9_DATABASE_CONTRACTS.docx, 'Complete Seed Values' section.
  - 2 additional params (stage8_penalty_threshold, stage8_confidence_threshold) from STAGE_9_TECHNICAL_CONTEXT.docx
    These are referenced by the runtime confidence formula but not in the
    Database Contracts seed table — confirmed addition per founder decision.
  - 2 additional params (max_forecast_vs_baseline, min_forecast_vs_baseline)
    from STAGE_9_TECHNICAL_CONTEXT.docx, Part 5 Sub-Stage 9.5 'Reasonableness
    Check'. Cap and floor for daily forecast vs rolling 90-day baseline ratio.
  - 9 additional params for Sub-Stage 9.5 confidence formula multipliers and
    risk-band cutoffs — these replace the hardcoded floats that backtesting.py /
    forecasting.py previously embedded. All read at runtime via TenantParams.get()
    so they evolve per customer as evidence accumulates.
  - 1 additional param (model_performance_stable_band)
    for the model performance aggregator. Added to
    make all runtime thresholds tenant-configurable per Principle 1.

Total: 58 parameters seeded per tenant at onboarding.

To update starting values, edit this file. Do not hardcode values elsewhere.
"""

from decimal import Decimal

from infrastructure.constants import Param, TenantMaturity

# List of (param_name, starting_value) tuples, grouped by concern.
TENANT_LEARNING_PARAMS_DEFAULTS: list[tuple[str, Decimal]] = [
    # Pattern-specific confidence bases (5)
    ("confidence_base_cold_start",          Decimal("0.50")),
    ("confidence_base_intermittent",        Decimal("0.60")),
    ("confidence_base_seasonal",            Decimal("0.80")),
    ("confidence_base_trending",            Decimal("0.70")),
    ("confidence_base_stable",              Decimal("0.90")),

    # Pattern-specific quantile buffers (5)
    ("quantile_cold_start",                 Decimal("0.90")),
    ("quantile_intermittent",               Decimal("0.90")),
    ("quantile_seasonal",                   Decimal("0.90")),
    ("quantile_trending",                   Decimal("0.80")),
    ("quantile_stable",                     Decimal("0.80")),

    # Service level (1)
    ("service_level_target",                Decimal("0.90")),

    # Confidence tier gates (3)
    ("decision_gate_threshold",             Decimal("0.70")),
    ("review_suggested_threshold",          Decimal("0.60")),
    ("review_required_threshold",           Decimal("0.45")),

    # Confidence math (6)
    ("exception_penalty",                   Decimal("0.80")),
    ("confidence_floor",                    Decimal("0.30")),
    ("confidence_ceiling",                  Decimal("0.95")),
    ("overconfidence_threshold",            Decimal("0.10")),
    # Stage 8 uncertainty inheritance thresholds .
    # If Stage 8 composite_confidence < stage8_penalty_threshold → confidence × 0.92.
    # If Stage 8 pattern_confidence  < stage8_confidence_threshold → confidence × 0.90.
    ("stage8_penalty_threshold",            Decimal("0.60")),
    ("stage8_confidence_threshold",         Decimal("0.65")),

    # Forecast reasonableness bounds.
    # daily_forecast > rolling_90d_avg × max_forecast_vs_baseline → flag unusually_high.
    # daily_forecast < rolling_90d_avg × min_forecast_vs_baseline → flag unusually_low.
    ("max_forecast_vs_baseline",            Decimal("5.00")),
    ("min_forecast_vs_baseline",            Decimal("0.10")),

    # Safety + learning rate (2)
    ("safety_stock_factor",                 Decimal("0.20")),
    ("calibration_update_rate",             Decimal("0.10")),

    # Thompson + features (2)
    ("thompson_exploration_budget",         Decimal("3.00")),
    ("feature_reliability_floor",           Decimal("0.30")),

    # Promo + clearance (3)
    ("max_promo_multiplier",                Decimal("3.00")),
    ("price_elasticity_clearance",          Decimal("-1.50")),
    ("clearance_markdown_threshold",        Decimal("0.15")),

    # Backtest windows (5)
    ("default_backtest_window",             Decimal("60.00")),
    ("min_backtest_window",                 Decimal("14.00")),
    ("max_backtest_window",                 Decimal("90.00")),
    ("backtest_short_obs_threshold",        Decimal("60.00")),
    ("backtest_exploit_obs_threshold",      Decimal("180.00")),

    # Size curves + warm-start (3)
    ("size_curve_smoothing_alpha",          Decimal("0.30")),
    ("warm_start_max_mape",                 Decimal("0.25")),
    ("warm_start_min_comps",                Decimal("3.00")),

    # Operational (1)
    ("micro_update_threshold_hours",        Decimal("18.00")),

    # Structural break (2)
    ("structural_break_sensitivity",        Decimal("0.30")),
    ("structural_break_confidence_penalty", Decimal("0.15")),

    # Exploit thresholds per tenant maturity (3) + Thompson confidence gate (1)
    ("exploit_threshold_new",               Decimal("8.00")),
    ("exploit_threshold_developing",        Decimal("5.00")),
    ("exploit_threshold_established",       Decimal("3.00")),
    ("thompson_exploit_confidence_threshold", Decimal("0.60")),

    # Sub-Stage 9.5 confidence formula multipliers (7) and risk-band cutoffs (2)
    # evolve per customer as calibration evidence accumulates.
    ("mape_cap_in_confidence",              Decimal("0.50")),
    ("overconfidence_mult",                 Decimal("0.90")),
    ("underconfidence_mult",                Decimal("1.10")),
    ("stage8_penalty_mult",                 Decimal("0.92")),
    ("insufficient_post_break_mult",        Decimal("0.75")),
    ("forecast_unusually_high_mult",        Decimal("0.85")),
    ("forecast_unusually_low_mult",         Decimal("0.90")),
    ("risk_low_min",                        Decimal("0.85")),
    ("risk_medium_min",                     Decimal("0.70")),

    # Model performance aggregator (1)
    ("model_performance_stable_band",       Decimal("0.02")),

    # Seasonal model guard (1)
    ("min_seasonal_obs_days",               Decimal("120.00")),

    # Channel split and learning calibration gates (3)
    # Moved from module-level constants so they evolve per tenant as evidence accumulates.
    ("channel_split_confidence_threshold",  Decimal("0.50")),
    ("min_learning_evidence_count",         Decimal("10.00")),
    ("quantile_calibration_step",           Decimal("0.02")),
]

assert len(TENANT_LEARNING_PARAMS_DEFAULTS) == 58, (
    f"Expected 58 params (37 from Database Contracts Table 1 + 4 from Tech Context "
    f"Parts 5 and 7 + 9 confidence-formula params + 1 model-perf param + "
    f"1 Thompson exploit confidence gate + 2 backtest obs thresholds + "
    f"1 seasonal obs guard + 3 calibration gates), "
    f"got {len(TENANT_LEARNING_PARAMS_DEFAULTS)}"
)

# Set of valid param names — used by seed.py for override validation.
VALID_PARAM_NAMES: frozenset[str] = frozenset(
    name for name, _ in TENANT_LEARNING_PARAMS_DEFAULTS
)

# Guard against accidental duplicate entries in the list above.
assert len(VALID_PARAM_NAMES) == len(TENANT_LEARNING_PARAMS_DEFAULTS), (
    "Duplicate param_name in TENANT_LEARNING_PARAMS_DEFAULTS"
)

# Guard against Param class and defaults drifting out of sync.
_PARAM_CLASS_VALUES: frozenset[str] = frozenset(
    v for k, v in vars(Param).items() if not k.startswith("_")
)
assert _PARAM_CLASS_VALUES == VALID_PARAM_NAMES, (
    f"Param class and TENANT_LEARNING_PARAMS_DEFAULTS are out of sync.\n"
    f"  In Param only: {sorted(_PARAM_CLASS_VALUES - VALID_PARAM_NAMES)}\n"
    f"  In defaults only: {sorted(VALID_PARAM_NAMES - _PARAM_CLASS_VALUES)}"
)

_TENANT_MATURITY_VALUES: frozenset[str] = frozenset(
    v for k, v in vars(TenantMaturity).items() if not k.startswith("_")
)
VALID_TENANT_MATURITY: frozenset[str] = _TENANT_MATURITY_VALUES
