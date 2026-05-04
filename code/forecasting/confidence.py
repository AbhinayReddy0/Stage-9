"""
Confidence formula for Sub-Stage 9.5.

Extracted from forecasting.py so compute_confidence() and each of its
five multiplicative steps can be unit-tested in isolation without
constructing a full pipeline context.

ForecastContext is defined here (not in forecasting.py) to avoid a
circular import — forecasting.py imports ForecastContext from this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from infrastructure.constants import CONFIDENCE_BASE_PARAM, Param
from infrastructure.tenant_params import TenantParams


# Exception flags that drive Step 4's exception_penalty multiplication.
# 'high_mape' is handled separately in determine_status (Step 5) — it forces
# 'needs_acknowledgment' regardless of confidence score.
EXCEPTION_PENALTY_FLAGS = frozenset({
    "stockout", "promo_spike", "unusual_drop", "high_volatility",
    "forecast_unusually_high", "forecast_unusually_low",
})


@dataclass
class ForecastContext:
    """
    Per-SKU run context assembled from earlier sub-stages:
      * training_data_truncated, insufficient_post_break  ← 9.4
      * effective_max_horizon, reorder_bias_factor        ← 9.1
      * oos_adjustment_factor, on_watchlist               ← preload / Stage 8
    """
    training_data_truncated: bool = False
    insufficient_post_break: bool = False
    effective_max_horizon: int = 365
    reorder_bias_factor: float = 1.0
    oos_adjustment_factor: float = 1.0
    on_watchlist: bool = False


def compute_confidence(
    *,
    pattern_label: str,
    backtest_mape: float,
    exception_flags: list[str],
    calibration_gap: Optional[float],
    stage8_confidence: float,
    reorder_bias_factor: float,
    ctx: ForecastContext,
    params: TenantParams,
    reasonableness_multiplier: float = 1.0,
) -> tuple[float, float]:
    """
    Returns (confidence_base, confidence_final).

    Step 1: base × (1 − min(mape, mape_cap)) × exception_penalty (if any flags)
    Step 2: calibration adjustment — symmetric around overconfidence_threshold:
              gap >  +threshold → × overconfidence_mult  (intervals too tight)
              gap < -threshold  → × underconfidence_mult (intervals too wide)
              |gap| ≤ threshold → no change
    Step 3: × stage8_penalty_mult when stage8_confidence < stage8_penalty_threshold
    Step 4: × reorder_bias_factor (per-SKU from 9.1)
    Step 5: × (1 − structural_break_confidence_penalty) when training truncated,
              OR × insufficient_post_break_mult when post-break data is too short
    Plus the reasonableness multiplier from forecasting Step 3.
    Clamp to [confidence_floor, confidence_ceiling].
    """
    base_param = CONFIDENCE_BASE_PARAM.get(pattern_label, "confidence_base_stable")
    base = params.get(base_param)

    # Step 1 — MAPE term
    mape_cap = params.get("mape_cap_in_confidence")
    mape = mape_cap if (
        backtest_mape is None
        or (isinstance(backtest_mape, float) and np.isnan(backtest_mape))
    ) else min(float(backtest_mape), mape_cap)

    confidence = base * (1.0 - mape)

    if any(flag in EXCEPTION_PENALTY_FLAGS for flag in exception_flags):
        confidence *= params.get("exception_penalty")

    # Step 2 — calibration gap
    if calibration_gap is not None:
        threshold = params.get("overconfidence_threshold")
        if calibration_gap > threshold:
            confidence *= params.get("overconfidence_mult")
        elif calibration_gap < -threshold:
            confidence *= params.get("underconfidence_mult")

    # Step 3 — Stage 8 signal quality
    if stage8_confidence is not None and stage8_confidence < params.get("stage8_penalty_threshold"):
        confidence *= params.get("stage8_penalty_mult")

    # Step 4 — reorder bias
    confidence *= reorder_bias_factor

    # Step 5 — structural break penalty
    if ctx.training_data_truncated:
        # structural_break_confidence_penalty is the reduction amount (e.g. 0.15 → ×0.85).
        # Let UnknownParamError propagate if the param is missing — that is a seeding bug.
        penalty = params.get(Param.STRUCTURAL_BREAK_CONFIDENCE_PENALTY)
        confidence *= max(0.0, 1.0 - penalty)
    elif ctx.insufficient_post_break:
        confidence *= params.get("insufficient_post_break_mult")

    confidence *= reasonableness_multiplier

    floor = params.get("confidence_floor")
    ceiling = params.get("confidence_ceiling")
    confidence_final = max(floor, min(ceiling, confidence))
    return float(base), float(confidence_final)
