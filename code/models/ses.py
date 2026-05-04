"""
models/ses.py — Atheera Stage 9 Forecasting Agent
==================================================
SESModel: Simple Exponential Smoothing for stable / steady SKUs.

Principle (Build Plan §3.5):
    Maintains a single smoothed level estimate.
    Update formula: S_t = alpha × y_t + (1 − alpha) × S_(t−1)

    More-recent observations receive exponentially higher weight. No trend.
    No seasonality. Optimal for demand that is flat with noise.

Assigned to: pattern_label = 'stable' (and 'steady' — Stage 8 alias)

HP search space — 5 configurations:
    smoothing_level (alpha): [0.1, 0.2, 0.3, 0.4, 0.5]
    default_hp: {'smoothing_level': 0.3}

CRITICAL RULE — optimized=False (Build Plan §06, §3.5):
    Always pass optimized=False to statsmodels SimpleExpSmoothing.
    Thompson Sampling supplies alpha from the HP dict explicitly.
    If optimized=True, statsmodels auto-selects alpha and overwrites the
    HP dict — the Thompson search becomes meaningless. Done Criterion D8
    verifies: fitted.params.smoothing_level == hp['smoothing_level'].

MANUAL FALLBACK (Build Plan §3.5):
    If statsmodels raises any exception, fall back to _ses_numpy().
    Never propagate a statsmodels error.

Build Plan §3.5, Technical Context §Part 12.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from models.base import BaseModel, ModelFitError
from models.bootstrap import bootstrap_quantiles
from infrastructure.constants import HORIZONS, FORECAST_COLUMN_MAP

log = logging.getLogger(__name__)

__all__ = ["SESModel"]

# ---------------------------------------------------------------------------
# HP search space — 5 configurations (single parameter)
# ---------------------------------------------------------------------------
_HP_SEARCH_SPACE: list[dict] = [
    {"smoothing_level": alpha}
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]
]
_DEFAULT_HP: dict = {"smoothing_level": 0.3}


class SESModel(BaseModel):
    """
    Simple Exponential Smoothing for stable, predictable demand patterns.

    Lower alpha → slower adaptation, weights history more heavily.
    Higher alpha → faster adaptation, emphasises recent demand.
    For stable products, lower alpha is typically better (demand isn't changing).

    Thread-pool safe: all mutable state stored per-instance.
    """

    def __init__(self, hp: dict) -> None:
        super().__init__(hp)
        self._fitted_result = None  # statsmodels result object (or None)
        self._level: float = 0.0  # final smoothed level (units/day)
        self._using_fallback: bool = False  # True when numpy fallback was used

    # -----------------------------------------------------------------------
    # 5 required properties
    # -----------------------------------------------------------------------

    @property
    def level(self) -> float:
        """Final smoothed level after fit(). 0.0 before fit is called."""
        return self._level

    @property
    def model_name(self) -> str:
        return "Simple Exponential Smoothing (SES)"

    @property
    def hp_search_space(self) -> list[dict]:
        return list(_HP_SEARCH_SPACE)

    @property
    def default_hp(self) -> dict:
        return dict(_DEFAULT_HP)

    @property
    def required_features(self) -> list[str]:
        return ["date", "qty"]

    @property
    def optional_features(self) -> list[str]:
        # SES uses only qty history; optional features are ignored.
        return []

    # -----------------------------------------------------------------------
    # 5 required methods
    # -----------------------------------------------------------------------

    def fit(
            self,
            df: pd.DataFrame,
            features: list[str],
            sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit Simple Exponential Smoothing with EXPLICIT alpha from hp dict.

        CRITICAL: optimized=False is mandatory.
        Falls back to _ses_numpy() if statsmodels raises any exception.
        sample_weights is silently ignored (SES cannot use them).
        """
        try:
            alpha = float(self.hp.get("smoothing_level", 0.3))
            # Sanitise: NaN/Inf replaced with 0 — corrupt rows must not propagate.
            series = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            if len(series) == 0:
                self._level = 0.0
                self._using_fallback = True
                return

            # Primary path: statsmodels with explicit alpha
            try:
                model = SimpleExpSmoothing(series, initialization_method="heuristic")
                # CRITICAL: optimized=False — Thompson supplies alpha explicitly.
                # If optimized=True, statsmodels ignores our alpha completely.
                result = model.fit(smoothing_level=alpha, optimized=False)
                self._fitted_result = result
                self._level = float(result.level.iloc[-1])
                self._using_fallback = False

                # Verify statsmodels respected our alpha.
                # Under optimized=False this should always match, but guard anyway.
                fitted_alpha = float(result.params["smoothing_level"])
                if abs(fitted_alpha - alpha) > 1e-4:
                    log.warning(
                        "SESModel: statsmodels alpha %.5f diverges from requested "
                        "%.5f — switching to numpy fallback",
                        fitted_alpha, alpha,
                    )
                    raise RuntimeError("alpha mismatch in statsmodels result")

            except Exception as inner_exc:
                # Numpy fallback — mandatory.
                # Log at DEBUG, so it doesn't flood production logs for transient issues.
                log.debug(
                    "SESModel: numpy fallback for alpha=%.2f: %s",
                    alpha, inner_exc,
                )
                self._level = _ses_numpy(series, alpha)
                self._fitted_result = None
                self._using_fallback = True

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"SESModel.fit failed [alpha={self.hp.get('smoothing_level')}]: {exc}"
            ) from exc

    def predict(
            self,
            df: pd.DataFrame,
            features: list[str],
            horizon: int,
    ) -> np.ndarray:
        """
        Return flat-level forecast array of length `horizon`.

        SES produces no trend — forecast is constant at the current smoothed level.
        Done Criterion D6: np.ndarray, len == horizon, all values ≥ 0.
        """
        try:
            level = max(0.0, self._level)
            return np.full(horizon, level, dtype=float)
        except Exception as exc:
            raise ModelFitError(f"SESModel.predict failed: {exc}") from exc

    def predict_all_horizons(
            self,
            df: pd.DataFrame,
            features: list[str],
            oos_factor: float = 1.0,
    ) -> dict:
        """
        Generate cumulative demand for all 8 HORIZONS.

        Formula: point(H) = level × H × oos_factor  (flat forecast, no trend).
        Pattern label 'stable' used for BOOTSTRAP_UNCERTAINTY lookup.
        """
        try:
            level = max(0.0, self._level)
            residuals = self.compute_residuals(df, features)
            result: dict = {}

            for H in HORIZONS:
                col = FORECAST_COLUMN_MAP[H]
                point = level * H * oos_factor
                result[col] = bootstrap_quantiles(point, residuals, "stable")

            return result

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"SESModel.predict_all_horizons failed: {exc}"
            ) from exc

    def compute_residuals(
            self,
            df: pd.DataFrame,
            features: list[str],
    ) -> np.ndarray:
        """
        Re-fit on df and return actual − fitted for the last 30 rows.
        """
        try:
            self.fit(df, features)
            fitted = self._get_fitted_values(df)
            # Sanitise actual values — NaN/Inf in residuals corrupt bootstrap_quantiles.
            resids = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0) - fitted
            return resids[-30:]

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"SESModel.compute_residuals failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_fitted_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return in-sample fitted values (S_1, S_2, ..., S_n) from statsmodels.

        Falls back to a constant-level array when numpy path was used.
        Clamps to non-negative — fitted values can theoretically go below 0
        for series with zero observations early in the history.
        """
        if self._fitted_result is not None and not self._using_fallback:
            return np.maximum(self._fitted_result.fittedvalues.values, 0.0)
        # Numpy fallback: constant level for every row
        return np.full(len(df), max(0.0, self._level), dtype=float)


# ===========================================================================
# NumPy fallback function — top-level for pickle compatibility
# ===========================================================================

def _ses_numpy(series: np.ndarray, alpha: float) -> float:
    """
    Simple Exponential Smoothing via direct Python iteration.

    Returns the final smoothed level S_n after processing all observations.

    Update formula:  S_t = alpha × y_t + (1 − alpha) × S_(t−1)
    Initialisation:  S_0 = series[0]   (first observation)

    Args:
        series: Observed demand values (float64 array). Length must be ≥ 1.
        alpha:  Smoothing factor in [0, 1]. Caller guarantees this range.

    Returns:
        Final smoothed level as float.

    """
    if len(series) == 0:
        return 0.0
    level = float(series[0])  # initialise with first observation
    for y in series[1:]:
        level = alpha * float(y) + (1.0 - alpha) * level
    return level
