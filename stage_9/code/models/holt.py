"""
models/holt.py — Atheera Stage 9 Forecasting Agent
====================================================
HoltLinearTrend: damped Holt's method for trending-demand SKUs.

Principle:
    Maintains two smoothed estimates:
        Level (L) — current demand baseline (units/day)
        Trend (T) — rate of change (units/day per day)

    A damping factor phi (0 < phi < 1) prevents unlimited extrapolation —
    growth realistically slows down over long horizons.

    Forecast at day h: L + T × (phi^1 + phi^2 + … + phi^h)
    Undamped (phi=1):  L + T × h

Assigned to: pattern_label = 'trending'

HP search space — 24 configurations (4 × 3 × 2):
    smoothing_level (alpha): [0.1, 0.2, 0.3, 0.4]
    smoothing_trend  (beta): [0.05, 0.1, 0.2]
    damped_trend:            [True, False]
    default_hp: {'smoothing_level': 0.3, 'smoothing_trend': 0.1, 'damped_trend': True}

CRITICAL RULES (Build Plan §06):
    1. Always pass optimized=False to statsmodels ExponentialSmoothing.
       Thompson supplies alpha and beta from the HP dict explicitly.
       Fitted params must match hp values.
    2. If statsmodels raises any exception, fall back to _holt_numpy().
       Never propagate a statsmodels error.

"""

from __future__ import annotations

import itertools
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.base import BaseModel, ModelFitError
from models.bootstrap import bootstrap_quantiles
from infrastructure.constants import HORIZONS, FORECAST_COLUMN_MAP, HOLT_DAMPING_COEFFICIENT

log = logging.getLogger(__name__)

__all__ = ["HoltLinearTrend"]

# ---------------------------------------------------------------------------
# HP search space — 24 configurations (4 alpha × 3 beta × 2 damped)
# ---------------------------------------------------------------------------
_HP_SEARCH_SPACE: list[dict] = [
    {
        "smoothing_level": alpha,
        "smoothing_trend": beta,
        "damped_trend": damped,
    }
    for alpha, beta, damped in itertools.product(
        [0.1, 0.2, 0.3, 0.4],
        [0.05, 0.1, 0.2],
        [True, False],
    )
]
_DEFAULT_HP: dict = {
    "smoothing_level": 0.3,
    "smoothing_trend": 0.1,
    "damped_trend": True,
}

# Fixed damping coefficient — standard literature value.


class HoltLinearTrend(BaseModel):
    """
    Damped Holt's Linear Trend for consistently growing or declining demand.

    Uses statsmodels ExponentialSmoothing with explicit alpha/beta
    (optimized=False). Falls back to _holt_numpy() on any statsmodels error.

    Thread-pool safe: all state per-instance.
    """

    def __init__(self, hp: dict) -> None:
        super().__init__(hp)
        self._fitted_result = None  # statsmodels ResultsWrapper or None
        self._level: float = 0.0  # final smoothed level L_n
        self._trend: float = 0.0  # final smoothed trend T_n
        self._using_fallback: bool = False  # True when numpy fallback was used

    # -----------------------------------------------------------------------
    # 5 required properties
    # -----------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "Holt's Linear Trend"

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
        Fit Holt's Linear Trend with EXPLICIT alpha and beta.

        CRITICAL: optimized=False is mandatory. Falls back to _holt_numpy()
        if statsmodels raises. sample_weights silently ignored.
        """
        try:
            alpha = float(self.hp.get("smoothing_level", 0.3))
            beta = float(self.hp.get("smoothing_trend", 0.1))
            damped = bool(self.hp.get("damped_trend", True))
            # Sanitise: NaN/Inf replaced with 0 — corrupt rows must not propagate.
            series = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            # Guard: need at least 2 points to estimate a trend
            if len(series) < 2:
                self._level = float(np.mean(series)) if len(series) else 0.0
                self._trend = 0.0
                self._using_fallback = True
                return

            # Primary path: statsmodels with explicit alpha, beta
            try:
                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    damped_trend=damped,
                    seasonal=None,
                    initialization_method="estimated",
                )
                # CRITICAL: optimized=False — Thompson supplies alpha, beta.
                result = model.fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    optimized=False,  # MANDATORY rule
                )
                self._fitted_result = result
                self._level = float(result.level.iloc[-1])
                self._trend = float(result.slope.iloc[-1])
                self._using_fallback = False

                # Verify statsmodels used our parameters.
                fitted_alpha = float(result.params["smoothing_level"])
                fitted_beta = float(result.params["smoothing_trend"])
                if abs(fitted_alpha - alpha) > 1e-4 or abs(fitted_beta - beta) > 1e-4:
                    log.warning(
                        "HoltLinearTrend: statsmodels params diverge "
                        "(alpha req=%.3f got=%.3f, beta req=%.3f got=%.3f) "
                        "— switching to numpy fallback",
                        alpha, fitted_alpha, beta, fitted_beta,
                    )
                    raise RuntimeError("params mismatch in statsmodels result")

            except Exception as inner_exc:
                # Numpy fallback
                log.debug(
                    "HoltLinearTrend: numpy fallback "
                    "[alpha=%.2f, beta=%.2f, damped=%s]: %s",
                    alpha, beta, damped, inner_exc,
                )
                phi = HOLT_DAMPING_COEFFICIENT if damped else 1.0
                self._level, self._trend = _holt_numpy(series, alpha, beta, phi=phi)
                self._fitted_result = None
                self._using_fallback = True

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"HoltLinearTrend.fit failed "
                f"[alpha={self.hp.get('smoothing_level')}, "
                f"beta={self.hp.get('smoothing_trend')}, "
                f"damped={self.hp.get('damped_trend')}]: {exc}"
            ) from exc

    def predict(
            self,
            df: pd.DataFrame,
            features: list[str],
            horizon: int,
    ) -> np.ndarray:
        """
        Return per-day demand forecast for the next `horizon` days.

        Applies damped-trend formula forward from the fitted (L, T).

        """
        try:
            return self._daily_forecasts(horizon)
        except Exception as exc:
            raise ModelFitError(f"HoltLinearTrend.predict failed: {exc}") from exc

    def predict_all_horizons(
            self,
            df: pd.DataFrame,
            features: list[str],
            oos_factor: float = 1.0,
    ) -> dict:
        """
        Generate cumulative demand for all 8 HORIZONS.

        Computes daily forecasts for max(HORIZONS)=365 days once, then
        extracts cumulative sums at each horizon boundary — single-pass approach.
        """
        try:
            residuals = self.compute_residuals(df, features)
            result: dict = {}

            # Single forward pass: compute daily forecasts up to 365
            max_h = max(HORIZONS)
            daily = self._daily_forecasts(max_h)

            for H in HORIZONS:
                col = FORECAST_COLUMN_MAP[H]
                point = float(np.sum(daily[:H])) * oos_factor
                point = max(0.0, point)
                result[col] = bootstrap_quantiles(point, residuals, "trending")

            return result

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"HoltLinearTrend.predict_all_horizons failed: {exc}"
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
                f"HoltLinearTrend.compute_residuals failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _daily_forecasts(self, horizon: int) -> np.ndarray:
        """
        Compute daily demand forecast for days 1 … horizon.

        Damped Holt formula:
            forecast at day h = L + T × phi × (1 − phi^h) / (1 − phi)
        Undamped (phi=1):
            forecast at day h = L + T × h

        Clamped to 0 — demand cannot be negative even if the trend is strongly
        negative and L + trend < 0 at some horizon.
        """
        level = self._level
        trend = self._trend
        damped = bool(self.hp.get("damped_trend", True))
        phi = HOLT_DAMPING_COEFFICIENT if damped else 1.0

        forecasts = np.empty(horizon, dtype=float)
        for h in range(1, horizon + 1):
            if damped and phi < 1.0:
                # Geometric series sum: phi^1 + phi^2 + … + phi^h
                phi_sum = phi * (1.0 - phi ** h) / (1.0 - phi)
            else:
                phi_sum = float(h)
            forecasts[h - 1] = max(0.0, level + trend * phi_sum)

        return forecasts

    def _get_fitted_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return in-sample fitted values from statsmodels, or constant-level fallback.
        """
        if self._fitted_result is not None and not self._using_fallback:
            return np.maximum(self._fitted_result.fittedvalues.values, 0.0)
        # Numpy fallback: use current level as constant approximation
        return np.full(len(df), max(0.0, self._level), dtype=float)


# ===========================================================================
# NumPy Holt fallback
# ===========================================================================

def _holt_numpy(
        series: np.ndarray,
        alpha: float,
        beta: float,
        phi: float = 0.95,
) -> Tuple[float, float]:
    """
    Damped Holt's Linear Trend via direct Python iteration.

    Returns (final_level L_n, final_trend T_n) for use in forward forecasting.

    Update equations:
        L_t = alpha × y_t + (1 − alpha) × (L_(t−1) + phi × T_(t−1))
        T_t = beta  × (L_t − L_(t−1)) + (1 − beta) × phi × T_(t−1)

    Initialisation (first two observations):
        L_0 = series[0]
        T_0 = series[1] − series[0]   (first difference as initial slope)

    Args:
        series: Observed demand values (float64). Length must be ≥ 2.
        alpha:  Level smoothing factor ∈ (0, 1).
        beta:   Trend smoothing factor ∈ (0, 1).
        phi:    Damping factor ∈ [0, 1]. phi=1 → undamped Holt.

    Returns:
        (level, trend) — final level and trend for forward forecasting.
    """
    level = float(series[0])
    trend = float(series[1]) - float(series[0])  # initial trend = first difference

    for y in series[2:]:
        level_prev = level
        trend_prev = trend
        # Level update: blend new observation with dampened extrapolation
        level = alpha * float(y) + (1.0 - alpha) * (level_prev + phi * trend_prev)
        # Trend update: blend new slope with dampened prior trend
        trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend_prev

    return level, trend
