"""
models/naive.py — Atheera Stage 9 Forecasting Agent
=====================================================
NaiveForecast: flat-level demand model for cold_start SKUs.

Principle:
    Computes a smoothed demand level from recent history and projects it flat
    across all horizons. No trend. No seasonality.
    Forecast for horizon H = level × H days.

Assigned to: pattern_label = 'cold_start'

HP search space — 9 configurations (3 lag_periods × 3 smoothing_methods):
    lag_periods:      [1, 7, 14]   — how many recent days define the level
    smoothing_method: ['last_value', 'mean_3d', 'mean_7d']
    default_hp:       {'lag_periods': 7, 'smoothing_method': 'mean_7d'}

NaiveForecast is also the E005 fallback target used by Sub-Stage 9.3 when
every HP config for the assigned model raises ModelFitError. Sub-Stage 9.3
catches ModelFitError and falls back to NaiveForecast with default_hp.

"""

from __future__ import annotations

import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd

from models.base import BaseModel, ModelFitError
from models.bootstrap import bootstrap_quantiles
from infrastructure.constants import HORIZONS, FORECAST_COLUMN_MAP

log = logging.getLogger(__name__)

__all__ = ["NaiveForecast"]

# ---------------------------------------------------------------------------
# HP search space — generated programmatically to prevent typos
# ---------------------------------------------------------------------------
_LAG_PERIODS: list[int] = [1, 7, 14]
_SMOOTHING_METHS: list[str] = ["last_value", "mean_3d", "mean_7d"]

_HP_SEARCH_SPACE: list[dict] = [
    {"lag_periods": lp, "smoothing_method": sm}
    for lp, sm in itertools.product(_LAG_PERIODS, _SMOOTHING_METHS)
]  # exactly 9 configurations

_DEFAULT_HP: dict = {"lag_periods": 7, "smoothing_method": "mean_7d"}


class NaiveForecast(BaseModel):
    """
    Flat-level demand model for cold_start (new or unclassifiable) SKUs.

    Computes a single scalar demand level (units/day) from the most recent
    `lag_periods` days and projects it flat across every forecast horizon.

    Thread-pool safe: all mutable state stored on instance, not at class level.
    """

    def __init__(self, hp: dict) -> None:
        super().__init__(hp)
        # Fitted state — populated by fit(), consumed by predict/predict_all_horizons
        self._level: float = 0.0

    # -----------------------------------------------------------------------
    # 5 required properties
    # -----------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "Naive Forecast"

    @property
    def hp_search_space(self) -> list[dict]:
        # Return a shallow copy — callers must not mutate the class-level list
        return list(_HP_SEARCH_SPACE)

    @property
    def default_hp(self) -> dict:
        return dict(_DEFAULT_HP)  # copy — prevent mutation of class default

    @property
    def required_features(self) -> list[str]:
        return ["date", "qty"]

    @property
    def optional_features(self) -> list[str]:
        # NaiveForecast uses only qty history — ignores all optional features.
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
        Compute the scalar demand level from recent history.

        sample_weights is accepted but intentionally ignored (Build Plan §2.1:
        only NeuralProphet / Prophet use sample_weights).
        """
        try:
            self._level = self._compute_level(df)
        except Exception as exc:
            raise ModelFitError(
                f"NaiveForecast.fit failed [lag={self.hp.get('lag_periods')}, "
                f"method={self.hp.get('smoothing_method')}]: {exc}"
            ) from exc

    def predict(
            self,
            df: pd.DataFrame,
            features: list[str],
            horizon: int,
    ) -> np.ndarray:
        """
        Return a flat-level forecast array of length `horizon`.

        Each element = self._level (units/day). All values ≥ 0.
        Done Criterion D6: returns np.ndarray, len == horizon.
        """
        try:
            level = max(0.0, self._level)
            return np.full(horizon, level, dtype=float)
        except Exception as exc:
            raise ModelFitError(
                f"NaiveForecast.predict failed: {exc}"
            ) from exc

    def predict_all_horizons(
            self,
            df: pd.DataFrame,
            features: list[str],
            oos_factor: float = 1.0,
    ) -> dict:
        """
        Generate cumulative demand forecasts for all 8 HORIZONS.

        Formula: point(H) = level × H × oos_factor
        Residuals are re-computed here to reflect the full training df.

        """
        try:
            level = max(0.0, self._level)
            residuals = self.compute_residuals(df, features)
            result: dict = {}

            for H in HORIZONS:
                col = FORECAST_COLUMN_MAP[H]
                point = level * H * oos_factor
                result[col] = bootstrap_quantiles(point, residuals, "cold_start")

            return result

        except ModelFitError:
            raise  # do not double-wrap
        except Exception as exc:
            raise ModelFitError(
                f"NaiveForecast.predict_all_horizons failed: {exc}"
            ) from exc

    def compute_residuals(
            self,
            df: pd.DataFrame,
            features: list[str],
    ) -> np.ndarray:
        """
        Re-fit on df and return actual − fitted for the last 30 rows.

        For NaiveForecast, the fitted value for every row is the constant level
        computed from the full df. Residuals = actual_t − level for each day.
        """
        try:
            self.fit(df, features)  # re-fit on full df
            level = max(0.0, self._level)
            fitted = np.full(len(df), level, dtype=float)
            # Sanitise actual values — NaN/Inf in residuals corrupt bootstrap_quantiles.
            resids = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0) - fitted
            return resids[-30:]  # last 30 rows only (Build Plan §06)

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"NaiveForecast.compute_residuals failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Internal helpers — never called by sub-stages directly
    # -----------------------------------------------------------------------

    def _compute_level(self, df: pd.DataFrame) -> float:
        """
        Compute the scalar demand level (units/day) from recent history.

        HP parameters used:
            lag_periods:      how many recent days to look at
            smoothing_method: how to aggregate those days into a single level

        Edge case:
            If df has fewer than 7 rows, fall back to the mean of all available
            rows regardless of lag_periods. Short-history guard prevents a
            lag_periods=14 request from returning a single point when only 3
            days exist.
        """
        # Sanitise input: replace NaN/Inf with 0 before any arithmetic.
        # Upstream data cleaning (Stage 7/8) should prevent this, but a single
        # corrupt row must not propagate inf/nan into the forecast output.
        qty = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        lag = int(self.hp.get("lag_periods", 7))
        meth = str(self.hp.get("smoothing_method", "mean_7d"))

        if len(qty) == 0:
            return 0.0

        # Short-history guard — not enough data for meaningful lag window
        if len(qty) < 7:
            return float(np.mean(qty))

        # Restrict to the most recent `lag` days for level estimation
        recent = qty[-lag:] if lag <= len(qty) else qty

        if meth == "last_value":
            # Most recent day only — highest recency weight
            return float(recent[-1])

        elif meth == "mean_3d":
            # Rolling 3-day mean of the most recent 3 days in the lag window
            window = recent[-3:] if len(recent) >= 3 else recent
            return float(np.mean(window))

        else:  # "mean_7d" — default
            # Rolling 7-day mean — balanced between recency and stability
            window = recent[-7:] if len(recent) >= 7 else recent
            return float(np.mean(window))
