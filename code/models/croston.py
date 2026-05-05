"""
models/croston.py
======================================================
CrostonMethod: demand model for intermittent-demand SKUs.

Principle:
    Standard averaging underestimates demand on days it occurs because zeros
    dilute the average. Croston fixes this by separately tracking:
        demand_smooth   — average SIZE of demand when it occurs (ignores zeros)
        interval_smooth — average NUMBER OF DAYS between demand events

    Combined: daily_rate = demand_smooth / interval_smooth

Assigned to: pattern_label = 'intermittent'

HP search space — 9 configurations (3 alpha × 3 interval_types):
    alpha:         [0.05, 0.10, 0.20]
    interval_type: ['classic', 'SBA', 'TSB']
    default_hp:    {'alpha': 0.10, 'interval_type': 'SBA'}

Variant definitions (Build Plan §3.2 table):
    classic — original Croston 1972; known upward bias
    SBA     — Syntetos-Boylan Approximation; bias-corrected; DEFAULT
    TSB     — Teunter-Syntetos-Babai; models extinction probability;
              iterates over ALL periods (including zeros) — not just demand events

EDGE CASE (if it has sales data instead of zero's or only one sale (which means not enough data) ):
    Trigger: len(non_zero_indices) < 2  (0 or 1 non-zero demand events)
    Fix: fall back to SESModel silently. DO NOT raise. DO NOT change
    ctx.pattern_label. Log 'no_zero_interval_detected' at INFO level.
    The Sub-Stage 9.3 caller receives a valid result as if Croston ran normally.

"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from models.base import BaseModel, ModelFitError
from models.bootstrap import bootstrap_quantiles
from models.ses import SESModel
from infrastructure.constants import HORIZONS, FORECAST_COLUMN_MAP, CROSTON_INTERVAL_FLOOR

log = logging.getLogger(__name__)

__all__ = ["CrostonMethod"]

# ---------------------------------------------------------------------------
# HP search space — 9 configurations
# ---------------------------------------------------------------------------
_HP_SEARCH_SPACE: list[dict] = [
    {"alpha": a, "interval_type": it}
    for a in [0.05, 0.10, 0.20]
    for it in ["classic", "SBA", "TSB"]
]
_DEFAULT_HP: dict = {"alpha": 0.10, "interval_type": "SBA"}


class CrostonMethod(BaseModel):
    """
    Croston's Method with SBA and TSB bias-correction variants.

    Variant behaviour:
        'classic' — iterates over demand events only; known upward bias
        'SBA'     — same iteration as classic, multiplied by (1 − alpha/2) correction
        'TSB'     — iterates over ALL time periods; tracks demand probability;
                    more appropriate when demand may become permanently zero

    """

    def __init__(self, hp: dict) -> None:
        super().__init__(hp)
        self._daily_rate: float = 0.0
        self._demand_smooth: float = 0.0
        self._interval_smooth: float = 1.0

        # fallback state
        self._using_fallback: bool = False
        self._fallback: Optional[SESModel] = None

    # -----------------------------------------------------------------------
    # 5 required properties
    # -----------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "Croston's Method"

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
        Fit Croston's Method on the demand series.

        fewer than 2 non-zero demand events → fall back to SES.
        TSB iterates ALL time steps; classic/SBA iterate non-zero events only.
        sample_weights silently ignored.
        """
        try:
            alpha = float(self.hp.get("alpha", 0.10))
            interval_type = str(self.hp.get("interval_type", "SBA"))
            # Sanitise: NaN/Inf replaced with 0 — corrupt rows must not propagate.
            series = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            non_zero_indices = np.where(series > 0)[0]

            # ------------------------------------------------------------------
            # MANDATORY CHECK
            # ------------------------------------------------------------------
            # Trigger: 0 or 1 non-zero events → cannot compute an inter-demand
            # interval. Fall back to SESModel with smoothing_level=0.3 silently.
            # DO NOT raise. DO NOT change ctx.pattern_label upstream.
            if len(non_zero_indices) < 2:
                log.info(
                    "CrostonMethod FallBack: only %d non-zero event(s) in series "
                    "(len=%d) — falling back to SES internally. "
                    "Caller should log 'no_zero_interval_detected'.",
                    len(non_zero_indices), len(series),
                )
                self._using_fallback = True
                self._fallback = SESModel(hp={"smoothing_level": 0.3})
                self._fallback.fit(df, features)
                self._daily_rate = self._fallback.level
                return

            # Standard path — sufficient demand events for Croston
            self._using_fallback = False
            self._fallback = None

            if interval_type == "TSB":
                # TSB MUST iterate over every time step (including zeros).
                # It tracks p_t (demand probability) which decays during zero runs.
                self._fit_tsb(series, alpha, non_zero_indices)
            else:
                # Classic and SBA iterate over demand events only.
                self._fit_classic_sba(series, alpha, interval_type, non_zero_indices)

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"CrostonMethod.fit failed "
                f"[alpha={self.hp.get('alpha')}, type={self.hp.get('interval_type')}]: {exc}"
            ) from exc

    def predict(
            self,
            df: pd.DataFrame,
            features: list[str],
            horizon: int,
    ) -> np.ndarray:
        """
        Return flat daily-rate forecast array of length `horizon`.

        Croston produces the same daily rate for all future periods — no trend,
        no seasonality. Done Criterion D6: np.ndarray, len == horizon, all ≥ 0.
        """
        try:
            if self._using_fallback and self._fallback is not None:
                return self._fallback.predict(df, features, horizon)
            rate = max(0.0, self._daily_rate)
            return np.full(horizon, rate, dtype=float)
        except Exception as exc:
            raise ModelFitError(f"CrostonMethod.predict failed: {exc}") from exc

    def predict_all_horizons(
            self,
            df: pd.DataFrame,
            features: list[str],
            oos_factor: float = 1.0,
    ) -> dict:
        """
        Generate cumulative demand for all 8 HORIZONS.

        Delegates to SES if fallback con was triggered.
        """
        try:
            if self._using_fallback and self._fallback is not None:
                # SES fallback returns pattern='stable'; we override to 'intermittent'
                # by calling bootstrap_quantiles separately.
                level = max(0.0, self._fallback.level)
                residuals = self._fallback.compute_residuals(df, features)
                result: dict = {}
                for H in HORIZONS:
                    col = FORECAST_COLUMN_MAP[H]
                    point = level * H * oos_factor
                    result[col] = bootstrap_quantiles(point, residuals, "intermittent")
                return result

            rate = max(0.0, self._daily_rate)
            residuals = self.compute_residuals(df, features)
            result = {}

            for H in HORIZONS:
                col = FORECAST_COLUMN_MAP[H]
                point = rate * H * oos_factor
                result[col] = bootstrap_quantiles(point, residuals, "intermittent")

            return result

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"CrostonMethod.predict_all_horizons failed: {exc}"
            ) from exc

    def compute_residuals(
            self,
            df: pd.DataFrame,
            features: list[str],
    ) -> np.ndarray:
        """
        Re-fit on df and return actual − fitted for the last 30 rows.

        For Croston, the fitted value for every row is the constant daily_rate
        (intermittent demand is modelled as a rate process, not per-day).
        """
        try:
            self.fit(df, features)

            if self._using_fallback and self._fallback is not None:
                return self._fallback.compute_residuals(df, features)

            rate = max(0.0, self._daily_rate)
            fitted = np.full(len(df), rate, dtype=float)
            # Sanitise actual values — NaN/Inf in residuals corrupt bootstrap_quantiles.
            resids = np.nan_to_num(df["qty"].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0) - fitted
            return resids[-30:]

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"CrostonMethod.compute_residuals failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Internal helpers — variant-specific fitting routines
    # -----------------------------------------------------------------------

    def _fit_classic_sba(
            self,
            series: np.ndarray,
            alpha: float,
            interval_type: str,
            non_zero_indices: np.ndarray,
    ) -> None:
        """
        Fit classic Croston (or SBA bias-correction) by iterating over demand events.

        Maintains two exponentially smoothed estimates:
            demand_smooth   — average demand SIZE on demand days
            interval_smooth — average INTERVAL (days) between demand days

        SBA correction:
            daily_rate = demand_smooth / interval_smooth × (1 − alpha/2)
            This removes the systematic upward bias in the classic formula.
        """
        # Initialise at first demand event
        demand_smooth = float(series[non_zero_indices[0]])
        interval_smooth = 1.0  # initialise interval to 1 day

        for i in range(1, len(non_zero_indices)):
            idx = non_zero_indices[i]
            prev_idx = non_zero_indices[i - 1]
            interval = float(idx - prev_idx)  # days since last demand
            demand = float(series[idx])

            # Exponential smoothing update (same formula for both variants)
            demand_smooth = alpha * demand + (1.0 - alpha) * demand_smooth
            interval_smooth = alpha * interval + (1.0 - alpha) * interval_smooth

        self._demand_smooth = demand_smooth
        self._interval_smooth = max(interval_smooth, CROSTON_INTERVAL_FLOOR)

        if interval_type == "SBA":
            # Syntetos-Boylan bias correction: multiply by (1 − alpha/2)
            raw_rate = demand_smooth / self._interval_smooth
            self._daily_rate = raw_rate * (1.0 - alpha / 2.0)
        else:
            # Classic (no correction)
            self._daily_rate = demand_smooth / self._interval_smooth

    def _fit_tsb(
            self,
            series: np.ndarray,
            alpha: float,
            non_zero_indices: np.ndarray,
    ) -> None:
        """
        Fit Teunter-Syntetos-Babai (TSB) variant by iterating over ALL time steps.

        TSB tracks two estimates:
            z_t — average demand SIZE (updated on demand events only)
            p_t — demand PROBABILITY  (updated on EVERY time step, including zeros)

        daily_rate = p_t × z_t

        TSB is preferable when demand may become permanently zero (e.g. product
        approaching end-of-life), because p_t decays toward zero during extended
        zero runs, eventually predicting near-zero demand.

        Iteration MUST cover every period (not just non-zero events) — this is
        the defining difference from classic/SBA.
        """
        # Initialise estimates at the first demand event
        first_idx = non_zero_indices[0]
        z = float(series[first_idx])  # initial demand size
        # Initial probability: 1/(days since launch + 1), clamped to (0, 1]
        p = 1.0 / float(first_idx + 1) if first_idx > 0 else 1.0

        # Iterate over ALL time steps from the first demand event onward
        for t in range(first_idx + 1, len(series)):
            demand_occurred = float(series[t]) > 0.0

            # p_t update: alpha × I(demand) + (1 − alpha) × p_(t−1)
            p = alpha * (1.0 if demand_occurred else 0.0) + (1.0 - alpha) * p

            # z_t update: only on demand events
            if demand_occurred:
                z = alpha * float(series[t]) + (1.0 - alpha) * z

        self._demand_smooth = z
        self._interval_smooth = max(p, CROSTON_INTERVAL_FLOOR)  # reused field for p
        # TSB daily rate: probability × average demand size
        self._daily_rate = z * p
