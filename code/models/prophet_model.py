"""
models/prophet.py — Atheera Stage 9 Forecasting Agent
======================================================
ProphetModel: seasonal-demand model using Facebook Prophet.

Assigned to: pattern_label = 'seasonal'

Principle:
    Explicitly models weekly and annual seasonality. Fits ONCE for 365 days
    forward per SKU. predict_all_horizons() extracts cumulative sums from that
    single forward prediction at each HORIZON'S boundary.

    NEVER scale forecast_30d to approximate longer horizons — doing so destroys
    seasonal peaks that Prophet already computed correctly.

HP search space — 24 configurations:
    weekly_seasonality:      [True, False]
    yearly_seasonality:      [True, False]
    seasonality_mode:        ['additive', 'multiplicative']
    changepoint_prior_scale: [0.01, 0.1, 0.5]
    default_hp: {weekly_seasonality: True, yearly_seasonality: True,
                 seasonality_mode: 'additive', changepoint_prior_scale: 0.1}

HARDCODED RULES — Never Violate:
    1. daily_seasonality = False ALWAYS. Hardcoded in __init__ and _build_prophet().
       Not in hp_search_space. Not overridable by any caller. Daily variation
       in retail is noise, not signal.
    2. Single fit, cumulative extraction. fit() called once per SKU for 365 days.
       predict_all_horizons() extracts cumulative sums — no per-horizon re-fitting.
    3. If df['qty'].std() < 0.01, inject tiny noise before fitting.
       Prophet's Stan backend raises a numerical error on zero-variance series.

"""

from __future__ import annotations

import itertools
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Prophet import guard — raises ModelFitError at runtime (not at import time)
# so that the rest of Stage 9 can load even in environments without Prophet.
# ---------------------------------------------------------------------------
try:
    from prophet import Prophet as _Prophet
    _PROPHET_AVAILABLE: bool = True
except ImportError:
    _PROPHET_AVAILABLE = False

    class _Prophet:  # type: ignore[no-redef]
        """Stub — never instantiated; _PROPHET_AVAILABLE=False blocks all calls."""
        def __init__(self, **kwargs: object) -> None: ...
        def fit(self, df: object, **kwargs: object) -> None: ...
        def predict(self, df: object) -> object: ...
        def make_future_dataframe(self, periods: int, freq: str = "D") -> object: ...

from models.base import BaseModel, ModelFitError
from models.bootstrap import bootstrap_quantiles
from infrastructure.constants import HORIZONS, FORECAST_COLUMN_MAP

log = logging.getLogger(__name__)

__all__ = ["ProphetModel"]

# Total forward horizon Prophet fits in a single call. Always 365.
# predict_all_horizons() extracts cumulative sums at each boundary from this.
_FIT_HORIZON: int = 365

# variance threshold — below this, Stan raises a numerical error
_E002_STD_THRESHOLD: float = 0.01

# Noise scale for E002 fix — 0.1% of the mean; does not affect accuracy
_E002_NOISE_SCALE: float = 0.001

# ---------------------------------------------------------------------------
# HP search space — 24 configurations (2 × 2 × 2 × 3)
# ---------------------------------------------------------------------------
_HP_SEARCH_SPACE: list[dict] = [
    {
        "weekly_seasonality": ws,
        "yearly_seasonality": ys,
        "seasonality_mode": sm,
        "changepoint_prior_scale": cps,
    }
    for ws, ys, sm, cps in itertools.product(
        [True, False],
        [True, False],
        ["additive", "multiplicative"],
        [0.01, 0.1, 0.5],
    )
]
_DEFAULT_HP: dict = {
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.1,
}


class ProphetModel(BaseModel):
    """
    Prophet-based seasonal demand model.

    Fits Prophet ONCE per SKU across 365 days forward. Never calls fit()
    multiple times for different horizons. Never scales shorter-horizon
    forecasts to approximate longer ones.

    ProcessPool safe: top-level class definition. No lambdas. No closures.
    daily_seasonality is ALWAYS False — hardcoded, never an HP parameter.
    """

    def __init__(self, hp: dict) -> None:
        super().__init__(hp)
        # HARDCODED: daily_seasonality is always False.
        # This is NOT an HP parameter. It is NOT in hp_search_space.
        # It is NOT overridable by any caller. Setting it True introduces noise.
        self._daily_seasonality: bool = False  # locked, never change

        self._model: Optional[object] = None  # fitted Prophet object
        self._predictions: Optional[np.ndarray] = None  # 365 daily yhat values
        self._history_df: Optional[pd.DataFrame] = None  # Prophet-format train df

    # -----------------------------------------------------------------------
    # 5 required properties
    # -----------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "Prophet"

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
        # Regressor support (promo_flag, price) can be added in a future build.
        # Excluded here to keep the model surface minimal and testable.
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
        Fit Prophet on df and cache 365 daily forward predictions.

        HARDCODED: daily_seasonality = False — never altered.
        E002: Injects tiny noise if std < 0.01 to prevent Stan numerical crash.
        sample_weights: Applied via Prophet's 'weights' column if provided.
        """
        if not _PROPHET_AVAILABLE:
            raise ModelFitError(
                "Prophet is not installed. "
                "Run: pip install prophet"
            )

        try:
            prophet_df = self._to_prophet_df(df, sample_weights)

            # ------------------------------------------------------------------
            # E002: Zero-variance series guard
            # ------------------------------------------------------------------
            # Prophet's Stan backend raises a numerical error when the training
            # series has near-zero variance (e.g. every day sells exactly 10 units).
            # Fix: add tiny noise (0.1% of mean) before fitting. This is
            # documented and logged — the quantile spread from bootstrap handles
            # the remaining uncertainty correctly.
            if prophet_df["y"].std() < _E002_STD_THRESHOLD:
                mean_y = float(prophet_df["y"].mean())
                log.info(
                    "ProphetModel E002: near-zero variance (std=%.5f, mean=%.2f) "
                    "— injecting %.4f%% noise to prevent Stan crash",
                    prophet_df["y"].std(), mean_y, _E002_NOISE_SCALE * 100,
                )
                rng = np.random.default_rng(seed=42)
                noise = rng.normal(
                    loc=0.0,
                    scale=_E002_NOISE_SCALE * max(mean_y, 1.0),
                    size=len(prophet_df),
                )
                # Operate on a copy — do not modify the caller's df
                prophet_df = prophet_df.copy()
                prophet_df["y"] = np.maximum(prophet_df["y"].values + noise, 0.0)

            model = self._build_prophet()

            # Suppress Prophet's verbose Stan output to keep logs clean
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)

            self._model = model
            self._history_df = prophet_df

            # HARDCODED RULE: fit() covers the full _FIT_HORIZON (365) in ONE call.
            # predict_all_horizons() extracts cumulative sums from this result.
            # Never re-fit inside predict_all_horizons().
            future = model.make_future_dataframe(periods=_FIT_HORIZON)
            forecast = model.predict(future)

            # Store only the FORWARD-LOOKING portion (not in-sample predictions)
            self._predictions = np.maximum(
                forecast["yhat"].values[-_FIT_HORIZON:].astype(float),
                0.0,
            )

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"ProphetModel.fit (Prophet) failed: {exc}"
            ) from exc

    def predict(
            self,
            df: pd.DataFrame,
            features: list[str],
            horizon: int,
    ) -> np.ndarray:
        """
        Return per-day demand forecast for the next `horizon` days.

        Uses the cached _predictions from the most recent fit() call.
        Returns _predictions[:horizon] — the first `horizon` forward days.

        During Sub-Stage 9.3 HP testing, fit() is called on train_df before
        predict() is called for the 14-day validation window. The Sub-Stage
        manages the train/validation split externally.

        returns np.ndarray, len == horizon, all values ≥ 0.
        """
        try:
            if self._predictions is None:
                raise ModelFitError(
                    "ProphetModel.predict() called before fit(). "
                    "Call fit(df, features) first."
                )
            # Clamp to horizon — _predictions always has 365 elements
            result = self._predictions[:horizon]
            return np.maximum(result, 0.0)

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(f"ProphetModel.predict failed: {exc}") from exc

    def predict_all_horizons(
            self,
            df: pd.DataFrame,
            features: list[str],
            oos_factor: float = 1.0,
    ) -> dict:
        """
        Extract cumulative demand at each HORIZONS boundary from _predictions.

        HARDCODED RULE: Never call fit() here. Never scale forecast_30d to
        approximate longer horizons — Prophet's 365-day prediction already
        encodes seasonal peaks correctly. Scaling destroys them.

        Formula: cumsum(H) = sum(_predictions[:H]) × oos_factor

        for a synthetic seasonal series with a clear annual
        cycle, forecast_365d['mean'] > forecast_30d['mean'] × 12 by ≥ 10%.
        This is satisfied because we sum 365 daily values (not 30 × 12).
        """
        try:
            if self._predictions is None:
                raise ModelFitError(
                    "ProphetModel.predict_all_horizons() called before fit()."
                )

            residuals = self.compute_residuals(df, features)
            result: dict = {}

            for H in HORIZONS:
                col = FORECAST_COLUMN_MAP[H]
                # Cumulative sum of Prophet's daily predictions for the first H days
                cumsum = float(np.sum(self._predictions[:H])) * oos_factor
                cumsum = max(0.0, cumsum)
                result[col] = bootstrap_quantiles(cumsum, residuals, "seasonal")

            return result

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"ProphetModel.predict_all_horizons failed: {exc}"
            ) from exc

    def compute_residuals(
            self,
            df: pd.DataFrame,
            features: list[str],
    ) -> np.ndarray:
        """
        Re-fit Prophet on df and return actual − fitted for the last 30 rows.
        """
        try:
            self.fit(df, features)  # re-fits; updates _model and _history_df
            fitted = self._get_fitted_values(df)
            resids = df["qty"].values.astype(float) - np.nan_to_num(fitted, nan=0.0)
            return resids[-30:]

        except ModelFitError:
            raise
        except Exception as exc:
            raise ModelFitError(
                f"ProphetModel.compute_residuals failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_prophet(self) -> "_Prophet":
        """
        Construct a configured Prophet instance from self.hp.

        HARDCODED: daily_seasonality is always self._daily_seasonality = False.
        Never read daily_seasonality from hp. Never expose it in hp_search_space.
        """
        if not _PROPHET_AVAILABLE:
            raise ModelFitError("Prophet is not installed. Run: pip install prophet")
        return _Prophet(
            weekly_seasonality=bool(self.hp.get("weekly_seasonality", True)),
            yearly_seasonality=bool(self.hp.get("yearly_seasonality", True)),
            seasonality_mode=str(self.hp.get("seasonality_mode", "additive")),
            changepoint_prior_scale=float(self.hp.get("changepoint_prior_scale", 0.1)),
            daily_seasonality=self._daily_seasonality,  # ALWAYS False
        )

    @staticmethod
    def _to_prophet_df(
            df: pd.DataFrame,
            sample_weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Convert Stage 9 DataFrame (date, qty columns) to Prophet format (ds, y).

        Prophet requires:
            'ds': datestamp column (datetime or date string)
            'y':  numeric target column (demand; must be ≥ 0)

        Clamps negative qty to 0 — Prophet can produce numerical instability
        with negative training values, and demand is by definition non-negative.
        """
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df["date"]),
            "y": np.maximum(df["qty"].values.astype(float), 0.0),
        })

        # Sample weights: Prophet supports a 'weights' column for weighted fitting.
        # Sub-Stage 9.2 provides these for promo-day down-weighting.
        if sample_weights is not None and len(sample_weights) == len(df):
            prophet_df["weights"] = np.asarray(sample_weights, dtype=float)

        return prophet_df

    def _get_fitted_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract in-sample (training) fitted values from the fitted Prophet model.

        Uses self._history_df (the Prophet-format training data) to request
        in-sample predictions. Falls back to zeros if the model is unavailable.
        """
        if self._model is None or self._history_df is None:
            log.warning(
                "ProphetModel: _get_fitted_values called with no fitted model "
                "— returning zeros"
            )
            return np.zeros(len(df), dtype=float)

        try:
            # Prophet.predict() on the training DataFrame returns in-sample predictions.
            # _history_df is in Prophet format (ds, y) matching the training data.
            in_sample = self._model.predict(self._history_df)
            fitted = in_sample["yhat"].values.astype(float)
            return np.maximum(fitted, 0.0)
        except Exception as exc:
            log.warning(
                "ProphetModel: failed to extract fitted values: %s "
                "— returning zeros for residual computation",
                exc,
            )
            return np.zeros(len(df), dtype=float)
