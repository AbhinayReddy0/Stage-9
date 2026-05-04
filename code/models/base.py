"""
models/base.py
====================================================
Abstract base class and shared error type for all Stage 9 forecasting models.

All five concrete model classes (NaiveForecast, SESModel, CrostonMethod,
HoltLinearTrend, ProphetModel) inherit from BaseModel and must implement
every abstract method and property defined here.

Sub-Stages 9.3 (Thompson Sampling), 9.4 (Backtesting), and 9.5 (Forecast
Generation) interact with models exclusively through the five methods and five
properties defined in this contract. No sub-stage ever accesses model-internal
attributes directly.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from infrastructure.constants import Model
from infrastructure.errors import Stage9Error

log = logging.getLogger(__name__)

__all__ = ["BaseModel", "ModelFitError", "get_model_class"]


# ===========================================================================
# Custom exception — the ONLY exception that may escape the model layer
# ===========================================================================

class ModelFitError(Stage9Error):
    """
    Raised when model.fit() or model.predict() encounters an unrecoverable error.

    Critical Rules:
    - Every exception raised inside fit() or predict() MUST be caught and
      re-raised as ModelFitError. Raw exceptions (ValueError, RuntimeError,
      Stan errors, statsmodels errors) MUST NOT escape the model layer.
    - Sub-Stage 9.3 catches ModelFitError and assigns mape = 1.0 for that
      HP config, then continues. Thompson penalises the config.
    - Sub-Stage 9.4 catches ModelFitError from compute_residuals() and applies
      PATTERN_FEEDBACK_PROXY_MAPE (0.50) as the fallback backtest MAPE.
    - ModelFitError inherits from Stage9Error (not from ValueError/KeyError)
      to prevent silent swallowing by accidental bare `except ValueError` blocks.
    """


# ===========================================================================
# Abstract base class — the complete model contract
# ===========================================================================

class BaseModel(ABC):
    """
    Interface that every Stage 9 forecasting model must satisfy.

    Concrete subclasses:
        NaiveForecast      → cold_start pattern
        CrostonMethod      → intermittent pattern
        NeuralProphetModel → seasonal pattern  (Prophet internally)
        HoltLinearTrend    → trending pattern
        SESModel           → stable / steady pattern

    ProcessPool safety (ProphetModel only ):
        ProphetModel must be defined at module top level (not as an inner class
        or closure) so it is picklable by ProcessPoolExecutor. ThreadPool models
        (Naive, SES, Croston, Holt) do not have this requirement.

    Method visibility:
        Sub-stages call ONLY the five public methods and read ONLY the five
        public properties. Private helpers (prefix _) are implementation
        details — never referenced externally.
    """

    def __init__(self, hp: dict) -> None:
        """
        Store the hyperparameter dict. DO NOT train. DO NOT touch the database.

        Args:
            hp: Complete HP configuration dict for this specific configuration.
                Thompson Sampling builds these dicts from hp_search_space and
                passes them here. Each concrete model defines the valid keys
                it expects and their defaults.
        """
        self.hp: dict = hp

    # -----------------------------------------------------------------------
    # 5 required properties
    # -----------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Human-readable model identifier stored in forecasts.assigned_model.

        Must EXACTLY match the human-readable values documented in the build
        plan. Case-sensitive. Sub-Stage 9.1 validates this against
        PATTERN_MODEL_MAP after model instantiation.

        Valid values:
            "Naive Forecast"
            "Croston's Method"
            "Prophet"
            "Holt's Linear Trend"
            "Simple Exponential Smoothing (SES)"
        """

    @property
    @abstractmethod
    def hp_search_space(self) -> list[dict]:
        """
        Complete list of candidate HP dicts for Thompson Sampling.

        Every dict must be a complete, valid HP configuration — no partial
        configs. Thompson Sampling iterates this list in select_configs().
        default_hp must be a valid entry in this list.

        Naive:  9 configs  (3 lag_periods × 3 smoothing_methods)
        Croston: 9 configs  (3 alpha × 3 interval_types)
        Prophet: 24 configs (2×2×2×3 — weekly/yearly/mode/cps)
        Holt:   24 configs  (4 alpha × 3 beta × 2 damped)
        SES:     5 configs  (5 alpha values)
        """

    @property
    @abstractmethod
    def default_hp(self) -> dict:
        """
        Starting HP used before Thompson Sampling has accumulated evidence.

        Used for:
          - New products on their first run (no Thompson state yet)
          Must be a valid entry in hp_search_space. Always return a copy so
        callers cannot mutate the class-level default.
        """

    @property
    @abstractmethod
    def required_features(self) -> list[str]:
        """
        Feature columns the model always needs. Minimum: ['date', 'qty'].

        Sub-Stage 9.2 never drops required features — reliability filtering
        in Step 1 applies to optional_features only. Sub-Stage 9.2 guarantees
        these columns are present in df before calling fit().
        """

    @property
    @abstractmethod
    def optional_features(self) -> list[str]:
        """
        Feature columns the model can use if they pass the reliability filter.

        Examples: ['promo_flag', 'day_of_week', 'is_weekend', 'price',
                   'discount_pct']

        Sub-Stage 9.2 Step 1 drops optional features whose reliability score
        is below feature_reliability_floor. Sub-Stage 9.2 Step 4 further
        filters by MAPE improvement. Silently ignored if absent from df.
        """

    # -----------------------------------------------------------------------
    # 5 required methods
    # -----------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        features: list[str],
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the model on the provided DataFrame.

        Args:
            df:             Training data. Always contains 'date' (datetime)
                            and 'qty' (float ≥ 0). May contain optional feature
                            columns. Rows are in chronological order.
            features:       Column names to use. Always a superset of
                            required_features. Sub-Stage 9.2 has already
                            filtered to passing optional features.
            sample_weights: Optional array of the same length as df. Only
                            ProphetModel uses this (promo weighting from
                            Sub-Stage 9.2). All other models MUST accept the
                            parameter and MUST ignore it silently — never raise
                            on an ignored argument.

        Raises:
            ModelFitError: Wrap ALL exceptions from the underlying library
                           (Stan, statsmodels, numpy) and re-raise. Never
                           swallow — Sub-Stage 9.3 needs the signal.

        Note:
            Does NOT commit to the database. Does NOT call BatchWriter.
            All DB writes happen in Sub-Stage 9.5 via BatchWriter.
        """

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        features: list[str],
        horizon: int,
    ) -> np.ndarray:
        """
        Return a daily-demand point-forecast array of length `horizon`.

        Uses the fitted state from the most recent fit() call. Does NOT re-fit.

        Args:
            df:      The DataFrame the model was fitted on (or held-out
                     validation data during backtesting in Sub-Stage 9.3).
                     Some models (Naive, SES, Holt) use df only to extract
                     the final fitted level/trend; others ignore it.
            features: Column names consistent with fit().
            horizon: Number of future days. During Sub-Stage 9.3 HP testing
                     this is always 14 (validation holdout). During Sub-Stage
                     9.4 backtesting it equals the backtest window (≤ 60 days).

        Returns:
            np.ndarray of shape (horizon,). Index 0 = day 1, index H-1 = day H.
            All values ≥ 0. Clamp negatives to 0 before returning.
            Done Criterion D6: isinstance(result, np.ndarray) and len == horizon.

        Raises:
            ModelFitError: any exception during prediction.
        """

    @abstractmethod
    def predict_all_horizons(
        self,
        df: pd.DataFrame,
        features: list[str],
        oos_factor: float = 1.0,
    ) -> dict:
        """
        Produce the complete forecast dict for all 8 HORIZONS.

        Called ONCE per SKU by Sub-Stage 9.5. Uses the model's current fitted
        state — does NOT call fit() internally (that would re-train unnecessarily).

        The oos_factor corrects for suppressed historical demand during stock-outs:
            oos_factor = 1 + (oos_pct_of_history × detection_confidence)
            'intermittent' are excluded from this; that mean multiplied by 1.
        Sub-Stage 9.1 caps oos_factor at 1.50 before passing it here.

        Args:
            df:         Full training DataFrame (the model has already been
                        fitted on this data by Sub-Stage 9.5's explicit fit call).
            features:   Feature list used during fit().
            oos_factor: Applied to ALL point forecasts BEFORE bootstrap. Default 1.0.

        Returns:
            dict with EXACTLY 8 string keys (Done Criterion D7):
                'forecast_7d', 'forecast_14d', 'forecast_30d', 'forecast_60d',
                'forecast_90d', 'forecast_150d', 'forecast_180d', 'forecast_365d'
            Each value is a dict:
                {'mean': float, 'p50': float, 'p80': float, 'p90': float}
            All floats ≥ 0. Invariant: p50 ≤ p80 ≤ p90 (enforced by bootstrap_quantiles).

        Raises:
            ModelFitError: any exception during horizon computation.
        """

    @abstractmethod
    def compute_residuals(
        self,
        df: pd.DataFrame,
        features: list[str],
    ) -> np.ndarray:
        """
        Fit the model on df and return (actual − fitted) for the last 30 rows.

        Called by Sub-Stage 9.4 to obtain residuals for bootstrap_quantiles()
        in Sub-Stage 9.5. Deliberately re-fits to ensure residuals reflect the
        complete training data, not a train-split subset.

        Implementation pattern for every concrete model:
            self.fit(df, features)
            fitted = self._get_fitted_values(df)
            residuals = df['qty'].values - fitted
            return residuals[-30:]      # last 30 rows ONLY

        Why last 30 only :
            More residuals add noise rather than signal to bootstrap resampling.
            30 rows captures recent model accuracy without diluting it with
            distant history where the model may have been less calibrated.

        Returns:
            np.ndarray (it may be empty if df has < 30 rows — bootstrap_quantiles
            handles len < 3 via log-normal proxy). Never raises on short series.

        Raises:
            ModelFitError: any exception during residual computation.
        """


# ---------------------------------------------------------------------------
# Model registry — lazy import to avoid circular deps at import time
# ---------------------------------------------------------------------------

_MODEL_CLASS: dict[str, type] | None = None


def get_model_class(model_name: str) -> type:
    """Return the concrete model class for a given model_name string.

    Lazy-imports all five model modules on first call so this module can be
    imported without triggering transitive imports of Prophet, statsmodels, etc.
    Falls back to SESModel for any unrecognised name (safe default for workers).
    """
    global _MODEL_CLASS
    if _MODEL_CLASS is None:
        from models.croston import CrostonMethod
        from models.holt import HoltLinearTrend
        from models.naive import NaiveForecast
        from models.prophet_model import ProphetModel
        from models.ses import SESModel
        _MODEL_CLASS = {
            Model.NAIVE:        NaiveForecast,
            Model.CROSTON:      CrostonMethod,
            Model.PROPHET:      ProphetModel,
            Model.HOLTS_LINEAR: HoltLinearTrend,
            Model.SES:          SESModel,
        }
    from models.ses import SESModel as _SES
    return _MODEL_CLASS.get(model_name, _SES)
