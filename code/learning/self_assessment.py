"""
self_assessment.py
=======================================================
SelfAssessmentEngine: runs in REPORTING state after all SKUs are processed.

Responsibilities:
  - Read model_performance_s9 to detect degrading models
  - Compute run statistics from in-memory SKUResult list
  - Write one row to stage9.stage9_self_assessment (direct conn, never BatchWriter)
  - Return SelfAssessmentResult containing model_health dict for emit_model_health
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from infrastructure.constants import (
    MODEL_DEGRADATION_MAPE_DELTA,
    SELF_ASSESSMENT_CACHE_HIT_RATE_WARN,
    SELF_ASSESSMENT_FALLBACK_RATE_WARN,
    SELF_ASSESSMENT_SUCCESS_RATE_CRITICAL,
    ForecastStatus,
    Model,
    ProcessingTier,
)

log = logging.getLogger(__name__)

__all__ = [
    "SKUResult",
    "ModelHealthEntry",
    "SelfAssessmentResult",
    "SelfAssessmentEngine",
]

# All primary model names iterated when building model_health dict.
_ALL_MODEL_NAMES: list[str] = [
    Model.NAIVE,
    Model.CROSTON,
    Model.PROPHET,
    Model.HOLTS_LINEAR,
    Model.SES,
]

_PRIMARY_HORIZON: int = 30  # degradation evaluated at the 30-day horizon

_INSERT_SQL = """
    INSERT INTO stage9.stage9_self_assessment (
        tenant_id, run_id,
        avg_mape_this_run, avg_mape_prev_run, mape_delta_pct,
        degradation_detected, recommendations, model_health_summary,
        total_skus_processed, cache_tier_count, partial_tier_count,
        full_tier_count, fallback_count, pattern_feedback_retry_count,
        execution_mode, run_duration_seconds
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (tenant_id, run_id) DO NOTHING;
"""


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class SKUResult:
    """One entry per SKU produced by the ACTING state. Passed into run()."""
    sku_id: str
    status: str  # ForecastStatus.*
    confidence_final: float
    processing_tier: str  # ProcessingTier.*
    assigned_model: str
    used_fallback: bool
    pattern_label: str
    backtest_mape: Optional[float] = None


@dataclass
class ModelHealthEntry:
    """Per-model health record derived from model_performance_s9."""
    model_name: str
    avg_mape_30d: Optional[float]  # None when no data yet (first run)
    prior_avg_mape: Optional[float]
    trend: str  # 'improving' / 'stable' / 'degrading' / 'unknown'
    mape_delta: Optional[float]  # current - prior; positive = worse
    is_degrading: bool


@dataclass
class SelfAssessmentResult:
    """Full output of SelfAssessmentEngine.run()."""
    tenant_id: str
    run_id: str

    # Model health
    model_health: dict[str, ModelHealthEntry]
    degrading_models: list[str]
    recommendations: list[str]

    # Run statistics
    total_skus_processed: int
    cache_hit_count: int
    partial_count: int
    full_count: int
    pct_auto_proceed: float
    pct_needs_review: float
    avg_confidence_all: float
    high_fallback_count: int
    pattern_feedback_failures_count: int

    # Alert flags
    high_fallback_alert: bool
    low_cache_hit_alert: bool
    low_success_rate_alert: bool
    degradation_detected: bool

    # Timing
    run_duration_seconds: float
    execution_mode: str


# ===========================================================================
# Engine
# ===========================================================================

class SelfAssessmentEngine:
    """
    Evaluates Stage 9 run health and writes stage9_self_assessment.

    Instantiated once per run and called from the REPORTING state handler.
    """

    def run(
            self,
            tenant_id: str,
            run_id: str,
            results: list[SKUResult],
            pattern_feedback_failures_count: int,
            run_start_time: float,
            execution_mode: str,
            conn: Any,
    ) -> SelfAssessmentResult:
        """
        Run the full self-assessment and write results to the DB.

        Args:
            tenant_id:                        Tenant UUID string.
            run_id:                           Run UUID string.
            results:                          One SKUResult per processed SKU.
            pattern_feedback_failures_count:  Count of pattern_feedback writes that
                                              failed all 3 retries in Sub-Stage 9.4.
            run_start_time:                   time.time() captured at run start.
            execution_mode:                   'full' or 'micro_update'.
            conn:                             Open psycopg2 connection.

        Returns:
            SelfAssessmentResult with all fields populated.
        """
        if pattern_feedback_failures_count > 0:
            log.warning(
                "self_assessment tenant=%s run=%s: %d pattern_feedback write(s) "
                "failed after 3 retries — possible DB write latency issue",
                tenant_id, run_id, pattern_feedback_failures_count,
            )

        model_perf_rows = self._fetch_model_performance(tenant_id, conn)
        model_health = self._compute_model_health(model_perf_rows)
        run_stats = self._compute_run_statistics(results)
        degrading = [name for name, e in model_health.items() if e.is_degrading]
        recommendations = self._build_recommendations(degrading, model_health, run_stats)

        assessment = SelfAssessmentResult(
            tenant_id=tenant_id,
            run_id=run_id,
            model_health=model_health,
            degrading_models=degrading,
            recommendations=recommendations,
            degradation_detected=len(degrading) > 0,
            run_duration_seconds=time.time() - run_start_time,
            execution_mode=execution_mode,
            pattern_feedback_failures_count=pattern_feedback_failures_count,
            **run_stats,
        )

        self._write_assessment(assessment, conn)
        return assessment

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_model_performance(
            tenant_id: str, conn: Any
    ) -> dict[str, Any]:
        """
        Fetch one row per assigned_model at horizon_days=30.

        Returns dict keyed by assigned_model. Empty dict on first run or DB error.
        """
        sql = """
            SELECT assigned_model, avg_mape_30d, trend, mape_delta, sample_count
              FROM stage9.model_performance_s9
             WHERE tenant_id   = %s
               AND horizon_days = %s
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tenant_id, _PRIMARY_HORIZON))
                rows = cur.fetchall()
            return {
                row[0]: {
                    "avg_mape_30d": float(row[1]) if row[1] is not None else None,
                    "trend": row[2] or "unknown",
                    "mape_delta": float(row[3]) if row[3] is not None else None,
                    "sample_count": row[4],
                }
                for row in rows
            }
        except Exception as exc:
            log.error(
                "self_assessment _fetch_model_performance failed tenant=%s: %s",
                tenant_id, exc,
            )
            return {}

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_model_health(
            model_perf_rows: dict[str, Any]
    ) -> dict[str, ModelHealthEntry]:
        """
        Build one ModelHealthEntry per model in _ALL_MODEL_NAMES.

        Entries for models not in model_perf_rows get trend='unknown' and
        is_degrading=False — no data is not the same as degradation.
        """
        health: dict[str, ModelHealthEntry] = {}

        for model_name in _ALL_MODEL_NAMES:
            row = model_perf_rows.get(model_name)

            if row is None:
                health[model_name] = ModelHealthEntry(
                    model_name=model_name,
                    avg_mape_30d=None,
                    prior_avg_mape=None,
                    trend="unknown",
                    mape_delta=None,
                    is_degrading=False,
                )
            else:
                avg_mape = row["avg_mape_30d"]
                delta = row["mape_delta"]
                # Degradation: strictly greater than threshold (not >=)
                is_degrading = (
                        delta is not None and delta > MODEL_DEGRADATION_MAPE_DELTA
                )
                prior = (avg_mape - delta) if (avg_mape is not None and delta is not None) else None
                health[model_name] = ModelHealthEntry(
                    model_name=model_name,
                    avg_mape_30d=avg_mape,
                    prior_avg_mape=prior,
                    trend=row["trend"],
                    mape_delta=delta,
                    is_degrading=is_degrading,
                )

        # Include Prophet fallback rows if present — reported under their own key
        prophet_row = model_perf_rows.get(Model.PROPHET)
        if prophet_row is not None:
            delta = prophet_row["mape_delta"]
            avg_mape = prophet_row["avg_mape_30d"]
            health[Model.PROPHET] = ModelHealthEntry(
                model_name=Model.PROPHET,
                avg_mape_30d=avg_mape,
                prior_avg_mape=(avg_mape - delta) if (avg_mape is not None and delta is not None) else None,
                trend=prophet_row["trend"],
                mape_delta=delta,
                is_degrading=delta is not None and delta > MODEL_DEGRADATION_MAPE_DELTA,
            )

        return health

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_run_statistics(results: list[SKUResult]) -> dict:
        """Pure computation from SKUResult list — no DB access."""
        total = len(results)

        cache_hit_count = sum(1 for r in results if r.processing_tier == ProcessingTier.CACHE)
        partial_count = sum(1 for r in results if r.processing_tier == ProcessingTier.PARTIAL)
        full_count = sum(1 for r in results if r.processing_tier == ProcessingTier.FULL)

        auto_proceed_count = sum(1 for r in results if r.status == ForecastStatus.FORECASTED)
        needs_review_count = sum(
            1 for r in results
            if r.status in {ForecastStatus.NEEDS_ACKNOWLEDGMENT, ForecastStatus.WATCHLIST_REVIEW}
        )

        pct_auto_proceed = auto_proceed_count / total if total > 0 else 0.0
        pct_needs_review = needs_review_count / total if total > 0 else 0.0

        confidences = [r.confidence_final for r in results if r.confidence_final is not None]
        avg_confidence_all = sum(confidences) / len(confidences) if confidences else 0.0

        high_fallback_count = sum(1 for r in results if r.used_fallback)

        high_fallback_alert = (
            (high_fallback_count / total) > SELF_ASSESSMENT_FALLBACK_RATE_WARN
            if total > 0 else False
        )
        low_cache_hit_alert = (
            (cache_hit_count / total) < SELF_ASSESSMENT_CACHE_HIT_RATE_WARN
            if total > 0 else False
        )
        low_success_rate_alert = (
            ((total - high_fallback_count) / total) < SELF_ASSESSMENT_SUCCESS_RATE_CRITICAL
            if total > 0 else False
        )

        return {
            "total_skus_processed": total,
            "cache_hit_count": cache_hit_count,
            "partial_count": partial_count,
            "full_count": full_count,
            "pct_auto_proceed": pct_auto_proceed,
            "pct_needs_review": pct_needs_review,
            "avg_confidence_all": avg_confidence_all,
            "high_fallback_count": high_fallback_count,
            "high_fallback_alert": high_fallback_alert,
            "low_cache_hit_alert": low_cache_hit_alert,
            "low_success_rate_alert": low_success_rate_alert,
        }

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendations(
            degrading_models: list[str],
            model_health: dict[str, ModelHealthEntry],
            run_stats: dict,
    ) -> list[str]:
        """Return actionable human-readable recommendation strings."""
        recommendations: list[str] = []

        for model_name in degrading_models:
            entry = model_health[model_name]
            delta_str = f"{entry.mape_delta:.1%}" if entry.mape_delta is not None else "N/A"
            recommendations.append(
                f"{model_name}: MAPE increased by {delta_str} over prior 30 days "
                f"(current={entry.avg_mape_30d:.3f}, prior={entry.prior_avg_mape:.3f}). "
                f"Suggested actions: "
                f"(1) Check upstream data quality for this pattern. "
                f"(2) Increase thompson_exploration_budget for this model. "
                f"(3) Verify no schema changes affected the training series."
            )

        if run_stats["high_fallback_alert"]:
            recommendations.append(
                f"HIGH FALLBACK RATE: {run_stats['high_fallback_count']}/"
                f"{run_stats['total_skus_processed']} SKUs used Naive fallback. "
                "Check for upstream data quality issues or a schema change in golden_table."
            )

        if run_stats["low_cache_hit_alert"]:
            recommendations.append(
                f"LOW CACHE HIT RATE: only {run_stats['cache_hit_count']}/"
                f"{run_stats['total_skus_processed']} SKUs served from cache. "
                "Unusual demand volatility or fingerprint logic issue — "
                "investigate data_fingerprint_cache."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Step 5
    # ------------------------------------------------------------------

    @staticmethod
    def _write_assessment(assessment: SelfAssessmentResult, conn: Any) -> None:
        """Write one row to stage9_self_assessment. ON CONFLICT DO NOTHING — safe to retry."""
        model_health_jsonb = {
            name: {
                "avg_mape": entry.avg_mape_30d,
                "trend": entry.trend,
                "delta": entry.mape_delta,
            }
            for name, entry in assessment.model_health.items()
        }

        avg_mape_this = _mean_mape(assessment.model_health, "avg_mape_30d")
        avg_mape_prev = _mean_mape(assessment.model_health, "prior_avg_mape")
        delta_pct = (
            (avg_mape_this - avg_mape_prev) / avg_mape_prev * 100.0
            if avg_mape_prev and avg_mape_prev > 0
            else None
        )

        try:
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_SQL,
                    (
                        assessment.tenant_id,
                        assessment.run_id,
                        avg_mape_this,
                        avg_mape_prev,
                        delta_pct,
                        assessment.degradation_detected,
                        json.dumps(assessment.recommendations),
                        json.dumps(model_health_jsonb),
                        assessment.total_skus_processed,
                        assessment.cache_hit_count,
                        assessment.partial_count,
                        assessment.full_count,
                        assessment.high_fallback_count,
                        assessment.pattern_feedback_failures_count,
                        assessment.execution_mode,
                        assessment.run_duration_seconds,
                    ),
                )
            conn.commit()
            log.debug(
                "self_assessment written tenant=%s run=%s degradation=%s",
                assessment.tenant_id, assessment.run_id, assessment.degradation_detected,
            )
        except Exception as exc:
            conn.rollback()
            log.error(
                "self_assessment _write_assessment failed tenant=%s run=%s: %s",
                assessment.tenant_id, assessment.run_id, exc,
            )
            raise


# ===========================================================================
# Internal helpers
# ===========================================================================

def _mean_mape(model_health: dict[str, ModelHealthEntry], attr: str) -> Optional[float]:
    """Mean of a float attribute across all ModelHealthEntry values that have data."""
    values = [
        getattr(e, attr)
        for e in model_health.values()
        if getattr(e, attr) is not None
    ]
    return sum(values) / len(values) if values else None
