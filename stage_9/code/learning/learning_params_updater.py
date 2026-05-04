"""
learning_params_updater.py
==============================================================
Nightly batch job (4:30 AM) that updates tenant_learning_params toward
evidence accumulated in forecast_outcomes, adaptive_quantile_state, and
cross_agent_signals. Runs after ModelPerformanceAggregator (4:00 AM).

Pipeline position:
    3:00 AM  OutcomeCollector          — writes forecast_outcomes
    4:00 AM  ModelPerformanceAggregator — aggregates rolling 30d MAPE
    4:30 AM  LearningParamsUpdater     ← THIS JOB
    5:00 AM  SimilarityRegistryUpdater — reads updated params

Transaction rule: all updates for one tenant commit together. If any
tp.update() raises, the entire tenant's updates roll back. Other tenants
are unaffected.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from infrastructure.constants import (
    PATTERN_MODEL_MAP,
    REORDER_BIAS_FACTOR_OVERSTOCK,
    REORDER_BIAS_FACTOR_STOCKOUT,
    REORDER_OVERSTOCK_THRESHOLD,
    REORDER_SIGNAL_LOOKBACK,
    REORDER_STOCKOUT_MIN_EVENTS,
    Agent,
    Model,
    Param,
    Pattern,
    SignalType,
)
from infrastructure.db import pg_conn
from infrastructure.tenant_params import TenantParams

log = logging.getLogger(__name__)

__all__ = ["LearningParamsUpdater", "compute_quantile_evidence"]

# Invert PATTERN_MODEL_MAP: {model: pattern}.
_MODEL_TO_PATTERN: dict[str, str] = {v: k for k, v in PATTERN_MODEL_MAP.items()}

# ---------------------------------------------------------------------------
# SQL strings
# ---------------------------------------------------------------------------

_MAPE_EVIDENCE_SQL = """
    SELECT
        assigned_model,
        AVG(error_mape)  AS avg_mape,
        COUNT(*)         AS outcome_count
    FROM stage9.forecast_outcomes
    WHERE tenant_id    = %s
      AND horizon_days = 30
      AND outcome_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY assigned_model
"""

_QUANTILE_EVIDENCE_SQL = """
    SELECT
        pattern_label,
        actual_coverage,
        target_quantile,
        sample_size
    FROM stage9.adaptive_quantile_state
    WHERE tenant_id    = %s
      AND horizon_days = 30
"""

_REORDER_SIGNAL_SQL = """
    SELECT payload
    FROM stage9.cross_agent_signals
    WHERE tenant_id  = %s
      AND signal_type = %s
      AND from_agent  = %s
      AND to_agent    = %s
      AND (expires_at IS NULL OR expires_at > NOW())
      AND processed   = FALSE
    ORDER BY created_at DESC
    LIMIT %s
"""

_FETCH_ALL_TENANTS_SQL = """
    SELECT DISTINCT tenant_id
    FROM stage9.tenant_learning_params
    ORDER BY tenant_id
"""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def compute_quantile_evidence(
        actual_coverage: float,
        target_quantile: float,
        current_quantile: float,
        step: float,
) -> float:
    """
    Return adjusted quantile evidence via binary-search step.

    If |gap| < 1 pp — within tolerance, no adjustment.
    Otherwise shift current_quantile toward target by `step`,
    clamped to [0.50, 0.99].
    """
    gap = target_quantile - actual_coverage
    if abs(gap) < 0.01:
        return current_quantile
    direction = 1.0 if gap > 0 else -1.0
    return min(0.99, max(0.50, current_quantile + direction * step))


def _safe_evidence(value: float) -> float:
    """Clamp evidence to [0.0, 1.0]. Log and zero-out NaN / Inf."""
    if not math.isfinite(value):
        log.warning("Non-finite evidence value %r — clamping to 0.0", value)
        return 0.0
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Data fetchers (module-level so tests can call them independently)
# ---------------------------------------------------------------------------

def _fetch_mape_evidence(
        tenant_id: str, conn: Any
) -> dict[str, tuple[float, int]]:
    """
    Query forecast_outcomes for last 30 days at horizon_days=30.
    Returns {assigned_model: (avg_mape, count)}.
    Empty dict on no rows or DB error.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(_MAPE_EVIDENCE_SQL, (tenant_id,))
            rows = cur.fetchall()
        return {
            row[0]: (float(row[1]), int(row[2]))
            for row in rows
            if row[0] is not None and row[1] is not None
        }
    except Exception as exc:
        log.error("_fetch_mape_evidence failed tenant=%s: %s", tenant_id, exc)
        return {}


def _fetch_quantile_evidence(
        tenant_id: str, conn: Any
) -> dict[str, tuple[float, float, int]]:
    """
    Query adaptive_quantile_state at horizon_days=30.
    Returns {pattern_label: (actual_coverage, target_quantile, sample_size)}.
    Empty dict on no rows or DB error.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(_QUANTILE_EVIDENCE_SQL, (tenant_id,))
            rows = cur.fetchall()
        return {
            row[0]: (float(row[1]), float(row[2]), int(row[3]))
            for row in rows
            if row[0] is not None
        }
    except Exception as exc:
        log.error("_fetch_quantile_evidence failed tenant=%s: %s", tenant_id, exc)
        return {}


def _fetch_reorder_signals(tenant_id: str, conn: Any) -> list[dict]:
    """
    Query cross_agent_signals for unprocessed reorder_outcome signals from Stage 10.
    Returns list of payload dicts (psycopg2 JSONB auto-deserialized).
    Empty list on no rows or DB error.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                _REORDER_SIGNAL_SQL,
                (
                    tenant_id,
                    SignalType.REORDER_OUTCOME,
                    Agent.STAGE_10,
                    Agent.STAGE_9,
                    REORDER_SIGNAL_LOOKBACK,
                ),
            )
            rows = cur.fetchall()
        return [row[0] for row in rows if row[0] is not None]
    except Exception as exc:
        log.error("_fetch_reorder_signals failed tenant=%s: %s", tenant_id, exc)
        return []


def _fetch_all_tenant_ids(conn: Any) -> list[str]:
    """Return all distinct tenant_ids that have rows in tenant_learning_params."""
    with conn.cursor() as cur:
        cur.execute(_FETCH_ALL_TENANTS_SQL)
        rows = cur.fetchall()
    return [str(row[0]) for row in rows]


# ---------------------------------------------------------------------------
# LearningParamsUpdater
# ---------------------------------------------------------------------------

class LearningParamsUpdater:
    """
    Nightly batch job: updates tenant_learning_params toward evidence.

    Instantiated once by main() and called once per tenant. Each tenant's
    updates commit together inside run(); a failure rolls back that tenant
    only — other tenants are unaffected.
    """

    def run(self, tenant_id: str, conn: Any) -> dict:
        """
        Run all parameter updates for one tenant.

        Returns a summary dict for logging:
            {"tenant_id": ..., "status": "ok" | "skipped_unseeded"}

        All tp.update() calls accumulate DB writes without committing.
        conn.commit() is called once at the end of this method. Callers
        must NOT commit or rollback between steps.
        """
        log.info("LearningParamsUpdater starting — tenant=%s", tenant_id)

        tp = TenantParams.load(tenant_id, conn)
        if len(tp) == 0:
            log.warning(
                "No params found for tenant=%s — skipping (not seeded?)",
                tenant_id,
            )
            return {"tenant_id": tenant_id, "status": "skipped_unseeded"}

        mape_evidence = _fetch_mape_evidence(tenant_id, conn)
        quantile_evidence = _fetch_quantile_evidence(tenant_id, conn)
        reorder_signals = _fetch_reorder_signals(tenant_id, conn)

        updated, skipped, unchanged = 0, 0, 0

        cb_u, cb_s = self._update_confidence_bases(mape_evidence, tp, conn)
        updated += cb_u
        skipped += cb_s

        q_u, q_s, q_uc = self._update_quantiles(quantile_evidence, tp, conn)
        updated += q_u
        skipped += q_s
        unchanged += q_uc

        ss_u, ss_uc = self._update_safety_stock(reorder_signals, tp, conn)
        updated += ss_u
        unchanged += ss_uc

        conn.commit()
        log.info(
            "LearningParamsUpdater complete — tenant=%s updated=%d skipped=%d unchanged=%d",
            tenant_id, updated, skipped, unchanged,
        )
        return {"tenant_id": tenant_id, "status": "ok"}

    # ------------------------------------------------------------------
    # Step 2 — confidence_base_{pattern}
    # ------------------------------------------------------------------

    @staticmethod
    def _update_confidence_bases(
            mape_evidence: dict[str, tuple[float, int]],
            tp: TenantParams,
            conn: Any,
    ) -> tuple[int, int]:
        """
        Update confidence_base_{pattern} for each model with >= 10 outcomes.

        Returns (updated, skipped).
        """
        updated, skipped = 0, 0

        rate = tp.get(Param.CALIBRATION_UPDATE_RATE)  # same for all models in this run
        min_evidence = int(tp.get(Param.MIN_LEARNING_EVIDENCE_COUNT))

        for model, (avg_mape, count) in mape_evidence.items():
            if count < min_evidence:
                log.info(
                    "  confidence_base — SKIPPED: model=%s count=%d < %d",
                    model, count, min_evidence,
                )
                skipped += 1
                continue

            pattern = _MODEL_TO_PATTERN.get(model)
            if pattern is None:
                continue

            param_name = f"confidence_base_{pattern}"
            evidence = _safe_evidence(1.0 - float(avg_mape))
            prior = tp.get(param_name)
            new_val = tp.update(param_name, evidence, conn)
            log.info(
                "  %s: prior=%.3f evidence=%.3f rate=%.3f new=%.3f",
                param_name, prior, evidence, rate, new_val,
            )
            updated += 1

        return updated, skipped

    # ------------------------------------------------------------------
    # Step 4 — quantile_{pattern}
    # ------------------------------------------------------------------

    @staticmethod
    def _update_quantiles(
            quantile_evidence: dict[str, tuple[float, float, int]],
            tp: TenantParams,
            conn: Any,
    ) -> tuple[int, int, int]:
        """
        Update quantile_{pattern} using binary-search calibration.
        Returns (updated, skipped, unchanged).
        """
        updated, skipped, unchanged = 0, 0, 0
        rate = tp.get(Param.CALIBRATION_UPDATE_RATE)  # same for all patterns in this run
        min_evidence = int(tp.get(Param.MIN_LEARNING_EVIDENCE_COUNT))
        q_step = float(tp.get(Param.QUANTILE_CALIBRATION_STEP))

        for pattern, (actual_coverage, target_quantile, sample_size) in quantile_evidence.items():
            if sample_size < min_evidence:
                log.info(
                    "  quantile_%s — SKIPPED: sample_size=%d < %d",
                    pattern, sample_size, min_evidence,
                )
                skipped += 1
                continue

            param_name = f"quantile_{pattern}"
            current_quantile = tp.get(param_name)
            evidence = compute_quantile_evidence(
                actual_coverage, target_quantile, current_quantile, q_step
            )

            if evidence == current_quantile:
                log.info(
                    "  %s — already_calibrated (coverage=%.3f target=%.3f)",
                    param_name, actual_coverage, target_quantile,
                )
                unchanged += 1
                continue

            new_val = tp.update(param_name, evidence, conn)
            log.info(
                "  %s: prior=%.3f evidence=%.3f rate=%.3f new=%.3f",
                param_name, current_quantile, evidence, rate, new_val,
            )
            updated += 1

        return updated, skipped, unchanged

    # ------------------------------------------------------------------
    # Step 6 — safety_stock_factor
    # ------------------------------------------------------------------

    @staticmethod
    def _update_safety_stock(
            signals: list[dict],
            tp: TenantParams,
            conn: Any,
    ) -> tuple[int, int]:
        """
        Update safety_stock_factor from reorder_outcome signal aggregate.
        Returns (updated, unchanged).
        """
        if not signals:
            return 0, 0

        stockout_count = sum(1 for s in signals if s.get("stockout") is True)
        overstock_vals = [s.get("overstock_pct", 0.0) for s in signals]
        avg_overstock = sum(overstock_vals) / len(overstock_vals)

        current = tp.get(Param.SAFETY_STOCK_FACTOR)

        if stockout_count >= REORDER_STOCKOUT_MIN_EVENTS:
            evidence = min(0.50, current * REORDER_BIAS_FACTOR_STOCKOUT)
        elif avg_overstock > REORDER_OVERSTOCK_THRESHOLD:
            evidence = max(0.05, current * REORDER_BIAS_FACTOR_OVERSTOCK)
        else:
            log.info(
                "  safety_stock_factor — neutral_no_update "
                "(stockouts=%d avg_overstock=%.3f)",
                stockout_count, avg_overstock,
            )
            return 0, 1

        rate = tp.get(Param.CALIBRATION_UPDATE_RATE)
        new_val = tp.update(Param.SAFETY_STOCK_FACTOR, evidence, conn)
        log.info(
            "  safety_stock_factor: prior=%.3f evidence=%.3f rate=%.3f new=%.3f",
            current, evidence, rate, new_val,
        )
        return 1, 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Iterate all tenants with seeded params and run learning updates.
    Called by the nightly scheduler at 4:30 AM.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    updater = LearningParamsUpdater()
    results: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}

    with pg_conn() as conn:
        tenant_ids = _fetch_all_tenant_ids(conn)

    for tenant_id in tenant_ids:
        try:
            with pg_conn() as conn:
                summary = updater.run(tenant_id, conn)
                bucket = summary["status"].split("_")[0]
                results[bucket] = results.get(bucket, 0) + 1
        except Exception as exc:
            log.error(
                "LearningParamsUpdater FAILED tenant=%s: %s", tenant_id, exc,
            )
            results["failed"] += 1

    log.info("LearningParamsUpdater finished: %s", results)


if __name__ == "__main__":
    main()
