"""
REPORTING handler — Task 11.

Post-run health report and final status:

  1. Run SelfAssessmentEngine to detect model degradation, compute run
     stats, and write the stage9_self_assessment row.
  2. Emit model_health broadcast signal → all agents (Stage 8, 10).
  3. Clean up the RunContext from the in-memory registry.

No FAILED edge from REPORTING (see VALID_TRANSITIONS) — any exception
surfaces as an unhandled error directly to the caller.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from infrastructure.constants import Agent, ForecastStatus, RunStatus, SignalType
from infrastructure.db_utils import DBConnection
from learning.self_assessment import SelfAssessmentEngine
from handlers._context import fetch, remove

try:
    from psycopg2.extras import Json as _Json  # type: ignore
except Exception:
    _Json = None

log = logging.getLogger(__name__)

_INSERT_MODEL_HEALTH_SIGNAL = """
    INSERT INTO stage9.cross_agent_signals
        (signal_id, tenant_id, from_agent, to_agent, signal_type,
         run_id, payload, processed, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, NOW())
"""

# UPDATE run.status — written once at end of REPORTING. The runs table
# lives in the orchestration schema and is owned upstream; we only UPDATE
# existing rows.
_SQL_UPDATE_RUN_STATUS = """
    UPDATE stage8.runs
       SET status = %s,
           updated_at = NOW()
     WHERE tenant_id = %s
       AND run_id    = %s
"""


def _set_run_status(db: DBConnection, tenant_id: str, run_id: str, status: str) -> None:
    """UPDATE stage8.runs.status — orchestration table is owned upstream."""
    try:
        with db.cursor() as cur:
            cur.execute(_SQL_UPDATE_RUN_STATUS, (status, tenant_id, run_id))
        db.commit()
    except Exception:
        # stage8.runs may not exist in dev environments; log but don't fail
        # the run — agent_state_log_s9 already records the terminal state,
        # which is the authoritative signal for monitoring.
        log.exception("reporting_handler _set_run_status failed tenant=%s run=%s",
                      tenant_id, run_id)


def _jsonb(obj: Any) -> Any:
    return _Json(obj) if _Json else json.dumps(obj)


def reporting_handler(*, tenant_id: str, run_id: str, db: DBConnection) -> None:
    log.info("reporting_handler starting tenant=%s run=%s", tenant_id, run_id)
    ctx = fetch(run_id)

    # ------------------------------------------------------------------
    # 1. Self-assessment — reads model_performance_s9, writes
    #    stage9_self_assessment, returns SelfAssessmentResult
    # ------------------------------------------------------------------
    engine = SelfAssessmentEngine()
    assessment = engine.run(
        tenant_id=tenant_id,
        run_id=run_id,
        results=ctx.sku_results,
        pattern_feedback_failures_count=ctx.pattern_feedback_failures_count,
        run_start_time=ctx.run_start_time,
        execution_mode=ctx.execution_mode,
        conn=db,
    )

    log.info(
        "reporting_handler self_assessment complete tenant=%s run=%s "
        "degradation=%s high_fallback=%s pct_auto=%.2f",
        tenant_id, run_id,
        assessment.degradation_detected,
        assessment.high_fallback_alert,
        assessment.pct_auto_proceed,
    )

    # ------------------------------------------------------------------
    # 2. Emit model_health broadcast signal → Stage 8 and Stage 10
    # ------------------------------------------------------------------
    model_health_payload = {
        model: {
            "avg_mape": entry.avg_mape_30d,
            "trend":    entry.trend,
            "delta":    entry.mape_delta,
        }
        for model, entry in assessment.model_health.items()
    }
    signal_payload = _jsonb({
        "run_id":              run_id,
        "degradation_detected": assessment.degradation_detected,
        "model_health":        model_health_payload,
        "recommendations":     assessment.recommendations,
    })

    for to_agent in (Agent.STAGE_8, Agent.STAGE_10):
        with db.cursor() as cur:
            cur.execute(
                _INSERT_MODEL_HEALTH_SIGNAL,
                (
                    str(uuid.uuid4()),
                    tenant_id,
                    Agent.STAGE_9,
                    to_agent,
                    SignalType.MODEL_HEALTH,
                    run_id,
                    signal_payload,
                ),
            )
    db.commit()
    log.info(
        "reporting_handler model_health signal emitted tenant=%s run=%s",
        tenant_id, run_id,
    )

    # ------------------------------------------------------------------
    # 3. Set run.status — 'needs_acknowledgment' if any SKU requires review,
    #    otherwise 'forecasted'. Watchlist SKUs always need ack.
    # ------------------------------------------------------------------
    needs_review_statuses = {
        ForecastStatus.NEEDS_ACKNOWLEDGMENT,
        ForecastStatus.WATCHLIST_REVIEW,
    }
    requires_ack = any(r.status in needs_review_statuses for r in ctx.sku_results)
    final_status = (
        RunStatus.NEEDS_ACKNOWLEDGMENT if requires_ack else RunStatus.FORECASTED
    )
    _set_run_status(db, tenant_id, run_id, final_status)

    # ------------------------------------------------------------------
    # 4. Clean up run context from in-memory registry
    # ------------------------------------------------------------------
    remove(run_id)

    log.info(
        "reporting_handler complete tenant=%s run=%s status=%s "
        "duration=%.1fs skus=%d",
        tenant_id, run_id, final_status,
        assessment.run_duration_seconds,
        assessment.total_skus_processed,
    )
