"""
PRELOADING handler .

Instantiates Preloader, runs all 7 bulk reads plus TenantParams and
signal_context loads, then stores a RunContext for all subsequent
handlers to share.

Also resolves the execution mode (FULL vs MICRO_UPDATE) by comparing
the time since the last completed run against the tenant's configured
threshold (micro_update_threshold_hours, default 18).

  >= threshold hours since last COMPLETE  →  ExecutionMode.FULL
  <  threshold hours since last COMPLETE  →  ExecutionMode.MICRO_UPDATE
  No prior COMPLETE row (first run)       →  ExecutionMode.FULL

No per-SKU DB reads happen after this point in the main process.
Stage 8 tables are read via fully-qualified stage8.* names.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from infrastructure.batch_writer import BatchWriter
from infrastructure.constants import ExecutionMode, Param
from infrastructure.db_utils import DBConnection
from signals import SignalConsumer
from pipeline.preloader import Preloader
from handlers._context import RunContext, store

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Execution-mode resolution
# ---------------------------------------------------------------------------

_SQL_LAST_COMPLETE = """
    SELECT transitioned_at
    FROM agent_state_log_s9
    WHERE tenant_id = %s
      AND to_state   = 'COMPLETE'
    ORDER BY transitioned_at DESC
    LIMIT 1
"""


def _resolve_execution_mode(tenant_id: str, db: DBConnection, params: Any) -> str:
    """
    Determine whether this run is FULL or MICRO_UPDATE.

    Reads micro_update_threshold_hours from tenant_learning_params (default 18).
    Returns FULL when no prior COMPLETE row exists (first run for this tenant).
    """
    try:
        threshold_hours = float(params.get(Param.MICRO_UPDATE_THRESHOLD_HOURS) or 18.0)
    except Exception:
        threshold_hours = 18.0

    with db.cursor() as cur:
        cur.execute(_SQL_LAST_COMPLETE, (tenant_id,))
        row = cur.fetchone()

    if row is None:
        # No prior completed run — always run full on first execution.
        return ExecutionMode.FULL

    last_complete_at = row[0]
    if last_complete_at.tzinfo is None:
        last_complete_at = last_complete_at.replace(tzinfo=timezone.utc)
    else:
        last_complete_at = last_complete_at.astimezone(timezone.utc)

    hours_since_full = (datetime.now(timezone.utc) - last_complete_at).total_seconds() / 3600

    mode = ExecutionMode.FULL if hours_since_full >= threshold_hours else ExecutionMode.MICRO_UPDATE
    log.info(
        "preloading_handler execution_mode=%s hours_since_full=%.2f threshold=%.1f tenant=%s",
        mode, hours_since_full, threshold_hours, tenant_id,
    )
    return mode


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def preloading_handler(*, tenant_id: str, run_id: str, db: DBConnection) -> None:
    log.info("preloading_handler starting tenant=%s run=%s", tenant_id, run_id)
    run_start = time.time()

    loader   = Preloader()
    preloaded = loader.load(tenant_id, db)

    execution_mode = _resolve_execution_mode(tenant_id, db, loader.params)

    # All SKU IDs in scope — every sku_id that has a pattern_history row.
    sku_ids = list(preloaded.pattern_ctx.keys())

    # End the read-only transaction so the connection is IDLE before
    # BatchWriter / SignalConsumer inspect its state.
    db.commit()

    ctx = RunContext(
        tenant_id=tenant_id,
        run_id=run_id,
        run_start_time=run_start,
        execution_mode=execution_mode,
        preloaded=preloaded,
        params=loader.params,
        batch_writer=BatchWriter(db),
        signal_consumer=SignalConsumer(db),
        sku_ids=sku_ids,
    )
    store(ctx)

    log.info(
        "preloading_handler complete tenant=%s run=%s mode=%s skus=%d elapsed=%.2fs",
        tenant_id, run_id, execution_mode, len(sku_ids), time.time() - run_start,
    )
