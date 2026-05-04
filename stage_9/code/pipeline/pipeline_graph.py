"""
Pipeline-level LangGraph — watches runs.status across all tenants and
dispatches the appropriate stage agent.

This is the "outer" graph the master spec prescribes:

    Orchestration: LangGraph (pipeline-level, not inside Stage 9)

It does NOT model Stage 9's internal IDLE → PRELOADING → ... flow. That
lives in state_machine.py and is kicked off here by calling
stage9.start_run(tenant_id, run_id, conn).

Run this as a long-lived process; it polls every N seconds and dispatches
any pending runs, then loops.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from infrastructure.constants import RunStatus
from pipeline.orchestrator import run as orchestrator_run

logger = logging.getLogger("pipeline.orchestrator")

POLL_INTERVAL_SECONDS = 30


class PipelineState(TypedDict, total=False):
    pending_runs: list[dict]   # [{tenant_id, run_id, status}, ...]
    current: Optional[dict]    # the run being dispatched right now


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _make_watch_node(conn):
    def watch(_state: PipelineState) -> dict:
        sql = """
            SELECT tenant_id, run_id, status
            FROM runs
            WHERE status IN (%s, %s, %s)
            ORDER BY updated_at ASC
            LIMIT 50;
        """
        cur = conn.cursor()
        try:
            cur.execute(sql, (
                RunStatus.DATA_READY,
                RunStatus.PATTERNS_DISCOVERED,
                RunStatus.FORECASTED,
            ))
            rows = [
                {"tenant_id": r[0], "run_id": r[1], "status": r[2]}
                for r in cur.fetchall()
            ]
        finally:
            cur.close()

        logger.info("watch: %d pending run(s)", len(rows))
        return {"pending_runs": rows}

    return watch


def _make_dispatch_node(conn):
    def dispatch(state: PipelineState) -> dict:
        for run in state.get("pending_runs", []):
            status = run["status"]
            tenant_id = run["tenant_id"]
            run_id = run["run_id"]

            try:
                if status == RunStatus.DATA_READY:
                    logger.info("dispatch: tenant=%s run=%s -> stage_8", tenant_id, run_id)
                    # stage8.start_run(tenant_id, run_id, conn)  # not in scope
                elif status == RunStatus.PATTERNS_DISCOVERED:
                    logger.info("dispatch: tenant=%s run=%s -> stage_9", tenant_id, run_id)
                    orchestrator_run(tenant_id, run_id, conn)
                elif status == RunStatus.FORECASTED:
                    logger.info("dispatch: tenant=%s run=%s -> stage_10", tenant_id, run_id)
                    # stage10.start_run(tenant_id, run_id, conn)  # not in scope
            except Exception:
                logger.exception(
                    "dispatch failed for tenant=%s run=%s status=%s",
                    tenant_id, run_id, status,
                )
                # One bad run never stops the dispatcher — same isolation
                # principle as Stage 9's per-SKU handling.

        return {"pending_runs": []}

    return dispatch


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_pipeline_graph(conn):
    g = StateGraph(PipelineState)
    g.add_node("watch",    _make_watch_node(conn))
    g.add_node("dispatch", _make_dispatch_node(conn))

    g.set_entry_point("watch")
    g.add_edge("watch", "dispatch")
    g.add_edge("dispatch", END)
    return g.compile()


def run_forever(conn, poll_interval: int = POLL_INTERVAL_SECONDS) -> None:
    """Long-lived process: poll, dispatch, sleep, repeat."""
    graph = build_pipeline_graph(conn)
    while True:
        try:
            graph.invoke({})
        except Exception:
            logger.exception("pipeline tick failed — sleeping and retrying")
        time.sleep(poll_interval)
