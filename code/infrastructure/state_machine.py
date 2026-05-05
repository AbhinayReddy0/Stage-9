# stage9/state_machine.py
"""
Stage 9 — State Machine
=======================
Controls the full execution lifecycle of the Stage 9 Forecasting Agent.

Every state transition is:
  1. Validated against VALID_TRANSITIONS before any side effect.
  2. Persisted to `agent_state_log_s9` with a DB COMMIT before the
     corresponding handler executes — so a handler crash still leaves
     a fully auditable trail.
  3. Logged at INFO level in structured key=value format for log aggregators.

Concurrency guarantee:
  One Redis NX lock per tenant ensures only one Stage 9 run executes at a
  time. A 4-hour TTL acts as a dead-man's switch: if the process crashes
  without reaching the finally-block, the key auto-expires and a fresh run
  can acquire the lock without manual intervention.

Dependency injection:
  `db` is an explicit parameter throughout. No module-level globals,
  no thread-locals, no environment-variable reads. This makes the
  module trivially testable — pass a mock DB connection.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Dict, List, Optional

from infrastructure.errors import InvalidStateTransitionError
from infrastructure.constants import RunStatus

logger = logging.getLogger(__name__)


# ===========================================================================
# Section 1 — AgentState Enum
# ===========================================================================

class AgentState(Enum):
    """
    The nine states of the Stage 9 agent lifecycle.

    Rules for future changes:
      - Do NOT add, rename, or remove members without simultaneously updating:
          * VALID_TRANSITIONS below
          * The CHECK constraints on `from_state` / `to_state` in the
            agent_state_log_s9 DDL
          * Any monitoring queries or dashboards that filter on state strings
      - String values (e.g. "IDLE") are used ONLY when persisting to the DB.
        Every dict key, comparison, function argument, and return value in
        application code must use the enum member (e.g. AgentState.IDLE).
    """
    IDLE = "IDLE"
    PRELOADING = "PRELOADING"
    PERCEIVING = "PERCEIVING"
    PLANNING = "PLANNING"
    ACTING = "ACTING"
    LEARNING = "LEARNING"
    REPORTING = "REPORTING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ===========================================================================
# Section 2 — Valid Transition Map
# ===========================================================================

# Keys and values are AgentState members — never bare strings.
#
# LEARNING and REPORTING intentionally have NO FAILED edge.
# Rationale: once ACTING succeeds, every SKU result has been computed and
# the BatchWriter is ready to flush. Any exception in LEARNING or REPORTING
# is genuinely unexpected (not a data or configuration problem) and should
# surface as an unhandled exception so monitoring alerts immediately.
# Silently transitioning to FAILED at that point would hide the root cause.
#
# KNOWN GAP — IDLE → FAILED is not in this map (matches spec exactly).
# Consequence: if the very first transition() call (IDLE → PRELOADING) itself
# fails (e.g. the DB is completely down), the except block in run() cannot
# write a FAILED log row. This edge case is handled defensively — see the
# inner try/except in run() for details.
VALID_TRANSITIONS: Dict[AgentState, List[AgentState]] = {
    AgentState.IDLE:       [AgentState.PRELOADING],
    AgentState.PRELOADING: [AgentState.PERCEIVING, AgentState.FAILED],
    AgentState.PERCEIVING: [AgentState.PLANNING,   AgentState.FAILED],
    AgentState.PLANNING:   [AgentState.ACTING,     AgentState.FAILED],
    AgentState.ACTING:     [AgentState.LEARNING,   AgentState.FAILED],
    AgentState.LEARNING:   [AgentState.REPORTING],
    AgentState.REPORTING:  [AgentState.COMPLETE],
    AgentState.COMPLETE:   [AgentState.IDLE],
    AgentState.FAILED:     [AgentState.IDLE],
}


# ===========================================================================
# Section 3 — Internal Helpers
# ===========================================================================

# ---- SQL ------------------------------------------------------------------

# Defined once at module level — avoids rebuilding the string on every
# transition() call. Uses psycopg2 named-parameter style (%(name)s), which
# is parameterized and therefore immune to SQL injection even if tenant_id
# or run_id contains special characters.
_INSERT_STATE_LOG_SQL = """
    INSERT INTO agent_state_log_s9
        (tenant_id, run_id, from_state, to_state, transitioned_at, reason)
    VALUES
        (%(tenant_id)s, %(run_id)s, %(from_state)s, %(to_state)s,
         NOW() AT TIME ZONE 'UTC', %(reason)s)
"""

# ---- Input validation -----------------------------------------------------

# SECURITY ADDITION (not in spec):
# tenant_id is formatted directly into a Redis key via str.format().
# Without validation, a malformed tenant_id (e.g. containing spaces, braces,
# or path separators) could pollute the Redis key space or cause lock
# collisions between tenants if the Atheera provisioning service ever sends
# a badly formed ID.
#
# Pattern mirrors: alphanumeric + hyphens,
# 1–64 characters. UPDATE THIS REGEX if the provisioning format changes
# (e.g. to UUIDs, which would also need underscores).
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9\-]{1,64}$")

# run_id follows the same character policy plus underscores, up to 128 chars
# (common in UUID4 + timestamp composite keys used by LangGraph).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9\-_]{1,128}$")

# Maximum characters written to the `reason` column on a FAILED transition.
# PostgreSQL TEXT has no hard byte limit, but extremely long tracebacks cause
# unnecessary storage bloat in the audit log. 2 000 characters captures any
# meaningful error message.
# SPEC NOTE: The spec does not specify a truncation length. Update this
# constant if your monitoring pipeline needs a different maximum.
_REASON_MAX_LEN = 2_000


def validate_ids(tenant_id: str, run_id: str) -> None:
    """
    Guard against malformed tenant_id and run_id before they reach Redis
    or the database.

    In normal operation these values come from the LangGraph orchestrator
    and are always well-formed. A ValueError here indicates a bug upstream
    in the calling layer, not a user-facing error.

    Raises:
        ValueError: Either id is not a string or does not match its pattern.
    """
    if not isinstance(tenant_id, str) or not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id {tenant_id!r}. "
            "Expected pattern: [A-Za-z0-9\\-]{1,64}. "
            "Update _TENANT_ID_RE in state_machine.py if the provisioning "
            "format has changed."
        )
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"Invalid run_id {run_id!r}. "
            "Expected pattern: [A-Za-z0-9\\-_]{1,128}."
        )


# ===========================================================================
# Section 4 — transition()
# ===========================================================================

def transition(
        db,
        tenant_id: str,
        run_id: str,
        current_state: AgentState,
        next_state: AgentState,
        reason: Optional[str] = None,
) -> AgentState:
    """
    Advance the state machine from current_state to next_state.

    Execution order:
      1. Validate. Raise InvalidStateTransitionError — without any DB write —
         if (current → next) is not in VALID_TRANSITIONS.
      2. INSERT one row into agent_state_log_s9.
      3. COMMIT so the log row is durable before any handler executes.
         A handler crash after this point still leaves an auditable trail.
      4. Return next_state so the caller can reassign its local variable:
             state = transition(...)

    Args:
        db:            psycopg2 connection (or psycopg3-compatible).
                       The caller owns the connection lifecycle — this
                       function never closes or reopens the connection.
        tenant_id:     Tenant identifier. Persisted to the log row.
        run_id:        Pipeline run identifier. Persisted to the log row.
        current_state: The AgentState the machine is currently in.
        next_state:    The AgentState to transition to.
        reason:        Human-readable reason string. Pass None on happy-path
                       transitions; pass str(exception) on FAILED transitions.

    Returns:
        next_state — the caller assigns: state = transition(...)

    Raises:
        InvalidStateTransitionError: Illegal (current → next) pair.
            No DB row is written in this case.
        Any DB exception (psycopg2.Error, etc.): Propagated to the caller.
            The caller must handle — typically by attempting a FAILED
            transition in its own except block.
    """
    # ---- Step 1: Validate ---------------------------------------------------
    # Check before any side effect so a programming error (wrong transition)
    # never pollutes the audit log with a bogus row.
    allowed = VALID_TRANSITIONS.get(current_state, [])
    if next_state not in allowed:
        raise InvalidStateTransitionError(
            f"Transition {current_state.value!r} -> {next_state.value!r} "
            f"is not allowed. "
            f"Valid targets from {current_state.value!r}: "
            f"{[s.value for s in allowed]}"
        )

    # ---- Step 2: Build log row params --------------------------------------
    # Use .value (the string form) only for the DB TEXT columns.
    # Every other reference in this file uses enum members.
    # Parameterized dict — psycopg2 substitutes %(name)s safely, no
    # string interpolation into SQL ever happens here.
    row_params = {
        "tenant_id": tenant_id,
        "run_id": run_id,
        "from_state": current_state.value,  # "IDLE", "PRELOADING", etc.
        "to_state": next_state.value,
        "reason": reason,  # None → SQL NULL on happy-path
    }

    # ---- Step 3: Insert log row --------------------------------------------
    # Explicit cursor: works with psycopg2 (which requires cursor()) and
    # psycopg3 (which supports conn.execute() directly but also supports
    # cursor()). Using cursor() keeps compatibility with both drivers.
    cursor = db.cursor()
    try:
        cursor.execute(_INSERT_STATE_LOG_SQL, row_params)
    finally:
        # Always close the cursor to return it to the connection pool,
        # even if execute() raised. Never leave cursors open after use.
        cursor.close()

    # ---- Step 4: Commit ----------------------------------------------------
    # Commit BEFORE returning so the log row is immediately visible to
    # monitoring dashboards and audit queries on other connections.
    # If commit fails, the exception propagates to the caller's except block,
    # which will attempt a FAILED transition.
    db.commit()

    # Structured log line: parseable by Datadog, CloudWatch, Splunk, etc.
    logger.info(
        "stage9_transition tenant_id=%s run_id=%s from=%s to=%s reason=%s",
        tenant_id,
        run_id,
        current_state.value,
        next_state.value,
        reason or "-",
    )

    # ---- Step 5: Return ----------------------------------------------------
    # Caller assigns: state = transition(...)
    return next_state


# ===========================================================================
# Section 5 — DB helpers
# ===========================================================================

def _check_trigger(conn, tenant_id: str, run_id: str) -> bool:
    """IDLE → PRELOADING precondition: runs.status = 'patterns_discovered'."""
    sql = """
        SELECT 1 FROM runs
        WHERE tenant_id = %(tenant_id)s
          AND run_id    = %(run_id)s
          AND status    = %(status)s
        LIMIT 1;
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, {
            "tenant_id": tenant_id,
            "run_id": run_id,
            "status": RunStatus.PATTERNS_DISCOVERED,
        })
        return cur.fetchone() is not None
    finally:
        cur.close()


def _already_completed(conn, tenant_id: str, run_id: str) -> bool:
    """Idempotency guard: skip if forecasts already written for this run."""
    sql = """
        SELECT 1 FROM forecasts
        WHERE tenant_id = %(tenant_id)s
          AND run_id    = %(run_id)s
        LIMIT 1;
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, {"tenant_id": tenant_id, "run_id": run_id})
        return cur.fetchone() is not None
    finally:
        cur.close()


def _set_run_status(conn, tenant_id: str, run_id: str, status: str) -> None:
    """Write runs.status — called by REPORTING (success) and FAILED (failure)."""
    sql = """
        UPDATE runs SET status = %(status)s, updated_at = NOW()
        WHERE tenant_id = %(tenant_id)s AND run_id = %(run_id)s;
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, {"tenant_id": tenant_id, "run_id": run_id, "status": status})
        conn.commit()
    finally:
        cur.close()


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "AgentState",
    "VALID_TRANSITIONS",
    "InvalidStateTransitionError",
    "transition",
    "validate_ids",
]
