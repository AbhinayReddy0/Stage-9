"""
SignalConsumer — read-side of the Stage 9 cross-agent signal bus.

peek_signals  — SELECT without modifying processed flag (thread-safe).
consume_signals — atomic FOR UPDATE SKIP LOCKED + UPDATE processed=TRUE.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from infrastructure.constants import REORDER_SIGNAL_LOOKBACK
from infrastructure.db_utils import DBConnection
from signals._base import (
    Signal, DEFAULT_PEEK_LIMIT, DEFAULT_CONSUME_LIMIT,
    _decode_jsonb, _maybe_warn_shared_conn,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# Thread-safe, read-only peek. Filters processed=FALSE only; sku_id cast
# to uuid; returns payloads only (not full Signal dataclass).
_PEEK_COMPAT_SQL = """
    SELECT payload
      FROM stage9.cross_agent_signals
     WHERE tenant_id = %s
       AND signal_type = %s
       AND (%s IS NULL OR sku_id = %s::uuid)
       AND processed = FALSE
       AND (expires_at IS NULL OR expires_at > NOW())
     ORDER BY created_at DESC
     LIMIT %s
"""

# Canonical column list for both peek (SELECT) and consume (RETURNING).
_PEEK_COLS: tuple = (
    "payload", "created_at", "sku_id", "signal_id", "from_agent", "to_agent",
)
_PEEK_COLS_CSV = ", ".join(_PEEK_COLS)
_CONSUME_RETURNING = "RETURNING " + ", ".join(f"s.{c}" for c in _PEEK_COLS)

_PEEK_SQL_BASE = (
    f"SELECT {_PEEK_COLS_CSV} "
    "FROM stage9.cross_agent_signals "
    "WHERE tenant_id = %s "
    "  AND signal_type = %s "
    "  AND expires_at > NOW() "
)

# Atomic FOR UPDATE SKIP LOCKED so concurrent consumers see disjoint rows.
_CONSUME_SQL_NO_LIMIT = (
    "WITH eligible AS ( "
    "  SELECT signal_id FROM stage9.cross_agent_signals "
    "  WHERE tenant_id = %s AND signal_type = %s "
    "    AND to_agent IS NOT DISTINCT FROM %s "
    "    AND processed = FALSE AND expires_at > NOW() "
    "  ORDER BY created_at DESC "
    "  FOR UPDATE SKIP LOCKED "
    ") "
    "UPDATE stage9.cross_agent_signals AS s "
    "SET processed = TRUE "
    "FROM eligible "
    "WHERE s.signal_id = eligible.signal_id "
    + _CONSUME_RETURNING
)
_CONSUME_SQL_LIMIT = (
    "WITH eligible AS ( "
    "  SELECT signal_id FROM stage9.cross_agent_signals "
    "  WHERE tenant_id = %s AND signal_type = %s "
    "    AND to_agent IS NOT DISTINCT FROM %s "
    "    AND processed = FALSE AND expires_at > NOW() "
    "  ORDER BY created_at DESC LIMIT %s "
    "  FOR UPDATE SKIP LOCKED "
    ") "
    "UPDATE stage9.cross_agent_signals AS s "
    "SET processed = TRUE "
    "FROM eligible "
    "WHERE s.signal_id = eligible.signal_id "
    + _CONSUME_RETURNING
)


def _row_to_signal(row: tuple) -> Signal:
    payload, created_at, sku_id, signal_id, from_agent, to_agent = row
    return Signal(
        payload=_decode_jsonb(payload),
        created_at=created_at,
        sku_id=sku_id,
        signal_id=str(signal_id),
        from_agent=from_agent,
        to_agent=to_agent,
    )


# ---------------------------------------------------------------------------
# SignalConsumer
# ---------------------------------------------------------------------------

class SignalConsumer:
    """
    Read-side counterpart to SignalEmitter.

    `peek_signals` returns matching rows without modifying processed.
    `consume_signals` atomically marks rows processed=TRUE under
    FOR UPDATE SKIP LOCKED so concurrent consumers can't double-process.

    Both methods default to LIMIT 1000 — pass `limit=None` to fetch
    everything (use with care; a tenant's 90-day backlog can be huge).
    """

    def __init__(self, conn: DBConnection) -> None:
        _maybe_warn_shared_conn(conn, label="SignalConsumer")
        self._conn = conn
        # psycopg2 connections are not thread-safe. Sub-stages running
        # inside the ThreadPoolExecutor share this consumer; the lock
        # serialises concurrent peek() calls on the shared connection.
        self._lock = threading.Lock()

    def peek(
        self,
        tenant_id: str,
        signal_type: str,
        sku_id: Optional[str] = None,
        limit: int = REORDER_SIGNAL_LOOKBACK,
    ) -> list[dict]:
        """
        Thread-safe, read-only peek. Returns up to `limit` raw payload
        dicts (newest-first) without setting processed=TRUE. Only returns
        unprocessed rows. Returns empty list on any DB error.
        """
        try:
            with self._lock:
                with self._conn.cursor() as cur:
                    cur.execute(
                        _PEEK_COMPAT_SQL,
                        (tenant_id, signal_type, sku_id, sku_id, limit),
                    )
                    rows = cur.fetchall()
            return [row[0] for row in rows if row[0] is not None]
        except Exception as exc:
            logger.error(
                "SignalConsumer.peek failed tenant=%s signal_type=%s sku=%s: %s",
                tenant_id, signal_type, sku_id, exc,
            )
            return []

    def peek_signals(
        self,
        tenant_id: str,
        *,
        signal_type: str,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        sku_id: Optional[str] = None,
        limit: Optional[int] = DEFAULT_PEEK_LIMIT,
    ) -> list[Signal]:
        """
        SELECT-only read. Excludes expired rows. Returns a list of Signal
        objects (payload + metadata) sorted DESC by created_at.
        """
        sql = _PEEK_SQL_BASE
        args: list[Any] = [tenant_id, signal_type]
        if from_agent is not None:
            sql += "  AND from_agent = %s "
            args.append(from_agent)
        if to_agent is not None:
            sql += "  AND to_agent = %s "
            args.append(to_agent)
        if sku_id is not None:
            sql += "  AND sku_id = %s "
            args.append(sku_id)
        sql += "ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            args.append(int(limit))

        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            rows = cur.fetchall()
        return [_row_to_signal(r) for r in rows]

    def consume_signals(
        self,
        tenant_id: str,
        *,
        signal_type: str,
        to_agent: Optional[str],
        limit: Optional[int] = DEFAULT_CONSUME_LIMIT,
    ) -> list[Signal]:
        """
        Atomic claim-and-mark-processed. FOR UPDATE SKIP LOCKED on the CTE
        inner SELECT so concurrent consumers see disjoint rows.

        `to_agent=None` matches broadcast (NULL) signals like model_health
        (uses IS NOT DISTINCT FROM).
        """
        if limit is None:
            sql = _CONSUME_SQL_NO_LIMIT
            args: tuple = (tenant_id, signal_type, to_agent)
        else:
            sql = _CONSUME_SQL_LIMIT
            args = (tenant_id, signal_type, to_agent, int(limit))

        with self._conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        try:
            self._conn.commit()
        except Exception:
            logger.debug(
                "consume_signals commit failed type=%s to_agent=%s",
                signal_type, to_agent, exc_info=True,
            )
        rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
        return [_row_to_signal(r) for r in rows_sorted]
