"""
Shared DB plumbing helpers used across Stage 9 modules.

These exist so multiple sub-stages (signal_bus, aggregator jobs, etc.)
don't each ship near-identical copies of the same defensive checks.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = ["DBConnection", "warn_if_shared_conn"]


@runtime_checkable
class DBConnection(Protocol):
    """Structural type for a psycopg2 (or compatible) database connection.

    Any object that exposes cursor(), commit(), and rollback() satisfies this
    Protocol — including real psycopg2 connections and test fakes — without
    requiring an explicit inheritance relationship.
    """

    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


def warn_if_shared_conn(conn: DBConnection, *, label: str) -> None:
    """
    psycopg2 connections expose `info.transaction_status`:
        0 = TRANSACTION_STATUS_IDLE
        1 = TRANSACTION_STATUS_ACTIVE
        2 = TRANSACTION_STATUS_INTRANS  (open transaction)
        3 = TRANSACTION_STATUS_INERROR

    Anything other than IDLE means the caller has an in-flight
    transaction on this connection — committing on it via the calling
    module would commit (or rollback) that other work too. Issue a
    UserWarning so production deploys see the gap.

    Best-effort: silently no-op when the conn doesn't expose .info
    (FakeConn in tests; non-psycopg2 drivers).
    """
    info = getattr(conn, "info", None)
    if info is None:
        return
    status = getattr(info, "transaction_status", None)
    if status is None or status == 0:
        return
    warnings.warn(
        f"{label} received a conn with transaction_status={status} "
        "(expected 0=IDLE). The caller commits on the supplied "
        "connection — sharing it with non-target writes risks "
        "committing or rolling back unrelated work. Pass a dedicated "
        "psycopg2 connection.",
        stacklevel=3,
    )
