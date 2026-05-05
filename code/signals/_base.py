"""
Shared constants, SQL strings, dataclasses, and low-level helpers for
the Stage 9 cross-agent signal bus. Imported by signals.emitter and
signals.consumer — never imported directly by application code (use
signal_bus or signals package instead).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from infrastructure.constants import REORDER_SIGNAL_LOOKBACK  # noqa: F401 — re-exported for consumer
from infrastructure.db_utils import warn_if_shared_conn

try:
    from psycopg2.extras import Json as _PsycopgJson  # type: ignore
except ImportError:  # pragma: no cover
    _PsycopgJson = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------

SIGNAL_TYPE_FORECAST_ACCURACY = "forecast_accuracy"
SIGNAL_TYPE_FORECAST_RISK = "forecast_risk"
SIGNAL_TYPE_CROSS_SKU_LEARNING = "cross_sku_learning"
SIGNAL_TYPE_MODEL_HEALTH = "model_health"

# ---------------------------------------------------------------------------
# TTLs (days)
# ---------------------------------------------------------------------------

TTL_FORECAST_ACCURACY_DAYS = 90
TTL_FORECAST_RISK_DAYS = 90
TTL_CROSS_SKU_LEARNING_DAYS = 60
TTL_MODEL_HEALTH_DAYS = 30

# ---------------------------------------------------------------------------
# Agent identifiers
# ---------------------------------------------------------------------------

FROM_AGENT_STAGE_9 = "stage9"
TO_AGENT_STAGE_8 = "stage8"
TO_AGENT_STAGE_9 = "stage9"
TO_AGENT_STAGE_10 = "stage10"
TO_AGENT_BROADCAST: Optional[str] = None  # tenant-level broadcast — NULL in DB

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

SIGNAL_MAX_RETRIES = 3
SIGNAL_RETRY_DELAY_S = 0.1

# ---------------------------------------------------------------------------
# Default fetch limits — guard against accidental full-backlog fetchall
# ---------------------------------------------------------------------------

DEFAULT_PEEK_LIMIT = 1000
DEFAULT_CONSUME_LIMIT = 1000

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT_SIGNAL_SQL = """
INSERT INTO stage9.cross_agent_signals (
    signal_id, tenant_id, from_agent, to_agent, signal_type,
    sku_id, run_id, payload, confidence, processed,
    created_at, expires_at
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, FALSE,
    NOW(), NOW() + make_interval(days => %s)
)
"""

# model_health is one-per-(tenant, run): delete any prior UN-CONSUMED rows
# for the same run before inserting the new one. Scoped to processed=FALSE
# so already-consumed history is preserved.
_DELETE_PRIOR_MODEL_HEALTH_SQL = (
    "DELETE FROM stage9.cross_agent_signals "
    "WHERE tenant_id = %s AND run_id = %s "
    "  AND signal_type = %s "
    "  AND processed = FALSE"
)

# ---------------------------------------------------------------------------
# Public dataclasses + exceptions
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A row read back from cross_agent_signals — payload + metadata."""
    payload: dict
    created_at: dt.datetime
    sku_id: Optional[str]
    signal_id: str
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None


class SignalEmitFailed(Exception):
    """
    Raised by SignalEmitter when raise_on_failure=True and a sacred
    write exhausts its retries. Callers can branch on .signal_type /
    .sku_id to route differently from generic DB errors.
    """

    def __init__(
        self,
        *,
        attempts: int,
        last_error: Optional[BaseException],
        signal_type: str,
        sku_id: Optional[str],
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.signal_type = signal_type
        self.sku_id = sku_id
        super().__init__(
            f"signal emit failed after {attempts} attempts type={signal_type} "
            f"sku={sku_id} last_error={last_error}"
        )

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _clamp_confidence(value: Optional[float]) -> Optional[float]:
    """
    Clamp to [0, 1] at the API boundary so callers passing an out-of-range
    value get a sensible row instead of a DB-level CHECK violation.
    """
    if value is None:
        return None
    f = float(value)
    if f < 0.0:
        logger.warning("clamping out-of-range confidence value=%s to 0.0", f)
        return 0.0
    if f > 1.0:
        logger.warning("clamping out-of-range confidence value=%s to 1.0", f)
        return 1.0
    return f


def _wrap_jsonb(payload: dict) -> Any:
    """Wrap a dict for the JSONB column; falls through in test environments."""
    if _PsycopgJson is None:
        return payload
    return _PsycopgJson(payload)


def _decode_jsonb(value: Any) -> dict:
    """
    Decode a JSONB value coming back from a SELECT.
    psycopg2 normally returns dict directly; tests may pass raw strings.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return json.loads(bytes(value).decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(
        f"unexpected JSONB value type: {type(value).__name__}; "
        "expected dict, str, bytes, bytearray, memoryview, or None"
    )


# Promoted to db_utils.warn_if_shared_conn; aliased here for call sites
# in emitter and consumer.
_maybe_warn_shared_conn = warn_if_shared_conn
