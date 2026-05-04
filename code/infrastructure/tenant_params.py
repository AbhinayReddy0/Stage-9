"""
Per-tenant snapshot of tenant_learning_params.

Every Stage 9 sub-stage reads parameters through this class. No sub-stage
queries tenant_learning_params directly.

Lifecycle:
  - PRELOADING: TenantParams.load(tenant_id, conn) runs ONE query and caches
    all rows in a dict keyed by param_name.
  - Sub-stages: params.get(name) returns current_value as float.
  - LEARNING: params.update(name, evidence, conn) applies exponential
    smoothing, writes to DB, and refreshes the in-memory cache so subsequent
    get() calls within the same run see the new value.

Missing params raise UnknownParamError — they are never silently defaulted.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from infrastructure.constants import Param
from infrastructure.errors import UnknownParamError

__all__ = ["TenantParams", "UnknownParamError"]

logger = logging.getLogger(__name__)


_SELECT_ALL = (
    "SELECT param_name, current_value "
    "FROM tenant_learning_params "
    "WHERE tenant_id = %s"
)

_UPDATE_ONE = (
    # Atomic read-modify-write in SQL — avoids the Python read-modify-write
    # race where two concurrent callers could each read the same prior value
    # and both write a stale result. The Redis run-lock prevents concurrent
    # Stage 9 runs for the same tenant, but this guard is correct regardless.
    "UPDATE tenant_learning_params "
    "SET current_value = current_value + %s * (%s - current_value), "
    "    last_updated_at = NOW() "
    "WHERE tenant_id = %s AND param_name = %s"
)


class TenantParams:
    def __init__(self, tenant_id: str, values: dict[str, Decimal]) -> None:
        self._tenant_id = tenant_id
        self._values: dict[str, Decimal] = dict(values)

    @classmethod
    def load(cls, tenant_id: str, conn) -> "TenantParams":
        """
        Load all current_values for tenant_id in a single SELECT.

        Called once during PRELOADING. All sub-stages then call get() on the
        returned instance — no further DB reads for params during the run.
        Raises nothing on an empty result; sub-stages will raise
        UnknownParamError on their first get() if the tenant was never seeded.
        """
        with conn.cursor() as cur:
            cur.execute(_SELECT_ALL, (tenant_id,))
            rows = cur.fetchall()
        instance = cls(tenant_id, {name: value for name, value in rows})
        logger.info(
            "TenantParams.load tenant_id=%s params_loaded=%d",
            tenant_id, len(instance),
        )
        return instance

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def __contains__(self, param_name: str) -> bool:
        return param_name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, param_name: str) -> float:
        """
        Return current_value for param_name as float.

        Raises UnknownParamError — never silently defaults — so that a missing
        seed row surfaces immediately rather than causing a silent wrong result.
        """
        try:
            return float(self._values[param_name])
        except KeyError:
            raise UnknownParamError(
                f"Param {param_name!r} not found for tenant {self._tenant_id}. "
                f"Was it seeded via seed_tenant_params?"
            ) from None

    def to_dict(self) -> dict[str, str]:
        """
        Serialise the snapshot to a plain {param_name: str(value)} dict.

        Used when passing TenantParams to ProcessPoolExecutor workers, which
        communicate via pickle. Converting Decimal → str avoids any precision
        loss during serialisation and is safely reconstructed by from_dict().
        Workers should call from_dict() on the other side and treat the result
        as read-only (updates require a DB connection, which workers own
        independently).
        """
        return {name: str(val) for name, val in self._values.items()}

    @classmethod
    def from_dict(cls, tenant_id: str, data: dict[str, str]) -> "TenantParams":
        """
        Reconstruct a TenantParams snapshot from the output of to_dict().

        Intended for use inside ProcessPoolExecutor workers. The reconstructed
        instance supports get() but not update() (workers do not persist
        learning updates — that is done by the main process after the pool
        completes).
        """
        return cls(tenant_id, {name: Decimal(val) for name, val in data.items()})

    def update(self, param_name: str, evidence_value: float, conn) -> float:
        """
        Apply exponential smoothing atomically in SQL and refresh in-memory cache.

        SQL form (atomic — avoids Python read-modify-write race):
            new_current = current_value + rate * (evidence - current_value)

        The Redis run-lock already prevents two Stage 9 runs for the same
        tenant from executing concurrently, but the atomic SQL form is correct
        regardless of locking strategy.

        Returns the new current_value as float. Caller owns commit/rollback —
        this method issues an UPDATE but does NOT call conn.commit().

        Raises:
            UnknownParamError: param_name not in snapshot, or
                calibration_update_rate missing, or DB row not found
                (snapshot out of sync with DB).
        """
        # --- guard: param must exist in snapshot ------------------------------
        if param_name not in self._values:
            raise UnknownParamError(
                f"Cannot update {param_name!r} — not in snapshot for tenant "
                f"{self._tenant_id}."
            )

        # --- guard: learning rate must be present -----------------------------
        # Use dict key access (not .get()) so a missing calibration_update_rate
        # raises UnknownParamError with a clear message rather than producing
        # a TypeError on the arithmetic below.
        if Param.CALIBRATION_UPDATE_RATE not in self._values:
            raise UnknownParamError(
                "calibration_update_rate missing from snapshot — cannot apply "
                "exponential smoothing. Seed this tenant first."
            )
        rate = self._values[Param.CALIBRATION_UPDATE_RATE]

        evidence = Decimal(str(evidence_value))

        # The SQL does the arithmetic atomically; we mirror it here so the
        # in-memory snapshot stays consistent with what was written to DB.
        new_current = self._values[param_name] + rate * (evidence - self._values[param_name])

        with conn.cursor() as cur:
            # Args order matches _UPDATE_ONE placeholders: rate, evidence, tenant, param
            cur.execute(_UPDATE_ONE, (rate, evidence, self._tenant_id, param_name))
            if cur.rowcount != 1:
                logger.error(
                    "TenantParams.update row not found tenant_id=%s param=%s — snapshot out of sync",
                    self._tenant_id, param_name,
                )
                raise UnknownParamError(
                    f"Row for ({self._tenant_id}, {param_name!r}) not found in DB "
                    f"during update — snapshot is out of sync."
                )

        self._values[param_name] = new_current
        logger.info(
            "TenantParams.update tenant_id=%s param=%s new_value=%s",
            self._tenant_id, param_name, float(new_current),
        )
        return float(new_current)
