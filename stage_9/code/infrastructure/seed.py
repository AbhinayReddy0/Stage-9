"""
Tenant onboarding seed helpers.

seed_tenant_params inserts the 54 rows of tenant_learning_params for a new
tenant. Idempotent via ON CONFLICT DO
NOTHING — safe to call on an already-seeded tenant.

tenant_maturity is validated but does NOT change which rows are inserted:
per STAGE_9_DATABASE_CONTRACTS.docx, exploit_threshold_new / _developing /
_established are three distinct rows with fixed starting values. Runtime
code reads the appropriate row based on the tenant's current maturity.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Mapping

from infrastructure.errors import UnknownParamError
from infrastructure.tenant_params_defaults import (
    TENANT_LEARNING_PARAMS_DEFAULTS,
    VALID_PARAM_NAMES,
    VALID_TENANT_MATURITY,
)

logger = logging.getLogger(__name__)

_INSERT_SQL = """
    INSERT INTO tenant_learning_params (tenant_id, param_name, starting_value, current_value)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (tenant_id, param_name) DO NOTHING
"""


def seed_tenant_params(
    tenant_id: str,
    tenant_maturity: str,
    overrides_dict: Mapping[str, Decimal | float | int] | None = None,
    conn=None,
) -> int:
    """
    Seed tenant_learning_params for tenant_id, or backfill rows that are
    missing after a migration adds new params to constants.Param and
    tenant_params_defaults.

    Idempotent: uses ON CONFLICT DO NOTHING, so re-running against an
    already-seeded tenant is safe and inserts only the missing rows.

    Migration path: when a new Param is added, re-run this function for all
    existing tenants. The perceiving_handler will raise TenantParamNotFoundError
    on the next Stage 9 run until the backfill completes — this is intentional
    (Principle 1: zero silent defaults).

    Args:
        tenant_id: UUID of the tenant. Must be a valid UUID string — invalid
            values raise ValueError immediately rather than producing a
            confusing psycopg2 error at DB execution time.
        tenant_maturity: One of 'new', 'developing', 'established'. Validated
            but not used to alter inserted values — runtime code reads the
            matching exploit_threshold_{maturity} row at execution time.
        overrides_dict: Optional per-customer overrides of starting values,
            e.g. {'service_level_target': 0.95}. Keys must be known param
            names; unknown keys raise UnknownParamError.
        conn: Open database connection (DB-API 2.0 compliant). Required.
            Caller owns commit/rollback.

    Returns:
        Number of rows actually inserted. 0 on a re-run of an already-seeded
        tenant (all rows existed). Returns the count of newly inserted rows
        when backfilling missing params after a migration.

    Raises:
        ValueError: invalid tenant_id UUID, invalid tenant_maturity, or
            conn is None.
        UnknownParamError: unknown key in overrides_dict.
    """
    # --- guard: connection must be provided -----------------------------------
    if conn is None:
        raise ValueError("conn is required")

    # --- guard: validate tenant_id is a well-formed UUID ----------------------
    # Catches typos and wrong-type arguments before they reach the DB driver,
    # which would otherwise surface as an opaque psycopg2 DataError.
    try:
        uuid.UUID(tenant_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"tenant_id must be a valid UUID string, got {tenant_id!r}"
        ) from exc

    # --- guard: validate tenant_maturity --------------------------------------
    if tenant_maturity not in VALID_TENANT_MATURITY:
        raise ValueError(
            f"Invalid tenant_maturity: {tenant_maturity!r}. "
            f"Must be one of {sorted(VALID_TENANT_MATURITY)}."
        )

    # --- guard: validate override keys before touching the DB -----------------
    overrides = dict(overrides_dict) if overrides_dict else {}
    if overrides:
        unknown = set(overrides.keys()) - VALID_PARAM_NAMES
        if unknown:
            logger.error(
                "seed_tenant_params rejected unknown override keys tenant_id=%s keys=%s",
                tenant_id, sorted(unknown),
            )
            raise UnknownParamError(
                f"Unknown param(s) in overrides_dict: {sorted(unknown)}. "
                f"Must be one of the {len(VALID_PARAM_NAMES)} defined params."
            )

    # --- build rows -----------------------------------------------------------
    # Both starting_value and current_value are set to the same value at seed
    # time. current_value diverges from starting_value as the learning jobs run.
    rows = []
    for param_name, default_value in TENANT_LEARNING_PARAMS_DEFAULTS:
        value = overrides.get(param_name, default_value)
        # Normalise to Decimal for consistent DB binding regardless of whether
        # the override arrived as float / int / Decimal.
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        rows.append((tenant_id, param_name, value, value))

    # --- insert ---------------------------------------------------------------
    # executemany with ON CONFLICT DO NOTHING is idempotent: re-running on an
    # already-seeded tenant is safe and inserts 0 rows.
    #
    # psycopg2 rowcount after executemany is unreliable — it reflects only the
    # last statement, not the total. We count inserted rows explicitly by
    # executing each row individually and accumulating rowcount, which is
    # accurate per DB-API 2.0 spec for single-statement execute().
    inserted = 0
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(_INSERT_SQL, row)
            # rowcount is 1 if the row was inserted, 0 if skipped by ON CONFLICT.
            if cur.rowcount == 1:
                inserted += 1

    logger.info(
        "seed_tenant_params complete tenant_id=%s maturity=%s inserted=%d skipped=%d overrides=%d",
        tenant_id, tenant_maturity, inserted, len(rows) - inserted, len(overrides),
    )
    return inserted
