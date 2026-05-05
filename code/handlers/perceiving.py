"""
PERCEIVING handler — Task 7.

Loads tenant_learning_params into an in-memory TenantParams snapshot
shared by all subsequent sub-stages. Raises TenantParamNotFoundError
if no params exist for the tenant — propagates to run()'s except block
which transitions the run to FAILED.

Also peeks at pattern_confidence signals from Stage 8 (read-only;
consumed=TRUE is never set here per the SignalConsumer PEEK contract).
"""
from __future__ import annotations

import logging

from infrastructure.constants import SignalType
from infrastructure.db_utils import DBConnection
from infrastructure.errors import TenantParamNotFoundError
from infrastructure.tenant_params import TenantParams
from infrastructure.tenant_params_defaults import VALID_PARAM_NAMES
from handlers._context import fetch

log = logging.getLogger(__name__)


def _assert_params_complete(tenant_id: str, params: TenantParams) -> None:
    """
    Catch a deploy-ahead-of-migration scenario: code expects a new param that
    was added to constants.Param and tenant_params_defaults, but the DB row
    hasn't been backfilled for this tenant yet.

    Raises TenantParamNotFoundError with the names of the missing params and
    the exact remediation command so the operator knows what to do.
    """
    missing = VALID_PARAM_NAMES - set(params)
    if missing:
        raise TenantParamNotFoundError(
            f"Tenant '{tenant_id}' is missing {len(missing)} param(s) "
            f"introduced in a recent migration: {sorted(missing)}. "
            f"Fix: re-run seed.seed_tenant_params(tenant_id, ...) — it uses "
            f"ON CONFLICT DO NOTHING and will INSERT only the missing rows."
        )


def perceiving_handler(*, tenant_id: str, run_id: str, db: DBConnection) -> None:
    log.info("perceiving_handler starting tenant=%s run=%s", tenant_id, run_id)
    ctx = fetch(run_id)

    # Load all tenant params in one SELECT. TenantParams.load() returns an
    # instance with len() == number of rows found.
    params = TenantParams.load(tenant_id, db)
    if len(params) == 0:
        raise TenantParamNotFoundError(
            f"tenant_learning_params has no rows for tenant={tenant_id!r}. "
            "Run seed.seed_tenant_params() to initialise this tenant before "
            "starting the Stage 9 pipeline."
        )

    # Guard against a deploy where new params were added to constants.Param
    # and tenant_params_defaults but the DB rows haven't been backfilled yet.
    _assert_params_complete(tenant_id, params)

    ctx.params = params

    # Peek at pattern_confidence signals from Stage 8.
    # These are read-only (PEEK contract — processed flag never touched here).
    pattern_signals = ctx.signal_consumer.peek(
        tenant_id, SignalType.PATTERN_CONFIDENCE
    )
    ctx.pattern_signals = pattern_signals or []
    if pattern_signals:
        log.info(
            "perceiving_handler tenant=%s: %d pattern_confidence signal(s) available",
            tenant_id, len(pattern_signals),
        )

    log.info(
        "perceiving_handler complete tenant=%s run=%s params=%d",
        tenant_id, run_id, len(params),
    )
