"""
Stage 9 nightly batch job — ModelPerformanceAggregator .

Schedule: 4 AM UTC, after OutcomeCollector (3 AM) so we read fresh
forecast_outcomes rows. Per tenant:

  1. Aggregate forecast_outcomes for the last 30 days, grouped by
     (assigned_model, horizon_days). Compute avg_mape, median_mape,
     p90_mape, avg_bias, sample_count.
  2. Aggregate the prior 30 days the same way.
  3. Compute improvement_vs_prior = current_avg_mape - prior_avg_mape.
       improvement < 0  → MAPE shrinking → 'improving'
       improvement > 0  → MAPE growing   → 'degrading'
       |improvement| within stable_band → 'stable'
       new model (no prior) → 'stable' (no signal yet)
  4. DELETE-then-INSERT into model_performance_s9 (idempotent on
     re-run; the table holds one row per (tenant, model, horizon)
     with the latest rolling window).

PostgreSQL-specific: PERCENTILE_CONT(...) WITHIN GROUP, execute_values,
SET LOCAL statement_timeout. Not portable.

Connection contract:
  * Pass a DEDICATED psycopg2 connection — the aggregator commits on
    it. Sharing the conn with other writes risks committing /
    rolling back unrelated work. The function emits a UserWarning
    via stage9.db_utils.warn_if_shared_conn when the caller passes
    a conn that's already in a transaction.

Concurrency:
  * Pass acquire_lock_fn / release_lock_fn (e.g. backed by Redis) to
    serialize concurrent runs for the same tenant. Optional —
    callers that already enforce single-runner discipline (cron with
    no overlap) can leave them None.

Failure model:
  * Per-tenant try/except: aggregator failures don't crash the
    nightly batch. Failures populate stats.failure_reason and call
    log_failure_fn(tenant_id, "<aggregator>", "<batch>", reason).
  * Idempotent on re-run against the same as_of date.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from infrastructure.constants import MODEL_PERFORMANCE_STABLE_BAND
from infrastructure.db import pg_conn
from infrastructure.db_utils import warn_if_shared_conn
from infrastructure.tenant_params import TenantParams

logger = logging.getLogger(__name__)

# psycopg2.extras.execute_values — used when the cursor is real psycopg2.
# FakeConn in tests doesn't expose `cur.connection`, so we fall through
# to executemany. Same pattern as BatchWriter.
try:
    from psycopg2.extras import execute_values as _execute_values  # type: ignore
except ImportError:  # pragma: no cover - psycopg2 missing in unit-test env
    _execute_values = None

__all__ = [
    "AggregatorStats",
    "ModelPerformanceRow",
    "PERIOD_DAYS",
    "STATEMENT_TIMEOUT_MS",
    "TREND_IMPROVING",
    "TREND_DEGRADING",
    "TREND_STABLE",
    "run_model_performance_aggregator",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIOD_DAYS = 30   # rolling window (spec)

# Tenant param name for the stable band.
_STABLE_BAND_PARAM = "model_performance_stable_band"

TREND_IMPROVING = "improving"
TREND_DEGRADING = "degrading"
TREND_STABLE = "stable"

# Per-statement timeout — guards against pathological query plans on a
# huge forecast_outcomes table holding up the entire 4 AM batch.
STATEMENT_TIMEOUT_MS = 5 * 60 * 1000   # 5 minutes

# How many INSERT rows to bundle into one execute_values batch.
INSERT_PAGE_SIZE = 200


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelPerformanceRow:
    """One aggregated row written to model_performance_s9."""
    tenant_id: str
    model_name: str
    horizon_days: int
    period_start: dt.date
    period_end: dt.date
    avg_mape: Optional[float]
    median_mape: Optional[float]
    p90_mape: Optional[float]
    avg_bias: Optional[float]
    sample_count: int
    prior_avg_mape: Optional[float]
    improvement_vs_prior: Optional[float]
    mape_trend: str


@dataclass
class AggregatorStats:
    """
    What run_model_performance_aggregator returns.

    `failure_reason` is the human-readable string for logs and audit
    rows. `failure_exception` carries the underlying BaseException so
    callers wanting to branch by type (e.g. distinguish a connection
    error from a constraint violation) can `isinstance` check it.
    Both are populated together on failure; both stay None on success.
    """
    rows_written: int = 0
    period_start: Optional[dt.date] = None
    period_end: Optional[dt.date] = None
    new_models: list[str] = field(default_factory=list)         # in current, not prior
    discontinued_models: list[str] = field(default_factory=list)  # in prior, not current
    failure_reason: Optional[str] = None
    failure_exception: Optional[BaseException] = None
    lock_acquired: bool = True            # False if acquire_lock_fn returned None


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# Aggregate by (model, horizon) within [start, end). Excludes NULL
# error_mape rows — never count missing measurements. Casts the AVG /
# PERCENTILE outputs to float so Python sees plain floats, not Decimal.
# ORDER BY makes the INSERT order deterministic.
_AGGREGATE_SQL = """
SELECT
    assigned_model,
    horizon_days,
    AVG(error_mape)::float AS avg_mape,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY error_mape)::float AS median_mape,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY error_mape)::float AS p90_mape,
    AVG(bias)::float AS avg_bias,
    COUNT(*)::int AS sample_count
FROM stage9.forecast_outcomes
WHERE tenant_id = %s
  AND outcome_date >= %s
  AND outcome_date <  %s
  AND error_mape IS NOT NULL
GROUP BY assigned_model, horizon_days
ORDER BY assigned_model, horizon_days
"""

# Idempotent write: clear the tenant's prior aggregator output, then
# bulk-insert the fresh rows. Doesn't depend on a unique constraint
# shape — works against the documented schema.
_DELETE_TENANT_ROWS_SQL = (
    "DELETE FROM stage9.model_performance_s9 WHERE tenant_id = %s"
)

# Canonical column list for the model_performance_s9 INSERT — must match
# the schema in db.py:SQL_CREATE_MODEL_PERFORMANCE_S9 EXACTLY. The
# aggregator's ModelPerformanceRow dataclass has additional rich-metric
# fields (median_mape, p90_mape, avg_bias, period_start, period_end,
# prior_avg_mape) that aren't persisted — they exist for in-memory
# analysis and logging only. _COL_TO_FIELD bridges the schema column
# names to the dataclass attribute names where they differ.
#
# `created_at` is set by the schema via DEFAULT NOW() and is NOT in this
# tuple (we don't write it explicitly).
_INSERT_COLS: tuple[str, ...] = (
    "tenant_id", "assigned_model", "horizon_days",
    "avg_mape_30d", "trend", "mape_delta", "sample_count",
)

# Schema column → dataclass attribute. Identity for columns whose names
# already match. Aggregator semantics:
#   improvement_vs_prior = current_avg_mape - prior_avg_mape
#   (positive = MAPE went up = degrading) — matches mape_delta convention
#   in self_assessment.py (current - prior; positive = worse).
_COL_TO_FIELD: dict[str, str] = {
    "assigned_model": "model_name",
    "avg_mape_30d":   "avg_mape",
    "trend":          "mape_trend",
    "mape_delta":     "improvement_vs_prior",
}

_INSERT_COL_LIST = ", ".join(_INSERT_COLS)
_INSERT_PLACEHOLDERS = ", ".join(["%s"] * len(_INSERT_COLS))

# Two flavors of the INSERT, both derived from _INSERT_COLS:
#   1. _INSERT_BULK_SQL    — for psycopg2.extras.execute_values. The
#                            outer %s placeholder is the VALUES list;
#                            _INSERT_BULK_TEMPLATE is the per-row shape.
#   2. _INSERT_ROW_SQL     — single-row INSERT for the executemany
#                            fallback (used when the cursor is a fake
#                            without `.connection`).
_INSERT_BULK_SQL = (
    f"INSERT INTO stage9.model_performance_s9 ({_INSERT_COL_LIST}) VALUES %s"
)
_INSERT_BULK_TEMPLATE = f"({_INSERT_PLACEHOLDERS})"
_INSERT_ROW_SQL = (
    f"INSERT INTO stage9.model_performance_s9 ({_INSERT_COL_LIST}) "
    f"VALUES ({_INSERT_PLACEHOLDERS})"
)

_SET_STATEMENT_TIMEOUT_SQL = "SET LOCAL statement_timeout = %s"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

LockAcquireFn = Callable[[str], Optional[str]]      # (tenant_id) -> token or None
LockReleaseFn = Callable[[str, str], bool]          # (tenant_id, token) -> released?


def run_model_performance_aggregator(
    conn: Any,
    *,
    tenant_id: str,
    params: Optional[TenantParams] = None,
    as_of: Optional[dt.date] = None,
    period_days: int = PERIOD_DAYS,
    stable_band: float = MODEL_PERFORMANCE_STABLE_BAND,
    statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
    log_failure_fn: Optional[Callable[[str, str, str, str], None]] = None,
    acquire_lock_fn: Optional[LockAcquireFn] = None,
    release_lock_fn: Optional[LockReleaseFn] = None,
) -> AggregatorStats:
    """
    Aggregate forecast_outcomes for [as_of-30, as_of) and the prior
    30-day window, compute trend, write to model_performance_s9.

    `as_of` defaults to today UTC (NOT host-local timezone). Windows
    are EXCLUSIVE on the right so today's still-being-collected
    outcomes don't pollute the average.

    `statement_timeout_ms` is set per-cursor via `SET LOCAL` so a
    pathological query plan can't hold up the entire batch.

    `acquire_lock_fn` / `release_lock_fn` (optional) serialize
    concurrent runs for the same tenant. Production callers wire
    these to a Redis lock keyed on the aggregator job. If
    acquire_lock_fn returns None, the run is skipped — another
    instance is already running, log it and move on.

    `log_failure_fn(tenant_id, run_id, sku_id, reason)` records batch
    failures for the audit trail; we use sku_id="<batch>" for
    tenant-level failures (no SKU context).

    `params` (optional TenantParams snapshot) — when supplied, the
    stable_band is read from `model_performance_stable_band` so the
    threshold is per-tenant tunable. The function-arg `stable_band`
    is the fallback when `params` is None (test-friendly).

    Returns AggregatorStats. On failure, stats.failure_reason is the
    human-readable label and stats.failure_exception is the original
    exception (for callers branching by type).
    """
    warn_if_shared_conn(conn, label="ModelPerformanceAggregator")

    # Tenant-tunable stable_band: read from params when supplied.
    if params is not None:
        stable_band = float(params.get(_STABLE_BAND_PARAM))

    if as_of is None:
        # Use UTC explicitly so the window doesn't shift by ±1 day
        # depending on which timezone the scheduler host runs in.
        as_of = dt.datetime.now(dt.timezone.utc).date()

    period_end = as_of
    period_start = as_of - dt.timedelta(days=period_days)
    prior_end = period_start
    prior_start = period_start - dt.timedelta(days=period_days)

    stats = AggregatorStats(
        period_start=period_start,
        period_end=period_end,
    )

    # Outer perimeter: the inner blocks already catch the two
    # expected failure modes (query, upsert) and return cleanly. This
    # try/except guards against ANYTHING else that escapes —
    # _build_rows raising, a future helper that doesn't catch its own
    # errors, an unexpected library exception. Without it, an
    # unhandled exception breaks the function's documented "always
    # returns AggregatorStats" contract.
    try:
        with _maybe_locked(
            tenant_id, acquire_lock_fn, release_lock_fn, stats,
        ) as lock_ok:
            if not lock_ok:
                # Another aggregator instance holds the lock. Log and
                # bail — not an error, just contention.
                logger.info(
                    "aggregator skip lock_held tenant=%s as_of=%s",
                    tenant_id, as_of,
                )
                return stats

            try:
                current = _aggregate_window(
                    conn, tenant_id, period_start, period_end,
                    statement_timeout_ms,
                )
                prior = _aggregate_window(
                    conn, tenant_id, prior_start, prior_end,
                    statement_timeout_ms,
                )
            except MemoryError:
                raise
            except Exception as e:
                logger.exception(
                    "aggregator query failed tenant=%s as_of=%s err=%s",
                    tenant_id, as_of, e,
                )
                stats.failure_reason = f"query_failed:{e}"
                stats.failure_exception = e
                if log_failure_fn:
                    _safe_log_failure(log_failure_fn, tenant_id, stats.failure_reason)
                return stats

            rows = _build_rows(
                tenant_id=tenant_id,
                current=current,
                prior=prior,
                period_start=period_start,
                period_end=period_end,
                stable_band=stable_band,
                stats=stats,
            )

            if not rows:
                logger.debug(
                    "aggregator no_rows tenant=%s window=[%s, %s)",
                    tenant_id, period_start, period_end,
                )
                return stats

            try:
                _delete_then_insert(conn, tenant_id, rows, statement_timeout_ms)
                conn.commit()
                stats.rows_written = len(rows)
            except MemoryError:
                raise
            except Exception as e:
                logger.exception(
                    "aggregator upsert failed tenant=%s err=%s", tenant_id, e,
                )
                try:
                    conn.rollback()
                except Exception:
                    logger.debug(
                        "rollback failed during aggregator upsert", exc_info=True,
                    )
                stats.failure_reason = f"upsert_failed:{e}"
                stats.failure_exception = e
                stats.rows_written = 0
                if log_failure_fn:
                    _safe_log_failure(log_failure_fn, tenant_id, stats.failure_reason)

    except MemoryError:
        raise
    except Exception as e:
        # Inner blocks didn't catch this — populate stats only if the
        # inner blocks haven't already (in which case we'd overwrite a
        # more diagnostic message with a generic one).
        logger.exception(
            "aggregator unexpected error tenant=%s err=%s", tenant_id, e,
        )
        if stats.failure_reason is None:
            stats.failure_reason = f"unexpected_error:{e}"
            stats.failure_exception = e
            if log_failure_fn:
                _safe_log_failure(log_failure_fn, tenant_id, stats.failure_reason)
        # Best-effort rollback so we don't leak an open txn back to
        # the caller.
        try:
            conn.rollback()
        except Exception:
            logger.debug(
                "rollback failed in outer perimeter", exc_info=True,
            )

    return stats


@contextmanager
def _maybe_locked(
    tenant_id: str,
    acquire_lock_fn: Optional[LockAcquireFn],
    release_lock_fn: Optional[LockReleaseFn],
    stats: AggregatorStats,
) -> Iterator[bool]:
    """
    Acquire-and-release wrapper. Yields True when the lock is held
    (or no locking was configured), False when another instance owns
    the lock.
    """
    if acquire_lock_fn is None:
        yield True
        return

    token = acquire_lock_fn(tenant_id)
    if token is None:
        stats.lock_acquired = False
        yield False
        return

    try:
        yield True
    finally:
        if release_lock_fn is not None:
            try:
                release_lock_fn(tenant_id, token)
            except Exception:
                logger.exception(
                    "aggregator lock release failed tenant=%s", tenant_id,
                )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _aggregate_window(
    conn: Any,
    tenant_id: str,
    start: dt.date,
    end: dt.date,
    statement_timeout_ms: int,
) -> dict[tuple[str, int], dict]:
    """
    Run the aggregation SQL against [start, end). Returns a dict keyed
    by (model_name, horizon_days) with the per-group stats.
    """
    out: dict[tuple[str, int], dict] = {}
    with conn.cursor() as cur:
        _set_statement_timeout(cur, statement_timeout_ms)
        cur.execute(_AGGREGATE_SQL, (tenant_id, start, end))
        for row in cur.fetchall():
            model, horizon, avg, med, p90, bias, count = row
            out[(model, int(horizon))] = {
                "avg_mape": _to_float(avg),
                "median_mape": _to_float(med),
                "p90_mape": _to_float(p90),
                "avg_bias": _to_float(bias),
                "sample_count": int(count),
            }
    return out


def _set_statement_timeout(cur: Any, timeout_ms: int) -> None:
    """
    SET LOCAL only applies for the current transaction, so each
    cursor block sets it again. A pathological forecast_outcomes
    plan should hit this cap, not block the batch indefinitely.
    """
    if timeout_ms <= 0:
        return
    cur.execute(_SET_STATEMENT_TIMEOUT_SQL, (int(timeout_ms),))


def _build_rows(
    *,
    tenant_id: str,
    current: dict,
    prior: dict,
    period_start: dt.date,
    period_end: dt.date,
    stable_band: float,
    stats: AggregatorStats,
) -> list[ModelPerformanceRow]:
    """
    Walk current-window groups, joining with prior-window groups on
    (model, horizon). Discontinued models (in prior, not current) are
    NOT written — model_performance_s9 reflects what's currently in
    play, not history. Stats-level new/discontinued tracking is at the
    MODEL level (across all horizons) to avoid the ambiguous "Prophet
    appears in both lists" case when only some horizons changed.
    """
    current_keys = set(current.keys())
    prior_keys = set(prior.keys())

    # Compare at model level — a model is "new" only if NONE of its
    # horizons existed in prior; "discontinued" only if NONE of its
    # horizons exist in current.
    current_models = {m for (m, _) in current_keys}
    prior_models = {m for (m, _) in prior_keys}
    stats.new_models = sorted(current_models - prior_models)
    stats.discontinued_models = sorted(prior_models - current_models)

    rows: list[ModelPerformanceRow] = []
    for key, agg in current.items():
        model, horizon = key
        prior_agg = prior.get(key)
        prior_avg = prior_agg["avg_mape"] if prior_agg else None

        improvement = _improvement(agg["avg_mape"], prior_avg)
        trend = _classify_trend(improvement, stable_band)

        rows.append(ModelPerformanceRow(
            tenant_id=tenant_id,
            model_name=model,
            horizon_days=horizon,
            period_start=period_start,
            period_end=period_end,
            avg_mape=agg["avg_mape"],
            median_mape=agg["median_mape"],
            p90_mape=agg["p90_mape"],
            avg_bias=agg["avg_bias"],
            sample_count=agg["sample_count"],
            prior_avg_mape=prior_avg,
            improvement_vs_prior=improvement,
            mape_trend=trend,
        ))
    return rows


def _improvement(
    current_mape: Optional[float], prior_mape: Optional[float],
) -> Optional[float]:
    """current - prior. Negative = MAPE went down (good)."""
    if current_mape is None or prior_mape is None:
        return None
    return float(current_mape) - float(prior_mape)


def _classify_trend(improvement: Optional[float], stable_band: float) -> str:
    """
    improvement < -stable_band → 'improving' (MAPE down beyond band)
    improvement > +stable_band → 'degrading' (MAPE up beyond band)
    else                       → 'stable'
    None (new or discontinued model) → 'stable' (no comparable signal)

    Rounds to 9 decimal places before comparing — without this, float
    noise on near-boundary deltas (e.g. 0.15 - 0.16 = -0.0100000000001)
    flips the classification.
    """
    if improvement is None:
        return TREND_STABLE
    imp = round(float(improvement), 9)
    band = round(float(stable_band), 9)
    if imp < -band:
        return TREND_IMPROVING
    if imp > band:
        return TREND_DEGRADING
    return TREND_STABLE


def _delete_then_insert(
    conn: Any,
    tenant_id: str,
    rows: list[ModelPerformanceRow],
    statement_timeout_ms: int,
) -> None:
    """
    Idempotent write: clear all model_performance_s9 rows for the
    tenant, then bulk-insert the fresh rows. Uses execute_values when
    the cursor is real psycopg2 (≥100× throughput vs executemany on
    bulk inserts); falls back to executemany for fakes in tests.
    """
    insert_args = [_row_to_args(r) for r in rows]
    with conn.cursor() as cur:
        _set_statement_timeout(cur, statement_timeout_ms)
        cur.execute(_DELETE_TENANT_ROWS_SQL, (tenant_id,))
        _bulk_insert(cur, insert_args)


def _row_to_args(row: ModelPerformanceRow) -> tuple:
    """Project a ModelPerformanceRow into the positional args tuple
    matching _INSERT_COLS order. Schema column names are mapped back to
    dataclass attribute names via _COL_TO_FIELD where they differ."""
    return tuple(
        getattr(row, _COL_TO_FIELD.get(col, col))
        for col in _INSERT_COLS
    )


def _bulk_insert(cur: Any, insert_args: list[tuple]) -> None:
    """
    execute_values needs a real psycopg2 cursor (it inspects
    cur.connection.encoding). Fake cursors in tests lack .connection,
    so fall through to executemany — same single-row INSERT, just
    chattier on the wire.
    """
    use_execute_values = (
        _execute_values is not None and hasattr(cur, "connection")
    )
    if use_execute_values:
        _execute_values(
            cur, _INSERT_BULK_SQL, insert_args,
            template=_INSERT_BULK_TEMPLATE,
            page_size=INSERT_PAGE_SIZE,
        )
    else:
        cur.executemany(_INSERT_ROW_SQL, insert_args)


def _to_float(value: Any) -> Optional[float]:
    """
    Coerce a DB-returned scalar to float. NaN is treated as missing —
    PostgreSQL's AVG normally drops NaN before aggregation, but if a
    pathological row sneaks through (or a custom adapter returns
    float('nan') for a NULL), we surface it as None so the column
    doesn't propagate NaN through the row build.
    """
    if value is None:
        return None
    f = float(value)
    if math.isnan(f):
        return None
    return f


def _safe_log_failure(
    log_failure_fn: Callable[[str, str, str, str], None],
    tenant_id: str,
    reason: str,
) -> None:
    """
    Tenant-level batch failures fill log_failure_fn args as:
        run_id  = "<aggregator>"  (the job that failed; not a real run)
        sku_id  = "<batch>"       (sentinel — no per-SKU context)
    """
    try:
        log_failure_fn(tenant_id, "<aggregator>", "<batch>", reason)
    except Exception:
        logger.exception(
            "log_failure_fn raised — failure not recorded tenant=%s",
            tenant_id,
        )


_FETCH_ALL_TENANTS_SQL = """
    SELECT DISTINCT tenant_id
    FROM stage9.tenant_learning_params
    ORDER BY tenant_id
"""


def main() -> None:
    """
    Iterate all tenants and run the rolling-window MAPE aggregation.
    Called by the nightly scheduler at 4:00 AM UTC (before LearningParamsUpdater).
    Each tenant gets a dedicated connection so commits are fully isolated.
    """
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_FETCH_ALL_TENANTS_SQL)
            tenant_ids = [str(row[0]) for row in cur.fetchall()]

    results: dict[str, int] = {"ok": 0, "failed": 0, "no_data": 0}

    for tenant_id in tenant_ids:
        try:
            with pg_conn() as conn:
                params = TenantParams.load(tenant_id, conn)
                stats = run_model_performance_aggregator(
                    conn,
                    tenant_id=tenant_id,
                    params=params if len(params) > 0 else None,
                )
            if stats.failure_reason:
                logger.error(
                    "aggregator FAILED tenant=%s reason=%s",
                    tenant_id, stats.failure_reason,
                )
                results["failed"] += 1
            elif stats.rows_written == 0:
                results["no_data"] += 1
            else:
                results["ok"] += 1
        except Exception as exc:
            logger.error(
                "aggregator FAILED tenant=%s: %s", tenant_id, exc,
            )
            results["failed"] += 1

    logger.info("ModelPerformanceAggregator finished: %s", results)


if __name__ == "__main__":
    main()
