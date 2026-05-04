"""
Stage 9 — OutcomeCollector Batch Job
=====================================

Compares past forecasts to actual sales after each horizon period closes.
Writes ground-truth rows to `forecast_outcomes` — the learning signal that
drives every downstream Stage 9 batch job (4 AM ModelPerformanceAggregator,
4:30 AM LearningParamsUpdater, 5 AM SimilarityRegistryUpdater(skipped for this phase)).

Scheduled at 3 AM tenant local time. The orchestration layer (cron / Lambda /
ECS) is responsible for tenant discovery and timezone math; this module
exposes one public function:

    run_for_tenant(tenant_id, conn) -> dict
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

from infrastructure.constants import HORIZONS

logger = logging.getLogger("stage9.outcome_collector")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Commit cadence . Bounds memory and rollback size.
BATCH_COMMIT_SIZE: int = 100

# Decimal quantizers for column scales.
_Q_ERROR = Decimal("0.001")   # DECIMAL(5,3)  — error_mape, error_wape, bias
_Q_VALUE = Decimal("0.0001")  # DECIMAL(12,4) — forecast_value, actual_value


# ---------------------------------------------------------------------------
# Pure helpers — unit-testable without a DB
# ---------------------------------------------------------------------------

def _quantile_key(selected_quantile: Any) -> str:
    """
    Map a selected_quantile value to the JSONB key name in the forecast_{H}d
    columns. Stage 9 stores p50, p80, p90 only. Any quantile above 0.80 falls
    back to p90 (Stage 9 → Stage 10 contract: p90 is the ceiling).

    Mapping:
        0.50        -> 'p50'
        0.80        -> 'p80'
        0.90 / other(overrides) -> 'p90'  (conservative default)

    Conversion goes through Decimal(str(...)) to avoid IEEE-754 float issues.
    """
    q = Decimal(str(selected_quantile))
    if q == Decimal("0.50"):
        return "p50"
    if q == Decimal("0.80"):
        return "p80"
    # 0.90 and any unrecognised value map to the p90 ceiling.
    return "p90"


def _to_decimal(value: Any) -> Optional[Decimal]:
    """
    Coerce a JSONB-extracted number to Decimal. Returns None if not numeric.

    Always stringifies first to avoid IEEE-754 float representation issues
    (e.g. Decimal(0.1 + 0.2) != Decimal("0.3")).
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError):
        return None


def _compute_errors(
    forecast_value: Decimal,
    actual_value: Decimal,
) -> tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
    """
    Compute (error_mape, error_wape, bias)

    Formulae:
        error_mape = |actual - forecast| / actual
        error_wape = same as MAPE for a single-row outcome (collapses to MAPE
                     because there is no cross-SKU weighting at this stage)
        bias       = (forecast - actual) / actual
                     positive -> over-forecast, negative -> under-forecast

    When actual_value == 0:
        All three are None — MAPE undefined. Do NOT write 0 or infinity.
        Writing 0 would falsely signal a perfect forecast; writing inf would
        poison every downstream AVG(error_mape). NULL tells the aggregator to
        exclude this row from percentage-error averages.

    Negative actuals are not expected (units_adjusted >= 0 upstream) but the
    formulae remain correct if they occur.
    """
    if actual_value == 0:
        return None, None, None

    abs_err = abs(actual_value - forecast_value)

    error_mape = (abs_err / actual_value).quantize(_Q_ERROR, rounding=ROUND_HALF_UP)

    # WAPE collapses to MAPE for a single-row outcome — confirmed simple per spec.
    error_wape = error_mape

    bias = ((forecast_value - actual_value) / actual_value).quantize(
        _Q_ERROR, rounding=ROUND_HALF_UP
    )
    return error_mape, error_wape, bias


# ---------------------------------------------------------------------------
# DB interactions — thin, mockable wrappers
# ---------------------------------------------------------------------------

def _fetch_pending(conn, tenant_id: str, horizon_days: int) -> list[dict]:
    """
    Anti-join: forecasts old enough for this horizon that have no matching
    forecast_outcomes row yet. Pulls every column needed to compute the
    outcome in a single round-trip .

    LEFT JOIN ... WHERE fo.tenant_id IS NULL pattern means re-runs are
    idempotent at the read side — already-written rows simply won't appear.

    Parameterised placeholders prevent SQL injection .
    """
    sql = """
        SELECT
            f.tenant_id,
            f.sku_id,
            f.run_id,
            f.assigned_model,
            f.selected_quantile,
            DATE(f.created_at)            AS forecast_date,
            f.forecast_7d,
            f.forecast_14d,
            f.forecast_30d,
            f.forecast_60d,
            f.forecast_90d,
            f.forecast_150d,
            f.forecast_180d,
            f.forecast_365d
        FROM forecasts f
        LEFT JOIN forecast_outcomes fo
               ON fo.tenant_id    = f.tenant_id
              AND fo.sku_id       = f.sku_id
              AND fo.run_id       = f.run_id
              AND fo.horizon_days = %(horizon_days)s
        WHERE f.tenant_id = %(tenant_id)s
          AND DATE(f.created_at) <= CURRENT_DATE - %(horizon_days)s::int
          AND fo.tenant_id IS NULL;
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, {"tenant_id": tenant_id, "horizon_days": horizon_days})
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()


def _fetch_actual(
    conn,
    tenant_id: str,
    sku_id: str,
    forecast_date: date,
    horizon_days: int,
) -> tuple[Optional[Decimal], int]:
    """
    Sum stage8.clean_orders.quantity_sold over [forecast_date, forecast_date + horizon_days).

    Half-open interval is intentional: day 0 included, day H excluded —
    gives exactly horizon_days calendar days of demand data .

    Returns:
        (actual_total, day_count)

        day_count == 0:
            No rows in clean_orders for this window yet. Actuals not available.
            Caller must skip this row and retry on the next scheduled run.

        day_count > 0, actual_total == Decimal("0"):
            Actuals ARE available; the SKU genuinely sold zero units.
            Caller proceeds and writes the row with NULL error metrics.
    """
    sql = """
        SELECT COALESCE(SUM(quantity_sold), 0)::NUMERIC AS actual_total,
               COUNT(*)                                 AS day_count
        FROM stage8.clean_orders
        WHERE tenant_id        = %(tenant_id)s
          AND canonical_sku_id = %(sku_id)s
          AND order_date      >= %(forecast_date)s
          AND order_date       < %(forecast_date)s + %(horizon_days)s::int;
    """
    cur = conn.cursor()
    try:
        cur.execute(
            sql,
            {
                "tenant_id": tenant_id,
                "sku_id": sku_id,
                "forecast_date": forecast_date,
                "horizon_days": horizon_days,
            },
        )
        row = cur.fetchone()
    finally:
        cur.close()

    if row is None:
        return None, 0

    actual_total_raw, day_count = row
    # Only convert to Decimal when rows exist; day_count == 0 means no data yet.
    actual_total = _to_decimal(actual_total_raw) if day_count else None
    return actual_total, int(day_count)


def _write_outcome(
    conn,
    *,
    tenant_id: str,
    sku_id: str,
    run_id: str,
    horizon_days: int,
    assigned_model: Optional[str],
    forecast_value: Decimal,
    actual_value: Decimal,
    error_mape: Optional[Decimal],
    error_wape: Optional[Decimal],
    bias: Optional[Decimal],
) -> None:
    """
    INSERT into forecast_outcomes. Idempotent via PRIMARY KEY conflict.

    ON CONFLICT DO NOTHING on (tenant_id, sku_id, run_id, horizon_days)
    means re-running the job on the same day produces no duplicates and no
    errors.

    outcome_date is set to CURRENT_DATE at the DB server — it records when
    the outcome was collected, not when the forecast was made.

    The caller is responsible for COMMIT cadence; this function never commits.
    """
    sql = """
        INSERT INTO forecast_outcomes (
            tenant_id, sku_id, run_id, horizon_days,
            assigned_model, forecast_value, actual_value,
            error_mape, error_wape, bias, outcome_date
        ) VALUES (
            %(tenant_id)s, %(sku_id)s, %(run_id)s, %(horizon_days)s,
            %(assigned_model)s, %(forecast_value)s, %(actual_value)s,
            %(error_mape)s, %(error_wape)s, %(bias)s, CURRENT_DATE
        )
        ON CONFLICT (tenant_id, sku_id, run_id, horizon_days) DO NOTHING;
    """
    cur = conn.cursor()
    try:
        cur.execute(
            sql,
            {
                "tenant_id": tenant_id,
                "sku_id": sku_id,
                "run_id": run_id,
                "horizon_days": horizon_days,
                "assigned_model": assigned_model,
                "forecast_value": forecast_value,
                "actual_value": actual_value,
                "error_mape": error_mape,
                "error_wape": error_wape,
                "bias": bias,
            },
        )
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# Per-horizon driver 
# ---------------------------------------------------------------------------

def _process_horizon(
    conn,
    tenant_id: str,
    horizon_days: int,
    uncommitted_counter: list[int],
) -> dict:
    """
    Drive one horizon for one tenant: fetch pending forecasts, compute
    outcomes, write rows, commit every BATCH_COMMIT_SIZE writes.

    Per-row sequence follows:
        Step 2  Fetch actual sales from golden_table for the horizon window.
        Step 3  Extract the quantile-specific forecast value from JSONB.
        Step 4  Compute error metrics. Round values to DB column scales.
        Step 5  Write the outcome row.
        Step 6  Commit every BATCH_COMMIT_SIZE rows.

    uncommitted_counter:
        A single-element list [int] used as a mutable integer shared with
        run_for_tenant(). Python integers are immutable so the list wrapper
        is the standard idiom for passing a counter by reference. This keeps
        the commit cadence continuous across all 8 horizons — we never force
        a commit just because a horizon boundary was crossed (spec §4 Step 6).

    Per-row isolation (spec §6, Stage 9 Principle 3):
        Exceptions inside the try/except are logged with full stack trace and
        the row is skipped. The row stays pending and retries on the next run.

    Returns:
        dict with keys: pending, written, skipped_no_actuals, errors
    """
    stats = {"pending": 0, "written": 0, "skipped_no_actuals": 0, "errors": 0}

    pending = _fetch_pending(conn, tenant_id, horizon_days)
    stats["pending"] = len(pending)

    if not pending:
        logger.info(
            "tenant=%s horizon=%d: no pending forecasts",
            tenant_id, horizon_days,
        )
        return stats

    # Derive the JSONB column name from the horizon constant.
    # HORIZONS is spec-locked so this always maps to a real column (forecast_7d
    # through forecast_365d).
    horizon_col = f"forecast_{horizon_days}d"

    for row in pending:
        sku_id = row["sku_id"]
        run_id = row["run_id"]

        try:
            # ── Step 2: fetch actuals ──────────────────────────────────────
            actual_value, day_count = _fetch_actual(
                conn,
                tenant_id=tenant_id,
                sku_id=sku_id,
                forecast_date=row["forecast_date"],
                horizon_days=horizon_days,
            )

            if day_count == 0 or actual_value is None:
                # No golden_table rows for this window yet — timing skip.
                # The self-healing WHERE clause picks it up on the next run.
                stats["skipped_no_actuals"] += 1
                logger.debug(
                    "tenant=%s sku=%s run=%s horizon=%d: actuals unavailable, skipping",
                    tenant_id, sku_id, run_id, horizon_days,
                )
                continue

            # ── Step 3: extract forecast value from JSONB ──────────────────
            # psycopg2's JSON adapter returns JSONB columns as Python dicts.
            jsonb = row[horizon_col]
            if jsonb is None:
                # NULL JSONB column is a data-quality issue on forecasts table.
                logger.warning(
                    "tenant=%s sku=%s run=%s horizon=%d: %s is NULL — skipping",
                    tenant_id, sku_id, run_id, horizon_days, horizon_col,
                )
                stats["errors"] += 1
                continue

            key = _quantile_key(row["selected_quantile"])
            # isinstance guard handles the unlikely case where psycopg2
            # returns JSONB as a raw string rather than a parsed dict.
            fc_raw = jsonb.get(key) if isinstance(jsonb, dict) else None
            forecast_value = _to_decimal(fc_raw)

            if forecast_value is None:
                logger.warning(
                    "tenant=%s sku=%s run=%s horizon=%d: forecast key %r missing "
                    "or non-numeric in %s — skipping",
                    tenant_id, sku_id, run_id, horizon_days, key, horizon_col,
                )
                stats["errors"] += 1
                continue

            # ── Step 4: compute errors, then round to DB column scales ─────
            # Errors computed at full precision first; rounding happens after
            # so the math is not affected by the scale reduction.
            error_mape, error_wape, bias = _compute_errors(
                forecast_value, actual_value
            )

            forecast_value_q = forecast_value.quantize(
                _Q_VALUE, rounding=ROUND_HALF_UP
            )
            actual_value_q = actual_value.quantize(
                _Q_VALUE, rounding=ROUND_HALF_UP
            )

            # ── Step 5: write outcome row ──────────────────────────────────
            _write_outcome(
                conn,
                tenant_id=tenant_id,
                sku_id=sku_id,
                run_id=run_id,
                horizon_days=horizon_days,
                assigned_model=row.get("assigned_model"),
                forecast_value=forecast_value_q,
                actual_value=actual_value_q,
                error_mape=error_mape,
                error_wape=error_wape,
                bias=bias,
            )
            stats["written"] += 1
            uncommitted_counter[0] += 1

            # ── Step 6: commit cadence ─────────────────────────────────────
            if uncommitted_counter[0] >= BATCH_COMMIT_SIZE:
                conn.commit()
                uncommitted_counter[0] = 0

        except Exception as exc:
            # Per-SKU isolation — log full stack trace and continue.
            # Bad row stays pending and retries on the next scheduled run.
            stats["errors"] += 1
            logger.warning(
                "tenant=%s sku=%s run=%s horizon=%d: row failed: %s",
                tenant_id, sku_id, run_id, horizon_days, exc,
                exc_info=True,
            )

    logger.info(
        "tenant=%s horizon=%d: pending=%d written=%d skipped_no_actuals=%d errors=%d",
        tenant_id, horizon_days,
        stats["pending"], stats["written"],
        stats["skipped_no_actuals"], stats["errors"],
    )
    return stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_for_tenant(tenant_id: str, conn) -> dict:
    """
    Run the OutcomeCollector for one tenant.

    Iterates through all 8 horizons in HORIZONS, finds forecast rows whose
    horizon has elapsed and have no outcome yet, queries golden_table for
    actuals, computes MAPE/WAPE/bias, writes rows to forecast_outcomes.

    Idempotency :
        Re-running the same day produces no duplicates — the anti-join in
        _fetch_pending returns no rows for already-written outcomes, and
        _write_outcome uses ON CONFLICT DO NOTHING as a second safety net.

    Self-healing :
        A missed run day is recovered automatically on the next run. The
        WHERE DATE(created_at) <= CURRENT_DATE - H predicate is the only
        recovery mechanism — no high-water-mark table or replay queue needed.

    Args:
        tenant_id: UUID string for the tenant.
        conn:      Open psycopg2 connection. Caller owns its lifecycle.

    Returns:
        Summary dict:
            {
                "tenant_id": str,
                "started_at": ISO-8601 UTC string,
                "finished_at": ISO-8601 UTC string,
                "duration_seconds": float,
                "per_horizon": {
                    7:  {"pending": int, "written": int,
                         "skipped_no_actuals": int, "errors": int},
                    ... (one entry per value in HORIZONS)
                },
                "total_written": int,
                "total_skipped_no_actuals": int,
                "total_errors": int,
            }

    Raises:
        Exception: any unrecovered tenant-level DB exception is re-raised
                   after rolling back all uncommitted writes.
    """
    started_at = datetime.now(timezone.utc)
    logger.info("tenant=%s: OutcomeCollector starting", tenant_id)

    per_horizon: dict[int, dict] = {}

    # Single-element list used as a mutable integer shared across all horizons.
    # Python integers are immutable — the list wrapper is the standard idiom
    # for passing a mutable counter by reference across function calls.
    uncommitted_counter = [0]

    try:
        for horizon_days in HORIZONS:
            per_horizon[horizon_days] = _process_horizon(
                conn, tenant_id, horizon_days, uncommitted_counter
            )

        # Final flush — commit any tail rows that never filled a full batch.
        if uncommitted_counter[0] > 0:
            conn.commit()

    except Exception:
        # Tenant-level failure: roll back all uncommitted writes so the DB
        # stays consistent. The next scheduled run re-processes pending rows.
        try:
            conn.rollback()
        except Exception:
            pass
        logger.exception("tenant=%s: OutcomeCollector failed", tenant_id)
        raise

    finished_at = datetime.now(timezone.utc)
    total_written = sum(s["written"] for s in per_horizon.values())
    total_skipped = sum(s["skipped_no_actuals"] for s in per_horizon.values())
    total_errors = sum(s["errors"] for s in per_horizon.values())

    summary = {
        "tenant_id": tenant_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "per_horizon": per_horizon,
        "total_written": total_written,
        "total_skipped_no_actuals": total_skipped,
        "total_errors": total_errors,
    }
    logger.info(
        "tenant=%s: OutcomeCollector finished — written=%d skipped=%d errors=%d duration=%.2fs",
        tenant_id, total_written, total_skipped, total_errors,
        summary["duration_seconds"],
    )
    return summary
