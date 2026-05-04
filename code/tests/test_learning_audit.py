"""
test_learning_audit.py — Unit-tests the three pure helpers in outcome_collector,
then seeds SKUs, drives the full pipeline to produce forecasts, backdates them,
seeds clean_orders actuals, and runs the three nightly batch jobs end-to-end.

Run:
    STAGE9_TEST_DSN="postgresql://postgres:Joyboy@localhost:5432/dev?sslmode=disable" \
    STAGE9_PROJECT_ROOT="M:/stage_9/code" \
    python -m pytest tests/test_learning_audit.py -v -s
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uuid
import pytest
import psycopg2
from psycopg2.extras import Json
from datetime import date, timedelta
from decimal import Decimal

from learning.outcome_collector import (
    _quantile_key,
    _to_decimal,
    _compute_errors,
    run_for_tenant,
)
from learning.model_performance_aggregator import run_model_performance_aggregator
from learning.learning_params_updater import LearningParamsUpdater

from infrastructure.state_machine import AgentState, transition
from handlers.preloading  import preloading_handler
from handlers.perceiving  import perceiving_handler
from handlers.planning    import planning_handler
from handlers.acting      import acting_handler
from handlers.learning    import learning_handler
from handlers.reporting   import reporting_handler
from infrastructure.seed import seed_tenant_params
from tests.stage9_data_factory import (
    gen_cold_start, gen_stable, gen_trending, gen_intermittent,
)

from infrastructure.config import DB_DSN as _DSN  # noqa: E402

# Skip if local stage8 is view-aliased (see _stage8_real_schema_required.py).
from tests._stage8_real_schema_required import skip_if_stage8_uses_views  # noqa: E402
skip_if_stage8_uses_views()

TENANT_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "learning-audit-tenant-001"))

# No seasonal SKU — Prophet Stan chains add ~8 s per run.
SKUS = [
    dict(code="LRNA-CS",  pattern="cold_start",   model_hint="Naive",   obs_days=25,
         df_fn=lambda: gen_cold_start("LRNA-CS",  n_days=25)),
    dict(code="LRNA-STB", pattern="stable",        model_hint="exponential_smoothing", obs_days=90,
         df_fn=lambda: gen_stable("LRNA-STB",     n_days=90, daily_mean=20.0)),
    dict(code="LRNA-TRN", pattern="trending",      model_hint="Holt",   obs_days=120,
         df_fn=lambda: gen_trending("LRNA-TRN",   n_days=120, daily_mean=8.0, trend_slope=0.08)),
    dict(code="LRNA-INT", pattern="intermittent",  model_hint="Croston", obs_days=180,
         df_fn=lambda: gen_intermittent("LRNA-INT", n_days=180, zero_ratio=0.65)),
]

STAGE9_TABLES = [
    "forecasts",
    "thompson_sampling_state",
    "feature_decisions_s9",
    "hyperparameter_decisions",
    "backtest_decisions",
    "data_fingerprint_cache",
    "sku_similarity_registry",
    "model_initialization_s9",
    "stage9_self_assessment",
    "cross_agent_signals",
    "stage9_sku_execution_log",
    "agent_state_log_s9",
    "forecast_outcomes",
    "model_performance_s9",
]


# ===========================================================================
# Pure-unit tests — no DB needed
# ===========================================================================

class TestQuantileKey:
    def test_p50(self):
        assert _quantile_key(0.50) == "p50"

    def test_p80(self):
        assert _quantile_key(0.80) == "p80"

    def test_p90(self):
        assert _quantile_key(0.90) == "p90"

    def test_unknown_maps_to_p90(self):
        assert _quantile_key(0.75) == "p90"

    def test_string_input(self):
        assert _quantile_key("0.50") == "p50"


class TestToDecimal:
    def test_none_returns_none(self):
        assert _to_decimal(None) is None

    def test_int_converts(self):
        assert _to_decimal(5) == Decimal("5")

    def test_float_converts(self):
        assert _to_decimal(1.5) == Decimal("1.5")

    def test_string_invalid_returns_none(self):
        assert _to_decimal("not-a-number") is None

    def test_ieee754_avoidance(self):
        # 0.1 + 0.2 as float has representation error; stringify avoids it
        result = _to_decimal(0.1)
        assert result is not None
        assert isinstance(result, Decimal)


class TestComputeErrors:
    def test_normal_over_forecast(self):
        # forecast=12, actual=10 → mape=0.200, bias=+0.200
        mape, wape, bias = _compute_errors(Decimal("12"), Decimal("10"))
        assert mape == Decimal("0.200")
        assert wape == Decimal("0.200")
        assert bias == Decimal("0.200")

    def test_normal_under_forecast(self):
        # forecast=8, actual=10 → mape=0.200, bias=-0.200
        mape, wape, bias = _compute_errors(Decimal("8"), Decimal("10"))
        assert mape == Decimal("0.200")
        assert bias == Decimal("-0.200")

    def test_perfect_forecast(self):
        mape, wape, bias = _compute_errors(Decimal("10"), Decimal("10"))
        assert mape == Decimal("0.000")
        assert bias == Decimal("0.000")

    def test_zero_actual_all_none(self):
        mape, wape, bias = _compute_errors(Decimal("5"), Decimal("0"))
        assert mape is None
        assert wape is None
        assert bias is None

    def test_wape_equals_mape_single_row(self):
        mape, wape, bias = _compute_errors(Decimal("15"), Decimal("10"))
        assert mape == wape


# ===========================================================================
# Integration helpers
# ===========================================================================

def _connect():
    conn = psycopg2.connect(_DSN, connect_timeout=5)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO stage9, public")
        cur.execute("SET lock_timeout = '10s'")
    conn.commit()
    return conn


def _kill_idle_test_connections(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE state IN ('idle', 'idle in transaction')
              AND pid <> pg_backend_pid()
              AND datname = current_database()
        """)
    conn.commit()


def _truncate_all(cur):
    for t in STAGE9_TABLES:
        cur.execute(f"TRUNCATE TABLE stage9.{t} CASCADE")
    cur.execute("DELETE FROM stage8.demand_history   WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.pattern_history  WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.feature_decisions WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.signal_context   WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.runs             WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.clean_orders     WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.canonical_sku    WHERE tenant_id = %s", (TENANT_ID,))


def _seed_sku(cur, run_id, sku):
    sku_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{TENANT_ID}-{sku['code']}"))

    cur.execute("""
        INSERT INTO stage8.canonical_sku (sku_id, tenant_id, vendor, product_type)
        VALUES (%s, %s, 'LearnVendor', 'learning_test')
        ON CONFLICT (sku_id) DO NOTHING
    """, (sku_uuid, TENANT_ID))

    df = sku["df_fn"]()
    rows = [
        (TENANT_ID, sku_uuid, row["order_date"], float(row["quantity"]))
        for _, row in df.iterrows()
    ]
    cur.executemany(
        "INSERT INTO stage8.demand_history (tenant_id, sku_id, sale_date, qty) VALUES (%s,%s,%s,%s)",
        rows,
    )

    lifecycle = "introduction" if sku["obs_days"] < 28 else "saturation"
    cur.execute("""
        INSERT INTO stage8.pattern_history
            (tenant_id, sku_id, run_id, pattern_label, confidence_calibrated,
             model_hint, on_watchlist, observation_days, lifecycle_stage, drift_detected)
        VALUES (%s,%s,%s,%s,0.85,%s,FALSE,%s,%s,FALSE)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
    """, (TENANT_ID, sku_uuid, run_id, sku["pattern"], sku["model_hint"],
          sku["obs_days"], lifecycle))

    rel = {"trend": 0.85, "seasonality": 0.85, "zero_ratio": 0.95, "cv": 0.90}
    cur.execute("""
        INSERT INTO stage8.feature_decisions
            (tenant_id, sku_id, run_id, feature_reliability_map,
             weekend_zero_ratio, velocity_signature)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
    """, (TENANT_ID, sku_uuid, run_id, Json(rel), 0.0, Json({})))

    return sku_uuid


def _seed_run(cur, run_id, n_skus):
    cur.execute("""
        INSERT INTO stage8.runs (run_id, tenant_id, status, created_at)
        VALUES (%s,%s,'patterns_discovered',NOW())
        ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status
    """, (run_id, TENANT_ID))

    cur.execute("""
        INSERT INTO stage8.signal_context
            (tenant_id, run_id, pipeline_mode, data_mode, tenant_maturity,
             channel_split_applied, total_sku_count, median_history_days)
        VALUES (%s,%s,'single_channel','normal','established',FALSE,%s,120)
        ON CONFLICT (tenant_id, run_id) DO NOTHING
    """, (TENANT_ID, run_id, n_skus))


def _seed_clean_orders(cur, run_id, sku_uuids: list[str], forecast_date: date) -> None:
    """Insert one clean_orders row per SKU per day for the 7-day horizon window."""
    for sku_uuid in sku_uuids:
        for offset in range(7):
            order_day = forecast_date + timedelta(days=offset)
            cur.execute("""
                INSERT INTO stage8.clean_orders
                    (tenant_id, canonical_sku_id, order_date, quantity_sold)
                VALUES (%s, %s, %s, %s)
            """, (TENANT_ID, sku_uuid, order_day, 10.0))


def _drive_pipeline(conn, run_id):
    state = AgentState.IDLE
    try:
        state = transition(conn, TENANT_ID, run_id, state, AgentState.PRELOADING)
        preloading_handler(tenant_id=TENANT_ID, run_id=run_id, db=conn)
        state = transition(conn, TENANT_ID, run_id, state, AgentState.PERCEIVING)
        perceiving_handler(tenant_id=TENANT_ID, run_id=run_id, db=conn)
        state = transition(conn, TENANT_ID, run_id, state, AgentState.PLANNING)
        planning_handler(tenant_id=TENANT_ID, run_id=run_id, db=conn)
        state = transition(conn, TENANT_ID, run_id, state, AgentState.ACTING)
        acting_handler(tenant_id=TENANT_ID, run_id=run_id, db=conn)
        state = transition(conn, TENANT_ID, run_id, state, AgentState.LEARNING)
        learning_handler(tenant_id=TENANT_ID, run_id=run_id, db=conn)
        state = transition(conn, TENANT_ID, run_id, state, AgentState.REPORTING)
        reporting_handler(tenant_id=TENANT_ID, run_id=run_id, db=conn)
        transition(conn, TENANT_ID, run_id, state, AgentState.COMPLETE)
    except Exception:
        try:
            transition(conn, TENANT_ID, run_id, state, AgentState.FAILED)
        except Exception:
            pass
        raise


# ===========================================================================
# Integration fixture
# ===========================================================================

@pytest.fixture(scope="module")
def learning_state():
    conn = _connect()
    run_id = str(uuid.uuid4())

    _kill_idle_test_connections(conn)
    with conn.cursor() as cur:
        _truncate_all(cur)
    conn.commit()

    seed_tenant_params(TENANT_ID, "established", conn=conn)

    sku_uuids = []
    with conn.cursor() as cur:
        _seed_run(cur, run_id, len(SKUS))
        for sku in SKUS:
            sku_uuid = _seed_sku(cur, run_id, sku)
            sku_uuids.append(sku_uuid)
    conn.commit()

    _drive_pipeline(conn, run_id)

    # Backdate forecasts so the 7-day horizon is eligible:
    # DATE(created_at) <= CURRENT_DATE - 7  requires at least 8 days back.
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE stage9.forecasts
            SET created_at = NOW() - INTERVAL '9 days'
            WHERE tenant_id = %s AND run_id = %s
        """, (TENANT_ID, run_id))
    conn.commit()

    # The forecast_date after backdating will be CURRENT_DATE - 9.
    forecast_date = date.today() - timedelta(days=9)

    with conn.cursor() as cur:
        _seed_clean_orders(cur, run_id, sku_uuids, forecast_date)
    conn.commit()

    # Run OutcomeCollector.
    outcome_summary = run_for_tenant(TENANT_ID, conn)

    # Run ModelPerformanceAggregator with as_of = tomorrow so today's outcomes
    # (outcome_date = CURRENT_DATE) fall inside the [as_of-30, as_of) window.
    agg_stats = run_model_performance_aggregator(
        conn,
        tenant_id=TENANT_ID,
        as_of=date.today() + timedelta(days=1),
    )

    # Run LearningParamsUpdater.
    updater_result = LearningParamsUpdater().run(TENANT_ID, conn)

    conn.close()
    return {
        "run_id": run_id,
        "sku_uuids": sku_uuids,
        "outcome_summary": outcome_summary,
        "agg_stats": agg_stats,
        "updater_result": updater_result,
    }


# ===========================================================================
# Tests
# ===========================================================================

class TestOutcomeCollector:

    def test_outcomes_written(self, learning_state):
        s = learning_state["outcome_summary"]
        assert s["total_written"] > 0, \
            f"Expected outcomes to be written, got total_written={s['total_written']}"

    def test_7d_horizon_processed(self, learning_state):
        h7 = learning_state["outcome_summary"]["per_horizon"][7]
        assert h7["written"] > 0, \
            f"Expected 7d horizon outcomes, got {h7}"

    def test_no_errors(self, learning_state):
        s = learning_state["outcome_summary"]
        assert s["total_errors"] == 0, \
            f"OutcomeCollector reported errors: {s['total_errors']}"

    def test_idempotent_rerun(self, learning_state):
        conn = _connect()
        try:
            second_run = run_for_tenant(TENANT_ID, conn)
        finally:
            conn.close()
        assert second_run["total_written"] == 0, \
            f"Re-run should write nothing (ON CONFLICT DO NOTHING), got {second_run['total_written']}"

    def test_error_metrics_present_for_nonzero_actuals(self, learning_state):
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM stage9.forecast_outcomes
                    WHERE tenant_id = %s
                      AND actual_value > 0
                      AND error_mape IS NULL
                """, (TENANT_ID,))
                bad = cur.fetchone()[0]
        finally:
            conn.close()
        assert bad == 0, \
            f"{bad} outcomes have actual_value > 0 but NULL error_mape"


class TestModelPerformanceAggregator:

    def test_no_failure(self, learning_state):
        stats = learning_state["agg_stats"]
        assert stats.failure_reason is None, \
            f"Aggregator failed: {stats.failure_reason}"

    def test_rows_written(self, learning_state):
        stats = learning_state["agg_stats"]
        assert stats.rows_written >= 1, \
            f"Expected at least 1 model_performance_s9 row, got {stats.rows_written}"

    def test_model_performance_in_db(self, learning_state):
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM stage9.model_performance_s9
                    WHERE tenant_id = %s
                """, (TENANT_ID,))
                rows = cur.fetchone()[0]
        finally:
            conn.close()
        assert rows >= 1, f"Expected model_performance_s9 rows, found {rows}"


class TestLearningParamsUpdater:

    def test_status_ok(self, learning_state):
        result = learning_state["updater_result"]
        assert result["status"] == "ok", \
            f"LearningParamsUpdater returned status={result['status']!r}"

    def test_tenant_id_matches(self, learning_state):
        result = learning_state["updater_result"]
        assert result["tenant_id"] == TENANT_ID


class TestLearningLoopAudit:

    def test_print_audit_report(self, learning_state, capsys):
        s   = learning_state["outcome_summary"]
        agg = learning_state["agg_stats"]
        upd = learning_state["updater_result"]

        with capsys.disabled():
            print("\n")
            print("=" * 80)
            print("  STAGE 9 LEARNING LOOP AUDIT REPORT")
            print(f"  Tenant: {TENANT_ID}   SKUs: {len(SKUS)}")
            print("=" * 80)

            print("\n  [OutcomeCollector]")
            print(f"    total_written          : {s['total_written']}")
            print(f"    total_skipped_no_actuals: {s['total_skipped_no_actuals']}")
            print(f"    total_errors           : {s['total_errors']}")
            print(f"    duration_seconds       : {s['duration_seconds']:.3f}s")
            print()
            for h, hst in sorted(s["per_horizon"].items()):
                print(
                    f"    horizon={h:>3}d  "
                    f"pending={hst['pending']}  "
                    f"written={hst['written']}  "
                    f"skipped={hst['skipped_no_actuals']}  "
                    f"errors={hst['errors']}"
                )

            print("\n  [ModelPerformanceAggregator]")
            print(f"    rows_written    : {agg.rows_written}")
            print(f"    failure_reason  : {agg.failure_reason}")
            if agg.new_models:
                print(f"    new_models      : {agg.new_models}")
            if agg.discontinued_models:
                print(f"    discontinued    : {agg.discontinued_models}")

            print("\n  [LearningParamsUpdater]")
            print(f"    status          : {upd['status']}")

            print("\n" + "=" * 80)
