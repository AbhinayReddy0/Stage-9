"""
test_rw_audit.py — Seeds one SKU per pattern, runs a full pipeline, then reads
back every stage9 table and reports null counts and row counts for every column.

Run:
    STAGE9_TEST_DSN="postgresql://postgres:Joyboy@localhost:5432/dev?sslmode=disable" \
    STAGE9_PROJECT_ROOT="M:/stage_9/code" \
    python -m pytest tests/test_rw_audit.py -v -s
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uuid
import pytest
import psycopg2
from psycopg2.extras import Json

from infrastructure.state_machine import AgentState, transition
from handlers.preloading import preloading_handler
from handlers.perceiving import perceiving_handler
from handlers.planning import planning_handler
from handlers.acting import acting_handler
from handlers.learning import learning_handler
from handlers.reporting import reporting_handler
from infrastructure.seed import seed_tenant_params
from tests.stage9_data_factory import (
    gen_cold_start, gen_stable, gen_trending, gen_seasonal, gen_intermittent,
)

from infrastructure.config import DB_DSN as _DSN  # noqa: E402

TENANT_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "audit-tenant-001"))

SKUS = [
    dict(code="AUDIT-CS", pattern="cold_start", model_hint="Naive", obs_days=25,
         df_fn=lambda: gen_cold_start("AUDIT-CS", n_days=25)),
    dict(code="AUDIT-STB", pattern="stable", model_hint="exponential_smoothing", obs_days=90,
         df_fn=lambda: gen_stable("AUDIT-STB", n_days=90, daily_mean=20.0)),
    dict(code="AUDIT-TRN", pattern="trending", model_hint="Holt", obs_days=120,
         df_fn=lambda: gen_trending("AUDIT-TRN", n_days=120, daily_mean=8.0, trend_slope=0.08)),
    dict(code="AUDIT-SEA", pattern="seasonal", model_hint="Prophet", obs_days=365,
         df_fn=lambda: gen_seasonal("AUDIT-SEA", n_days=365)),
    dict(code="AUDIT-INT", pattern="intermittent", model_hint="Croston", obs_days=180,
         df_fn=lambda: gen_intermittent("AUDIT-INT", n_days=180, zero_ratio=0.65)),
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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect():
    conn = psycopg2.connect(_DSN, connect_timeout=5)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO stage9, public")
        cur.execute("SET lock_timeout = '10s'")
    conn.commit()
    return conn


def _kill_idle_test_connections(conn) -> None:
    """Terminate all idle connections in the test DB to release stale locks."""
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
    # also clear stage8 seed data for our tenant
    cur.execute("DELETE FROM stage8.demand_history   WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.pattern_history  WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.feature_decisions WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.signal_context   WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.runs             WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.canonical_sku    WHERE tenant_id = %s", (TENANT_ID,))


def _seed_sku(cur, run_id, sku):
    sku_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{TENANT_ID}-{sku['code']}"))

    # canonical_sku
    cur.execute("""
        INSERT INTO stage8.canonical_sku (sku_id, tenant_id, vendor, product_type)
        VALUES (%s, %s, 'AuditVendor', 'audit_test')
        ON CONFLICT (sku_id) DO NOTHING
    """, (sku_uuid, TENANT_ID))

    # demand_history
    df = sku["df_fn"]()
    rows = [
        (TENANT_ID, sku_uuid, row["order_date"], float(row["quantity"]))
        for _, row in df.iterrows()
    ]
    cur.executemany(
        "INSERT INTO stage8.demand_history (tenant_id, sku_id, sale_date, qty) VALUES (%s,%s,%s,%s)",
        rows,
    )

    # pattern_history
    lifecycle = "introduction" if sku["obs_days"] < 28 else "saturation"
    cur.execute("""
        INSERT INTO stage8.pattern_history
            (tenant_id, sku_id, run_id, pattern_label, confidence_calibrated,
             model_hint, on_watchlist, observation_days, lifecycle_stage, drift_detected)
        VALUES (%s,%s,%s,%s,0.85,%s,FALSE,%s,%s,FALSE)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
    """, (TENANT_ID, sku_uuid, run_id, sku["pattern"], sku["model_hint"],
          sku["obs_days"], lifecycle))

    # feature_decisions
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


def _audit_table(cur, table: str) -> dict:
    cur.execute(f"SELECT COUNT(*) FROM stage9.{table} WHERE tenant_id = %s", (TENANT_ID,))
    row_count = cur.fetchone()[0]

    # get column names
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'stage9' AND table_name = %s
        ORDER BY ordinal_position
    """, (table,))
    cols = [r[0] for r in cur.fetchall()]

    null_counts = {}
    for col in cols:
        cur.execute(
            f"SELECT COUNT(*) FROM stage9.{table} WHERE tenant_id = %s AND {col} IS NULL",
            (TENANT_ID,),
        )
        null_counts[col] = cur.fetchone()[0]

    return {"rows": row_count, "nulls": null_counts, "cols": cols}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestReadWriteAudit:

    @pytest.fixture(scope="class")
    def audit_result(self):
        conn = _connect()
        run_id = str(uuid.uuid4())

        _kill_idle_test_connections(conn)
        with conn.cursor() as cur:
            _truncate_all(cur)
        conn.commit()

        seed_tenant_params(TENANT_ID, "established", conn=conn)

        with conn.cursor() as cur:
            _seed_run(cur, run_id, len(SKUS))
            for sku in SKUS:
                _seed_sku(cur, run_id, sku)
        conn.commit()

        _drive_pipeline(conn, run_id)

        results = {}
        with conn.cursor() as cur:
            for table in STAGE9_TABLES:
                try:
                    results[table] = _audit_table(cur, table)
                except Exception as e:
                    results[table] = {"error": str(e)}

        conn.close()
        return results

    def test_print_audit_report(self, audit_result, capsys):
        """Print the full read/write audit report."""
        with capsys.disabled():
            print("\n")
            print("=" * 80)
            print("  STAGE 9 READ/WRITE AUDIT REPORT")
            print(f"  Tenant: {TENANT_ID}   SKUs: {len(SKUS)}")
            print("=" * 80)

            for table, data in audit_result.items():
                if "error" in data:
                    print(f"\n  [ERROR] stage9.{table}: {data['error']}")
                    continue

                rows = data["rows"]
                nulls = data["nulls"]
                cols = data["cols"]

                null_cols = {c: n for c, n in nulls.items() if n > 0}
                ok_cols = [c for c in cols if nulls.get(c, 0) == 0]

                status = "OK" if not null_cols else "NULLS"
                print(f"\n  [{status}] stage9.{table}  ({rows} rows)")

                if null_cols:
                    print(f"    Columns with NULLs:")
                    for col, n in null_cols.items():
                        pct = int(100 * n / rows) if rows else 0
                        print(f"      {col:<35s}  {n}/{rows} NULL  ({pct}%)")

                print(f"    Fully populated: {', '.join(ok_cols)}")

            print("\n" + "=" * 80)

    def test_forecasts_written(self, audit_result):
        data = audit_result["forecasts"]
        assert "error" not in data, data.get("error")
        assert data["rows"] == len(SKUS), \
            f"Expected {len(SKUS)} forecast rows, got {data['rows']}"

    def test_forecasts_no_null_required_cols(self, audit_result):
        required = [
            "tenant_id", "sku_id", "run_id", "forecast_date",
            "assigned_model", "pattern_label", "selected_quantile",
            "confidence_base", "confidence_final", "status",
            "lifecycle_stage", "processing_tier",
            "forecast_7d", "forecast_14d", "forecast_30d",
        ]
        nulls = audit_result["forecasts"]["nulls"]
        rows = audit_result["forecasts"]["rows"]
        bad = {c: nulls[c] for c in required if nulls.get(c, 0) > 0}
        assert not bad, f"Required forecast columns have NULLs: {bad} (of {rows} rows)"

    def test_thompson_state_written(self, audit_result):
        data = audit_result["thompson_sampling_state"]
        assert "error" not in data
        assert data["rows"] >= len(SKUS)

    def test_fingerprint_cache_written(self, audit_result):
        data = audit_result["data_fingerprint_cache"]
        assert "error" not in data
        assert data["rows"] == len(SKUS)
        nulls = data["nulls"]
        assert nulls.get("fingerprint", 0) == 0, "fingerprint must not be NULL"
        assert nulls.get("tier", 0) == 0, "tier must not be NULL"
        assert nulls.get("pattern_label", 0) == 0, "pattern_label must not be NULL"
        assert nulls.get("demand_total", 0) == 0, "demand_total must not be NULL"

    def test_similarity_registry_written(self, audit_result):
        data = audit_result["sku_similarity_registry"]
        assert "error" not in data
        assert data["rows"] >= 1, "At least one SKU should be written to similarity registry"

    def test_feature_decisions_written(self, audit_result):
        data = audit_result["feature_decisions_s9"]
        assert "error" not in data
        assert data["rows"] >= 1

    def test_execution_log_present(self, audit_result):
        # execution log only written on errors/fallbacks — zero rows on a clean run is correct
        data = audit_result["stage9_sku_execution_log"]
        assert "error" not in data

    def test_self_assessment_written(self, audit_result):
        data = audit_result["stage9_self_assessment"]
        assert "error" not in data
        assert data["rows"] >= 1
