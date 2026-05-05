"""
test_stage8_contract.py — Stage 8 scaffold contract verification.

Asserts that every stage8.* table Stage 9 depends on:
  1. Exists in the database
  2. Has the critical columns with correct data types
  3. Enforces NOT NULL on columns Stage 9 relies on being non-null
  4. Accepts the exact writes Stage 9 performs (write contract)

These tests are structural — they require no seed data and run in < 1s.
They catch schema drift between the scaffold DDL and the SQL Stage 9 actually
executes (the class of bug where updated_at goes missing from stage8.runs).

Run:
    STAGE9_TEST_DSN="postgresql://postgres:Joyboy@localhost:5432/dev?sslmode=disable" \
    python -m pytest tests/test_stage8_contract.py -v
"""
from __future__ import annotations

import os
import sys
import uuid

import psycopg2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from infrastructure.config import DB_DSN as _DSN  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level fixture — load all schema info in one pass
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def schema_info():
    try:
        conn = psycopg2.connect(_DSN)
    except Exception as e:
        pytest.skip(f"DB unavailable: {e}")

    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name
          FROM information_schema.tables
         WHERE table_schema = 'stage8'
           AND table_type   = 'BASE TABLE'
    """)
    tables = {row[0] for row in cur.fetchall()}

    cur.execute("""
        SELECT table_name, column_name, data_type, is_nullable
          FROM information_schema.columns
         WHERE table_schema = 'stage8'
         ORDER BY table_name, ordinal_position
    """)
    columns: dict[str, dict[str, dict]] = {}
    for table, col, dtype, nullable in cur.fetchall():
        columns.setdefault(table, {})[col] = {
            "data_type":   dtype,
            "is_nullable": nullable,
        }

    cur.execute("""
        SELECT tc.table_name, kcu.column_name
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema    = kcu.table_schema
         WHERE tc.table_schema      = 'stage8'
           AND tc.constraint_type   = 'UNIQUE'
    """)
    unique_constraints: dict[str, set[str]] = {}
    for table, col in cur.fetchall():
        unique_constraints.setdefault(table, set()).add(col)

    cur.close()
    conn.close()
    return {"tables": tables, "columns": columns, "unique": unique_constraints}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(schema_info, table, column):
    return schema_info["columns"].get(table, {}).get(column)

def _assert_col(schema_info, table, column, expected_type):
    info = _col(schema_info, table, column)
    assert info is not None, f"stage8.{table}.{column} missing"
    assert info["data_type"] == expected_type, (
        f"stage8.{table}.{column}: expected {expected_type!r}, got {info['data_type']!r}"
    )

def _assert_not_null(schema_info, table, column):
    info = _col(schema_info, table, column)
    assert info is not None, f"stage8.{table}.{column} missing"
    assert info["is_nullable"] == "NO", (
        f"stage8.{table}.{column} should be NOT NULL"
    )


# ---------------------------------------------------------------------------
# 1. All tables exist
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "demand_history",
    "pattern_history",
    "feature_decisions",
    "signal_context",
    "oos_impact_estimates",
    "channel_demand_splits",
    "promo_decisions",
    "portfolio_intelligence_reports",
    "tenant_thresholds",
    "clean_orders",
    "runs",
    "pattern_feedback",
    "canonical_sku",
}

class TestStage8TablesExist:
    def test_all_tables_present(self, schema_info):
        missing = EXPECTED_TABLES - schema_info["tables"]
        assert not missing, f"Missing stage8 tables: {sorted(missing)}"

    def test_no_unexpected_gaps(self, schema_info):
        # Ensures the scaffold wasn't partially applied
        assert len(schema_info["tables"] & EXPECTED_TABLES) == len(EXPECTED_TABLES)


# ---------------------------------------------------------------------------
# 2. Column type contracts
# ---------------------------------------------------------------------------

class TestColumnTypes:
    def test_demand_history(self, schema_info):
        _assert_col(schema_info, "demand_history", "tenant_id",  "uuid")
        _assert_col(schema_info, "demand_history", "sku_id",     "uuid")
        _assert_col(schema_info, "demand_history", "sale_date",  "date")
        _assert_col(schema_info, "demand_history", "qty",        "numeric")

    def test_pattern_history(self, schema_info):
        _assert_col(schema_info, "pattern_history", "tenant_id",        "uuid")
        _assert_col(schema_info, "pattern_history", "sku_id",           "uuid")
        _assert_col(schema_info, "pattern_history", "pattern_label",    "character varying")
        _assert_col(schema_info, "pattern_history", "observation_days", "integer")
        _assert_col(schema_info, "pattern_history", "drift_detected",   "boolean")
        _assert_col(schema_info, "pattern_history", "weekend_zero_ratio", "numeric")

    def test_feature_decisions(self, schema_info):
        _assert_col(schema_info, "feature_decisions", "tenant_id",              "uuid")
        _assert_col(schema_info, "feature_decisions", "sku_id",                 "uuid")
        _assert_col(schema_info, "feature_decisions", "feature_reliability_map", "jsonb")

    def test_signal_context(self, schema_info):
        _assert_col(schema_info, "signal_context", "tenant_id",       "uuid")
        _assert_col(schema_info, "signal_context", "pipeline_mode",   "character varying")
        _assert_col(schema_info, "signal_context", "tenant_maturity", "character varying")
        _assert_col(schema_info, "signal_context", "on_watchlist",    "boolean")

    def test_oos_impact_estimates(self, schema_info):
        _assert_col(schema_info, "oos_impact_estimates", "tenant_id",            "uuid")
        _assert_col(schema_info, "oos_impact_estimates", "sku_id",               "uuid")
        _assert_col(schema_info, "oos_impact_estimates", "oos_pct",              "numeric")
        _assert_col(schema_info, "oos_impact_estimates", "detection_confidence", "numeric")
        _assert_col(schema_info, "oos_impact_estimates", "created_at",           "timestamp without time zone")

    def test_channel_demand_splits(self, schema_info):
        _assert_col(schema_info, "channel_demand_splits", "tenant_id", "uuid")
        _assert_col(schema_info, "channel_demand_splits", "sku_id",    "uuid")
        _assert_col(schema_info, "channel_demand_splits", "sale_date", "date")
        _assert_col(schema_info, "channel_demand_splits", "qty",       "numeric")
        _assert_col(schema_info, "channel_demand_splits", "channel",   "character varying")

    def test_promo_decisions(self, schema_info):
        _assert_col(schema_info, "promo_decisions", "tenant_id",  "uuid")
        _assert_col(schema_info, "promo_decisions", "sku_id",     "uuid")
        _assert_col(schema_info, "promo_decisions", "promo_date", "date")
        _assert_col(schema_info, "promo_decisions", "multiplier", "numeric")

    def test_portfolio_intelligence_reports(self, schema_info):
        _assert_col(schema_info, "portfolio_intelligence_reports", "tenant_id",  "uuid")
        _assert_col(schema_info, "portfolio_intelligence_reports", "alert_type", "character varying")
        _assert_col(schema_info, "portfolio_intelligence_reports", "payload",    "jsonb")
        _assert_col(schema_info, "portfolio_intelligence_reports", "created_at", "timestamp without time zone")

    def test_tenant_thresholds(self, schema_info):
        _assert_col(schema_info, "tenant_thresholds", "tenant_id",        "uuid")
        _assert_col(schema_info, "tenant_thresholds", "confidence_floor", "numeric")
        _assert_col(schema_info, "tenant_thresholds", "confidence_ceiling", "numeric")

    def test_clean_orders(self, schema_info):
        _assert_col(schema_info, "clean_orders", "tenant_id",        "uuid")
        _assert_col(schema_info, "clean_orders", "canonical_sku_id", "uuid")
        _assert_col(schema_info, "clean_orders", "order_date",       "date")
        _assert_col(schema_info, "clean_orders", "quantity_sold",    "numeric")

    def test_runs(self, schema_info):
        _assert_col(schema_info, "runs", "run_id",     "uuid")
        _assert_col(schema_info, "runs", "tenant_id",  "uuid")
        _assert_col(schema_info, "runs", "status",     "character varying")
        _assert_col(schema_info, "runs", "created_at", "timestamp without time zone")
        _assert_col(schema_info, "runs", "updated_at", "timestamp without time zone")

    def test_pattern_feedback(self, schema_info):
        _assert_col(schema_info, "pattern_feedback", "tenant_id",           "uuid")
        _assert_col(schema_info, "pattern_feedback", "sku_id",              "uuid")
        _assert_col(schema_info, "pattern_feedback", "run_id",              "uuid")
        _assert_col(schema_info, "pattern_feedback", "forecast_error_mape", "numeric")
        _assert_col(schema_info, "pattern_feedback", "model_used",          "character varying")
        _assert_col(schema_info, "pattern_feedback", "fallback_used",       "boolean")

    def test_canonical_sku(self, schema_info):
        _assert_col(schema_info, "canonical_sku", "sku_id",               "uuid")
        _assert_col(schema_info, "canonical_sku", "tenant_id",            "uuid")
        _assert_col(schema_info, "canonical_sku", "vendor",               "character varying")
        _assert_col(schema_info, "canonical_sku", "product_type",         "character varying")
        _assert_col(schema_info, "canonical_sku", "shelf_life_days",      "integer")
        _assert_col(schema_info, "canonical_sku", "planned_end_date",     "date")
        _assert_col(schema_info, "canonical_sku", "criticality_tier",     "character")
        _assert_col(schema_info, "canonical_sku", "parent_style_id",      "uuid")
        _assert_col(schema_info, "canonical_sku", "service_level_target", "numeric")
        _assert_col(schema_info, "canonical_sku", "seed_daily_demand",    "numeric")


# ---------------------------------------------------------------------------
# 3. NOT NULL constraints on columns Stage 9 requires non-null
# ---------------------------------------------------------------------------

class TestNotNullConstraints:
    def test_demand_history_required_cols(self, schema_info):
        for col in ("tenant_id", "sku_id", "sale_date", "qty"):
            _assert_not_null(schema_info, "demand_history", col)

    def test_pattern_history_required_cols(self, schema_info):
        for col in ("tenant_id", "sku_id", "pattern_label"):
            _assert_not_null(schema_info, "pattern_history", col)

    def test_clean_orders_required_cols(self, schema_info):
        for col in ("tenant_id", "canonical_sku_id", "order_date"):
            _assert_not_null(schema_info, "clean_orders", col)

    def test_runs_required_cols(self, schema_info):
        for col in ("run_id", "tenant_id", "status", "updated_at"):
            _assert_not_null(schema_info, "runs", col)

    def test_oos_impact_estimates_required_cols(self, schema_info):
        for col in ("tenant_id", "sku_id", "oos_pct", "detection_confidence"):
            _assert_not_null(schema_info, "oos_impact_estimates", col)

    def test_canonical_sku_required_cols(self, schema_info):
        for col in ("sku_id", "tenant_id"):
            _assert_not_null(schema_info, "canonical_sku", col)


# ---------------------------------------------------------------------------
# 4. Write contract — Stage 9's actual writes against stage8.*
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def write_conn():
    try:
        conn = psycopg2.connect(_DSN)
    except Exception as e:
        pytest.skip(f"DB unavailable: {e}")
    yield conn
    conn.rollback()
    conn.close()


class TestWriteContract:
    """
    Verify that the exact SQL Stage 9 executes against stage8.* succeeds.
    All writes are rolled back — no persistent data changes.
    """

    def test_runs_status_update(self, write_conn):
        run_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        cur = write_conn.cursor()
        cur.execute(
            "INSERT INTO stage8.runs (run_id, tenant_id, status) VALUES (%s, %s, 'pending')",
            (run_id, tenant_id),
        )
        cur.execute(
            "UPDATE stage8.runs SET status = %s, updated_at = NOW() WHERE tenant_id = %s AND run_id = %s",
            ("forecasted", tenant_id, run_id),
        )
        cur.execute("SELECT status FROM stage8.runs WHERE run_id = %s", (run_id,))
        assert cur.fetchone()[0] == "forecasted"
        write_conn.rollback()

    def test_pattern_feedback_upsert(self, write_conn):
        run_id    = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        sku_id    = str(uuid.uuid4())
        cur = write_conn.cursor()
        cur.execute("""
            INSERT INTO stage8.pattern_feedback
                (tenant_id, sku_id, run_id, pattern_label, model_used, fallback_used)
            VALUES (%s, %s, %s, 'stable', 'simple_exponential_smoothing', FALSE)
            ON CONFLICT (tenant_id, sku_id, run_id) DO UPDATE
                SET pattern_label = EXCLUDED.pattern_label
        """, (tenant_id, sku_id, run_id))
        cur.execute(
            "SELECT pattern_label FROM stage8.pattern_feedback WHERE run_id = %s", (run_id,)
        )
        assert cur.fetchone()[0] == "stable"
        write_conn.rollback()

    def test_clean_orders_insert(self, write_conn):
        tenant_id = str(uuid.uuid4())
        sku_id    = str(uuid.uuid4())
        cur = write_conn.cursor()
        cur.execute("""
            INSERT INTO stage8.clean_orders (tenant_id, canonical_sku_id, order_date, quantity_sold)
            VALUES (%s, %s, CURRENT_DATE, 42.0)
        """, (tenant_id, sku_id))
        cur.execute(
            "SELECT quantity_sold FROM stage8.clean_orders WHERE tenant_id = %s AND canonical_sku_id = %s",
            (tenant_id, sku_id),
        )
        assert float(cur.fetchone()[0]) == 42.0
        write_conn.rollback()

    def test_pattern_feedback_unique_constraint(self, write_conn):
        run_id    = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        sku_id    = str(uuid.uuid4())
        cur = write_conn.cursor()
        cur.execute("""
            INSERT INTO stage8.pattern_feedback (tenant_id, sku_id, run_id, fallback_used)
            VALUES (%s, %s, %s, FALSE)
        """, (tenant_id, sku_id, run_id))
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute("""
                INSERT INTO stage8.pattern_feedback (tenant_id, sku_id, run_id, fallback_used)
                VALUES (%s, %s, %s, FALSE)
            """, (tenant_id, sku_id, run_id))
        write_conn.rollback()
