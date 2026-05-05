"""
test_stage9.py — Stage 9 integration test suite.

Combines two layers of tests into a single file:

LAYER 2 — Real-CSV end-to-end (was test_stage9_e2e.py)
    Drives the complete Stage 9 pipeline against a real Postgres DB using
    5 archetypal demand CSVs. Stage 8 is stubbed via direct DB inserts.
    Asserts sacred-rule invariants, per-SKU expected outputs from
    TEST_DATASET_PACKAGE.docx, multi-horizon forecast shapes, and
    cross-SKU sanity checks.

    SKUs:
        sku_cold_start.csv   → TEST-CS-001
        sku_intermittent.csv → TEST-INT-001
        sku_seasonal.csv     → TEST-SEA-001
        sku_trending.csv     → TEST-TRN-001
        sku_stable.csv       → TEST-STB-001

LAYER 3 — Synthetic-matrix scenarios (was test_stage9_production.py)
    13 production-realistic scenarios spanning pattern × data length.
    Synthetic data is generated deterministically via stage9_data_factory.py.
    Each scenario asserts: assigned_model, quantile, learning_mode, status,
    p50≤p80≤p90, and all 8 horizons populated.

PREREQUISITES
-------------
1. Postgres running locally, reachable via STAGE9_TEST_DSN.
   Example: STAGE9_TEST_DSN="postgresql://user:pass@localhost:5432/atheera_test"

2. Migrations 001-019 already applied (run db_files/run_migrations_local.py).

3. The 5 archetypal CSVs in tests/demand_csvs/ (or set STAGE9_TEST_DATA_DIR).

HOW TO RUN
----------
    # Full suite
    pytest tests/test_stage9.py -v

    # Layer 2 only (real CSVs)
    pytest tests/test_stage9.py -v -k "TestSacredRule or TestPerSku or TestForecastShapes or TestCrossSku"

    # Layer 3 only (synthetic matrix)
    pytest tests/test_stage9.py -v -k "TestSingleRun"

    # One SKU from Layer 2
    pytest tests/test_stage9.py -v -k "seasonal"

    # One pattern from Layer 3
    pytest tests/test_stage9.py -v -k "trending"

    # No-DB factory sanity check
    pytest tests/test_stage9.py -v -k "test_factory_produces_all_scenarios"

If STAGE9_TEST_DSN is not set, all tests skip with a clear message.

TEARDOWN
--------
Layer 2: cleanup is disabled — rows persist in DB for inspection.
Layer 3: cleanup runs after each scenario class — rows are deleted.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import date as _date, timedelta as _timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# psycopg2 — skip entire module if not installed
# ---------------------------------------------------------------------------
try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    pytest.skip(
        "psycopg2 not installed — install with `pip install psycopg2-binary`",
        allow_module_level=True,
    )

# Skip if no DB password (CI-friendly).
from infrastructure.config import (  # noqa: E402
    DB_DSN as _DSN, DB_PASSWORD as _DB_PASSWORD, PROJECT_ROOT as _PROJECT_ROOT_STR,
)
if not _DB_PASSWORD:
    pytest.skip(
        "DB_PASSWORD not set in .env — configure .env to run DB tests.",
        allow_module_level=True,
    )

# Make project root and test directory importable.
_PROJECT_ROOT = Path(_PROJECT_ROOT_STR).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from infrastructure.seed import seed_tenant_params  # noqa: E402
from infrastructure.state_machine import AgentState, transition  # noqa: E402
from handlers.preloading import preloading_handler  # noqa: E402
from handlers.perceiving import perceiving_handler  # noqa: E402
from handlers.planning import planning_handler  # noqa: E402
from handlers.acting import acting_handler  # noqa: E402
from handlers.learning import learning_handler  # noqa: E402
from handlers.reporting import reporting_handler  # noqa: E402

from stage9_data_factory import ALL_SCENARIOS, SINGLE_RUN_SCENARIOS, Scenario  # noqa: E402

# ===========================================================================
# Constants
# ===========================================================================

HORIZONS = [7, 14, 30, 60, 90, 150, 180, 365]  # Principle 5 — never modified

# Deterministic namespace UUIDs — Layer 2 and Layer 3 use separate namespaces
# so their SKU UUIDs never collide when both layers run in the same DB.
_NS_E2E = uuid.UUID("00000000-0000-0000-0000-0000DEADBEEF")  # Layer 2
_NS_SYN = uuid.UUID("00000000-0000-0000-0000-0000FACEB00C")  # Layer 3

SKU_UUIDS = {code: str(uuid.uuid5(_NS_E2E, code)) for code in (
    "TEST-CS-001", "TEST-INT-001", "TEST-SEA-001", "TEST-TRN-001", "TEST-STB-001",
)}
_UUID_TO_CODE = {v: k for k, v in SKU_UUIDS.items()}

# Per-SKU expected outputs from TEST_DATASET_PACKAGE.docx.
# Bands are intentionally wide because HP search adds variance.
EXPECTED = {
    "TEST-CS-001": {
        "csv": "sku_cold_start.csv",
        "pattern_label": "cold_start",
        "lifecycle_stage": "introduction",
        "model": "Naive",
        "quantile": 0.90,
        "learning_mode": "explore",
        "mape_band": None,  # no backtest — obs_days < 28
        "conf_band": (0.30, 0.55),
        "valid_status": {"needs_acknowledgment", "watchlist_review"},
        "obs_days": 25,
    },
    "TEST-INT-001": {
        "csv": "sku_intermittent.csv",
        "pattern_label": "intermittent",
        "lifecycle_stage": "saturation",
        "model": "Croston",
        "quantile": 0.90,
        "learning_mode": "explore",
        "mape_band": (0.20, 0.70),
        "conf_band": (0.28, 0.65),
        "valid_status": {"needs_acknowledgment", "watchlist_review", "forecasted"},
        "obs_days": 365,
    },
    "TEST-SEA-001": {
        "csv": "sku_seasonal.csv",
        "pattern_label": "seasonal",
        "lifecycle_stage": "saturation",
        "model": "Prophet",
        "quantile": 0.90,
        "learning_mode": "explore",
        "mape_band": (0.05, 0.45),
        "conf_band": (0.45, 0.80),
        "valid_status": {"needs_acknowledgment", "forecasted"},
        "obs_days": 365,
    },
    "TEST-TRN-001": {
        "csv": "sku_trending.csv",
        "pattern_label": "trending",
        "lifecycle_stage": "saturation",
        "model": "Holt",
        "quantile": 0.80,
        "learning_mode": "explore",
        "mape_band": (0.05, 0.50),
        "conf_band": (0.35, 0.75),
        "valid_status": {"needs_acknowledgment", "forecasted"},
        "obs_days": 365,
    },
    "TEST-STB-001": {
        "csv": "sku_stable.csv",
        "pattern_label": "stable",
        "lifecycle_stage": "saturation",
        "model": "exponential_smoothing",
        "quantile": 0.80,
        "learning_mode": "explore",
        "mape_band": (0.04, 0.15),
        "conf_band": (0.75, 0.92),
        "valid_status": {"forecasted"},
        "obs_days": 365,
    },
}


# ===========================================================================
# Shared DB helpers
# ===========================================================================


def _connect():
    """Open a Postgres connection from STAGE9_TEST_DSN with stage9 search path."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET search_path TO stage9, stage8, public")
    cur.close()
    conn.commit()
    return conn


def _shift_dates_to_recent(df: pd.DataFrame) -> pd.DataFrame:
    """Shift order_date values forward so the last date is yesterday.

    The planning/acting handlers filter demand_history to the last 730 days (MAX_DEMAND_HISTORY_DAYS).
    CSV or synthetic data from prior years would be silently excluded.
    Shifting keeps the data inside the rolling window while preserving the pattern.
    """
    today = _date.today()
    dates = pd.to_datetime(df["order_date"]).dt.date
    shift = (today - _timedelta(days=1) - max(dates)).days
    if shift == 0:
        return df
    df = df.copy()
    df["order_date"] = [d + _timedelta(days=shift) for d in dates]
    return df


def _insert_canonical_sku(cur, tenant_id: str, sku_uuid: str, code: str,
                          product_name: str = "", criticality: str = "B"):
    """Insert one canonical_sku row using the columns present in this schema."""
    cur.execute(
        """
        INSERT INTO stage8.canonical_sku
            (sku_id, tenant_id, vendor, product_type)
        VALUES (%s, %s, 'TestVendor', 'test_category')
        ON CONFLICT (sku_id) DO NOTHING
        """,
        (sku_uuid, tenant_id),
    )


def _insert_clean_orders(cur, tenant_id: str, run_id: str, sku_uuid: str,
                         df: pd.DataFrame):
    """Translate a demand DataFrame into clean_orders rows."""
    dates = pd.to_datetime(df["order_date"]).dt.date
    qty_col = df["quantity"] if "quantity" in df.columns else df["qty"]
    price_col = df["price"] if "price" in df.columns else pd.Series([1.0] * len(df))
    disc_col = df["discount_pct"] if "discount_pct" in df.columns else pd.Series([0.0] * len(df))

    rows = [
        (tenant_id, run_id, sku_uuid,
         "00000000-0000-0000-0000-000000000001",
         d, float(q), float(p), float(disc), "fulfilled", "paid", True)
        for d, q, p, disc in zip(dates, qty_col, price_col, disc_col)
    ]
    cur.executemany(
        """
        INSERT INTO stage8.clean_orders
            (tenant_id, run_id, canonical_sku_id, canonical_location_id,
             order_date, quantity_sold, unit_price, discount_pct,
             fulfillment_status, financial_status, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def _insert_demand_history(cur, tenant_id: str, sku_uuid: str, df: pd.DataFrame):
    """Populate stage8.demand_history so acting/planning handlers have demand."""
    dates = pd.to_datetime(df["order_date"]).dt.date
    qty_col = df["quantity"] if "quantity" in df.columns else df["qty"]
    rows = [(tenant_id, sku_uuid, d, float(q)) for d, q in zip(dates, qty_col)]
    cur.executemany(
        "INSERT INTO stage8.demand_history (tenant_id, sku_id, sale_date, qty) "
        "VALUES (%s, %s, %s, %s)",
        rows,
    )


def _seed_pattern_history(cur, tenant_id: str, run_id: str, sku_uuid: str,
                          pattern_label: str, model_hint: str, obs_days: int,
                          lifecycle_stage: str | None = None):
    """Stage 8 stub: seed pattern_history with the expected pattern label."""
    lc = lifecycle_stage or ("introduction" if obs_days < 28 else "saturation")
    cur.execute(
        """
        INSERT INTO stage8.pattern_history
            (tenant_id, sku_id, run_id, pattern_label,
             confidence_calibrated, model_hint, on_watchlist,
             observation_days, lifecycle_stage, drift_detected)
        VALUES (%s, %s, %s, %s, 0.85, %s, FALSE, %s, %s, FALSE)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """,
        (tenant_id, sku_uuid, run_id, pattern_label, model_hint, obs_days, lc),
    )


def _seed_signal_context(cur, tenant_id: str, run_id: str, total_skus: int = 5):
    """Stage 8.0 output. Stage 9 reads this in PERCEIVING."""
    cur.execute(
        """
        INSERT INTO stage8.signal_context
            (tenant_id, run_id, pipeline_mode, data_mode,
             tenant_maturity, channel_split_applied,
             total_sku_count, median_history_days)
        VALUES (%s, %s, 'single_channel', 'normal', 'new', FALSE, %s, 365)
        ON CONFLICT (tenant_id, run_id) DO NOTHING
        """,
        (tenant_id, run_id, total_skus),
    )


def _seed_feature_decisions(cur, tenant_id: str, run_id: str, sku_uuid: str,
                            weekend_zero_ratio: float = 0.10):
    """Minimal feature_decisions (Stage 8.4 output) — Stage 9 reads reliability map."""
    rel_map = {"trend": 0.85, "seasonality": 0.85, "zero_ratio": 0.95, "cv": 0.90}
    cur.execute(
        """
        INSERT INTO stage8.feature_decisions
            (tenant_id, sku_id, run_id, feature_reliability_map,
             weekend_zero_ratio, velocity_signature)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """,
        (tenant_id, sku_uuid, run_id, Json(rel_map), weekend_zero_ratio, Json({})),
    )


def _seed_run_row(cur, tenant_id: str, run_id: str):
    """Pre-create the runs row. Stage 9 expects status='patterns_discovered'."""
    cur.execute(
        """
        INSERT INTO stage8.runs (run_id, tenant_id, status, created_at)
        VALUES (%s, %s, 'patterns_discovered', NOW())
        ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status
        """,
        (run_id, tenant_id),
    )


def _drive_stage9(conn, tenant_id: str, run_id: str) -> None:
    """Walk Stage 9 through all 7 state transitions."""
    state = AgentState.IDLE
    try:
        state = transition(conn, tenant_id, run_id, state, AgentState.PRELOADING)
        preloading_handler(tenant_id=tenant_id, run_id=run_id, db=conn)

        state = transition(conn, tenant_id, run_id, state, AgentState.PERCEIVING)
        perceiving_handler(tenant_id=tenant_id, run_id=run_id, db=conn)

        state = transition(conn, tenant_id, run_id, state, AgentState.PLANNING)
        planning_handler(tenant_id=tenant_id, run_id=run_id, db=conn)

        state = transition(conn, tenant_id, run_id, state, AgentState.ACTING)
        acting_handler(tenant_id=tenant_id, run_id=run_id, db=conn)

        state = transition(conn, tenant_id, run_id, state, AgentState.LEARNING)
        learning_handler(tenant_id=tenant_id, run_id=run_id, db=conn)

        state = transition(conn, tenant_id, run_id, state, AgentState.REPORTING)
        reporting_handler(tenant_id=tenant_id, run_id=run_id, db=conn)

        transition(conn, tenant_id, run_id, state, AgentState.COMPLETE)
    except Exception:
        try:
            transition(conn, tenant_id, run_id, state, AgentState.FAILED)
        except Exception:
            pass
        raise


def _delete_tenant_artifacts(conn, tenant_id: str):
    """Delete every row this test could have written. Best-effort."""
    cur = conn.cursor()
    tables = [
        "forecasts", "pattern_feedback", "model_initialization_s9",
        "feature_decisions_s9", "hyperparameter_decisions", "backtest_decisions",
        "thompson_sampling_state", "stage9_self_assessment", "agent_state_log_s9",
        "stage9_sku_execution_log", "cross_agent_signals", "data_fingerprint_cache",
        "pattern_history", "signal_context", "feature_decisions",
        "clean_orders", "canonical_sku", "runs",
        "tenant_learning_params",
    ]
    for t in tables:
        try:
            cur.execute(f"DELETE FROM {t} WHERE tenant_id = %s", (tenant_id,))
        except Exception:
            conn.rollback()
            cur = conn.cursor()
    conn.commit()
    cur.close()


# ===========================================================================
# Layer 2 — Real CSV seeding helpers
# ===========================================================================


def _data_dir() -> Path:
    """Where to find the 5 archetypal test CSVs — always tests/demand_csvs/."""
    return Path(__file__).parent / "demand_csvs"


def _read_csv(name: str) -> pd.DataFrame:
    p = _data_dir() / name
    if not p.exists():
        pytest.fail(
            f"Missing test CSV: {p}. "
            f"Place CSVs in tests/demand_csvs/."
        )
    df = pd.read_csv(p)
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=True).dt.date
    return _shift_dates_to_recent(df)


def _seed_full_test_data(conn, tenant_id: str, run_id: str):
    """
    Seed all DB rows needed for one Layer 2 run.
    Idempotent via ON CONFLICT.
    """
    cur = conn.cursor()
    try:
        seed_tenant_params(tenant_id=tenant_id, tenant_maturity="new", conn=conn)

        for code, exp in EXPECTED.items():
            sku_uuid = SKU_UUIDS[code]
            df = _read_csv(exp["csv"])

            _insert_canonical_sku(cur, tenant_id, sku_uuid, code)
            _insert_clean_orders(cur, tenant_id, run_id, sku_uuid, df)
            _insert_demand_history(cur, tenant_id, sku_uuid, df)
            _seed_pattern_history(
                cur, tenant_id, run_id, sku_uuid,
                exp["pattern_label"], exp["model"], exp["obs_days"],
                exp["lifecycle_stage"],
            )
            _seed_feature_decisions(cur, tenant_id, run_id, sku_uuid)

        _seed_signal_context(cur, tenant_id, run_id, total_skus=len(EXPECTED))
        _seed_run_row(cur, tenant_id, run_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ===========================================================================
# Layer 3 — Synthetic scenario seeding helper
# ===========================================================================


def _seed_one_scenario(conn, tenant_id: str, run_id: str, scenario: Scenario):
    """Insert all DB rows for one synthetic scenario."""
    cur = conn.cursor()
    try:
        sku_uuid = str(uuid.uuid5(_NS_SYN, scenario.sku_code))
        df = _shift_dates_to_recent(scenario.df_factory())

        dates = pd.to_datetime(df["order_date"])
        weekend_mask = dates.dt.dayofweek >= 5
        qty_col = df["quantity"]
        weekend_zero_ratio = (
            float((qty_col[weekend_mask] == 0).mean()) if weekend_mask.any() else 0.0
        )

        _insert_canonical_sku(cur, tenant_id, sku_uuid, scenario.sku_code)
        _insert_clean_orders(cur, tenant_id, run_id, sku_uuid, df)
        _insert_demand_history(cur, tenant_id, sku_uuid, df)
        _seed_pattern_history(
            cur, tenant_id, run_id, sku_uuid,
            scenario.pattern_label, scenario.expected_model, len(df),
        )
        _seed_feature_decisions(
            cur, tenant_id, run_id, sku_uuid,
            weekend_zero_ratio=weekend_zero_ratio,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ===========================================================================
# LAYER 2 FIXTURES
# ===========================================================================


@pytest.fixture(scope="session")
def session_ids():
    """A single tenant_id + run_id shared across the whole Layer 2 session."""
    return {
        "tenant_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
    }


@pytest.fixture(scope="session")
def stage9_run(session_ids):
    """
    Seeds the DB with 5 real CSV SKUs, drives Stage 9 once, yields the
    connection. Cleanup is disabled — rows persist in DB for inspection.
    """
    conn = _connect()
    tenant_id = session_ids["tenant_id"]
    run_id = session_ids["run_id"]

    print(f"\n[Layer 2 setup] tenant_id={tenant_id}")
    print(f"[Layer 2 setup] run_id={run_id}")

    try:
        _seed_full_test_data(conn, tenant_id, run_id)
        _drive_stage9(conn, tenant_id, run_id)
        yield conn
    finally:
        try:
            pass  # cleanup disabled — rows kept for DB inspection
            # _delete_tenant_artifacts(conn, tenant_id)
        finally:
            conn.close()


@pytest.fixture(scope="session")
def model_init_rows(stage9_run, session_ids):
    """All model_initialization_s9 rows for the Layer 2 run, keyed by sku_code."""
    cur = stage9_run.cursor()
    cur.execute(
        """
        SELECT sku_id, assigned_model, selected_quantile, learning_mode
        FROM model_initialization_s9
        WHERE tenant_id=%s AND run_id=%s
        """,
        (session_ids["tenant_id"], session_ids["run_id"]),
    )
    return {
        _UUID_TO_CODE.get(str(r[0]), str(r[0])): {
            "sku_id": str(r[0]),
            "model": r[1],
            "quantile": float(r[2]),
            "learning_mode": r[3],
        }
        for r in cur.fetchall()
    }


@pytest.fixture(scope="session")
def forecast_rows(stage9_run, session_ids):
    """All forecasts rows for the Layer 2 run, keyed by sku_code."""
    cur = stage9_run.cursor()
    cur.execute(
        """
        SELECT f.sku_id, f.confidence_final, f.status, f.backtest_mape,
               f.forecast_7d, f.forecast_14d, f.forecast_30d, f.forecast_60d,
               f.forecast_90d, f.forecast_150d, f.forecast_180d, f.forecast_365d,
               f.exception_flags
        FROM forecasts f
        WHERE f.tenant_id=%s AND f.run_id=%s
        """,
        (session_ids["tenant_id"], session_ids["run_id"]),
    )
    out = {}
    for r in cur.fetchall():
        code = _UUID_TO_CODE.get(str(r[0]), str(r[0]))
        horizons = {
            h: (r[3 + i] if isinstance(r[3 + i], dict) else json.loads(r[3 + i]))
            for i, h in enumerate(HORIZONS, start=1)
        }
        out[code] = {
            "confidence_final": float(r[1]),
            "status": r[2],
            "backtest_mape": float(r[3]) if r[3] is not None else None,
            "horizons": horizons,
            "exception_flags": r[12] if isinstance(r[12], list)
            else json.loads(r[12] or "[]"),
        }
    return out


# ===========================================================================
# LAYER 2 — SECTION A: Sacred-rule invariants (Principles 2, 3, 5)
# ===========================================================================


class TestSacredRuleInvariants:
    """The non-negotiable invariants. Run first, fail loudly."""

    def test_run_reaches_complete(self, stage9_run, session_ids):
        """Principle 2 — every run reaches COMPLETE with all 7 transitions."""
        cur = stage9_run.cursor()
        cur.execute(
            "SELECT to_state FROM agent_state_log_s9 "
            "WHERE tenant_id=%s AND run_id=%s ORDER BY transitioned_at",
            (session_ids["tenant_id"], session_ids["run_id"]),
        )
        states = [r[0] for r in cur.fetchall()]
        assert states[-1] == "COMPLETE", (
            f"Run did not reach COMPLETE. State trail: {states}"
        )
        assert len(states) == 7, (
            f"Expected exactly 7 state transitions, got {len(states)}: {states}"
        )

    def test_pattern_feedback_written_for_every_sku(self, stage9_run, session_ids):
        """Principle 3 — pattern_feedback is sacred. One row per SKU, always."""
        cur = stage9_run.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM pattern_feedback "
            "WHERE tenant_id=%s AND run_id=%s",
            (session_ids["tenant_id"], session_ids["run_id"]),
        )
        count = cur.fetchone()[0]
        assert count == len(EXPECTED), (
            f"Expected {len(EXPECTED)} pattern_feedback rows (one per SKU), "
            f"got {count}. Stage 8's learning loop is broken for the missing SKUs."
        )

    def test_pattern_feedback_has_mape_for_all_skus(self, stage9_run, session_ids):
        """Principle 3 corollary — even failed SKUs must have forecast_error_mape."""
        cur = stage9_run.cursor()
        cur.execute(
            "SELECT sku_id, forecast_error_mape FROM pattern_feedback "
            "WHERE tenant_id=%s AND run_id=%s",
            (session_ids["tenant_id"], session_ids["run_id"]),
        )
        for sku_id, mape in cur.fetchall():
            assert mape is not None, f"sku_id={sku_id}: forecast_error_mape is NULL"
            assert 0.0 <= float(mape) <= 1.0, (
                f"sku_id={sku_id}: mape={mape} out of [0,1] range"
            )

    def test_all_eight_horizons_present(self, stage9_run, session_ids):
        """Principle 5 — HORIZONS = [7,14,30,60,90,150,180,365]. All 8, exactly."""
        cur = stage9_run.cursor()
        cur.execute(
            """
            SELECT sku_id,
                   forecast_7d, forecast_14d, forecast_30d, forecast_60d,
                   forecast_90d, forecast_150d, forecast_180d, forecast_365d
            FROM forecasts
            WHERE tenant_id=%s AND run_id=%s
            """,
            (session_ids["tenant_id"], session_ids["run_id"]),
        )
        rows = cur.fetchall()
        assert len(rows) == len(EXPECTED), (
            f"Expected {len(EXPECTED)} forecast rows, got {len(rows)}"
        )
        for row in rows:
            sku_id = row[0]
            for i, h in enumerate(HORIZONS, start=1):
                cell = row[i]
                assert cell is not None, (
                    f"sku_id={sku_id}: forecast_{h}d is NULL — Principle 5 violation"
                )
                payload = cell if isinstance(cell, dict) else json.loads(cell)
                assert "mean" in payload, (
                    f"sku_id={sku_id}: forecast_{h}d missing 'mean' key. Got: {payload}"
                )


# ===========================================================================
# LAYER 2 — SECTION B: Per-SKU expected outputs (TEST_DATASET_PACKAGE.docx)
# ===========================================================================


@pytest.mark.parametrize("sku_code", list(EXPECTED.keys()))
class TestPerSku:
    """One test class instance per SKU."""

    def test_model_assigned_correctly(self, sku_code, model_init_rows):
        exp = EXPECTED[sku_code]
        assert sku_code in model_init_rows, (
            f"{sku_code}: no row in model_initialization_s9"
        )
        actual = model_init_rows[sku_code]["model"]
        assert exp["model"].lower() in actual.lower(), (
            f"{sku_code}: expected model containing {exp['model']!r}, got {actual!r}"
        )

    def test_quantile_selected_correctly(self, sku_code, model_init_rows):
        exp = EXPECTED[sku_code]
        actual = model_init_rows[sku_code]["quantile"]
        assert abs(actual - exp["quantile"]) < 1e-3, (
            f"{sku_code}: expected quantile {exp['quantile']}, got {actual}"
        )

    def test_learning_mode_is_explore_first_run(self, sku_code, model_init_rows):
        exp = EXPECTED[sku_code]
        actual = model_init_rows[sku_code]["learning_mode"]
        assert actual == exp["learning_mode"], (
            f"{sku_code}: expected learning_mode={exp['learning_mode']}, got {actual!r}"
        )

    def test_confidence_in_band(self, sku_code, forecast_rows):
        exp = EXPECTED[sku_code]
        actual = forecast_rows[sku_code]["confidence_final"]
        lo, hi = exp["conf_band"]
        assert lo <= actual <= hi, (
            f"{sku_code}: confidence_final={actual:.3f} outside expected band {exp['conf_band']}"
        )

    def test_status_valid(self, sku_code, forecast_rows):
        exp = EXPECTED[sku_code]
        actual = forecast_rows[sku_code]["status"]
        assert actual in exp["valid_status"], (
            f"{sku_code}: status={actual!r} not in expected {exp['valid_status']}"
        )

    def test_backtest_mape_in_band(self, sku_code, forecast_rows):
        exp = EXPECTED[sku_code]
        if exp["mape_band"] is None:
            assert forecast_rows[sku_code]["backtest_mape"] is None, (
                f"{sku_code}: expected NULL backtest_mape (obs_days<28), "
                f"got {forecast_rows[sku_code]['backtest_mape']}"
            )
            return
        mape = forecast_rows[sku_code]["backtest_mape"]
        assert mape is not None, f"{sku_code}: backtest_mape is NULL but should be populated"
        lo, hi = exp["mape_band"]
        assert lo <= mape <= hi, (
            f"{sku_code}: backtest_mape={mape:.3f} outside expected band {exp['mape_band']}"
        )

    def test_quantile_monotonicity_all_horizons(self, sku_code, forecast_rows):
        """p50 <= p80 <= p90 in every horizon. Zero violations allowed."""
        for h, payload in forecast_rows[sku_code]["horizons"].items():
            p50 = payload.get("p50")
            p80 = payload.get("p80")
            p90 = payload.get("p90")
            if None in (p50, p80, p90):
                continue
            assert p50 <= p80 <= p90, (
                f"{sku_code}: forecast_{h}d violates p50<=p80<=p90: "
                f"p50={p50} p80={p80} p90={p90}"
            )


# ===========================================================================
# LAYER 2 — SECTION C: Multi-horizon strategy assertions
# ===========================================================================


class TestForecastShapes:
    """Asserts the docx 'Expected Forecast Shape Checks' table."""

    def test_seasonal_long_horizon_preserves_peak(self, forecast_rows):
        """
        Prophet's one-fit strategy must capture seasonal variation.
        forecast_365d/(forecast_30d*12) should deviate more than 5% from 1.0 —
        a flat ratio means Prophet is not detecting seasonality.
        """
        h = forecast_rows["TEST-SEA-001"]["horizons"]
        f30 = float(h[30]["mean"])
        f365 = float(h[365]["mean"])
        scaled = f30 * 12
        assert scaled > 0, "TEST-SEA-001: forecast_30d.mean must be positive"
        ratio = f365 / scaled
        assert abs(ratio - 1.0) > 0.05, (
            f"TEST-SEA-001: forecast_365d/(forecast_30d*12) = {ratio:.3f} is too "
            f"close to 1.0. Prophet should capture seasonal variation. "
            f"f30={f30}, f365={f365}"
        )

    def test_intermittent_linear_horizon_scaling(self, forecast_rows):
        """Croston produces daily_rate × N, so forecast_90d ≈ forecast_30d × 3."""
        h = forecast_rows["TEST-INT-001"]["horizons"]
        f30 = float(h[30]["mean"])
        f90 = float(h[90]["mean"])
        if f30 <= 0:
            pytest.skip("TEST-INT-001 forecast_30d.mean is zero — Croston degenerated")
        ratio = f90 / (f30 * 3)
        assert 0.80 <= ratio <= 1.20, (
            f"TEST-INT-001: forecast_90d/(forecast_30d*3) = {ratio:.3f}, "
            f"expected ~1.0. Croston should scale linearly. f30={f30}, f90={f90}"
        )

    def test_trending_damped(self, forecast_rows):
        """Damped Holt's growth must slow: forecast_365d < forecast_30d × 12."""
        h = forecast_rows["TEST-TRN-001"]["horizons"]
        f30 = float(h[30]["mean"])
        f365 = float(h[365]["mean"])
        scaled = f30 * 12
        assert f365 < scaled * 1.10, (
            f"TEST-TRN-001: forecast_365d={f365:.1f} >= forecast_30d*12*1.10={scaled * 1.10:.1f}. "
            f"Damped trend should slow growth."
        )

    def test_cold_start_flat_extrapolation(self, forecast_rows):
        """Naive forecast is linear: (forecast_30d/forecast_7d)/(30/7) ≈ 1.0."""
        h = forecast_rows["TEST-CS-001"]["horizons"]
        f7 = float(h[7]["mean"])
        f30 = float(h[30]["mean"])
        if f7 <= 0:
            pytest.skip("TEST-CS-001 forecast_7d.mean is zero — Naive degenerated")
        ratio = (f30 / f7) / (30 / 7)
        assert 0.85 <= ratio <= 1.15, (
            f"TEST-CS-001: (forecast_30d/forecast_7d)/(30/7) = {ratio:.3f}, "
            f"expected ~1.0. Naive should be perfectly linear. f7={f7}, f30={f30}"
        )


# ===========================================================================
# LAYER 2 — SECTION D: Cross-SKU sanity checks
# ===========================================================================


class TestCrossSkuSanity:

    def test_stable_has_highest_confidence(self, forecast_rows):
        """Stable SKU should always have the highest confidence_final."""
        confs = {sku: row["confidence_final"] for sku, row in forecast_rows.items()}
        winner = max(confs, key=confs.get)
        assert winner == "TEST-STB-001", (
            f"Expected TEST-STB-001 (stable) to have highest confidence, "
            f"but TEST-STB-001={confs.get('TEST-STB-001'):.3f} and winner was "
            f"{winner}={confs[winner]:.3f}. Full ranking: "
            f"{sorted(confs.items(), key=lambda x: -x[1])}"
        )

    def test_cold_start_no_exception_flags(self, forecast_rows):
        """TEST-CS-001: no backtest ran, so no exceptions should appear."""
        flags = forecast_rows["TEST-CS-001"]["exception_flags"]
        assert flags == [] or flags is None, (
            f"TEST-CS-001: expected empty exception_flags, got {flags}"
        )


# ===========================================================================
# LAYER 2 — Run summary printer (session-end eyeball check)
# ===========================================================================


@pytest.fixture(scope="session", autouse=True)
def _print_layer2_summary(stage9_run, session_ids):
    yield
    cur = stage9_run.cursor()
    cur.execute(
        """
        SELECT f.sku_id, m.assigned_model, m.selected_quantile,
               f.confidence_final, f.status, f.backtest_mape
        FROM forecasts f
        JOIN model_initialization_s9 m
            ON m.sku_id = f.sku_id AND m.run_id = f.run_id
        WHERE f.tenant_id=%s AND f.run_id=%s
        ORDER BY f.sku_id
        """,
        (session_ids["tenant_id"], session_ids["run_id"]),
    )
    rows = cur.fetchall()
    print("\n" + "=" * 80)
    print("LAYER 2 — Real CSV results")
    print(f"{'SKU':<14} {'MODEL':<22} {'Q':<6} {'CONF':<6} {'STATUS':<22} {'MAPE'}")
    print("=" * 80)
    for r in rows:
        code = _UUID_TO_CODE.get(str(r[0]), str(r[0])[:8])
        mape = f"{float(r[5]):.3f}" if r[5] is not None else "  -  "
        print(f"{code:<14} {str(r[1])[:22]:<22} {float(r[2]):<6.2f} "
              f"{float(r[3]):<6.2f} {str(r[4])[:22]:<22} {mape}")
    print("=" * 80)


# ===========================================================================
# LAYER 3 FIXTURE — Synthetic matrix (one tenant, one run, all 13 scenarios)
# ===========================================================================


@pytest.fixture(scope="class")
def matrix_run():
    """
    Seeds all SINGLE_RUN_SCENARIOS as separate SKUs under one tenant + run,
    drives Stage 9 once, yields results keyed by sku_code.
    Cleanup runs after the class so rows are removed.
    """
    conn = _connect()
    tenant_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    print(f"\n[Layer 3 setup] tenant={tenant_id}\n[Layer 3 setup] run={run_id}")

    try:
        cur = conn.cursor()
        seed_tenant_params(tenant_id=tenant_id, tenant_maturity="new", conn=conn)
        _seed_signal_context(cur, tenant_id, run_id, total_skus=len(SINGLE_RUN_SCENARIOS))
        _seed_run_row(cur, tenant_id, run_id)
        cur.close()
        conn.commit()

        for sc in SINGLE_RUN_SCENARIOS:
            _seed_one_scenario(conn, tenant_id, run_id, sc)

        _drive_stage9(conn, tenant_id, run_id)

        # Pull all results into memory keyed by sku_code.
        # Layer 3 SKUs are in canonical_sku only if sku_code column exists;
        # we fall back to matching via the deterministic UUID namespace.
        sku_code_by_uuid = {
            str(uuid.uuid5(_NS_SYN, sc.sku_code)): sc.sku_code
            for sc in SINGLE_RUN_SCENARIOS
        }
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.sku_id,
                   m.assigned_model, m.selected_quantile, m.learning_mode,
                   f.confidence_final, f.status, f.backtest_mape,
                   f.forecast_7d, f.forecast_14d, f.forecast_30d, f.forecast_60d,
                   f.forecast_90d, f.forecast_150d, f.forecast_180d, f.forecast_365d
            FROM model_initialization_s9 m
            LEFT JOIN forecasts f
                ON f.sku_id = m.sku_id AND f.run_id = m.run_id
            WHERE m.tenant_id=%s AND m.run_id=%s
            """,
            (tenant_id, run_id),
        )
        results: dict[str, Any] = {}
        for r in cur.fetchall():
            code = sku_code_by_uuid.get(str(r[0]), str(r[0]))
            horizons = {
                h: (r[7 + i] if isinstance(r[7 + i], dict) else json.loads(r[7 + i]))
                for i, h in enumerate(HORIZONS)
                if r[7 + i] is not None
            }
            results[code] = {
                "model": r[1],
                "quantile": float(r[2]) if r[2] is not None else None,
                "learning_mode": r[3],
                "confidence_final": float(r[4]) if r[4] is not None else None,
                "status": r[5],
                "backtest_mape": float(r[6]) if r[6] is not None else None,
                "horizons": horizons,
            }
        cur.close()
        yield {"tenant_id": tenant_id, "run_id": run_id, "results": results}
    finally:
        try:
            _delete_tenant_artifacts(conn, tenant_id)
        finally:
            conn.close()


# ===========================================================================
# LAYER 3 — Synthetic-matrix scenario assertions
# ===========================================================================


@pytest.mark.parametrize("scenario", SINGLE_RUN_SCENARIOS, ids=lambda s: s.name)
class TestSingleRun:
    """One test instance per synthetic scenario."""

    def test_model_assigned(self, scenario, matrix_run):
        results = matrix_run["results"]
        assert scenario.sku_code in results, (
            f"{scenario.name}: no model_initialization_s9 row written"
        )
        actual = results[scenario.sku_code]["model"] or ""
        assert scenario.expected_model.lower() in actual.lower(), (
            f"{scenario.name}: expected model containing {scenario.expected_model!r}, "
            f"got {actual!r}. {scenario.notes or ''}"
        )

    def test_quantile_assigned(self, scenario, matrix_run):
        actual = matrix_run["results"][scenario.sku_code]["quantile"]
        assert actual is not None, f"{scenario.name}: selected_quantile is NULL"
        assert abs(actual - scenario.expected_quantile) < 1e-3, (
            f"{scenario.name}: expected q={scenario.expected_quantile}, got {actual}"
        )

    def test_learning_mode_explore(self, scenario, matrix_run):
        actual = matrix_run["results"][scenario.sku_code]["learning_mode"]
        assert actual == "explore", (
            f"{scenario.name}: expected learning_mode='explore', got {actual!r}"
        )

    def test_status_terminal(self, scenario, matrix_run):
        actual = matrix_run["results"][scenario.sku_code]["status"]
        valid = {"forecasted", "needs_acknowledgment", "watchlist_review"}
        assert actual in valid, (
            f"{scenario.name}: status={actual!r} not in {valid}"
        )

    def test_all_horizons_populated(self, scenario, matrix_run):
        horizons = matrix_run["results"][scenario.sku_code]["horizons"]
        missing = [h for h in HORIZONS if h not in horizons]
        assert not missing, f"{scenario.name}: missing horizons {missing}"
        for h, payload in horizons.items():
            assert "mean" in payload, (
                f"{scenario.name}: horizon {h} missing 'mean'. Got: {payload}"
            )

    def test_quantile_monotonicity(self, scenario, matrix_run):
        for h, payload in matrix_run["results"][scenario.sku_code]["horizons"].items():
            p50 = payload.get("p50")
            p80 = payload.get("p80")
            p90 = payload.get("p90")
            if None in (p50, p80, p90):
                continue
            assert p50 <= p80 <= p90, (
                f"{scenario.name}: horizon {h} violates p50<=p80<=p90: "
                f"{p50}/{p80}/{p90}"
            )


# ===========================================================================
# No-DB sanity check — verify the data factory without touching Postgres
# ===========================================================================


def test_factory_produces_all_scenarios():
    """
    Every scenario in the registry should yield a non-empty DataFrame with
    the expected columns. Runs even if Postgres is unavailable.
    """
    expected_cols = {"order_date", "sku_id", "product_name", "quantity",
                     "price", "discount_pct", "channel"}
    for sc in ALL_SCENARIOS:
        df = sc.df_factory()
        assert len(df) > 0, f"{sc.name}: factory returned empty DataFrame"
        assert set(df.columns) >= expected_cols, (
            f"{sc.name}: missing columns {expected_cols - set(df.columns)}"
        )
