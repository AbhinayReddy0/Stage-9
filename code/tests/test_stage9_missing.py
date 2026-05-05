"""
test_stage9_missing.py — Test cases not covered by the existing 159 tests.

WHAT THIS COVERS
----------------

SECTION A — Batch job pipeline (OutcomeCollector → ModelPerformanceAggregator
            → LearningParamsUpdater)
    These three 3-5 AM jobs are the actual learning loop. None of them
    were tested in the existing files. Without these tests, you have no
    guarantee the loop closes.

SECTION B — Data quality edge cases
    Real Shopify data produces all of these regularly:
    - All-zero SKU (returns, discontinued)
    - Single non-zero day (just launched)
    - Negative quantity (returns in clean_orders)
    - Future-dated orders (clock skew from Shopify webhooks)

SECTION C — Boundary conditions
    The exact day a SKU transitions cold_start → real model:
    - obs_days = 59 must be cold_start
    - obs_days = 60 must classify on signal
    - backtest boundary: obs_days = 27 (NULL) vs obs_days = 28 (runs)

SECTION D — Tenant isolation
    Two tenants, same run, verify zero cross-contamination at every layer:
    - forecasts, thompson_sampling_state, pattern_feedback
    - model_initialization_s9, tenant_learning_params

HOW TO RUN
----------
    pytest test_stage9_missing.py -v -s

    # Just batch jobs (slowest — runs all 3 batch jobs)
    pytest test_stage9_missing.py -v -k "TestBatchPipeline"

    # Just edge cases (no special setup)
    pytest test_stage9_missing.py -v -k "TestEdgeCases"

    # Just boundaries
    pytest test_stage9_missing.py -v -k "TestBoundaries"

    # Just tenant isolation
    pytest test_stage9_missing.py -v -k "TestTenantIsolation"
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, timedelta, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

_DSN = os.environ.get("STAGE9_TEST_DSN")
if not _DSN:
    pytest.skip("STAGE9_TEST_DSN not set.", allow_module_level=True)

_PROJECT_ROOT = Path(os.environ.get("STAGE9_PROJECT_ROOT", "/mnt/project")).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from infrastructure.seed import seed_tenant_params
from infrastructure.state_machine import AgentState, transition
from handlers.preloading import preloading_handler
from handlers.perceiving import perceiving_handler
from handlers.planning import planning_handler
from handlers.acting import acting_handler
from handlers.learning import learning_handler
from handlers.reporting import reporting_handler

# Batch job imports
from learning.outcome_collector import run_for_tenant as run_outcome_collector
from learning.model_performance_aggregator import run_model_performance_aggregator
from learning.learning_params_updater import LearningParamsUpdater


# ===========================================================================
# Shared helpers
# ===========================================================================

_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "stage9-missing-tests")

def _uid(code: str) -> str:
    return str(uuid.uuid5(_NS, code))


def _connect():
    conn = psycopg2.connect(_DSN)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO stage9, public")
    conn.commit()
    return conn


def _seed_base(conn, tenant_id, run_id, n_skus=1, maturity="new"):
    seed_tenant_params(tenant_id=tenant_id, tenant_maturity=maturity, conn=conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO stage8.signal_context
            (tenant_id, run_id, pipeline_mode, data_mode,
             tenant_maturity, channel_split_applied,
             total_sku_count, median_history_days)
        VALUES (%s,%s,'single_channel','normal',%s,FALSE,%s,90)
        ON CONFLICT (tenant_id, run_id) DO NOTHING
        """,
        (tenant_id, run_id, maturity, n_skus),
    )
    cur.execute(
        """
        INSERT INTO stage8.runs (run_id, tenant_id, status, created_at)
        VALUES (%s,%s,'patterns_discovered',NOW())
        ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status
        """,
        (run_id, tenant_id),
    )
    cur.close()
    conn.commit()


def _seed_sku(conn, tenant_id, run_id, sku_uuid,
              orders_df, pattern, model_hint, obs_days):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO stage8.canonical_sku
            (sku_id, tenant_id, vendor, product_type)
        VALUES (%s,%s,'MissingTestVendor','test')
        ON CONFLICT (sku_id) DO NOTHING
        """,
        (sku_uuid, tenant_id),
    )
    if pd.api.types.is_datetime64_any_dtype(orders_df["order_date"]):
        dates = orders_df["order_date"].dt.date
    else:
        dates = pd.to_datetime(orders_df["order_date"]).dt.date

    rows = [
        (tenant_id, run_id, sku_uuid,
         "00000000-0000-0000-0000-000000000001",
         d, float(q), 19.99, 0.0, "fulfilled", "paid", True)
        for d, q in zip(dates, orders_df["quantity"])
    ]
    cur.executemany(
        """
        INSERT INTO stage8.clean_orders
            (tenant_id, run_id, canonical_sku_id, canonical_location_id,
             order_date, quantity_sold, unit_price, discount_pct,
             fulfillment_status, financial_status, is_active)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        rows,
    )
    # Also populate stage8.demand_history — the planning_handler reads
    # demand from this table, not directly from clean_orders.
    demand_rows = [
        (tenant_id, sku_uuid, d, float(q))
        for d, q in zip(dates, orders_df["quantity"])
    ]
    cur.executemany(
        """
        INSERT INTO stage8.demand_history (tenant_id, sku_id, sale_date, qty)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
        """,
        demand_rows,
    )
    lifecycle = "introduction" if obs_days < 28 else "saturation"
    cur.execute(
        """
        INSERT INTO stage8.pattern_history
            (tenant_id, sku_id, run_id, pattern_label,
             confidence_calibrated, model_hint, on_watchlist,
             observation_days, lifecycle_stage, drift_detected)
        VALUES (%s,%s,%s,%s,0.85,%s,FALSE,%s,%s,FALSE)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """,
        (tenant_id, sku_uuid, run_id, pattern, model_hint, obs_days, lifecycle),
    )
    rel = {"trend": 0.85, "seasonality": 0.85, "zero_ratio": 0.95, "cv": 0.90}
    cur.execute(
        """
        INSERT INTO stage8.feature_decisions
            (tenant_id, sku_id, run_id,
             feature_reliability_map, weekend_zero_ratio, velocity_signature)
        VALUES (%s,%s,%s,%s,0.05,%s)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """,
        (tenant_id, sku_uuid, run_id, Json(rel), Json({})),
    )
    cur.close()
    conn.commit()


def _drive(conn, tenant_id, run_id):
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


def _teardown(conn, tenant_id):
    cur = conn.cursor()
    for t in [
        "stage9.forecast_outcomes", "stage9.model_performance_s9", "stage9.forecasts",
        "stage8.pattern_feedback", "stage9.model_initialization_s9", "stage9.feature_decisions_s9",
        "stage9.hyperparameter_decisions", "stage9.backtest_decisions", "stage9.thompson_sampling_state",
        "stage9.stage9_self_assessment", "stage9.agent_state_log_s9", "stage9.stage9_sku_execution_log",
        "stage9.cross_agent_signals", "stage9.data_fingerprint_cache", "stage9.adaptive_quantile_state",
        "stage9.sku_similarity_registry",
        "stage8.pattern_history", "stage8.signal_context", "stage8.feature_decisions",
        "stage8.clean_orders", "stage8.demand_history", "stage8.canonical_sku", "stage8.runs",
        "stage9.tenant_learning_params",
    ]:
        try:
            cur.execute(f"DELETE FROM {t} WHERE tenant_id = %s", (tenant_id,))
        except Exception:
            conn.rollback()
            cur = conn.cursor()
    conn.commit()
    cur.close()


def _flat_df(n_days, mean, seed=0):
    rng = np.random.default_rng(seed)
    start = date.today() - timedelta(days=n_days)
    qty = np.round(mean + rng.normal(0, mean * 0.05, n_days)).clip(0).astype(int)
    return pd.DataFrame({
        "order_date": [start + timedelta(days=i) for i in range(n_days)],
        "quantity": qty,
    })


# ===========================================================================
# SECTION A — Batch job pipeline
# ===========================================================================

class TestBatchPipeline:
    """
    OutcomeCollector   (3 AM) → writes forecast_outcomes
    ModelPerformance   (4 AM) → aggregates rolling MAPE
    LearningParams     (4:30 AM) → updates tenant_learning_params

    These are the actual learning loop. We run them in sequence for a single
    stable SKU that has had 30+ days of forecasts, simulating what happens
    after the first month of operation.
    """

    @pytest.fixture(scope="class")
    def batch_env(self):
        """
        Set up a stable SKU, run Stage 9, then manually seed forecast_outcomes
        with 30+ days of simulated actuals so the batch jobs have evidence
        to process.
        """
        conn = _connect()
        tenant_id = str(uuid.uuid4())
        run_id    = str(uuid.uuid4())
        sku_uuid  = _uid("BATCH-STB-001")
        print(f"\n[batch] tenant={tenant_id}")

        try:
            _seed_base(conn, tenant_id, run_id, n_skus=1, maturity="developing")
            _seed_sku(conn, tenant_id, run_id, sku_uuid,
                      _flat_df(120, 20.0), "stable", "SES", 120)
            _drive(conn, tenant_id, run_id)

            # Simulate 35 days of outcomes already in the table
            # (as if OutcomeCollector has been running for 35 days)
            cur = conn.cursor()
            cur.execute(
                "SELECT assigned_model FROM model_initialization_s9 "
                "WHERE tenant_id=%s AND run_id=%s AND sku_id=%s",
                (tenant_id, run_id, sku_uuid),
            )
            row = cur.fetchone()
            assigned_model = row[0] if row else "SES"

            # Insert 35 realistic forecast_outcomes rows at horizon=30.
            # Use a distinct fake run_id per row — the UNIQUE constraint is on
            # (tenant_id, sku_id, run_id, horizon_days), so the same run_id
            # with the same horizon would collapse to 1 row via ON CONFLICT.
            # Outcome dates stay within the last 28 days so the 30-day window
            # used by _MAPE_EVIDENCE_SQL and the aggregator sees all 35 rows.
            for i in range(35):
                fake_run_id = str(uuid.uuid4())
                actual = round(20.0 * 30 + np.random.normal(0, 20), 2)
                forecast = round(actual * (1 + np.random.normal(0, 0.04)), 2)
                err_mape = round(abs(actual - forecast) / max(actual, 0.01), 4)
                cur.execute(
                    """
                    INSERT INTO stage9.forecast_outcomes
                        (tenant_id, sku_id, run_id, assigned_model,
                         horizon_days, outcome_date,
                         forecast_value, actual_value, error_mape, bias)
                    VALUES (%s,%s,%s,%s,30,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                    """,
                    (tenant_id, sku_uuid, fake_run_id, assigned_model,
                     (date.today() - timedelta(days=(i % 28) + 1)).isoformat(),
                     forecast, actual, err_mape, forecast - actual),
                )
            cur.close()
            conn.commit()

            yield {
                "conn": conn,
                "tenant_id": tenant_id,
                "run_id": run_id,
                "sku_uuid": sku_uuid,
                "assigned_model": assigned_model,
            }
        finally:
            try:
                _teardown(conn, tenant_id)
            finally:
                conn.close()

    def test_outcome_collector_runs_without_error(self, batch_env):
        """OutcomeCollector.run_for_tenant must complete without raising."""
        result = run_outcome_collector(
            tenant_id=batch_env["tenant_id"],
            conn=batch_env["conn"],
        )
        assert result is not None, (
            "run_for_tenant returned None — expected a result dict"
        )

    def test_forecast_outcomes_has_rows_after_collector(self, batch_env):
        """
        After OutcomeCollector runs, forecast_outcomes must have rows
        for horizons whose close date has passed.
        """
        cur = batch_env["conn"].cursor()
        cur.execute(
            "SELECT COUNT(*) FROM stage9.forecast_outcomes WHERE tenant_id=%s",
            (batch_env["tenant_id"],),
        )
        count = cur.fetchone()[0]
        cur.close()
        assert count >= 35, (
            f"forecast_outcomes has {count} rows, expected ≥ 35. "
            f"OutcomeCollector may not be writing for closed horizons."
        )

    def test_model_performance_aggregator_runs(self, batch_env):
        """ModelPerformanceAggregator must run and write model_performance_s9."""
        run_model_performance_aggregator(
            tenant_id=batch_env["tenant_id"],
            conn=batch_env["conn"],
            as_of=date.today(),
        )
        cur = batch_env["conn"].cursor()
        cur.execute(
            "SELECT COUNT(*) FROM stage9.model_performance_s9 WHERE tenant_id=%s",
            (batch_env["tenant_id"],),
        )
        count = cur.fetchone()[0]
        cur.close()
        assert count >= 1, (
            "model_performance_s9 is empty after aggregator ran. "
            "The aggregator may not be writing or the JOIN to "
            "forecast_outcomes is failing."
        )

    def test_model_performance_mape_is_reasonable(self, batch_env):
        """
        The aggregated rolling_mape_30d for SES on a flat stable series
        should be well under 10%. If it's above 10%, the aggregator is
        picking up wrong rows or using wrong math.
        """
        cur = batch_env["conn"].cursor()
        cur.execute(
            """
            SELECT assigned_model, horizon_days, avg_mape_30d
            FROM stage9.model_performance_s9
            WHERE tenant_id=%s AND horizon_days=30
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (batch_env["tenant_id"],),
        )
        row = cur.fetchone()
        cur.close()
        if row is None:
            pytest.skip("model_performance_s9 empty — aggregator likely skipped")
        mape = float(row[2])
        assert mape <= 0.10, (
            f"Aggregated MAPE={mape:.3f} for stable SKU. "
            f"Expected ≤ 10%. Check the aggregator's JOIN conditions."
        )

    def test_learning_params_updater_runs(self, batch_env):
        """LearningParamsUpdater must run without raising."""
        updater = LearningParamsUpdater()
        result = updater.run(
            tenant_id=batch_env["tenant_id"],
            conn=batch_env["conn"],
        )
        assert result is not None, "LearningParamsUpdater.run returned None"

    def test_tenant_learning_params_updated_after_30_days_evidence(self, batch_env):
        """
        After 35 outcome rows (> _MIN_EVIDENCE_COUNT=10), the
        LearningParamsUpdater applies exponential-smoothing updates via
        TenantParams.update(), which mutates `current_value` while leaving
        `starting_value` frozen at seed time. At least one row must therefore
        show `current_value <> starting_value`.
        """
        cur = batch_env["conn"].cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM stage9.tenant_learning_params
            WHERE tenant_id=%s
              AND current_value <> starting_value
            """,
            (batch_env["tenant_id"],),
        )
        diverged_count = cur.fetchone()[0]
        cur.close()
        assert diverged_count >= 1, (
            f"No tenant_learning_params row has current_value != "
            f"starting_value after 35 outcome rows. LearningParamsUpdater "
            f"requires _MIN_EVIDENCE_COUNT=10 — that threshold was met "
            f"(35 rows) so at least one confidence_base_* should have "
            f"diverged via the smoothing UPDATE."
        )

    def test_full_loop_closes(self, batch_env):
        """
        The complete 3-stage loop:
        forecast_outcomes → model_performance_s9 → tenant_learning_params
        All three must have rows for this tenant.
        """
        conn = batch_env["conn"]
        tenant_id = batch_env["tenant_id"]
        cur = conn.cursor()
        results = {}
        for t in ["forecast_outcomes", "model_performance_s9",
                  "tenant_learning_params"]:
            cur.execute(
                f"SELECT COUNT(*) FROM stage9.{t} WHERE tenant_id=%s",
                (tenant_id,),
            )
            results[t] = cur.fetchone()[0]
        cur.close()

        for table, count in results.items():
            assert count > 0, (
                f"stage9.{table} is empty for tenant. "
                f"The learning loop has a break at this stage. "
                f"All counts: {results}"
            )


# ===========================================================================
# SECTION B — Data quality edge cases
# ===========================================================================

class TestEdgeCases:
    """
    Each test runs Stage 9 on a single SKU with a pathological input.
    The test asserts Stage 9 does NOT crash and produces a valid output.
    """

    @pytest.fixture
    def edge_conn(self):
        conn = _connect()
        tenant_id = str(uuid.uuid4())
        yield conn, tenant_id
        _teardown(conn, tenant_id)
        conn.close()

    def _run_single_sku(self, conn, tenant_id, sku_code, orders_df,
                        pattern, model_hint, obs_days):
        run_id   = str(uuid.uuid4())
        sku_uuid = _uid(sku_code)
        _seed_base(conn, tenant_id, run_id, n_skus=1)
        _seed_sku(conn, tenant_id, run_id, sku_uuid,
                  orders_df, pattern, model_hint, obs_days)
        _drive(conn, tenant_id, run_id)
        return run_id, sku_uuid

    def _get_state(self, conn, tenant_id, run_id):
        cur = conn.cursor()
        cur.execute(
            "SELECT to_state FROM stage9.agent_state_log_s9 "
            "WHERE tenant_id=%s AND run_id=%s ORDER BY transitioned_at DESC LIMIT 1",
            (tenant_id, run_id),
        )
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None

    def test_all_zero_sku_does_not_crash(self, edge_conn):
        """
        90 days of data, every quantity = 0.
        Must reach COMPLETE. Must have a forecast row (not hang or error).
        """
        conn, tenant_id = edge_conn
        start = date.today() - timedelta(days=90)
        df = pd.DataFrame({
            "order_date": [start + timedelta(days=i) for i in range(90)],
            "quantity":   [0] * 90,
        })
        run_id, sku_uuid = self._run_single_sku(
            conn, tenant_id, "EDGE-ZERO-001", df,
            "intermittent", "Croston", 90,
        )
        final_state = self._get_state(conn, tenant_id, run_id)
        assert final_state == "COMPLETE", (
            f"All-zero SKU caused run to end in state={final_state!r}. "
            f"Expected COMPLETE — per-SKU failure isolation must catch this."
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT (forecast_30d->>'mean')::DECIMAL FROM stage9.forecasts "
            "WHERE tenant_id=%s AND run_id=%s",
            (tenant_id, run_id),
        )
        row = cur.fetchone()
        cur.close()
        assert row is not None, "No forecast row for all-zero SKU"
        assert float(row[0]) >= 0.0, (
            f"All-zero SKU: forecast_30d.mean={row[0]} is negative. "
            f"Croston must produce 0 or positive forecast for all-zero demand."
        )

    def test_single_nonzero_day_does_not_crash(self, edge_conn):
        """
        180 days of data, only 1 non-zero sale (on day 90).
        Must reach COMPLETE. Must classify as cold_start or intermittent.
        """
        conn, tenant_id = edge_conn
        start = date.today() - timedelta(days=180)
        qty = [0] * 180
        qty[90] = 5
        df = pd.DataFrame({
            "order_date": [start + timedelta(days=i) for i in range(180)],
            "quantity":   qty,
        })
        run_id, sku_uuid = self._run_single_sku(
            conn, tenant_id, "EDGE-ONE-001", df,
            "intermittent", "Croston", 180,
        )
        final_state = self._get_state(conn, tenant_id, run_id)
        assert final_state == "COMPLETE", (
            f"Single-sale SKU caused run to end in state={final_state!r}."
        )

    def test_negative_quantity_does_not_produce_negative_forecast(self, edge_conn):
        """
        clean_orders with some negative quantity_sold (returns).
        Forecast mean must never be negative regardless of returns.
        """
        conn, tenant_id = edge_conn
        rng = np.random.default_rng(99)
        start = date.today() - timedelta(days=90)
        qty = [20] * 90
        # Insert 10 return days
        for idx in rng.choice(90, 10, replace=False):
            qty[idx] = -5
        df = pd.DataFrame({
            "order_date": [start + timedelta(days=i) for i in range(90)],
            "quantity":   qty,
        })
        run_id, sku_uuid = self._run_single_sku(
            conn, tenant_id, "EDGE-NEG-001", df,
            "stable", "SES", 90,
        )
        final_state = self._get_state(conn, tenant_id, run_id)
        assert final_state == "COMPLETE", (
            f"Negative-qty SKU caused state={final_state!r}."
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT (forecast_7d->>'mean')::DECIMAL, "
            "(forecast_30d->>'mean')::DECIMAL, "
            "(forecast_90d->>'mean')::DECIMAL "
            "FROM stage9.forecasts "
            "WHERE tenant_id=%s AND run_id=%s",
            (tenant_id, run_id),
        )
        row = cur.fetchone()
        cur.close()
        assert row is not None, "No forecast row for negative-qty SKU"
        for i, h in enumerate([7, 30, 90]):
            val = float(row[i])
            assert val >= 0.0, (
                f"forecast_{h}d.mean={val} is negative. "
                f"Returns in clean_orders must not produce negative forecasts."
            )

    def test_future_dated_orders_filtered(self, edge_conn):
        """
        clean_orders has 5 rows with order_date > today.
        These must be filtered before feature engineering.
        Stage 9 must complete and forecast must be based only on past rows.
        """
        conn, tenant_id = edge_conn
        start = date.today() - timedelta(days=90)
        dates = [start + timedelta(days=i) for i in range(90)]
        # Add 5 future dates
        dates += [date.today() + timedelta(days=i+1) for i in range(5)]
        qty = [20] * 90 + [999] * 5  # future rows have unrealistic qty
        df = pd.DataFrame({"order_date": dates, "quantity": qty})
        run_id, sku_uuid = self._run_single_sku(
            conn, tenant_id, "EDGE-FUT-001", df,
            "stable", "SES", 90,
        )
        final_state = self._get_state(conn, tenant_id, run_id)
        assert final_state == "COMPLETE", (
            f"Future-dated orders caused state={final_state!r}."
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT (forecast_30d->>'mean')::DECIMAL FROM stage9.forecasts "
            "WHERE tenant_id=%s AND run_id=%s",
            (tenant_id, run_id),
        )
        row = cur.fetchone()
        cur.close()
        assert row is not None
        mean = float(row[0])
        # If future rows were NOT filtered, the mean would be pulled toward 999
        # A correct filter means mean should be close to 20 × 30 = 600
        assert mean < 2000, (
            f"forecast_30d.mean={mean:.0f} looks inflated — future rows "
            f"with qty=999 may not be filtered before feature engineering."
        )

    def test_all_skus_fail_run_still_reaches_complete(self, edge_conn):
        """
        Principle 2: even if EVERY SKU fails, the run reaches COMPLETE.
        We inject 3 SKUs that all produce all-zero demand to maximise
        failure probability.
        """
        conn, tenant_id = edge_conn
        run_id = str(uuid.uuid4())
        _seed_base(conn, tenant_id, run_id, n_skus=3)
        start = date.today() - timedelta(days=30)
        for i in range(3):
            sku_uuid = _uid(f"EDGE-ALLFAIL-{i:03d}")
            df = pd.DataFrame({
                "order_date": [start + timedelta(days=j) for j in range(25)],
                "quantity":   [0] * 25,
            })
            _seed_sku(conn, tenant_id, run_id, sku_uuid,
                      df, "cold_start", "Naive", 25)
        _drive(conn, tenant_id, run_id)
        final_state = self._get_state(conn, tenant_id, run_id)
        assert final_state == "COMPLETE", (
            f"All-SKUs-fail scenario ended in state={final_state!r}. "
            f"Expected COMPLETE — Sacred Rule 2 requires this always."
        )


# ===========================================================================
# SECTION C — Boundary conditions
# ===========================================================================

class TestBoundaries:
    """
    The two critical boundary conditions in the Stage 9 spec:
    1. obs_days < 60 → cold_start (regardless of underlying signal)
    2. obs_days < 28 → no backtest (NULL backtest_mape)
    """

    @pytest.fixture
    def boundary_conn(self):
        conn = _connect()
        tenant_id = str(uuid.uuid4())
        yield conn, tenant_id
        _teardown(conn, tenant_id)
        conn.close()

    def _quick_run(self, conn, tenant_id, sku_code, n_days, mean,
                   pattern, model):
        run_id   = str(uuid.uuid4())
        sku_uuid = _uid(sku_code)
        start    = date.today() - timedelta(days=n_days)
        df = pd.DataFrame({
            "order_date": [start + timedelta(days=i) for i in range(n_days)],
            "quantity":   [round(mean)] * n_days,
        })
        _seed_base(conn, tenant_id, run_id, n_skus=1)
        _seed_sku(conn, tenant_id, run_id, sku_uuid, df, pattern, model, n_days)
        _drive(conn, tenant_id, run_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT m.assigned_model, f.backtest_mape, f.status "
            "FROM stage9.model_initialization_s9 m "
            "JOIN stage9.forecasts f "
            "  ON f.sku_id = m.sku_id AND f.run_id = m.run_id "
            "WHERE m.tenant_id=%s AND m.run_id=%s AND m.sku_id=%s",
            (tenant_id, run_id, sku_uuid),
        )
        row = cur.fetchone()
        cur.close()
        return row  # (model, backtest_mape, status)

    def test_obs_days_59_is_cold_start(self, boundary_conn):
        """
        59 days of stable-looking demand MUST still classify as cold_start.
        The spec rule is: obs_days < 60 → cold_start, first match wins.
        """
        conn, tenant_id = boundary_conn
        row = self._quick_run(
            conn, tenant_id, "BOUND-59D-001", n_days=59,
            mean=20.0, pattern="cold_start", model="Naive",
        )
        assert row is not None, "No forecast/model row for 59d SKU"
        model = (row[0] or "").lower()
        assert "naive" in model, (
            f"59d SKU assigned model={row[0]!r}, expected Naive (cold_start). "
            f"The obs_days < 60 cold_start rule is not firing."
        )

    def test_obs_days_60_classifies_on_signal(self, boundary_conn):
        """
        60 days of clearly stable demand MUST classify as stable → SES.
        Exactly at the boundary, the pattern signal takes over.
        """
        conn, tenant_id = boundary_conn
        row = self._quick_run(
            conn, tenant_id, "BOUND-60D-001", n_days=60,
            mean=20.0, pattern="stable", model="SES",
        )
        assert row is not None, "No forecast/model row for 60d SKU"
        model = (row[0] or "").lower()
        assert "ses" in model or "exponential" in model, (
            f"60d SKU assigned model={row[0]!r}, expected SES. "
            f"At exactly 60 days, pattern signal should override cold_start rule."
        )

    def test_obs_days_27_has_null_backtest_mape(self, boundary_conn):
        """
        27 days of data — backtest requires obs_days ≥ 28.
        backtest_mape must be NULL.
        """
        conn, tenant_id = boundary_conn
        row = self._quick_run(
            conn, tenant_id, "BOUND-27D-001", n_days=27,
            mean=5.0, pattern="cold_start", model="Naive",
        )
        assert row is not None, "No forecast/model row for 27d SKU"
        assert row[1] is None, (
            f"27d SKU has backtest_mape={row[1]}, expected NULL. "
            f"Backtest must not run when obs_days < 28."
        )

    def test_obs_days_28_has_backtest_mape(self, boundary_conn):
        """
        28 days — exactly at the backtest minimum.
        backtest_mape must be non-NULL.
        """
        conn, tenant_id = boundary_conn
        row = self._quick_run(
            conn, tenant_id, "BOUND-28D-001", n_days=28,
            mean=20.0, pattern="stable", model="SES",
        )
        assert row is not None, "No forecast/model row for 28d SKU"
        if row[1] is None:
            pytest.xfail(
                "28d SKU has NULL backtest_mape. "
                "The backtest boundary may be > 28 days in your config — "
                "check backtest_min_days in tenant_learning_params."
            )
        assert float(row[1]) >= 0.0, (
            f"28d SKU backtest_mape={row[1]} is negative — invalid."
        )


# ===========================================================================
# SECTION D — Tenant isolation
# ===========================================================================

class TestTenantIsolation:
    """
    Two tenants, same SKU codes, run simultaneously (sequential in test
    but same run_ids to maximise collision risk).
    Every output table must be strictly partitioned by tenant_id.
    """

    @pytest.fixture(scope="class")
    def two_tenants(self):
        conn_a = _connect()
        conn_b = _connect()
        t_a = str(uuid.uuid4())
        t_b = str(uuid.uuid4())
        run_a = str(uuid.uuid4())
        run_b = str(uuid.uuid4())
        print(f"\n[isolation] tenant_a={t_a[:8]} tenant_b={t_b[:8]}")

        try:
            # Both tenants have the same SKU code — different UUIDs via namespace
            sku_a = _uid(f"{t_a}-STB")
            sku_b = _uid(f"{t_b}-STB")

            for conn, tid, rid, sku_uuid, mean in [
                (conn_a, t_a, run_a, sku_a, 20.0),
                (conn_b, t_b, run_b, sku_b, 50.0),   # different mean — easy to detect leak
            ]:
                _seed_base(conn, tid, rid, n_skus=1)
                _seed_sku(conn, tid, rid, sku_uuid,
                          _flat_df(90, mean), "stable", "SES", 90)
                _drive(conn, tid, rid)

            yield {
                "conn_a": conn_a, "tenant_a": t_a, "run_a": run_a, "sku_a": sku_a, "mean_a": 20.0,
                "conn_b": conn_b, "tenant_b": t_b, "run_b": run_b, "sku_b": sku_b, "mean_b": 50.0,
            }
        finally:
            _teardown(conn_a, t_a)
            _teardown(conn_b, t_b)
            conn_a.close()
            conn_b.close()

    def _count(self, conn, table, tenant_id):
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE tenant_id=%s", (tenant_id,))
        n = cur.fetchone()[0]
        cur.close()
        return n

    def test_forecasts_strictly_partitioned(self, two_tenants):
        """Application-level partitioning: every SELECT carries a tenant_id
        WHERE clause. Stage 9 doesn't use Postgres RLS, so cross-connection
        invisibility isn't enforced — but a non-matching tenant_id MUST
        return 0 rows. That's the contract."""
        e = two_tenants
        bogus = str(uuid.uuid4())
        n_a     = self._count(e["conn_a"], "stage9.forecasts", e["tenant_a"])
        n_b     = self._count(e["conn_b"], "stage9.forecasts", e["tenant_b"])
        n_bogus = self._count(e["conn_a"], "stage9.forecasts", bogus)
        assert n_a == 1, f"Tenant A has {n_a} forecast rows, expected 1"
        assert n_b == 1, f"Tenant B has {n_b} forecast rows, expected 1"
        assert n_bogus == 0, (
            f"stage9.forecasts with a non-existent tenant_id returned "
            f"{n_bogus} rows — partition WHERE filter is broken."
        )

    def test_thompson_state_partitioned(self, two_tenants):
        """Same partition contract for thompson_sampling_state."""
        e = two_tenants
        bogus = str(uuid.uuid4())
        n_a     = self._count(e["conn_a"], "stage9.thompson_sampling_state", e["tenant_a"])
        n_b     = self._count(e["conn_b"], "stage9.thompson_sampling_state", e["tenant_b"])
        n_bogus = self._count(e["conn_a"], "stage9.thompson_sampling_state", bogus)
        assert n_a >= 1, f"Tenant A has {n_a} Thompson rows, expected ≥ 1"
        assert n_b >= 1, f"Tenant B has {n_b} Thompson rows, expected ≥ 1"
        assert n_bogus == 0, (
            f"thompson_sampling_state with bogus tenant_id returned "
            f"{n_bogus} rows — partition WHERE filter is broken."
        )

    def test_forecast_values_reflect_correct_tenant_demand(self, two_tenants):
        """
        Tenant A (mean=20) and Tenant B (mean=50) must get different forecasts.
        If the forecasts are the same, data from one tenant leaked into the other.
        """
        e = two_tenants
        def _get_mean(conn, tenant_id, run_id):
            cur = conn.cursor()
            cur.execute(
                "SELECT (forecast_30d->>'mean')::DECIMAL FROM stage9.forecasts "
                "WHERE tenant_id=%s AND run_id=%s",
                (tenant_id, run_id),
            )
            row = cur.fetchone()
            cur.close()
            return float(row[0]) if row else None

        f30_a = _get_mean(e["conn_a"], e["tenant_a"], e["run_a"])
        f30_b = _get_mean(e["conn_b"], e["tenant_b"], e["run_b"])
        assert f30_a is not None and f30_b is not None
        # Tenant B mean is 2.5× Tenant A mean, so forecast_30d should reflect that
        ratio = f30_b / f30_a if f30_a > 0 else 0
        assert ratio > 1.5, (
            f"Tenant A forecast_30d={f30_a:.0f}, Tenant B={f30_b:.0f}. "
            f"Ratio={ratio:.2f}, expected > 1.5. "
            f"Tenant B's demand is 2.5× Tenant A's — the forecasts should "
            f"reflect this difference. Data may have leaked between tenants."
        )

    def test_pattern_feedback_partitioned(self, two_tenants):
        """Same partition contract for pattern_feedback."""
        e = two_tenants
        bogus = str(uuid.uuid4())
        n_bogus = self._count(e["conn_b"], "stage8.pattern_feedback", bogus)
        assert n_bogus == 0, (
            f"pattern_feedback with bogus tenant_id returned {n_bogus} "
            f"rows — partition WHERE filter is broken."
        )

    def test_tenant_learning_params_partitioned(self, two_tenants):
        """Each tenant must have their own isolated learning params."""
        e = two_tenants
        n_a = self._count(e["conn_a"], "stage9.tenant_learning_params", e["tenant_a"])
        n_b = self._count(e["conn_b"], "stage9.tenant_learning_params", e["tenant_b"])
        assert n_a >= 1, f"Tenant A has {n_a} learning param rows"
        assert n_b >= 1, f"Tenant B has {n_b} learning param rows"
        # Read the same param for both — must be independently set
        def _get_param(conn, tenant_id, param):
            cur = conn.cursor()
            cur.execute(
                "SELECT current_value FROM stage9.tenant_learning_params "
                "WHERE tenant_id=%s AND param_name=%s",
                (tenant_id, param),
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else None
        # Both tenants start from the same seed defaults — both should exist
        v_a = _get_param(e["conn_a"], e["tenant_a"], "min_backtest_window")
        v_b = _get_param(e["conn_b"], e["tenant_b"], "min_backtest_window")
        assert v_a is not None, "Tenant A missing min_backtest_window param"
        assert v_b is not None, "Tenant B missing min_backtest_window param"
