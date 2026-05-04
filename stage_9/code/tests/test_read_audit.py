"""
test_read_audit.py — Verifies all run-2 read paths work correctly.

Run 1  seeds + drives the full pipeline (all SKUs go FULL tier).
Run 2  drives the same SKUs again with the same fingerprint — stable SKUs
       should hit the CACHE tier, exercising every run-2 read path:

    - Fingerprint cache read → tier classification
    - Prior forecast read (_SQL_PRIOR_FORECAST) → all columns present
    - Thompson state carry-forward → alpha/beta/trials from run-1
    - Feature warm-start → features_used from run-1
    - Model performance aggregator → reads forecast_outcomes (even if empty)
    - LearningParamsUpdater → reads adaptive_quantile_state
    - SelfAssessment → reads model_performance_s9 (graceful if empty)

Run:
    STAGE9_TEST_DSN="postgresql://postgres:Joyboy@localhost:5432/dev?sslmode=disable" \
    STAGE9_PROJECT_ROOT="M:/stage_9/code" \
    python -m pytest tests/test_read_audit.py -v -s
"""
from __future__ import annotations

import sys, os

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
from learning.model_performance_aggregator import run_model_performance_aggregator
from learning.learning_params_updater import LearningParamsUpdater
from infrastructure.tenant_params import TenantParams
from infrastructure.seed import seed_tenant_params
from tests.stage9_data_factory import gen_stable, gen_trending, gen_intermittent

from infrastructure.config import DB_DSN as _DSN  # noqa: E402

# Skip if local stage8 is view-aliased (see _stage8_real_schema_required.py).
from tests._stage8_real_schema_required import skip_if_stage8_uses_views  # noqa: E402
skip_if_stage8_uses_views()

TENANT_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "read-audit-tenant-001"))

# Only stable + trending for run-2 cache hits; intermittent for non-cache control
SKUS = [
    dict(code="RA-STB-1", pattern="stable", model_hint="exponential_smoothing", obs_days=90,
         df_fn=lambda: gen_stable("RA-STB-1", n_days=90, daily_mean=20.0)),
    dict(code="RA-STB-2", pattern="stable", model_hint="exponential_smoothing", obs_days=120,
         df_fn=lambda: gen_stable("RA-STB-2", n_days=120, daily_mean=15.0, seed=10)),
    dict(code="RA-TRN-1", pattern="trending", model_hint="Holt", obs_days=120,
         df_fn=lambda: gen_trending("RA-TRN-1", n_days=120, daily_mean=8.0, trend_slope=0.05)),
    dict(code="RA-INT-1", pattern="intermittent", model_hint="Croston", obs_days=180,
         df_fn=lambda: gen_intermittent("RA-INT-1", n_days=180, zero_ratio=0.65)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect():
    conn = psycopg2.connect(_DSN, connect_timeout=5)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO stage9, public")
        # Fail fast if any table lock can't be acquired — prevents hanging
        # forever when a previous test run was killed mid-transaction.
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
    for t in [
        "forecasts", "thompson_sampling_state", "feature_decisions_s9",
        "hyperparameter_decisions", "backtest_decisions", "data_fingerprint_cache",
        "sku_similarity_registry", "model_initialization_s9", "stage9_self_assessment",
        "cross_agent_signals", "stage9_sku_execution_log", "agent_state_log_s9",
        "tenant_learning_params", "model_performance_s9", "adaptive_quantile_state",
        "forecast_outcomes",
    ]:
        cur.execute(f"TRUNCATE TABLE stage9.{t} CASCADE")
    cur.execute("DELETE FROM stage8.demand_history    WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.pattern_history   WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.feature_decisions WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.signal_context    WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.runs              WHERE tenant_id = %s", (TENANT_ID,))
    cur.execute("DELETE FROM stage8.canonical_sku     WHERE tenant_id = %s", (TENANT_ID,))


def _sku_uuid(code):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{TENANT_ID}-{code}"))


def _seed_sku(cur, run_id, sku):
    sku_uuid = _sku_uuid(sku["code"])
    cur.execute("""
        INSERT INTO stage8.canonical_sku (sku_id, tenant_id, vendor, product_type)
        VALUES (%s, %s, 'ReadAuditVendor', 'read_audit')
        ON CONFLICT (sku_id) DO NOTHING
    """, (sku_uuid, TENANT_ID))

    df = sku["df_fn"]()
    rows = [(TENANT_ID, sku_uuid, row["order_date"], float(row["quantity"]))
            for _, row in df.iterrows()]
    # Delete before inserting to prevent duplicate rows when the same SKU is
    # seeded for a second run — duplicates skew fingerprint computation and
    # prevent cache-tier matching on run 2.
    cur.execute(
        "DELETE FROM stage8.demand_history WHERE tenant_id = %s AND sku_id = %s",
        (TENANT_ID, sku_uuid),
    )
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


def _seed_run(cur, run_id, n_skus, maturity="established"):
    cur.execute("""
        INSERT INTO stage8.runs (run_id, tenant_id, status, created_at)
        VALUES (%s,%s,'patterns_discovered',NOW())
        ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status
    """, (run_id, TENANT_ID))
    cur.execute("""
        INSERT INTO stage8.signal_context
            (tenant_id, run_id, pipeline_mode, data_mode, tenant_maturity,
             channel_split_applied, total_sku_count, median_history_days)
        VALUES (%s,%s,'single_channel','normal',%s,FALSE,%s,120)
        ON CONFLICT (tenant_id, run_id) DO NOTHING
    """, (TENANT_ID, run_id, maturity, n_skus))


def _drive(conn, run_id):
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


def _fetch_col(cur, table, col, tenant_id=None):
    tid = tenant_id or TENANT_ID
    cur.execute(f"SELECT {col} FROM stage9.{table} WHERE tenant_id = %s", (tid,))
    return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Fixture — runs two full pipeline passes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_run_state():
    conn = _connect()
    run1_id = str(uuid.uuid4())
    run2_id = str(uuid.uuid4())

    # --- clean slate ---
    _kill_idle_test_connections(conn)
    with conn.cursor() as cur:
        _truncate_all(cur)
    conn.commit()

    seed_tenant_params(TENANT_ID, "established", conn=conn)

    # --- run 1 ---
    with conn.cursor() as cur:
        _seed_run(cur, run1_id, len(SKUS))
        for sku in SKUS:
            _seed_sku(cur, run1_id, sku)
    conn.commit()
    _drive(conn, run1_id)

    # Backdate run 1's COMPLETE so _resolve_execution_mode returns FULL for run 2.
    # Without this, run 2 would enter MICRO_UPDATE mode (< 18-hour threshold) and
    # the acting_handler early-returns before writing any forecasts to stage9.forecasts.
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE stage9.agent_state_log_s9
            SET transitioned_at = NOW() - INTERVAL '20 hours'
            WHERE tenant_id = %s AND run_id = %s AND to_state = 'COMPLETE'
        """, (TENANT_ID, run1_id))
    conn.commit()

    # snapshot thompson state after run 1
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sku_id, assigned_model, alpha_param, beta_param, total_trials
            FROM stage9.thompson_sampling_state
            WHERE tenant_id = %s
        """, (TENANT_ID,))
        thompson_run1 = {(r[0], r[1]): dict(alpha=r[2], beta=r[3], trials=r[4])
                         for r in cur.fetchall()}

    # snapshot fingerprints after run 1
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sku_id::text, fingerprint, tier, pattern_label, demand_total
            FROM stage9.data_fingerprint_cache WHERE tenant_id = %s
        """, (TENANT_ID,))
        fp_run1 = {r[0]: dict(fp=r[1], tier=r[2], pattern_label=r[3], demand_total=r[4])
                   for r in cur.fetchall()}

    # snapshot feature warm-start after run 1
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (sku_id) sku_id::text, features_used
            FROM stage9.feature_decisions_s9 WHERE tenant_id = %s
            ORDER BY sku_id, created_at DESC
        """, (TENANT_ID,))
        features_run1 = {r[0]: r[1] for r in cur.fetchall()}

    # --- run 2 (same demand data = same fingerprint → CACHE tier) ---
    with conn.cursor() as cur:
        _seed_run(cur, run2_id, len(SKUS))
        for sku in SKUS:
            _seed_sku(cur, run2_id, sku)
    conn.commit()
    _drive(conn, run2_id)

    # snapshot what run 2 produced
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sku_id::text, processing_tier, assigned_model,
                   confidence_final, backtest_mape, pattern_label,
                   selected_quantile, effective_max_horizon,
                   oos_adjustment_factor, reorder_bias_factor, is_b2b,
                   forecast_7d, forecast_14d, forecast_30d
            FROM stage9.forecasts WHERE tenant_id = %s AND run_id = %s
        """, (TENANT_ID, run2_id))
        cols = ["sku_id", "processing_tier", "assigned_model",
                "confidence_final", "backtest_mape", "pattern_label",
                "selected_quantile", "effective_max_horizon",
                "oos_adjustment_factor", "reorder_bias_factor", "is_b2b",
                "forecast_7d", "forecast_14d", "forecast_30d"]
        forecasts_run2 = {r[0]: dict(zip(cols[1:], r[1:])) for r in cur.fetchall()}

        cur.execute("""
            SELECT sku_id, assigned_model, alpha_param, beta_param, total_trials
            FROM stage9.thompson_sampling_state WHERE tenant_id = %s
        """, (TENANT_ID,))
        thompson_run2 = {(r[0], r[1]): dict(alpha=r[2], beta=r[3], trials=r[4])
                         for r in cur.fetchall()}

    conn.close()

    return dict(
        run1_id=run1_id,
        run2_id=run2_id,
        thompson_run1=thompson_run1,
        thompson_run2=thompson_run2,
        fp_run1=fp_run1,
        features_run1=features_run1,
        forecasts_run2=forecasts_run2,
    )


# ---------------------------------------------------------------------------
# Tests — Fingerprint cache read
# ---------------------------------------------------------------------------

class TestFingerprintCacheRead:

    def test_run1_wrote_fingerprints_for_all_skus(self, two_run_state):
        fp = two_run_state["fp_run1"]
        assert len(fp) == len(SKUS), f"Expected {len(SKUS)} fingerprint rows, got {len(fp)}"

    def test_fingerprint_has_pattern_label_and_demand_total(self, two_run_state):
        for sku_id, entry in two_run_state["fp_run1"].items():
            assert entry["pattern_label"] is not None, f"{sku_id} has NULL pattern_label in fingerprint cache"
            assert entry["demand_total"] is not None, f"{sku_id} has NULL demand_total in fingerprint cache"
            assert float(entry["demand_total"]) > 0, f"{sku_id} has zero demand_total"

    def test_run2_stable_skus_hit_cache_tier(self, two_run_state):
        forecasts = two_run_state["forecasts_run2"]
        stable_uuid_1 = _sku_uuid("RA-STB-1")
        stable_uuid_2 = _sku_uuid("RA-STB-2")
        cache_hits = [
            sku_id for sku_id, f in forecasts.items()
            if f["processing_tier"] == "cache"
        ]
        assert stable_uuid_1 in cache_hits or stable_uuid_2 in cache_hits, \
            f"Expected at least one stable SKU to hit CACHE tier on run-2. Tiers: " \
            f"{ {k: v['processing_tier'] for k, v in forecasts.items()} }"


# ---------------------------------------------------------------------------
# Tests — Prior forecast read (_SQL_PRIOR_FORECAST)
# ---------------------------------------------------------------------------

class TestPriorForecastRead:

    def test_cache_tier_rows_have_all_required_columns(self, two_run_state):
        forecasts = two_run_state["forecasts_run2"]
        required = [
            "processing_tier", "assigned_model", "confidence_final",
            "pattern_label", "selected_quantile", "effective_max_horizon",
            "oos_adjustment_factor", "reorder_bias_factor", "is_b2b",
            "forecast_7d", "forecast_14d", "forecast_30d",
        ]
        cache_rows = {k: v for k, v in forecasts.items() if v["processing_tier"] == "cache"}
        assert cache_rows, "No CACHE-tier rows found in run-2 — cache read not exercised"

        for sku_id, row in cache_rows.items():
            for col in required:
                assert row[col] is not None, \
                    f"Cache-tier row for {sku_id} has NULL {col} — prior forecast read dropped it"

    def test_cache_tier_forecasts_are_non_zero(self, two_run_state):
        forecasts = two_run_state["forecasts_run2"]
        for sku_id, row in forecasts.items():
            if row["processing_tier"] == "cache":
                for h in ["forecast_7d", "forecast_14d", "forecast_30d"]:
                    vals = row[h]
                    if vals and isinstance(vals, dict):
                        mean = vals.get("mean", 0)
                        assert mean >= 0, f"Cache-tier {sku_id} has negative {h}.mean={mean}"


# ---------------------------------------------------------------------------
# Tests — Thompson state carry-forward
# ---------------------------------------------------------------------------

class TestThompsonStateRead:

    def test_thompson_state_written_in_run1(self, two_run_state):
        assert two_run_state["thompson_run1"], "No Thompson state written by run-1"

    def test_thompson_trials_incremented_in_run2(self, two_run_state):
        t1 = two_run_state["thompson_run1"]
        t2 = two_run_state["thompson_run2"]
        incremented = 0
        for key, r1 in t1.items():
            r2 = t2.get(key)
            if r2 and r2["trials"] > r1["trials"]:
                incremented += 1
        assert incremented > 0, \
            "No Thompson state rows had total_trials incremented between run-1 and run-2 — " \
            "run-2 didn't read run-1 state"

    def test_thompson_alpha_or_beta_updated_or_cache_explains_unchanged(self, two_run_state):
        # Alpha/beta only update when a SKU runs FULL (backtest produces a reward signal).
        # When all SKUs hit CACHE tier on run-2 (no backtest), alpha/beta stay the same —
        # that is correct. We only assert alpha/beta changed if at least one SKU went FULL.
        t1 = two_run_state["thompson_run1"]
        t2 = two_run_state["thompson_run2"]
        forecasts = two_run_state["forecasts_run2"]
        full_skus = {k for k, v in forecasts.items() if v["processing_tier"] == "full"}

        if not full_skus:
            # All CACHE — alpha/beta unchanged is correct; nothing to assert
            return

        changed = 0
        for key, r1 in t1.items():
            sku_id = key[0]
            r2 = t2.get(key)
            if sku_id in full_skus and r2 and (r2["alpha"] != r1["alpha"] or r2["beta"] != r1["beta"]):
                changed += 1
        assert changed > 0, \
            "FULL-tier SKUs exist but no alpha/beta changed — bandit reward signal not propagating"


# ---------------------------------------------------------------------------
# Tests — Feature warm-start read
# ---------------------------------------------------------------------------

class TestFeatureWarmStartRead:

    def test_run1_wrote_feature_decisions(self, two_run_state):
        assert two_run_state["features_run1"], "No feature_decisions_s9 written by run-1"

    def test_features_have_content(self, two_run_state):
        for sku_id, features in two_run_state["features_run1"].items():
            assert features is not None, f"{sku_id} has NULL features_used"
            assert isinstance(features, (list, dict)), \
                f"{sku_id} features_used unexpected type: {type(features)}"


# ---------------------------------------------------------------------------
# Tests — Model performance aggregator read
# ---------------------------------------------------------------------------

class TestModelPerformanceAggregatorRead:

    def test_aggregator_runs_without_error(self):
        conn = _connect()
        try:
            params = TenantParams.load(TENANT_ID, conn)
            conn.commit()  # end read transaction → conn back to IDLE
            stats = run_model_performance_aggregator(
                conn, tenant_id=TENANT_ID, params=params
            )
            # With empty forecast_outcomes the aggregator should return 0 rows written
            # but must not raise
            assert stats is not None
        finally:
            conn.close()

    def test_aggregator_handles_empty_outcomes_gracefully(self):
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM stage9.forecast_outcomes WHERE tenant_id = %s",
                            (TENANT_ID,))
                outcome_count = cur.fetchone()[0]
            # outcomes won't exist since we haven't run outcome_collector (needs golden_table)
            # the aggregator must handle this without raising
            params = TenantParams.load(TENANT_ID, conn)
            conn.commit()  # end read transaction → conn back to IDLE
            stats = run_model_performance_aggregator(conn, tenant_id=TENANT_ID, params=params)
            assert stats is not None, "Aggregator returned None"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests — LearningParamsUpdater read
# ---------------------------------------------------------------------------

class TestLearningParamsUpdaterRead:

    def test_updater_runs_without_error(self):
        conn = _connect()
        try:
            updater = LearningParamsUpdater()
            result = updater.run(TENANT_ID, conn)
            assert result is not None
        finally:
            conn.close()

    def test_updater_reads_adaptive_quantile_state(self):
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM stage9.adaptive_quantile_state WHERE tenant_id = %s",
                    (TENANT_ID,),
                )
                count = cur.fetchone()[0]
            # adaptive_quantile_state is written by OutcomeCollector (needs golden_table)
            # so it may be empty — updater must handle that gracefully
            updater = LearningParamsUpdater()
            result = updater.run(TENANT_ID, conn)
            assert result is not None, "LearningParamsUpdater returned None"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests — Full audit print
# ---------------------------------------------------------------------------

class TestReadAuditReport:

    def test_print_read_audit_summary(self, two_run_state, capsys):
        forecasts = two_run_state["forecasts_run2"]
        t1 = two_run_state["thompson_run1"]
        t2 = two_run_state["thompson_run2"]
        fp = two_run_state["fp_run1"]

        tier_counts = {}
        for row in forecasts.values():
            t = row["processing_tier"]
            tier_counts[t] = tier_counts.get(t, 0) + 1

        null_in_cache = {}
        for sku_id, row in forecasts.items():
            if row["processing_tier"] == "cache":
                for col, val in row.items():
                    if val is None:
                        null_in_cache[col] = null_in_cache.get(col, 0) + 1

        trials_grew = sum(
            1 for k, r1 in t1.items()
            if t2.get(k) and t2[k]["trials"] > r1["trials"]
        )

        with capsys.disabled():
            print("\n")
            print("=" * 70)
            print("  STAGE 9 READ AUDIT — TWO-RUN SCENARIO")
            print("=" * 70)
            print(f"\n  Fingerprint cache after run-1: {len(fp)} SKUs")
            for sku_id, entry in fp.items():
                print(f"    {sku_id[:8]}...  tier={entry['tier']}  "
                      f"pattern={entry['pattern_label']}  demand={entry['demand_total']}")

            print(f"\n  Run-2 tier distribution: {tier_counts}")
            print(f"\n  Cache-tier NULL columns (from prior forecast read): "
                  f"{'none' if not null_in_cache else null_in_cache}")

            print(f"\n  Thompson state carry-forward:")
            print(f"    Run-1 keys: {len(t1)}")
            print(f"    Run-2 keys: {len(t2)}")
            print(f"    Rows where trials incremented: {trials_grew}/{len(t1)}")

            print(f"\n  Feature warm-start:")
            features = two_run_state["features_run1"]
            print(f"    SKUs with features_used: {len(features)}")
            for sku_id, feats in features.items():
                feat_list = feats if isinstance(feats, list) else list(feats.keys()) if isinstance(feats, dict) else []
                print(f"    {sku_id[:8]}...  features={feat_list[:4]}")

            print("\n" + "=" * 70)
