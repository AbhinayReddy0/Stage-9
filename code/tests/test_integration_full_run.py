"""
tests/test_integration_full_run.py

Integration tests — full IDLE → COMPLETE run against real handlers.

No handler mocks. Every handler (preloading, perceiving, planning, acting,
learning, reporting) runs its actual implementation. The only substitution
is the DB connection, replaced with InMemoryDB which:

  - Routes SELECT queries to seeded in-memory tables so real code gets
    realistic data back without a live Postgres.
  - Validates every INSERT column list against the declared table schema
    (schema drift guard — catches column renames before they reach prod).
  - Records every SQL statement for post-run assertions.

Test scenario: "empty tenant, first run."
  - tenant_learning_params seeded with all 58 production defaults.
  - No SKUs in pattern_history → no forecasts, no signals.
  - All 7 state transitions fire, RunContext is cleaned up,
    run.status ends as 'forecasted'.

This scenario exercises the full handler chain with real code paths and
real data structures. Adding a SKU scenario is straightforward: seed
InMemoryDB.seed_table("pattern_history", [...]) with one row.

Run with:
    python -m pytest tests/test_integration_full_run.py -v
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infrastructure.tenant_params_defaults import TENANT_LEARNING_PARAMS_DEFAULTS
from infrastructure.constants import RunStatus
from infrastructure.errors import RunAlreadyInProgressError
from handlers._context import _REGISTRY
from pipeline.orchestrator import run


# ===========================================================================
# Schema registry — INSERT column validation (schema drift guard)
# ===========================================================================

# Declare every column for every table Stage 9 writes to.
# Add a column here when the DB schema gains one.
# Remove a column here when one is dropped.
# An INSERT referencing a column absent from this dict → AssertionError.
_TABLE_SCHEMAS: dict[str, set[str]] = {
    "agent_state_log_s9": {
        "tenant_id", "run_id", "from_state", "to_state",
        "transitioned_at", "reason",
    },
    "stage8.runs": {
        "status", "updated_at", "tenant_id", "run_id",
    },
    "stage9.cross_agent_signals": {
        "signal_id", "tenant_id", "from_agent", "to_agent", "signal_type",
        "sku_id", "run_id", "payload", "confidence", "processed",
        "created_at", "expires_at",
    },
    "stage9.thompson_sampling_state": {
        "tenant_id", "sku_id", "assigned_model", "config_hash",
        "config_json", "alpha_param", "beta_param", "total_trials",
        "last_updated_at",
    },
    "stage9.sku_similarity_registry": {
        "tenant_id", "sku_id", "pattern_label", "best_model_config",
        "best_features", "avg_mape", "last_updated",
    },
    "stage9.data_fingerprint_cache": {
        "tenant_id", "sku_id", "fingerprint", "tier", "updated_at",
    },
    "stage9.stage9_self_assessment": {
        "tenant_id", "run_id",
        "avg_mape_this_run", "avg_mape_prev_run", "mape_delta_pct",
        "degradation_detected", "recommendations", "model_health_summary",
        "total_skus_processed", "cache_tier_count", "partial_tier_count",
        "full_tier_count", "fallback_count", "pattern_feedback_retry_count",
        "execution_mode", "run_duration_seconds",
    },
    "stage9.forecasts": {
        "tenant_id", "sku_id", "run_id", "assigned_model",
        "selected_quantile", "confidence_base", "confidence_final",
        "confidence_tier", "backtest_mape", "exception_flags",
        "status", "lifecycle_stage",
        "forecast_7d", "forecast_14d", "forecast_30d", "forecast_60d",
        "forecast_90d", "forecast_150d", "forecast_180d", "forecast_365d",
    },
    "stage9.pattern_feedback": {
        "tenant_id", "sku_id", "run_id", "pattern_label",
        "stage8_confidence", "mape", "wape", "bias",
        "model_used", "model_hint", "fallback_used",
        "classification_quality", "created_at",
    },
    "stage9.stage9_sku_execution_log": {
        "tenant_id", "sku_id", "run_id", "sub_stage", "reason", "created_at",
    },
}

# Regex to extract (table_name, column_list_str) from an INSERT statement.
_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+([\w.]+)\s*\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)


def _validate_insert(sql: str) -> None:
    """Raise AssertionError if INSERT references a column not in _TABLE_SCHEMAS."""
    match = _INSERT_RE.search(sql)
    if not match:
        return
    table = match.group(1).strip().lower()
    cols = {c.strip().lower() for c in match.group(2).split(",")}
    if table not in _TABLE_SCHEMAS:
        return   # table not yet declared in schema registry — skip silently
    unknown = cols - _TABLE_SCHEMAS[table]
    assert not unknown, (
        f"INSERT into {table!r} references unknown column(s): {sorted(unknown)}.\n"
        f"Known columns: {sorted(_TABLE_SCHEMAS[table])}\n"
        f"SQL: {sql[:300]}"
    )


# ===========================================================================
# InMemoryDB — real data for SELECT, validation for INSERT/UPDATE
# ===========================================================================

class _InMemoryCursor:
    """Cursor that routes queries to the owning InMemoryDB."""

    def __init__(self, db: "InMemoryDB") -> None:
        self._db = db
        self._rows: list = []
        self.description = None
        self.rowcount: int = -1

    def execute(self, sql: str, params=None) -> None:
        self._db.sql_log.append((sql, params))
        upper = sql.upper().strip()
        if upper.startswith("SELECT") or upper.startswith("WITH"):
            self._rows = list(self._db._route_select(sql, params))
        elif "INSERT INTO" in upper:
            _validate_insert(sql)
            self._rows = []
            self.rowcount = 1
        elif upper.startswith("UPDATE"):
            self._rows = []
            self.rowcount = 1
        else:
            self._rows = []

    def fetchall(self) -> list:
        rows, self._rows = self._rows, []
        return rows

    def fetchone(self):
        if self._rows:
            row, self._rows = self._rows[0], self._rows[1:]
            return row
        return None

    def close(self) -> None:
        pass

    def __enter__(self) -> "_InMemoryCursor":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class InMemoryDB:
    """
    Fake psycopg2 connection for integration tests.

    Call seed_table(name, rows) before the test run to pre-populate SELECT
    results for a given table pattern. tenant_learning_params is always
    pre-seeded with the 58 production defaults.
    """

    def __init__(self, tenant_id: str) -> None:
        self.sql_log: list[tuple[str, Any]] = []
        self.commit_count: int = 0
        self.rollback_count: int = 0
        # Each entry: {pattern: str, rows: list}
        # pattern is a substring matched against the SELECT SQL (case-insensitive).
        self._seeded: list[dict] = []
        # Auto-seed tenant_learning_params from production defaults.
        self.seed_table(
            "tenant_learning_params",
            [(name, val) for name, val in TENANT_LEARNING_PARAMS_DEFAULTS],
        )

    def seed_table(self, name_fragment: str, rows: list) -> None:
        """Seed SELECT results for any query whose SQL contains name_fragment."""
        self._seeded.append({"pattern": name_fragment.lower(), "rows": rows})

    def _route_select(self, sql: str, params) -> list:
        lower_sql = sql.lower()
        for entry in self._seeded:
            if entry["pattern"] in lower_sql:
                return list(entry["rows"])
        return []

    def cursor(self) -> _InMemoryCursor:
        return _InMemoryCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    # ---- helpers for assertions ------------------------------------------

    def sqls(self) -> list[str]:
        return [s for s, _ in self.sql_log]

    def inserts_for(self, table: str) -> list[str]:
        return [s for s in self.sqls() if "INSERT INTO" in s.upper() and table.lower() in s.lower()]

    def params_for(self, table: str) -> list:
        return [p for s, p in self.sql_log if "INSERT INTO" in s.upper() and table.lower() in s.lower()]

    def tables_written(self) -> set[str]:
        tables: set[str] = set()
        for sql in self.sqls():
            m = _INSERT_RE.search(sql)
            if m:
                tables.add(m.group(1).strip().lower())
            if sql.upper().strip().startswith("UPDATE"):
                words = sql.split()
                if len(words) >= 2:
                    tables.add(words[1].lower())
        return tables


def _clean_redis() -> MagicMock:
    r = MagicMock()
    r.set.return_value = True
    return r


def _run(tenant_id: str, run_id: str, db: InMemoryDB, redis=None) -> None:
    run(tenant_id, run_id, db, redis_client=redis or _clean_redis())


# ===========================================================================
# Test: 7-transition INSERT sequence
# ===========================================================================

class TestTransitionSequence(unittest.TestCase):
    TENANT = "acme-01"
    RUN    = "run-seq-001"

    EXPECTED = [
        ("IDLE",       "PRELOADING"),
        ("PRELOADING", "PERCEIVING"),
        ("PERCEIVING", "PLANNING"),
        ("PLANNING",   "ACTING"),
        ("ACTING",     "LEARNING"),
        ("LEARNING",   "REPORTING"),
        ("REPORTING",  "COMPLETE"),
    ]

    def test_seven_rows_written_in_order(self):
        db = InMemoryDB(self.TENANT)
        _run(self.TENANT, self.RUN, db)

        rows = [
            p for s, p in db.sql_log
            if isinstance(p, dict) and "from_state" in p
        ]
        self.assertEqual(len(rows), 7,
            f"Expected 7 state-log rows; got {len(rows)}.\n"
            f"Pairs: {[(r['from_state'], r['to_state']) for r in rows]}")

        for i, (exp_from, exp_to) in enumerate(self.EXPECTED):
            self.assertEqual(rows[i]["from_state"], exp_from)
            self.assertEqual(rows[i]["to_state"],   exp_to)

    def test_commit_at_least_once_per_transition(self):
        db = InMemoryDB(self.TENANT)
        _run(self.TENANT, "run-seq-002", db)
        self.assertGreaterEqual(db.commit_count, 7)


# ===========================================================================
# Test: real TenantParams — UnknownParamError surfaces correctly
# ===========================================================================

class TestTenantParams(unittest.TestCase):
    """Verify the run uses real TenantParams seeded from production defaults."""

    TENANT = "acme-params"
    RUN    = "run-params-001"

    def test_run_succeeds_with_full_param_set(self):
        """All 58 production defaults are present — no UnknownParamError."""
        db = InMemoryDB(self.TENANT)
        _run(self.TENANT, self.RUN, db)   # would raise if any param missing

    def test_run_fails_when_params_missing(self):
        """perceiving_handler raises TenantParamNotFoundError on empty params table."""
        from infrastructure.errors import TenantParamNotFoundError
        db = InMemoryDB(self.TENANT)
        # Override the tenant_learning_params seed with an empty table.
        db._seeded = [e for e in db._seeded if "tenant_learning_params" not in e["pattern"]]
        db.seed_table("tenant_learning_params", [])

        with self.assertRaises(TenantParamNotFoundError):
            _run(self.TENANT, "run-params-fail", db)


# ===========================================================================
# Test: Schema drift guard — INSERT columns validated against registry
# ===========================================================================

class TestSchemaContract(unittest.TestCase):
    """INSERT statements reference only declared columns."""

    TENANT = "acme-schema"
    RUN    = "run-schema-001"

    def setUp(self):
        self.db = InMemoryDB(self.TENANT)
        _run(self.TENANT, self.RUN, self.db)

    def test_state_log_table_written(self):
        self.assertIn("agent_state_log_s9", self.db.tables_written())

    def test_runs_table_written(self):
        self.assertIn("stage8.runs", self.db.tables_written())

    def test_self_assessment_written(self):
        self.assertIn("stage9.stage9_self_assessment", self.db.tables_written())

    def test_all_inserts_pass_schema_validation(self):
        """Re-validate every INSERT in the log — fails if schema dict is wrong."""
        for sql, _ in self.db.sql_log:
            if "INSERT INTO" in sql.upper():
                _validate_insert(sql)   # raises AssertionError on unknown columns


# ===========================================================================
# Test: run.status output
# ===========================================================================

class TestRunStatusOutput(unittest.TestCase):
    TENANT = "acme-status"

    def test_forecasted_on_empty_tenant(self):
        db = InMemoryDB(self.TENANT)
        _run(self.TENANT, "run-status-ok", db)

        runs_params = next(
            (p for s, p in db.sql_log if "stage8.runs" in s.lower() and p), None
        )
        self.assertIsNotNone(runs_params, "stage8.runs UPDATE not found")
        # params tuple is (status, updated_at/NOW(), tenant_id, run_id) — status at [0]
        actual = runs_params[0] if isinstance(runs_params, (list, tuple)) else None
        self.assertEqual(actual, RunStatus.FORECASTED)


# ===========================================================================
# Test: lock lifecycle
# ===========================================================================

class TestLockLifecycle(unittest.TestCase):
    TENANT = "acme-lock"

    def test_lock_acquired_and_released_on_success(self):
        redis = _clean_redis()
        db = InMemoryDB(self.TENANT)
        _run(self.TENANT, "run-lock-ok", db, redis=redis)
        redis.set.assert_called_once()
        redis.delete.assert_called_once()

    def test_lock_released_on_handler_failure(self):
        from infrastructure.errors import TenantParamNotFoundError
        redis = _clean_redis()
        db = InMemoryDB(self.TENANT)
        db._seeded = [e for e in db._seeded if "tenant_learning_params" not in e["pattern"]]
        db.seed_table("tenant_learning_params", [])

        with self.assertRaises(TenantParamNotFoundError):
            _run(self.TENANT, "run-lock-fail", db, redis=redis)

        redis.delete.assert_called_once()

    def test_second_run_blocked_while_lock_held(self):
        locked_redis = MagicMock()
        locked_redis.set.return_value = None
        with self.assertRaises(RunAlreadyInProgressError):
            _run(self.TENANT, "run-blocked", InMemoryDB(self.TENANT), redis=locked_redis)


# ===========================================================================
# Test: RunContext lifecycle
# ===========================================================================

class TestRunContextLifecycle(unittest.TestCase):
    TENANT = "acme-ctx"
    RUN    = "run-ctx-001"

    def test_context_removed_after_complete_run(self):
        db = InMemoryDB(self.TENANT)
        _run(self.TENANT, self.RUN, db)
        self.assertNotIn(self.RUN, _REGISTRY,
            "RunContext must be removed from the registry after COMPLETE")

    def test_params_set_before_acting(self):
        """Real perceiving_handler must populate ctx.params before planning runs."""
        seen: list = []

        original_planning = None

        import handlers.planning as _planning_mod
        original_planning = _planning_mod.planning_handler

        def _spy_planning(*, tenant_id, run_id, db):
            from handlers._context import fetch
            ctx = fetch(run_id)
            seen.append(ctx.params)
            original_planning(tenant_id=tenant_id, run_id=run_id, db=db)

        import unittest.mock as _mock
        with _mock.patch("pipeline.orchestrator.planning_handler", new=_spy_planning):
            db = InMemoryDB(self.TENANT)
            _run(self.TENANT, "run-ctx-spy", db)

        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0], "ctx.params must be set before planning_handler runs")
        # Verify it's a real TenantParams (not a stub) by checking it has 58 params.
        self.assertEqual(len(seen[0]), 58,
            f"Expected 58 params from production defaults, got {len(seen[0])}")


# ===========================================================================
# Test: run_one_sku pipeline (per-SKU end-to-end without dual_pool)
# ===========================================================================

class TestRunOneSku(unittest.TestCase):
    """
    Call run_one_sku directly — tests the full 9.2→9.3→9.4→9.5 pipeline
    for a single SKU without the dual_pool executor layer.
    """

    def _make_sku_input(self):
        from pipeline.dual_pool import SkuPipelineInput
        from handlers.acting import set_test_invariants
        from infrastructure.tenant_params import TenantParams
        from infrastructure.tenant_params_defaults import TENANT_LEARNING_PARAMS_DEFAULTS

        tenant_id = "acme-sku"
        params = TenantParams.from_dict(
            tenant_id,
            {name: str(val) for name, val in TENANT_LEARNING_PARAMS_DEFAULTS},
        )
        set_test_invariants(
            tenant_id=tenant_id,
            tenant_params=params.to_dict(),
            invariants={"feature_reliability": {}},
        )

        # Minimal preloaded_data contract — every REQUIRED_PRELOAD_KEY present.
        from handlers.acting import REQUIRED_PRELOAD_KEYS
        preloaded_data = {
            "demand_series":         [float(i % 20 + 1) for i in range(120)],
            "promo_weights":         {},
            "pattern_label":         "stable",
            "lifecycle_stage":       "mature",
            "assigned_model":        "SES",
            "selected_quantile":     0.80,
            "effective_max_horizon": 365,
            "learning_mode":         "standard",
            "oos_adjustment_factor": 1.0,
            "reorder_bias_factor":   1.0,
            "on_watchlist":          False,
            "pattern_confidence":    0.75,
            "thompson_state":        {},
            "calibrated_window_days": 60,
            "calibration_gap":       None,
            "tier":                  "full",
            "weekend_zero_ratio":    0.0,
            "criticality_tier":      None,
            "parent_style_id":       None,
            "tenant_params":         params.to_dict(),
        }
        # Fill any remaining required keys with safe defaults.
        for key in REQUIRED_PRELOAD_KEYS - preloaded_data.keys():
            preloaded_data[key] = None

        return SkuPipelineInput(
            sku_id="sku-test-001",
            assigned_model="SES",
            sku_data={},
            preloaded_data=preloaded_data,
        )

    def test_run_one_sku_returns_status_and_confidence(self):
        from handlers.acting import run_one_sku
        db = InMemoryDB("acme-sku")

        sku_input = self._make_sku_input()
        result = run_one_sku(sku_input, "acme-sku", "run-sku-001", db)

        self.assertIn("sku_id", result)
        self.assertIn("status", result)
        self.assertIn("confidence_final", result)
        self.assertIn("backtest_mape", result)
        self.assertEqual(result["sku_id"], "sku-test-001")
        self.assertIn(result["status"], {"forecasted", "needs_acknowledgment", "watchlist_review"})
        self.assertGreaterEqual(result["confidence_final"], 0.0)
        self.assertLessEqual(result["confidence_final"], 1.0)

    def test_run_one_sku_backtest_mape_is_real_float(self):
        from handlers.acting import run_one_sku
        db = InMemoryDB("acme-sku")

        sku_input = self._make_sku_input()
        result = run_one_sku(sku_input, "acme-sku", "run-sku-002", db)

        mape = result.get("backtest_mape")
        if mape is not None:
            self.assertIsInstance(mape, float)
            self.assertGreaterEqual(mape, 0.0)

    def test_run_one_sku_batch_rows_contains_forecasts(self):
        from handlers.acting import run_one_sku
        db = InMemoryDB("acme-sku")

        sku_input = self._make_sku_input()
        result = run_one_sku(sku_input, "acme-sku", "run-sku-003", db)

        batch_rows = result.get("batch_rows") or {}
        self.assertIn("forecasts", batch_rows,
            "run_one_sku must return a 'forecasts' entry in batch_rows")
        rows = batch_rows["forecasts"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for horizon in (7, 14, 30, 60, 90, 150, 180, 365):
            self.assertIn(f"forecast_{horizon}d", row,
                f"forecast_{horizon}d missing from forecasts row")


if __name__ == "__main__":
    unittest.main()
