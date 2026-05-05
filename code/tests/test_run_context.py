"""
unit/handlers/test_run_context.py — RunContext registry coverage.

Module is small (one dataclass + 3 module-level helpers) but every handler
depends on it. These tests pin the contract:
    * store / fetch / remove round-trip
    * fetch raises KeyError on unknown run_id
    * default field shapes (lists vs dicts vs ints, etc.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[3]
for p in (str(_CODE), str(_CODE / "handlers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from handlers._context import RunContext, store, fetch, remove


# ---------------------------------------------------------------------------
# Default shape
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_required_fields_set_at_construction(self):
        ctx = RunContext(tenant_id="t1", run_id="r1")
        assert ctx.tenant_id == "t1"
        assert ctx.run_id == "r1"

    def test_default_execution_mode_is_full(self):
        assert RunContext(tenant_id="t", run_id="r").execution_mode == "full"

    def test_default_run_start_time_is_recent(self):
        import time
        before = time.time()
        ctx = RunContext(tenant_id="t", run_id="r")
        after = time.time()
        assert before <= ctx.run_start_time <= after + 0.01

    def test_default_collections_are_empty_and_independent(self):
        a = RunContext(tenant_id="t", run_id="r-a")
        b = RunContext(tenant_id="t", run_id="r-b")
        # Mutating one must not bleed into the other (default_factory test).
        a.sku_ids.append("x")
        a.calibrated_cache["k"] = "v"
        a.sku_results.append("r")
        a.demand_data["sku"] = "df"
        assert b.sku_ids == []
        assert b.calibrated_cache == {}
        assert b.sku_results == []
        assert b.demand_data == {}

    def test_default_pattern_feedback_failures_count_is_zero(self):
        assert RunContext(tenant_id="t", run_id="r").pattern_feedback_failures_count == 0

    def test_default_optional_fields_are_none(self):
        ctx = RunContext(tenant_id="t", run_id="r")
        for field in ("preloaded", "batch_writer", "signal_consumer", "params"):
            assert getattr(ctx, field) is None, f"{field} should default None"


# ---------------------------------------------------------------------------
# store / fetch / remove
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_store_then_fetch_returns_same_instance(self):
        ctx = RunContext(tenant_id="t", run_id="reg-1")
        store(ctx)
        try:
            assert fetch("reg-1") is ctx
        finally:
            remove("reg-1")

    def test_fetch_unknown_run_raises_keyerror(self):
        with pytest.raises(KeyError, match="No RunContext"):
            fetch("never-stored-xyz")

    def test_remove_clears_registry_entry(self):
        ctx = RunContext(tenant_id="t", run_id="reg-2")
        store(ctx)
        remove("reg-2")
        with pytest.raises(KeyError):
            fetch("reg-2")

    def test_remove_unknown_is_noop(self):
        """Idempotent — calling remove on a key that was never stored
        must not raise (used in finally blocks)."""
        remove("never-existed")  # no exception

    def test_store_overwrites_same_run_id(self):
        first  = RunContext(tenant_id="t", run_id="reg-3")
        second = RunContext(tenant_id="t", run_id="reg-3")
        store(first)
        store(second)
        try:
            assert fetch("reg-3") is second
        finally:
            remove("reg-3")

    def test_multiple_runs_isolated(self):
        a = RunContext(tenant_id="t", run_id="reg-A")
        b = RunContext(tenant_id="t", run_id="reg-B")
        store(a)
        store(b)
        try:
            a.sku_ids.append("only-A")
            b.sku_ids.append("only-B")
            assert fetch("reg-A").sku_ids == ["only-A"]
            assert fetch("reg-B").sku_ids == ["only-B"]
        finally:
            remove("reg-A")
            remove("reg-B")
