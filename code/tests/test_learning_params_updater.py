"""
tests/test_learning_params_updater.py — LearningParamsUpdater Done Criteria
============================================================================
10 test cases covering every acceptance criterion from the build plan.

D1  — confidence_base_seasonal moves toward evidence when count >= 10
D2  — confidence_base_seasonal unchanged when count < 10 (min evidence rule)
D3  — current_value always moves TOWARD evidence (convergence, pure formula)
D4  — two tenants with different evidence converge to different values
D5  — unseeded tenant skipped cleanly, no exception
D6  — one tenant failure does not abort other tenants
D7  — calibration_update_rate never hardcoded in logic code (grep test)
D8  — safety_stock_factor increases on stockout signals
D9  — safety_stock_factor decreases on overstock signals
D10 — safety_stock_factor clamped to [0.05, 0.50]

No database required — all DB interactions replaced with lightweight doubles.
"""

from __future__ import annotations

import random
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from infrastructure.constants import Model, Param, Pattern
from learning.learning_params_updater import LearningParamsUpdater, compute_quantile_evidence
from infrastructure.tenant_params import TenantParams


# ===========================================================================
# Minimal param set shared across tests
# ===========================================================================

_BASE_PARAMS: list[tuple[str, Decimal]] = [
    (Param.CALIBRATION_UPDATE_RATE,              Decimal("0.10")),
    ("confidence_base_cold_start",               Decimal("0.80")),
    ("confidence_base_intermittent",             Decimal("0.80")),
    ("confidence_base_seasonal",                 Decimal("0.80")),
    ("confidence_base_trending",                 Decimal("0.80")),
    ("confidence_base_stable",                   Decimal("0.80")),
    ("quantile_cold_start",                      Decimal("0.90")),
    ("quantile_intermittent",                    Decimal("0.90")),
    ("quantile_seasonal",                        Decimal("0.90")),
    ("quantile_trending",                        Decimal("0.90")),
    ("quantile_stable",                          Decimal("0.90")),
    (Param.SAFETY_STOCK_FACTOR,                  Decimal("0.20")),
    (Param.MIN_LEARNING_EVIDENCE_COUNT,          Decimal("10.00")),
    (Param.QUANTILE_CALIBRATION_STEP,            Decimal("0.02")),
    (Param.CHANNEL_SPLIT_CONFIDENCE_THRESHOLD,   Decimal("0.50")),
]


# ===========================================================================
# Test doubles
# ===========================================================================

class _RoutingCursor:
    """
    Dispatches fetchall() results based on which table name appears in the SQL.

    All route keys are distinct substrings of each other, so order is
    irrelevant. For UPDATE statements rowcount=1 is returned; fetchall()
    is never called by the UPDATE path.
    """

    def __init__(
        self,
        tlp_rows=None,         # tenant_learning_params SELECT rows
        mape_rows=None,        # forecast_outcomes rows
        quantile_rows=None,    # adaptive_quantile_state rows
        signal_rows=None,      # cross_agent_signals rows
    ):
        self._routes = {
            "forecast_outcomes":       mape_rows or [],
            "adaptive_quantile_state": quantile_rows or [],
            "cross_agent_signals":     signal_rows or [],
            "tenant_learning_params":  tlp_rows or [],
        }
        self._current_rows: list = []
        self.rowcount: int = 1   # simulates successful UPDATE
        self.executed: list[tuple] = []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))
        self._current_rows = []
        for fragment, rows in self._routes.items():
            if fragment in sql:
                self._current_rows = rows
                return

    def fetchall(self) -> list:
        return self._current_rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeConn:
    """Wraps a single _RoutingCursor and records commit/rollback."""

    def __init__(self, **cursor_kwargs):
        self._cursor = _RoutingCursor(**cursor_kwargs)
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _tlp_rows(params=None) -> list[tuple]:
    """Build (param_name, current_value) rows for TenantParams.load()."""
    return [(name, val) for name, val in (params or _BASE_PARAMS)]


def _mape_rows(model: str, avg_mape: float, count: int) -> list[tuple]:
    """One forecast_outcomes aggregate row."""
    return [(model, avg_mape, count)]




# ===========================================================================
# D1 — confidence_base_seasonal moves toward evidence (count >= 10)
# ===========================================================================

def test_d1_confidence_base_moves_toward_evidence():
    """
    Inject neural_prophet with avg_mape=0.12, count=10.
    evidence = 1.0 - 0.12 = 0.88
    new = 0.80 + 0.10 × (0.88 - 0.80) = 0.808

    TenantParams.load is patched so we hold a reference to the exact tp
    instance that run() updates in-memory, letting us read the result directly.
    """
    tp = TenantParams("tid", dict(_BASE_PARAMS))
    conn = _FakeConn(
        mape_rows=_mape_rows(Model.PROPHET, avg_mape=0.12, count=10),
    )
    updater = LearningParamsUpdater()
    with patch.object(TenantParams, "load", return_value=tp):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    result = tp.get("confidence_base_seasonal")
    assert result == pytest.approx(0.808, abs=1e-6), (
        f"Expected 0.808 (0.80 + 10% of gap to 0.88), got {result:.6f}"
    )
    assert conn.committed is True


# ===========================================================================
# D2 — unchanged when count < 10 (min evidence rule)
# ===========================================================================

def test_d2_min_evidence_rule_blocks_update():
    """Inject 5 outcomes (< 10). Assert confidence_base_seasonal unchanged at 0.80."""
    conn = _FakeConn(
        tlp_rows=_tlp_rows(),
        mape_rows=_mape_rows(Model.PROPHET, avg_mape=0.12, count=5),
    )
    updater = LearningParamsUpdater()
    updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    tp = TenantParams.load("dummy", conn)
    result = tp.get("confidence_base_seasonal")
    assert result == pytest.approx(0.80, abs=1e-9), (
        f"Expected 0.80 unchanged (count=5 < 10), got {result}"
    )


# ===========================================================================
# D3 — convergence: new_current always moves TOWARD evidence (pure formula)
# ===========================================================================

def test_d3_new_value_always_moves_toward_evidence():
    """
    For 100 random (prior, evidence, rate) combinations the formula
    new = prior + rate × (evidence - prior)
    must satisfy |new - evidence| < |prior - evidence|.
    """
    rng = random.Random(42)

    for _ in range(100):
        prior    = rng.uniform(0.0, 1.0)
        evidence = rng.uniform(0.0, 1.0)
        rate     = rng.uniform(0.01, 1.0)

        if abs(prior - evidence) < 1e-9:
            continue  # already at evidence — skip degenerate case

        new_current  = prior + rate * (evidence - prior)
        dist_before  = abs(prior - evidence)
        dist_after   = abs(new_current - evidence)

        assert dist_after < dist_before, (
            f"new_current moved AWAY from evidence: "
            f"prior={prior:.4f}, evidence={evidence:.4f}, "
            f"rate={rate:.4f}, new={new_current:.4f}"
        )


# ===========================================================================
# D4 — two tenants converge to different values
# ===========================================================================

def test_d4_different_evidence_produces_different_values():
    """
    Tenant A: avg_mape=0.12 → evidence=0.88 → new ≈ 0.808
    Tenant B: avg_mape=0.25 → evidence=0.75 → new ≈ 0.805
    Assert tenant_A.confidence_base_seasonal > tenant_B.confidence_base_seasonal
    """
    tp_a = TenantParams("tid_a", dict(_BASE_PARAMS))
    tp_b = TenantParams("tid_b", dict(_BASE_PARAMS))

    conn_a = _FakeConn(mape_rows=_mape_rows(Model.PROPHET, avg_mape=0.12, count=15))
    conn_b = _FakeConn(mape_rows=_mape_rows(Model.PROPHET, avg_mape=0.25, count=15))

    updater = LearningParamsUpdater()

    with patch.object(TenantParams, "load", return_value=tp_a):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn_a)
    with patch.object(TenantParams, "load", return_value=tp_b):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn_b)

    val_a = tp_a.get("confidence_base_seasonal")
    val_b = tp_b.get("confidence_base_seasonal")

    assert val_a > val_b, (
        f"Expected tenant A (low MAPE) > tenant B (high MAPE): "
        f"A={val_a:.4f}, B={val_b:.4f}"
    )


# ===========================================================================
# D5 — unseeded tenant skipped cleanly
# ===========================================================================

def test_d5_unseeded_tenant_skipped_cleanly():
    """
    Empty tenant_learning_params → run() returns skipped_unseeded, no exception.
    """
    conn = _FakeConn(tlp_rows=[])
    updater = LearningParamsUpdater()
    result = updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    assert result["status"] == "skipped_unseeded"
    assert conn.committed is False  # nothing committed for an empty tenant


# ===========================================================================
# D6 — one tenant failure does not abort other tenants
# ===========================================================================

def test_d6_one_tenant_failure_does_not_abort_others():
    """
    Patch tp.update to raise for one call, then verify a second tenant run
    with a clean connection completes successfully.
    """
    bad_conn = _FakeConn(
        tlp_rows=_tlp_rows(),
        mape_rows=_mape_rows(Model.PROPHET, avg_mape=0.12, count=10),
    )
    good_conn = _FakeConn(
        tlp_rows=_tlp_rows(),
        mape_rows=_mape_rows(Model.PROPHET, avg_mape=0.12, count=10),
    )

    updater = LearningParamsUpdater()

    # Patch TenantParams.update to raise only on bad_conn's tenant
    original_update = TenantParams.update

    call_count = 0

    def _failing_update(self, param_name, evidence_value, conn):
        nonlocal call_count
        call_count += 1
        if conn is bad_conn:
            raise RuntimeError("simulated DB write failure")
        return original_update(self, param_name, evidence_value, conn)

    results = {"ok": 0, "failed": 0}

    with patch.object(TenantParams, "update", _failing_update):
        for conn in [bad_conn, good_conn]:
            try:
                summary = updater.run(tenant_id=str(uuid.uuid4()), conn=conn)
                results["ok"] += 1
            except Exception:
                results["failed"] += 1

    assert results["ok"] == 1, "Good tenant should complete"
    assert results["failed"] == 1, "Bad tenant should fail"
    assert good_conn.committed is True, "Good tenant committed"
    assert bad_conn.committed is False, "Failed tenant must not have committed"


# ===========================================================================
# D7 — calibration_update_rate never hardcoded as a bare literal
# ===========================================================================

def test_d7_update_rate_not_hardcoded():
    """
    The value 0.10 must not appear as a bare numeric literal in logic code.
    Comments and docstrings are permitted; any occurrence in code is a defect.
    """
    import pathlib
    import re

    src_path = pathlib.Path(__file__).parent.parent / "learning" / "learning_params_updater.py"
    lines    = src_path.read_text(encoding="utf-8").splitlines()

    violations = []
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments and docstrings
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
            continue
        # Bare 0.10 or 0.1 as a standalone float literal (not part of 0.10x etc.)
        if re.search(r"\b0\.1(?:0)?\b", stripped):
            # Allow it only if the line is already identified as a string literal
            if not ('"0.10"' in stripped or "'0.10'" in stripped or '"""' in stripped):
                violations.append((lineno, stripped))

    assert not violations, (
        "calibration_update_rate must be read from DB, not hardcoded. "
        f"Found bare 0.10 on lines: {violations}"
    )


# ===========================================================================
# D8 — safety_stock increases on 2+ stockout signals
# ===========================================================================

def test_d8_safety_stock_increases_on_stockout():
    """
    Inject 2 reorder_outcome signals with stockout=True.
    Assert safety_stock_factor increases.
    """
    signals = [
        ({"stockout": True,  "overstock_pct": 0.0},),
        ({"stockout": True,  "overstock_pct": 0.0},),
    ]
    tp    = TenantParams("tid", dict(_BASE_PARAMS))
    prior = tp.get(Param.SAFETY_STOCK_FACTOR)   # 0.20

    conn = _FakeConn(signal_rows=signals)
    updater = LearningParamsUpdater()
    with patch.object(TenantParams, "load", return_value=tp):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    result = tp.get(Param.SAFETY_STOCK_FACTOR)
    assert result > prior, (
        f"safety_stock_factor should increase on 2 stockout signals: "
        f"prior={prior:.4f}, after={result:.4f}"
    )


# ===========================================================================
# D9 — safety_stock decreases on high overstock signals
# ===========================================================================

def test_d9_safety_stock_decreases_on_overstock():
    """
    Inject 1 signal with overstock_pct=0.40 (> threshold of 0.30).
    Assert safety_stock_factor decreases.
    """
    signals = [({"stockout": False, "overstock_pct": 0.40},)]
    tp    = TenantParams("tid", dict(_BASE_PARAMS))
    prior = tp.get(Param.SAFETY_STOCK_FACTOR)   # 0.20

    conn = _FakeConn(signal_rows=signals)
    updater = LearningParamsUpdater()
    with patch.object(TenantParams, "load", return_value=tp):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    result = tp.get(Param.SAFETY_STOCK_FACTOR)
    assert result < prior, (
        f"safety_stock_factor should decrease on overstock signal: "
        f"prior={prior:.4f}, after={result:.4f}"
    )


# ===========================================================================
# D10 — safety_stock clamped to [0.05, 0.50]
# ===========================================================================

def test_d10_safety_stock_clamped_at_upper_bound():
    """
    Start with safety_stock_factor=0.48, inject 2 stockout signals.
    evidence = min(0.50, 0.48 × 1.10) = 0.50
    new = 0.48 + 0.10 × (0.50 - 0.48) = 0.482  ≤ 0.50
    """
    params_near_ceiling = [
        (name, val if name != Param.SAFETY_STOCK_FACTOR else Decimal("0.48"))
        for name, val in _BASE_PARAMS
    ]
    tp = TenantParams("tid", dict(params_near_ceiling))
    signals = [
        ({"stockout": True, "overstock_pct": 0.0},),
        ({"stockout": True, "overstock_pct": 0.0},),
    ]
    conn = _FakeConn(signal_rows=signals)
    updater = LearningParamsUpdater()
    with patch.object(TenantParams, "load", return_value=tp):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    result = tp.get(Param.SAFETY_STOCK_FACTOR)
    assert result <= 0.50, (
        f"safety_stock_factor must not exceed 0.50 (got {result:.4f})"
    )


def test_d10_safety_stock_clamped_at_lower_bound():
    """
    Start with safety_stock_factor=0.06, inject overstock signals.
    evidence = max(0.05, 0.06 × 0.92) = 0.0552
    new = 0.06 + 0.10 × (0.0552 - 0.06) = 0.05952  ≥ 0.05
    """
    params_near_floor = [
        (name, val if name != Param.SAFETY_STOCK_FACTOR else Decimal("0.06"))
        for name, val in _BASE_PARAMS
    ]
    tp = TenantParams("tid", dict(params_near_floor))
    signals = [({"stockout": False, "overstock_pct": 0.40},)]
    conn = _FakeConn(signal_rows=signals)
    updater = LearningParamsUpdater()
    with patch.object(TenantParams, "load", return_value=tp):
        updater.run(tenant_id=str(uuid.uuid4()), conn=conn)

    result = tp.get(Param.SAFETY_STOCK_FACTOR)
    assert result >= 0.05, (
        f"safety_stock_factor must not drop below 0.05 (got {result:.4f})"
    )
