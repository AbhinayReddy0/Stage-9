"""
tests/conftest.py — Atheera Stage 9 Test Suite
================================================
Shared pytest fixtures used across all test modules.

Fixtures provided:
    df_normal       — 60 rows, clean daily demand (5–15 units/day)
    df_seasonal     — 2 years of sinusoidal demand with weekly cycle
    df_trending     — 90 rows of steadily increasing demand
    df_sparse       — 10 rows (below validation holdout threshold)
    df_nan          — 60 rows with NaN/Inf injected every 5th row
    df_all_nonzero  — 30 rows, every day has demand (Croston E001 trigger)
    df_single_nz    — 20 rows, only last day has demand (Croston E001 trigger)
    df_constant     — 100 rows, identical qty every day (Prophet E002 trigger)
    mock_ctx        — minimal LearningContext object
    mock_params     — TenantParams stub returning sensible defaults
    mock_bw         — BatchWriter stub that records queued rows
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infrastructure.constants import Param


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_normal() -> pd.DataFrame:
    """60 days of clean, stable demand (5–15 units/day)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "qty":  rng.uniform(5, 15, 60),
    })


@pytest.fixture
def df_seasonal() -> pd.DataFrame:
    """2 years of sinusoidal demand — weekly + annual cycles."""
    t   = np.arange(730)
    qty = (
        50
        + 20 * np.sin(2 * np.pi * t / 365)   # annual cycle
        + 5  * np.sin(2 * np.pi * t / 7)      # weekly cycle
        + np.random.default_rng(1).normal(0, 2, 730)
    )
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=730),
        "qty":  np.maximum(qty, 0),
    })


@pytest.fixture
def df_trending() -> pd.DataFrame:
    """90 days of steadily growing demand starting at 10 units/day."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=90),
        "qty":  np.arange(90, dtype=float) * 0.5 + 10,
    })


@pytest.fixture
def df_sparse() -> pd.DataFrame:
    """10 rows — below the 14-day validation holdout threshold."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "qty":  np.ones(10) * 5.0,
    })


@pytest.fixture
def df_nan() -> pd.DataFrame:
    """60 rows with NaN and Inf injected at every 5th row."""
    qty = np.arange(60, dtype=float)
    qty[::5]  = float("nan")
    qty[1::5] = float("inf")
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "qty":  qty,
    })


@pytest.fixture
def df_all_nonzero() -> pd.DataFrame:
    """30 rows all with demand > 0 — Croston E001 boundary condition."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "qty":  np.ones(30) * 5.0,
    })


@pytest.fixture
def df_single_nz() -> pd.DataFrame:
    """20 rows with only the last day having demand — Croston E001 trigger."""
    qty      = np.zeros(20)
    qty[-1]  = 10.0
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "qty":  qty,
    })


@pytest.fixture
def df_constant() -> pd.DataFrame:
    """100 rows with identical qty — Prophet E002 (zero-variance) trigger."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "qty":  np.full(100, 10.0),
    })


@pytest.fixture
def df_intermittent() -> pd.DataFrame:
    """30 rows of sporadic demand — zeros interspersed with demand events."""
    qty = np.zeros(30)
    qty[[2, 7, 14, 21, 28]] = [8.0, 5.0, 12.0, 6.0, 9.0]
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "qty":  qty,
    })


@pytest.fixture
def df_all_zeros() -> pd.DataFrame:
    """60 rows with all-zero demand — slow-moving or delisted SKU."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "qty":  np.zeros(60),
    })


@pytest.fixture
def df_one_row() -> pd.DataFrame:
    """Single-row DataFrame — brand-new SKU with one day of history."""
    return pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "qty": [10.0]})


@pytest.fixture
def df_first_nonzero() -> pd.DataFrame:
    """20 rows with only the first day having demand — discontinued after launch."""
    qty = np.zeros(20)
    qty[0] = 10.0
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "qty":  qty,
    })


@pytest.fixture
def df_declining() -> pd.DataFrame:
    """90 rows of steadily declining demand from 50 toward 0 — end-of-life SKU."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=90),
        "qty":  np.maximum(50 - np.arange(90, dtype=float) * 0.5, 0),
    })


@pytest.fixture
def df_extinction() -> pd.DataFrame:
    """60 rows: demand events in the first 20 days, then 40 consecutive zeros."""
    qty = np.zeros(60)
    qty[[2, 5, 9, 14, 18]] = [8.0, 6.0, 10.0, 7.0, 5.0]
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "qty":  qty,
    })


@pytest.fixture
def df_flat() -> pd.DataFrame:
    """30 rows of identical qty=10 — SES convergence fixture."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "qty":  [10.0] * 30,
    })


@pytest.fixture
def df_flat_long() -> pd.DataFrame:
    """120 rows of identical qty=20 — for residual / horizon tests."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=120),
        "qty":  [20.0] * 120,
    })


@pytest.fixture
def df_noisy_stable() -> pd.DataFrame:
    """60 rows of stable demand with light noise (range 10–20)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "qty":  rng.uniform(10, 20, 60),
    })


@pytest.fixture
def df_with_nan_inf() -> pd.DataFrame:
    """60 rows with NaN every 5th and Inf every 5th+1 — sanitisation test."""
    qty = np.arange(60, dtype=float)
    qty[::5]  = float("nan")
    qty[1::5] = float("inf")
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "qty":  qty,
    })


@pytest.fixture
def df_single_row() -> pd.DataFrame:
    """One row — the smallest possible non-empty input."""
    return pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")],
        "qty":  [10.0],
    })


@pytest.fixture
def df_empty() -> pd.DataFrame:
    """Zero rows — boundary condition every model must handle."""
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "qty":  pd.Series(dtype="float64"),
    })


# ---------------------------------------------------------------------------
# Fake DB cursor / connection — for unit tests that verify emitted SQL
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, canned_rows: list | None = None) -> None:
        self.executed: list[tuple] = []
        self._rows = list(canned_rows) if canned_rows else []
        self._idx = 0

    def execute(self, sql, params=None) -> None:
        self.executed.append((sql, params))
        self._idx = 0

    def fetchone(self):
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def fetchall(self):
        out = list(self._rows[self._idx:])
        self._idx = len(self._rows)
        return out

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


class _FakeConn:
    def __init__(self, canned_rows: list | None = None) -> None:
        self.cur = _FakeCursor(canned_rows)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, **kwargs) -> _FakeCursor:
        return self.cur

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


@pytest.fixture
def fake_conn() -> _FakeConn:
    return _FakeConn()


@pytest.fixture
def fake_conn_factory():
    return _FakeConn


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class _MockCtx:
    """Minimal LearningContext for unit tests."""
    sku_id           = "sku-test-001"
    tenant_id        = "tenant-test-001"
    run_id           = "run-test-001"
    assigned_model   = "Naive Forecast"
    pattern_label    = "cold_start"
    selected_features = ["date", "qty"]
    sample_weights   = None
    baseline_mape    = 0.5
    is_b2b           = False
    lifecycle_stage  = None


class _MockParams:
    """TenantParams stub — returns sensible starting values."""
    _values = {
        Param.THOMPSON_EXPLORATION_BUDGET: 3,
        Param.FEATURE_RELIABILITY_FLOOR:   0.30,
        Param.MAX_PROMO_MULTIPLIER:        3.0,
    }

    def get(self, key: str) -> float:
        if key not in self._values:
            raise KeyError(f"MockParams: unknown param '{key}'")
        return self._values[key]


class _MockBatchWriter:
    """BatchWriter stub that records every queued row for assertion."""
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.tables: list[str] = []

    def queue(self, table: str, row: dict) -> None:
        self.tables.append(table)
        self.rows.append(row)

    def row_for(self, table: str) -> dict:
        """Return first row queued for `table`. Raises if not found."""
        for t, r in zip(self.tables, self.rows):
            if t == table:
                return r
        raise AssertionError(f"No row queued for table '{table}'")


@pytest.fixture
def mock_ctx() -> _MockCtx:
    return _MockCtx()


@pytest.fixture
def mock_params() -> _MockParams:
    return _MockParams()


@pytest.fixture
def mock_bw() -> _MockBatchWriter:
    return _MockBatchWriter()


# ---------------------------------------------------------------------------
# Common constants
# ---------------------------------------------------------------------------

EXPECTED_HORIZON_KEYS = {
    "forecast_7d", "forecast_14d", "forecast_30d", "forecast_60d",
    "forecast_90d", "forecast_150d", "forecast_180d", "forecast_365d",
}

FEATURES = ["date", "qty"]
