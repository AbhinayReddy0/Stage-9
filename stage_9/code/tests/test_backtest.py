"""
Unit tests for stage9.backtest (Sub-Stage 9.4).

Uses in-process fake cursors/connections — no Postgres required.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from backtesting.backtesting import (
    BacktestContext,
    BacktestMetrics,
    SkuBacktestInput,
    detect_exceptions,
    detect_structural_break,
    prefetch_calibrated_windows,
    run_backtest,
    run_substage_94,
    run_substage_94_parallel,
    select_backtest_window,
    write_pattern_feedback,
)
from infrastructure.batch_writer import BatchWriter
from infrastructure.constants import PATTERN_FEEDBACK_PROXY_MAPE
from infrastructure.tenant_params import TenantParams


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, select_rows=None, raise_on_execute_n=0):
        self.select_rows = select_rows or []
        self._pending = None
        self.executed: list[tuple] = []
        self._raise_remaining = raise_on_execute_n
        self.rowcount = 0

    def execute(self, sql, args=None):
        self.executed.append((sql, args))
        if self._raise_remaining > 0:
            self._raise_remaining -= 1
            raise RuntimeError("fake DB failure")
        if sql.lstrip().startswith("SELECT"):
            self._pending = list(self.select_rows)
            self.rowcount = len(self._pending) if self._pending else 0

    def executemany(self, sql, rows):
        for r in rows:
            self.executed.append((sql, r))

    def fetchone(self):
        return self._pending[0] if self._pending else None

    def fetchall(self):
        return list(self._pending or [])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class FakeConn:
    def __init__(self, select_rows=None, fail_writes=0):
        self._select_rows = select_rows
        self._fail_writes = fail_writes
        self.committed = 0
        self.rolled_back = 0
        self.closed = False
        self._cur = FakeCursor(select_rows=select_rows, raise_on_execute_n=fail_writes)

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        self.closed = True


def _params(**overrides):
    base = {
        "default_backtest_window":          Decimal("60"),
        "min_backtest_window":              Decimal("14"),
        "max_backtest_window":              Decimal("90"),
        "calibration_update_rate":          Decimal("0.10"),
        "structural_break_sensitivity":     Decimal("0.30"),
        "structural_break_confidence_penalty": Decimal("0.15"),
        "backtest_short_obs_threshold":     Decimal("60"),
        "backtest_exploit_obs_threshold":   Decimal("180"),
    }
    base.update({k: Decimal(str(v)) for k, v in overrides.items()})
    return TenantParams("t1", base)


# ---------------------------------------------------------------------------
# Step 1 — window selection
# ---------------------------------------------------------------------------

def test_window_uses_default_when_no_calibrated_row():
    conn = FakeConn(select_rows=[])
    w = select_backtest_window(
        conn, "t1", "seasonal", "Prophet", _params(),
        obs_days=365, ultra_sparse=False, learning_mode="explore",
    )
    assert w == 60


def test_window_uses_calibrated_when_present():
    conn = FakeConn(select_rows=[(45,)])
    w = select_backtest_window(
        conn, "t1", "seasonal", "Prophet", _params(),
        obs_days=365, ultra_sparse=False, learning_mode="explore",
    )
    assert w == 45


def test_window_ultra_sparse_goes_to_min():
    conn = FakeConn(select_rows=[(45,)])
    w = select_backtest_window(
        conn, "t1", "intermittent", "Croston's Method", _params(),
        obs_days=365, ultra_sparse=True, learning_mode="explore",
    )
    assert w == 14


def test_window_short_history_uses_third():
    conn = FakeConn(select_rows=[])
    w = select_backtest_window(
        conn, "t1", "stable", "SES", _params(),
        obs_days=45, ultra_sparse=False, learning_mode="explore",
    )
    # obs_days<60: max(min_w=14, 45//3=15) -> 15
    assert w == 15


def test_window_short_history_respects_min_floor():
    conn = FakeConn(select_rows=[])
    w = select_backtest_window(
        conn, "t1", "stable", "SES", _params(),
        obs_days=30, ultra_sparse=False, learning_mode="explore",
    )
    # obs_days<60: max(14, 30//3=10) -> 14
    assert w == 14


def test_window_exploit_established_uses_max():
    conn = FakeConn(select_rows=[(60,)])
    w = select_backtest_window(
        conn, "t1", "seasonal", "Prophet", _params(),
        obs_days=365, ultra_sparse=False, learning_mode="exploit",
    )
    assert w == 90


def test_window_clamped_to_range():
    conn = FakeConn(select_rows=[(500,)])
    w = select_backtest_window(
        conn, "t1", "seasonal", "Prophet", _params(),
        obs_days=365, ultra_sparse=False, learning_mode="explore",
    )
    assert w == 90  # max_backtest_window


def test_window_reduced_when_gte_obs_days():
    conn = FakeConn(select_rows=[])
    w = select_backtest_window(
        conn, "t1", "stable", "SES", _params(),
        obs_days=20, ultra_sparse=False, learning_mode="explore",
    )
    # obs_days<60 branch: max(14, 20//3=6)=14 — but 14<20 so fine; still fits.
    assert w == 14


# ---------------------------------------------------------------------------
# Step 2 — backtest metrics
# ---------------------------------------------------------------------------

def _df(values):
    return pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=len(values)),
                         "y": values})


def test_run_backtest_mape_masks_zero_actuals():
    # train has 3 obs, test has 3 obs (including a zero)
    df = _df([10.0, 10.0, 10.0, 10.0, 0.0, 10.0])
    def fit(train, test_len):
        return np.array([10.0, 10.0, 10.0])
    m = run_backtest(df, window=3, fit_predict_fn=fit)
    # zero actual masked -> perfect MAPE on remaining
    assert m.mape == pytest.approx(0.0)
    assert m.wape == pytest.approx(abs(0 - 10) / 20)  # 10/20 = 0.5
    assert m.bias == pytest.approx(10 / 20)  # (yhat-actual)=10, actual_sum=20


def test_run_backtest_all_zero_actuals_returns_nan_mape():
    df = _df([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
    def fit(train, test_len):
        return np.array([1.0, 1.0, 1.0])
    m = run_backtest(df, window=3, fit_predict_fn=fit)
    assert np.isnan(m.mape)
    assert np.isnan(m.wape)
    assert np.isnan(m.bias)


def test_run_backtest_rejects_bad_prediction_length():
    df = _df([1.0] * 10)
    def fit(train, test_len):
        return np.array([1.0])
    with pytest.raises(ValueError, match="returned 1 values"):
        run_backtest(df, window=3, fit_predict_fn=fit)


def test_run_backtest_rejects_window_gte_len():
    df = _df([1.0] * 5)
    with pytest.raises(ValueError):
        run_backtest(df, window=5, fit_predict_fn=lambda t, n: np.zeros(n))


# ---------------------------------------------------------------------------
# Step 3 — exception detection
# ---------------------------------------------------------------------------

def test_stockout_3_consecutive_zeros():
    arr = np.array([2, 3, 0, 0, 0, 5, 6], dtype=float)
    flags = detect_exceptions(arr, backtest_mape=0.10)
    assert "stockout" in flags


def test_stockout_2_zeros_not_flagged():
    arr = np.array([2, 0, 0, 5, 0, 0], dtype=float)
    flags = detect_exceptions(arr, backtest_mape=0.10)
    assert "stockout" not in flags


def test_promo_spike_ratio_rule():
    arr = np.array([5, 5, 5, 5, 5, 5, 5, 50], dtype=float)  # last is 10x baseline
    flags = detect_exceptions(arr, backtest_mape=0.10)
    assert "promo_spike" in flags


def test_unusual_drop_detected():
    # baseline near 10, then 3 consecutive days at 2 (80% below)
    arr = np.array([10, 10, 10, 10, 10, 10, 10, 2, 2, 2], dtype=float)
    flags = detect_exceptions(arr, backtest_mape=0.10)
    assert "unusual_drop" in flags


def test_high_volatility_cv_ge_1():
    arr = np.array([0, 0, 100, 0, 0, 100, 0], dtype=float)
    flags = detect_exceptions(arr, backtest_mape=0.10)
    assert "high_volatility" in flags


def test_high_mape_flag_at_threshold():
    arr = np.array([10.0] * 10)
    assert "high_mape" not in detect_exceptions(arr, backtest_mape=0.50)
    assert "high_mape" in detect_exceptions(arr, backtest_mape=0.51)


def test_stable_series_has_no_flags():
    arr = np.array([10.0] * 10)
    assert detect_exceptions(arr, backtest_mape=0.10) == []


# ---------------------------------------------------------------------------
# Step 4 — structural break
# ---------------------------------------------------------------------------

def test_structural_break_skipped_when_no_portfolio_alerts():
    ctx = BacktestContext()
    arr = np.concatenate([np.ones(40), np.full(40, 10.0)])
    detected, idx = detect_structural_break(arr, [], ctx)
    assert detected is False
    assert idx is None
    assert ctx.training_data_truncated is False


def test_structural_break_short_series_noop():
    ctx = BacktestContext()
    arr = np.ones(10)
    detected, idx = detect_structural_break(arr, ["alert"], ctx)
    assert detected is False


def test_structural_break_truncated_branch():
    pytest.importorskip("ruptures")
    # Clear regime change in the middle; both halves >= 30.
    arr = np.concatenate([np.ones(50), np.full(50, 20.0)])
    ctx = BacktestContext()
    detected, idx = detect_structural_break(arr, ["alert"], ctx, penalty=1)
    assert detected is True
    assert idx is not None
    assert ctx.training_data_truncated is True
    assert ctx.insufficient_post_break is False


def test_structural_break_insufficient_post_break_branch():
    pytest.importorskip("ruptures")
    # Break late — post-break < 30
    arr = np.concatenate([np.ones(60), np.full(10, 20.0)])
    ctx = BacktestContext()
    detected, idx = detect_structural_break(arr, ["alert"], ctx, penalty=1)
    assert detected is True
    assert ctx.insufficient_post_break is True
    assert ctx.training_data_truncated is False


# ---------------------------------------------------------------------------
# Step 5 — pattern_feedback writer
# ---------------------------------------------------------------------------

def test_pattern_feedback_happy_path_commits_once():
    conn = FakeConn()
    ok = write_pattern_feedback(
        conn,
        tenant_id=str(uuid.uuid4()), sku_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()),
        pattern_label="seasonal", stage8_confidence=0.85,
        mape=0.10, wape=0.08, bias=0.02,
        model_used="Prophet", model_hint="Prophet",
    )
    assert ok is True
    assert conn.committed == 1
    sql, args = conn._cur.executed[0]
    assert "INSERT INTO stage8.pattern_feedback" in sql
    # Last positional is fallback_used=False
    assert args[-1] is False
    # hint_matched (position -3) is True because model_used == model_hint
    assert args[-3] is True


def test_pattern_feedback_hint_matched_exact_name():
    conn = FakeConn()
    write_pattern_feedback(
        conn,
        tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="stable", stage8_confidence=0.9,
        mape=0.10, wape=None, bias=None,
        model_used="SES", model_hint="SES",
    )
    _, args = conn._cur.executed[0]
    assert args[-3] is True


def test_pattern_feedback_hint_matched_mismatch():
    conn = FakeConn()
    write_pattern_feedback(
        conn,
        tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="stable", stage8_confidence=0.9,
        mape=0.10, wape=None, bias=None,
        model_used="SES", model_hint="Prophet",
    )
    _, args = conn._cur.executed[0]
    assert args[-3] is False


def test_pattern_feedback_classification_good():
    conn = FakeConn()
    write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.10, wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
    )
    _, args = conn._cur.executed[0]
    assert args[-2] == "good"


def test_pattern_feedback_classification_acceptable():
    conn = FakeConn()
    write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.30, wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
    )
    _, args = conn._cur.executed[0]
    assert args[-2] == "acceptable"


def test_pattern_feedback_classification_poor():
    conn = FakeConn()
    write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.60, wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
    )
    _, args = conn._cur.executed[0]
    assert args[-2] == "poor"


def test_pattern_feedback_fallback_writes_proxy_mape_and_quality():
    conn = FakeConn()
    ok = write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="cold_start", stage8_confidence=0.5,
        mape=None, wape=None, bias=None,
        model_used="Naive Forecast", model_hint="Naive Forecast",
        fallback_used=True,
    )
    assert ok is True
    _, args = conn._cur.executed[0]
    # positional layout: ..., mape, wape, bias, model_used, horizon, hint_matched, quality, fallback
    mape_arg = args[5]
    assert mape_arg == PATTERN_FEEDBACK_PROXY_MAPE
    assert args[-2] == "proxy"
    assert args[-1] is True


def test_pattern_feedback_retries_and_eventually_succeeds():
    conn = FakeConn(fail_writes=2)  # first 2 attempts raise
    ok = write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.10, wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
        retry_delay_seconds=0.0,
    )
    assert ok is True
    # 2 failed + 1 success = 3 attempts logged
    assert len(conn._cur.executed) == 3
    assert conn.committed == 1
    assert conn.rolled_back == 2


def test_pattern_feedback_returns_false_after_all_retries_fail():
    conn = FakeConn(fail_writes=5)  # more failures than retries
    ok = write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.10, wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
        retry_delay_seconds=0.0,
    )
    assert ok is False
    assert len(conn._cur.executed) == 3  # default max_retries=3
    assert conn.committed == 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_orchestrator_writes_pattern_feedback_before_batch_flush():
    conn = FakeConn(select_rows=[])  # no calibrated row
    params = _params()
    bw = BatchWriter(conn, batch_size=100)

    df = _df([10.0] * 90)
    def fit(train, test_len):
        return np.array([10.0] * test_len)

    payload = SkuBacktestInput(
        sku_id="s1",
        assigned_model="Prophet",
        pattern_label="seasonal",
        model_hint="Prophet",
        stage8_confidence=0.8,
        df=df,
        obs_days=90,
        ultra_sparse=False,
        learning_mode="explore",
    )
    results = run_substage_94(
        conn, tenant_id="t1", run_id="r1",
        skus=[payload], params=params,
        fit_predict_fn=fit, batch_writer=bw,
    )
    assert "s1" in results
    r = results["s1"]
    assert r.fallback_used is False
    assert r.backtest_mape == pytest.approx(0.0)
    # pattern_feedback was committed (direct write), backtest_decisions queued
    # (not yet flushed because batch_size=100).
    pf_rows = [e for e in conn._cur.executed if "pattern_feedback" in e[0]]
    assert len(pf_rows) == 1
    assert conn.committed == 1  # only pattern_feedback's commit
    assert bw.count == 1  # queued, not flushed


def test_orchestrator_fallback_on_model_failure():
    conn = FakeConn(select_rows=[])
    params = _params()
    bw = BatchWriter(conn, batch_size=100)

    df = _df([10.0] * 90)
    def broken_fit(train, test_len):
        raise RuntimeError("model exploded")

    payload = SkuBacktestInput(
        sku_id="s1", assigned_model="Prophet", pattern_label="seasonal",
        model_hint="Prophet", stage8_confidence=0.8,
        df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
    )
    results = run_substage_94(
        conn, tenant_id="t1", run_id="r1",
        skus=[payload], params=params,
        fit_predict_fn=broken_fit, batch_writer=bw,
    )
    r = results["s1"]
    assert r.fallback_used is True
    pf_rows = [e for e in conn._cur.executed if "pattern_feedback" in e[0]]
    assert len(pf_rows) == 1
    # Proxy MAPE written
    _, args = pf_rows[0]
    assert args[5] == PATTERN_FEEDBACK_PROXY_MAPE


def test_orchestrator_isolates_per_sku_failures():
    conn = FakeConn(select_rows=[])
    params = _params()
    bw = BatchWriter(conn, batch_size=100)

    df_ok = _df([10.0] * 90)

    # Only the "bad" SKU's fit blows up — the "good" SKU must still succeed
    # and pattern_feedback must be written for both (Principle 3 + 4).
    def fit(train, test_len):
        if "BAD_MARKER" in train.attrs:
            raise RuntimeError("model exploded for bad sku")
        return np.array([10.0] * test_len)

    df_bad = _df([10.0] * 90)
    df_bad.attrs["BAD_MARKER"] = True

    payloads = [
        SkuBacktestInput(
            sku_id="good", assigned_model="Prophet", pattern_label="seasonal",
            model_hint="Prophet", stage8_confidence=0.8,
            df=df_ok, obs_days=90, ultra_sparse=False, learning_mode="explore",
        ),
        SkuBacktestInput(
            sku_id="bad", assigned_model="Prophet", pattern_label="seasonal",
            model_hint="Prophet", stage8_confidence=0.8,
            df=df_bad, obs_days=90, ultra_sparse=False, learning_mode="explore",
        ),
    ]
    results = run_substage_94(
        conn, tenant_id="t1", run_id="r1",
        skus=payloads, params=params,
        fit_predict_fn=fit, batch_writer=bw,
    )
    assert results["good"].fallback_used is False
    assert results["bad"].fallback_used is True
    # pattern_feedback written for BOTH SKUs (Principle 4).
    pf_rows = [e for e in conn._cur.executed if "pattern_feedback" in e[0]]
    assert len(pf_rows) == 2


# ---------------------------------------------------------------------------
# Done When acceptance criteria — explicit coverage per Task 14
# ---------------------------------------------------------------------------

def test_done_when_stockout_on_3_consecutive_zeros():
    """DONE WHEN: Stockout detected on 3+ consecutive zeros in test data."""
    arr = np.array([5, 4, 0, 0, 0, 7, 8], dtype=float)
    assert "stockout" in detect_exceptions(arr, backtest_mape=0.10)

    # Boundary — exactly 2 consecutive zeros must NOT fire stockout.
    arr_two = np.array([5, 0, 0, 5, 0, 0, 5], dtype=float)
    assert "stockout" not in detect_exceptions(arr_two, backtest_mape=0.10)


def test_done_when_promo_spike_on_3x_baseline():
    """DONE WHEN: promo_spike detected when day > 3× 7-day baseline."""
    # Seven days of uniform baseline = 10, then a day at 31 (3.1× baseline).
    arr = np.array([10, 10, 10, 10, 10, 10, 10, 31], dtype=float)
    assert "promo_spike" in detect_exceptions(arr, backtest_mape=0.10)

    # A day at exactly the baseline must NOT flag.
    calm = np.array([10] * 10, dtype=float)
    assert "promo_spike" not in detect_exceptions(calm, backtest_mape=0.10)


def test_done_when_pattern_feedback_written_for_every_sku_including_failures():
    """DONE WHEN: pattern_feedback row exists for EVERY SKU including failed."""
    conn = FakeConn(select_rows=[])
    params = _params()
    bw = BatchWriter(conn, batch_size=100)

    df_ok = _df([10.0] * 90)
    df_bad = _df([10.0] * 90)
    df_bad.attrs["FAIL"] = True

    def fit(train, test_len):
        if "FAIL" in train.attrs:
            raise RuntimeError("boom")
        return np.array([10.0] * test_len)

    payloads = [
        SkuBacktestInput(
            sku_id=f"sku-{i}", assigned_model="Prophet", pattern_label="seasonal",
            model_hint="Prophet", stage8_confidence=0.8,
            df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
        )
        for i, df in enumerate([df_ok, df_bad, df_ok, df_bad, df_ok])
    ]

    run_substage_94(
        conn, tenant_id="t1", run_id="r1",
        skus=payloads, params=params,
        fit_predict_fn=fit, batch_writer=bw,
    )

    pf_rows = [e for e in conn._cur.executed if "pattern_feedback" in e[0]]
    assert len(pf_rows) == 5  # one per SKU, success AND failure
    written_sku_ids = {args[1] for _, args in pf_rows}
    assert written_sku_ids == {f"sku-{i}" for i in range(5)}
    # Each failed SKU row carries proxy MAPE + fallback_used=True.
    fallback_rows = [args for _, args in pf_rows if args[-1] is True]
    assert len(fallback_rows) == 2
    for args in fallback_rows:
        assert args[5] == PATTERN_FEEDBACK_PROXY_MAPE
        assert args[-2] == "proxy"


def test_done_when_hint_matched_prophet_vs_prophet():
    """DONE WHEN: hint_matched=True when model_used and model_hint are the same model."""
    conn = FakeConn()
    write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.12, wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
    )
    _, args = conn._cur.executed[0]
    # args layout: ..., model_used, horizon, hint_matched, quality, fallback
    assert args[-3] is True

    # Mismatched models do not match.
    conn2 = FakeConn()
    write_pattern_feedback(
        conn2, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=0.12, wape=None, bias=None,
        model_used="Prophet", model_hint="SES",
    )
    _, args2 = conn2._cur.executed[0]
    assert args2[-3] is False


def test_done_when_pattern_feedback_committed_before_later_writes():
    """
    DONE WHEN: pattern_feedback timestamp < forecasts timestamp per SKU.

    Forecasts are written by Sub-Stage 9.5 (after this substage returns).
    The invariant this substage owns: pattern_feedback is committed
    immediately, BEFORE any batched downstream writes. We assert the
    ordering of operations recorded by the fake connection.
    """
    conn = FakeConn(select_rows=[])
    params = _params()
    bw = BatchWriter(conn, batch_size=1)  # force flush per SKU

    df = _df([10.0] * 90)
    def fit(train, test_len):
        return np.array([10.0] * test_len)

    payload = SkuBacktestInput(
        sku_id="s1", assigned_model="Prophet", pattern_label="seasonal",
        model_hint="Prophet", stage8_confidence=0.8,
        df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
    )
    run_substage_94(
        conn, tenant_id="t1", run_id="r1",
        skus=[payload], params=params,
        fit_predict_fn=fit, batch_writer=bw,
    )

    # Find the first pattern_feedback INSERT and the first backtest_decisions
    # INSERT in execution order.
    pf_idx = next(
        i for i, (sql, _) in enumerate(conn._cur.executed)
        if "pattern_feedback" in sql
    )
    bd_idx = next(
        (i for i, (sql, _) in enumerate(conn._cur.executed)
         if "backtest_decisions" in sql),
        None,
    )
    assert bd_idx is not None, "backtest_decisions was never flushed"
    # pattern_feedback must precede backtest_decisions in wall-clock order.
    # Since forecasts are written even later (Sub-Stage 9.5, after this
    # substage returns), this transitively guarantees pattern_feedback
    # precedes forecasts for every SKU.
    assert pf_idx < bd_idx
    # And pattern_feedback's commit happened (direct write, not batched).
    assert conn.committed >= 2  # one for pattern_feedback, one for batch flush


# ---------------------------------------------------------------------------
# P1 production-fix tests
# ---------------------------------------------------------------------------

def test_prefetch_calibrated_windows_returns_dict_keyed_by_pattern_model():
    """P1-3: one bulk SELECT loads every (pattern, model) -> window."""
    rows = [
        ("seasonal", "Prophet", 45, 30),
        ("stable",   "SES",     20, 30),
        ("seasonal", "Prophet", 60, 90),  # larger horizon -> wins for same key
    ]
    conn = FakeConn(select_rows=rows)
    cache = prefetch_calibrated_windows(conn, "t1")
    assert cache == {("seasonal", "Prophet"): 60, ("stable", "SES"): 20}
    # Exactly one SELECT was issued — no N+1.
    selects = [e for e in conn._cur.executed if e[0].lstrip().startswith("SELECT")]
    assert len(selects) == 1


def test_prefetch_skips_rows_with_null_window():
    rows = [
        ("seasonal", "Prophet", None, 30),
        ("stable",   "SES",     20,   30),
    ]
    conn = FakeConn(select_rows=rows)
    cache = prefetch_calibrated_windows(conn, "t1")
    assert cache == {("stable", "SES"): 20}


def test_select_backtest_window_uses_cache_skips_db_query():
    """P1-3: when a cache is provided, no DB call is issued."""
    conn = FakeConn(select_rows=[(99,)])  # would be picked up if hit
    cache = {("seasonal", "Prophet"): 45}
    w = select_backtest_window(
        conn, "t1", "seasonal", "Prophet", _params(),
        obs_days=365, ultra_sparse=False, learning_mode="explore",
        calibrated_cache=cache,
    )
    assert w == 45
    selects = [e for e in conn._cur.executed if e[0].lstrip().startswith("SELECT")]
    assert selects == []  # cache hit -> zero queries


def test_select_backtest_window_cache_miss_falls_back_to_default():
    """Empty cache hit -> uses TenantParams default, still no DB call."""
    conn = FakeConn(select_rows=[(45,)])
    w = select_backtest_window(
        conn, "t1", "seasonal", "Prophet", _params(),
        obs_days=365, ultra_sparse=False, learning_mode="explore",
        calibrated_cache={},  # empty dict, not None
    )
    assert w == 60  # default_backtest_window
    selects = [e for e in conn._cur.executed if e[0].lstrip().startswith("SELECT")]
    assert selects == []


def test_orchestrator_uses_dedicated_pf_conn():
    """P1-4: pattern_feedback writes go to pf_conn, not the main conn."""
    main_conn = FakeConn(select_rows=[])
    pf_conn = FakeConn(select_rows=[])
    bw = BatchWriter(main_conn, batch_size=1)

    df = _df([10.0] * 90)
    payload = SkuBacktestInput(
        sku_id="s1", assigned_model="Prophet", pattern_label="seasonal",
        model_hint="Prophet", stage8_confidence=0.8,
        df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
    )
    run_substage_94(
        main_conn, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        fit_predict_fn=lambda t, n: np.array([10.0] * n),
        batch_writer=bw, pf_conn=pf_conn,
    )

    # pattern_feedback hit pf_conn only.
    pf_main = [e for e in main_conn._cur.executed if "pattern_feedback" in e[0]]
    pf_alt = [e for e in pf_conn._cur.executed if "pattern_feedback" in e[0]]
    assert pf_main == []
    assert len(pf_alt) == 1
    # pf_conn committed once (the sacred write); main_conn committed once
    # for the batched backtest_decisions flush.
    assert pf_conn.committed == 1
    assert main_conn.committed == 1


def test_pattern_feedback_nan_mape_forces_fallback():
    """P2-8 (bundled with P1-4): NaN MAPE → fallback proxy row."""
    conn = FakeConn()
    write_pattern_feedback(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        pattern_label="seasonal", stage8_confidence=0.8,
        mape=float("nan"), wape=None, bias=None,
        model_used="Prophet", model_hint="Prophet",
    )
    _, args = conn._cur.executed[0]
    # mape replaced with proxy, quality='proxy', fallback_used=True
    assert args[5] == PATTERN_FEEDBACK_PROXY_MAPE
    assert args[-2] == "proxy"
    assert args[-1] is True


def test_batch_writer_uses_executemany_fallback_for_fake_cursor():
    """P2-1: with a fake cursor (no .connection), BatchWriter falls
    back to executemany — proves the path works in unit-test env."""
    conn = FakeConn()
    bw = BatchWriter(conn, batch_size=2)
    bw.queue("backtest_decisions", {"a": 1, "b": 2})
    bw.queue("backtest_decisions", {"a": 3, "b": 4})
    bw.flush_if_needed()
    # Two rows recorded by FakeCursor.executemany (one tuple each).
    rows = [e for e in conn._cur.executed if "backtest_decisions" in e[0]]
    assert len(rows) == 2  # both rows of the executemany call
    assert conn.committed == 1


def test_batch_writer_flush_failure_clears_buffer_and_rolls_back():
    """P2-6: a flush failure must not leave the buffer holding stale rows."""
    class FailingConn(FakeConn):
        def __init__(self):
            super().__init__()
            self._cur = _RaisingCursor()

    class _RaisingCursor(FakeCursor):
        def executemany(self, sql, rows):
            raise RuntimeError("disk full")

    conn = FailingConn()
    bw = BatchWriter(conn, batch_size=10)
    bw.queue("backtest_decisions", {"a": 1})
    with pytest.raises(RuntimeError, match="disk full"):
        bw.flush()
    assert bw.count == 0  # buffer cleared even on failure
    assert conn.rolled_back == 1


# ---------------------------------------------------------------------------
# Parallel orchestrator (P1-2)
# ---------------------------------------------------------------------------

def _toplevel_fit(train, test_len):
    """Module-level so the parallel executor can pickle/import it."""
    return np.array([10.0] * test_len)


class _SyncExecutor:
    """In-process stand-in for ProcessPoolExecutor — runs each submitted
    callable immediately and returns a future-like wrapping its result."""

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def submit(self, fn, *args, **kwargs):
        class _F:
            def __init__(self, value):
                self._value = value
            def result(self):
                return self._value
        return _F(fn(*args, **kwargs))


def test_parallel_orchestrator_runs_each_chunk_with_isolated_conns():
    """P1-2: each worker opens its own conn + pf_conn via the factories."""
    df = _df([10.0] * 90)
    skus = [
        SkuBacktestInput(
            sku_id=f"sku-{i}", assigned_model="Prophet", pattern_label="seasonal",
            model_hint="Prophet", stage8_confidence=0.8,
            df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
        )
        for i in range(6)
    ]
    opened: list[FakeConn] = []

    def connect_fn():
        c = FakeConn(select_rows=[])
        opened.append(c)
        return c

    def pf_connect_fn():
        c = FakeConn()
        opened.append(c)
        return c

    results = run_substage_94_parallel(
        tenant_id="t1", run_id="r1", skus=skus,
        params=_params(),
        fit_predict_fn=_toplevel_fit,
        connect_fn=connect_fn,
        pf_connect_fn=pf_connect_fn,
        max_workers=3,
        executor_factory=_SyncExecutor,
    )
    assert set(results.keys()) == {f"sku-{i}" for i in range(6)}
    # 3 workers × 2 conns each = 6 connections opened.
    assert len(opened) == 6
    # Every conn was closed exactly once (asserted via FakeConn closed flag below).
    for c in opened:
        assert getattr(c, "closed", False) is True


def test_parallel_orchestrator_empty_skus_short_circuits():
    """No work, no workers, no conns opened."""
    opened = []
    def cf():
        opened.append(FakeConn())
        return opened[-1]
    out = run_substage_94_parallel(
        tenant_id="t1", run_id="r1", skus=[],
        params=_params(),
        fit_predict_fn=_toplevel_fit,
        connect_fn=cf, pf_connect_fn=cf,
        executor_factory=_SyncExecutor,
    )
    assert out == {}
    assert opened == []


# ---------------------------------------------------------------------------
# Audit fix coverage — P3 outer-perimeter isolation + execution_log writes
# ---------------------------------------------------------------------------

def test_outer_loop_survives_batch_writer_failure_and_logs():
    """
    P3 audit: a flush failure inside batch_writer must not crash the run.
    The orchestrator catches it, calls log_failure_fn, and continues.
    """
    class _BadCursor(FakeCursor):
        def executemany(self, sql, rows):
            raise RuntimeError("disk full mid-flush")

    class _BadConn(FakeConn):
        def __init__(self):
            super().__init__(select_rows=[])
            self._cur = _BadCursor(select_rows=[])

    conn = _BadConn()
    bw = BatchWriter(conn, batch_size=1)  # flush after every add
    params = _params()
    failures: list = []

    df = _df([10.0] * 90)
    payloads = [
        SkuBacktestInput(
            sku_id=f"sku-{i}", assigned_model="Prophet", pattern_label="seasonal",
            model_hint="Prophet", stage8_confidence=0.8,
            df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
        )
        for i in range(3)
    ]
    results = run_substage_94(
        conn, tenant_id="t1", run_id="r1",
        skus=payloads, params=params,
        fit_predict_fn=lambda t, n: np.array([10.0] * n),
        batch_writer=bw,
        log_failure_fn=lambda *a: failures.append(a),
    )
    # All 3 SKUs got a result row even though every flush exploded.
    assert set(results) == {"sku-0", "sku-1", "sku-2"}
    # log_failure_fn fired for each; reason starts with backtest_outer_failure.
    outer_fails = [f for f in failures if "backtest_outer_failure" in f[3]]
    assert len(outer_fails) == 3


def test_pattern_feedback_retry_exhaustion_calls_log_failure():
    """
    P4 audit: when write_pattern_feedback returns False after 3 retries,
    the orchestrator must call log_failure_fn so stage9_sku_execution_log
    captures the audit gap.
    """
    main = FakeConn(select_rows=[])
    # pf_conn that fails every write (3 retries → False return).
    pf = FakeConn(fail_writes=99)
    bw = BatchWriter(main, batch_size=10)
    params = _params()
    failures: list = []

    df = _df([10.0] * 90)
    payload = SkuBacktestInput(
        sku_id="s1", assigned_model="Prophet", pattern_label="seasonal",
        model_hint="Prophet", stage8_confidence=0.8,
        df=df, obs_days=90, ultra_sparse=False, learning_mode="explore",
    )
    run_substage_94(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=params,
        fit_predict_fn=lambda t, n: np.array([10.0] * n),
        batch_writer=bw, pf_conn=pf,
        log_failure_fn=lambda *a: failures.append(a),
    )
    # log_failure recorded the exhausted-retry case explicitly.
    reasons = [f[3] for f in failures]
    assert "pattern_feedback_write_exhausted" in reasons


def test_constants_imported_from_stage9_constants():
    """All spec-locked thresholds now live in stage9.constants — no
    file-local literals. Touchstone test: importing them yields the
    expected values."""
    from infrastructure.constants import (
        HIGH_MAPE_FLAG_THRESHOLD, HIGH_VOLATILITY_CV, MIN_POST_BREAK_LEN,
        PATTERN_FEEDBACK_HORIZON_DAYS, PATTERN_FEEDBACK_PROXY_MAPE,
        PROMO_SPIKE_RATIO, PROMO_SPIKE_Z, QUALITY_ACCEPTABLE_MAX,
        QUALITY_GOOD_MAX, ROLLING_BASELINE_DAYS, STOCKOUT_MIN_ZERO_STREAK,
        UNUSUAL_DROP_KEEP_RATIO, UNUSUAL_DROP_MIN_STREAK,
    )
    assert HIGH_MAPE_FLAG_THRESHOLD == 0.50
    assert HIGH_VOLATILITY_CV == 1.0
    assert MIN_POST_BREAK_LEN == 30
    assert PATTERN_FEEDBACK_HORIZON_DAYS == 30
    # high_mape detection threshold and proxy MAPE happen to share a
    # value but are distinct concepts — they're separate constants.
    assert HIGH_MAPE_FLAG_THRESHOLD == PATTERN_FEEDBACK_PROXY_MAPE
    assert PROMO_SPIKE_RATIO == 2.0 and PROMO_SPIKE_Z == 3.0
    assert QUALITY_GOOD_MAX == 0.15 and QUALITY_ACCEPTABLE_MAX == 0.40
    assert ROLLING_BASELINE_DAYS == 7
    assert STOCKOUT_MIN_ZERO_STREAK == UNUSUAL_DROP_MIN_STREAK == 3
    assert UNUSUAL_DROP_KEEP_RATIO == 0.4
