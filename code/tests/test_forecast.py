"""
Unit tests for stage9.forecast (Sub-Stage 9.5).

Uses in-process fakes; no Postgres / no real model fits.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from infrastructure.batch_writer import BatchWriter
from infrastructure.constants import HORIZONS
from forecasting.forecasting import (
    EXCEPTION_PENALTY_FLAGS,
    ForecastBundle,
    ForecastContext,
    SkuForecastInput,
    bootstrap_quantiles_for_horizons,
    compute_confidence,
    determine_risk_level,
    determine_status,
    determine_tier,
    emit_forecast_risk_signal,
    generate_horizons,
    prefetch_calibration_gaps,
    reasonableness_check,
    run_substage_95,
    run_substage_95_parallel,
)
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

    def execute(self, sql, args=None):
        self.executed.append((sql, args))
        if self._raise_remaining > 0:
            self._raise_remaining -= 1
            raise RuntimeError("fake DB failure")
        if sql.lstrip().startswith("SELECT"):
            self._pending = list(self.select_rows)

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
        "confidence_base_seasonal": Decimal("0.80"),
        "confidence_base_stable":   Decimal("0.90"),
        "confidence_base_cold_start": Decimal("0.50"),
        "confidence_floor":         Decimal("0.30"),
        "confidence_ceiling":       Decimal("0.95"),
        "decision_gate_threshold":  Decimal("0.70"),
        "review_suggested_threshold": Decimal("0.60"),
        "review_required_threshold":  Decimal("0.45"),
        "exception_penalty":        Decimal("0.80"),
        "overconfidence_threshold": Decimal("0.10"),
        "stage8_penalty_threshold": Decimal("0.60"),
        "max_forecast_vs_baseline": Decimal("3.00"),
        "min_forecast_vs_baseline": Decimal("0.30"),
        "structural_break_confidence_penalty": Decimal("0.15"),
        "calibration_update_rate":  Decimal("0.10"),
        # Sub-Stage 9.5 confidence multipliers (now tenant-tunable)
        "mape_cap_in_confidence":   Decimal("0.50"),
        "overconfidence_mult":      Decimal("0.90"),
        "underconfidence_mult":     Decimal("1.10"),
        "stage8_penalty_mult":      Decimal("0.92"),
        "insufficient_post_break_mult": Decimal("0.75"),
        "forecast_unusually_high_mult": Decimal("0.85"),
        "forecast_unusually_low_mult":  Decimal("0.90"),
        # Risk-band cutoffs
        "risk_low_min":             Decimal("0.85"),
        "risk_medium_min":          Decimal("0.70"),
    }
    base.update({k: Decimal(str(v)) for k, v in overrides.items()})
    return TenantParams("t1", base)


def _df(values):
    return pd.DataFrame({
        "ds":  pd.date_range("2026-01-01", periods=len(values)),
        "y":   values,   # Prophet internal convention
        "qty": values,   # Stage 9 canonical column name
    })


# A linear-scale forecast: 10 units/day every horizon. Importable for
# the parallel-orchestrator test (must survive pickle).
def linear_10_per_day(model_name, train_df, horizons):
    return ForecastBundle(
        points_per_horizon={h: 10.0 * h for h in horizons},
        residuals=np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1] * 5),
    )


def deterministic_bootstrap(point, residuals, pattern):
    p = float(point)
    return {"mean": p, "p50": p, "p80": p * 1.10, "p90": p * 1.20}


# ---------------------------------------------------------------------------
# Step 1 — generate_horizons
# ---------------------------------------------------------------------------

def test_generate_horizons_returns_all_8_horizons():
    df = _df([10.0] * 90)
    points = generate_horizons(
        "Prophet", df, forecast_fn=linear_10_per_day,
        ctx=ForecastContext(), params=_params(),
    )
    assert sorted(points) == sorted(HORIZONS)
    assert points[7] == pytest.approx(70)
    assert points[365] == pytest.approx(3650)


def test_generate_horizons_applies_oos_factor():
    df = _df([10.0] * 90)
    ctx = ForecastContext(oos_adjustment_factor=1.20)
    points = generate_horizons(
        "Prophet", df, forecast_fn=linear_10_per_day,
        ctx=ctx, params=_params(),
    )
    assert points[30] == pytest.approx(300 * 1.20)


def test_generate_horizons_caps_to_effective_max():
    df = _df([10.0] * 90)
    ctx = ForecastContext(effective_max_horizon=30)
    points = generate_horizons(
        "Prophet", df, forecast_fn=linear_10_per_day,
        ctx=ctx, params=_params(),
    )
    # Past 30d the value should hold flat at the 30d level.
    assert points[30] == pytest.approx(300)
    for h in (60, 90, 150, 180, 365):
        assert points[h] == pytest.approx(300)


def test_generate_horizons_raises_on_missing_horizons():
    df = _df([10.0] * 90)
    def bad_fn(model, df, hs):
        return ForecastBundle(points_per_horizon={7: 70}, residuals=np.array([]))
    with pytest.raises(ValueError, match="HORIZONS keys"):
        generate_horizons("Prophet", df, forecast_fn=bad_fn,
                          ctx=ForecastContext(), params=_params())


# ---------------------------------------------------------------------------
# Step 2 — bootstrap quantiles + monotonicity
# ---------------------------------------------------------------------------

def test_bootstrap_returns_4_quantiles_per_horizon():
    points = {h: 100.0 for h in HORIZONS}
    out = bootstrap_quantiles_for_horizons(
        points, residuals=None, pattern="seasonal",
        bootstrap_fn=deterministic_bootstrap,
    )
    for h in HORIZONS:
        q = out[h]
        assert set(q.keys()) == {"mean", "p50", "p80", "p90"}


def test_bootstrap_enforces_p50_le_p80_le_p90():
    # Hand back deliberately disordered quantiles.
    def bad_fn(point, residuals, pattern):
        return {"mean": 100, "p50": 110, "p80": 105, "p90": 95}
    out = bootstrap_quantiles_for_horizons(
        {h: 100.0 for h in HORIZONS}, None, "seasonal", bad_fn,
    )
    for h in HORIZONS:
        q = out[h]
        assert q["p50"] <= q["p80"] <= q["p90"]


# ---------------------------------------------------------------------------
# Step 3 — reasonableness check
# ---------------------------------------------------------------------------

def test_reasonableness_no_baseline_returns_no_flags():
    df = _df([10.0] * 30)  # < 90 rows
    flags, mult = reasonableness_check(300.0, df, _params())
    assert flags == []
    assert mult == 1.0


def test_reasonableness_unusually_high():
    df = _df([10.0] * 90)  # baseline 10/day
    # daily_30d = 5000/30 = 166 — way over 10*3=30
    flags, mult = reasonableness_check(5000.0, df, _params())
    assert "forecast_unusually_high" in flags
    assert mult == pytest.approx(0.85)


def test_reasonableness_unusually_low():
    df = _df([10.0] * 90)
    # daily_30d = 30/30 = 1 — under 10*0.3=3
    flags, mult = reasonableness_check(30.0, df, _params())
    assert "forecast_unusually_low" in flags
    assert mult == pytest.approx(0.90)


def test_reasonableness_in_band_no_flags():
    df = _df([10.0] * 90)
    # daily_30d = 300/30 = 10 — exactly baseline
    flags, mult = reasonableness_check(300.0, df, _params())
    assert flags == []
    assert mult == 1.0


# ---------------------------------------------------------------------------
# Step 4 — confidence formula
# ---------------------------------------------------------------------------

def test_confidence_clean_path():
    """No exceptions, no break, mid mape → straight formula."""
    base, final = compute_confidence(
        pattern_label="seasonal",
        backtest_mape=0.10,
        exception_flags=[],
        calibration_gap=0.05,           # < threshold 0.10 → no penalty
        stage8_confidence=0.80,         # > threshold 0.60 → no penalty
        reorder_bias_factor=1.0,
        ctx=ForecastContext(),
        params=_params(),
    )
    # base 0.80 × (1 - 0.10) = 0.72
    assert base == pytest.approx(0.80)
    assert final == pytest.approx(0.72)


def test_confidence_applies_exception_penalty_when_any_flag():
    _, final_clean = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10, exception_flags=[],
        calibration_gap=None, stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    _, final_flagged = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=["stockout"],
        calibration_gap=None, stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    assert final_flagged == pytest.approx(final_clean * 0.80)  # exception_penalty


def test_confidence_caps_mape_at_50():
    """High mape (0.90) is treated as 0.50 — confidence still positive."""
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.90,
        exception_flags=[], calibration_gap=None, stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    # 0.80 * (1 - 0.50) = 0.40 → above floor 0.30
    assert final == pytest.approx(0.40)


def test_confidence_overconfidence_penalty():
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=0.20,  # > +0.10 threshold
        stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    # 0.72 × 0.90 = 0.648
    assert final == pytest.approx(0.72 * 0.90)


def test_confidence_underconfidence_reward():
    """Symmetric branch: gap < -threshold → ×1.10 boost."""
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=-0.20,  # < -0.10 threshold
        stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    # 0.72 × 1.10 = 0.792
    assert final == pytest.approx(0.72 * 1.10)


def test_confidence_calibration_within_band_no_change():
    """|gap| ≤ threshold → no calibration multiplier applied."""
    _, final_pos = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=0.05,   # within ±0.10
        stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    _, final_neg = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=-0.05,
        stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    assert final_pos == pytest.approx(0.72)
    assert final_neg == pytest.approx(0.72)


def test_confidence_stage8_penalty():
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=None,
        stage8_confidence=0.40,  # < 0.60 threshold
        reorder_bias_factor=1.0, ctx=ForecastContext(), params=_params(),
    )
    assert final == pytest.approx(0.72 * 0.92)


def test_confidence_structural_break_truncated_branch():
    ctx = ForecastContext(training_data_truncated=True)
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=None, stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ctx, params=_params(),
    )
    # 1 - structural_break_confidence_penalty (0.15 default) = 0.85
    assert final == pytest.approx(0.72 * 0.85)


def test_confidence_insufficient_post_break_branch():
    ctx = ForecastContext(insufficient_post_break=True)
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=None, stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ctx, params=_params(),
    )
    assert final == pytest.approx(0.72 * 0.75)


def test_confidence_clamped_to_floor():
    """Stack many penalties → would go below floor 0.30 → clamps."""
    ctx = ForecastContext(insufficient_post_break=True)
    _, final = compute_confidence(
        pattern_label="cold_start", backtest_mape=0.50,
        exception_flags=["stockout", "promo_spike"],
        calibration_gap=0.30, stage8_confidence=0.40,
        reorder_bias_factor=0.92, ctx=ctx, params=_params(),
    )
    assert final == pytest.approx(0.30)


def test_confidence_clamped_to_ceiling():
    """Strong base × high reorder bias should never exceed 0.95."""
    _, final = compute_confidence(
        pattern_label="stable", backtest_mape=0.0,
        exception_flags=[], calibration_gap=None, stage8_confidence=0.95,
        reorder_bias_factor=1.50,  # cranked up
        ctx=ForecastContext(), params=_params(),
    )
    assert final == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# confidence.py — module isolation
# ---------------------------------------------------------------------------

from forecasting.confidence import (
    EXCEPTION_PENALTY_FLAGS,
    ForecastContext as _ConfForecastContext,
    compute_confidence as _compute_confidence_direct,
)


def test_exception_penalty_flags_contains_all_six_strings():
    assert EXCEPTION_PENALTY_FLAGS == {
        "stockout", "promo_spike", "unusual_drop", "high_volatility",
        "forecast_unusually_high", "forecast_unusually_low",
    }


def test_exception_penalty_flags_is_frozenset():
    assert isinstance(EXCEPTION_PENALTY_FLAGS, frozenset)


def test_compute_confidence_importable_from_confidence_module():
    _, final = _compute_confidence_direct(
        pattern_label="seasonal",
        backtest_mape=0.10,
        exception_flags=[],
        calibration_gap=None,
        stage8_confidence=0.80,
        reorder_bias_factor=1.0,
        ctx=_ConfForecastContext(),
        params=_params(),
    )
    assert 0.0 < final <= 1.0


def test_confidence_module_and_forecasting_reexport_are_same_object():
    from forecasting.forecasting import compute_confidence as _via_forecasting
    assert _via_forecasting is _compute_confidence_direct


# ---------------------------------------------------------------------------
# Step 5 — status / tier / risk
# ---------------------------------------------------------------------------

def test_status_watchlist_wins_first():
    ctx = ForecastContext(on_watchlist=True)
    s = determine_status(0.95, ["high_mape"], ctx, _params())
    assert s == "watchlist_review"


def test_status_high_mape_forces_needs_acknowledgment():
    s = determine_status(0.95, ["high_mape"], ForecastContext(), _params())
    assert s == "needs_acknowledgment"


def test_status_low_confidence_needs_acknowledgment():
    s = determine_status(0.50, [], ForecastContext(), _params())
    assert s == "needs_acknowledgment"


def test_status_clean_forecasted():
    s = determine_status(0.85, [], ForecastContext(), _params())
    assert s == "forecasted"


def test_tier_thresholds():
    p = _params()
    assert determine_tier(0.95, p) == "auto_proceed"
    assert determine_tier(0.65, p) == "review_suggested"
    assert determine_tier(0.50, p) == "review_required"
    assert determine_tier(0.30, p) == "manual_override"


def test_risk_level_bands():
    p = _params()
    assert determine_risk_level(0.85, p) == "low"
    assert determine_risk_level(0.95, p) == "low"
    assert determine_risk_level(0.84, p) == "medium"
    assert determine_risk_level(0.70, p) == "medium"
    assert determine_risk_level(0.69, p) == "high"


# ---------------------------------------------------------------------------
# cross_agent_signals direct write
# ---------------------------------------------------------------------------

def test_emit_forecast_risk_signal_commits_directly():
    conn = FakeConn()
    ok = emit_forecast_risk_signal(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        confidence_final=0.55, confidence_tier="review_required",
        risk_level="high", exception_flags=["high_mape"],
        mape_30d=0.20, forecast_30d_selected=300.0, selected_quantile=0.90,
    )
    assert ok is True
    assert conn.committed == 1
    sql, args = conn._cur.executed[0]
    assert "INSERT INTO stage9.cross_agent_signals" in sql
    assert "forecast_risk" in sql
    # confidence is at args[5]
    assert args[5] == pytest.approx(0.55)


def test_emit_forecast_risk_signal_retries():
    conn = FakeConn(fail_writes=2)
    ok = emit_forecast_risk_signal(
        conn, tenant_id="t1", sku_id="s1", run_id="r1",
        confidence_final=0.55, confidence_tier="review_required",
        risk_level="high", exception_flags=[],
        mape_30d=0.10, forecast_30d_selected=300.0, selected_quantile=0.90,
        retry_delay_seconds=0.0,
    )
    assert ok is True
    assert len(conn._cur.executed) == 3
    assert conn.committed == 1


# ---------------------------------------------------------------------------
# prefetch_calibration_gaps
# ---------------------------------------------------------------------------

def test_prefetch_calibration_gaps_prefers_horizon_30():
    rows = [
        ("seasonal", "Prophet", 0.05, 30),
        ("seasonal", "Prophet", 0.20, 90),  # bigger horizon, but 30 wins
        ("stable",   "SES",     0.08, 60),  # only one row → wins
    ]
    conn = FakeConn(select_rows=rows)
    cache = prefetch_calibration_gaps(conn, "t1")
    assert cache == {("seasonal", "Prophet"): 0.05, ("stable", "SES"): 0.08}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _input(sku_id="s1", **overrides):
    base = dict(
        sku_id=sku_id,
        assigned_model="Prophet",
        pattern_label="seasonal",
        selected_quantile=0.90,
        df=_df([10.0] * 90),
        backtest_mape=0.10,
        exception_flags=[],
        stage8_confidence=0.80,
    )
    base.update(overrides)
    return SkuForecastInput(**base)


def test_orchestrator_writes_forecasts_and_emits_signal():
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=1)

    payload = _input()
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={payload.sku_id: ForecastContext()},
        signal_conn=sig,
    )
    r = results["s1"]
    # One forecasts row was queued + flushed (batch_size=1).
    fc_rows = [e for e in main._cur.executed if "forecasts" in e[0]]
    assert len(fc_rows) == 1
    # One forecast_risk signal hit signal_conn (NOT the main conn).
    sig_rows = [e for e in sig._cur.executed if "cross_agent_signals" in e[0]]
    assert len(sig_rows) == 1
    main_sig = [e for e in main._cur.executed if "cross_agent_signals" in e[0]]
    assert main_sig == []
    # 0.80 (base) * (1 - 0.10) (mape) = 0.72 — above 0.70 gate so 'forecasted',
    # below 0.85 so risk='medium'.
    assert r.status == "forecasted"
    assert r.risk_level == "medium"


def test_orchestrator_failure_falls_back_to_needs_acknowledgment():
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)

    def broken_fn(model, df, hs):
        raise RuntimeError("model exploded")

    payload = _input()
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=broken_fn,
        batch_writer=bw,
        contexts={payload.sku_id: ForecastContext()},
        signal_conn=sig,
    )
    r = results["s1"]
    assert r.status == "needs_acknowledgment"
    assert "forecast_failed" in r.exception_flags
    assert r.confidence_final == pytest.approx(0.30)  # floor


def test_orchestrator_uses_calibration_gap_cache():
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)
    cache = {("seasonal", "Prophet"): 0.20}  # over-confident

    payload = _input()
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={payload.sku_id: ForecastContext()},
        signal_conn=sig,
        calibration_gaps=cache,
    )
    # 0.72 * 0.90 (overconfidence) = 0.648
    assert results["s1"].confidence_final == pytest.approx(0.72 * 0.90)


def test_orchestrator_uses_dedicated_signal_conn():
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)

    run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[_input()], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={"s1": ForecastContext()},
        signal_conn=sig,
    )
    # Signal commit happened on signal_conn, not main.
    assert sig.committed == 1


# ---------------------------------------------------------------------------
# Parallel orchestrator
# ---------------------------------------------------------------------------

class _SyncExecutor:
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


def test_parallel_orchestrator_dispatches_chunks_and_closes_conns():
    payloads = [_input(f"sku-{i}") for i in range(6)]
    contexts = {p.sku_id: ForecastContext() for p in payloads}
    opened: list[FakeConn] = []

    def cf():
        c = FakeConn(select_rows=[])
        opened.append(c)
        return c

    def sf():
        c = FakeConn()
        opened.append(c)
        return c

    results = run_substage_95_parallel(
        tenant_id="t1", run_id="r1",
        skus=payloads, contexts=contexts, params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        connect_fn=cf, signal_connect_fn=sf,
        max_workers=3, executor_factory=_SyncExecutor,
    )
    assert set(results) == {f"sku-{i}" for i in range(6)}
    assert len(opened) == 6  # 3 workers × 2 conns
    for c in opened:
        assert c.closed is True


def test_parallel_orchestrator_empty_skus_short_circuits():
    out = run_substage_95_parallel(
        tenant_id="t1", run_id="r1", skus=[], contexts={},
        params=_params(),
        forecast_fn=linear_10_per_day,
        connect_fn=FakeConn, signal_connect_fn=FakeConn,
        executor_factory=_SyncExecutor,
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Done-When acceptance
# ---------------------------------------------------------------------------

def test_done_when_all_8_horizons_in_forecasts_row():
    """Every forecasts row must carry forecast_7d through forecast_365d."""
    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=1)
    run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[_input()], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={"s1": ForecastContext()},
        signal_conn=FakeConn(),
    )
    # The BatchWriter recorded one INSERT INTO forecasts(...) call;
    # the column list must include all 8 horizons.
    sql_calls = [e[0] for e in main._cur.executed if "forecasts" in e[0]]
    assert sql_calls
    cols = sql_calls[0]
    for h in HORIZONS:
        assert f"forecast_{h}d" in cols


def test_done_when_p50_le_p80_le_p90_for_every_horizon():
    """Quantile monotonicity must hold post-bootstrap."""
    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=1)
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[_input()], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={"s1": ForecastContext()},
        signal_conn=FakeConn(),
    )
    for q in results["s1"].forecasts.values():
        assert q["p50"] <= q["p80"] <= q["p90"]


def test_done_when_high_mape_drives_needs_acknowledgment():
    """Status logic: high_mape always wins over a healthy confidence."""
    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=1)
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[_input(exception_flags=["high_mape"])], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={"s1": ForecastContext()},
        signal_conn=FakeConn(),
    )
    assert results["s1"].status == "needs_acknowledgment"


def test_done_when_signal_emitted_per_sku():
    """One forecast_risk signal per SKU, written directly + committed."""
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)
    run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[_input(f"sku-{i}") for i in range(3)],
        params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={f"sku-{i}": ForecastContext() for i in range(3)},
        signal_conn=sig,
    )
    sig_rows = [e for e in sig._cur.executed if "cross_agent_signals" in e[0]]
    assert len(sig_rows) == 3
    assert sig.committed == 3


def test_done_when_risk_bands_match_spec():
    """≥0.85=low, 0.70-0.84=medium, <0.70=high."""
    p = _params()
    assert determine_risk_level(0.95, p) == "low"
    assert determine_risk_level(0.85, p) == "low"
    assert determine_risk_level(0.84999, p) == "medium"
    assert determine_risk_level(0.70, p) == "medium"
    assert determine_risk_level(0.6999, p) == "high"


# A seasonal-shaped fixture: forecast_365d outpaces forecast_30d × 12 by 11%.
# Production this comes from Prophet's "ONE fit for 365 days, extract cumulative
# sums at boundaries" strategy. Here we just hardcode the curve so we can prove
# the orchestrator passes it through unchanged (NEVER scales from 30d).
def seasonal_curve_fixture(model_name, train_df, horizons):
    points = {
        7:   70.0,
        14:  140.0,
        30:  300.0,
        60:  610.0,    # slight super-linear
        90:  920.0,
        150: 1560.0,
        180: 1900.0,
        365: 4000.0,   # 300*12*1.111 = 4000  (≥10% over linear)
    }
    return ForecastBundle(points_per_horizon=points, residuals=np.array([]))


def test_done_when_seasonal_365d_exceeds_30d_times_12_by_at_least_10_percent():
    """
    Done-When #1: a seasonal SKU's forecast_365d must be ≥110% of
    forecast_30d × 12. The orchestrator must NOT linear-scale Prophet
    output from 30d.
    """
    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=1)
    payload = _input(sku_id="seasonal-1", assigned_model="Prophet")
    payload.pattern_label = "seasonal"

    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=seasonal_curve_fixture,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={"seasonal-1": ForecastContext()},
        signal_conn=FakeConn(),
    )
    forecasts = results["seasonal-1"].forecasts
    f30 = forecasts[30]["mean"]
    f365 = forecasts[365]["mean"]
    linear = f30 * 12
    ratio = f365 / linear
    assert ratio >= 1.10, (
        f"forecast_365d ({f365}) must be ≥110% of forecast_30d × 12 "
        f"({linear}); got ratio={ratio:.3f}"
    )


def test_done_when_all_8_jsonb_values_populated_no_nulls():
    """
    Done-When #2: every forecast_Xd JSONB column must hold a populated
    dict — never None, never empty. Inspect the row queued to BatchWriter.
    """
    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=10)
    run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[_input()], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw,
        contexts={"s1": ForecastContext()},
        signal_conn=FakeConn(),
    )
    # The pending buffer (not yet flushed) holds the raw row dict.
    rows = bw._buffer["forecasts"]
    assert len(rows) == 1
    row = rows[0]
    for h in HORIZONS:
        col = f"forecast_{h}d"
        assert col in row, f"missing column {col}"
        val = row[col]
        # _jsonb returns the dict directly when psycopg2 isn't loaded
        # (test path), so val is either a dict or a Json wrapper.
        inner = val.adapted if hasattr(val, "adapted") else val
        assert inner is not None, f"{col} is NULL"
        assert isinstance(inner, dict) and inner, f"{col} is empty"
        assert {"mean", "p50", "p80", "p90"}.issubset(inner.keys())


def test_done_when_p50_le_p80_le_p90_across_many_rows():
    """
    Done-When #3: monotonicity must hold IN EVERY ROW. Sweep 25 SKUs
    with a noisy bootstrap that intentionally returns swapped quantiles —
    _enforce_quantile_monotonicity must repair every one.
    """
    rng = np.random.default_rng(42)

    def noisy_bootstrap(point, residuals, pattern):
        # Return three quantiles in random (often wrong) order.
        vals = sorted([point * (0.95 + rng.random() * 0.3) for _ in range(3)])
        rng.shuffle(vals)
        return {
            "mean": float(point),
            "p50": float(vals[0]),
            "p80": float(vals[1]),
            "p90": float(vals[2]),
        }

    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=100)
    payloads = [_input(f"sku-{i}") for i in range(25)]
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=payloads, params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=noisy_bootstrap,
        batch_writer=bw,
        contexts={p.sku_id: ForecastContext() for p in payloads},
        signal_conn=FakeConn(),
    )
    violations = []
    for sku_id, r in results.items():
        for h, q in r.forecasts.items():
            if not (q["p50"] <= q["p80"] <= q["p90"]):
                violations.append((sku_id, h, q))
    assert violations == [], f"{len(violations)} monotonicity violations"


def test_done_when_watchlist_wins_regardless_of_confidence():
    """
    Done-When #4: on_watchlist=True forces status='watchlist_review' for
    EVERY confidence level — not just when other flags fire.
    """
    p = _params()
    ctx_watch = ForecastContext(on_watchlist=True)
    ctx_normal = ForecastContext(on_watchlist=False)

    # Sweep low / mid / high confidence; with watchlist on, every one
    # must produce 'watchlist_review'. With watchlist off, the normal
    # gate logic applies.
    for conf in (0.95, 0.80, 0.65, 0.50, 0.30):
        assert determine_status(conf, [], ctx_watch, p) == "watchlist_review"
    # Sanity check that the normal path differs at the gate.
    assert determine_status(0.95, [], ctx_normal, p) == "forecasted"
    assert determine_status(0.50, [], ctx_normal, p) == "needs_acknowledgment"


def test_done_when_watchlist_wins_even_with_high_mape():
    """
    Watchlist must beat high_mape too — first-match-wins ordering.
    """
    ctx = ForecastContext(on_watchlist=True)
    s = determine_status(0.95, ["high_mape"], ctx, _params())
    assert s == "watchlist_review"


# ---------------------------------------------------------------------------
# Deep-review fix coverage — new contracts after the audit
# ---------------------------------------------------------------------------

def test_failed_sku_still_writes_forecasts_row():
    """T15 P1-1: Stage 10 must NEVER see a missing forecasts row."""
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)

    def broken(model, df, hs):
        raise RuntimeError("model exploded")

    payload = SkuForecastInput(
        sku_id="boom", assigned_model="Prophet", pattern_label="seasonal",
        selected_quantile=0.90, df=_df([10.0] * 90),
        backtest_mape=0.10, exception_flags=[], stage8_confidence=0.80,
    )
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=broken, bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw, contexts={"boom": ForecastContext()},
        signal_conn=sig,
    )
    # Failed SKU still has a forecasts row queued + populated horizons.
    rows = bw._buffer["forecasts"]
    assert len(rows) == 1
    row = rows[0]
    for h in HORIZONS:
        col = f"forecast_{h}d"
        val = row[col]
        inner = val.adapted if hasattr(val, "adapted") else val
        assert isinstance(inner, dict) and inner
    assert row["status"] == "needs_acknowledgment"
    assert "forecast_failed" in results["boom"].exception_flags


def test_signal_conn_default_emits_warning():
    """T15 P1-2: defaulting signal_conn to conn must warn."""
    main = FakeConn(select_rows=[])
    bw = BatchWriter(main, batch_size=10)
    payload = SkuForecastInput(
        sku_id="s1", assigned_model="Prophet", pattern_label="seasonal",
        selected_quantile=0.90, df=_df([10.0] * 90),
        backtest_mape=0.10, exception_flags=[], stage8_confidence=0.80,
    )
    with pytest.warns(UserWarning, match="signal_conn defaulted"):
        run_substage_95(
            main, tenant_id="t1", run_id="r1",
            skus=[payload], params=_params(),
            forecast_fn=linear_10_per_day,
            bootstrap_fn=deterministic_bootstrap,
            batch_writer=bw, contexts={"s1": ForecastContext()},
        )


def test_low_risk_skus_skip_signal_emit_by_default():
    """T15 P1-3: default risk_levels_to_emit = {medium, high} — low SKUs no signal."""
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)

    # Build a SKU whose confidence lands in 'low' risk (>=0.85).
    # base 0.90 (stable) * (1 - 0.0) = 0.90 → low risk.
    payload = SkuForecastInput(
        sku_id="calm", assigned_model="SES", pattern_label="stable",
        selected_quantile=0.80, df=_df([10.0] * 90),
        backtest_mape=0.0, exception_flags=[], stage8_confidence=0.85,
    )
    results = run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw, contexts={"calm": ForecastContext()},
        signal_conn=sig,
    )
    assert results["calm"].risk_level == "low"
    sig_rows = [e for e in sig._cur.executed if "cross_agent_signals" in e[0]]
    assert sig_rows == []  # filtered out


def test_low_risk_can_be_emitted_when_explicitly_enabled():
    """Caller can override with risk_levels_to_emit={'low','medium','high'}."""
    main = FakeConn(select_rows=[])
    sig = FakeConn()
    bw = BatchWriter(main, batch_size=10)
    payload = SkuForecastInput(
        sku_id="calm", assigned_model="SES", pattern_label="stable",
        selected_quantile=0.80, df=_df([10.0] * 90),
        backtest_mape=0.0, exception_flags=[], stage8_confidence=0.85,
    )
    run_substage_95(
        main, tenant_id="t1", run_id="r1",
        skus=[payload], params=_params(),
        forecast_fn=linear_10_per_day,
        bootstrap_fn=deterministic_bootstrap,
        batch_writer=bw, contexts={"calm": ForecastContext()},
        signal_conn=sig,
        risk_levels_to_emit={"low", "medium", "high"},
    )
    sig_rows = [e for e in sig._cur.executed if "cross_agent_signals" in e[0]]
    assert len(sig_rows) == 1


def test_structural_break_penalty_uses_tenant_param():
    """T15 P1-4: STRUCTURAL_BREAK_MULT now derived from tenant param."""
    # Override the penalty to 0.30 (×0.70) and verify the formula uses it.
    p = _params(structural_break_confidence_penalty=0.30)
    ctx = ForecastContext(training_data_truncated=True)
    _, final = compute_confidence(
        pattern_label="seasonal", backtest_mape=0.10,
        exception_flags=[], calibration_gap=None, stage8_confidence=0.80,
        reorder_bias_factor=1.0, ctx=ctx, params=p,
    )
    # 0.80 * (1 - 0.10) * (1 - 0.30) = 0.504
    assert final == pytest.approx(0.504, abs=1e-3)


def test_reasonableness_check_works_below_90_days():
    """T15 P2-4: SKUs with 60d of history still get a sanity check."""
    df = _df([10.0] * 60)  # 60 days, baseline 10/day
    flags, mult = reasonableness_check(5000.0, df, _params())  # daily 166 vs band 30
    assert "forecast_unusually_high" in flags
    assert mult == pytest.approx(0.85)


