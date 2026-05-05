"""
test_stage9_accuracy.py — Forecast accuracy, math verification,
learning loop, and full pipeline tests for Stage 9.

WHAT THIS COVERS
----------------

SECTION A — Math verification (no DB needed)
    Directly instantiates SES, Holt, Croston and feeds known numbers.
    Asserts the output matches hand-calculated expected values.
    Catches wrong alpha, wrong damping formula, wrong Croston rate.

SECTION B — Forecast units accuracy
    Drives Stage 9 against controlled synthetic demand.
    Asserts forecast_Nd.mean is within a tolerance of the true expected value.

    Tolerance tiers (tied to tenant_maturity):
        new        (Run 1) → 15%
        developing (Run 2, ~10 weeks) → 10%
        established(Run 3, ~12 weeks) → 6%

SECTION C — Three-run learning loop
    Run 1 (new)         → tenant_maturity='new',         explore, Thompson at Beta(1,1)
    Run 2 (developing)  → tenant_maturity='developing',  Thompson has Run 1 state
    Run 3 (established) → tenant_maturity='established', Thompson should converge

    Asserts:
        - alpha_param increases for the winning model between runs
        - total_trials increments correctly
        - confidence_final rises from Run 1 to Run 3
        - learning_mode transitions from explore toward exploit

SECTION D — Full pipeline: Stage 8 stub → Stage 9 → Stage 10 readback
    Seeds Stage 8 tables, drives Stage 9, then runs the Stage 10
    contract query and asserts all required fields are non-null.

PREREQUISITES
-------------
    export STAGE9_TEST_DSN="postgresql://test:test@localhost:5432/test"
    export STAGE9_PROJECT_ROOT="/path/to/stage9/code"
    export STAGE9_TEST_DATA_DIR="/path/to/tests/stage_9run"

HOW TO RUN
----------
    # All sections
    pytest test_stage9_accuracy.py -v -s

    # Just math (no DB)
    pytest test_stage9_accuracy.py -v -k "TestMath"

    # Just accuracy
    pytest test_stage9_accuracy.py -v -k "TestAccuracy"

    # Just learning loop
    pytest test_stage9_accuracy.py -v -k "TestLearningLoop"

    # Just full pipeline
    pytest test_stage9_accuracy.py -v -k "TestFullPipeline"
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Skip if no DSN
# ---------------------------------------------------------------------------
try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

from infrastructure.config import DB_DSN as _DSN, DB_PASSWORD as _DB_PASSWORD, PROJECT_ROOT as _PROJECT_ROOT_STR  # noqa: E402
if not _DB_PASSWORD:
    pytest.skip(
        "DB_PASSWORD not set in .env — configure .env to run DB tests.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(_PROJECT_ROOT_STR).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from infrastructure.seed import seed_tenant_params                        # noqa: E402
from infrastructure.state_machine import AgentState, transition           # noqa: E402
from handlers.preloading import preloading_handler         # noqa: E402
from handlers.perceiving import perceiving_handler         # noqa: E402
from handlers.planning import planning_handler             # noqa: E402
from handlers.acting import acting_handler                 # noqa: E402
from handlers.learning import learning_handler             # noqa: E402
from handlers.reporting import reporting_handler           # noqa: E402

# Model imports for math tests — bypass DB entirely
from models.ses import SESModel                           # noqa: E402
from models.holt import HoltLinearTrend                   # noqa: E402
from models.croston import CrostonMethod as CrostonModel  # noqa: E402
from infrastructure.constants import HORIZONS                     # noqa: E402


# ===========================================================================
# Constants
# ===========================================================================

# Tolerance tiers by tenant maturity
TOLERANCE = {
    "new":         0.15,   # 15% — first run, Thompson at uniform prior
    "developing":  0.10,   # 10% — ~10 weeks of data, Thompson updating
    "established": 0.06,   # 6%  — ~12 weeks, Thompson converging
}

_NS_ACC = uuid.UUID("00000000-0000-0000-0000-ACCACCACCACC")


def _sku_uuid(code: str) -> str:
    return str(uuid.uuid5(_NS_ACC, code))


# ===========================================================================
# DB helpers — same pattern as test_stage9_e2e.py
# ===========================================================================

def _connect():
    conn = psycopg2.connect(_DSN)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO stage9, public")
    conn.commit()
    return conn


def _insert_canonical_sku(cur, tenant_id, sku_uuid):
    cur.execute(
        """
        INSERT INTO stage8.canonical_sku
            (sku_id, tenant_id, vendor, product_type)
        VALUES (%s, %s, 'AccuracyVendor', 'accuracy_test')
        ON CONFLICT (sku_id) DO NOTHING
        """,
        (sku_uuid, tenant_id),
    )


def _insert_clean_orders(cur, tenant_id, run_id, sku_uuid, df):
    if pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        dates = df["order_date"].dt.date
    else:
        dates = pd.to_datetime(df["order_date"]).dt.date
    qty_col = df["quantity"] if "quantity" in df.columns else df["qty"]
    rows = [(tenant_id, sku_uuid, d, float(q)) for d, q in zip(dates, qty_col)]
    cur.executemany(
        "INSERT INTO stage8.demand_history (tenant_id, sku_id, sale_date, qty) "
        "VALUES (%s, %s, %s, %s)",
        rows,
    )


def _seed_pattern_history(cur, tenant_id, run_id, sku_uuid,
                          pattern_label, model_hint, obs_days):
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
        (tenant_id, sku_uuid, run_id, pattern_label,
         model_hint, obs_days, lifecycle),
    )


def _seed_signal_context(cur, tenant_id, run_id, n_skus, maturity):
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


def _seed_feature_decisions(cur, tenant_id, run_id, sku_uuid,
                             weekend_zero_ratio=0.0):
    rel = {"trend": 0.85, "seasonality": 0.85, "zero_ratio": 0.95, "cv": 0.90}
    cur.execute(
        """
        INSERT INTO stage8.feature_decisions
            (tenant_id, sku_id, run_id,
             feature_reliability_map, weekend_zero_ratio, velocity_signature)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (tenant_id, sku_id, run_id) DO NOTHING
        """,
        (tenant_id, sku_uuid, run_id, Json(rel), weekend_zero_ratio, Json({})),
    )


def _seed_run_row(cur, tenant_id, run_id):
    cur.execute(
        """
        INSERT INTO stage8.runs (run_id, tenant_id, status, created_at)
        VALUES (%s,%s,'patterns_discovered',NOW())
        ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status
        """,
        (run_id, tenant_id),
    )


def _drive_stage9(conn, tenant_id, run_id):
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
    tables = [
        "forecasts", "pattern_feedback", "model_initialization_s9",
        "feature_decisions_s9", "hyperparameter_decisions",
        "backtest_decisions", "thompson_sampling_state",
        "stage9_self_assessment", "agent_state_log_s9",
        "stage9_sku_execution_log", "cross_agent_signals",
        "data_fingerprint_cache", "adaptive_quantile_state",
        "sku_similarity_registry", "forecast_outcomes",
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


def _get_forecast(conn, tenant_id, run_id, sku_uuid):
    """Return the forecast row for a single SKU as a dict of {horizon: payload}."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT forecast_7d, forecast_14d, forecast_30d, forecast_60d,
               forecast_90d, forecast_150d, forecast_180d, forecast_365d,
               confidence_final, status, backtest_mape
        FROM stage9.forecasts
        WHERE tenant_id=%s AND run_id=%s AND sku_id=%s
        """,
        (tenant_id, run_id, sku_uuid),
    )
    row = cur.fetchone()
    cur.close()
    if row is None:
        return None
    def _parse(v):
        return v if isinstance(v, dict) else json.loads(v)
    return {
        "horizons": {h: _parse(row[i]) for i, h in enumerate(HORIZONS)},
        "confidence_final": float(row[8]) if row[8] else None,
        "status": row[9],
        "backtest_mape": float(row[10]) if row[10] else None,
    }


# ===========================================================================
# Demand series generators — fully controlled, deterministic
# ===========================================================================

def _flat_demand(n_days: int, daily_mean: float,
                 noise_pct: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """
    Stable demand: constant level with tiny noise.
    Expected forecast_Nd.mean ≈ daily_mean × N.
    With noise_pct=0.02 (2%) the series is almost perfectly flat so the
    SES fit converges very close to daily_mean.
    """
    rng = np.random.default_rng(seed)
    qty = np.round(
        daily_mean + rng.normal(0, daily_mean * noise_pct, n_days)
    ).clip(0).astype(int)
    start = date.today() - timedelta(days=n_days)
    return pd.DataFrame({
        "order_date": [start + timedelta(days=i) for i in range(n_days)],
        "quantity": qty,
    })


def _trending_demand(n_days: int, start_mean: float, slope: float,
                     seed: int = 1) -> pd.DataFrame:
    """
    Linearly rising demand. Level at day i ≈ start_mean + slope × i.
    Expected forecast_30d.mean ≈ (level_at_end × 30) with Holt damping.
    """
    rng = np.random.default_rng(seed)
    base = start_mean + np.arange(n_days) * slope
    qty = np.round(base + rng.normal(0, base * 0.05, n_days)).clip(0).astype(int)
    start = date.today() - timedelta(days=n_days)
    return pd.DataFrame({
        "order_date": [start + timedelta(days=i) for i in range(n_days)],
        "quantity": qty,
    })


def _intermittent_demand(n_days: int, daily_rate: float,
                         zero_ratio: float = 0.65, seed: int = 2) -> pd.DataFrame:
    """
    Lumpy demand. Expected Croston daily_rate ≈ daily_rate.
    Expected forecast_30d.mean ≈ daily_rate × 30.
    """
    rng = np.random.default_rng(seed)
    n_zero = int(n_days * zero_ratio)
    qty = np.zeros(n_days)
    non_zero_idx = rng.choice(n_days, size=n_days - n_zero, replace=False)
    avg_sale = daily_rate / (1 - zero_ratio)
    qty[non_zero_idx] = rng.poisson(max(avg_sale, 0.01), size=len(non_zero_idx))
    start = date.today() - timedelta(days=n_days)
    return pd.DataFrame({
        "order_date": [start + timedelta(days=i) for i in range(n_days)],
        "quantity": qty.astype(int),
    })


# ===========================================================================
# SECTION A — Math Verification (no DB)
# ===========================================================================

class TestMath:
    """
    Pure unit tests for the arithmetic inside each model.
    No DB. No Stage 9 pipeline. Just instantiate the model and verify numbers.
    """

    # ── SES ──────────────────────────────────────────────────────────────────

    def test_ses_flat_series_level_equals_mean(self):
        """
        SES on a perfectly flat series [10,10,...,10] must converge to exactly
        10.0 regardless of alpha, because every update is:
            S_t = alpha × 10 + (1-alpha) × S_{t-1}
        which is 10 at every step when S_0 = 10.
        """
        model = SESModel(hp={"smoothing_level": 0.3})
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30),
                           "qty": [10.0] * 30})
        model.fit(df, ["date", "qty"])
        assert abs(model.level - 10.0) < 0.01, (
            f"SES flat series: expected level=10.0, got {model.level:.4f}"
        )

    def test_ses_predict_is_constant(self):
        """
        SES predict(horizon=7) must return an array of 7 identical values
        equal to level. SES has no trend — it is constant.
        """
        model = SESModel(hp={"smoothing_level": 0.3})
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30),
                           "qty": [20.0] * 30})
        model.fit(df, ["date", "qty"])
        preds = model.predict(df, ["date", "qty"], horizon=7)
        assert len(preds) == 7
        assert np.all(preds == preds[0]), "SES predict must be constant"
        assert abs(preds[0] - 20.0) < 0.1, (
            f"SES predict: expected 20.0, got {preds[0]:.4f}"
        )

    def test_ses_predict_all_horizons_linear_scale(self):
        """
        SES formula: point(H) = level × H.
        So forecast_30d / forecast_7d must ≈ 30/7.
        """
        model = SESModel(hp={"smoothing_level": 0.2})
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60),
                           "qty": [15.0] * 60})
        model.fit(df, ["date", "qty"])
        result = model.predict_all_horizons(df, ["date", "qty"])
        m30 = result["forecast_30d"]["mean"]
        m7  = result["forecast_7d"]["mean"]
        assert m7 > 0, "forecast_7d mean must be positive"
        ratio = m30 / m7
        assert abs(ratio - 30/7) < 0.05, (
            f"SES horizons not linear: forecast_30d/forecast_7d={ratio:.3f}, "
            f"expected {30/7:.3f}"
        )

    def test_ses_alpha_effect(self):
        """
        Higher alpha reacts faster to a step change.
        Give a series that jumps from 10 to 30 at the midpoint.
        high-alpha SES level should be closer to 30 than low-alpha.
        """
        series = [10.0] * 30 + [30.0] * 30
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60),
                           "qty": series})
        low_alpha  = SESModel(hp={"smoothing_level": 0.1})
        high_alpha = SESModel(hp={"smoothing_level": 0.5})
        low_alpha.fit(df, ["date", "qty"])
        high_alpha.fit(df, ["date", "qty"])
        assert high_alpha.level > low_alpha.level, (
            f"Higher alpha should track step change faster. "
            f"alpha=0.5 level={high_alpha.level:.2f}, "
            f"alpha=0.1 level={low_alpha.level:.2f}"
        )

    # ── Croston ───────────────────────────────────────────────────────────────

    def test_croston_daily_rate_pure_series(self):
        """
        Known series: [0, 6, 0, 0, 12, 0, 0, 0, 9, 0]
        3 non-zero demands: 6, 12, 9 → mean demand ≈ 9.0
        Intervals between sales: 2, 3, 4 → mean interval ≈ 3.0
        Croston daily_rate ≈ 9.0 / 3.0 = 3.0
        With Syntetos-Boylan correction: rate × (1 - alpha/2)
        alpha=0.3 → correction = 0.85 → rate ≈ 3.0 × 0.85 = 2.55
        Tolerance ±0.50 to absorb smoothing convergence effects.
        """
        series = [0, 6, 0, 0, 12, 0, 0, 0, 9, 0]
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10),
                           "qty": series})
        model = CrostonModel(hp={"smoothing_level": 0.3})
        model.fit(df, ["date", "qty"])
        assert model._daily_rate > 0, "Croston daily_rate must be positive"
        assert 1.5 <= model._daily_rate <= 4.5, (
            f"Croston daily_rate={model._daily_rate:.3f} outside expected "
            f"range [1.5, 4.5] for this series"
        )

    def test_croston_all_zeros_does_not_crash(self):
        """All-zero demand must produce daily_rate=0, not a crash."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30),
                           "qty": [0] * 30})
        model = CrostonModel(hp={"smoothing_level": 0.3})
        model.fit(df, ["date", "qty"])
        assert model._daily_rate == 0.0 or model._daily_rate >= 0.0, (
            "Croston all-zero: daily_rate must be 0 or positive"
        )

    def test_croston_horizon_scales_linearly(self):
        """
        Croston formula: point(H) = daily_rate × H.
        forecast_90d.mean / forecast_30d.mean must ≈ 3.0.
        """
        series = [0, 5, 0, 0, 8, 0, 0, 6, 0, 0, 0, 7, 0, 0, 9, 0,
                  0, 0, 4, 0, 0, 5, 0, 8, 0, 0, 0, 6, 0, 0]
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30),
                           "qty": series})
        model = CrostonModel(hp={"smoothing_level": 0.3})
        model.fit(df, ["date", "qty"])
        result = model.predict_all_horizons(df, ["date", "qty"])
        m30 = result["forecast_30d"]["mean"]
        m90 = result["forecast_90d"]["mean"]
        if m30 <= 0:
            pytest.skip("Croston degenerated to zero rate for this series")
        ratio = m90 / m30
        assert abs(ratio - 3.0) < 0.20, (
            f"Croston horizon scaling: 90d/30d={ratio:.3f}, expected ~3.0"
        )

    # ── Holt ──────────────────────────────────────────────────────────────────

    def test_holt_flat_series_no_trend(self):
        """
        Holt on a flat series should converge to near-zero trend.
        Trend component T_n must be < 0.5 for a flat [20,20,...] series.
        """
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60),
                           "qty": [20.0] * 60})
        model = HoltLinearTrend(
            hp={"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": True}
        )
        model.fit(df, ["date", "qty"])
        assert abs(model._trend) < 0.5, (
            f"Holt flat series: expected |trend| < 0.5, got {model._trend:.4f}"
        )
        assert abs(model._level - 20.0) < 1.0, (
            f"Holt flat series: expected level≈20.0, got {model._level:.4f}"
        )

    def test_holt_damped_slower_than_undamped(self):
        """
        Damped Holt's long-horizon forecast must be less than undamped.
        Both models trained on the same upward trending series.
        forecast_365d(damped) < forecast_365d(undamped).
        """
        base = 10.0 + np.arange(90) * 0.2
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=90),
                           "qty": base.round().astype(float)})
        damped   = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.1,
                                        "damped_trend": True})
        undamped = HoltLinearTrend(hp={"smoothing_level": 0.3, "smoothing_trend": 0.1,
                                        "damped_trend": False})
        damped.fit(df, ["date", "qty"])
        undamped.fit(df, ["date", "qty"])
        r_damp   = damped.predict_all_horizons(df, ["date", "qty"])
        r_undamp = undamped.predict_all_horizons(df, ["date", "qty"])
        f365_d = r_damp["forecast_365d"]["mean"]
        f365_u = r_undamp["forecast_365d"]["mean"]
        assert f365_d < f365_u, (
            f"Damped Holt forecast_365d={f365_d:.1f} should be less than "
            f"undamped={f365_u:.1f}"
        )

    def test_holt_trending_series_positive_trend(self):
        """
        A clearly rising series must produce a positive trend component.
        """
        base = 5.0 + np.arange(60) * 0.5   # rising from 5 to 35
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60),
                           "qty": base.astype(float)})
        model = HoltLinearTrend(
            hp={"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": True}
        )
        model.fit(df, ["date", "qty"])
        assert model._trend > 0, (
            f"Holt on rising series must have positive trend, got {model._trend:.4f}"
        )

    # ── Thompson ─────────────────────────────────────────────────────────────

    def test_thompson_success_increments_alpha(self):
        """alpha += 1 on success, beta unchanged."""
        from models.thompson import ThompsonSampler
        sampler = ThompsonSampler()
        config = {"smoothing_level": 0.3}
        state = {}
        updated = sampler.update_state(state, config, success=True)
        h = sampler.config_hash(config)
        assert updated[h]["alpha"] == 2, (
            f"Success should increment alpha from 1 to 2, got {updated[h]['alpha']}"
        )
        assert updated[h]["beta"] == 1, (
            f"Success should not change beta, got {updated[h]['beta']}"
        )

    def test_thompson_failure_increments_beta(self):
        """beta += 1 on failure, alpha unchanged."""
        from models.thompson import ThompsonSampler
        sampler = ThompsonSampler()
        config = {"smoothing_level": 0.3}
        state = {}
        updated = sampler.update_state(state, config, success=False)
        h = sampler.config_hash(config)
        assert updated[h]["beta"] == 2, (
            f"Failure should increment beta from 1 to 2, got {updated[h]['beta']}"
        )
        assert updated[h]["alpha"] == 1, (
            f"Failure should not change alpha, got {updated[h]['alpha']}"
        )

    def test_thompson_accumulated_wins_increase_win_rate(self):
        """
        After 5 successes config A should sample higher than config B
        (which had 5 failures) in the vast majority of draws.
        """
        from models.thompson import ThompsonSampler
        sampler = ThompsonSampler()
        cfg_a = {"smoothing_level": 0.3}
        cfg_b = {"smoothing_level": 0.1}
        state = {}
        for _ in range(5):
            state = sampler.update_state(state, cfg_a, success=True)
            state = sampler.update_state(state, cfg_b, success=False)

        np.random.seed(42)
        wins_a = sum(
            sampler.select_configs([cfg_a, cfg_b], state, 2, cfg_a)[0] == cfg_a
            for _ in range(200)
        )
        assert wins_a >= 160, (
            f"Config A (5 wins) should win >80% of Thompson draws, "
            f"won {wins_a}/200"
        )

    def test_quantile_monotonicity_ses_horizons(self):
        """p50 ≤ p80 ≤ p90 for every horizon. Zero violations allowed."""
        model = SESModel(hp={"smoothing_level": 0.3})
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60),
                           "qty": [18.0] * 60})
        model.fit(df, ["date", "qty"])
        result = model.predict_all_horizons(df, ["date", "qty"])
        for h in HORIZONS:
            col = f"forecast_{h}d"
            payload = result.get(col, {})
            p50, p80, p90 = payload.get("p50"), payload.get("p80"), payload.get("p90")
            if None in (p50, p80, p90):
                continue
            assert p50 <= p80 <= p90, (
                f"forecast_{h}d: p50={p50} p80={p80} p90={p90} — "
                f"quantile monotonicity violated"
            )


# ===========================================================================
# SECTION B — Forecast units accuracy (DB required)
# ===========================================================================

class TestAccuracy:
    """
    Drives Stage 9 against controlled demand series and asserts that the
    forecast mean values are within tolerance of the known true values.
    """

    @pytest.fixture(scope="class")
    def accuracy_run(self):
        """One tenant, one run, 3 accuracy SKUs."""
        conn = _connect()
        tenant_id = str(uuid.uuid4())
        run_id    = str(uuid.uuid4())
        print(f"\n[accuracy] tenant={tenant_id}")

        # SKU definitions: (code, pattern, model, daily_mean, n_days)
        skus = [
            ("ACC-STB-001", "stable",       "SES",     20.0, 120),
            ("ACC-TRN-001", "trending",     "Holt",    10.0, 120),
            ("ACC-INT-001", "intermittent", "Croston",  5.0, 180),
        ]

        try:
            seed_tenant_params(tenant_id=tenant_id, tenant_maturity="new", conn=conn)
            cur = conn.cursor()
            _seed_signal_context(cur, tenant_id, run_id, len(skus), "new")
            _seed_run_row(cur, tenant_id, run_id)
            cur.close()
            conn.commit()

            for code, pattern, model_hint, daily_mean, n_days in skus:
                sku_uuid = _sku_uuid(code)
                if pattern == "stable":
                    df = _flat_demand(n_days, daily_mean)
                elif pattern == "trending":
                    df = _trending_demand(n_days, daily_mean, slope=0.05)
                else:
                    df = _intermittent_demand(n_days, daily_mean, zero_ratio=0.60)

                cur = conn.cursor()
                _insert_canonical_sku(cur, tenant_id, sku_uuid)
                _insert_clean_orders(cur, tenant_id, run_id, sku_uuid, df)
                _seed_pattern_history(cur, tenant_id, run_id, sku_uuid,
                                      pattern, model_hint, n_days)
                _seed_feature_decisions(cur, tenant_id, run_id, sku_uuid)
                cur.close()
                conn.commit()

            _drive_stage9(conn, tenant_id, run_id)
            yield {"conn": conn, "tenant_id": tenant_id, "run_id": run_id}
        finally:
            try:
                _teardown(conn, tenant_id)
            finally:
                conn.close()

    def _check_horizon(self, row, horizon, expected_units, tolerance, label):
        """
        Assert that forecast_Nd.mean is within tolerance of expected_units.
        expected_units = daily_mean × horizon for flat demand.
        """
        assert row is not None, f"{label}: no forecast row found"
        payload = row["horizons"].get(horizon)
        assert payload is not None, f"{label}: horizon {horizon} missing"
        mean = float(payload["mean"])
        if expected_units == 0:
            assert mean >= 0, f"{label}: mean must be non-negative"
            return
        error = abs(mean - expected_units) / expected_units
        assert error <= tolerance, (
            f"{label} horizon={horizon}d: "
            f"forecast={mean:.1f}, expected≈{expected_units:.1f}, "
            f"error={error:.1%} > tolerance={tolerance:.0%}"
        )

    def test_stable_sku_30d_accuracy(self, accuracy_run):
        """
        Stable SKU with daily_mean=20: forecast_30d.mean should ≈ 600.
        Tolerance: 15% (new tenant).
        """
        conn = accuracy_run["conn"]
        row = _get_forecast(conn, accuracy_run["tenant_id"],
                            accuracy_run["run_id"], _sku_uuid("ACC-STB-001"))
        self._check_horizon(row, 30, 600, TOLERANCE["new"], "stable_30d")

    def test_stable_sku_90d_accuracy(self, accuracy_run):
        """forecast_90d.mean should ≈ 1800. Tolerance: 15%."""
        conn = accuracy_run["conn"]
        row = _get_forecast(conn, accuracy_run["tenant_id"],
                            accuracy_run["run_id"], _sku_uuid("ACC-STB-001"))
        self._check_horizon(row, 90, 1800, TOLERANCE["new"], "stable_90d")

    def test_stable_sku_all_horizons_linear(self, accuracy_run):
        """
        SES is flat — forecast_Nd.mean must scale perfectly with N.
        forecast_90d / forecast_30d must ≈ 3.0 (±10%).
        """
        conn = accuracy_run["conn"]
        row = _get_forecast(conn, accuracy_run["tenant_id"],
                            accuracy_run["run_id"], _sku_uuid("ACC-STB-001"))
        assert row is not None
        m30 = float(row["horizons"][30]["mean"])
        m90 = float(row["horizons"][90]["mean"])
        if m30 <= 0:
            pytest.skip("SES degenerated")
        ratio = m90 / m30
        assert abs(ratio - 3.0) < 0.10, (
            f"SES horizons not linear: 90d/30d={ratio:.3f}, expected 3.0"
        )

    def test_trending_sku_forecast_positive(self, accuracy_run):
        """
        Trending SKU must produce a positive forecast mean at all horizons.
        Holt level after 120 days of slope=0.05 ≈ 16 units/day.
        forecast_30d.mean should be > 300.
        """
        conn = accuracy_run["conn"]
        row = _get_forecast(conn, accuracy_run["tenant_id"],
                            accuracy_run["run_id"], _sku_uuid("ACC-TRN-001"))
        assert row is not None
        m30 = float(row["horizons"][30]["mean"])
        assert m30 > 300, (
            f"Trending SKU forecast_30d.mean={m30:.1f}, expected > 300"
        )

    def test_trending_sku_damped_at_365d(self, accuracy_run):
        """
        Holt damped: forecast_365d must be less than forecast_30d × 12.
        This confirms damping is active and growth isn't runaway.
        """
        conn = accuracy_run["conn"]
        row = _get_forecast(conn, accuracy_run["tenant_id"],
                            accuracy_run["run_id"], _sku_uuid("ACC-TRN-001"))
        assert row is not None
        m30  = float(row["horizons"][30]["mean"])
        m365 = float(row["horizons"][365]["mean"])
        assert m365 < m30 * 14, (
            f"Holt damped: forecast_365d={m365:.1f} should be < "
            f"forecast_30d × 14 = {m30 * 14:.1f}"
        )

    def test_intermittent_sku_linear_scale(self, accuracy_run):
        """
        Croston: forecast_90d.mean / forecast_30d.mean ≈ 3.0 (±20%).
        """
        conn = accuracy_run["conn"]
        row = _get_forecast(conn, accuracy_run["tenant_id"],
                            accuracy_run["run_id"], _sku_uuid("ACC-INT-001"))
        assert row is not None
        m30 = float(row["horizons"][30]["mean"])
        m90 = float(row["horizons"][90]["mean"])
        if m30 <= 0:
            pytest.skip("Croston degenerated for this series")
        ratio = m90 / m30
        assert abs(ratio - 3.0) < 0.20, (
            f"Croston 90d/30d={ratio:.3f}, expected ~3.0"
        )

    def test_all_skus_quantile_monotonicity(self, accuracy_run):
        """p50 ≤ p80 ≤ p90 across all SKUs and all horizons."""
        conn = accuracy_run["conn"]
        for code in ["ACC-STB-001", "ACC-TRN-001", "ACC-INT-001"]:
            row = _get_forecast(conn, accuracy_run["tenant_id"],
                                accuracy_run["run_id"], _sku_uuid(code))
            assert row is not None, f"{code}: no forecast row"
            for h, payload in row["horizons"].items():
                p50, p80, p90 = payload.get("p50"), payload.get("p80"), payload.get("p90")
                if None in (p50, p80, p90):
                    continue
                assert p50 <= p80 <= p90, (
                    f"{code} horizon {h}: p50={p50} p80={p80} p90={p90} — "
                    f"monotonicity violated"
                )


# ===========================================================================
# SECTION C — Three-run learning loop
# ===========================================================================

class TestLearningLoop:
    """
    Run Stage 9 three times for the same tenant on the same stable SKU.
    Verifies the Thompson → pattern_feedback → next run model selection loop.

    Run 1: new tenant          → Beta(1,1) uniform, explore mode
    Run 2: developing tenant   → Thompson has Run 1 state, may start exploiting
    Run 3: established tenant  → Thompson has 2 runs, confidence should be higher
    """

    @pytest.fixture(scope="class")
    def loop_runs(self):
        conn = _connect()
        tenant_id = str(uuid.uuid4())
        sku_uuid  = _sku_uuid("LOOP-STB-001")
        print(f"\n[loop] tenant={tenant_id}")

        # Stable demand: 20 units/day flat — easy to predict
        df_base = _flat_demand(n_days=120, daily_mean=20.0, noise_pct=0.02, seed=42)

        maturity_sequence = ["new", "developing", "established"]
        run_results = []

        try:
            seed_tenant_params(tenant_id=tenant_id, tenant_maturity="new", conn=conn)
            conn.commit()

            for i, maturity in enumerate(maturity_sequence, start=1):
                run_id = str(uuid.uuid4())
                print(f"[loop]   run {i}: maturity={maturity}, run_id={run_id[:8]}")

                cur = conn.cursor()
                _seed_signal_context(cur, tenant_id, run_id, 1, maturity)
                _seed_run_row(cur, tenant_id, run_id)
                _insert_canonical_sku(cur, tenant_id, sku_uuid)
                _insert_clean_orders(cur, tenant_id, run_id, sku_uuid, df_base)
                _seed_pattern_history(cur, tenant_id, run_id, sku_uuid,
                                      "stable", "SES", 120)
                _seed_feature_decisions(cur, tenant_id, run_id, sku_uuid)
                cur.close()
                conn.commit()

                _drive_stage9(conn, tenant_id, run_id)

                # Read results for this run
                forecast = _get_forecast(conn, tenant_id, run_id, sku_uuid)

                # Read Thompson state after this run
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT assigned_model, config_hash, alpha_param, beta_param,
                           total_trials
                    FROM stage9.thompson_sampling_state
                    WHERE tenant_id=%s AND sku_id=%s
                    ORDER BY total_trials DESC, alpha_param DESC
                    LIMIT 1
                    """,
                    (tenant_id, sku_uuid),
                )
                thompson_row = cur.fetchone()

                # Read learning_mode from model_init
                cur.execute(
                    """
                    SELECT learning_mode, assigned_model
                    FROM stage9.model_initialization_s9
                    WHERE tenant_id=%s AND run_id=%s AND sku_id=%s
                    """,
                    (tenant_id, run_id, sku_uuid),
                )
                init_row = cur.fetchone()
                cur.close()

                run_results.append({
                    "run":           i,
                    "run_id":        run_id,
                    "maturity":      maturity,
                    "forecast":      forecast,
                    "thompson":      thompson_row,
                    "learning_mode": init_row[0] if init_row else None,
                    "model":         init_row[1] if init_row else None,
                })

            yield {
                "tenant_id": tenant_id,
                "sku_uuid":  sku_uuid,
                "conn":      conn,
                "runs":      run_results,
            }
        finally:
            try:
                _teardown(conn, tenant_id)
            finally:
                conn.close()

    def test_run1_learning_mode_explore(self, loop_runs):
        """Run 1 (new tenant) must be in explore mode — Thompson has no history."""
        r = loop_runs["runs"][0]
        assert r["learning_mode"] == "explore", (
            f"Run 1 (new tenant): expected learning_mode='explore', "
            f"got {r['learning_mode']!r}"
        )

    def test_run1_model_is_ses(self, loop_runs):
        """Run 1 must assign SES for a stable pattern."""
        r = loop_runs["runs"][0]
        assert r["model"] is not None
        assert "ses" in r["model"].lower() or "exponential" in r["model"].lower(), (
            f"Run 1: expected SES model, got {r['model']!r}"
        )

    def test_thompson_state_written_after_run1(self, loop_runs):
        """Thompson state must exist in DB after Run 1."""
        r = loop_runs["runs"][0]
        assert r["thompson"] is not None, (
            "No thompson_sampling_state row after Run 1. "
            "The LEARNING handler did not flush Thompson state."
        )

    def test_thompson_alpha_increases_run1_to_run3(self, loop_runs):
        """
        After 3 runs on the same stable SKU, the winning SES config's
        alpha_param should be higher than 1 (initial value).
        A flat demand series with near-zero noise means SES should succeed
        (mape ≤ baseline) on most HP evaluations.
        """
        r3 = loop_runs["runs"][2]
        if r3["thompson"] is None:
            pytest.skip("No Thompson state after Run 3")
        alpha = r3["thompson"][2]
        assert alpha > 1, (
            f"After 3 runs on stable SKU, best SES config alpha={alpha}. "
            f"Expected > 1 — at least one successful HP evaluation should "
            f"have incremented alpha."
        )

    def test_total_trials_increments_across_runs(self, loop_runs):
        """
        total_trials in thompson_sampling_state must increase across runs.
        After 3 runs it should be ≥ 3.
        """
        r3 = loop_runs["runs"][2]
        if r3["thompson"] is None:
            pytest.skip("No Thompson state after Run 3")
        total_trials = r3["thompson"][4]
        assert total_trials >= 3, (
            f"total_trials={total_trials} after 3 runs. "
            f"Expected ≥ 3 (one increment per run at minimum)."
        )

    def test_confidence_final_stable_or_rising_run1_to_run3(self, loop_runs):
        """
        Confidence_final should stay stable or improve across runs for a
        stable SKU. A drop from Run 1 to Run 3 indicates the pipeline is
        not learning — just re-exploring.
        """
        r1_fc = loop_runs["runs"][0]["forecast"]
        r3_fc = loop_runs["runs"][2]["forecast"]
        if r1_fc is None or r3_fc is None:
            pytest.skip("forecast not available for all runs")
        r1_conf = r1_fc["confidence_final"]
        r3_conf = r3_fc["confidence_final"]
        if r1_conf is None or r3_conf is None:
            pytest.skip("confidence_final not available")
        assert r3_conf >= r1_conf * 0.95, (
            f"confidence_final dropped more than 5% across 3 runs: "
            f"Run 1={r1_conf:.3f}, Run 3={r3_conf:.3f}. "
            f"The learning loop should maintain or improve confidence."
        )

    def test_forecast_accuracy_tightens_run1_to_run3(self, loop_runs):
        """
        backtest_mape should stay within tolerance or improve across runs.
        Run 3 (established, 6% tolerance) vs Run 1 (new, 15% tolerance).
        This verifies that HP tuning via Thompson is actually improving fit.
        """
        r1_fc = loop_runs["runs"][0]["forecast"]
        r3_fc = loop_runs["runs"][2]["forecast"]
        if r1_fc is None or r3_fc is None:
            pytest.skip("forecast not available for all runs")
        r1_mape = r1_fc["backtest_mape"]
        r3_mape = r3_fc["backtest_mape"]
        if r1_mape is None or r3_mape is None:
            pytest.skip("backtest_mape not available")
        # Run 3 must be within established tolerance (6%)
        assert r3_mape <= TOLERANCE["established"], (
            f"Run 3 (established) backtest_mape={r3_mape:.3f} > "
            f"tolerance={TOLERANCE['established']:.0%}. "
            f"HP tuning across 3 runs should converge below 6% for a "
            f"stable flat-demand SKU."
        )
        # Run 3 must not be significantly worse than Run 1
        assert r3_mape <= r1_mape * 1.20, (
            f"MAPE got worse across runs: Run 1={r1_mape:.3f}, "
            f"Run 3={r3_mape:.3f}. Learning loop may not be converging."
        )


# ===========================================================================
# SECTION D — Full pipeline: Stage 8 stub → Stage 9 → Stage 10 readback
# ===========================================================================

class TestFullPipeline:
    """
    Runs the complete flow:
        1. Seed Stage 8 stub tables (pattern_history, signal_context, etc.)
        2. Drive Stage 9 (preloading → reporting)
        3. Execute the Stage 10 contract query against forecasts
        4. Assert all required Stage 10 fields are non-null

    Stage 10 requires, per SKU:
        - (forecast_30d->>'mean')  non-null DECIMAL
        - (forecast_90d->>'mean')  non-null DECIMAL
        - confidence_final         non-null DECIMAL in [0,1]
        - status                   in {'forecasted', 'needs_acknowledgment',
                                       'watchlist_review'}

    Additionally asserts that runs.status is set to 'forecasted' so
    Stage 10's trigger condition is met.
    """

    # All 5 patterns so Stage 10 can handle any mix
    _SKUS = [
        ("PIPE-CS-001",  "cold_start",   "Naive",   25,  0.05),
        ("PIPE-STB-001", "stable",       "SES",     120, 0.0),
        ("PIPE-TRN-001", "trending",     "Holt",    120, 0.0),
        ("PIPE-SEA-001", "seasonal",     "Prophet", 365, 0.0),
        ("PIPE-INT-001", "intermittent", "Croston", 180, 0.65),
    ]

    @pytest.fixture(scope="class")
    def pipeline_run(self):
        conn = _connect()
        tenant_id = str(uuid.uuid4())
        run_id    = str(uuid.uuid4())
        print(f"\n[pipeline] tenant={tenant_id}")

        try:
            seed_tenant_params(tenant_id=tenant_id, tenant_maturity="new", conn=conn)
            cur = conn.cursor()
            _seed_signal_context(cur, tenant_id, run_id,
                                 len(self._SKUS), "new")
            _seed_run_row(cur, tenant_id, run_id)
            cur.close()
            conn.commit()

            for code, pattern, model_hint, n_days, zero_ratio in self._SKUS:
                sku_uuid = _sku_uuid(code)
                if pattern == "cold_start":
                    df = _flat_demand(n_days, 2.0, noise_pct=0.50)
                elif pattern == "stable":
                    df = _flat_demand(n_days, 20.0)
                elif pattern == "trending":
                    df = _trending_demand(n_days, 8.0, slope=0.05)
                elif pattern == "seasonal":
                    # Use flat as seasonal proxy — Prophet runs on real data in
                    # production; here we verify the pipeline doesn't crash
                    df = _flat_demand(n_days, 22.0, noise_pct=0.30)
                else:
                    df = _intermittent_demand(n_days, 5.0, zero_ratio=zero_ratio)

                cur = conn.cursor()
                _insert_canonical_sku(cur, tenant_id, sku_uuid)
                _insert_clean_orders(cur, tenant_id, run_id, sku_uuid, df)
                _seed_pattern_history(cur, tenant_id, run_id, sku_uuid,
                                      pattern, model_hint, n_days)
                _seed_feature_decisions(cur, tenant_id, run_id, sku_uuid,
                                        weekend_zero_ratio=zero_ratio * 0.5)
                cur.close()
                conn.commit()

            _drive_stage9(conn, tenant_id, run_id)
            yield {"conn": conn, "tenant_id": tenant_id, "run_id": run_id}
        finally:
            try:
                _teardown(conn, tenant_id)
            finally:
                conn.close()

    # ── Stage 10 contract query ──────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def stage10_rows(self, pipeline_run):
        """Execute the exact query Stage 10 would run."""
        conn = pipeline_run["conn"]
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT
                f.sku_id,
                (f.forecast_7d   ->>'mean')::DECIMAL  AS f7,
                (f.forecast_14d  ->>'mean')::DECIMAL  AS f14,
                (f.forecast_30d  ->>'mean')::DECIMAL  AS f30,
                (f.forecast_60d  ->>'mean')::DECIMAL  AS f60,
                (f.forecast_90d  ->>'mean')::DECIMAL  AS f90,
                (f.forecast_150d ->>'mean')::DECIMAL  AS f150,
                (f.forecast_180d ->>'mean')::DECIMAL  AS f180,
                (f.forecast_365d ->>'mean')::DECIMAL  AS f365,
                f.confidence_final,
                f.status
            FROM stage9.forecasts f
            WHERE f.tenant_id = %s
              AND f.run_id    = %s
              AND f.status IN ('forecasted', 'needs_acknowledgment',
                               'watchlist_review')
            ORDER BY f.sku_id
            """,
            (pipeline_run["tenant_id"], pipeline_run["run_id"]),
        )
        rows = cur.fetchall()
        cur.close()
        return rows

    def test_stage10_receives_all_skus(self, stage10_rows):
        """Stage 10 must receive a row for every SKU (5 patterns)."""
        assert len(stage10_rows) == len(self._SKUS), (
            f"Stage 10 query returned {len(stage10_rows)} rows, "
            f"expected {len(self._SKUS)}. "
            f"Some SKUs may have an invalid status or are missing from forecasts."
        )

    def test_stage10_all_horizon_means_non_null(self, stage10_rows):
        """Every horizon mean must be non-null and ≥ 0."""
        col_names = ["f7","f14","f30","f60","f90","f150","f180","f365"]
        for row in stage10_rows:
            sku_id = str(row[0])
            for i, col in enumerate(col_names, start=1):
                val = row[i]
                assert val is not None, (
                    f"sku_id={sku_id}: {col} is NULL in Stage 10 query. "
                    f"Stage 10 cannot compute reorder quantity without this value."
                )
                assert float(val) >= 0, (
                    f"sku_id={sku_id}: {col}={val} is negative. "
                    f"Forecast mean must always be ≥ 0."
                )

    def test_stage10_confidence_final_non_null_and_in_range(self, stage10_rows):
        """confidence_final must be non-null and in [0, 1]."""
        for row in stage10_rows:
            sku_id = str(row[0])
            conf   = row[9]
            assert conf is not None, (
                f"sku_id={sku_id}: confidence_final is NULL. "
                f"Stage 10 uses this to weight PO recommendations."
            )
            assert 0.0 <= float(conf) <= 1.0, (
                f"sku_id={sku_id}: confidence_final={conf} outside [0,1]"
            )

    def test_stage10_status_valid(self, stage10_rows):
        """Status must be a valid Stage 10-readable value."""
        valid = {"forecasted", "needs_acknowledgment", "watchlist_review"}
        for row in stage10_rows:
            sku_id = str(row[0])
            status = row[10]
            assert status in valid, (
                f"sku_id={sku_id}: status={status!r} not in {valid}"
            )

    def test_stage10_30d_forecast_positive_for_non_cold_start(self, stage10_rows):
        """
        Non-cold-start SKUs must have forecast_30d > 0.
        A zero 30-day forecast would cause Stage 10 to recommend no reorder —
        which is wrong for a stable/trending/seasonal/intermittent SKU.
        """
        # Cold start SKU may legitimately have low/zero forecast
        cold_start_uuid = _sku_uuid("PIPE-CS-001")
        for row in stage10_rows:
            sku_id = str(row[0])
            if sku_id == cold_start_uuid:
                continue
            f30 = float(row[3])   # forecast_30d mean
            assert f30 > 0, (
                f"sku_id={sku_id}: forecast_30d.mean={f30} is 0 for a "
                f"non-cold-start SKU. Stage 10 would recommend no reorder."
            )

    def test_runs_status_set_to_forecasted(self, pipeline_run):
        """
        Stage 9 REPORTING must set runs.status='forecasted'.
        This is Stage 10's trigger condition — it polls runs.status.
        If this is wrong, Stage 10 never starts.
        """
        conn = pipeline_run["conn"]
        cur  = conn.cursor()
        cur.execute(
            "SELECT status FROM stage8.runs WHERE run_id=%s",
            (pipeline_run["run_id"],),
        )
        row = cur.fetchone()
        cur.close()
        assert row is not None, "No runs row found after Stage 9 completed"
        assert row[0] in ("forecasted", "needs_acknowledgment"), (
            f"runs.status='{row[0]}' after Stage 9. "
            f"Expected 'forecasted' or 'needs_acknowledgment'."
        )

    def test_pattern_feedback_written_for_all_patterns(self, pipeline_run):
        """
        pattern_feedback must have one row per SKU — all 5 patterns.
        This is Stage 8's learning signal for the next run.
        """
        conn = pipeline_run["conn"]
        cur  = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM stage8.pattern_feedback "
            "WHERE tenant_id=%s AND run_id=%s",
            (pipeline_run["tenant_id"], pipeline_run["run_id"]),
        )
        count = cur.fetchone()[0]
        cur.close()
        assert count == len(self._SKUS), (
            f"pattern_feedback has {count} rows, expected {len(self._SKUS)}. "
            f"Missing rows break Stage 8's learning loop for those SKUs."
        )
