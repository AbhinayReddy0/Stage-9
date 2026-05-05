"""
tests/load_test.py — Stage 9 Load / Throughput Test
=====================================================
Runs a configurable number of SKUs through the full thread-pool pipeline
for all four non-Prophet models (Naive, SES, Holt, Croston).

Three phases, each run concurrently with ThreadPoolExecutor:

  Phase 1  raw_model   — fit() + predict_all_horizons() only
  Phase 2  sub_92      — Sub-Stage 9.2 feature engineering only
  Phase 3  full        — run_one_sku() — the complete 9.2→9.3→9.4→9.5
                         pipeline using InMemoryDB + real TenantParams
                         (52 production defaults) — same fixtures as
                         test_integration_full_run.TestRunOneSku.

Prophet / NeuralProphet are excluded — they require separate OS processes
and PyTorch/GPU overhead outside the scope of thread-pool load testing.

Usage (from M:/stage_9/code):
    python tests/load_test.py              # 5 000 SKUs, 16 threads, all phases
    python tests/load_test.py --skus 10000
    python tests/load_test.py --skus 5000 --workers 8
    python tests/load_test.py --phase raw  # Phase 1 only

Output: per-phase ASCII table with throughput, latency percentiles, error counts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import statistics as _stats
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infrastructure.constants import Pattern, THREAD_POOL_WORKERS
from models.naive import NaiveForecast
from models.ses import SESModel
from models.holt import HoltLinearTrend
from models.croston import CrostonMethod
from forecasting.feature_engg import run_feature_engineering
from infrastructure.tenant_params import TenantParams
from infrastructure.tenant_params_defaults import TENANT_LEARNING_PARAMS_DEFAULTS
from handlers.acting import run_one_sku, set_test_invariants, REQUIRED_PRELOAD_KEYS
from pipeline.dual_pool import SkuPipelineInput

# ── Model / pattern slots ─────────────────────────────────────────────────────
# Each slot: (ModelClass, model_name, pattern, n_history_days, default_hp).
# n_history_days ≥ 60 so sub_stage_93 holdout (14 days) always fires.

_SLOTS = [
    (NaiveForecast, "naive_forecast", Pattern.COLD_START,
     60, {"lag_periods": 7, "smoothing_method": "mean_7d"}),
    (SESModel, "simple_exponential_smoothing", Pattern.STABLE, 90, {"smoothing_level": 0.3}),
    (HoltLinearTrend, "holts_linear_trend", Pattern.TRENDING, 90,
     {"smoothing_level": 0.3, "smoothing_trend": 0.1, "damped_trend": True}),
    (CrostonMethod, "croston", Pattern.INTERMITTENT,
     60, {"alpha": 0.10, "interval_type": "SBA"}),
]

_N_SLOTS = len(_SLOTS)

# ── Shared real TenantParams (52 production defaults) ────────────────────────

_TENANT_ID = "load-test-tenant"
_RUN_ID = "load-test-run"

_PARAMS: TenantParams = TenantParams.from_dict(
    _TENANT_ID,
    {name: str(val) for name, val in TENANT_LEARNING_PARAMS_DEFAULTS},
)

# Wire up worker invariants once at process start so run_one_sku can access
# them without a DB round-trip — mirrors the dual_pool _init_worker pattern.
set_test_invariants(
    tenant_id=_TENANT_ID,
    tenant_params=_PARAMS.to_dict(),
    invariants={"feature_reliability": {}},
)

# ── Shared preload stubs for Phase 1 & 2 ─────────────────────────────────────

_PRELOADED_92 = {"feature_reliability": {}, "promo_decisions": {}, "feature_history": {}}
_PRELOADED_93 = {"thompson_state": {}}


class _NullBatchWriter:
    """Discards queue() calls — zero allocation overhead (Phase 1 & 2 only)."""

    def queue(self, table: str, row: dict) -> None:
        pass


_SHARED_BW = _NullBatchWriter()


# ── InMemoryDB (Phase 3 — same design as test_integration_full_run.py) ───────

class _InMemoryCursor:
    def __init__(self, db: "_InMemoryDB") -> None:
        self._db = db
        self._rows: list = []
        self.description = None
        self.rowcount: int = -1

    def execute(self, sql: str, params=None) -> None:
        upper = sql.upper().strip()
        if upper.startswith("SELECT") or upper.startswith("WITH"):
            self._rows = list(self._db._route_select(sql, params))
        else:
            self._rows = []
            self.rowcount = 1

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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _InMemoryDB:
    """Lightweight InMemoryDB — same pattern as test_integration_full_run.InMemoryDB.

    Routes SELECT queries to seeded in-memory tables; discards INSERTs/UPDATEs.
    Auto-seeded with all 52 production tenant_learning_params defaults.
    """

    _SEEDED = [
        {
            "pattern": "tenant_learning_params",
            "rows": [(name, val) for name, val in TENANT_LEARNING_PARAMS_DEFAULTS],
        },
    ]

    def _route_select(self, sql: str, params) -> list:
        lower = sql.lower()
        for entry in self._SEEDED:
            if entry["pattern"] in lower:
                return list(entry["rows"])
        return []

    def cursor(self) -> _InMemoryCursor:
        return _InMemoryCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


# ── Demand series generator ───────────────────────────────────────────────────

def _make_df(pattern: str, n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_days, dtype=float)

    if pattern == Pattern.STABLE:
        qty = rng.uniform(5, 15, n_days)
    elif pattern == Pattern.TRENDING:
        qty = t * 0.35 + 8.0 + rng.normal(0, 1.5, n_days)
    elif pattern == Pattern.COLD_START:
        qty = rng.uniform(1, 10, n_days)
    elif pattern == Pattern.INTERMITTENT:
        qty = np.zeros(n_days)
        n_events = max(2, n_days // 5)
        idx = rng.choice(n_days, size=n_events, replace=False)
        qty[idx] = rng.uniform(2, 20, n_events)
    else:
        qty = np.maximum(40 - t * 0.25 + rng.normal(0, 1, n_days), 0)

    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n_days),
        "qty": np.maximum(qty, 0.0),
    })


# ── Per-SKU context (Phase 1 & 2) ────────────────────────────────────────────

class _Ctx:
    """Minimal LearningContext for sub-stage-level phases. Not shared across threads."""
    __slots__ = (
        "sku_id", "tenant_id", "run_id", "assigned_model", "pattern_label",
        "is_b2b", "selected_features", "sample_weights",
        "baseline_mape", "lifecycle_stage", "best_hp", "validation_mape",
    )

    def __init__(self, sku_id: str, model_name: str, pattern: str) -> None:
        self.sku_id = sku_id
        self.tenant_id = _TENANT_ID
        self.run_id = _RUN_ID
        self.assigned_model = model_name
        self.pattern_label = pattern
        self.is_b2b = False
        self.selected_features = ["date", "qty"]
        self.sample_weights = None
        self.baseline_mape = 0.5
        self.lifecycle_stage = None
        self.best_hp = {}
        self.validation_mape = 1.0


# ── Phase work functions ──────────────────────────────────────────────────────

def _work_raw(i: int) -> tuple[float, str | None]:
    """Phase 1 — fit + predict_all_horizons only."""
    model_cls, _, pattern, n_days, hp = _SLOTS[i % _N_SLOTS]
    t0 = time.perf_counter()
    try:
        df = _make_df(pattern, n_days, seed=i)
        m = model_cls(hp=hp)
        m.fit(df, ["date", "qty"])
        out = m.predict_all_horizons(df, ["date", "qty"])
        assert len(out) == 8, f"expected 8 horizon keys, got {len(out)}"
        return time.perf_counter() - t0, None
    except Exception as exc:
        return time.perf_counter() - t0, type(exc).__name__


def _work_92(i: int) -> tuple[float, str | None]:
    """Phase 2 — Sub-Stage 9.2 feature engineering only, with real TenantParams."""
    model_cls, model_name, pattern, n_days, hp = _SLOTS[i % _N_SLOTS]
    t0 = time.perf_counter()
    try:
        df = _make_df(pattern, n_days, seed=i)
        ctx = _Ctx(f"sku-{i:08d}", model_name, pattern)
        model = model_cls(hp=hp)
        res = run_feature_engineering(ctx, df, model, _PRELOADED_92, _PARAMS, _SHARED_BW)
        assert res.df_train is not None, "df_train must not be None after sub_stage_92"
        assert len(res.selected_features) >= 2, "selected_features must have at least date+qty"
        return time.perf_counter() - t0, None
    except Exception as exc:
        return time.perf_counter() - t0, type(exc).__name__


def _make_sku_input(i: int) -> SkuPipelineInput:
    """Build a SkuPipelineInput for run_one_sku.

    Mirrors TestRunOneSku._make_sku_input from test_integration_full_run.py:
    real TenantParams snapshot + all REQUIRED_PRELOAD_KEYS populated.
    """
    _, model_name, pattern, n_days, _ = _SLOTS[i % _N_SLOTS]
    n_days = max(n_days, 120)  # enough history for HP tuning holdout

    demand_series = list(_make_df(pattern, n_days, seed=i)["qty"].astype(float))

    quantile_param = f"quantile_{pattern}" if f"quantile_{pattern}" in {
        n for n, _ in TENANT_LEARNING_PARAMS_DEFAULTS
    } else "quantile_stable"

    preloaded_data: dict = {
        "demand_series": demand_series,
        "promo_weights": {},
        "pattern_label": pattern,
        "lifecycle_stage": "mature",
        "assigned_model": model_name,
        "selected_quantile": float(_PARAMS.get(quantile_param)),
        "effective_max_horizon": 365,
        "learning_mode": "standard",
        "oos_adjustment_factor": 1.0,
        "reorder_bias_factor": 1.0,
        "on_watchlist": False,
        "pattern_confidence": 0.75,
        "thompson_state": {},
        "calibrated_window_days": 60,
        "calibration_gap": None,
        "tier": "full",
        "weekend_zero_ratio": 0.0,
        "criticality_tier": None,
        "parent_style_id": None,
        "tenant_params": _PARAMS.to_dict(),
        "feature_history": {},
        "feature_reliability": {},
        "promo_decisions": {},
    }
    # Populate any remaining required keys not yet set.
    for key in REQUIRED_PRELOAD_KEYS - preloaded_data.keys():
        preloaded_data[key] = None

    return SkuPipelineInput(
        sku_id=f"sku-{i:08d}",
        assigned_model=model_name,
        sku_data={},
        preloaded_data=preloaded_data,
    )


def _work_full(i: int) -> tuple[float, str | None]:
    """Phase 3 — full run_one_sku pipeline (9.2 → 9.3 → 9.4 → 9.5).

    Uses InMemoryDB + real TenantParams — same fixtures as
    test_integration_full_run.TestRunOneSku.
    """
    t0 = time.perf_counter()
    try:
        db = _InMemoryDB()
        sku_input = _make_sku_input(i)
        result = run_one_sku(sku_input, _TENANT_ID, _RUN_ID, db)

        assert "status" in result, "result missing 'status'"
        assert "confidence_final" in result, "result missing 'confidence_final'"
        assert "backtest_mape" in result, "result missing 'backtest_mape'"
        assert 0.0 <= result["confidence_final"] <= 1.0, (
            f"confidence_final {result['confidence_final']!r} out of [0, 1]"
        )
        return time.perf_counter() - t0, None
    except Exception as exc:
        return time.perf_counter() - t0, type(exc).__name__


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = max(0, int(p / 100 * len(sorted_d)) - 1)
    return sorted_d[idx]


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _print_separator(width: int = 72) -> None:
    print("-" * width)


def _report_phase(
        phase: str,
        timings: list[float],
        errors: dict[str, int],
        wall_seconds: float,
        n_skus: int,
        n_workers: int,
) -> None:
    n_err = sum(errors.values())
    n_ok = n_skus - n_err
    err_pct = 100 * n_err / n_skus if n_skus else 0
    tps = n_skus / wall_seconds if wall_seconds > 0 else 0

    _print_separator()
    print(f"  Phase : {phase}")
    print(f"  SKUs  : {n_skus:,}  |  Workers : {n_workers}  |  Wall time : {wall_seconds:.2f}s")
    print(f"  OK    : {n_ok:,}  |  Errors   : {n_err} ({err_pct:.1f}%)")
    print(f"  Throughput : {tps:,.0f} SKUs/s")
    if timings:
        print(
            f"  Latency (per SKU)  "
            f"mean={_fmt_ms(_stats.mean(timings))}  "
            f"p50={_fmt_ms(_pct(timings, 50))}  "
            f"p95={_fmt_ms(_pct(timings, 95))}  "
            f"p99={_fmt_ms(_pct(timings, 99))}  "
            f"max={_fmt_ms(max(timings))}"
        )
    if errors:
        top = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top errors : {', '.join(f'{k} x{v}' for k, v in top)}")
    _print_separator()


# ── Phase runner ──────────────────────────────────────────────────────────────

def _run_phase(label: str, work_fn, n_skus: int, n_workers: int) -> None:
    timings: list[float] = []
    errors: dict[str, int] = {}

    print(f"\nStarting phase '{label}'  ({n_skus:,} SKUs, {n_workers} workers) ...")
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(work_fn, i): i for i in range(n_skus)}
        done = 0
        report_every = max(1, n_skus // 10)

        for fut in as_completed(futs):
            elapsed, err = fut.result()
            timings.append(elapsed)
            if err:
                errors[err] = errors.get(err, 0) + 1
            done += 1
            if done % report_every == 0 or done == n_skus:
                pct = 100 * done / n_skus
                print(f"  {done:>{len(str(n_skus))}}/{n_skus} ({pct:5.1f}%)  ...")

    wall = time.perf_counter() - t_start
    _report_phase(label, timings, errors, wall, n_skus, n_workers)

    err_rate = sum(errors.values()) / n_skus if n_skus else 0
    if err_rate > 0.01:
        raise SystemExit(
            f"FAIL: phase '{label}' error rate {err_rate:.1%} exceeds 1% threshold"
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 9 load / throughput test across all thread-pool models."
    )
    parser.add_argument(
        "--skus", type=int, default=5_000,
        help="Total SKUs to process per phase (default: 5000)",
    )
    parser.add_argument(
        "--workers", type=int, default=THREAD_POOL_WORKERS,
        help=f"ThreadPoolExecutor workers (default: {THREAD_POOL_WORKERS})",
    )
    parser.add_argument(
        "--phase", choices=["all", "raw", "92", "full"], default="all",
        help="Which phase(s) to run (default: all)",
    )
    args = parser.parse_args()

    n_skus = args.skus
    n_workers = args.workers
    phase = args.phase

    print()
    print("=" * 72)
    print("  Stage 9 Load Test")
    print(f"  SKUs: {n_skus:,}  |  Workers: {n_workers}  |  Phase: {phase}")
    print(f"  Models: Naive × SES × Holt × Croston  (round-robin, {n_skus // _N_SLOTS:,} each)")
    print(f"  TenantParams: {len(_PARAMS)} production defaults")
    print("=" * 72)

    print("\nWarming up ...", end="", flush=True)
    _work_raw(0)
    _work_92(0)
    _work_full(0)
    print(" done.")

    if phase in ("all", "raw"):
        _run_phase("raw_model — fit + predict_all_horizons", _work_raw, n_skus, n_workers)

    if phase in ("all", "92"):
        _run_phase("sub_stage_92 — feature engineering", _work_92, n_skus, n_workers)

    if phase in ("all", "full"):
        _run_phase("full_pipeline — run_one_sku (9.2→9.3→9.4→9.5)", _work_full, n_skus, n_workers)

    print("\nLoad test complete.")


if __name__ == "__main__":
    main()
