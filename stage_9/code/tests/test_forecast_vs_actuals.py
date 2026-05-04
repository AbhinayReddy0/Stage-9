"""
test_forecast_vs_actuals.py — Automated accuracy check.

Replicates the manual procedure in `accuracy_check.xlsx`:

  1. Load forecasted_sales.csv (Stage 9 output, 5 SKUs × 5 horizons).
  2. Load actual_sales.csv (daily real sales).
  3. Reconcile: actuals summed from forecast_run_date over N days must equal
     the actual_Nd_total in the forecast file (within rounding).
  4. MAPE per SKU per horizon must lie within the pattern-specific tolerance.
  5. Scorecard: 30d MAPE drives the PASS/WARN/FAIL verdict.

Tolerance tiers (from the xlsx):
    stable                    : PASS ≤ 6%   WARN 6–9%    FAIL > 9%
    trending / seasonal       : PASS ≤ 15%  WARN 15–22%  FAIL > 22%
    cold_start / intermittent : PASS ≤ 30%  WARN 30–45%  FAIL > 45%

Data path can be overridden via env var STAGE9_ACCURACY_DATA_DIR; defaults
to `code/tests/data/`.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = [7, 14, 30, 60, 90]
PRIMARY_HORIZON = 30  # Scorecard verdict is driven by this horizon

# (pass_max, warn_max). MAPE strictly above warn_max is FAIL.
TOLERANCE: dict[str, tuple[float, float]] = {
    "stable":       (0.06, 0.09),
    "trending":     (0.15, 0.22),
    "seasonal":     (0.15, 0.22),
    "cold_start":   (0.30, 0.45),
    "intermittent": (0.30, 0.45),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    override = os.environ.get("STAGE9_ACCURACY_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "data"


def _mape(actual: float, forecast: float) -> float:
    """MAPE = |actual − forecast| / actual. Returns 0.0 when actual is 0
    and forecast is also 0 (perfect on a zero-demand day); otherwise large
    when actual is 0 to flag the divergence."""
    if actual == 0:
        return 0.0 if forecast == 0 else 1.0
    return abs(actual - forecast) / abs(actual)


def _verdict(mape: float, pattern: str) -> str:
    pass_max, warn_max = TOLERANCE[pattern]
    if mape <= pass_max:
        return "PASS"
    if mape <= warn_max:
        return "WARN"
    return "FAIL"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def forecast_df() -> pd.DataFrame:
    path = _data_dir() / "forecasted_sales.csv"
    if not path.exists():
        pytest.skip(f"forecasted_sales.csv not found at {path}")
    df = pd.read_csv(path)
    df["forecast_run_date"] = pd.to_datetime(df["forecast_run_date"]).dt.date
    return df


@pytest.fixture(scope="module")
def actuals_df() -> pd.DataFrame:
    path = _data_dir() / "actual_sales.csv"
    if not path.exists():
        pytest.skip(f"actual_sales.csv not found at {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@pytest.fixture(scope="module")
def post_run_window_days(forecast_df, actuals_df) -> int:
    """Number of post-run-date days actually present in the actuals file.
    Horizons longer than this window can't be fairly validated because
    `actual_Nd_total` clamps to whatever days are available."""
    run_date = forecast_df["forecast_run_date"].iloc[0]
    return int((actuals_df[actuals_df["date"] >= run_date]
                .groupby("sku_id")["date"].nunique()).min())


# ---------------------------------------------------------------------------
# Section 1 — File structure & coverage
# ---------------------------------------------------------------------------

class TestFiles:

    def test_forecast_has_five_skus(self, forecast_df):
        assert len(forecast_df) == 5, (
            f"Expected 5 SKUs in forecasted_sales.csv, got {len(forecast_df)}"
        )

    def test_forecast_has_one_run_date(self, forecast_df):
        assert forecast_df["forecast_run_date"].nunique() == 1, (
            "All forecasts in this file should share one run date"
        )

    def test_actuals_cover_at_least_30_days_post_run(
        self, forecast_df, actuals_df
    ):
        run_date = forecast_df["forecast_run_date"].iloc[0]
        post_run = actuals_df[actuals_df["date"] >= run_date]
        days_per_sku = post_run.groupby("sku_id")["date"].nunique()
        for sku_id, n_days in days_per_sku.items():
            assert n_days >= 30, (
                f"{sku_id}: only {n_days} post-run actuals — need ≥30 to "
                f"validate the primary 30-day horizon."
            )

    def test_actuals_cover_all_forecast_skus(self, forecast_df, actuals_df):
        forecast_skus = set(forecast_df["sku_id"])
        actual_skus   = set(actuals_df["sku_id"])
        missing = forecast_skus - actual_skus
        assert not missing, f"Actuals missing for SKUs: {missing}"


# ---------------------------------------------------------------------------
# Section 2 — Reconciliation: daily actuals → horizon totals
# ---------------------------------------------------------------------------

class TestReconciliation:
    """The actual_Nd_total column in forecasted_sales.csv must equal the
    sum of actual_qty in actual_sales.csv from run_date for the available
    days (capped at min(N, available_days) since post-run window is finite).
    """

    @pytest.mark.parametrize("horizon", [7, 14, 30])
    def test_actual_horizon_totals_match_daily_sum(
        self, horizon, forecast_df, actuals_df
    ):
        run_date = forecast_df["forecast_run_date"].iloc[0]
        end_date = run_date + timedelta(days=horizon)
        for _, fc in forecast_df.iterrows():
            sku = fc["sku_id"]
            window = actuals_df[
                (actuals_df["sku_id"] == sku)
                & (actuals_df["date"] >= run_date)
                & (actuals_df["date"] < end_date)
            ]
            daily_sum = int(window["actual_qty"].sum())
            reported  = int(fc[f"actual_{horizon}d_total"])
            assert daily_sum == reported, (
                f"{sku} {horizon}d: daily-sum reconciliation failed "
                f"({daily_sum} from CSV vs {reported} reported)"
            )


# ---------------------------------------------------------------------------
# Section 3 — Per-SKU per-horizon MAPE tolerance
# ---------------------------------------------------------------------------

class TestMAPE:
    """For each SKU, MAPE at every horizon must not exceed the FAIL
    threshold for that pattern. Borderline (WARN-tier) values pass with
    a captured warning; only FAIL-tier values fail the assertion."""

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_mape_within_fail_threshold(
        self, horizon, forecast_df, post_run_window_days
    ):
        if horizon > post_run_window_days:
            pytest.skip(
                f"Only {post_run_window_days} days of post-run actuals "
                f"available — {horizon}d MAPE would be misleading "
                f"(actual_{horizon}d_total is clamped)."
            )
        for _, fc in forecast_df.iterrows():
            actual   = float(fc[f"actual_{horizon}d_total"])
            forecast = float(fc[f"forecast_{horizon}d_mean"])
            pattern  = fc["pattern"]
            mape     = _mape(actual, forecast)
            verdict  = _verdict(mape, pattern)
            _, fail_threshold = TOLERANCE[pattern]
            assert verdict != "FAIL", (
                f"{fc['sku_id']} ({pattern}) {horizon}d: "
                f"MAPE={mape*100:.2f}% exceeds FAIL threshold "
                f"{fail_threshold*100:.0f}% — actual={actual}, "
                f"forecast={forecast}"
            )

    def test_primary_30d_mape_per_pattern(self, forecast_df):
        """The 30-day horizon is the purchasing-decision horizon. Every
        SKU must at least meet WARN-tier here (i.e. NOT FAIL)."""
        for _, fc in forecast_df.iterrows():
            actual   = float(fc[f"actual_{PRIMARY_HORIZON}d_total"])
            forecast = float(fc[f"forecast_{PRIMARY_HORIZON}d_mean"])
            pattern  = fc["pattern"]
            mape     = _mape(actual, forecast)
            verdict  = _verdict(mape, pattern)
            assert verdict in ("PASS", "WARN"), (
                f"{fc['sku_id']} ({pattern}): 30d primary horizon FAIL — "
                f"MAPE={mape*100:.2f}% (PASS≤{TOLERANCE[pattern][0]*100:.0f}%, "
                f"WARN≤{TOLERANCE[pattern][1]*100:.0f}%)"
            )


# ---------------------------------------------------------------------------
# Section 4 — Scorecard verdicts (xlsx 🏆 Scorecard)
# ---------------------------------------------------------------------------

class TestScorecard:
    """Aggregate verdict per SKU based on the 30-day primary horizon and
    cross-checks against the worst horizon. Mirrors the human-readable
    scorecard sheet."""

    def _scorecard(self, fc) -> dict:
        pattern = fc["pattern"]
        primary_mape = _mape(
            float(fc[f"actual_{PRIMARY_HORIZON}d_total"]),
            float(fc[f"forecast_{PRIMARY_HORIZON}d_mean"]),
        )
        primary_verdict = _verdict(primary_mape, pattern)

        # Bias direction by 30d horizon: positive (actual > forecast) =
        # under-forecast → stockout risk; negative = over-forecast.
        diff = (
            float(fc[f"actual_{PRIMARY_HORIZON}d_total"])
            - float(fc[f"forecast_{PRIMARY_HORIZON}d_mean"])
        )
        if abs(diff) <= 0.05 * float(fc[f"actual_{PRIMARY_HORIZON}d_total"] or 1):
            bias = "balanced"
        elif diff > 0:
            bias = "under_forecast"
        else:
            bias = "over_forecast"

        return {
            "sku_id": fc["sku_id"],
            "pattern": pattern,
            "primary_mape": primary_mape,
            "primary_verdict": primary_verdict,
            "bias": bias,
        }

    def test_no_sku_fails_primary_horizon(self, forecast_df):
        cards = [self._scorecard(fc) for _, fc in forecast_df.iterrows()]
        fails = [c for c in cards if c["primary_verdict"] == "FAIL"]
        assert not fails, (
            "These SKUs FAIL the 30d primary horizon: "
            + ", ".join(
                f"{c['sku_id']} ({c['pattern']}, MAPE={c['primary_mape']*100:.1f}%)"
                for c in fails
            )
        )

    def test_at_least_one_sku_passes_cleanly(self, forecast_df):
        cards = [self._scorecard(fc) for _, fc in forecast_df.iterrows()]
        passes = [c for c in cards if c["primary_verdict"] == "PASS"]
        assert passes, (
            "Every SKU is at WARN or FAIL — no clean PASS. "
            "Forecast quality is degraded across all patterns."
        )


# ---------------------------------------------------------------------------
# Section 5 — Diagnostic report (always passes, prints summary)
# ---------------------------------------------------------------------------

class TestReport:

    def test_print_full_accuracy_report(self, forecast_df, capsys):
        rows = []
        for _, fc in forecast_df.iterrows():
            row = {"sku_id": fc["sku_id"], "pattern": fc["pattern"]}
            for h in HORIZONS:
                actual   = float(fc[f"actual_{h}d_total"])
                forecast = float(fc[f"forecast_{h}d_mean"])
                row[f"{h}d_mape"]    = round(_mape(actual, forecast) * 100, 2)
                row[f"{h}d_verdict"] = _verdict(_mape(actual, forecast), fc["pattern"])
            rows.append(row)
        report = pd.DataFrame(rows)
        with capsys.disabled():
            print("\n=== Per-SKU per-horizon MAPE report ===")
            print(report.to_string(index=False))
        # Always passes — this is a diagnostic print, not an assertion.
        assert True
