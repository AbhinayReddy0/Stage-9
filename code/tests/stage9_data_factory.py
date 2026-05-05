"""
stage9_data_factory.py — Synthetic SKU demand generator for Stage 9 testing.

Produces controllable daily demand series so tests can exercise:
    - pattern type    (cold_start, stable, trending, seasonal, intermittent)
    - data length     (25, 60, 90, 120, 180, 365 days, etc.)
    - severity        (low CV stable vs noisy stable; mild trend vs steep)

Design principles:
    - Deterministic via seed — re-runs produce identical data.
    - Realistic noise — uses Poisson for low-volume, normal for high-volume.
    - All non-negative integers (real sales).
    - Returns DataFrames with the same columns as the project's existing
      sku_*.csv files so the seeding helpers in test_stage9_e2e.py work
      without modification.

Usage:
    from stage9_data_factory import (
        gen_cold_start, gen_stable, gen_trending, gen_seasonal, gen_intermittent,
        ALL_SCENARIOS,
    )

    df = gen_stable(sku_id="SYN-STB-90D", n_days=90, daily_mean=20.0, cv=0.10)
    # df has columns: order_date, sku_id, product_name, quantity, price,
    #                 discount_pct, channel
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ===========================================================================
# Core generator — every pattern routes through this with different params
# ===========================================================================

def _generate(
    sku_id: str,
    n_days: int,
    *,
    daily_mean: float,
    cv: float = 0.15,
    trend_slope: float = 0.0,            # absolute units/day delta
    seasonal_weekly_amp: float = 0.0,    # 0.0..1.0 multiplier swing
    seasonal_annual_amp: float = 0.0,    # 0.0..1.0 multiplier swing
    annual_peak_doy: int = 350,          # day-of-year of the peak (350 = mid-Dec)
    zero_ratio: float = 0.0,             # fraction of random days zeroed out
    weekend_zero: bool = False,          # B2B mode: weekends always zero
    seed: int = 42,
    start_date: Optional[date] = None,
    product_name: str = "Synthetic SKU",
    price: float = 19.99,
) -> pd.DataFrame:
    """
    The single workhorse. Every named generator below calls this with a
    different parameter combination.

    Returns a DataFrame with the same columns as the project's sku_*.csv
    files: order_date, sku_id, product_name, quantity, price, discount_pct,
    channel.
    """
    rng = np.random.default_rng(seed)

    if start_date is None:
        # Anchor the END of the series to today so `obs_days` calculations
        # work the way Stage 9 expects on the day the test runs.
        start_date = date.today() - timedelta(days=n_days)

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    base = np.full(n_days, daily_mean, dtype=float)

    # -- trend -----------------------------------------------------------
    if trend_slope != 0.0:
        base = base + np.arange(n_days) * trend_slope

    # -- weekly seasonality (peak Mon–Fri, trough weekend) ---------------
    if seasonal_weekly_amp > 0.0:
        dows = np.array([d.weekday() for d in dates])    # 0=Mon ... 6=Sun
        # cosine wave with peak at Wed (dow=2), trough at Sat/Sun
        wk_factor = 1.0 + seasonal_weekly_amp * np.cos((dows - 2) * 2 * np.pi / 7)
        base = base * wk_factor

    # -- annual seasonality (peak around mid-December by default) -------
    if seasonal_annual_amp > 0.0:
        doys = np.array([d.timetuple().tm_yday for d in dates])
        # cosine with peak at annual_peak_doy
        yr_factor = 1.0 + seasonal_annual_amp * np.cos(
            (doys - annual_peak_doy) * 2 * np.pi / 365
        )
        base = base * yr_factor

    base = np.maximum(base, 0.0)

    # -- noise -----------------------------------------------------------
    # Poisson preserves integer/non-negative semantics for low volumes;
    # normal noise looks more realistic for high volumes.
    if daily_mean < 5.0:
        qty = rng.poisson(np.maximum(base, 0.01)).astype(float)
    else:
        sigma = base * cv
        qty = base + rng.normal(0.0, sigma, n_days)
        qty = np.maximum(qty, 0.0)
        qty = np.round(qty).astype(float)

    # -- weekend zeros (B2B intermittent pattern) ------------------------
    if weekend_zero:
        for i, d in enumerate(dates):
            if d.weekday() >= 5:
                qty[i] = 0.0

    # -- random zeros (intermittent, sparse demand) ---------------------
    if zero_ratio > 0.0:
        n_zero = int(n_days * zero_ratio)
        zero_idx = rng.choice(n_days, size=n_zero, replace=False)
        qty[zero_idx] = 0.0

    return pd.DataFrame({
        "order_date":   dates,
        "sku_id":       sku_id,
        "product_name": product_name,
        "quantity":     qty.astype(int),
        "price":        price,
        "discount_pct": 0,
        "channel":      "shopify",
    })


# ===========================================================================
# Named generators — one per pattern, with sensible defaults
# ===========================================================================

def gen_cold_start(sku_id: str, n_days: int = 25, seed: int = 1) -> pd.DataFrame:
    """
    New product. Short history, low volume, lots of zero days.
    Triggers cold_start by either obs_days < 60 OR total_units < 10.
    """
    return _generate(
        sku_id, n_days,
        daily_mean=2.0, cv=0.50, zero_ratio=0.50, seed=seed,
        product_name="New Lipstick Matte Red", price=18.99,
    )


def gen_stable(sku_id: str, n_days: int = 90, daily_mean: float = 18.0,
               cv: float = 0.10, seed: int = 2) -> pd.DataFrame:
    """
    Predictable daily demand. Low CV, no trend, no zeros.
    """
    return _generate(
        sku_id, n_days,
        daily_mean=daily_mean, cv=cv, seed=seed,
        product_name="Castor Oil 100ml", price=12.99,
    )


def gen_trending(sku_id: str, n_days: int = 120, daily_mean: float = 8.0,
                 trend_slope: float = 0.05, seed: int = 3) -> pd.DataFrame:
    """
    Direction is consistent (positive or negative slope).
    """
    return _generate(
        sku_id, n_days,
        daily_mean=daily_mean, cv=0.08, trend_slope=trend_slope, seed=seed,
        product_name="Vitamin C Serum 30ml", price=34.99,
    )


def gen_seasonal(sku_id: str, n_days: int = 365, daily_mean: float = 22.0,
                 weekly_amp: float = 0.35, annual_amp: float = 0.40,
                 seed: int = 4) -> pd.DataFrame:
    """
    Strong weekly + annual cycles. Default annual peak is mid-December.
    """
    return _generate(
        sku_id, n_days,
        daily_mean=daily_mean, cv=0.10,
        seasonal_weekly_amp=weekly_amp,
        seasonal_annual_amp=annual_amp,
        seed=seed,
        product_name="Castor Oil Gift Set 250ml", price=24.99,
    )


def gen_intermittent(sku_id: str, n_days: int = 180, zero_ratio: float = 0.65,
                     daily_mean: float = 6.0, seed: int = 5) -> pd.DataFrame:
    """
    Lumpy demand. High zero ratio. Sells in irregular bursts.
    """
    return _generate(
        sku_id, n_days,
        daily_mean=daily_mean, cv=0.40, zero_ratio=zero_ratio, seed=seed,
        product_name="Hydraulic Seal 25mm", price=47.50,
    )


def gen_intermittent_b2b(sku_id: str, n_days: int = 180, seed: int = 6) -> pd.DataFrame:
    """
    B2B pattern: weekday-only sales (every weekend = zero).
    Stage 8 should detect this via weekend_zero_ratio > 0.60.
    """
    return _generate(
        sku_id, n_days,
        daily_mean=8.0, cv=0.30, weekend_zero=True, seed=seed,
        product_name="Industrial Bearing 6203", price=22.50,
    )


# ===========================================================================
# Scenario registry — every test scenario in one place
# ===========================================================================

@dataclass
class Scenario:
    """One test scenario. Maps to one row in the matrix table."""
    name:               str          # human-readable, used in test ids
    sku_code:           str          # unique within a run
    pattern_label:      str          # what Stage 8 should label this as
    expected_model:     str          # what Stage 9 should pick (substring match)
    expected_quantile:  float        # what Stage 9 should pick
    obs_days:           int          # length of generated series
    df_factory:         "callable"   # () -> DataFrame
    notes:              str = ""


def _scenarios() -> list[Scenario]:
    """The complete production-like matrix."""
    s = []

    # --- pattern × length matrix (single-run) ---------------------------
    s.append(Scenario(
        name="cold_start_25d",          sku_code="SYN-CS-25D",
        pattern_label="cold_start",     expected_model="Naive",  expected_quantile=0.90,
        obs_days=25,                    df_factory=lambda: gen_cold_start("SYN-CS-25D", n_days=25),
        notes="Short history triggers cold_start regardless of signal",
    ))
    s.append(Scenario(
        name="cold_start_45d_lowvol",   sku_code="SYN-CS-45D",
        pattern_label="cold_start",     expected_model="Naive",  expected_quantile=0.90,
        obs_days=45,                    df_factory=lambda: gen_cold_start("SYN-CS-45D", n_days=45, seed=10),
        notes="< 60 days still cold_start by spec rule",
    ))

    s.append(Scenario(
        name="stable_60d",              sku_code="SYN-STB-60D",
        pattern_label="stable",         expected_model="exponential_smoothing",    expected_quantile=0.80,
        obs_days=60,                    df_factory=lambda: gen_stable("SYN-STB-60D", n_days=60, daily_mean=18.0),
    ))
    s.append(Scenario(
        name="stable_90d",              sku_code="SYN-STB-90D",
        pattern_label="stable",         expected_model="exponential_smoothing",    expected_quantile=0.80,
        obs_days=90,                    df_factory=lambda: gen_stable("SYN-STB-90D", n_days=90, daily_mean=20.0),
    ))
    s.append(Scenario(
        name="stable_365d",             sku_code="SYN-STB-365D",
        pattern_label="stable",         expected_model="exponential_smoothing",    expected_quantile=0.80,
        obs_days=365,                   df_factory=lambda: gen_stable("SYN-STB-365D", n_days=365, daily_mean=22.0),
    ))

    s.append(Scenario(
        name="trending_60d_up",         sku_code="SYN-TRN-60D-UP",
        pattern_label="trending",       expected_model="Holt",   expected_quantile=0.80,
        obs_days=60,
        df_factory=lambda: gen_trending("SYN-TRN-60D-UP", n_days=60, daily_mean=8.0, trend_slope=0.10),
    ))
    s.append(Scenario(
        name="trending_120d_up",        sku_code="SYN-TRN-120D",
        pattern_label="trending",       expected_model="Holt",   expected_quantile=0.80,
        obs_days=120,
        df_factory=lambda: gen_trending("SYN-TRN-120D", n_days=120, daily_mean=8.0, trend_slope=0.08),
    ))
    s.append(Scenario(
        name="trending_180d_down",      sku_code="SYN-TRN-180D-DN",
        pattern_label="trending",       expected_model="Holt",   expected_quantile=0.80,
        obs_days=180,
        df_factory=lambda: gen_trending("SYN-TRN-180D-DN", n_days=180, daily_mean=25.0, trend_slope=-0.05),
        notes="Negative slope — model still Holt; quantile still 0.80",
    ))

    s.append(Scenario(
        name="seasonal_180d_weekly",    sku_code="SYN-SEA-180D",
        pattern_label="seasonal",       expected_model="Prophet", expected_quantile=0.90,
        obs_days=180,
        df_factory=lambda: gen_seasonal("SYN-SEA-180D", n_days=180, weekly_amp=0.40, annual_amp=0.0),
        notes="Weekly cycle only — half a year",
    ))
    s.append(Scenario(
        name="seasonal_365d_annual",    sku_code="SYN-SEA-365D",
        pattern_label="seasonal",       expected_model="Prophet", expected_quantile=0.90,
        obs_days=365,                   df_factory=lambda: gen_seasonal("SYN-SEA-365D", n_days=365),
        notes="Full year — both weekly and December peak",
    ))

    s.append(Scenario(
        name="intermittent_90d",        sku_code="SYN-INT-90D",
        pattern_label="intermittent",   expected_model="Croston", expected_quantile=0.90,
        obs_days=90,                    df_factory=lambda: gen_intermittent("SYN-INT-90D", n_days=90, zero_ratio=0.60),
    ))
    s.append(Scenario(
        name="intermittent_180d_b2b",   sku_code="SYN-INT-B2B",
        pattern_label="intermittent",   expected_model="Croston", expected_quantile=0.90,
        obs_days=180,                   df_factory=lambda: gen_intermittent_b2b("SYN-INT-B2B", n_days=180),
        notes="B2B weekday-only — Stage 8 should detect via weekend_zero_ratio",
    ))
    s.append(Scenario(
        name="intermittent_365d_extreme", sku_code="SYN-INT-365D",
        pattern_label="intermittent",   expected_model="Croston", expected_quantile=0.90,
        obs_days=365,
        df_factory=lambda: gen_intermittent("SYN-INT-365D", n_days=365, zero_ratio=0.80, daily_mean=4.0),
        notes="80% zero days — Croston should still produce a reasonable rate",
    ))

    return s


ALL_SCENARIOS: list[Scenario] = _scenarios()
SINGLE_RUN_SCENARIOS = list(ALL_SCENARIOS)


# ===========================================================================
# CLI: dump every scenario to /tmp for visual inspection
# ===========================================================================

if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage9_synthetic"
    import os
    os.makedirs(out_dir, exist_ok=True)
    for sc in ALL_SCENARIOS:
        df = sc.df_factory()
        path = f"{out_dir}/{sc.name}.csv"
        df.to_csv(path, index=False)
        print(f"  {sc.name:<40s}  {len(df):4d} rows  {sc.expected_model:<8s}  q={sc.expected_quantile}  →  {path}")
    print(f"\nWrote {len(ALL_SCENARIOS)} scenarios.")
