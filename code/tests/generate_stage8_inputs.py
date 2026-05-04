"""
generate_stage8_inputs.py
=========================
Augments a demand CSV with data-driven Stage 8 mock inputs for use with
stage9_harness.py.

All derived values are computed from the actual demand signal so Stage 8
inputs are consistent with what Stage 8 would genuinely produce. Using
this script keeps harness accuracy numbers reliable — feeding random or
constant values would inflate or deflate MAPE results.

Usage
-----
    # CSV already uses standard column names (sku_id, date, qty)
    python generate_stage8_inputs.py --csv demand.csv --output demand_s8.csv

    # CSV uses non-standard column names
    python generate_stage8_inputs.py --csv demand.csv --output demand_s8.csv \\
        --col-sku product_id --col-date order_date --col-qty quantity

    # Override criticality threshold (default: top 10% of SKUs by volume = tier A)
    python generate_stage8_inputs.py --csv demand.csv --output demand_s8.csv \\
        --pct-critical 0.20

Flags
-----
    --csv PATH          Input demand CSV (required)
    --output PATH       Output path for augmented CSV (required)
    --pct-critical N    Fraction of SKUs (by total volume) assigned criticality tier A
                        (default: 0.10 — top 10%)
    --col-sku COL       CSV column name for SKU ID (default: sku_id)
    --col-date COL      CSV column name for date (default: date)
    --col-qty COL       CSV column name for quantity (default: qty)

Added columns
-------------
    oos_pct               Fraction of days inside consecutive zero-streaks (>=3 days)
                          Uses same STOCKOUT_MIN_ZERO_STREAK constant as Sub-Stage 9.2
    detection_confidence  Based on average OOS streak length — longer = more confident
                          Capped at 1.0 using streak_len / 60 formula
    promo_weight          <1.0 on rows where qty > 2x rolling baseline AND > baseline + 3 sigma
                          Uses same PROMO_SPIKE_RATIO and PROMO_SPIKE_Z as exception detection
    on_watchlist          True when mid-series mean shifts by >2x or CV > 1.5
    confidence_calibrated Derived from coefficient of variation: 1.0 - 0.4*CV, clamped to [0.4, 1.0]
    weekend_zero_ratio    Fraction of Saturday/Sunday rows with qty=0
    criticality_tier      'A' for top --pct-critical SKUs by total volume, 'B' for rest
    lifecycle_stage       Left blank — no forecast effect, reserved for future use

Notes
-----
- The output CSV keeps all original columns and appends the Stage 8 columns.
- If a Stage 8 column already exists in the input, it is overwritten.
- pattern_label is NOT added — supply it separately or use --default-pattern
  when running stage9_harness.py.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

# Use the same spike thresholds as Sub-Stage 9.2 promo detection and
# backtesting exception detection so derived values are consistent.
from infrastructure.constants import (
    PROMO_SPIKE_RATIO,
    PROMO_SPIKE_Z,
    ROLLING_BASELINE_DAYS,
    STOCKOUT_MIN_ZERO_STREAK,
)


def _sku_features(sku_df: pd.DataFrame) -> dict[str, Any]:
    """Compute all data-driven Stage 8 features for one SKU."""
    qty   = sku_df["qty"].values.astype(float)
    dates = sku_df["date"]
    n     = len(qty)

    # ------------------------------------------------------------------
    # 1. OOS detection — consecutive zero-streaks ≥ STOCKOUT_MIN_ZERO_STREAK
    # ------------------------------------------------------------------
    oos_mask     = np.zeros(n, dtype=bool)
    streak       = 0
    streak_start = 0
    streak_lengths: list[int] = []

    for i in range(n):
        if qty[i] == 0:
            if streak == 0:
                streak_start = i
            streak += 1
        else:
            if streak >= STOCKOUT_MIN_ZERO_STREAK:
                oos_mask[streak_start:i] = True
                streak_lengths.append(streak)
            streak = 0
    if streak >= STOCKOUT_MIN_ZERO_STREAK:
        oos_mask[streak_start:] = True
        streak_lengths.append(streak)

    oos_pct = float(oos_mask.mean())

    # Detection confidence: longer average streak → clearer OOS signal → higher confidence.
    # If no OOS detected, value is irrelevant (oos_pct=0 → factor stays 1.0).
    if oos_pct > 0 and streak_lengths:
        avg_streak = float(np.mean(streak_lengths))
        # Asymptotically approaches 0.95 as avg streak length grows (saturates ~60d).
        detection_confidence = round(min(0.95, 0.60 + avg_streak / 60.0), 3)
    else:
        detection_confidence = 0.5

    # ------------------------------------------------------------------
    # 2. Promo weight — per row
    # Spike criteria: qty > rolling_mean × PROMO_SPIKE_RATIO
    #             AND qty > rolling_mean + PROMO_SPIKE_Z × rolling_std
    # Weight = organic fraction = rolling_mean / qty, floored at 0.50.
    # ------------------------------------------------------------------
    qty_series   = pd.Series(qty)
    rolling_mean = qty_series.rolling(ROLLING_BASELINE_DAYS, min_periods=3).mean().values
    rolling_std  = qty_series.rolling(ROLLING_BASELINE_DAYS, min_periods=3).std().fillna(0).values

    promo_weights: list[float] = []
    for i in range(n):
        rm  = rolling_mean[i]
        rs  = rolling_std[i]
        q   = qty[i]
        if (
            rm is not None
            and not np.isnan(rm)
            and rm > 0
            and q > 0
            and q > rm * PROMO_SPIKE_RATIO
            and q > rm + PROMO_SPIKE_Z * rs
        ):
            promo_weights.append(round(max(0.50, min(0.95, rm / q)), 3))
        else:
            promo_weights.append(1.0)

    # ------------------------------------------------------------------
    # 3. Confidence calibrated — derived from coefficient of variation
    # Excludes OOS days so stockout zeros don't inflate the CV.
    # CV ≤ 0.3 → ~0.88,  CV = 1.0 → ~0.60,  CV ≥ 1.5 → capped at 0.55.
    # ------------------------------------------------------------------
    organic = qty[~oos_mask & (qty > 0)]
    if len(organic) >= 7:
        cv = float(np.std(organic) / np.mean(organic)) if np.mean(organic) > 0 else 1.0
        confidence_calibrated = round(max(0.55, min(0.95, 1.0 - 0.4 * cv)), 3)
    else:
        confidence_calibrated = 0.65  # insufficient data — conservative default

    # ------------------------------------------------------------------
    # 4. On watchlist — significant mid-series demand shift or extreme volatility
    # ------------------------------------------------------------------
    on_watchlist = False
    mid = n // 2
    if mid >= 14:
        first_half  = qty[:mid]
        second_half = qty[mid:]
        m1 = float(np.mean(first_half[first_half > 0])) if (first_half > 0).any() else 0.0
        m2 = float(np.mean(second_half[second_half > 0])) if (second_half > 0).any() else 0.0
        if m1 > 0 and m2 > 0:
            ratio = m2 / m1
            on_watchlist = bool(ratio > 2.0 or ratio < 0.5)
        if not on_watchlist and len(organic) >= 7:
            on_watchlist = bool(cv > 1.5)  # cv computed above

    # ------------------------------------------------------------------
    # 5. Weekend zero ratio — fraction of Sat/Sun rows with qty=0
    # ------------------------------------------------------------------
    is_weekend  = dates.dt.dayofweek >= 5
    weekend_qty = qty[is_weekend.values]
    wzr = round(float((weekend_qty == 0).mean()), 3) if len(weekend_qty) > 0 else 0.0

    return {
        "oos_pct":               round(oos_pct, 3),
        "detection_confidence":  detection_confidence,
        "promo_weights":         promo_weights,   # per-row list, same length as sku_df
        "confidence_calibrated": confidence_calibrated,
        "on_watchlist":          on_watchlist,
        "weekend_zero_ratio":    wzr,
    }


def generate(
    csv_path: str,
    output_path: str,
    pct_critical: float = 0.10,
    col_sku: str = "sku_id",
    col_date: str = "date",
    col_qty: str = "qty",
) -> None:
    df = pd.read_csv(csv_path)

    rename = {}
    if col_sku != "sku_id" and col_sku in df.columns:
        rename[col_sku] = "sku_id"
    if col_date != "date" and col_date in df.columns:
        rename[col_date] = "date"
    if col_qty != "qty" and col_qty in df.columns:
        rename[col_qty] = "qty"
    if rename:
        df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"])
    df["qty"]  = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0).clip(lower=0)
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    sku_ids = df["sku_id"].unique().tolist()

    # ------------------------------------------------------------------
    # Criticality tier — top pct_critical SKUs by total demand volume.
    # High-volume SKUs are most likely to be business-critical (tier A).
    # ------------------------------------------------------------------
    total_vol    = df.groupby("sku_id")["qty"].sum().sort_values(ascending=False)
    n_critical   = max(1, round(len(sku_ids) * pct_critical))
    critical_set = set(total_vol.index[:n_critical])

    # ------------------------------------------------------------------
    # Compute per-SKU features and build output columns
    # ------------------------------------------------------------------
    oos_pct_col        = np.zeros(len(df))
    det_conf_col       = np.full(len(df), 0.5)
    promo_weight_col   = np.ones(len(df))
    conf_cal_col       = np.full(len(df), 0.80)
    on_watchlist_col   = np.zeros(len(df), dtype=bool)
    wzr_col            = np.zeros(len(df))
    criticality_col    = np.full(len(df), "B", dtype=object)

    stats: list[dict] = []

    for sku_id in sku_ids:
        mask   = df["sku_id"] == sku_id
        idx    = np.where(mask)[0]
        sku_df = df[mask].copy()

        feat = _sku_features(sku_df)

        oos_pct_col[idx]      = feat["oos_pct"]
        det_conf_col[idx]     = feat["detection_confidence"]
        promo_weight_col[idx] = feat["promo_weights"]
        conf_cal_col[idx]     = feat["confidence_calibrated"]
        on_watchlist_col[idx] = feat["on_watchlist"]
        wzr_col[idx]          = feat["weekend_zero_ratio"]
        criticality_col[idx]  = "A" if sku_id in critical_set else "B"

        n_promo = sum(1 for w in feat["promo_weights"] if w < 1.0)
        stats.append({
            "sku_id":      sku_id,
            "oos_pct":     feat["oos_pct"],
            "conf_cal":    feat["confidence_calibrated"],
            "on_watchlist": feat["on_watchlist"],
            "promo_days":  n_promo,
            "wzr":         feat["weekend_zero_ratio"],
            "tier":        "A" if sku_id in critical_set else "B",
        })

    df["oos_pct"]               = oos_pct_col
    df["detection_confidence"]  = det_conf_col
    df["promo_weight"]          = promo_weight_col
    df["confidence_calibrated"] = conf_cal_col
    df["on_watchlist"]          = on_watchlist_col
    df["weekend_zero_ratio"]    = wzr_col
    df["criticality_tier"]      = criticality_col
    df["lifecycle_stage"]       = ""   # reserved for future use — no forecast effect

    df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_oos       = sum(1 for s in stats if s["oos_pct"] > 0)
    n_watchlist = sum(1 for s in stats if s["on_watchlist"])
    n_promo_skus = sum(1 for s in stats if s["promo_days"] > 0)
    total_promo_rows = int((promo_weight_col < 1.0).sum())

    print(f"Written {len(df):,} rows | {len(sku_ids)} SKUs -> {output_path}")
    print(f"  OOS SKUs       : {n_oos} / {len(sku_ids)}  "
          f"(avg oos_pct={np.mean([s['oos_pct'] for s in stats]):.3f})")
    print(f"  Watchlist SKUs : {n_watchlist} / {len(sku_ids)}")
    print(f"  Promo SKUs     : {n_promo_skus} / {len(sku_ids)}  "
          f"({total_promo_rows:,} promo rows)")
    print(f"  Critical (A)   : {n_critical} / {len(sku_ids)}  "
          f"(top {100*pct_critical:.0f}% by volume)")
    print(f"  Avg conf_cal   : {np.mean([s['conf_cal'] for s in stats]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment a demand CSV with data-driven Stage 8 mock inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",          required=True, help="Input demand CSV")
    parser.add_argument("--output",       required=True, help="Output augmented CSV path")
    parser.add_argument("--pct-critical", type=float, default=0.10,
                        help="Fraction of SKUs assigned criticality tier A "
                             "by total volume (default 0.10)")
    parser.add_argument("--col-sku",  default="sku_id",   metavar="COL",
                        help="Source column name for SKU ID (default: sku_id)")
    parser.add_argument("--col-date", default="date",     metavar="COL",
                        help="Source column name for date (default: date)")
    parser.add_argument("--col-qty",  default="qty",      metavar="COL",
                        help="Source column name for quantity (default: qty)")
    args = parser.parse_args()

    try:
        generate(
            csv_path=args.csv,
            output_path=args.output,
            pct_critical=args.pct_critical,
            col_sku=args.col_sku,
            col_date=args.col_date,
            col_qty=args.col_qty,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
