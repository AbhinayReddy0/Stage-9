# Stage 9 — Complete Data Flow, Calculations & Learning Reference

> Source: STAGE_9_MASTER_SPEC.docx, STAGE_9_TECHNICAL_CONTEXT.docx,
> STAGE_9_DATABASE_CONTRACTS.docx, ER_DIAGRAM_DATA_FLOW.docx,
> STAGE9_TO_STAGE10_API_CONTRACT.docx, constants.py, all model files
> Version 3.0 | April 2026

---

## Table of Contents

1. [How Data Enters Stage 9](#1-how-data-enters-stage-9)
2. [Agent State Machine — The Outer Shell](#2-agent-state-machine--the-outer-shell)
3. [PRELOADING — Bulk Data Ingestion](#3-preloading--bulk-data-ingestion)
4. [Sub-Stage 9.1 — Model Initialisation](#4-sub-stage-91--model-initialisation)
5. [Sub-Stage 9.2 — Feature Engineering](#5-sub-stage-92--feature-engineering)
6. [Sub-Stage 9.3 — Hyperparameter Tuning (Thompson Sampling)](#6-sub-stage-93--hyperparameter-tuning-thompson-sampling)
7. [Sub-Stage 9.4 — Backtesting & Exception Detection](#7-sub-stage-94--backtesting--exception-detection)
8. [Sub-Stage 9.5 — Forecast Generation & Confidence Scoring](#8-sub-stage-95--forecast-generation--confidence-scoring)
9. [Sub-Stage 9.6 — Size Curve Distribution](#9-sub-stage-96--size-curve-distribution)
10. [The Two Data Transformation Pipelines](#10-the-two-data-transformation-pipelines)
11. [How Stage 9 Exits — Output Contract with Stage 10](#11-how-stage-9-exits--output-contract-with-stage-10)
12. [How Stage 9 Learns](#12-how-stage-9-learns)
13. [The Three Feedback Loops](#13-the-three-feedback-loops)
14. [Daily Batch Jobs — Learning Happens Overnight](#14-daily-batch-jobs--learning-happens-overnight)
15. [Key Definitions & Glossary](#15-key-definitions--glossary)

---

## 1. How Data Enters Stage 9

Stage 9 never pulls data on demand during SKU processing. **All reads happen once, upfront, in PRELOADING state.** Data arrives from three upstream sources, all via database tables — never via function calls or API calls.

### 1.1 Upstream Sources

```
Stages 1–7          Stage 8                 Stage 10 (prior runs)
──────────          ───────                 ─────────────────────
golden_table        pattern_history         cross_agent_signals
clean_orders        signal_context            └─ reorder_outcome signals
clean_inventory     channel_demand_splits
canonical_sku       oos_impact_estimates
                    feature_decisions
                    portfolio_intelligence_reports
                    promo_decisions
                    tenant_thresholds
```

### 1.2 The Trigger

Stage 9 sits idle until Stage 8 finishes. Stage 8 writes `run.status = 'patterns_discovered'` to the `runs` table. LangGraph detects this and starts Stage 9.

```
Stage 8 completes
    → writes run.status = 'patterns_discovered'
        → LangGraph detects change
            → Stage 9 starts (IDLE → PRELOADING)
```

### 1.3 What Each Source Provides

| Source Table | Owner | What Stage 9 Needs From It |
|---|---|---|
| `golden_table` | Stages 1–7 | **Primary demand series.** Daily `units_adjusted` per SKU with OOS masking already applied. Preferred over `clean_orders`. |
| `clean_orders` | Stages 1–7 | Fallback demand series when `golden_table` is unavailable for a SKU. |
| `clean_inventory` | Stages 1–7 | Daily inventory snapshots. Used in Sub-Stage 9.4 for stockout detection. |
| `canonical_sku` | Stages 1–7 | SKU metadata: `vendor`, `product_type`, `shelf_life_days`, `planned_end_date`, `criticality_tier`, `parent_style_id`, `service_level_target`, `seed_daily_demand`. |
| `pattern_history` | Stage 8 | **Primary input.** `pattern_label`, `lifecycle_stage`, `confidence_calibrated`, `model_hint`, `on_watchlist`, `observation_days`, `composite_confidence`, `drift_detected`. |
| `signal_context` | Stage 8 | Run-level context: `pipeline_mode`, `data_mode`, `tenant_maturity`, `channel_split_applied`, `total_sku_count`. |
| `channel_demand_splits` | Stage 8 | Per-SKU per-day organic vs paid demand split. Used when `split_confidence >= 0.50` to strip ad-driven demand inflation from training data. |
| `oos_impact_estimates` | Stage 8 | How many days each SKU was out-of-stock and how much demand was suppressed. Used to compute OOS adjustment factor. |
| `feature_decisions` (S8) | Stage 8 | `feature_reliability_map` per SKU — scores for each optional feature. Low-reliability features are dropped before model training. |
| `portfolio_intelligence_reports` | Stage 8 | `market_shift` and `channel_count_changed` alerts. Triggers structural break detection in Sub-Stage 9.4. |
| `promo_decisions` | Stage 8 | Per-`(sku_id, date)` promo weights. Used to cap or down-weight promo-day demand during training. |
| `tenant_thresholds` | Stage 8 | `confidence_floor`, `confidence_ceiling` — used as bounds for Stage 9's confidence output. |
| `cross_agent_signals` | Stage 10 | `reorder_outcome` signals from prior cycles — did previous forecasts lead to stockouts or overstock? Used to compute `reorder_bias_factor`. |
| `tenant_learning_params` | Stage 9 | All 41 learned parameters — every threshold, rate, and buffer. No hardcoded numbers anywhere in Stage 9 logic. |

### 1.4 Execution Modes

Before any per-SKU work begins, Stage 9 decides which execution mode to use:

| Mode | When | What Runs | Target Time |
|---|---|---|---|
| **FULL** | Daily at 2 AM tenant local time. Also on Sync Now if last full run was > 18 hours ago. | All 6 sub-stages. All learning updates. All signals emitted. | 4–25 min depending on catalog size |
| **MICRO-UPDATE** | Sync Now when last full run < 18 hours ago. | Pull new orders only. Update SES level estimates. Re-check exception flags. No model retraining. | < 15 seconds for any catalog size |

> **Example:** An inventory manager clicks Sync Now at 11 AM after the 2 AM full run. Because only 9 hours have passed, Stage 9 runs micro-update mode — it incorporates the morning's sales data without re-running Thompson Sampling, backtesting, or model training.

---

## 2. Agent State Machine — The Outer Shell

Every run passes through these states in order. Every transition is logged to `agent_state_log_s9`. There are no silent transitions.

```
IDLE
  │  trigger: run.status = 'patterns_discovered'
  ▼
PRELOADING     ← 5–7 bulk DB reads into memory dicts, fingerprinting, tier classification
  ▼
PERCEIVING     ← read signal_context, load tenant_learning_params, consume cross-agent signals
  ▼
PLANNING       ← split SKUs into ProcessPool (NeuralProphet/Prophet) and ThreadPool (all others)
  ▼
ACTING         ← run full pipeline for all SKUs concurrently, write via BatchWriter
  ▼
LEARNING       ← flush BatchWriter, update Thompson state, register converged SKUs, emit signals
  ▼
REPORTING      ← run self-assessment, write stage9_self_assessment, set run.status
  ▼
COMPLETE       ← release run lock
```

If any unrecoverable error occurs in any state: → `FAILED` → release run lock → retry on next trigger.

### Dual-Pool Concurrent Execution (PLANNING → ACTING)

Stage 9 splits SKUs into two pools that run simultaneously:

| Pool | Model Types | Workers | Timeout | Why |
|---|---|---|---|---|
| `ProcessPoolExecutor` | NeuralProphet, Prophet | 4 | 120s per SKU | Separate OS processes bypass Python GIL. PyTorch benefits from process isolation. |
| `ThreadPoolExecutor` | Naive, Croston, Holt, SES | 16 | 30s per SKU | Thread-safe. Shared memory is fine. Much faster than processes for these models. |

> **Why this matters:** Without dual-pool concurrency, a 5,000-SKU catalog with 20% seasonal SKUs would take ~40 minutes (Prophet is slow). With both pools starting simultaneously, the same catalog takes 8–12 minutes.

### SKU Processing Tiers (set in PRELOADING, used in ACTING)

Each SKU is classified using a SHA-256 fingerprint of its last 7 days of demand data, compared to the fingerprint from the previous run:

| Tier | Condition | Sub-Stages That Run | ~% of Daily SKUs |
|---|---|---|---|
| **cache** | Fingerprint identical — no demand change | Skip all sub-stages. Retrieve prior forecast. Apply micro SES update only. | ~70% |
| **partial** | 7-day average demand shifted < 5% | Refit model with updated data. Reuse prior HP config. Skip Thompson Sampling. | ~20% |
| **full** | Demand shifted ≥ 5%, or no prior record | Run complete sub-stages 9.1–9.5 (and 9.6 if applicable). | ~10% |

> **Example:** A stable SKU selling 15 units/day that sells 14 yesterday → fingerprint nearly identical → cache tier → prior forecast reused with a small SES level update. A SKU that just ran a promotion and spiked from 20 to 80 units → full tier → everything re-runs.

---

## 3. PRELOADING — Bulk Data Ingestion

**Rule: no database reads happen inside the per-SKU processing loop.** Everything is loaded once here and stored in Python dicts keyed by `sku_id`.

### What Gets Loaded

```
Read 1: pattern_history JOIN feature_decisions JOIN canonical_sku
        → pattern_label, lifecycle_stage, confidence, watchlist flag,
          feature_reliability_map, weekend_zero_ratio,
          parent_style_id, shelf_life_days, planned_end_date,
          criticality_tier, service_level_target

Read 2: oos_impact_estimates
        → oos_pct_of_history, detection_confidence, suppressed_demand_estimate

Read 3: channel_demand_splits (multi-channel tenants only)
        → daily organic_units, paid_ratio, split_confidence per SKU

Read 4: promo_decisions
        → Dict[(sku_id, date), promo_weight] — O(1) lookup during training

Read 5: portfolio_intelligence_reports
        → break alerts (market_shift, channel_count_changed)

Read 6: tenant_thresholds (Stage 8's)
        → confidence_floor, confidence_ceiling

Read 7: tenant_learning_params (Stage 9's)
        → all 41 parameters for this tenant
```

### OOS Adjustment Factor (computed here, used in Sub-Stage 9.5)

When a product was out of stock for a portion of its history, recorded sales understate true demand. The OOS factor corrects for this:

```
adjustment_factor = 1 + (oos_pct_of_history × detection_confidence)
adjustment_factor = min(adjustment_factor, 1.50)    ← hard cap at +50% uplift
```

> **Why dampen by `detection_confidence`?** If inventory data is poor (low confidence), you don't want to overcorrect. Example:
> - `oos_pct = 0.35`, `detection_confidence = 0.90` (good data) → `1 + (0.35 × 0.90) = 1.315` → 31.5% uplift
> - `oos_pct = 0.35`, `detection_confidence = 0.25` (poor data) → `1 + (0.35 × 0.25) = 1.088` → only 8.8% uplift

### Channel Split Decision (computed here)

For multi-channel tenants (e.g. Shopify + Amazon Ads), organic demand must be separated from ad-driven demand before training. Without this, cutting the ad budget looks like a demand collapse.

```
IF channel_demand_splits.split_confidence >= 0.50:
    use organic_units series for training   (channel_adjusted = TRUE)
ELSE:
    use total demand series                 (channel_adjusted = FALSE)
```

---

## 4. Sub-Stage 9.1 — Model Initialisation

**One row written to `model_initialization_s9` per SKU.** This is the decision record — every choice made here is permanent for this run.

### 4.1 Model Assignment (Locked)

The model is determined entirely by the demand pattern Stage 8 assigned. There is no model selection, no competition, no override.

| Pattern | Assigned Model | Why |
|---|---|---|
| `cold_start` | Naive Forecast | Insufficient history. Conservative flat estimate until more data accumulates. |
| `intermittent` | Croston's Method | Demand occurs in bursts with silence between. Separates demand SIZE from demand FREQUENCY. Standard averaging underestimates intermittent demand. |
| `seasonal` | NeuralProphet (primary) / Prophet (fallback if obs_days ≤ 90) | Explicitly models weekly and annual cycles. The December peak must appear in the December forecast, not be averaged away. |
| `trending` | Holt's Linear Trend | Consistently growing or declining. Extrapolates direction with damping — growth doesn't continue at full rate forever. |
| `stable` | SES — Simple Exponential Smoothing | Predictable, consistent demand. Optimal for stationary demand; the simplest model that works. |

> **Example:** A spare parts SKU that sells 0, 0, 0, 8, 0, 0, 12, 0 units across 8 days → Stage 8 labels it `intermittent` → Sub-Stage 9.1 assigns Croston's Method, no further model selection happens.

### 4.2 Quantile Selection (Priority Chain)

The quantile determines the ordering buffer — higher quantile = more conservative. The first matching rule wins:

```
Priority 1: criticality_tier = 'A'    → use 0.99 (or sku.service_level_target if set)
Priority 2: sku.service_level_target   → use that value (SKU-level override)
Priority 3: lifecycle_stage = clearance → use 0.90
Priority 4: no override               → use pattern default from tenant_learning_params
                                         (starts at 0.90 for cold/intermittent/seasonal,
                                          0.80 for trending/stable)
```

> **Example (criticality_tier):** A spare part for a production line (tier A) gets quantile 0.99 — the system orders enough to cover 99% of plausible demand scenarios. A routine commodity (tier C) uses the standard 0.80.

### 4.3 Effective Max Horizon

Forecasting past a product's usable life is waste. The horizon is capped:

```
effective_max_horizon = min(365, shelf_life_days, days_until_planned_end_date)
```

> **Example:** A food product with `shelf_life_days = 90` gets a maximum forecast horizon of 90 days — forecast_150d, forecast_180d, forecast_365d will all be capped at or derived from day 90.

### 4.4 Reorder Bias Factor

Stage 9 reads `reorder_outcome` signals written by Stage 10 from prior cycles. If this SKU's forecasts have been consistently leading to stockouts or overstock, the bias factor corrects the forecast:

```
IF stockout_count >= 2 in last 5 Stage 10 signals:
    reorder_bias_factor = 1.10    ← order 10% more

IF avg_overstock_pct > 0.30 in last 5 signals:
    reorder_bias_factor = 0.92    ← order 8% less

OTHERWISE:
    reorder_bias_factor = 1.00    ← no correction
```

This factor is applied in Sub-Stage 9.5 during confidence computation (Step 4) and effectively adjusts the final forecast conservatism.

### 4.5 Learning Mode

```
explore:  Thompson Sampling tests multiple HP configurations this run
exploit:  uses the historically best config directly (no sampling overhead)
```

Mode is determined by run count vs the tenant's maturity threshold:

| Maturity | Exploit After N Runs |
|---|---|
| new | 8 runs |
| developing | 5 runs |
| established | 3 runs |

---

## 5. Sub-Stage 9.2 — Feature Engineering

Produces the cleaned, weighted training DataFrame and the final feature list that Sub-Stage 9.3 will use for HP tuning. Runs **four steps in order**. Each step can fail independently — a step failure logs and continues; the SKU is never abandoned.

### Step 1 — Reliability Filtering

Stage 8 provides a `feature_reliability_map` scoring each optional feature (e.g., `promo_flag`, `day_of_week`, `channel_split`) from 0.0 to 1.0. Features below `feature_reliability_floor` (starts at 0.30) are dropped before training.

```
for each optional feature:
    if reliability_score < feature_reliability_floor:
        DROP feature
    else:
        KEEP feature
```

Required features (`date`, `qty`) are never dropped.

### Step 2 — B2B Mode Filter

If `weekend_zero_ratio > 0.60` (i.e., this SKU records zero sales on more than 60% of weekends), it's a B2B product — weekends aren't real demand, they're closed-office days. Training on weekends would dilute the model.

```
IF weekend_zero_ratio > 0.60:
    filter training data to weekdays only (Mon–Fri)

    IF filtered rows == 0 (weekend-only seller — edge case E006):
        disable B2B mode for this SKU, use all days
        set flag: 'b2b_mode_disabled'
```

### Step 3 — Promo-Weighted Training Data

Promotional spikes in history must not train the model to expect those spikes every cycle.

**For Prophet/NeuralProphet:** build a `sample_weights` array (lower weight on promo days).

**For all other models:** cap promo-day demand at `baseline × max_promo_multiplier`:
```
rolling_baseline = 14-day rolling mean of qty
cap = rolling_baseline × max_promo_multiplier    (starts at 3.0)
promo_day_qty = min(actual_qty, cap)
```

> **Example:** A product normally sells 50 units/day. On a promo day it sold 280 units. With `max_promo_multiplier = 3.0`, the training series sees 150 instead of 280. The model learns there was elevated demand, but not that 280 is the new baseline.

### Step 4 — Additive Feature Search

Greedy forward search: add features one at a time, keep if MAPE improves by ≥ 2%.

```
budget = 4 configs max
early_stop if MAPE < 0.08

for each candidate feature (not yet selected):
    test_mape = MAPE with this feature added
    if test_mape <= best_mape × (1 - 0.02):
        accept feature
        update best_mape
    if best_mape < 0.08:
        stop searching
```

**Output written to `feature_decisions_s9`:**
- `features_used` — final feature list
- `b2b_mode_applied` — whether weekday filter ran
- `promo_weighting_applied` — whether promo adjustment ran
- `baseline_mape` — MAPE using required features only
- `improved_mape` — MAPE after feature search

---

## 6. Sub-Stage 9.3 — Hyperparameter Tuning (Thompson Sampling)

Finds the optimal hyperparameter configuration for this SKU's assigned model. Uses Thompson Sampling — a Bayesian bandit that learns which configurations are most likely to produce low MAPE.

### What HP Spaces Look Like

Each model has a fixed set of configurations Thompson can choose from:

| Model | HP Parameters | Total Configs |
|---|---|---|
| Naive Forecast | `lag_periods` ∈ {1, 7, 14} × `smoothing_method` ∈ {last_value, mean_3d, mean_7d} | 9 |
| Croston's Method | `alpha` ∈ {0.05, 0.10, 0.20} × `interval_type` ∈ {classic, SBA, TSB} | 9 |
| NeuralProphet/Prophet | `weekly_seasonality` × `yearly_seasonality` × `seasonality_mode` × `changepoint_prior_scale` | 24 |
| Holt's Linear Trend | `smoothing_level` × `smoothing_trend` × `damped_trend` | 24 |
| SES | `smoothing_level` ∈ {0.1, 0.2, 0.3, 0.4, 0.5} | 5 |

### How Thompson Sampling Works

Each configuration has a Beta(α, β) distribution stored in `thompson_sampling_state`. α counts successes, β counts failures.

```
Each run:
  1. For each config: sample theta ~ Beta(alpha, beta)
  2. Sort configs by theta descending
  3. Test top N configs (N = thompson_exploration_budget, starts at 3)
  4. ALWAYS include the prior best config (safe bet guarantee)

After testing each config:
  5. If validation_mape <= baseline_mape × 0.98 (improved ≥ 2%):
         alpha += 1  (success)
     Else:
         beta  += 1  (failure)

Early stop: if any config achieves validation_mape < 0.10, stop testing.
```

> **Example — Thompson converging over time:**
> - Run 1: All configs have Beta(1,1). Random theta → test 3 random configs.
> - Run 5: Config A (alpha=4, beta=1) consistently wins. Beta(4,1) has mean 0.80 and almost always samples > 0.5. Config F (alpha=1, beta=4) almost never gets sampled.
> - Run 20: The system has converged — it effectively always picks Config A. Exploration nearly stops.

### Validation Split

HP testing uses the **last 14 days of training data** as the validation holdout:
```
train_split = df_train[:-14 days]
val_split   = df_train[-14 days:]
MAPE computed on val_split predictions vs actual
```

**Special case — introduction lifecycle:** If `lifecycle_stage = 'introduction'`, HP tuning is skipped entirely and the **CategoryComps pipeline** runs instead (see Section 10.1). The product has < 28 days of history; there's not enough data to meaningfully tune HPs.

### Output written to `hyperparameter_decisions`:
- `hyperparameters` — the winning HP config (JSONB)
- `validation_mape` — MAPE on the 14-day holdout
- `config_hash` — SHA-256 of the config for Thompson state lookup
- `thompson_score` — α/(α+β) for the selected config
- `early_stopped` — whether the search stopped early

---

## 7. Sub-Stage 9.4 — Backtesting & Exception Detection

Measures actual model accuracy on held-out data and detects anomalies in the demand history. The backtest MAPE is the most important accuracy signal in the entire pipeline — it flows directly to Stage 8 as `pattern_feedback`.

### 7.1 Backtest Window Selection

The window adapts per SKU's data density:

| Data Density | Condition | Window |
|---|---|---|
| `ultra_sparse` | observation_days < 14 | `min_backtest_window` (starts 14 days) |
| `sparse` | observation_days < 60 | `max(min_window, obs_days / 3)` |
| `normal` | standard | `default_backtest_window` (starts 60 days) |
| (ceiling) | established tenants | up to `max_backtest_window` (starts 90 days) |

### 7.2 The Four Exception Detectors

Run on the backtest period. Each appends a string to `exception_flags` if triggered:

| Flag | Detection Rule | What It Means |
|---|---|---|
| `stockout` | ≥ 3 consecutive zero-sale days | Product was likely unavailable. Demand is suppressed, not zero. |
| `promo_spike` | Any day > 200% of 7-day rolling baseline **OR** z-score > 3.0 | Promotional demand is inflating the history. |
| `unusual_drop` | 3 consecutive periods each < 40% of 7-day baseline (no promo) | Something structurally changed. Not explainable by a promotion. |
| `high_volatility` | Coefficient of Variation (std / mean) ≥ 1.0 | Demand is too erratic to forecast with standard confidence. |

> **Example — stockout detection:** A SKU with demand series `[20, 22, 18, 0, 0, 0, 21, 19]` — three consecutive zeros. These aren't real zero-demand days, the product was out of stock. `stockout` flag is set, confidence is penalised, and Stage 10 is warned to add a safety buffer.

### 7.3 Structural Break Detection

Only runs if `portfolio_intelligence_reports` has a `market_shift` or `channel_count_changed` alert for this SKU. Uses the **PELT algorithm** (ruptures library) to find the exact date when demand shifted to a new regime.

```
IF portfolio_intelligence alert exists for this SKU:
    run PELT algorithm on full demand history
    IF break detected AND break date is within ±14 days of alert date:
        truncate training data to post-break period only
        set structural_break_found = TRUE
        set break_date
        (confidence will be penalised × 0.85 in Sub-Stage 9.5)
```

> **Example:** A brand starts selling on Amazon in addition to Shopify. Stage 8 fires a `channel_count_changed` alert. Sub-Stage 9.4 detects the regime change on March 15th. All training data before March 15th is discarded — the model learns the new multi-channel demand level, not a blended history.

### 7.4 Pattern Feedback Write (CRITICAL)

**Immediately after backtesting, before Sub-Stage 9.5 starts**, Stage 9 writes one row to `pattern_feedback` for this SKU. This is the most critical write in the entire pipeline.

```
pattern_feedback row:
  pattern_label        ← what Stage 8 classified this SKU as
  forecast_error_mape  ← backtest MAPE on 30-day horizon (or 0.50 proxy if model failed)
  bias                 ← (forecast - actual) / actual
  model_used           ← what model Stage 9 actually used
  hint_matched         ← did Stage 8's model_hint match what Stage 9 used?
  classification_quality ← 'good' (MAPE < 0.15) / 'acceptable' (0.15–0.40) / 'poor' (> 0.40) / 'proxy'
  fallback_used        ← TRUE if model failed and 0.50 proxy was written
```

**Write guarantees:**
- Written via direct `conn.execute()` + `conn.commit()` — **never via BatchWriter**
- Written even for failed SKUs (use MAPE = 0.50 proxy, `fallback_used = TRUE`)
- Retried 3 times on DB failure (100ms between attempts), never silently skipped
- `run.status` is NOT set to `'forecasted'` until all pattern_feedback rows for this run exist

Stage 8 reads every row of this table on its next run to validate and improve its pattern classifications.

---

## 8. Sub-Stage 9.5 — Forecast Generation & Confidence Scoring

Generates the 8-horizon forecasts and computes the final confidence score. The `forecasts` table row written here is the primary output Stage 10 reads.

### 8.1 Multi-Horizon Forecast Generation

Each model uses a different strategy to produce forecasts for all 8 horizons `[7, 14, 30, 60, 90, 150, 180, 365]`:

| Model | Strategy | Why |
|---|---|---|
| **NeuralProphet / Prophet** | Fit **once** for 365 days forward. Extract cumulative sums at each horizon boundary. **Never scale from forecast_30d.** | Scaling from 30d destroys seasonal peaks. A Christmas-peaking product's H365 must show the December peak — multiplying a October forecast by 12 gives a flat underestimate. |
| **SES** | `level × N days` for each horizon. Linear scaling is correct — no seasonal component. | SES models a flat level. Demand for the next N days = level × N. |
| **Holt's Linear Trend** | `(level + trend × N) × damping_factor(N)` | The damping factor (φ = 0.95) prevents unlimited extrapolation — growth realistically slows over long horizons. |
| **Croston's Method** | `daily_rate × N days` for each horizon. | Croston produces a constant daily rate. Cumulative demand = rate × days. |
| **Naive Forecast** | `level × N days` (flat). | Cold-start products have no trend or seasonality to model. |

**OOS factor applied to all models:**
```
point_forecast = model_output × oos_factor    ← uplift for suppressed OOS demand
```

### 8.2 Bootstrap Quantile Generation

After each horizon's point forecast is computed, `bootstrap_quantiles()` generates the four quantile values stored in each JSONB column:

```
{
  "mean": 540.0,    ← point forecast (display only)
  "p50":  520.0,    ← 50th percentile (median scenario)
  "p80":  620.0,    ← 80th percentile (trending/stable ordering)
  "p90":  680.0     ← 90th percentile (seasonal/intermittent/cold_start ordering)
}
```

**Algorithm:**
1. Take last 30 residuals from `compute_residuals()` (actual − fitted)
2. If < 3 residuals: use log-normal proxy parameterised by pattern uncertainty factor
3. Resample residuals 1000 times with replacement
4. Each sample = point + random residual
5. Clamp all samples ≥ 0
6. Take 50th, 80th, 90th percentiles
7. Sort to guarantee p50 ≤ p80 ≤ p90 (floating-point safety)

**Log-normal uncertainty factors (for < 3 residuals case, Edge Case E004):**

| Pattern | Factor | Interpretation |
|---|---|---|
| `cold_start` | 0.60 | Wide spread — no history at all |
| `intermittent` | 0.50 | Moderate-high — sporadic demand |
| `seasonal` | 0.40 | Cycle direction known, magnitude uncertain |
| `trending` | 0.35 | Direction known, rate uncertain |
| `stable` | 0.25 | Most predictable — tightest spread |

### 8.3 Confidence Formula — Five Steps

Every value read from `tenant_learning_params`. Nothing hardcoded.

```
Step 1 — Base:
    confidence = confidence_base_{pattern}          (e.g. 0.80 for seasonal)
              × (1 - min(backtest_MAPE, 0.50))      (MAPE capped at 0.50)
              × exception_penalty                    (starts 0.80 — applied if any exceptions)

Step 2 — Calibration:
    if historically over-confident (from adaptive_quantile_state):
        confidence × 0.90
    if historically under-confident:
        confidence × 1.10

Step 3 — Stage 8 quality inheritance:
    if Stage 8 composite_confidence < 0.60:
        confidence × 0.92

Step 4 — Reorder feedback:
    confidence × reorder_bias_factor       (1.10 / 1.00 / 0.92 from Sub-Stage 9.1)

Step 5 — Structural break penalty:
    if training_data_truncated (break detected in Sub-Stage 9.4):
        confidence × 0.85

Final clamp:
    confidence = CLAMP(confidence, confidence_floor, confidence_ceiling)
              = CLAMP(confidence, 0.30, 0.95)
```

> **Full example — seasonal SKU with a promo spike:**
> ```
> confidence_base_seasonal = 0.80
> backtest_MAPE            = 0.22
> exception_penalty        = 0.80  (promo_spike detected)
>
> Step 1: 0.80 × (1 - 0.22) × 0.80 = 0.80 × 0.78 × 0.80 = 0.499
> Step 2: historically calibrated → no change → 0.499
> Step 3: Stage 8 confidence = 0.75 (above 0.60 threshold) → no penalty → 0.499
> Step 4: no prior stockouts → reorder_bias_factor = 1.00 → 0.499
> Step 5: no structural break → no penalty → 0.499
> Clamp:  max(0.30, min(0.95, 0.499)) = 0.499
>
> Final confidence: 0.499
> → confidence_tier: review_required  (0.45 ≤ 0.499 < 0.60)
> ```

### 8.4 Confidence Tier Mapping

| Tier | Condition | Meaning |
|---|---|---|
| `auto_proceed` | confidence ≥ 0.70 | System is confident. Stage 10 generates PO automatically. |
| `review_suggested` | 0.60 ≤ confidence < 0.70 | Probably fine but something slightly unusual detected. Quick human check recommended. |
| `review_required` | 0.45 ≤ confidence < 0.60 | Genuine uncertainty. Manager should review recent demand. |
| `manual_override` | confidence < 0.45 | System lacks sufficient reliable information. Human decision required. |

### 8.5 Forecast Status (Stage 10 instruction)

```
IF on_watchlist = TRUE:
    status = 'watchlist_review'             (overrides everything)
ELIF 'high_mape' in exception_flags:
    status = 'needs_acknowledgment'
ELIF confidence < decision_gate_threshold:  (starts 0.70)
    status = 'needs_acknowledgment'
ELSE:
    status = 'forecasted'
```

### 8.6 Reasonableness Check

Before writing to the `forecasts` table, a sanity check flags implausible forecasts:

```
daily_rate = forecast_30d_mean / 30

IF daily_rate > rolling_90d_avg × 5.0:    flag 'unusually_high' + needs_acknowledgment
IF daily_rate < rolling_90d_avg × 0.10:   flag 'unusually_low'
```

---

## 9. Sub-Stage 9.6 — Size Curve Distribution

**Only runs when `canonical_sku.parent_style_id IS NOT NULL`.** This is the mechanism for fashion/apparel brands where a style (e.g., "Women's Knit Jumper — Navy") has multiple size variants (XS, S, M, L, XL).

Sub-Stage 9.5 generates the aggregate forecast at the parent style level. Sub-Stage 9.6 distributes it across sizes.

### How Distribution Works

```
1. Load size curve from size_curve_registry for (style_id, season)
   If not found: build from history, borrow from prior season, or use category default

2. Normalise: each_share = each_share / sum(all_shares)
   (guarantees shares sum to 1.0 — prevents float drift edge case E008)

3. For each child SKU (each size):
   child_forecast_Nd = parent_forecast_Nd × size_share_pct
   (applied to all 8 horizons × all 4 quantiles)

4. If planned_end_date exists:
   projected_sell_through = min(1.0, forecast_until_end / current_inventory)
```

> **Example:**
> Parent style forecast_30d p90 = 1,000 units
> Size curve: S=18%, M=35%, L=28%, XL=14%, XXL=5%
> → S gets 180 units, M gets 350, L gets 280, XL gets 140, XXL gets 50

---

## 10. The Two Data Transformation Pipelines

These are not statistical models. They transform the training data before it is passed to the five standard models. Both are triggered by SKU attributes — no sector-specific code anywhere.

### 10.1 CategoryComps Warm-Start

**Trigger:** `lifecycle_stage = 'introduction'` (set by Stage 8 when product has < 28 days of history)

**Problem:** A brand new product has almost no history. Handing 5 days of data to NeuralProphet produces garbage. A flat Naive forecast ignores everything learned from similar products.

**Solution:** Find comparable products that have already converged, borrow their demand trajectory, and build a synthetic training series.

```
1. Query sku_similarity_registry for:
   - same pattern_label
   - similar vendor + product_type
   - avg_mape < 0.25 (only well-calibrated comps)
   - already exited introduction phase
   - minimum 3 comps required

2. Build trajectory:
   - Normalise each comp's first-60-day trajectory by its mean
   - Take the p50 trajectory across all comps

3. Adjust for early signals:
   IF actual day-3 demand > comp p80:  scale curve up 15%
   IF actual day-3 demand < comp p20:  scale curve down 15%
   IF within p20–p80:                  use curve as-is

4. Pass synthetic series to the assigned model
   (flag: training_data_source = 'category_comps')

Fallback: if < 3 comps found → Naive Forecast, flag 'insufficient_comps_for_warm_start'
```

### 10.2 ClearanceAdjustment

**Trigger:** `lifecycle_stage = 'clearance'` OR `discount_pct > 0.15` for ≥ 5 consecutive days

**Problem:** Historical demand at full price cannot predict clearance demand. A product marked 40% off will sell much faster — but how much faster?

**Solution:** Apply price elasticity to project clearance demand from the pre-markdown baseline.

```
baseline = average demand in 14 days before markdown start

adjusted_demand = baseline × (1 - discount_pct) ^ elasticity

elasticity = price_elasticity_clearance from tenant_learning_params
             (starts -1.50, evolves from actual clearance outcomes)
```

> **Example:**
> `baseline = 50 units/day`
> `discount_pct = 0.40` (40% off)
> `elasticity = -1.50`
> `adjusted = 50 × (1 - 0.40)^(-1.50) = 50 × (0.60)^(-1.50) = 50 × 2.15 = 107.5 units/day`
>
> The markdown is projected to more than double daily sales — which drives `projected_sell_through` computation for buyer decisions.

---

## 11. How Stage 9 Exits — Output Contract with Stage 10

### 11.1 What Gets Written

After ACTING completes, the LEARNING state flushes the BatchWriter. The primary output is the `forecasts` table — one row per SKU per run.

**The forecasts row:**
```
forecasts.forecast_30d example:
{
  "mean": 540.00,
  "p50":  520.00,
  "p80":  620.00,
  "p90":  680.00
}
```

All 8 horizons have this structure. Stage 10 reads `selected_quantile` (e.g. 0.90) to know which key to extract.

### 11.2 Nearest-Ceiling Horizon Rule (Critical for Stage 10)

Stage 10 must NOT scale from `forecast_30d` to approximate longer planning horizons. Correct approach:

```
H = lead_time_days + review_days

IF   H <= 7:    use forecast_7d,   demand = forecast_7d[p90]   × (H/7)
ELIF H <= 14:   use forecast_14d,  demand = forecast_14d[p90]  × (H/14)
ELIF H <= 30:   use forecast_30d,  demand = forecast_30d[p90]  × (H/30)
ELIF H <= 60:   use forecast_60d,  demand = forecast_60d[p90]  × (H/60)
ELIF H <= 90:   use forecast_90d,  ...
ELIF H <= 150:  use forecast_150d, ...
ELIF H <= 180:  use forecast_180d, ...
ELSE:           use forecast_365d, demand = forecast_365d[p90] × (H/365)
```

> **Why this matters:** If a supplier has a 45-day lead time, Stage 10 uses `forecast_60d × (45/60)`. If it mistakenly used `forecast_30d × (45/30)`, it multiplies a 30-day forecast by 1.5 — completely ignoring that Prophet already computed the correct 60-day seasonal shape with peaks and troughs.

### 11.3 Status and What Stage 10 Does With It

| Status | Stage 10 Action |
|---|---|
| `forecasted` | Generate PO recommendation automatically. No human review needed. |
| `needs_acknowledgment` | Generate PO but mark `pending_ack`. User must Accept / Override / Skip before Stage 11 processes it. |
| `watchlist_review` | Same as `needs_acknowledgment`. Show Stage 8's watchlist reason in the dashboard. |

### 11.4 Cross-Agent Signals Emitted on Exit

Stage 9 writes two signals after every SKU:

| Signal | To | Payload | Purpose |
|---|---|---|---|
| `forecast_accuracy` | Stage 8 | `{pattern_label, model_used, mape, bias, quality, hint_matched}` | Stage 8 reads this to validate its classifications. |
| `forecast_risk` | Stage 10 | `{confidence, confidence_tier, risk_level, exception_flags, mape_30d, forecast_30d_selected, selected_quantile}` | Stage 10 uses `risk_level` to add safety buffers: low=none, medium=+5%, high=+10%. |

After REPORTING completes, Stage 9 also emits:

| Signal | To | Purpose |
|---|---|---|
| `model_health` | All stages | Which models are degrading. Stage 10 adds extra buffers for degrading models. |
| `cross_sku_learning` | Stage 9 (self) | Converged SKU configs for warm-starting new SKUs in future runs. |

### 11.5 Run Status Set

```
REPORTING state:

IF any SKU has status = 'needs_acknowledgment' or 'watchlist_review':
    run.status = 'needs_acknowledgment'   ← LangGraph: Stage 10 waits for user
ELSE:
    run.status = 'forecasted'             ← LangGraph: Stage 10 auto-starts

PRECONDITION: all pattern_feedback rows for this run must be written before
              run.status is set. Never set status before pattern_feedback is complete.
```

---

## 12. How Stage 9 Learns

Stage 9 never has fixed parameters. Every threshold, buffer, and model configuration evolves from evidence. Three mechanisms drive this.

### 12.1 tenant_learning_params — Per-Customer Parameter Evolution

Every number that influences forecasting is stored in `tenant_learning_params` as a `current_value`. On day 1, `current_value = starting_value` (research-informed defaults). Over time, the `LearningParamsUpdater` batch job nudges each value toward what evidence shows is optimal for *this specific customer*:

```
new_current_value = prior_current_value
                  + calibration_update_rate × (evidence_value - prior_current_value)

calibration_update_rate = 0.10 (starts)  →  10% of gap per update
```

This is exponential smoothing — the parameter moves 10% of the way toward the evidence value each run.

> **Convergence timeline:**
> - After 15 runs: parameter is ~79% of the way from starting_value to evidence_value
> - After 30 runs: ~95% converged
> - After 60 runs: essentially fully converged

**What each parameter group converges toward:**

| Parameter Group | Evidence Source | Converges Toward |
|---|---|---|
| `confidence_base_{pattern}` | `avg(1 - actual_MAPE)` for that pattern | Accuracy actually achieved by each model for this customer |
| `quantile_{pattern}` | Binary search: quantile where actual coverage = `service_level_target` | The quantile that actually achieves the target fill rate for this customer |
| `safety_stock_factor` | Minimise (stockouts × cost + overstock × holding_cost) from Stage 10 signals | The buffer that minimises inventory cost given this customer's demand volatility |
| `decision_gate_threshold` | ROC analysis on forecast outcomes | The confidence level that actually predicts forecast reliability for this customer |
| `price_elasticity_clearance` | Actual clearance outcomes | How this customer's products respond to markdowns |

### 12.2 Thompson Sampling State — Per-SKU HP Convergence

The `thompson_sampling_state` table holds a Beta(α, β) distribution per `(tenant_id, sku_id, assigned_model, config_hash)`. As evidence accumulates, the distribution concentrates around the winning HP configuration.

```
After N runs of a SES SKU:
  alpha=18, beta=2 for config {alpha=0.30}
  alpha=3,  beta=9 for config {alpha=0.10}

  Expected win rate: 0.30 config → 18/20 = 90%
                     0.10 config → 3/12  = 25%

  Thompson will almost always select the 0.30 config.
  Exploration has effectively stopped for this SKU.
```

### 12.3 Adaptive Quantile State — Calibration Feedback

The `adaptive_quantile_state` table tracks whether the selected quantile (p80 or p90) actually achieves its theoretical coverage. If Stage 9 claims p90 but only 75% of actual outcomes fall below the forecast, the system is over-confident.

```
calibration_gap = actual_coverage - target_quantile

  positive gap → over-confident → Step 2 of confidence formula applies × 0.90
  negative gap → under-confident → Step 2 applies × 1.10
```

This is updated by the `OutcomeCollector` batch job after each horizon closes.

---

## 13. The Three Feedback Loops

### Loop 1 — Immediate Feedback (Stage 9 → Stage 8)

**Direction:** Stage 9 writes → Stage 8 reads on next run  
**Table:** `pattern_feedback`  
**What it does:** Stage 8 uses the MAPE scores to validate and improve its pattern classifications. If Stage 8 consistently classifies a product type as `stable` but Stage 9's backtest shows high MAPE, Stage 8 adjusts its thresholds.

```
Stage 9 run N:    backtests complete → writes pattern_feedback (MAPE, bias, model_used)
Stage 8 run N+1:  reads pattern_feedback → adjusts confidence thresholds
                  → better pattern classifications
                      → better model assignments in Stage 9
                          → lower MAPE
```

**Why it matters:** Without this loop, a wrong pattern classification (e.g., labelling a trending SKU as stable) would produce bad forecasts forever. With the loop, Stage 8 detects the systematic MAPE elevation for that pattern label and corrects within a few runs.

### Loop 2 — Delayed Feedback (Reality → Stage 9)

**Direction:** OutcomeCollector writes → LearningParamsUpdater reads → parameters update  
**Table:** `forecast_outcomes`  
**What it does:** After each horizon period closes (e.g. 30 days after a forecast is made), the actual units sold are compared to what was forecast. These outcomes update `tenant_learning_params` and `adaptive_quantile_state`.

```
Day 0:      Stage 9 forecasts 450 units for the next 30 days
Day 30:     OutcomeCollector queries golden_table for actual sales = 380 units
            Writes forecast_outcomes: forecast=450, actual=380, MAPE=0.18, bias=+0.184
Day 31 AM:  LearningParamsUpdater reads these outcomes
            confidence_base_seasonal nudges toward (1 - 0.18) = 0.82 from current 0.80
            quantile_seasonal nudges toward coverage that actually captured this demand
```

### Loop 3 — Cross-Agent Feedback (Stage 10 → Stage 9)

**Direction:** Stage 10 writes `reorder_outcome` signals → Stage 9 reads in Sub-Stage 9.1  
**Table:** `cross_agent_signals`  
**What it does:** When purchase orders generated from Stage 9's forecasts lead to stockouts or overstock, Stage 10 reports back. Stage 9 adjusts its `reorder_bias_factor` per SKU.

```
Stage 10 cycle closes for SKU X:
  stockout = TRUE, stockout_days = 4, overstock_pct = 0.00
  → writes reorder_outcome signal

Next Stage 9 run, Sub-Stage 9.1:
  reads last 5 reorder_outcome signals for SKU X
  stockout_count = 2 (in last 5) → reorder_bias_factor = 1.10
  → all 8 horizon forecasts for SKU X are implicitly 10% more conservative
  → Stage 10 generates slightly larger PO → fewer stockouts going forward
```

**Why this loop is unique:** No competitor has a direct channel from inventory outcomes back to forecast calibration. The forecast and the reorder result are normally separate systems that never communicate.

---

## 14. Daily Batch Jobs — Learning Happens Overnight

Four jobs run automatically each night, in dependency order:

| Time | Job | Reads From | Writes To | What It Does |
|---|---|---|---|---|
| 3:00 AM | **OutcomeCollector** | `golden_table` (actual sales), `forecasts` | `forecast_outcomes`, `adaptive_quantile_state` | For each horizon that has closed, computes actual vs forecast MAPE, WAPE, bias. Stores ground truth. |
| 4:00 AM | **ModelPerformanceAggregator** | `forecast_outcomes` | `model_performance_s9` | Aggregates rolling 30-day MAPE per model per horizon per tenant. Used by self-assessment and Thompson mode decisions. |
| 4:30 AM | **LearningParamsUpdater** | `forecast_outcomes`, `cross_agent_signals` (reorder outcomes) | `tenant_learning_params` | Updates all `current_value`s via exponential smoothing toward evidence. This is where the actual parameter learning happens. |
| 5:00 AM | **SimilarityRegistryUpdater** | `hyperparameter_decisions`, `forecast_outcomes` | `sku_similarity_registry` | Registers SKUs that have run ≥ 5 times with avg_MAPE < 0.25. Enables warm-start for new similar SKUs. |

> **What a missed night looks like:** If `OutcomeCollector` misses a day, the `cutoff_date` check automatically catches up on the next run — no manual intervention needed. All other jobs also catch up naturally.

---

## 15. Key Definitions & Glossary

| Term | Definition |
|---|---|
| **MAPE** | Mean Absolute Percentage Error. `abs(actual - forecast) / actual`. 0.15 = 15% average error. The primary accuracy metric. |
| **WAPE** | Weighted Absolute Percentage Error. Like MAPE but weights errors by volume — high-volume days count more. More representative than MAPE for intermittent demand. |
| **Bias** | `(forecast - actual) / actual`. Positive = systematically over-forecasting (ordering too much). Negative = under-forecasting (stockouts). |
| **p50 / p80 / p90** | Percentile forecasts. p90 means "actual demand will be below this value 90% of the time." Higher percentile = larger order = safer against stockout. |
| **Quantile** | The p-value used for ordering. p90 is more conservative than p80. Pattern defaults: 0.90 for cold_start/intermittent/seasonal; 0.80 for trending/stable. |
| **OOS** | Out-of-stock. A period when a product was unavailable for purchase. OOS days show zero demand but this is suppressed demand, not real zero demand. |
| **Demand suppression** | When OOS causes recorded sales to be lower than actual customer desire. Stage 8 estimates suppressed demand; Stage 9 applies the correction via OOS adjustment factor. |
| **pattern_label** | The demand shape Stage 8 assigned to a SKU: `cold_start`, `intermittent`, `seasonal`, `trending`, or `stable`. Determines the entire forecasting approach. |
| **Thompson Sampling** | A Bayesian method for choosing which HP configurations to test. Builds a Beta(α, β) distribution per config — more successes → higher α → more likely to be tested again. Converges to the best config over runs. |
| **HP / Hyperparameter** | A configuration parameter of a model (e.g., the smoothing alpha in SES, or the changepoint_prior_scale in Prophet). Not learned from data directly — tuned by Thompson Sampling. |
| **Backtest** | Hold out the most recent N days of history, train the model on older data, and measure its error on the held-out period. Simulates how well the model will predict future demand. |
| **Structural break** | A point in time where the demand pattern fundamentally changed (e.g., a new competitor launched, a brand went viral, a channel was added). Data before the break is no longer representative. |
| **PELT algorithm** | Pruned Exact Linear Time — the algorithm used to detect structural breaks. Implemented via the `ruptures` Python library. |
| **Bootstrap** | Resampling technique: randomly draw from historical residuals 1,000 times to simulate the distribution of plausible future outcomes. Used to generate p50/p80/p90 from a point forecast. |
| **Residuals** | The error between what the model fitted in-sample and what actually happened: `actual - fitted`. The spread of residuals determines how wide the p80/p90 bands are. |
| **confidence_final** | The output confidence score (0.30–0.95) after all five formula steps. Drives the confidence tier and the `status` instruction to Stage 10. |
| **reorder_bias_factor** | A multiplier (1.10 / 1.00 / 0.92) applied to forecast conservatism based on whether prior orders led to stockouts or overstock. Written to `model_initialization_s9`. |
| **SKU** | Stock Keeping Unit. One specific product in one specific variant (size, colour, etc.). The atomic unit of forecasting. |
| **size curve** | The historical distribution of demand across sizes for a parent style (e.g., M=35%, L=28%, S=18%...). Used in Sub-Stage 9.6 to split parent forecasts into per-size forecasts. |
| **CategoryComps** | Warm-start pipeline for new products. Builds a synthetic training series from the demand trajectories of similar already-converged products. |
| **ClearanceAdjustment** | Pipeline for products under markdown. Uses price elasticity to project how much faster the product will sell at the discounted price. |
| **sell-through projection** | `min(1.0, forecast_until_end_date / current_inventory)`. What fraction of current stock is expected to sell before the product is discontinued. A value < 0.85 suggests a markdown may be needed. |
| **BatchWriter** | Accumulates all DB writes during ACTING state and flushes every 100 SKUs. Reduces ~40,000 individual INSERT calls to ~400 batch inserts. `pattern_feedback` is the one exception — always written directly. |
| **data fingerprint** | SHA-256 of a SKU's last 7 days of demand. Compared to the prior run's fingerprint to classify the SKU as cache / partial / full tier. |
| **tenant_maturity** | How many calibrated runs a tenant has completed: new / developing / established. Controls how quickly Stage 9 switches from explore mode to exploit mode. |
| **cross_agent_signals** | The inter-stage signal bus table. All communication between Stage 8, 9, and 10 passes through this table. Nothing communicates via function calls or APIs. TTL-based expiry. |
| **PEEK vs CONSUME** | Two patterns for reading signals. PEEK: read without marking `processed = TRUE` (signal stays available for future runs). CONSUME: read and mark `processed = TRUE` (one-time action). |
