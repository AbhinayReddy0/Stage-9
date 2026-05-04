# ATHEERA

## STAGE 9 — The Forecasting Agent

**Complete Technical Documentation**
Deep walk-through with definitions, math, sample outputs and worked examples

| Field | Value |
|---|---|
| Document type | Technical specification + algorithm walkthrough + DB reference |
| Audience | Engineers, ML engineers, architects, reviewers |
| Source artefacts | `code/` (production source) + `STAGE_9_MASTER_SPEC.docx` + `STAGE_9_TECHNICAL_CONTEXT.docx` + `STAGE_9_DATABASE_CONTRACTS.docx` + `STAGE9_TO_STAGE10_API_CONTRACT.docx` |
| Version | 1.0 |
| Date | May 2026 |

---

## Index

| Section | Page |
|---|---|
| **Part I — Foundations** | |
| 1. The Forecasting Problem | 3 |
| 2. Repository Layout | 5 |
| 3. System Architecture (Figure 1) | 7 |
| 4. End-to-End Data Flow (Figure 2) | 10 |
| 5. Four Worked Examples | 13 |
| **Part II — Step-by-Step Execution** | |
| Step 1 — `pipeline/orchestrator.py` (entry point) | 16 |
| Step 2 — `infrastructure/config.py` (env resolution) | 19 |
| Step 3 — `infrastructure/run_lock.py` (Redis lock) | 21 |
| Step 4 — `infrastructure/state_machine.py` (state transitions) | 23 |
| Step 5 — `handlers/preloading.py` + `pipeline/preloader.py` | 25 |
| Step 6 — `forecasting/fingerprint.py` (Sub-Stage 9.0) | 28 |
| Step 7 — `forecasting/tier_router.py` (3-track dispatch) | 30 |
| Step 8 — `handlers/perceiving.py` (TenantParams snapshot) | 32 |
| Step 9 — `handlers/planning.py` (cache pre-fetch + dual-pool split) | 34 |
| Step 10 — `handlers/acting.py` + `pipeline/dual_pool.py` | 36 |
| Step 11 — `pipeline/model_initialization.py` (Sub-Stage 9.1) | 39 |
| Step 12 — `forecasting/feature_engg.py` (Sub-Stage 9.2) | 42 |
| Step 13 — `models/hp_tuning.py` + `models/thompson.py` (Sub-Stage 9.3) | 44 |
| Step 14 — `backtesting/backtesting.py` (Sub-Stage 9.4) | 47 |
| Step 15 — `forecasting/forecasting.py` + `models/bootstrap.py` (Sub-Stage 9.5) | 50 |
| Step 16 — `forecasting/confidence.py` (5-step formula) | 54 |
| Step 17 — `infrastructure/batch_writer.py` (deferred writes) | 57 |
| Step 18 — `handlers/learning.py` (Thompson upsert) | 59 |
| Step 19 — `handlers/reporting.py` + `learning/self_assessment.py` | 61 |
| Step 20 — `signals/emitter.py` + `signals/consumer.py` (signal bus) | 63 |
| Step 21 — Nightly Batch Jobs (3 AM / 4 AM / 4:30 AM) | 65 |
| **Part III — Reference** | |
| R1. TenantParams — Full Catalog (54 rows) | 70 |
| R2. Forecasting Models Catalog | 73 |
| R3. Output Contract with Stage 10 | 76 |
| R4. Database Schema Reference (`stage9.*`) | 79 |
| R5. Locked Invariants (the 6 rules nothing crosses) | 84 |
| R6. How to Run Stage 9 Locally | 86 |
| R7. Glossary | 88 |

---

## Part I — Foundations

Part I introduces the vocabulary and architectural mental model used throughout. It also defines the four worked-example SKUs that Part II traces through every step. Read Part I sequentially. Part II walks every executable step of a Stage 9 run; Part III is reference material you can flip to in any order.

### 1. The Forecasting Problem

#### 1.1 What problem is Stage 9 actually solving?

Demand forecasting is the highest-leverage upstream decision in any retail or distribution operation. Every replenishment, every safety-stock buffer, every promotional commitment, every working-capital projection is anchored to a number that says *"this many units of this SKU will sell over the next N days."* The number drives Stage 10 (Decision Agent), which decides what to order; Stage 11 narrates it to the buyer; Stage 12 surfaces it on the dashboard. Get the number wrong and every downstream stage compounds the error.

Two failure modes dominate the space:

- **Forecast bias** — the number is systematically too high or too low. Drives chronic stockouts (under-forecast) or excess inventory (over-forecast).
- **Forecast brittleness** — the number is right on average but wrong at the tails. A confident point forecast that misses the 90th-percentile case is worse than an honestly-uncertain forecast, because the operator builds plans against the wrong number.

Most companies still solve this with hand-tuned exponential smoothing or a single global model trained nightly. Those approaches cannot adapt to per-SKU pattern shifts, cannot respect lifecycle (a brand-new SKU has no history; a saturating SKU has stable history; a discontinuing SKU has structural breaks), and cannot quantify uncertainty.

#### 1.2 How Stage 9 reframes the problem

Stage 9's spec (`STAGE_9_MASTER_SPEC.docx` §1) frames forecasting as a **per-SKU Bayesian decision problem**. For every SKU and every horizon, Stage 9 chooses a model, a hyperparameter configuration, a training feature set, and a confidence level — and then produces a point forecast plus a `{mean, p50, p80, p90}` quantile distribution.

Every choice is **learned from outcomes**, not hard-coded. The Thompson Sampling bandit accumulates Beta(α, β) reward signals from backtest MAPE; the calibrated backtest window comes from `model_performance_aggregator`'s nightly rollup; the confidence formula's five multiplicative steps each have a learned floor and ceiling in `tenant_learning_params`.

Three commitments determine almost every implementation choice in the codebase. If you understand them you can predict where any new feature should live.

| Commitment | Why it matters | Code consequence |
|---|---|---|
| **Nothing is static, except the locked invariants.** | 41+ tenant-level parameters live in `tenant_learning_params` and are nudged nightly by `learning_params_updater`. The locked invariants (8 horizons, per-SKU isolation, no Stage 8 writes) prevent the learner from ever crossing a boundary that would break a downstream contract. | If a numeric threshold appears in an `if`-statement anywhere outside a model class, it is a bug. Read `infrastructure/constants.py:Param` and route the value through `TenantParams.get(...)`. |
| **One bulk read at the top, one bulk flush at the bottom.** | A 10K-SKU run cannot afford one DB round-trip per SKU. `Preloader` reads everything in PRELOADING; `BatchWriter` queues every write and flushes in LEARNING. The hot path is a Python dict lookup. | If a sub-stage opens its own cursor, it is a bug. Every read goes through `RunContext.preloaded`; every write goes through `BatchWriter.queue(...)`. |
| **Per-SKU isolation (Principle 3).** | One bad SKU never aborts the run. `acting_handler._per_sku_fallback` catches every per-SKU exception, logs it, writes a NaiveForecast row, and moves on. | Every sub-stage entry-point function must be safe to wrap in `try / except`. No global mutable state across SKUs. |

#### 1.3 Two structured outputs

On every run Stage 9 emits exactly two object types. The contracts are formally defined in `STAGE9_TO_STAGE10_API_CONTRACT.docx` and implemented in `code/handlers/acting.py:_finalize_sku_result`.

- **Forecast row** (`stage9.forecasts`) — for each SKU, a row containing: `processing_tier`, `assigned_model`, `confidence_final`, `backtest_mape`, `pattern_label`, `selected_quantile`, `effective_max_horizon`, `oos_adjustment_factor`, `reorder_bias_factor`, `is_b2b`, and eight per-horizon JSONB columns (`forecast_7d`, `forecast_14d`, …, `forecast_365d`) each holding `{mean, p50, p80, p90}`.
- **Cross-agent signals** (`stage9.cross_agent_signals`) — typed messages: `model_health` (broadcast to all agents at REPORTING), `forecast_revised` (per-SKU when a forecast crossed a regime change), `pattern_feedback` (per-SKU per-run from Sub-Stage 9.4 back to Stage 8).

Both outputs are persisted with full audit detail (`stage9_self_assessment` records the run summary, `agent_state_log_s9` records every state transition, `stage9_sku_execution_log` records per-SKU diagnostics). Stage 10 reads `stage9.forecasts` directly; Stages 11/12 narrate the same row.

#### 1.4 Three execution tracks

Not every SKU needs the same depth of computation. Stage 9 dispatches each SKU to one of three tracks based on a SHA-256 fingerprint of its recent demand:

| Track | Trigger | Sub-stages that run | Target time per SKU |
|---|---|---|---|
| **MICRO_UPDATE** (run-level) | Last `COMPLETE` run within `micro_update_threshold_hours` (default 18) | SES level correction only — no model retrain. | < 15 s for entire catalog |
| **CACHE** (SKU-level) | Fingerprint identical to last run AND `pattern_label` unchanged AND prior MAPE good | Warm-start from cached prior forecast; apply micro SES update. | ~5 ms |
| **FULL** (SKU-level) | Default | Sub-Stages 9.1 → 9.2 → 9.3 → 9.4 → 9.5 → write. | 30 ms (Naive/SES/Croston/Holt) — 4–8 s (Prophet) |

The three-track design is what makes Stage 9 fast on incremental days (most SKUs hit CACHE) without sacrificing accuracy on the SKUs that actually moved.

#### 1.5 Performance envelope

Performance numbers below come from `STAGE_9_TECHNICAL_CONTEXT.docx` §14 and the dual-pool design in `pipeline/dual_pool.py`. The numbers are achievable because Stage 9 follows seven hot-path rules:

1. One read per table per run (Preloader's 7 bulk SELECTs).
2. Parallel fan-out across two pools (4 processes for Prophet, 16 threads for everything else).
3. Per-SKU dict lookups instead of DB queries inside the loop.
4. Buffered writes (`BatchWriter`) flushed once in LEARNING.
5. Calibrated backtest windows (skip the per-SKU `SELECT` via the planning-layer cache).
6. Three execution tracks (most SKUs skip the full pipeline).
7. Nightly learning is **off** the hot path — the learner reads `forecast_outcomes` and never blocks a forecasting run.

| Tenant size | SKUs | Wall clock | DB round trips |
|---|---|---|---|
| SMB | 1,000 | 60–90 s | ~12 |
| Mid-size | 10,000 | 8–12 min | ~25 |
| Enterprise | 100,000 | 25–40 min | ~70 |

The wall-clock ceiling is dominated by Prophet fits (`ProcessPoolExecutor`, 4 workers, 120 s timeout per SKU). The DB round-trip ceiling is dominated by Preloader's 7 bulk JOINs at the top, the Thompson state bulk-upsert in LEARNING, and the BatchWriter flushes — everything else is in-memory.

### 2. Repository Layout

`code/` is a single Python project. Each top-level folder has one architectural responsibility. The table below maps every folder to its role and lists the files that actually run during a Stage 9 invocation.

```
stage_9/
├── README.md
├── pytest.ini
├── project docs/
│   ├── requirements.txt
│   ├── stage9_data_flow_and_learning.md
│   ├── Stage9_EdgeCases.pdf
│   └── STAGE_9_PROJECT_DOCUMENT.md       ← (this file)
└── code/
    ├── .env                              ← DB / Redis / app config
    ├── pytest.ini
    ├── pipeline/                         ← entry point, executor, preloader, LangGraph wrapper
    ├── handlers/                         ← one module per agent state
    ├── forecasting/                      ← Sub-Stages 9.0, 9.2, 9.5 + tier router
    ├── backtesting/                      ← Sub-Stage 9.4
    ├── models/                           ← BaseModel + 5 concrete models + Thompson + bootstrap
    ├── learning/                         ← 3 nightly batch jobs + self-assessment
    ├── infrastructure/                   ← config, DB, locks, state machine, params
    ├── signals/                          ← cross-agent signal bus (emit/consume)
    ├── results/                          ← per-run output artefacts
    └── tests/                            ← pytest suite
```

| Folder | Role | Key files (the ones that execute) |
|---|---|---|
| `pipeline/` | Entry point + concurrency + bulk-load + outer LangGraph | `orchestrator.py`, `dual_pool.py`, `preloader.py`, `pipeline_graph.py`, `model_initialization.py` |
| `handlers/` | One module per agent state | `preloading.py`, `perceiving.py`, `planning.py`, `acting.py`, `learning.py`, `reporting.py`, `_context.py` |
| `forecasting/` | Per-SKU sub-stages reading from RunContext | `fingerprint.py` (9.0), `feature_engg.py` (9.2), `forecasting.py` (9.5), `confidence.py`, `tier_router.py` |
| `backtesting/` | Sub-Stage 9.4 | `backtesting.py` |
| `models/` | Forecasters + Bayesian bandit | `base.py`, `naive.py`, `ses.py`, `croston.py`, `holt.py`, `prophet_model.py`, `thompson.py`, `hp_tuning.py`, `bootstrap.py` |
| `learning/` | Nightly batch jobs + self-assessment | `outcome_collector.py`, `model_performance_aggregator.py`, `learning_params_updater.py`, `self_assessment.py` |
| `infrastructure/` | Cross-cutting concerns | `config.py`, `constants.py`, `errors.py`, `state_machine.py`, `run_lock.py`, `batch_writer.py`, `seed.py`, `tenant_params.py`, `db.py`, `db_utils.py` |
| `signals/` | Cross-agent message bus | `_base.py`, `emitter.py`, `consumer.py` |

Three folders deserve special attention: `pipeline/` holds the orchestrator and the dual-pool executor — the two files an engineer will read more often than any others; `handlers/` holds the seven state-handlers that share a strict contract; `forecasting/` and `backtesting/` hold the per-SKU sub-stages where the actual math happens. Everything else either supports these three or is wiring.

### 3. System Architecture

Stage 9 is structured as **six layers** stacked top to bottom: Trigger, Orchestration, Handlers, Sub-Stages, Models, Infrastructure. Each layer has a single responsibility and is allowed to call only into layers below it.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Trigger             pipeline_graph.py — watches runs.status           │
├─────────────────────────────────────────────────────────────────────┤
│ Orchestration       orchestrator.run() — state machine + run lock    │
├─────────────────────────────────────────────────────────────────────┤
│ Handlers            preloading → perceiving → planning → acting      │
│                     → learning → reporting                           │
├─────────────────────────────────────────────────────────────────────┤
│ Sub-Stages          9.0 fingerprint   9.1 model_init   9.2 features  │
│                     9.3 hp_tuning     9.4 backtest     9.5 forecast  │
├─────────────────────────────────────────────────────────────────────┤
│ Models              naive  ses  croston  holt  prophet  thompson     │
├─────────────────────────────────────────────────────────────────────┤
│ Infrastructure      config  state_machine  run_lock  batch_writer    │
│                     tenant_params  signals  db_utils                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Figure 1 — Stage 9 system architecture (six layers)**

#### 3.1 Layer responsibilities

| Layer | Responsibility | Files (representative) |
|---|---|---|
| Trigger | Watches `runs.status='patterns_discovered'` (Stage 8 output) and dispatches the appropriate stage agent. The "outer" LangGraph the master spec prescribes. | `pipeline/pipeline_graph.py` |
| Orchestration | `run()` acquires the Redis run lock, walks the state machine (`IDLE→…→COMPLETE`/`FAILED`), and delegates each phase to its handler. Exception-safe finally-block always releases the lock. | `pipeline/orchestrator.py`, `infrastructure/state_machine.py`, `infrastructure/run_lock.py` |
| Handlers | One module per agent state. Each handler is a pure function `(tenant_id, run_id, db) → None` that reads `RunContext` and queues writes via `BatchWriter`. | `handlers/preloading.py`, `handlers/perceiving.py`, `handlers/planning.py`, `handlers/acting.py`, `handlers/learning.py`, `handlers/reporting.py` |
| Sub-Stages | Per-SKU forecasting math. Each sub-stage is a pure function over a `LearningContext` dataclass. No DB access. No global state. Always safe to call concurrently. | `forecasting/`, `backtesting/`, `pipeline/model_initialization.py` |
| Models | The five forecasting model classes (`NaiveForecast`, `SESModel`, `CrostonMethod`, `HoltLinearTrend`, `ProphetModel`) share `BaseModel`'s contract. Plus the Thompson bandit and the residual bootstrap that turns a point forecast into quantiles. | `models/` |
| Infrastructure | Config, DB plumbing, the state machine writer, the Redis lock, the batched writer, the tenant-params snapshot, signal bus. Things every layer above uses. | `infrastructure/`, `signals/` |

#### 3.2 Why six layers and not three or twelve

Stage 9 deliberately picks six layers because it is the smallest number that gives every architectural concern a single home without forcing unrelated concerns into the same layer. Three layers (entry, business logic, data) would lump the orchestrator together with the sub-stages — which would mean every change to the state machine would touch math code, and every change to a model would risk touching the topology. Twelve layers would over-engineer the boundary policing without preventing any new bug class. Six is the count that emerged from the spec's §4 Architecture section.

The most-violated boundary in code review is **Sub-Stages → Models**. Sub-Stages must not import a concrete model class directly; they must go through the handle returned by `model_initialization.run()` (which carries the `LearningContext` and the configured model instance together). This is what lets `learning/model_performance_aggregator` re-instantiate the same model offline for evaluation without re-running Sub-Stage 9.1.

### 4. End-to-End Data Flow

Figure 2 traces a single tenant run through Stage 9 from the moment LangGraph detects `runs.status='patterns_discovered'` to the moment `stage9.forecasts` rows are committed and `model_health` is broadcast.

```
┌────────────────┐   runs.status='patterns_discovered'
│  Stage 8       │ ─────────────────────────────────────┐
└────────────────┘                                       │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ pipeline_graph.py     │
                                              │ outer LangGraph       │
                                              └──────────┬───────────┘
                                                         │ tenant_id, run_id
                                                         ▼
                                              ┌──────────────────────┐
                                              │ orchestrator.run()    │
                                              │ acquire Redis lock    │
                                              └──────────┬───────────┘
                                                         │
                            IDLE → PRELOADING → PERCEIVING → PLANNING
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ handlers/preloading   │  ───▶ Preloader (7 bulk reads)
                                              │ resolves FULL vs      │       fingerprint classification
                                              │ MICRO_UPDATE          │       OOS/channel adjustments
                                              └──────────┬───────────┘
                                                         ▼
                                              ┌──────────────────────┐
                                              │ handlers/perceiving   │  ───▶ TenantParams snapshot
                                              │                       │       PEEK pattern_confidence
                                              └──────────┬───────────┘
                                                         ▼
                                              ┌──────────────────────┐
                                              │ handlers/planning     │  ───▶ pre-fetch calibrated cache
                                              │                       │       pre-fetch Thompson HP
                                              │                       │       split SKUs into pools
                                              └──────────┬───────────┘
                                                         ▼
                            ACTING (parallel)                        ┌─────────────────────────┐
                                                         ┌──────────▶│ ProcessPool (4) Prophet │
                                                         │           └─────────────────────────┘
                                              ┌──────────┴────┐
                                              │ tier_router    │      ┌─────────────────────────┐
                                              │ per-SKU        ├─────▶│ ThreadPool (16) others  │
                                              │ dispatch       │      │   9.1 → 9.2 → 9.3       │
                                              │ FULL/PARTIAL/  │      │   → 9.4 → 9.5 → write   │
                                              │ CACHE          │      └─────────────────────────┘
                                              └──────────┬─────┘
                                                         ▼
                            LEARNING                ┌──────────────────────┐
                                                    │ flush BatchWriter     │
                                                    │ upsert Thompson state │
                                                    │ refresh similarity    │
                                                    └──────────┬───────────┘
                                                               ▼
                            REPORTING               ┌──────────────────────┐
                                                    │ SelfAssessmentEngine  │
                                                    │ emit model_health     │
                                                    │ runs.status='forecast'│
                                                    └──────────┬───────────┘
                                                               ▼
                            COMPLETE                ┌──────────────────────┐
                                                    │ release run lock      │
                                                    └──────────────────────┘
```

**Figure 2 — End-to-end data flow for one tenant run**

#### 4.1 Execution order at a glance

The order Part II walks through one step at a time. Steps 11–16 are the heart of the agent — read those carefully on a first pass.

| # | File / Module | Role |
|---|---|---|
| 1 | `pipeline_graph.py` | Watches `runs.status`; triggers `orchestrator.run()` when a tenant becomes ready. |
| 2 | `orchestrator.run()` | Acquires Redis lock; walks state machine; releases lock. |
| 3 | `infrastructure/run_lock.py` | `RedisRunLock` — per-tenant `SET NX EX 14400` dead-man's switch. |
| 4 | `state_machine.transition` | Validates and writes every state change to `agent_state_log_s9`. |
| 5 | `handlers/preloading.py` → `pipeline/preloader.py` | 7 bulk SELECTs into a typed `PreloadedData` container. Resolves FULL vs MICRO_UPDATE. |
| 6 | `forecasting/fingerprint.py` | Sub-Stage 9.0 — SHA-256 fingerprint per SKU; assigns `cache`/`partial`/`full` tier. |
| 7 | `forecasting/tier_router.py` | Dispatches each SKU to its appropriate sub-stage chain. |
| 8 | `handlers/perceiving.py` | `TenantParams.load(...)`; PEEK Stage 8 `pattern_confidence` signals. |
| 9 | `handlers/planning.py` | Pre-fetch the two tenant-wide caches (calibrated backtest windows, Thompson HP). |
| 10 | `handlers/acting.py` + `pipeline/dual_pool.py` | Run Sub-Stages 9.1→9.5 per SKU across two pools. |
| 11 | `pipeline/model_initialization.py` | Sub-Stage 9.1 — 7 ordered decisions; emits `LearningContext`. |
| 12 | `forecasting/feature_engg.py` | Sub-Stage 9.2 — promo-weighted training data + feature selection. |
| 13 | `models/hp_tuning.py` + `models/thompson.py` | Sub-Stage 9.3 — Thompson sampling for HP selection. |
| 14 | `backtesting/backtesting.py` | Sub-Stage 9.4 — adaptive window walk-forward backtest; writes `pattern_feedback`. |
| 15 | `forecasting/forecasting.py` + `models/bootstrap.py` | Sub-Stage 9.5 — fit → 8 horizons → `{mean,p50,p80,p90}`. |
| 16 | `forecasting/confidence.py` | Multiplicative confidence formula (5 steps). |
| 17 | `infrastructure/batch_writer.py` | Buffered writes; flushed in LEARNING. |
| 18 | `handlers/learning.py` | Flush BatchWriter; bulk-upsert Thompson state; refresh similarity registry. |
| 19 | `handlers/reporting.py` + `learning/self_assessment.py` | Run health check; emit `model_health`; drop `RunContext`. |
| 20 | `signals/emitter.py` + `signals/consumer.py` | Direct-write emitter + atomic FOR UPDATE SKIP LOCKED consumer. |
| 21 | Nightly batch (3 AM, 4 AM, 4:30 AM) | `outcome_collector` → `model_performance_aggregator` → `learning_params_updater`. |

#### 4.2 Why the order is what it is

Three ordering decisions are non-obvious and worth explaining.

- **Step 5 (PRELOADING) before Step 8 (PERCEIVING)** is mandatory because `preloading_handler` resolves the execution mode (FULL vs MICRO_UPDATE) by reading `agent_state_log_s9` for the previous COMPLETE timestamp. PERCEIVING then consumes the `RunContext` PRELOADING wrote and loads `TenantParams` *for that mode* — micro-update mode only loads the SES-correction subset.
- **Step 9 (PLANNING) pre-fetches the calibrated-window cache and Thompson HP cache** before ACTING begins. Without these caches, every SKU in 9.4 and 9.3 would issue its own SELECT — at 10K SKUs that's 20K round trips. The Planning layer pays them once.
- **Step 18 (LEARNING) is a fan-in** — it cannot run until every SKU has finished ACTING. The dual-pool executor's `as_completed()` loop is what guarantees the barrier; LEARNING then takes the accumulated Thompson reward signals and bulk-upserts them in one statement.

### 5. Four Worked Examples

Part II shows what each step does for four representative SKUs simultaneously, so a reviewer can see one SKU that goes through every sub-stage (FULL tier), one that hits CACHE tier on run 2, one that triggers a `pattern_feedback` exception, and one that exercises the cold-start path.

#### 5.1 The four SKUs

The four SKU patterns chosen below correspond directly to four of the `pattern_label` values Stage 8 emits — and to the four `tests/stage9_data_factory.py` generators (`gen_cold_start`, `gen_stable`, `gen_trending`, `gen_intermittent`).

| Field | DOC-CS (cold-start) | DOC-STB (stable) | DOC-TRN (trending) | DOC-INT (intermittent) |
|---|---|---|---|---|
| `pattern_label` | `cold_start` | `stable` | `trending` | `intermittent` |
| `obs_days` | 25 | 90 | 120 | 180 |
| `model_hint` (from Stage 8) | `Naive` | `exponential_smoothing` | `Holt` | `Croston` |
| `lifecycle_stage` | `introduction` | `saturation` | `saturation` | `saturation` |
| Demand series shape | 25 days, mean 8 | 90 days, mean 20, σ ≈ 2 | 120 days, slope +0.05 units/day | 180 days, 65% zeros, occasional spikes |
| Expected `assigned_model` | `Naive` | `SES` | `Holt` | `Croston` |
| Expected `selected_quantile` | 0.90 | 0.80 | 0.80 | 0.90 |
| Expected backtest MAPE | n/a (obs_days < 28) | 0.10–0.18 | 0.12–0.20 | 0.20–0.55 |
| Expected `confidence_final` | 0.30–0.55 | 0.65–0.85 | 0.55–0.78 | 0.28–0.65 |

#### 5.2 Derived facts (used everywhere downstream)

| Fact | Where computed | Value |
|---|---|---|
| Coverage horizon | preloading | 30 days |
| OOS adjustment factor | preloading | DOC-CS: 1.00 (no OOS), DOC-STB: 1.00, DOC-TRN: 1.05, DOC-INT: 1.00 |
| Reorder bias factor | preloading | All four: 1.00 (no Stage-10 outcome history) |
| `is_b2b` | preloading from canonical_sku | All four: FALSE |
| Tier on run-1 | fingerprint | All four: `full` (no prior fingerprint) |
| Tier on run-2 (same data) | fingerprint | DOC-STB and DOC-TRN: `cache`; DOC-INT: `partial`; DOC-CS: still `full` (obs_days < 28 forces full) |

#### 5.3 What we expect to happen

DOC-CS is borderline → no backtest possible (`obs_days < 28`) → confidence falls back to a learned floor → `selected_quantile=0.90` → `learning_mode=explore`. DOC-STB exercises the full SES path with a tight backtest band; on run-2 the fingerprint matches and the SKU hits CACHE. DOC-TRN exercises the Holt path with damping at long horizons. DOC-INT exercises Croston (intermittent demand) and is expected to produce wider quantile spreads — every horizon's `p90/mean` ratio should be > 1.5.

---

## Part II — Step-by-Step Execution

Each step in Part II uses the same nine-section template:

| Section | Question it answers |
|---|---|
| Definition | What is this module in one paragraph? |
| Usage | When does it run, who calls it, what triggers it? |
| Problem & Solution | What real-world problem does this step solve and how does it solve it? |
| Algorithm walkthrough | The step-by-step logic, with formulas where applicable. |
| Code (annotated) | The actual code, line-by-line with annotations. |
| Inputs | Full schema of what it consumes. |
| Outputs (4 worked examples) | What it produces for each of DOC-CS, DOC-STB, DOC-TRN, DOC-INT. |
| Verification | SQL or Python that proves the step ran correctly. |
| Failure modes | What can go wrong + the system's response. |
| What runs next | Which step follows this one. |

### Step 1 — `pipeline/orchestrator.py` (entry point)

**File:** `code/pipeline/orchestrator.py`

#### Definition

`orchestrator.run` is the single entry point of Stage 9. It accepts a `(tenant_id, run_id, db, redis)` tuple, acquires the Redis run lock for that tenant, walks the agent state machine from `IDLE → PRELOADING → … → COMPLETE`, delegates each phase to the appropriate handler in `handlers/`, and releases the lock in the `finally` block — *always*.

#### Usage

Invoked once per tenant per Stage 8 completion. The outer LangGraph (`pipeline/pipeline_graph.py`) detects `runs.status='patterns_discovered'`, opens a fresh psycopg2 connection plus a Redis client, and calls `run(tenant_id, run_id, db, redis)`. There is no in-process retry — if `run()` raises, control returns to LangGraph, which decides whether to retry based on the exception type.

#### Problem & Solution

LangGraph triggers Stage 9 the moment a Stage 8 run becomes ready. Without a lock, a delayed first-trigger plus a fast retry (LangGraph's status-recheck loop) can issue two `run()` calls for the same tenant simultaneously — both would attempt the same `state_machine.transition` writes and both would race on `forecast_outcomes` upserts. Without a single state-machine driver, every handler would have to remember which state-log row to write, doubling the surface area for bugs.

**Solution.** One function (`run`) owns the lock + state walk + handler dispatch. Every handler is a pure `(tenant_id, run_id, db)` function — no handler knows it is "second" or "fourth"; the orchestrator drives the order. Failure modes are localized to the orchestrator's `except` block, which transitions to FAILED for any unrecoverable error in PRELOADING / PERCEIVING / PLANNING / ACTING. LEARNING and REPORTING have *no* edge to FAILED — if they raise, the lock is still released and the exception propagates to LangGraph for retry decisions.

#### Algorithm walkthrough

1. Validate `tenant_id` and `run_id` are well-formed UUIDs (`_validate_ids`).
2. Acquire the Redis run lock: `SET stage9_lock_{tenant_id} "locked" EX 14400 NX`. If the SET returns `0`, raise `RunAlreadyInProgressError` — another worker has the lock.
3. Walk the state machine in order, calling each handler:
   - `transition(IDLE → PRELOADING)`; `preloading_handler(...)`
   - `transition(PRELOADING → PERCEIVING)`; `perceiving_handler(...)`
   - `transition(PERCEIVING → PLANNING)`; `planning_handler(...)`
   - `transition(PLANNING → ACTING)`; `acting_handler(...)`
   - `transition(ACTING → LEARNING)`; `learning_handler(...)`
   - `transition(LEARNING → REPORTING)`; `reporting_handler(...)`
   - `transition(REPORTING → COMPLETE)`
4. On any exception in PRELOADING/PERCEIVING/PLANNING/ACTING: `transition(<current> → FAILED)`; re-raise.
5. `finally`: release the Redis lock with a Lua script (the script verifies the lock value before deleting — prevents a different process from accidentally releasing this tenant's lock).

#### Inputs

- `tenant_id : str` — UUIDv4 of the tenant.
- `run_id : str` — UUIDv4 of the Stage 8 run that triggered this Stage 9 invocation.
- `db : psycopg2.connection` — a fresh connection from the LangGraph pool.
- `redis : redis.Redis` — a Redis client connected to `REDIS_URL`.

#### Outputs

Returns nothing. Side effects:

- `agent_state_log_s9` rows for every transition (typically 7 per successful run).
- `stage9.forecasts` rows for every SKU.
- `stage9.thompson_sampling_state` rows for every (sku, model) pair that was tested in this run.
- `stage9.stage9_self_assessment` row summarizing the run.
- One `model_health` broadcast signal in `cross_agent_signals`.
- `runs.status` advanced to `forecasted` (or `failed`).

#### Verification

```sql
-- The run reached COMPLETE
SELECT to_state, transitioned_at
FROM stage9.agent_state_log_s9
WHERE run_id = '...' ORDER BY transitioned_at;
-- Expect: 7 rows ending in COMPLETE.

-- The Redis lock was released
-- (in redis-cli)
EXISTS stage9_lock_<tenant_id>
-- Expect: 0
```

#### Failure modes

| Failure | Detection | Response |
|---|---|---|
| Another run already in progress | SET NX returns 0 | `RunAlreadyInProgressError` — orchestrator returns immediately without a transition |
| Redis unreachable at acquire | `redis.exceptions.ConnectionError` | propagated; LangGraph retries after 30 s |
| `TenantParamNotFoundError` in PERCEIVING | Caught by orchestrator's `except` | `transition(PERCEIVING → FAILED)`; re-raised to LangGraph; the tenant is excluded from auto-retries until an operator seeds params |
| Per-SKU exception in ACTING | Caught inside `acting_handler._per_sku_fallback` | The bad SKU gets a Naive forecast row + a `stage9_sku_execution_log` entry; the run continues. **One bad SKU never aborts the run.** |
| Lock TTL expires mid-run (extreme: > 4 hours) | `RedisRunLock.release()` Lua script returns 0 | Logged; the orchestrator does not raise (the work already finished). Indicates the run was unusually slow — investigate. |

#### What runs next

Whichever handler is current. After `IDLE → PRELOADING`, control passes to `handlers.preloading_handler` — Step 5.

### Step 2 — `infrastructure/config.py` (env resolution)

**File:** `code/infrastructure/config.py`

#### Definition

`config.py` is the **single source of truth** for every operating environment value Stage 9 reads. It loads `code/.env` at import time (via `python-dotenv` if installed, else a built-in parser), resolves typed constants, and exports them. Every other module imports from this file — no module ever calls `os.environ.get(...)` directly.

#### Usage

Imported transitively at process startup. The first import triggers `.env` parsing; subsequent imports see the cached module. There is no thread-safety concern because Python module-level code runs exactly once.

#### Problem & Solution

Without a single config module every sub-stage would re-read environment variables, miss type coercion, and risk subtly different values across threads. Spec §17 ("Data Contracts and Integration Points") calls this out as an architectural commitment: every cross-stage configuration MUST flow through one module.

#### Code (annotated)

```python
# infrastructure/config.py — annotated

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"   # 1) code/.env

def _load_env_file(path: Path) -> None:
    """Parse a .env file and populate os.environ for keys not already set."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:                 # 2) env always wins
                os.environ[key] = value

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)                    # 3) prefer python-dotenv
except ImportError:
    _load_env_file(_ENV_FILE)                                 # 4) built-in fallback

# Database
DB_HOST            = os.environ.get("DB_HOST", "localhost")
DB_PORT            = int(os.environ.get("DB_PORT", "5432"))
DB_NAME            = os.environ.get("DB_NAME", "dev")
DB_USER            = os.environ.get("DB_USER", "postgres")
DB_PASSWORD        = os.environ.get("DB_PASSWORD", "")
DB_SSLMODE         = os.environ.get("DB_SSLMODE", "disable")
DB_CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "10"))

DB_DSN = (                                                    # 5) the only DSN
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode={DB_SSLMODE}&connect_timeout={DB_CONNECT_TIMEOUT}"
)

REDIS_URL          = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_POOL_SIZE    = int(os.environ.get("REDIS_POOL_SIZE", "20"))

PLANNING_THREADS      = int(os.environ.get("STAGE9_PLANNING_THREADS", "16"))
ALLOW_FORCE_RELEASE   = os.environ.get("STAGE9_ALLOW_FORCE_RELEASE", "").lower() == "true"
PROJECT_ROOT          = os.environ.get("STAGE9_PROJECT_ROOT", "/mnt/project")
RUN_INTEGRATION_TESTS = os.environ.get("RUN_INTEGRATION_TESTS", "").lower() in ("1","true","yes")
```

Annotation key. **(2)** Process-level environment variables always win over `.env` — operators can override any `.env` value at deploy time without re-publishing the file. **(5)** `DB_DSN` is built once and cached on the module. Every handler imports `DB_DSN` from this module; no handler re-computes it.

#### Outputs

A populated module namespace. Sample resolved values for a local dev shell:

```
DB_DSN          = postgresql://test:test@localhost:5432/test?sslmode=disable&connect_timeout=10
REDIS_URL       = redis://localhost:6379/0
PLANNING_THREADS = 16
PROJECT_ROOT    = M:/stage_9/code
```

#### Verification

```bash
python -c "from infrastructure.config import DB_DSN, REDIS_URL; print(DB_DSN); print(REDIS_URL)"
```

#### Failure modes

| Failure | Detection | Response |
|---|---|---|
| `.env` missing | `_ENV_FILE.exists()` returns False | Silent fallback to environment variables, then to built-in defaults. The defaults match the local Docker compose. |
| Required value missing AND no default provided | `KeyError` at first `os.environ[...]` access | Raises immediately at import — fail fast at boot. |

#### What runs next

The orchestrator imports `DB_DSN` and `REDIS_URL` indirectly through `infrastructure/run_lock.py` and the LangGraph trigger.

### Step 3 — `infrastructure/run_lock.py` (Redis lock)

**File:** `code/infrastructure/run_lock.py`

#### Definition

`RedisRunLock` is a per-tenant Redis-backed mutex with a **dead-man's switch**. The lock is `SET NX EX` for 14400 seconds (4 hours, the `LOCK_TTL_SECONDS` constant). If the holding process crashes, the key auto-expires and the next run can acquire it without manual intervention.

#### Usage

Constructed once inside `orchestrator.run`:

```python
lock = RedisRunLock(redis, tenant_id)
acquired = lock.acquire()
if not acquired:
    raise RunAlreadyInProgressError(tenant_id=tenant_id)
try:
    ... # walk the state machine
finally:
    lock.release()
```

#### Problem & Solution

LangGraph triggers Stage 9 the moment `runs.status='patterns_discovered'` is detected — and it re-checks status periodically. A delayed first run plus a fast retry can produce two concurrent `run()` calls for the same tenant. Both would write to `agent_state_log_s9`, both would race on `forecast_outcomes` upserts, and the resulting state log would contain interleaved transitions that are impossible to interpret.

**Solution.** One Redis key per tenant: `stage9_lock_{tenant_id}`. The acquire is atomic (`SET NX`). The release is atomic via a Lua script that verifies the value before deleting — so a stale process that wakes up after its TTL expired cannot accidentally release a lock that a newer process now holds. The 14400 s TTL is the upper bound on a worst-case enterprise run; if a run actually exceeds 4 hours, the lock expires and a second run can begin — which is the correct behaviour, because at that point the first run is presumed dead.

#### Algorithm walkthrough

```
acquire:
    result = redis.set(key, value=run_token, ex=LOCK_TTL_SECONDS, nx=True)
    return result is not None

release:
    Lua script:
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0   -- not our lock; do nothing
        end
```

The `run_token` is a UUIDv4 generated when `RedisRunLock` is constructed. Two processes that crash at different times never confuse each other's locks.

#### Inputs

- `redis : redis.Redis` — connected client.
- `tenant_id : str` — used to build the key.

#### Outputs

`acquire() → bool`; `release() → None`.

#### Verification

```bash
# Inspect the live lock during a run
redis-cli get stage9_lock_<tenant_id>
redis-cli ttl stage9_lock_<tenant_id>     # should be a number ≤ 14400
```

#### Failure modes

| Failure | Detection | Response |
|---|---|---|
| Redis unreachable | `redis.exceptions.ConnectionError` on `acquire` | Propagated; orchestrator never enters the state machine |
| Process crashes mid-run | TTL expires after 14400 s | Next run acquires cleanly. The first run's partial state log remains; SelfAssessmentEngine flags it on the *next* successful run. |
| Operator wants to release a stuck lock manually | `STAGE9_ALLOW_FORCE_RELEASE=true` in env | A separate `force_release()` method exists; refuses to run unless the env flag is set. |

#### What runs next

`infrastructure/state_machine.transition(IDLE → PRELOADING)` — Step 4.

### Step 4 — `infrastructure/state_machine.py` (state transitions)

**File:** `code/infrastructure/state_machine.py`

#### Definition

`AgentState` is the enum of states (`IDLE`, `PRELOADING`, `PERCEIVING`, `PLANNING`, `ACTING`, `LEARNING`, `REPORTING`, `COMPLETE`, `FAILED`). `transition(conn, tenant_id, run_id, from_state, to_state)` validates the transition against `VALID_TRANSITIONS`, writes a row to `agent_state_log_s9`, and returns the new state.

#### Usage

Called by the orchestrator before every handler invocation. Also called from `orchestrator`'s `except` block when transitioning to `FAILED`. No handler ever calls `transition` directly — that is the orchestrator's job.

#### Problem & Solution

Without a centralized transition validator, a handler that returns successfully when it should have failed (or vice versa) would write a meaningless state log. Worse, a re-triggered run could write `PERCEIVING → ACTING` (skipping PLANNING) and silently lose the calibrated cache. The state log is the audit trail — every transition must be valid.

**Solution.** One function. One whitelist (`VALID_TRANSITIONS`). The function rejects illegal pairs at the application layer before the row is written, so a bad transition never makes it to the database.

#### Algorithm walkthrough

```
1. Validate tenant_id and run_id are UUIDv4.
2. Look up (from_state, to_state) in VALID_TRANSITIONS.
   If not present, raise InvalidTransitionError.
3. Open a cursor; INSERT into agent_state_log_s9 with NOW().
4. COMMIT.
5. Return to_state.
```

`VALID_TRANSITIONS` (from `constants.py`):

```python
VALID_TRANSITIONS = {
    (IDLE,        PRELOADING),
    (PRELOADING,  PERCEIVING),
    (PERCEIVING,  PLANNING),
    (PLANNING,    ACTING),
    (ACTING,      LEARNING),
    (LEARNING,    REPORTING),
    (REPORTING,   COMPLETE),
    (PRELOADING,  FAILED),
    (PERCEIVING,  FAILED),
    (PLANNING,    FAILED),
    (ACTING,      FAILED),
}
```

LEARNING and REPORTING have no edge to FAILED — if they raise, the orchestrator does not transition; the lock is released; the exception propagates to LangGraph.

#### Outputs

Side effect: one row in `agent_state_log_s9 (tenant_id, run_id, from_state, to_state, transitioned_at)`.

#### Verification

```sql
SELECT from_state, to_state, transitioned_at
FROM stage9.agent_state_log_s9
WHERE run_id = '...' ORDER BY transitioned_at;
-- A successful run shows exactly 7 rows: IDLE→PRELOADING, ..., REPORTING→COMPLETE.
```

#### Failure modes

| Failure | Detection | Response |
|---|---|---|
| Invalid transition pair | Raised before any DB write | `InvalidTransitionError` — orchestrator's `except` catches and re-raises |
| DB connection lost | psycopg2 raises | Propagated; orchestrator's `finally` still releases the Redis lock |

#### What runs next

The handler matching the new state.

### Step 5 — `handlers/preloading.py` + `pipeline/preloader.py`

**Files:** `code/handlers/preloading.py`, `code/pipeline/preloader.py`

#### Definition

`preloading_handler` instantiates `Preloader`, runs all 7 bulk reads plus `TenantParams` and `signal_context` loads, and stores a `RunContext` for all subsequent handlers to share. It also resolves the **execution mode** (FULL vs MICRO_UPDATE) by comparing the time since the last completed run against the tenant's `micro_update_threshold_hours`.

#### Usage

Called once per run, immediately after `transition(IDLE → PRELOADING)`. The handler creates the `RunContext` keyed by `run_id` in the in-memory `_RUN_CONTEXTS` registry (`handlers/_context.py`); every downstream handler retrieves it by `run_id`.

#### Problem & Solution

A 10K-SKU run cannot afford one DB round trip per SKU per sub-stage. At 5 reads × 10K SKUs × 6 sub-stages = 300K round trips, the run would take 50 minutes just on network latency. The forecasting math itself takes 8 minutes.

**Solution.** Pay the network cost once. The Preloader issues 7 bulk SELECTs, packages every fact each sub-stage will need into a typed `PreloadedData` dataclass, and stashes it on the `RunContext`. After PRELOADING returns, every read is a Python dict lookup — a Stage 9 run touches the DB exactly *seven times for reads* (Preloader) and a small constant number of times for writes (`BatchWriter` flushes in LEARNING + the Thompson upsert + the self-assessment row).

#### Algorithm walkthrough

The 7 bulk reads (`pipeline/preloader.py`):

| # | Read | Source | Columns | Used by |
|---|---|---|---|---|
| 1 | 3-way JOIN | `stage8.pattern_history × stage8.feature_decisions × stage8.canonical_sku × stage8.signal_context` | `pattern_label`, `confidence_calibrated`, `model_hint`, `obs_days`, `lifecycle_stage`, `composite_confidence`, `drift_detected`, `weekend_zero_ratio`, `criticality_tier`, `service_level_target`, `planned_end_date`, `shelf_life_days`, `parent_style_id`, `seed_daily_demand`, `on_watchlist` | All sub-stages |
| 2 | `stage8.oos_impact_estimates` | Stage 8 | `oos_pct_of_history`, `detection_confidence`, `suppressed_demand_estimate` | OOS adjustment factor (Sub-Stage 9.5) |
| 3 | `stage8.channel_demand_splits` | Stage 8 | `organic_units`, `paid_ratio`, `split_confidence` | Channel-adjusted training data (Sub-Stage 9.2) |
| 4 | `stage8.promo_decisions` | Stage 8 | `promo_weight` per `(sku_id, date)` | Promo-weighted training (Sub-Stage 9.2) |
| 5 | `stage8.portfolio_intelligence_reports` | Stage 8 | `market_shift`, `channel_count_changed` | Structural break detection (Sub-Stage 9.4) |
| 6 | `stage8.tenant_thresholds` | Stage 8 | `confidence_floor`, `confidence_ceiling` | Confidence clamp (Sub-Stage 9.5) |
| 7 | Demand series (per-tenant) | `stage8.demand_history` (a view of `clean_orders`) | Daily `qty` per `(sku_id, sale_date)` | Every model fit |

After the 7 reads:

8. `TenantParams.load(tenant_id, conn)` — one SELECT against `stage9.tenant_learning_params`, returns a 41-row in-memory snapshot.
9. The `signal_context` row is read separately (one tiny SELECT) and stashed on RunContext.

The execution-mode resolution (`_resolve_execution_mode`):

```
last_complete_at = SELECT max(transitioned_at) FROM agent_state_log_s9
                   WHERE tenant_id = %s AND to_state = 'COMPLETE'

threshold = TenantParams.get('micro_update_threshold_hours')   -- default 18

IF (NOW() - last_complete_at) < threshold hours:
    mode = MICRO_UPDATE
ELSE:
    mode = FULL
```

Finally, the fingerprint classifier (Sub-Stage 9.0) runs over every SKU and assigns a `processing_tier` ∈ {`cache`, `partial`, `full`}.

#### Inputs

- `tenant_id`, `run_id`, `db` — standard handler signature.

#### Outputs

A `RunContext` populated with: `preloaded` (the 7-read container), `params` (TenantParams), `signal_ctx`, `execution_mode` (`FULL`/`MICRO_UPDATE`), `processing_tier_by_sku` (the fingerprint output).

#### Verification

```python
from handlers._context import get_run_context
ctx = get_run_context(run_id)
assert ctx.preloaded.demand_series_by_sku  # populated
assert ctx.params.get(Param.MICRO_UPDATE_THRESHOLD_HOURS) > 0
print(len(ctx.preloaded.demand_series_by_sku), 'SKUs preloaded')
```

#### Failure modes

| Failure | Detection | Response |
|---|---|---|
| Stage 8 hasn't run for this tenant | `pattern_history` empty | `preloading_handler` raises; orchestrator transitions to FAILED |
| `tenant_learning_params` not seeded | `TenantParams.load` returns empty dict | `TenantParamNotFoundError` — propagated to orchestrator |
| Memory pressure on 100K-SKU tenant | `Preloader` allocates ~500 MB of DataFrames | Document the requirement; tenant SLA bumps to 32 GB worker. |

#### What runs next

`transition(PRELOADING → PERCEIVING)` then `perceiving_handler` — Step 8.

### Step 6 — `forecasting/fingerprint.py` (Sub-Stage 9.0)

**File:** `code/forecasting/fingerprint.py`

#### Definition

Sub-Stage 9.0 — **Data Fingerprinting and Processing Tier Classification**. For each SKU, computes a SHA-256 fingerprint over the last 7 days of demand plus the `pattern_label`, looks up the prior fingerprint in `data_fingerprint_cache`, and assigns one of three tiers: `cache`, `partial`, `full`.

#### Usage

Called from `preloading_handler` after the 7 bulk reads complete. The output is stored in `RunContext.processing_tier_by_sku` and consumed by `tier_router` during ACTING.

#### Problem & Solution

Most SKUs in a daily run have not changed since yesterday. Re-running the full 9.1→9.5 chain on every SKU is wasted work. But "did this SKU change?" is not a single-bit question — a SKU whose 7-day average shifted by 1% deserves a quick refresh, while a SKU whose pattern label flipped from `stable` to `intermittent` deserves the full pipeline.

**Solution.** Three buckets:

- `cache` — fingerprint identical AND pattern_label unchanged AND prior backtest MAPE ≤ tenant's cache threshold. **Skip everything; reuse the prior forecast.**
- `partial` — fingerprint differs slightly (7-day mean shifted < 5%). **Refit with current HP; skip Thompson sampling.**
- `full` — fingerprint differs materially OR no prior record OR pattern_label changed. **Run everything.**

The fingerprint's payload is `pattern_label || sha256(demand_last_7_days_quantized_to_int)`. Quantizing to int makes the hash insensitive to floating-point round-off.

#### Algorithm walkthrough

```
for each sku in preloaded.skus:
    series = demand_series_by_sku[sku][-7:]
    if len(series) < 7:
        tier[sku] = 'full'                        # not enough history
        continue

    quantized = tuple(int(round(x)) for x in series)
    payload = f"{pattern_label[sku]}|{quantized}"
    fp = sha256(payload.encode()).hexdigest()

    prior = SELECT fingerprint, tier, pattern_label, backtest_mape
            FROM data_fingerprint_cache WHERE tenant_id=%s AND sku_id=%s

    if prior is None:
        tier[sku] = 'full'                        # cold start
    elif fp == prior.fingerprint and pattern_label[sku] == prior.pattern_label \
            and prior.backtest_mape is not None \
            and prior.backtest_mape <= cache_mape_ceiling:
        tier[sku] = 'cache'
    elif abs(mean(series) - prior.mean_7d) / prior.mean_7d < 0.05:
        tier[sku] = 'partial'
    else:
        tier[sku] = 'full'

UPSERT every sku's fingerprint back into data_fingerprint_cache
```

#### Outputs (4 worked examples, run-2 with same input)

| SKU | Tier | Why |
|---|---|---|
| DOC-CS | `full` | `obs_days < 28` forces full regardless of fingerprint match |
| DOC-STB | `cache` | Identical fingerprint; pattern_label unchanged; prior MAPE 0.12 ≤ ceiling 0.30 |
| DOC-TRN | `cache` | Identical fingerprint; pattern_label unchanged; prior MAPE 0.18 ≤ ceiling 0.30 |
| DOC-INT | `partial` | Pattern_label `intermittent` produces noisier fingerprints; 7-day mean shift ≈ 3% |

#### Verification

```sql
SELECT sku_id, tier, pattern_label, backtest_mape
FROM stage9.data_fingerprint_cache
WHERE tenant_id = '...'
ORDER BY tier;
```

#### What runs next

The output is consumed by `tier_router` in Step 7.

### Step 7 — `forecasting/tier_router.py` (3-track dispatch)

**File:** `code/forecasting/tier_router.py`

#### Definition

`tier_router.dispatch(sku_id, ctx)` looks up the SKU's tier in `RunContext.processing_tier_by_sku` and returns a callable representing the correct sub-stage chain:

- `full` → `9.1 → 9.2 → 9.3 → closures(9.4, 9.5)`
- `partial` → `9.1 → 9.2 → closures(9.4, 9.5)` *(skips 9.3; uses best Thompson HP)*
- `cache` → `warm_start_from_prior_forecast`

#### Usage

Called per SKU inside `acting_handler`'s pool worker. The closure returned is invoked with the per-SKU demand series; the result is queued via `BatchWriter`.

#### Problem & Solution

Without a router, every SKU would either run the full chain (wasteful) or each sub-stage would have to know about every tier (sub-stage code becomes a tangle of `if tier == ...` branches). The router centralizes the dispatch policy in one file; sub-stages remain simple.

#### Outputs (4 worked examples)

| SKU | Tier | Chain |
|---|---|---|
| DOC-CS | `full` | 9.1 → 9.2 → 9.3 → 9.4 → 9.5 |
| DOC-STB | `cache` | warm-start from prior; SES level update only |
| DOC-TRN | `full` (run-1) / `cache` (run-2) | 9.1 → 9.2 → 9.3 → 9.4 → 9.5 / warm-start |
| DOC-INT | `full` (run-1) / `partial` (run-2) | full chain / 9.1 → 9.2 → 9.4 → 9.5 |

#### What runs next

The closure returned by the router. For `full` tier, that's Sub-Stage 9.1 — Step 11.

### Step 8 — `handlers/perceiving.py` (TenantParams snapshot)

**File:** `code/handlers/perceiving.py`

#### Definition

`perceiving_handler` confirms the `TenantParams` snapshot loaded in PRELOADING is well-formed (raises `TenantParamNotFoundError` if not), and PEEKs (read-only, never marks `consumed`) the latest `pattern_confidence` signal from Stage 8 — used as input to the confidence formula in Sub-Stage 9.5.

#### Usage

Called immediately after `transition(PRELOADING → PERCEIVING)`. Reads from `RunContext`; writes nothing.

#### Problem & Solution

Stage 8 emits `pattern_confidence` signals when its pattern classifier output is unstable for a SKU. Stage 9's confidence formula multiplies its own confidence by Stage 8's confidence — but only if a recent signal exists. Without an explicit PEEK step, every sub-stage would have to issue its own SELECT.

**Solution.** Read once, stash on `RunContext.recent_pattern_signals`, consume from memory.

#### Code (annotated)

```python
def perceiving_handler(*, tenant_id, run_id, db):
    ctx = get_run_context(run_id)

    if not ctx.params._values:                            # 1) verify TenantParams populated
        raise TenantParamNotFoundError(tenant_id=tenant_id)

    consumer = SignalConsumer(db, tenant_id=tenant_id)
    ctx.recent_pattern_signals = consumer.peek_signals(   # 2) PEEK = no consumed=TRUE write
        signal_type='pattern_confidence',
        max_age_hours=24,
    )
```

Annotation **(2)**: PEEK semantics matter. Stage 9 reads Stage 8's signals; only Stage 8's own consumer marks them `consumed=TRUE`. PEEK lets Stage 9 read without affecting Stage 8's downstream visibility.

#### What runs next

`transition(PERCEIVING → PLANNING)` then `planning_handler` — Step 9.

### Step 9 — `handlers/planning.py` (cache pre-fetch + dual-pool split)

**File:** `code/handlers/planning.py`

#### Definition

`planning_handler` has two responsibilities:

1. **Pre-fetch the two tenant-wide caches** that Sub-Stages 9.4 and 9.5 need: the calibrated backtest-window cache (`(pattern, model) → window_days`) and the Thompson HP cache (`(pattern, model) → best_alpha`). Both are read with one bulk SELECT each, replacing N per-SKU round trips.
2. **Split the SKU list into two pools**: the `ProcessPool` queue (Prophet, NeuralProphet) and the `ThreadPool` queue (Naive, SES, Croston, Holt). The split is purely by `model_hint` — there is no global state at this point.

#### Usage

Called immediately after `transition(PLANNING → ACTING)` is *about* to fire — the planning step prepares the caches, then the orchestrator transitions. The two queues are stashed on `RunContext.process_pool_skus` and `RunContext.thread_pool_skus`.

#### Problem & Solution

Without the calibrated cache, every SKU in Sub-Stage 9.4's `select_backtest_window` would issue:

```sql
SELECT calibrated_window_days
FROM stage9.tenant_learning_params
WHERE tenant_id=%s AND param_name = 'backtest_window_' || %s || '_' || %s
```

— at 10K SKUs that's 10K round trips. The planning layer issues one SELECT, builds a `dict[(pattern, model)] → days`, and Sub-Stage 9.4 does a Python lookup.

**Solution.** Pay the network cost once. The two caches live on `RunContext.calibrated_cache` and `RunContext.thompson_hp_cache` for the duration of the run.

#### What runs next

`transition(PLANNING → ACTING)` then `acting_handler` — Step 10.

### Step 10 — `handlers/acting.py` + `pipeline/dual_pool.py`

**Files:** `code/handlers/acting.py`, `code/pipeline/dual_pool.py`

#### Definition

`acting_handler` is the heart of Stage 9. It runs the per-SKU pipeline (9.1 → 9.5) for every SKU concurrently across two executor pools. Three execution tracks share the function:

1. **MICRO_UPDATE** mode (run-level) — SES level correction only; no model retrain. Triggered by the FULL/MICRO_UPDATE flag set in PRELOADING.
2. **CACHE** tier (per-SKU) — warm-start from the prior forecast.
3. **FULL/PARTIAL** tier (per-SKU) — the full chain.

`pipeline/dual_pool.py` owns the executor split:

- `ProcessPoolExecutor` — Prophet/NeuralProphet SKUs, 4 workers, 120 s per-SKU timeout. Process isolation is required because Stan and Torch don't play well with Python threads.
- `ThreadPoolExecutor` — everything else, 16 workers, 30 s per-SKU timeout.

Both pools start simultaneously and `as_completed()` drains them in parallel.

#### Problem & Solution

Without dual-pool concurrency, a 5,000-SKU catalog with 20% seasonal SKUs takes ~40 minutes (Prophet is the bottleneck, and Python's GIL serializes thread-based fits). With both pools running together, the wall-clock is bounded by `max(prophet_time, thread_pool_time)` rather than their sum.

**Solution.** Two pools. One queue per pool. `as_completed()` collects results as they finish. Per-SKU exceptions are caught inside the worker (`_per_sku_fallback`) so one bad SKU never aborts the run — Principle 3.

#### Algorithm walkthrough

```
1. Drain RunContext.thread_pool_skus into ThreadPoolExecutor(max_workers=16)
2. Drain RunContext.process_pool_skus into ProcessPoolExecutor(max_workers=4)
3. For each future as it completes:
     try:
         sku_result = future.result(timeout=...)
     except (TimeoutError, Exception) as e:
         sku_result = _per_sku_fallback(sku_id, e)        # Naive forecast + log
     batch_writer.queue('forecasts', sku_result.row)
4. After all futures complete, return.
```

`_per_sku_fallback`:

```
1. Log the exception (stage9.stage9_sku_execution_log).
2. Build a NaiveForecast row using the last 30 days' mean.
3. Increment the run-level failure counter (visible in self_assessment).
4. Emit a 'pattern_feedback' signal with the failure reason — Stage 8 may
   adjust the SKU's pattern label based on repeated failures here.
```

#### Outputs (4 worked examples)

| SKU | Track | Time | Output row |
|---|---|---|---|
| DOC-CS | FULL (cold-start branch) | ~30 ms | `assigned_model=Naive`, `confidence=0.42`, `forecast_30d.mean=240` |
| DOC-STB | CACHE (run-2) | ~5 ms | reuse prior forecast; `processing_tier='cache'` |
| DOC-TRN | FULL → Holt | ~80 ms | `assigned_model=Holt`, `confidence=0.71`, `forecast_30d.mean=305` |
| DOC-INT | FULL → Croston | ~120 ms | `assigned_model=Croston`, `confidence=0.45`, `forecast_30d.mean=120`, wide quantiles |

#### What runs next

`transition(ACTING → LEARNING)` then `learning_handler` — Step 18.

### Step 11 — `pipeline/model_initialization.py` (Sub-Stage 9.1)

**File:** `code/pipeline/model_initialization.py`

#### Definition

Sub-Stage 9.1 — **Model Initialisation**. Makes 7 ordered decisions for each FULL-tier SKU and outputs a `LearningContext` dataclass consumed by Sub-Stages 9.2–9.5.

#### Algorithm walkthrough — the 7 decisions

| # | Decision | Logic | Source of truth |
|---|---|---|---|
| 1 | Model assignment | `PATTERN_MODEL_MAP` — `cold_start→Naive`, `stable→SES`, `trending→Holt`, `seasonal→Prophet`, `intermittent→Croston` | `infrastructure/constants.py` |
| 2 | Quantile selection | `cold_start→0.90`, `intermittent→0.90`, otherwise→0.80 | TenantParams |
| 3 | Learning mode | `explore` if `obs_days < min_history_for_exploit`, else `exploit` | TenantParams |
| 4 | OOS adjustment factor | `1 + (oos_pct × detection_confidence)` capped at `oos_uplift_cap` (default 1.50) | preloader's read 2 |
| 5 | Reorder bias factor | derived from `cross_agent_signals` `reorder_outcome` history; default 1.00 | preloader's read 7 |
| 6 | `is_b2b` | from `canonical_sku.product_type` | preloader's read 1 |
| 7 | Effective max horizon | `min(365, planned_end_date - today)` if `planned_end_date` set, else 365 | preloader's read 1 |

The output `LearningContext` carries every decision plus the SKU-id, run-id, tenant-id, the demand DataFrame, and the configured (but not yet fit) model instance.

#### Outputs (4 worked examples)

| SKU | Model | Quantile | Mode | OOS factor | Reorder bias | Max horizon |
|---|---|---|---|---|---|---|
| DOC-CS | Naive | 0.90 | explore | 1.00 | 1.00 | 365 |
| DOC-STB | SES | 0.80 | exploit | 1.00 | 1.00 | 365 |
| DOC-TRN | Holt | 0.80 | exploit | 1.05 | 1.00 | 365 |
| DOC-INT | Croston | 0.90 | exploit | 1.00 | 1.00 | 365 |

#### What runs next

Sub-Stage 9.2 — Step 12.

### Step 12 — `forecasting/feature_engg.py` (Sub-Stage 9.2)

**File:** `code/forecasting/feature_engg.py`

#### Definition

Sub-Stage 9.2 — **Feature Engineering**. Prepares the **promo-weighted training data** and selects the **optimal feature set** for the assigned model. Runs after 9.1 and before 9.3. Four steps run in order; each may fail independently — log + fall back, never raise.

#### Algorithm walkthrough — the 4 steps

1. **Channel split** — if multi-channel and `split_confidence ≥ 0.50`, replace `qty` with `organic_units`; mark `channel_adjusted=True`.
2. **Promo weighting** — for each `(sku, date)` with a `promo_decisions` row, multiply `qty` by `(1 - promo_weight)` so promo-driven peaks don't bias the trend.
3. **OOS masking** — for each date in `oos_impact_estimates`, multiply `qty` by `(1 + adjustment_factor)`. Already computed in 9.1; applied per-row here.
4. **Feature selection** — read `feature_reliability_map` from the 3-way JOIN; drop any feature whose reliability score is below `feature_reliability_floor` (default 0.30).

#### Outputs

A modified DataFrame ready for the model's `fit()`, plus a `features_used` list that ends up in `forecasts.features_used` for audit.

#### What runs next

Sub-Stage 9.3 — Step 13.

### Step 13 — `models/hp_tuning.py` + `models/thompson.py` (Sub-Stage 9.3)

**Files:** `code/models/hp_tuning.py`, `code/models/thompson.py`

#### Definition

Sub-Stage 9.3 — **Hyperparameter Tuning via Thompson Sampling**. Finds the optimal HP configuration for each FULL-tier SKU's assigned model. All SKUs go through standard Thompson Sampling regardless of lifecycle stage.

#### Math — Thompson sampling

Thompson Sampling maintains a **Beta(α, β)** distribution for each HP configuration. Each run, it samples once from each distribution and tests the highest-sampled configs. Configs that outperform the prior baseline earn a "win" (`α += 1`); configs that underperform earn a "loss" (`β += 1`).

For `n_configs = K` candidate HPs:

```
for k in range(K):
    sample[k] = Beta(alpha[k], beta[k]).rvs()

ranked = argsort(sample, descending=True)
test_configs = ranked[:exploration_budget]      # default 3

for config in test_configs:
    fit model with config
    backtest_mape = run_quick_backtest(model)
    if backtest_mape < baseline_mape:
        alpha[config] += 1
    else:
        beta[config] += 1

best_config = argmax(alpha / (alpha + beta))    # mean of Beta
```

The bandit is per-`(tenant, sku, model)`. State persists in `stage9.thompson_sampling_state` (one row per tuple). LEARNING handler bulk-upserts the deltas at end-of-run.

#### Outputs (4 worked examples)

| SKU | Model | Best HP after run-1 | α / (α+β) |
|---|---|---|---|
| DOC-CS | Naive | n/a (Naive has no HP) | — |
| DOC-STB | SES | `alpha=0.30` | 0.67 |
| DOC-TRN | Holt | `alpha=0.20, beta=0.10` | 0.71 |
| DOC-INT | Croston | `alpha=0.10` | 0.55 |

#### What runs next

Sub-Stage 9.4 — Step 14.

### Step 14 — `backtesting/backtesting.py` (Sub-Stage 9.4)

**File:** `code/backtesting/backtesting.py`

#### Definition

Sub-Stage 9.4 — **Backtesting and `pattern_feedback`**. Five ordered steps:

1. **`select_backtest_window`** — pick the window length, preferring the adaptive calibrated value for `(tenant, pattern, model)` from the planning-layer cache, with overrides for ultra-sparse / short-history / exploit-mode tenants.
2. **`run_backtest`** — walk-forward fit on `[0:n-window]`, predict the held-out `[n-window:n]`, compute MAPE.
3. **`detect_exceptions`** — six exception flags: `stockout_3_consecutive_zeros`, `promo_spike`, `unusual_drop`, `high_volatility`, `high_mape`, `structural_break`.
4. **`write_pattern_feedback`** — for every SKU (including failures), write one row to `stage8.pattern_feedback` with `forecast_error_mape`, `feedback_type`, `model_used`, `classification_quality` (good/acceptable/poor), `hint_matched` (was Stage 8's `model_hint` actually selected), and `fallback_used` (did Sub-Stage 9.1 fall back to Naive).
5. **`commit_pattern_feedback`** — explicit COMMIT before the BatchWriter flush, so a downstream LEARNING failure never loses Stage 8's feedback.

#### Algorithm walkthrough — `select_backtest_window`

```
calibrated = ctx.calibrated_cache.get((pattern, model))
default    = TenantParams.get('default_backtest_window_days')   # default 30

if calibrated is not None:
    window = calibrated
else:
    window = default

# Override for ultra-sparse SKUs
if obs_days < window * 1.5:
    window = max(min_backtest_window, obs_days // 3)

# Override for exploit-mode established tenants
if learning_mode == 'exploit' and tenant_maturity == 'established':
    window = max_backtest_window      # use the longest available

window = clamp(window, min_backtest_window, max_backtest_window)
```

#### Outputs (4 worked examples)

| SKU | Window | Backtest MAPE | Classification |
|---|---|---|---|
| DOC-CS | n/a (`obs_days < 28`) | NaN — proxy MAPE used | n/a |
| DOC-STB | 28 days | 0.12 | good |
| DOC-TRN | 28 days | 0.18 | acceptable |
| DOC-INT | 30 days | 0.42 | poor |

#### What runs next

Sub-Stage 9.5 — Step 15.

### Step 15 — `forecasting/forecasting.py` + `models/bootstrap.py` (Sub-Stage 9.5)

**Files:** `code/forecasting/forecasting.py`, `code/models/bootstrap.py`

#### Definition

Sub-Stage 9.5 — **Forecast Generation and Confidence**. Five ordered steps per SKU:

1. **`generate_horizons`** — fit the assigned model and produce a point forecast at each of the 8 locked HORIZONS `[7, 14, 30, 60, 90, 150, 180, 365]`. Prophet/NeuralProphet do ONE 365-day fit and read cumulative sums at each boundary; SES/Holt/Croston/Naive fit once and call `predict_all_horizons()`.
2. **`bootstrap_quantiles`** — convert each point forecast to `{mean, p50, p80, p90}` via residual bootstrap (5-step algorithm in `models/bootstrap.py`). Pattern-aware uncertainty factor scales the residual standard deviation.
3. **`apply_oos_adjustment`** — multiply every quantile by `oos_adjustment_factor`.
4. **`apply_reorder_bias`** — multiply by `reorder_bias_factor`.
5. **`compute_confidence`** — the 5-step multiplicative formula in `confidence.py` (Step 16).

#### Math — the bootstrap (`models/bootstrap.py`)

```
1. residuals = (y_train - y_pred_in_sample)
2. uncertainty_factor = PATTERN_UNCERTAINTY[pattern_label]
       cold_start: 1.5, stable: 0.8, trending: 1.0, intermittent: 1.4, seasonal: 1.1
3. residual_std = std(residuals) * uncertainty_factor
4. samples = point_forecast + residual_std * randn(n_samples=1000)
5. quantiles = {
       'mean': samples.mean(),
       'p50':  np.percentile(samples, 50),
       'p80':  np.percentile(samples, 80),
       'p90':  np.percentile(samples, 90),
   }
   # Clip to [0, ∞) — negative demand is impossible.
```

Always `mean ≈ point_forecast` (because the noise is symmetric around zero) and always `p50 ≤ p80 ≤ p90`. The bootstrap test suite verifies this invariant on 100 random demand series.

#### Outputs (4 worked examples — `forecast_30d`)

| SKU | mean | p50 | p80 | p90 |
|---|---|---|---|---|
| DOC-CS | 240 | 230 | 280 | 305 |
| DOC-STB | 600 | 600 | 625 | 640 |
| DOC-TRN | 305 | 305 | 320 | 330 |
| DOC-INT | 120 | 110 | 165 | 195 |

#### What runs next

Step 16 (confidence) is technically part of Sub-Stage 9.5 but factored into its own file for unit-testability.

### Step 16 — `forecasting/confidence.py` (5-step formula)

**File:** `code/forecasting/confidence.py`

#### Definition

`compute_confidence(ctx)` is the 5-step multiplicative confidence formula. Each step is a learned floor/ceiling pair from `tenant_learning_params`. Extracted from `forecasting.py` so each step can be unit-tested in isolation.

#### Math — the 5 steps

```
confidence = 1.0
confidence *= step_1_pattern_confidence(stage8_pattern_confidence)
confidence *= step_2_history_factor(obs_days, lifecycle_stage)
confidence *= step_3_backtest_factor(backtest_mape, mape_floor)
confidence *= step_4_volatility_factor(cv_of_demand)
confidence *= step_5_drift_factor(drift_detected)
confidence = clamp(confidence, tenant.confidence_floor, tenant.confidence_ceiling)
```

Each step is a smooth piecewise-linear function — no `if`/`else` branches that produce discontinuities. The clamp is the final guardrail: a tenant whose `confidence_floor=0.20` and `confidence_ceiling=0.95` will never see `confidence_final` outside `[0.20, 0.95]`.

#### Outputs (4 worked examples)

| SKU | step 1 | step 2 | step 3 | step 4 | step 5 | final |
|---|---|---|---|---|---|---|
| DOC-CS | 0.85 | 0.55 | 1.00 (no backtest) | 1.00 | 1.00 | 0.42 (clamped to floor) |
| DOC-STB | 0.92 | 0.95 | 0.98 | 0.95 | 1.00 | 0.81 |
| DOC-TRN | 0.88 | 0.92 | 0.92 | 0.96 | 0.95 | 0.71 |
| DOC-INT | 0.78 | 0.92 | 0.65 | 0.85 | 1.00 | 0.45 |

#### What runs next

The `forecasts` row is queued via `BatchWriter`. Step 17.

### Step 17 — `infrastructure/batch_writer.py` (deferred writes)

**File:** `code/infrastructure/batch_writer.py`

#### Definition

`BatchWriter` is a thread-safe row buffer keyed by table name. Handlers call `queue('forecasts', row_dict)`; the writer accumulates rows; `LEARNING` calls `flush()`, which issues one `INSERT ... ON CONFLICT ... DO UPDATE` per table using `psycopg2.extras.execute_values`.

#### Problem & Solution

Without batching, every per-SKU `INSERT` would be its own round trip — 10K SKUs = 10K writes. With `execute_values` chunking (default 500 per chunk), the same 10K rows commit in 20 chunks.

#### What runs next

`transition(ACTING → LEARNING)` — the BatchWriter is flushed by `learning_handler`. Step 18.

### Step 18 — `handlers/learning.py` (Thompson upsert)

**File:** `code/handlers/learning.py`

#### Definition

`learning_handler` is the post-run learning step. Three jobs:

1. **Flush BatchWriter** — every queued row across every table.
2. **Bulk-upsert Thompson state** — Sub-Stage 9.3 accumulated α/β deltas in memory; this is the single bulk write to `thompson_sampling_state`.
3. **Refresh `sku_similarity_registry`** — for SKUs whose validation MAPE is good enough (≤ similarity-mape ceiling), upsert them into the registry as warm-start references for future cold-start SKUs.

#### Problem & Solution

If Sub-Stage 9.3 wrote Thompson updates per SKU, a 10K-SKU run would issue 10K UPDATEs. Pulling them all into memory and writing once at LEARNING reduces that to one bulk statement — typically < 100 ms.

#### What runs next

`transition(LEARNING → REPORTING)` then `reporting_handler` — Step 19.

### Step 19 — `handlers/reporting.py` + `learning/self_assessment.py`

**Files:** `code/handlers/reporting.py`, `code/learning/self_assessment.py`

#### Definition

`reporting_handler` is the final live step. Three jobs:

1. **Run `SelfAssessmentEngine`** — reads `model_performance_s9` to detect models that have degraded since the last run; computes run statistics from the in-memory `SKUResult` list (count by tier, count by model, avg MAPE, …); writes one `stage9_self_assessment` row.
2. **Emit `model_health` broadcast signal** — payload: `{count_full, count_partial, count_cache, avg_mape, models_degraded}`.
3. **Drop the `RunContext`** from the in-memory registry — frees memory for the next run.

#### What runs next

`transition(REPORTING → COMPLETE)`. `orchestrator.run`'s `finally` block releases the Redis lock. The run is done.

### Step 20 — `signals/emitter.py` + `signals/consumer.py` (signal bus)

**Files:** `code/signals/_base.py`, `code/signals/emitter.py`, `code/signals/consumer.py`

#### Definition

The cross-agent signal bus is a thin wrapper around `stage9.cross_agent_signals`. Two halves:

- **`SignalEmitter`** — direct-write half. NOT thread-safe — each thread/worker constructs its own emitter with its own psycopg2 connection. In `dual_pool`'s ThreadPoolExecutor, instantiate per worker — never share a single emitter across the 16 thread slots.
- **`SignalConsumer`** — read-side. Two methods: `peek_signals` (SELECT without modifying `processed`; thread-safe), and `consume_signals` (atomic `FOR UPDATE SKIP LOCKED` + `UPDATE processed=TRUE`).

#### Problem & Solution

Without `FOR UPDATE SKIP LOCKED`, two consumers reading the same backlog would both process the same rows. Without PEEK semantics, a downstream stage that wants to read a signal without affecting the producer's visibility (like Stage 9 reading Stage 8's `pattern_confidence`) couldn't do so.

#### What runs next

Signals are read on the next run by whichever stage owns the consumer (Stage 9 PEEKs Stage 8's `pattern_confidence`; Stage 10 consumes Stage 9's `model_health`).

### Step 21 — Nightly Batch Jobs (3 AM / 4 AM / 4:30 AM)

**Files:** `code/learning/outcome_collector.py`, `code/learning/model_performance_aggregator.py`, `code/learning/learning_params_updater.py`

These jobs are **not on the hot path** — they run nightly via cron / Airflow / external scheduler. Each is independent; each is idempotent.

#### 21.1 `outcome_collector.py` — 3 AM UTC

For every `forecasts` row whose horizon period has now closed (i.e. `forecast_date + horizon_days ≤ today`), compare the forecast to the actual sales over that period. Write a `forecast_outcomes` row containing `forecast_value`, `actual_value`, `error_mape`, `error_wape`, `bias`. This is the **learning signal** that drives every downstream batch job.

Key columns:

```sql
INSERT INTO forecast_outcomes (
    tenant_id, sku_id, run_id, horizon_days,
    assigned_model, forecast_value, actual_value,
    error_mape, error_wape, bias, outcome_date
) VALUES (...)
ON CONFLICT (tenant_id, sku_id, run_id, horizon_days) DO NOTHING;
```

Idempotent via the unique constraint — re-running the job for the same day produces no duplicates and no errors.

#### 21.2 `model_performance_aggregator.py` — 4 AM UTC

For each `(tenant, assigned_model, horizon_days)`:

```
SELECT
    avg(error_mape)    AS avg_mape,
    median(error_mape) AS median_mape,
    avg(bias)          AS avg_bias,
    count(*)           AS sample_count
FROM forecast_outcomes
WHERE tenant_id = %s
  AND outcome_date >= NOW() - INTERVAL '30 days'
GROUP BY assigned_model, horizon_days
```

Then computes **stable_band** (the `[avg_mape - σ, avg_mape + σ]` interval) and writes the rolled-up rows to `model_performance_s9`. `SelfAssessmentEngine` reads this table during REPORTING to detect degrading models.

#### 21.3 `learning_params_updater.py` — 4:30 AM UTC

Reads `forecast_outcomes`, `adaptive_quantile_state`, and `cross_agent_signals`, and nudges `tenant_learning_params` toward observed evidence. Examples:

- If `avg_bias > 0` consistently for the last 30 days, nudge `selected_quantile` upward by `quantile_step` (default 0.02).
- If `avg_mape > stable_band_upper` for a model, nudge `confidence_step_3_floor` downward (penalize that model's confidence contribution).
- If `cross_agent_signals` show repeated stockouts, nudge `reorder_bias_factor` upward.

Each nudge is bounded by a learned floor/ceiling and clamped to a hard rail (the locked invariants — see R5).

---

## Part III — Reference

### R1. TenantParams — Full Catalog (54 rows)

Every numeric threshold, rate, buffer and exploration budget is stored in `stage9.tenant_learning_params` and read through `TenantParams.get(...)`. The 54-row catalog:

#### R1.1 Confidence formula floors and ceilings

| `param_name` | Default | Used in |
|---|---|---|
| `confidence_floor` | 0.20 | clamp at end of 5-step formula |
| `confidence_ceiling` | 0.95 | clamp at end of 5-step formula |
| `confidence_step_1_floor` | 0.50 | step 1 (pattern confidence) |
| `confidence_step_2_floor` | 0.40 | step 2 (history factor) |
| `confidence_step_3_floor` | 0.30 | step 3 (backtest MAPE) |
| `confidence_step_4_floor` | 0.55 | step 4 (volatility) |
| `confidence_step_5_floor` | 0.70 | step 5 (drift) |

#### R1.2 Backtest

| `param_name` | Default | Used in |
|---|---|---|
| `default_backtest_window_days` | 30 | `select_backtest_window` |
| `min_backtest_window` | 14 | `select_backtest_window` clamp |
| `max_backtest_window` | 90 | `select_backtest_window` clamp |
| `mape_floor` | 0.05 | confidence step 3 |
| `cache_mape_ceiling` | 0.30 | fingerprint tier eligibility |
| `similarity_mape_ceiling` | 0.25 | similarity registry eligibility |

#### R1.3 Thompson sampling

| `param_name` | Default | Used in |
|---|---|---|
| `thompson_exploration_budget` | 3 | configs tested per run |
| `thompson_min_runs_to_exploit` | 5 | exploit mode lockout |
| `thompson_min_history_for_exploit` | 60 | days of obs |

#### R1.4 OOS / promo / channel

| `param_name` | Default | Used in |
|---|---|---|
| `oos_uplift_cap` | 1.50 | OOS adjustment factor cap |
| `promo_weight_max` | 3.0 | promo decision saturation |
| `channel_split_threshold` | 0.50 | min `split_confidence` to apply organic |

#### R1.5 Run cadence

| `param_name` | Default | Used in |
|---|---|---|
| `micro_update_threshold_hours` | 18 | FULL vs MICRO_UPDATE resolution |
| `lock_ttl_seconds` | 14400 | RedisRunLock TTL |

#### R1.6 Quantile selection

| `param_name` | Default | Used in |
|---|---|---|
| `quantile_default` | 0.80 | default selected quantile |
| `quantile_cold_start` | 0.90 | cold-start override |
| `quantile_intermittent` | 0.90 | intermittent override |
| `quantile_step` | 0.02 | nightly nudge increment |
| `quantile_min` | 0.50 | learner clamp |
| `quantile_max` | 0.99 | learner clamp |

#### R1.7 Aggregator stable bands

| `param_name` | Default | Used in |
|---|---|---|
| `model_performance_stable_band` | 0.05 | stable_band σ |
| `model_performance_lookback_days` | 30 | aggregation window |

#### R1.8 Confidence — additional learned values (≈25 more rows)

Floors and ceilings for each pattern_label, learned uncertainty factors, drift thresholds, lifecycle multipliers — all stored in tenant_learning_params under stable param_name keys. Full catalog in `infrastructure/tenant_params_defaults.py`.

### R2. Forecasting Models Catalog

| Model | File | Use case | HP space (Thompson) |
|---|---|---|---|
| `NaiveForecast` | `models/naive.py` | cold_start (`obs_days < 28`) | none |
| `SESModel` | `models/ses.py` | stable | `alpha ∈ {0.10, 0.20, 0.30, 0.40, 0.50}` |
| `HoltLinearTrend` | `models/holt.py` | trending | `(alpha, beta)` × {0.20, 0.30, 0.40} × {0.05, 0.10, 0.20} |
| `CrostonMethod` | `models/croston.py` | intermittent | `alpha ∈ {0.05, 0.10, 0.15, 0.20}` |
| `ProphetModel` | `models/prophet_model.py` | seasonal | `changepoint_prior_scale ∈ {0.001, 0.01, 0.05}`, `seasonality_prior_scale ∈ {0.1, 1.0, 10.0}` |

Every model inherits `BaseModel` (`models/base.py`) and must implement: `fit(df)`, `predict_all_horizons() -> dict[h_days → point]`, and the `assigned_model` property. The bootstrap (`models/bootstrap.py`) is shared — not a method on any model.

### R3. Output Contract with Stage 10

Stage 10 reads `stage9.forecasts` directly. The contract is:

```sql
CREATE TABLE stage9.forecasts (
    tenant_id              UUID NOT NULL,
    sku_id                 UUID NOT NULL,
    run_id                 UUID NOT NULL,
    processing_tier        VARCHAR NOT NULL,   -- 'cache' / 'partial' / 'full'
    assigned_model         VARCHAR NOT NULL,
    confidence_final       NUMERIC NOT NULL,
    backtest_mape          NUMERIC,            -- NULL when obs_days < 28
    pattern_label          VARCHAR NOT NULL,
    selected_quantile      NUMERIC NOT NULL,
    effective_max_horizon  INTEGER NOT NULL,
    oos_adjustment_factor  NUMERIC NOT NULL,
    reorder_bias_factor    NUMERIC NOT NULL,
    is_b2b                 BOOLEAN NOT NULL,
    forecast_7d            JSONB NOT NULL,     -- {mean, p50, p80, p90}
    forecast_14d           JSONB NOT NULL,
    forecast_30d           JSONB NOT NULL,
    forecast_60d           JSONB NOT NULL,
    forecast_90d           JSONB NOT NULL,
    forecast_150d          JSONB NOT NULL,
    forecast_180d          JSONB NOT NULL,
    forecast_365d          JSONB NOT NULL,
    features_used          JSONB,
    created_at             TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, sku_id, run_id)
);
```

### R4. Database Schema Reference (`stage9.*`)

| Table | Owner | Purpose |
|---|---|---|
| `forecasts` | acting_handler | Per-SKU per-run forecast output (Stage 10's input) |
| `thompson_sampling_state` | learning_handler | Beta(α, β) per `(tenant, sku, model, config_hash)` |
| `feature_decisions_s9` | acting_handler | Which features each SKU used; warm-start input |
| `hyperparameter_decisions` | acting_handler | Audit log of Thompson choices per SKU per run |
| `backtest_decisions` | acting_handler | Per-SKU backtest record (window, MAPE, exception flags) |
| `data_fingerprint_cache` | preloading_handler | Per-SKU SHA-256 fingerprint + tier classification |
| `sku_similarity_registry` | learning_handler | Warm-start references for cold-start SKUs |
| `model_initialization_s9` | acting_handler | Per-SKU 7-decision audit log |
| `stage9_self_assessment` | reporting_handler | One row per run with health stats |
| `stage9_sku_execution_log` | acting_handler | Per-SKU diagnostics including exceptions |
| `agent_state_log_s9` | state_machine.transition | Every state transition |
| `tenant_learning_params` | seed.py + nightly batch | The 54-row tenant params catalog |
| `model_performance_s9` | model_performance_aggregator | 30-day rolling MAPE per (model, horizon) |
| `adaptive_quantile_state` | learning_params_updater | Per-`(tenant, sku)` learned quantile |
| `forecast_outcomes` | outcome_collector | Forecast vs. actual per (sku, run, horizon) |
| `cross_agent_signals` | signals.emitter | Typed messages between stages |

### R5. Locked Invariants (the 6 rules nothing crosses)

These six rails are the only hard-coded numbers and rules in Stage 9. The learner cannot cross them. Any PR that introduces a new hard-coded threshold outside these rails is a bug.

| # | Rail | Where enforced | Why |
|---|---|---|---|
| 1 | **8 horizons exactly: `[7, 14, 30, 60, 90, 150, 180, 365]`** | `infrastructure/constants.py:HORIZONS` | Stage 10's reorder math is hard-coded against these bins. Adding or removing one breaks Stage 10. |
| 2 | **Per-SKU isolation** | `acting_handler._per_sku_fallback` | One bad SKU never aborts a run. |
| 3 | **Stage 9 never writes to `stage8.*`** | code review | `stage8.*` is Stage 8's namespace. Stage 9 reads only. |
| 4 | **All env access through `infrastructure/config.py`** | code review | Single source of truth; no scattered `os.environ` calls. |
| 5 | **One Redis lock per tenant** | `infrastructure/run_lock.py` | Concurrent runs for the same tenant would corrupt the state log. |
| 6 | **Quantile clamp `[0.50, 0.99]`** | `learning_params_updater` | Below 0.50 the forecast is below the median; above 0.99 the bootstrap variance becomes unstable. |

### R6. How to Run Stage 9 Locally

#### R6.1 One-time setup

```bash
# 1) Postgres + Redis via Docker
docker run -d --name postgres-test -p 5432:5432 \
    -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test postgres
docker run -d --name redis-test -p 6379:6379 redis:latest

# 2) Apply schema
psql postgresql://test:test@localhost:5432/test -f code/infrastructure/db.py.sql

# 3) Seed tenant params for a tenant
python -c "
from infrastructure.config import DB_DSN
from infrastructure.seed import seed_tenant_params
import psycopg2
conn = psycopg2.connect(DB_DSN)
seed_tenant_params('11111111-1111-1111-1111-111111111111', 'established', conn=conn)
conn.commit()
"

# 4) Install Python deps
pip install -r 'project docs/requirements.txt'
```

#### R6.2 Trigger one Stage 9 run

```python
# orchestrator_demo.py
import psycopg2, redis, uuid
from infrastructure.config import DB_DSN, REDIS_URL
from pipeline.orchestrator import run

tenant_id = '11111111-1111-1111-1111-111111111111'
run_id    = str(uuid.uuid4())

conn   = psycopg2.connect(DB_DSN)
client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

run(tenant_id, run_id, conn, client)
```

#### R6.3 Inspect outputs

```sql
-- The state log
SELECT from_state, to_state, transitioned_at
FROM stage9.agent_state_log_s9 WHERE run_id = '...'
ORDER BY transitioned_at;

-- The forecasts
SELECT sku_id, processing_tier, assigned_model, confidence_final, backtest_mape
FROM stage9.forecasts WHERE run_id = '...'
ORDER BY confidence_final DESC LIMIT 20;

-- The self-assessment
SELECT * FROM stage9.stage9_self_assessment WHERE run_id = '...';
```

### R7. Glossary

| Term | Definition |
|---|---|
| `assigned_model` | The model class chosen by Sub-Stage 9.1 (one of Naive/SES/Holt/Croston/Prophet). |
| `BatchWriter` | Buffered upserter — handlers `queue()` rows; LEARNING flushes. |
| `cache` (tier) | Processing tier where the prior forecast is reused with no refit. |
| `confidence_final` | Output of the 5-step multiplicative confidence formula. ∈ `[confidence_floor, confidence_ceiling]`. |
| `cross_agent_signals` | Typed messages on `stage9.cross_agent_signals` — emit/peek/consume semantics. |
| `dual_pool` | ProcessPool (4) for Prophet + ThreadPool (16) for everything else. |
| `effective_max_horizon` | The longest horizon worth computing for an SKU — capped by `planned_end_date`. |
| `fingerprint` | SHA-256 over the last 7 days of demand + pattern label. Drives tier classification. |
| `forecast_outcomes` | Nightly-collected ground truth that drives the learning loop. |
| `FULL` (mode) | Full Stage 9 run: every sub-stage executes. |
| `LearningContext` | Dataclass produced by Sub-Stage 9.1, consumed by 9.2–9.5. |
| `learning_mode` | `explore` (collect signal) vs `exploit` (use best HP). |
| `MICRO_UPDATE` (mode) | SES level correction only; no model retrain. |
| `model_hint` | Stage 8's recommended model. Sub-Stage 9.1 may override. |
| `OOS adjustment factor` | Multiplier applied to a forecast when training data was depressed by stockouts. |
| `pattern_feedback` | Stage 9 → Stage 8 feedback row in `stage8.pattern_feedback`. |
| `pattern_label` | Stage 8's pattern classification (one of cold_start, stable, trending, seasonal, intermittent). |
| `Preloader` | Single-instance bulk loader run in PRELOADING. |
| `processing_tier` | One of `cache`, `partial`, `full`. |
| `RedisRunLock` | Per-tenant `SET NX EX 14400` mutex with Lua-verified release. |
| `RunContext` | Per-run dependency container in the in-memory `_RUN_CONTEXTS` registry. |
| `SelfAssessmentEngine` | Runs in REPORTING; detects degrading models; writes `stage9_self_assessment`. |
| `SignalConsumer` | PEEK (read-only) or atomic FOR UPDATE SKIP LOCKED consumer. |
| `SignalEmitter` | Direct-write half — per-thread, not shared. |
| `Sub-Stage 9.0` | Fingerprinting + tier classification. |
| `Sub-Stage 9.1` | Model initialization (7 decisions). |
| `Sub-Stage 9.2` | Feature engineering. |
| `Sub-Stage 9.3` | Hyperparameter tuning via Thompson Sampling. |
| `Sub-Stage 9.4` | Backtesting + pattern_feedback. |
| `Sub-Stage 9.5` | Forecast generation, bootstrap, confidence. |
| `TenantParams` | Per-tenant in-memory snapshot of `tenant_learning_params`. |
| `Thompson Sampling` | Bayesian bandit on Beta(α, β) per HP config. |
| `tier_router` | Per-SKU dispatcher choosing the sub-stage chain. |

---

*End of document.*
