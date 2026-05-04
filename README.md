# Stage 9 — Forecasting Agent

Per-SKU demand forecasting with Bayesian hyperparameter tuning, adaptive
backtesting, and a nightly learning loop. Driven by an agent state machine
(`IDLE → PRELOADING → PERCEIVING → PLANNING → ACTING → LEARNING → REPORTING
→ COMPLETE`) and triggered by LangGraph when a Stage 8 run reaches
`patterns_discovered`.

## Repository layout

```
stage_9/
├── README.md                  ← (this file)
├── pytest.ini                 ← testpaths = code/tests, pythonpath = code
├── project docs/
│   ├── requirements.txt
│   ├── stage9_data_flow_and_learning.md
│   └── Stage9_EdgeCases.pdf
└── code/
    ├── .env                   ← DB / Redis / app config (see config.py)
    ├── pytest.ini
    ├── backtesting/           ← Sub-Stage 9.4
    ├── forecasting/           ← Sub-Stages 9.0, 9.2, 9.5 + tier router
    ├── handlers/              ← one module per agent state
    ├── infrastructure/        ← config, DB, locks, state machine, params
    ├── learning/              ← nightly batch jobs + self-assessment
    ├── models/                ← forecasting models + Thompson sampler
    ├── pipeline/              ← orchestrator, dual-pool executor, preloader
    ├── signals/               ← cross-agent signal bus (emit/consume)
    ├── results/               ← run output artefacts
    └── tests/                 ← pytest suite
```

## Module guide

### `pipeline/` — execution skeleton

| File | Role |
|---|---|
| `orchestrator.py` | `run()` entry point. Acquires Redis run lock, walks the state machine, delegates each phase to its handler, releases the lock in `finally`. Re-exports `run_one_sku`, `build_sku_pipeline_input`, `REQUIRED_PRELOAD_KEYS`. |
| `pipeline_graph.py` | Outer LangGraph that watches `runs.status` across tenants and dispatches the appropriate stage agent. |
| `preloader.py` | Single-instance bulk loader: 7 SELECTs + TenantParams + signal_context, packaged into a typed `PreloadedData` container handed to every sub-stage. |
| `dual_pool.py` | Concurrent SKU executor. ProcessPool (4 workers, 120s) for Prophet; ThreadPool (16 slots) for everything else. |
| `model_initialization.py` | Sub-Stage 9.1 — 7 ordered decisions per FULL-tier SKU; emits the `LearningContext` consumed by 9.2–9.6. |

### `handlers/` — one per agent state

| State | Handler | Responsibility |
|---|---|---|
| PRELOADING | `preloading.py` | Run Preloader, stash a `RunContext`, resolve FULL vs MICRO_UPDATE based on `micro_update_threshold_hours`. |
| PERCEIVING | `perceiving.py` | Snapshot tenant params; PEEK Stage 8 `pattern_confidence` signals. |
| PLANNING | `planning.py` | Pre-fetch the two tenant-wide caches (calibrated backtest windows, Thompson HP cache) so 9.4/9.5 don't make N round-trips. |
| ACTING | `acting.py` | Run Sub-Stages 9.1 → 9.5 per SKU across three tracks (MICRO_UPDATE / CACHE / FULL). One bad SKU never stops the run (`_per_sku_fallback`). |
| LEARNING | `learning.py` | Flush BatchWriter, bulk-upsert Thompson state, refresh `sku_similarity_registry`. |
| REPORTING | `reporting.py` | Run `SelfAssessmentEngine`, emit `model_health` broadcast, drop the RunContext. |
| `_context.py` | — | In-memory `RunContext` registry shared across handlers (created in PRELOADING, removed in REPORTING). |

### `forecasting/` — per-SKU sub-stages

| File | Role |
|---|---|
| `fingerprint.py` | Sub-Stage 9.0 — classifies each SKU into `cache` / `partial` / `full` tier using a content fingerprint. |
| `tier_router.py` | Dispatches each SKU to the right sub-stage chain: `full` (9.1→9.2→9.3), `partial` (9.1→9.2, skips 9.3), `cache` (warm-start). |
| `feature_engg.py` | Sub-Stage 9.2 — promo-weighted training data + feature selection. |
| `forecasting.py` | Sub-Stage 9.5 — fit, generate 8 horizon forecasts, bootstrap quantiles. |
| `confidence.py` | Multiplicative confidence formula (5 steps) extracted for unit-testability. |

### `backtesting/` — Sub-Stage 9.4

`backtesting.py` — adaptive window selection, walk-forward backtest, MAPE,
and `pattern_feedback` write-back.

### `models/` — forecasters + bandit

`base.py` defines `BaseModel`. Concrete models: `naive.py`, `ses.py`,
`croston.py`, `holt.py`, `prophet_model.py`. `bootstrap.py` converts a point
forecast into `{mean, p50, p80, p90}`. `thompson.py` is the Beta(α, β)
hyperparameter bandit. `hp_tuning.py` is Sub-Stage 9.3 — runs Standard
Thompson Sampling regardless of lifecycle stage.

### `learning/` — nightly batch + self-assessment

| File | Schedule | Role |
|---|---|---|
| `outcome_collector.py` | 3 AM UTC | Compares closed-horizon forecasts to actuals, writes `forecast_outcomes`. |
| `model_performance_aggregator.py` | 4 AM UTC | Rolls up the last 30 days of `forecast_outcomes` into `model_performance_s9` per (model, horizon). |
| `learning_params_updater.py` | 4:30 AM UTC | Nudges `tenant_learning_params` toward observed evidence. |
| `self_assessment.py` | In REPORTING | Detects degrading models, computes run stats, writes `stage9_self_assessment`. |

### `infrastructure/` — cross-cutting

| File | Role |
|---|---|
| `config.py` | Loads `code/.env` (via python-dotenv if present, else built-in parser). Exports DB/Redis/app constants and `DB_DSN`. Single source of truth for env. |
| `constants.py` | Enums + string constants (`Param`, `Table`, `LOCK_KEY_TEMPLATE`, `LOCK_TTL_SECONDS`, etc.). |
| `errors.py` | Domain exceptions (`Stage9Error`, `RunAlreadyInProgressError`, `TenantParamNotFoundError`, …). |
| `state_machine.py` | `AgentState` enum + `transition()` writer to `agent_state_log_s9`. |
| `run_lock.py` | `RedisRunLock` — per-tenant SET NX EX dead-man's switch (default TTL 14400 s). Prevents concurrent runs for the same tenant. |
| `batch_writer.py` | Buffered upserter — handlers `queue()` rows; `LEARNING` flushes. |
| `seed.py` | `seed_tenant_params()` — idempotent insert of 54 default rows into `tenant_learning_params`. |
| `tenant_params.py` | Per-tenant in-memory snapshot read by every sub-stage. Fail-fast `UnknownParamError` on missing keys. |
| `tenant_params_defaults.py` | The 54-row defaults used by `seed.py`. |
| `db.py` | DDL strings — owns the `stage9` schema migration. |
| `db_utils.py` | Shared psycopg2 helpers (defensive checks, search-path setup). |

### `signals/` — cross-agent message bus

`_base.py` (shared SQL + dataclasses), `emitter.py` (per-thread direct
write — **not** thread-safe across the dual-pool worker slots) and
`consumer.py` (`peek_signals` / atomic `consume_signals` via `FOR UPDATE
SKIP LOCKED`).

## State machine

```
IDLE → PRELOADING → PERCEIVING → PLANNING → ACTING → LEARNING → REPORTING → COMPLETE
                                                  │
                                                  └── any unrecoverable error → FAILED
```

`LEARNING` and `REPORTING` have **no** edge to `FAILED` — if they raise,
the lock is still released and the exception propagates to LangGraph for
retry decisions.

## Configuration

All env reads go through `infrastructure/config.py`. Drop a `.env` file at
`code/.env`:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=test
DB_USER=test
DB_PASSWORD=test
DB_SSLMODE=disable
DB_CONNECT_TIMEOUT=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=20

# Application
STAGE9_PLANNING_THREADS=16
STAGE9_ALLOW_FORCE_RELEASE=false
STAGE9_PROJECT_ROOT=M:/stage_9/code
RUN_INTEGRATION_TESTS=true
```

## Dependencies

From `project docs/requirements.txt`:

```
numpy pandas scipy statsmodels prophet neuralprophet ruptures
psycopg2-binary redis langgraph
```

Install with:

```bash
pip install numpy pandas scipy statsmodels prophet neuralprophet ruptures \
            psycopg2-binary redis langgraph
```

## External services

| Service | Purpose |
|---|---|
| PostgreSQL | All Stage 8 reads + Stage 9 writes. Two schemas: `stage8` (read-only inputs), `stage9` (everything this agent owns). |
| Redis | Per-tenant run lock (`stage9_lock_{tenant_id}`, TTL 14400 s). |

## Three execution tracks (per SKU)

| Track | Trigger | Path |
|---|---|---|
| **MICRO_UPDATE** | Last `COMPLETE` run within `micro_update_threshold_hours` | SES level correction only — no model retrain (target < 15 s). |
| **CACHE** | Fingerprint match + `pattern_label` + good prior MAPE | Warm-start from cached prior forecast. |
| **FULL** | Default | Sub-Stages 9.1 → 9.2 → 9.3 → 9.4 → 9.5 → write. |

## Locked invariants

- **8 horizons** (Principle 5): `[7, 14, 30, 60, 90, 150, 180, 365]` — never modified.
- **Per-SKU isolation** (Principle 3): one bad SKU never aborts a run.
- **Stage 9 never writes to Stage 8** — only reads `stage8.*`.
- **All env access through `infrastructure/config.py`** — no `os.environ` calls scattered through sub-stages.
