"""
db.py — Atheera Stage 9: Database Migration
============================================
Version 3.0  |  April 2026  |  Confidential

SCHEMA LAYOUT
-------------
Both Stage 8 and Stage 9 share the same PostgreSQL server but live in
separate schemas. This file owns everything under the stage9 schema.
canonical_sku is created in the stage8 schema — it is the shared product
reference table read by all pipeline stages.

    PostgreSQL server (shared)
    ├── stage8.*   — owned by Stage 8; includes canonical_sku and all input tables
    └── stage9.*   — created and managed by this file

Stage 9 reads Stage 8 tables using fully-qualified stage8.* names
(pattern_history, signal_context, canonical_sku, etc.) through a read-only
connection. The one write Stage 9 performs into stage8 is inserting rows into
stage8.pattern_feedback — that table is owned by Stage 8 but Stage 9 is its
designated writer per the pipeline data contract.

TABLE OWNERSHIP RULE
-------------------------------------
Every table has exactly one owner — the stage whose migration created it.
No stage modifies a table it does not own. stage8.canonical_sku is created
here with CREATE TABLE IF NOT EXISTS — safe to re-run, and never drops
pre-existing rows.

WHAT THIS FILE MANAGES
-----------------------
  Schema     : stage9  (created on first run_up call)
  Creates    : stage9 schema, all 17 stage9.* tables, stage8.canonical_sku
               with all pre-existing + 7 Stage 9 columns

TABLE CREATION ORDER
--------------------
Tables are created in dependency order. No table references another that
has not already been created.

  1.  tenant_learning_params        — replaces all hardcoded thresholds; no FK deps
  2.  model_initialization_s9       — Sub-Stage 9.1 decisions per SKU per run
  3.  feature_decisions_s9          — Sub-Stage 9.2 feature selection decisions
  4.  hyperparameter_decisions      — Sub-Stage 9.3 Thompson Sampling winner
  5.  backtest_decisions            — Sub-Stage 9.4 backtest results and exceptions
  6.  forecasts                     — PRIMARY output; Stage 10 reads this
  7.  forecast_outcomes             — ground-truth actuals written by OutcomeCollector
  8.  cross_agent_signals           — inter-stage signal bus shared by Stage 8/9/10
  9.  thompson_sampling_state       — Beta(alpha, beta) per SKU-model-config pair
  10. sku_similarity_registry       — converged SKU configs for new product warm-start
  11. data_fingerprint_cache        — SHA256 per SKU enabling incremental processing
  12. adaptive_quantile_state       — tracks actual vs target quantile coverage
  13. size_curve_registry           — per-style size distributions updated each season
  14. stage9_self_assessment        — post-run health report and degradation detection
  15. model_performance_s9          — rolling 30-day MAPE per model per horizon
  16. agent_state_log_s9            — every state machine transition for every run
  17. stage9_sku_execution_log      — per-SKU success / fallback / failure log

DOWN MIGRATION SAFETY
---------------------
run_down() drops only stage9.* tables in reverse creation order and removes
only the 7 stage8.canonical_sku columns this file added. It never touches
other stage8.* tables or any Stage 1-7 tables.

RUNTIME LOCK
------------
Concurrent run prevention is handled by run_lock.py (RedisRunLock).
db.py has no dependency on Redis.

DEPENDENCIES
------------
  pip install psycopg2-binary
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extensions

# ---------------------------------------------------------------------------
# Public interface — only these names are intended for import by other modules.
# ---------------------------------------------------------------------------
__all__ = [
    "pg_conn",
    "get_connection",
    "run_up",
    "run_down",
    "run_verify",
    "DSN",
]

# ---------------------------------------------------------------------------
# Logging
#
# NullHandler is the correct default for a library/module — it prevents
# "No handlers could be found for logger" warnings when the caller has not
# configured logging. The application's entry point (not this file) should
# call logging.basicConfig() or attach its own handlers.
# ---------------------------------------------------------------------------
log = logging.getLogger("stage9.db")
log.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Schema names — single source of truth.
# If schema names ever change, update only these two constants.
# All SQL strings below reference S9 and S8, never literal schema names.
# ---------------------------------------------------------------------------
S9 = "stage9"   # schema owned by Stage 9; this file creates it
S8 = "stage8"   # schema owned by Stage 8; Stage 9 reads from it; canonical_sku lives here

# ---------------------------------------------------------------------------
# Database connection — all values come from infrastructure/config.py which
# loads them from the .env file.  Edit .env to change connection settings.
# ---------------------------------------------------------------------------
from infrastructure.config import (  # noqa: E402
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    DB_SSLMODE, DB_CONNECT_TIMEOUT,
)

if not DB_PASSWORD:
    log.warning(
        "DB_PASSWORD is not set — add it to .env or set the environment "
        "variable before connecting."
    )

DSN = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode={DB_SSLMODE}&connect_timeout={DB_CONNECT_TIMEOUT}"
)

# Password-masked DSN safe for log output — never log the real DSN.
SAFE_DSN = (
    f"postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode={DB_SSLMODE}&connect_timeout={DB_CONNECT_TIMEOUT}"
)


# ===========================================================================
# SECTION 1 — PostgreSQL connection helper
# ===========================================================================

@contextmanager
def pg_conn(dsn: str = DSN) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context-manager connection factory — the standard way to get a connection
    in Stage 9 application code (sub-stages, batch jobs, cross_agent.py, etc.).

    Opens a connection, yields it, then commits on clean exit or rolls back on
    any exception. Connection is always closed in the finally-block.

    Autocommit is intentionally left OFF so every caller's work runs inside
    an explicit transaction.

    Usage:
        from db import pg_conn

        with pg_conn() as conn:
            conn.cursor().execute("SELECT ...")
    """
    # Log the safe (masked) DSN — never log the real DSN which contains the password.
    log.debug("Opening connection: %s", SAFE_DSN)
    conn = psycopg2.connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        log.debug("Connection closed")


def get_connection(dsn: str = DSN) -> psycopg2.extensions.connection:
    """
    Raw connection factory — for code that cannot use a context manager.

    Returns an open psycopg2 connection with autocommit OFF.
    THE CALLER is responsible for:
      - conn.commit() or conn.rollback()
      - conn.close() — ALWAYS in a finally-block, never in a bare except

    When to use this instead of pg_conn():
      - ProcessPool subprocess workers (E003 edge case) — the worker function
        must be a plain function, not a generator, so 'with pg_conn()' cannot
        be used across the subprocess boundary.
      - Any finally block that needs to close a connection it opened earlier.

    SCALABILITY WARNING:
      Every call to get_connection() opens a brand-new database connection.
      In application code that runs per-SKU or in tight loops, this will
      exhaust the database's max_connections limit quickly. For high-frequency
      application code, use a connection pool (e.g. psycopg2.pool.ThreadedConnectionPool
      or SQLAlchemy's pool) rather than calling get_connection() per operation.
      This function is safe for use in:
        - ProcessPool subprocess workers (one connection per worker process)
        - pattern_feedback direct writes (one connection per SKU, immediately closed)
        - Migration functions (one connection for the entire migration)

    Usage (subprocess worker — E003 pattern):
        from db import get_connection

        def _subprocess_worker(args):
            conn = None
            try:
                conn = get_connection()
                # ... do work ...
                conn.commit()
                return result
            except Exception:
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()   # always executes — even if worker is killed
    """
    log.debug("Opening raw connection: %s", SAFE_DSN)
    return psycopg2.connect(dsn)


def execute_sql(conn: psycopg2.extensions.connection, sql: str, label: str = "") -> None:
    """
    Execute one DDL statement and log the result.

    label is a short human-readable description used in log output.
    If omitted, the first 80 characters of sql are logged instead.
    Raises on any psycopg2 error so the caller's transaction rolls back.
    """
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            log.info("OK    %s", label or sql[:80].strip())
        except psycopg2.Error as exc:
            log.error("FAIL  %s — %s", label, exc.pgerror or str(exc))
            raise


# ===========================================================================
# SECTION 2 — DDL: schema
# ===========================================================================

SQL_CREATE_SCHEMA = f"CREATE SCHEMA IF NOT EXISTS {S9};"

# ===========================================================================
# SECTION 3 — DDL: stage9.* tables (17 tables, creation order)
# ===========================================================================

# 1. tenant_learning_params — no FK dependencies
SQL_CREATE_TENANT_LEARNING_PARAMS = f"""
CREATE TABLE IF NOT EXISTS {S9}.tenant_learning_params (
    tenant_id               UUID            NOT NULL,
    param_name              VARCHAR(100)    NOT NULL,
    starting_value          DECIMAL(12,6)   NOT NULL,
    current_value           DECIMAL(12,6)   NOT NULL,
    confidence_in_value     DECIMAL(3,2)    DEFAULT 0.10,
    total_evidence_runs     INTEGER         DEFAULT 0,
    last_updated_run_id     UUID,
    last_updated_at         TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (tenant_id, param_name)
);
"""

# 2. model_initialization_s9 — Sub-Stage 9.1 decisions per SKU per run
SQL_CREATE_MODEL_INITIALIZATION_S9 = f"""
CREATE TABLE IF NOT EXISTS {S9}.model_initialization_s9 (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    run_id                  UUID            NOT NULL,
    assigned_model                  VARCHAR(60)     NOT NULL,
    insufficient_seasonal_history   BOOLEAN         NOT NULL DEFAULT FALSE,
    pattern_label                   VARCHAR(30)     NOT NULL,
    lifecycle_stage                 VARCHAR(30),
    selected_quantile               DECIMAL(4,3)    NOT NULL,
    quantile_source         VARCHAR(30)     NOT NULL,
    effective_max_horizon   INTEGER         NOT NULL,
    learning_mode           VARCHAR(20)     NOT NULL,
    oos_adjustment_factor   DECIMAL(6,4)    NOT NULL,
    is_b2b                  BOOLEAN         NOT NULL,
    reorder_bias_factor     DECIMAL(6,4)    NOT NULL,
    created_at              TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, sku_id, run_id)
);
"""

# 3. feature_decisions_s9 — Sub-Stage 9.2 feature selection decisions
SQL_CREATE_FEATURE_DECISIONS_S9 = f"""
CREATE TABLE IF NOT EXISTS {S9}.feature_decisions_s9 (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    run_id                  UUID            NOT NULL,
    features_used           JSONB           NOT NULL,
    reliability_map_applied BOOLEAN         NOT NULL DEFAULT FALSE,
    b2b_mode_applied        BOOLEAN         NOT NULL DEFAULT FALSE,
    promo_weighting_applied BOOLEAN         NOT NULL DEFAULT FALSE,
    baseline_mape           DECIMAL(8,6),
    improved_mape           DECIMAL(8,6),
    created_at              TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, sku_id, run_id)
);
"""

# 4. hyperparameter_decisions — Sub-Stage 9.3 Thompson Sampling winner
SQL_CREATE_HYPERPARAMETER_DECISIONS = f"""
CREATE TABLE IF NOT EXISTS {S9}.hyperparameter_decisions (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    run_id                  UUID            NOT NULL,
    hyperparameters         JSONB           NOT NULL,
    validation_mape         DECIMAL(8,6),
    config_hash             VARCHAR(64)     NOT NULL,
    thompson_score          DECIMAL(6,4),
    early_stopped           BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, sku_id, run_id)
);
"""

# 5. backtest_decisions — Sub-Stage 9.4 backtest results and exception flags
SQL_CREATE_BACKTEST_DECISIONS = f"""
CREATE TABLE IF NOT EXISTS {S9}.backtest_decisions (
    id                          BIGSERIAL       PRIMARY KEY,
    tenant_id                   UUID            NOT NULL,
    sku_id                      UUID            NOT NULL,
    run_id                      UUID            NOT NULL,
    backtest_mape               DECIMAL(8,6),
    backtest_wape               DECIMAL(8,6),
    backtest_bias               DECIMAL(8,6),
    exception_flags             JSONB,
    backtest_window_days        INTEGER,
    structural_break_detected   BOOLEAN         DEFAULT FALSE,
    break_index                 INTEGER         NULL,
    training_data_truncated     BOOLEAN         DEFAULT FALSE,
    created_at                  TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, sku_id, run_id)
);
"""

# 6. forecasts — PRIMARY output; Stage 10 reads this
# Column-per-horizon design: each of the 8 horizons gets its own JSONB column
# containing {{mean, p50, p80, p90}} quantile values (see FORECAST_COLUMN_MAP).
SQL_CREATE_FORECASTS = f"""
CREATE TABLE IF NOT EXISTS {S9}.forecasts (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    run_id                  UUID            NOT NULL,
    forecast_date           DATE,
    assigned_model          VARCHAR(60)     NOT NULL,
    pattern_label           VARCHAR(30),
    selected_quantile       DECIMAL(4,3)    NOT NULL,
    confidence_final        DECIMAL(4,3),
    confidence_base         DECIMAL(4,3)    NULL,
    confidence_tier         VARCHAR(30),
    status                  VARCHAR(30)     NOT NULL,
    exception_flags         JSONB,
    backtest_mape           DECIMAL(8,6)    NULL,
    lifecycle_stage         VARCHAR(30)     NULL,
    effective_max_horizon   INTEGER,
    oos_adjustment_factor   DECIMAL(6,4),
    reorder_bias_factor     DECIMAL(6,4),
    is_b2b                  BOOLEAN,
    execution_mode          VARCHAR(20),
    processing_tier         VARCHAR(20),
    forecast_7d             JSONB,
    forecast_14d            JSONB,
    forecast_30d            JSONB,
    forecast_60d            JSONB,
    forecast_90d            JSONB,
    forecast_150d           JSONB,
    forecast_180d           JSONB,
    forecast_365d           JSONB,
    created_at              TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, sku_id, run_id)
);
"""

SQL_IDX_FORECASTS_TENANT_SKU = f"""
CREATE INDEX IF NOT EXISTS idx_forecasts_tenant_sku
    ON {S9}.forecasts (tenant_id, sku_id);
"""

# 7. forecast_outcomes — ground-truth actuals written by OutcomeCollector
# UNIQUE on (tenant_id, sku_id, run_id, horizon_days) makes ON CONFLICT DO NOTHING
# re-runs idempotent (spec §4 Step 5, §8 Done-When #2).
SQL_CREATE_FORECAST_OUTCOMES = f"""
CREATE TABLE IF NOT EXISTS {S9}.forecast_outcomes (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    run_id                  UUID            NOT NULL,
    horizon_days            INTEGER         NOT NULL,
    assigned_model          VARCHAR(60),
    forecast_value          DECIMAL(12,4),
    actual_value            DECIMAL(12,4),
    error_mape              DECIMAL(5,3),
    error_wape              DECIMAL(5,3),
    bias                    DECIMAL(5,3),
    outcome_date            DATE            NOT NULL DEFAULT CURRENT_DATE,
    UNIQUE (tenant_id, sku_id, run_id, horizon_days)
);
"""

SQL_IDX_FORECAST_OUTCOMES_TENANT_SKU = f"""
CREATE INDEX IF NOT EXISTS idx_forecast_outcomes_tenant_sku
    ON {S9}.forecast_outcomes (tenant_id, sku_id, horizon_days);
"""

# 9. cross_agent_signals — inter-stage signal bus shared by Stage 8/9/10
# sku_id is nullable: tenant-level signals (e.g. model_health) have no SKU scope.
SQL_CREATE_CROSS_AGENT_SIGNALS = f"""
CREATE TABLE IF NOT EXISTS {S9}.cross_agent_signals (
    id                      BIGSERIAL       PRIMARY KEY,
    signal_id               UUID            NULL,
    tenant_id               UUID            NOT NULL,
    from_agent              VARCHAR(30)     NOT NULL,
    to_agent                VARCHAR(30)     NOT NULL,
    signal_type             VARCHAR(50)     NOT NULL,
    sku_id                  UUID,
    run_id                  UUID,
    payload                 JSONB,
    confidence              DECIMAL(4,3),
    expires_at              TIMESTAMP,
    processed               BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMP       DEFAULT NOW()
);
"""

# Covers the PEEK query in cross_agent.py: filters on tenant_id, signal_type,
# sku_id, processed, and expires_at in that order.
SQL_IDX_CROSS_AGENT_SIGNALS_PEEK = f"""
CREATE INDEX IF NOT EXISTS idx_cross_agent_signals_peek
    ON {S9}.cross_agent_signals (tenant_id, signal_type, sku_id, processed, expires_at);
"""

# 10. thompson_sampling_state — Beta(alpha, beta) per SKU-model-config pair
SQL_CREATE_THOMPSON_SAMPLING_STATE = f"""
CREATE TABLE IF NOT EXISTS {S9}.thompson_sampling_state (
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    assigned_model          VARCHAR(60)     NOT NULL,
    config_hash             VARCHAR(64)     NOT NULL,
    config_json             JSONB           NOT NULL,
    alpha_param             DECIMAL(10,4)   NOT NULL DEFAULT 1.0,
    beta_param              DECIMAL(10,4)   NOT NULL DEFAULT 1.0,
    total_trials            INTEGER         NOT NULL DEFAULT 0,
    last_mape               DECIMAL(8,6),
    last_updated_at         TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (tenant_id, sku_id, assigned_model, config_hash)
);
"""

# 11. sku_similarity_registry — converged SKU configs for new product warm-start
SQL_CREATE_SKU_SIMILARITY_REGISTRY = f"""
CREATE TABLE IF NOT EXISTS {S9}.sku_similarity_registry (
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    pattern_label           VARCHAR(30),
    vendor                  VARCHAR(255),
    product_type            VARCHAR(255),
    parent_style_id         UUID,
    cv                      DECIMAL(10,4),
    observation_days        INTEGER,
    avg_daily_qty           DECIMAL(12,4),
    weekend_zero_ratio      DECIMAL(5,4),
    best_model_config       JSONB,
    best_features           JSONB,
    avg_mape                DECIMAL(8,6),
    convergence_run         INTEGER,
    last_updated            TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (tenant_id, sku_id)
);
"""

# 12. data_fingerprint_cache — SHA256 per SKU enabling incremental processing
SQL_CREATE_DATA_FINGERPRINT_CACHE = f"""
CREATE TABLE IF NOT EXISTS {S9}.data_fingerprint_cache (
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    fingerprint             VARCHAR(64)     NOT NULL,
    tier                    VARCHAR(20)     NOT NULL
        CHECK (tier IN ('cache', 'partial', 'full')),
    pattern_label           VARCHAR(30)     NULL,
    demand_total            DECIMAL(12,4)   NULL,
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (tenant_id, sku_id)
);
"""

# 13. adaptive_quantile_state — tracks actual vs target quantile coverage
SQL_CREATE_ADAPTIVE_QUANTILE_STATE = f"""
CREATE TABLE IF NOT EXISTS {S9}.adaptive_quantile_state (
    tenant_id               UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    assigned_model          VARCHAR(60)     NOT NULL,
    pattern_label           VARCHAR(30)     NOT NULL,
    horizon_days            INTEGER         NOT NULL,
    backtest_window_days    INTEGER,
    sample_size             INTEGER,
    calibration_gap         DECIMAL(6,4),
    target_quantile         DECIMAL(4,3),
    actual_coverage         DECIMAL(4,3),
    last_updated            TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (tenant_id, sku_id, assigned_model, pattern_label, horizon_days)
);
"""

# 14. size_curve_registry — per-style size distributions updated each season
SQL_CREATE_SIZE_CURVE_REGISTRY = f"""
CREATE TABLE IF NOT EXISTS {S9}.size_curve_registry (
    tenant_id               UUID            NOT NULL,
    style_id                UUID            NOT NULL,
    season                  VARCHAR(20)     NOT NULL,
    curve                   JSONB           NOT NULL,
    confidence              DECIMAL(4,3),
    observation_days        INTEGER,
    last_updated            TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (tenant_id, style_id, season)
);
"""

# 15. stage9_self_assessment — post-run health report and degradation detection
SQL_CREATE_STAGE9_SELF_ASSESSMENT = f"""
CREATE TABLE IF NOT EXISTS {S9}.stage9_self_assessment (
    id                              BIGSERIAL       PRIMARY KEY,
    tenant_id                       UUID            NOT NULL,
    run_id                          UUID            NOT NULL,
    avg_mape_this_run               DECIMAL(8,6),
    avg_mape_prev_run               DECIMAL(8,6),
    mape_delta_pct                  DECIMAL(8,4),
    degradation_detected            BOOLEAN         NOT NULL DEFAULT FALSE,
    recommendations                 JSONB,
    model_health_summary            JSONB,
    total_skus_processed            INTEGER,
    cache_tier_count                INTEGER,
    partial_tier_count              INTEGER,
    full_tier_count                 INTEGER,
    fallback_count                  INTEGER,
    pattern_feedback_retry_count    INTEGER,
    execution_mode                  VARCHAR(20),
    run_duration_seconds            DECIMAL(10,3),
    created_at                      TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, run_id)
);
"""

# 16. model_performance_s9 — rolling 30-day MAPE per model per horizon
# One row per (tenant, assigned_model, horizon_days) — updated each run.
SQL_CREATE_MODEL_PERFORMANCE_S9 = f"""
CREATE TABLE IF NOT EXISTS {S9}.model_performance_s9 (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    assigned_model          VARCHAR(60)     NOT NULL,
    horizon_days            INTEGER         NOT NULL,
    avg_mape_30d            DECIMAL(8,6),
    trend                   VARCHAR(20),
    mape_delta              DECIMAL(8,6),
    sample_count            INTEGER,
    created_at              TIMESTAMP       DEFAULT NOW(),
    UNIQUE (tenant_id, assigned_model, horizon_days)
);
"""

SQL_IDX_MODEL_PERF_TENANT_MODEL = f"""
CREATE INDEX IF NOT EXISTS idx_model_performance_tenant_model
    ON {S9}.model_performance_s9 (tenant_id, assigned_model, horizon_days);
"""

# 17. agent_state_log_s9 — every state machine transition for every run
# tenant_id and run_id are VARCHAR (not UUID): the state machine validates these
# as alphanumeric strings; run_id may include timestamp suffixes (not pure UUID).
SQL_CREATE_AGENT_STATE_LOG_S9 = f"""
CREATE TABLE IF NOT EXISTS {S9}.agent_state_log_s9 (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               VARCHAR(64)     NOT NULL,
    run_id                  VARCHAR(128)    NOT NULL,
    from_state              VARCHAR(20)     NOT NULL
        CHECK (from_state IN ('IDLE','PRELOADING','PERCEIVING','PLANNING',
                               'ACTING','LEARNING','REPORTING','COMPLETE','FAILED')),
    to_state                VARCHAR(20)     NOT NULL
        CHECK (to_state IN ('IDLE','PRELOADING','PERCEIVING','PLANNING',
                             'ACTING','LEARNING','REPORTING','COMPLETE','FAILED')),
    transitioned_at         TIMESTAMP       NOT NULL,
    reason                  TEXT
);
"""

SQL_IDX_AGENT_STATE_LOG_TENANT_RUN = f"""
CREATE INDEX IF NOT EXISTS idx_agent_state_log_tenant_run
    ON {S9}.agent_state_log_s9 (tenant_id, run_id);
"""

# 18. stage9_sku_execution_log — per-SKU success / fallback / failure log
SQL_CREATE_STAGE9_SKU_EXECUTION_LOG = f"""
CREATE TABLE IF NOT EXISTS {S9}.stage9_sku_execution_log (
    id                      BIGSERIAL       PRIMARY KEY,
    tenant_id               UUID            NOT NULL,
    run_id                  UUID            NOT NULL,
    sku_id                  UUID            NOT NULL,
    status                  VARCHAR(20)     NOT NULL
        CHECK (status IN ('success', 'fallback', 'failed')),
    fallback_model          VARCHAR(60),
    error_code              VARCHAR(10),
    error_message           TEXT,
    sub_stage               VARCHAR(10),
    execution_ms            INTEGER,
    created_at              TIMESTAMP       DEFAULT NOW()
);
"""

# ===========================================================================
# SECTION 4 — stage8.canonical_sku
#
# The single source of truth for product metadata. Every stage reads from it.
# Lives in the stage8 schema alongside all other Stage 8 input tables.
#
# Pre-existing columns: the standard product reference columns every stage
# needs (sku_id, tenant_id, vendor, product_type, vendor_id, lead_time,
# moq, pack_size, created_at, updated_at).
#
# Stage 9 additions (7 nullable columns): all columns Stage 9 needs to
# drive forecasting behaviour. All nullable — no existing rows are affected
# if this table already exists with the pre-existing columns only.
#
# CREATE TABLE IF NOT EXISTS — safe to re-run on a database that already
# has the table. Columns are added with IF NOT EXISTS for the same reason.
#
# Stage 9 reads this in PRELOADING as part of the bulk
# pattern_history + canonical_sku join (Technical Context Part 4, Read #1).
# ===========================================================================

SQL_CREATE_CANONICAL_SKU = f"""
CREATE TABLE IF NOT EXISTS {S8}.canonical_sku (

    -- ── Pre-existing columns ──────────────────────────────────────────────
    -- Standard product reference columns used by all stages.

    sku_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id              UUID            NOT NULL,

    -- Source platform identifier (Shopify product_id, Amazon ASIN, etc.)
    external_id            VARCHAR(255),

    -- Human-readable SKU code (e.g. 'SKU-ABC-123').
    sku_code               VARCHAR(100),

    -- Product display name.
    product_name           VARCHAR(500),

    -- Vendor / supplier name. Used in similarity registry matching (warm-start).
    vendor                 VARCHAR(255),

    -- Vendor UUID for Stage 10 reorder formula (lead_time, MOQ, pack_size joins).
    vendor_id              UUID,

    -- Product category. Used in similarity registry matching (warm-start).
    product_type           VARCHAR(255),

    -- Supplier lead time in days. Stage 10 uses this in the reorder formula.
    lead_time_days         INTEGER,

    -- Minimum order quantity. Stage 10 rounds up to nearest MOQ.
    moq                    INTEGER,

    -- Units per pack. Stage 10 rounds to the nearest pack boundary.
    pack_size              INTEGER         DEFAULT 1,

    -- Standard audit timestamps.
    created_at             TIMESTAMP       DEFAULT NOW(),
    updated_at             TIMESTAMP       DEFAULT NOW(),

    -- ── Stage 9 additions (7 nullable columns) ───────────────────────────
    -- All nullable — pre-existing rows are unaffected.

    -- Stage 9 caps all forecast horizons at min(365, shelf_life_days).
    -- Prevents generating purchase orders for stock that will expire before sale.
    shelf_life_days        INTEGER         NULL,

    -- Stage 9 caps horizons at the number of days remaining until this date.
    -- Sub-Stage 9.6 uses it to compute projected_sell_through for fashion SKUs.
    planned_end_date       DATE            NULL,

    -- Drives quantile override in Sub-Stage 9.1 (first match wins):
    --   A = critical — override quantile to 0.99 (MRO / spare parts; cannot stock out)
    --   B = standard — no override; use pattern default from tenant_learning_params
    --   C = routine  — lower buffer acceptable (slow movers, cheap commodities)
    criticality_tier       CHAR(1)         NULL
        CHECK (criticality_tier IN ('A', 'B', 'C')),

    -- Links a size variant SKU to its parent style SKU (self-referential).
    -- Sub-Stage 9.6 activates for every SKU where this column IS NOT NULL.
    parent_style_id        UUID            NULL
        REFERENCES {S8}.canonical_sku (sku_id) ON DELETE SET NULL,

    -- SKU-level ordering buffer override. Supersedes the tenant-default
    -- quantile_{{pattern}} param in tenant_learning_params for this specific SKU.
    service_level_target   DECIMAL(3,2)    NULL
        CHECK (service_level_target BETWEEN 0.50 AND 0.99),

    -- Affects lifecycle routing and size curve behaviour in Sub-Stage 9.6.
    -- 'NOS'              = never-out-of-season basics; no sell-through projection
    -- 'seasonal_fashion' = lifecycle-aware forecasting; sell-through projection active
    -- NULL               = system infers from pattern_label and lifecycle_stage signals
    product_lifecycle_type VARCHAR(30)     NULL,

    -- Manual demand estimate for brand-new products with zero sales history.
    -- Used as the baseline daily rate when cold_start pattern is assigned
    -- and CategoryComps finds fewer than 3 comparable SKUs in the registry.
    seed_daily_demand      DECIMAL(10,4)   NULL
);
"""

SQL_IDX_CANONICAL_SKU_TENANT = f"""
CREATE INDEX IF NOT EXISTS idx_canonical_sku_tenant
    ON {S8}.canonical_sku (tenant_id);
"""

SQL_IDX_CANONICAL_SKU_VENDOR = f"""
CREATE INDEX IF NOT EXISTS idx_canonical_sku_vendor
    ON {S8}.canonical_sku (tenant_id, vendor);
"""

SQL_IDX_CANONICAL_SKU_PARENT_STYLE = f"""
CREATE INDEX IF NOT EXISTS idx_canonical_sku_parent_style
    ON {S8}.canonical_sku (parent_style_id)
    WHERE parent_style_id IS NOT NULL;
"""

# ===========================================================================
# SECTION 5 — Table and column registries for down migration
# ===========================================================================

# Reverse of creation order so dependent tables are dropped before their parents.
STAGE9_TABLES = [
    f"{S9}.stage9_sku_execution_log",
    f"{S9}.agent_state_log_s9",
    f"{S9}.model_performance_s9",
    f"{S9}.stage9_self_assessment",
    f"{S9}.size_curve_registry",
    f"{S9}.adaptive_quantile_state",
    f"{S9}.data_fingerprint_cache",
    f"{S9}.sku_similarity_registry",
    f"{S9}.thompson_sampling_state",
    f"{S9}.cross_agent_signals",
    f"{S9}.forecast_outcomes",
    f"{S9}.forecasts",
    f"{S9}.backtest_decisions",
    f"{S9}.hyperparameter_decisions",
    f"{S9}.feature_decisions_s9",
    f"{S9}.model_initialization_s9",
    f"{S9}.tenant_learning_params",
]

# Only the 7 columns this file added to stage8.canonical_sku.
# Referenced in run_verify() to confirm all Stage 9 columns are present.
CANONICAL_SKU_STAGE9_COLUMNS = [
    "shelf_life_days",
    "planned_end_date",
    "criticality_tier",
    "parent_style_id",
    "service_level_target",
    "product_lifecycle_type",
    "seed_daily_demand",
]

# Full qualified table name used in run_down() DROP
CANONICAL_SKU_TABLE = f"{S8}.canonical_sku"


# ===========================================================================
# SECTION 6 — Stage 8 scaffold DDL (dev/test only)
#
# These tables are owned by Stage 8 in production.  Until Stage 9 is
# integrated with the real Stage 8 service, run_up_stage8_scaffold() creates
# minimal stand-in tables so that local runs and integration tests work
# without a live Stage 8.  Drop this section (and the function below) once
# the real Stage 8 schema is in place.
# ===========================================================================

_SQL_S8_SCHEMA = "CREATE SCHEMA IF NOT EXISTS stage8;"

_SQL_S8_DEMAND_HISTORY = """
CREATE TABLE IF NOT EXISTS stage8.demand_history (
    id          BIGSERIAL   PRIMARY KEY,
    tenant_id   UUID        NOT NULL,
    sku_id      UUID        NOT NULL,
    sale_date   DATE        NOT NULL,
    qty         NUMERIC     NOT NULL DEFAULT 0
);
"""

_SQL_S8_PATTERN_HISTORY = """
CREATE TABLE IF NOT EXISTS stage8.pattern_history (
    id                      BIGSERIAL   PRIMARY KEY,
    tenant_id               UUID        NOT NULL,
    sku_id                  UUID        NOT NULL,
    pattern_label           VARCHAR(30) NOT NULL,
    confidence_calibrated   NUMERIC(5,4),
    model_hint              VARCHAR(60),
    observation_days        INTEGER,
    lifecycle_stage         VARCHAR(30),
    composite_confidence    NUMERIC(5,4),
    drift_detected          BOOLEAN     NOT NULL DEFAULT FALSE,
    weekend_zero_ratio      NUMERIC(5,4),
    velocity_signature      JSONB
);
"""

_SQL_S8_FEATURE_DECISIONS = """
CREATE TABLE IF NOT EXISTS stage8.feature_decisions (
    id                      BIGSERIAL   PRIMARY KEY,
    tenant_id               UUID        NOT NULL,
    sku_id                  UUID        NOT NULL,
    feature_reliability_map JSONB       NOT NULL DEFAULT '{}'
);
"""

_SQL_S8_SIGNAL_CONTEXT = """
CREATE TABLE IF NOT EXISTS stage8.signal_context (
    id              BIGSERIAL   PRIMARY KEY,
    tenant_id       UUID        NOT NULL,
    sku_id          UUID,
    pipeline_mode   VARCHAR(30) NOT NULL DEFAULT 'standard',
    tenant_maturity VARCHAR(30) NOT NULL DEFAULT 'new',
    on_watchlist    BOOLEAN     NOT NULL DEFAULT FALSE
);
"""

_SQL_S8_OOS_IMPACT_ESTIMATES = """
CREATE TABLE IF NOT EXISTS stage8.oos_impact_estimates (
    id                  BIGSERIAL   PRIMARY KEY,
    tenant_id           UUID        NOT NULL,
    sku_id              UUID        NOT NULL,
    oos_pct             NUMERIC(5,4) NOT NULL DEFAULT 0,
    detection_confidence NUMERIC(5,4) NOT NULL DEFAULT 0,
    expires_at          TIMESTAMP,
    created_at          TIMESTAMP   NOT NULL DEFAULT NOW()
);
"""

_SQL_S8_CHANNEL_DEMAND_SPLITS = """
CREATE TABLE IF NOT EXISTS stage8.channel_demand_splits (
    id          BIGSERIAL   PRIMARY KEY,
    tenant_id   UUID        NOT NULL,
    sku_id      UUID        NOT NULL,
    sale_date   DATE        NOT NULL,
    qty         NUMERIC     NOT NULL DEFAULT 0,
    channel     VARCHAR(60) NOT NULL
);
"""

_SQL_S8_PROMO_DECISIONS = """
CREATE TABLE IF NOT EXISTS stage8.promo_decisions (
    id          BIGSERIAL   PRIMARY KEY,
    tenant_id   UUID        NOT NULL,
    sku_id      UUID        NOT NULL,
    promo_date  DATE        NOT NULL,
    multiplier  NUMERIC(6,4) NOT NULL DEFAULT 1.0
);
"""

_SQL_S8_PORTFOLIO_INTELLIGENCE_REPORTS = """
CREATE TABLE IF NOT EXISTS stage8.portfolio_intelligence_reports (
    id          BIGSERIAL   PRIMARY KEY,
    tenant_id   UUID        NOT NULL,
    alert_type  VARCHAR(60) NOT NULL,
    payload     JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);
"""

_SQL_S8_TENANT_THRESHOLDS = """
CREATE TABLE IF NOT EXISTS stage8.tenant_thresholds (
    id                  BIGSERIAL   PRIMARY KEY,
    tenant_id           UUID        NOT NULL UNIQUE,
    confidence_floor    NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    confidence_ceiling  NUMERIC(5,4) NOT NULL DEFAULT 1.0
);
"""

_SQL_S8_CLEAN_ORDERS = """
CREATE TABLE IF NOT EXISTS stage8.clean_orders (
    id                  BIGSERIAL   PRIMARY KEY,
    tenant_id           UUID        NOT NULL,
    canonical_sku_id    UUID        NOT NULL,
    order_date          DATE        NOT NULL,
    quantity_sold       NUMERIC     NOT NULL DEFAULT 0
);
"""

_SQL_S8_RUNS = """
CREATE TABLE IF NOT EXISTS stage8.runs (
    run_id      UUID        PRIMARY KEY,
    tenant_id   UUID        NOT NULL,
    status      VARCHAR(30) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);
"""

_SQL_S8_PATTERN_FEEDBACK = """
CREATE TABLE IF NOT EXISTS stage8.pattern_feedback (
    id                      BIGSERIAL   PRIMARY KEY,
    tenant_id               UUID        NOT NULL,
    sku_id                  UUID        NOT NULL,
    run_id                  UUID        NOT NULL,
    pattern_label           VARCHAR(30),
    stage8_confidence       NUMERIC(5,4),
    forecast_error_mape     NUMERIC(5,3),
    forecast_error_wape     NUMERIC(5,3),
    bias                    NUMERIC(5,3),
    model_used              VARCHAR(60),
    horizon_days            INTEGER,
    hint_matched            BOOLEAN,
    classification_quality  VARCHAR(30),
    fallback_used           BOOLEAN,
    UNIQUE (tenant_id, sku_id, run_id)
);
"""

_STAGE8_SCAFFOLD_STEPS = [
    (_SQL_S8_SCHEMA,                          "CREATE SCHEMA stage8"),
    (_SQL_S8_DEMAND_HISTORY,                  "CREATE stage8.demand_history"),
    (_SQL_S8_PATTERN_HISTORY,                 "CREATE stage8.pattern_history"),
    (_SQL_S8_FEATURE_DECISIONS,               "CREATE stage8.feature_decisions"),
    (_SQL_S8_SIGNAL_CONTEXT,                  "CREATE stage8.signal_context"),
    (_SQL_S8_OOS_IMPACT_ESTIMATES,            "CREATE stage8.oos_impact_estimates"),
    (_SQL_S8_CHANNEL_DEMAND_SPLITS,           "CREATE stage8.channel_demand_splits"),
    (_SQL_S8_PROMO_DECISIONS,                 "CREATE stage8.promo_decisions"),
    (_SQL_S8_PORTFOLIO_INTELLIGENCE_REPORTS,  "CREATE stage8.portfolio_intelligence_reports"),
    (_SQL_S8_TENANT_THRESHOLDS,               "CREATE stage8.tenant_thresholds"),
    (_SQL_S8_CLEAN_ORDERS,                    "CREATE stage8.clean_orders"),
    (_SQL_S8_RUNS,                            "CREATE stage8.runs"),
    (_SQL_S8_PATTERN_FEEDBACK,                "CREATE stage8.pattern_feedback"),
]


def run_up_stage8_scaffold(dsn: str = DSN) -> None:
    """
    Create minimal stage8.* stand-in tables for dev/test use.

    Safe to re-run (all statements use IF NOT EXISTS).  These tables are
    owned by Stage 8 in production — this function exists only to unblock
    local development and integration tests until the real Stage 8 schema
    is present.  Remove once Stage 8 is integrated.
    """
    log.info("Stage 8 scaffold UP — creating %d stand-in tables", len(_STAGE8_SCAFFOLD_STEPS) - 1)
    with pg_conn(dsn) as conn:
        for sql, label in _STAGE8_SCAFFOLD_STEPS:
            execute_sql(conn, sql, label)
    log.info("Stage 8 scaffold UP — complete")


# ===========================================================================
# SECTION 7 — Migration functions
# ===========================================================================

def run_up(dsn: str = DSN) -> None:
    """
    Apply the Stage 9 UP migration.

    Creates the stage9 schema, all 17 stage9.* tables with their indexes,
    and stage8.canonical_sku with the 7 Stage 9 columns and its indexes.
    All statements use IF NOT EXISTS — safe to re-run on an existing database.
    """
    ddl_steps = [
        # Schema
        (SQL_CREATE_SCHEMA,                       "CREATE SCHEMA stage9"),
        # stage9.* tables in dependency order
        (SQL_CREATE_TENANT_LEARNING_PARAMS,       "CREATE stage9.tenant_learning_params"),
        (SQL_CREATE_MODEL_INITIALIZATION_S9,      "CREATE stage9.model_initialization_s9"),
        (SQL_CREATE_FEATURE_DECISIONS_S9,         "CREATE stage9.feature_decisions_s9"),
        (SQL_CREATE_HYPERPARAMETER_DECISIONS,     "CREATE stage9.hyperparameter_decisions"),
        (SQL_CREATE_BACKTEST_DECISIONS,           "CREATE stage9.backtest_decisions"),
        (SQL_CREATE_FORECASTS,                    "CREATE stage9.forecasts"),
        (SQL_CREATE_FORECAST_OUTCOMES,            "CREATE stage9.forecast_outcomes"),
        (SQL_CREATE_CROSS_AGENT_SIGNALS,          "CREATE stage9.cross_agent_signals"),
        (SQL_CREATE_THOMPSON_SAMPLING_STATE,      "CREATE stage9.thompson_sampling_state"),
        (SQL_CREATE_SKU_SIMILARITY_REGISTRY,      "CREATE stage9.sku_similarity_registry"),
        (SQL_CREATE_DATA_FINGERPRINT_CACHE,       "CREATE stage9.data_fingerprint_cache"),
        (SQL_CREATE_ADAPTIVE_QUANTILE_STATE,      "CREATE stage9.adaptive_quantile_state"),
        (SQL_CREATE_SIZE_CURVE_REGISTRY,          "CREATE stage9.size_curve_registry"),
        (SQL_CREATE_STAGE9_SELF_ASSESSMENT,       "CREATE stage9.stage9_self_assessment"),
        (SQL_CREATE_MODEL_PERFORMANCE_S9,         "CREATE stage9.model_performance_s9"),
        (SQL_CREATE_AGENT_STATE_LOG_S9,           "CREATE stage9.agent_state_log_s9"),
        (SQL_CREATE_STAGE9_SKU_EXECUTION_LOG,     "CREATE stage9.stage9_sku_execution_log"),
        # Indexes on stage9.*
        (SQL_IDX_FORECASTS_TENANT_SKU,            "INDEX  stage9.forecasts(tenant_id, sku_id)"),
        (SQL_IDX_FORECAST_OUTCOMES_TENANT_SKU,    "INDEX  stage9.forecast_outcomes(tenant_id, sku_id, horizon_days)"),
        (SQL_IDX_CROSS_AGENT_SIGNALS_PEEK,        "INDEX  stage9.cross_agent_signals(peek)"),
        (SQL_IDX_MODEL_PERF_TENANT_MODEL,         "INDEX  stage9.model_performance_s9(tenant_id, assigned_model)"),
        (SQL_IDX_AGENT_STATE_LOG_TENANT_RUN,      "INDEX  stage9.agent_state_log_s9(tenant_id, run_id)"),
        # stage8.canonical_sku
        (SQL_CREATE_CANONICAL_SKU,                "CREATE stage8.canonical_sku"),
        (SQL_IDX_CANONICAL_SKU_TENANT,            "INDEX  stage8.canonical_sku(tenant_id)"),
        (SQL_IDX_CANONICAL_SKU_VENDOR,            "INDEX  stage8.canonical_sku(tenant_id, vendor)"),
        (SQL_IDX_CANONICAL_SKU_PARENT_STYLE,      "INDEX  stage8.canonical_sku(parent_style_id) partial"),
    ]

    log.info("Stage 9 UP migration — starting (%d DDL steps)", len(ddl_steps))

    with pg_conn(dsn) as conn:
        for step_num, (sql, label) in enumerate(ddl_steps, 1):
            log.info("[%d/%d] %s", step_num, len(ddl_steps), label)
            execute_sql(conn, sql, label)

    log.info("Stage 9 UP migration — complete")


def run_down(dsn: str = DSN) -> None:
    """
    Apply the Stage 9 DOWN migration.

    Drops all 17 stage9.* tables in reverse creation order, drops
    stage8.canonical_sku, then drops the stage9 schema.

    SAFETY: Only stage9.* objects and stage8.canonical_sku are ever dropped.
    Other stage8.* tables are never touched.

    *** WARNING — THIS IS IRREVERSIBLE ***
    All Stage 9 data (forecasts, learning state, Thompson Sampling history,
    size curves, etc.) will be permanently deleted. Run only when you intend
    a full teardown. There is no confirmation prompt — the caller is responsible
    for ensuring this is intentional.
    """
    log.warning("Stage 9 DOWN migration — starting (DESTRUCTIVE — all stage9 data will be deleted)")
    log.warning("Connection: %s", SAFE_DSN)

    with pg_conn(dsn) as conn:
        for table in STAGE9_TABLES:
            execute_sql(conn, f"DROP TABLE IF EXISTS {table} CASCADE;", f"DROP {table}")

        execute_sql(
            conn,
            f"DROP TABLE IF EXISTS {CANONICAL_SKU_TABLE} CASCADE;",
            f"DROP {CANONICAL_SKU_TABLE}",
        )

        execute_sql(conn, f"DROP SCHEMA IF EXISTS {S9} CASCADE;", "DROP SCHEMA stage9")

    log.info("Stage 9 DOWN migration — complete")


def run_verify(dsn: str = DSN) -> None:
    """
    Verify the Stage 9 migration was applied correctly.

    Queries information_schema to confirm all 17 stage9.* tables,
    stage8.canonical_sku existence, and all 7 Stage 9 columns on it.
    Logs a pass/fail line for each object.
    Raises RuntimeError if anything is missing.
    """
    log.info("Stage 9 migration verification — starting")

    missing_tables:  list[str] = []
    missing_columns: list[str] = []

    # Use a dedicated read-only connection for verification.
    # We set autocommit=True so no implicit transaction is opened —
    # information_schema queries never need a transaction and should not hold locks.
    conn = get_connection(dsn)
    try:
        conn.autocommit = True
        # Check all 17 stage9.* tables
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema || '.' || table_name
                  FROM information_schema.tables
                 WHERE table_schema = %s
                   AND table_type   = 'BASE TABLE';
                """,
                (S9,),
            )
            existing_tables = {row[0] for row in cur.fetchall()}

        for table in reversed(STAGE9_TABLES):
            if table in existing_tables:
                log.info("  ✓ %s", table)
            else:
                log.error("  ✗ MISSING %s", table)
                missing_tables.append(table)

        # Check stage8.canonical_sku exists
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                  FROM information_schema.tables
                 WHERE table_schema = %s
                   AND table_name   = 'canonical_sku'
                   AND table_type   = 'BASE TABLE';
                """,
                (S8,),
            )
            if cur.fetchone():
                log.info("  ✓ stage8.canonical_sku")
            else:
                log.error("  ✗ MISSING stage8.canonical_sku")
                missing_tables.append("stage8.canonical_sku")

        # Check all 7 Stage 9 columns exist on stage8.canonical_sku
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = %s
                   AND table_name   = 'canonical_sku';
                """,
                (S8,),
            )
            existing_cols = {row[0] for row in cur.fetchall()}

        for col in CANONICAL_SKU_STAGE9_COLUMNS:
            if col in existing_cols:
                log.info("  ✓ stage8.canonical_sku.%s", col)
            else:
                log.error("  ✗ MISSING stage8.canonical_sku.%s", col)
                missing_columns.append(col)

    finally:
        conn.close()

    if missing_tables or missing_columns:
        raise RuntimeError(
            f"Verification failed — "
            f"{len(missing_tables)} missing tables, "
            f"{len(missing_columns)} missing columns"
        )

    log.info("Stage 9 migration verification — all 17 tables, stage8.canonical_sku, and 7 columns present")


# ===========================================================================
# SECTION 7 — CLI entry point
#
# Allows running the migration directly from the command line:
#
#   python db.py up       — apply UP migration (create all tables)
#   python db.py down     — apply DOWN migration (drop all tables — DESTRUCTIVE)
#   python db.py verify   — verify all tables and columns exist
#
# All three commands use the hardcoded DSN defined at the top of this file.
# To override the connection, pass a DSN as the second argument:
#
#   python db.py up "postgresql://user:pass@host/db"
# ===========================================================================

if __name__ == "__main__":
    import sys

    # Configure a basic formatter for CLI use only.
    # When imported as a module, the caller configures logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cmd  = sys.argv[1] if len(sys.argv) > 1 else "verify"
    _dsn = sys.argv[2] if len(sys.argv) > 2 else DSN

    if cmd == "up":
        run_up(_dsn)
    elif cmd == "down":
        # Extra confirmation guard for destructive operation
        confirm = input(
            "\n⚠  This will permanently delete ALL Stage 9 data.\n"
            "   Type 'yes' to confirm: "
        ).strip().lower()
        if confirm == "yes":
            run_down(_dsn)
        else:
            print("Aborted.")
            sys.exit(1)
    elif cmd == "verify":
        run_verify(_dsn)
    else:
        print(f"Unknown command: {cmd!r}. Use: up | down | verify")
        sys.exit(1)
