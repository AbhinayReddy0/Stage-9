"""
LEARNING handler.

Post-run learning tasks after all SKUs have been processed:

  1. Flush any remaining BatchWriter rows to DB.
  2. Upsert updated Thompson sampling state — Sub-Stage 9.3 accumulates
     all alpha/beta updates in memory; this is the single bulk write.
  3. Upsert sku_similarity_registry for SKUs whose validation MAPE is
     good enough to serve as warm-start references for new products.
  4. Emit forecast_accuracy signal → Stage 8 (aggregate MAPE per model).

No FAILED edge from LEARNING (see VALID_TRANSITIONS) — any exception
here surfaces as an unhandled error directly to the caller.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from infrastructure.constants import Agent, Param, SignalType
from infrastructure.db_utils import DBConnection
from handlers._context import fetch

_SIGNAL_MAX_RETRIES = 3
_SIGNAL_RETRY_DELAY_S = 0.1

try:
    from psycopg2.extras import Json as _Json  # type: ignore
except Exception:
    _Json = None

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thompson state upsert
# ---------------------------------------------------------------------------

_UPSERT_THOMPSON = """
    INSERT INTO stage9.thompson_sampling_state
        (tenant_id, sku_id, assigned_model, config_hash,
         config_json, alpha_param, beta_param, total_trials, last_updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (tenant_id, sku_id, assigned_model, config_hash)
    DO UPDATE SET
        alpha_param     = EXCLUDED.alpha_param,
        beta_param      = EXCLUDED.beta_param,
        total_trials    = stage9.thompson_sampling_state.total_trials + 1,
        last_updated_at = NOW()
"""

# ---------------------------------------------------------------------------
# SKU similarity registry upsert
# ---------------------------------------------------------------------------

_UPSERT_SIMILARITY = """
    INSERT INTO stage9.sku_similarity_registry
        (tenant_id, sku_id, pattern_label, vendor, product_type, parent_style_id,
         observation_days, weekend_zero_ratio, best_model_config,
         best_features, avg_mape, last_updated)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (tenant_id, sku_id)
    DO UPDATE SET
        pattern_label     = EXCLUDED.pattern_label,
        vendor            = EXCLUDED.vendor,
        product_type      = EXCLUDED.product_type,
        parent_style_id   = EXCLUDED.parent_style_id,
        observation_days  = EXCLUDED.observation_days,
        weekend_zero_ratio = EXCLUDED.weekend_zero_ratio,
        best_model_config = EXCLUDED.best_model_config,
        best_features     = EXCLUDED.best_features,
        avg_mape          = EXCLUDED.avg_mape,
        last_updated      = NOW()
"""

# ---------------------------------------------------------------------------
# forecast_accuracy signal INSERT
# ---------------------------------------------------------------------------

_UPSERT_FINGERPRINT = """
    INSERT INTO stage9.data_fingerprint_cache
        (tenant_id, sku_id, fingerprint, tier, pattern_label, demand_total, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (tenant_id, sku_id)
    DO UPDATE SET
        fingerprint   = EXCLUDED.fingerprint,
        tier          = EXCLUDED.tier,
        pattern_label = EXCLUDED.pattern_label,
        demand_total  = EXCLUDED.demand_total,
        updated_at    = NOW()
"""

_INSERT_SIGNAL = """
    INSERT INTO stage9.cross_agent_signals
        (signal_id, tenant_id, from_agent, to_agent, signal_type,
         run_id, payload, processed, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, NOW())
"""

_INSERT_CROSS_SKU_SIGNAL = """
    INSERT INTO stage9.cross_agent_signals
        (signal_id, tenant_id, from_agent, to_agent, signal_type,
         run_id, payload, processed, created_at, expires_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, NOW(), NOW() + make_interval(hours => 1440))
"""


def _jsonb(obj: Any) -> Any:
    return _Json(obj) if _Json else json.dumps(obj)


def _emit_cross_sku_signal(
        db: DBConnection,
        *,
        tenant_id: str,
        run_id: str,
        converged_sku_ids: list[str],
) -> None:
    """Write cross_sku_learning signal with 3-attempt retry (matches forecast_risk)."""
    args = (
        str(uuid.uuid4()),
        tenant_id,
        Agent.STAGE_9,
        Agent.STAGE_9,
        SignalType.CROSS_SKU_LEARNING,
        run_id,
        _jsonb({"run_id": run_id, "converged_sku_ids": converged_sku_ids}),
    )
    last_err: Any = None
    for attempt in range(1, _SIGNAL_MAX_RETRIES + 1):
        try:
            with db.cursor() as cur:
                cur.execute(_INSERT_CROSS_SKU_SIGNAL, args)
            db.commit()
            log.info(
                "learning_handler cross_sku_learning signal emitted tenant=%s converged=%d",
                tenant_id, len(converged_sku_ids),
            )
            return
        except Exception as e:
            last_err = e
            log.warning(
                "cross_sku_learning signal attempt %d/%d failed tenant=%s err=%s",
                attempt, _SIGNAL_MAX_RETRIES, tenant_id, e,
            )
            try:
                db.rollback()
            except Exception:
                pass
            if attempt < _SIGNAL_MAX_RETRIES:
                time.sleep(_SIGNAL_RETRY_DELAY_S)
    log.error(
        "cross_sku_learning signal FAILED after %d attempts tenant=%s err=%s",
        _SIGNAL_MAX_RETRIES, tenant_id, last_err,
    )


def learning_handler(*, tenant_id: str, run_id: str, db: DBConnection) -> None:
    log.info("learning_handler starting tenant=%s run=%s", tenant_id, run_id)
    ctx = fetch(run_id)

    # ------------------------------------------------------------------
    # 1. Flush any remaining BatchWriter rows
    # ------------------------------------------------------------------
    ctx.batch_writer.flush()
    log.info("learning_handler batch_writer flushed tenant=%s run=%s", tenant_id, run_id)

    # ------------------------------------------------------------------
    # 2. Upsert Thompson sampling state in bulk
    #    preloaded.thompson_state was mutated in-memory by Sub-Stage 9.3
    #    Format: {(sku_id, model): {config_hash: {"alpha", "beta", "config"}}}
    # ------------------------------------------------------------------
    thompson_rows = []
    for (sku_id, model), configs in ctx.preloaded.thompson_state.items():
        for cfg_hash, state in configs.items():
            thompson_rows.append((
                tenant_id,
                sku_id,
                model,
                cfg_hash,
                _jsonb(state.get("config", {})),
                state.get("alpha", 1.0),
                state.get("beta", 1.0),
                state.get("total_trials", 0) + 1,
            ))

    if thompson_rows:
        with db.cursor() as cur:
            for row in thompson_rows:
                cur.execute(_UPSERT_THOMPSON, row)
        db.commit()
        log.info(
            "learning_handler thompson_state flushed tenant=%s configs=%d",
            tenant_id, len(thompson_rows),
        )

    # ------------------------------------------------------------------
    # 3. Update sku_similarity_registry for converged SKUs
    #    A SKU is converged when its backtest + validation MAPE is low
    #    enough to serve as a warm-start reference for new products.
    # ------------------------------------------------------------------
    convergence_threshold = (
        ctx.params.get(Param.WARM_START_MAX_MAPE) if ctx.params is not None else 0.25
    )

    similarity_rows = []
    for result in ctx.sku_results:
        bt_mape = result.backtest_mape
        if bt_mape is None:
            continue
        if bt_mape <= convergence_threshold:
            preloaded = ctx.preloaded
            features = preloaded.feature_history.get(result.sku_id, [])
            thompson_key = (result.sku_id, result.assigned_model)
            configs = preloaded.thompson_state.get(thompson_key, {})
            best_cfg = {}
            if configs:
                best_cfg_hash = max(
                    configs,
                    key=lambda h: configs[h].get("alpha", 1.0) / (
                            configs[h].get("alpha", 1.0) + configs[h].get("beta", 1.0)
                    ),
                )
                best_cfg = configs[best_cfg_hash].get("config", {})

            pctx = preloaded.pattern_ctx.get(result.sku_id, {})
            similarity_rows.append((
                tenant_id,
                result.sku_id,
                result.pattern_label,
                pctx.get("vendor"),
                pctx.get("product_type"),
                pctx.get("parent_style_id"),
                int(pctx["obs_days"]) if pctx.get("obs_days") is not None else None,
                float(pctx["weekend_zero_ratio"]) if pctx.get("weekend_zero_ratio") is not None else None,
                _jsonb(best_cfg),
                _jsonb(features),
                bt_mape,
            ))

    if similarity_rows:
        with db.cursor() as cur:
            for row in similarity_rows:
                cur.execute(_UPSERT_SIMILARITY, row)
        db.commit()
        log.info(
            "learning_handler similarity_registry updated tenant=%s converged=%d",
            tenant_id, len(similarity_rows),
        )

        # Emit cross_sku_learning signal — real-time broadcast to concurrent
        # Stage 9 runs so they can warm-start from sku_similarity_registry
        # without polling. 60-day TTL (1440 hours) per spec. 3-retry for
        # durability parity with forecast_risk.
        converged_sku_ids = [row[1] for row in similarity_rows]
        _emit_cross_sku_signal(db, tenant_id=tenant_id, run_id=run_id,
                               converged_sku_ids=converged_sku_ids)

    # ------------------------------------------------------------------
    # 4. Emit forecast_accuracy signal → Stage 8
    #    Aggregate MAPE per model across this run's processed SKUs.
    # ------------------------------------------------------------------
    mape_by_model: dict[str, list[float]] = {}
    for result in ctx.sku_results:
        if not result.used_fallback and result.backtest_mape is not None:
            mape_by_model.setdefault(result.assigned_model, []).append(
                float(result.backtest_mape)
            )

    if mape_by_model:
        accuracy_payload = {
            model: round(sum(v) / len(v), 6)
            for model, v in mape_by_model.items()
        }
        with db.cursor() as cur:
            cur.execute(
                _INSERT_SIGNAL,
                (
                    str(uuid.uuid4()),
                    tenant_id,
                    Agent.STAGE_9,
                    Agent.STAGE_8,
                    SignalType.FORECAST_ACCURACY,
                    run_id,
                    _jsonb({"run_id": run_id, "mape_by_model": accuracy_payload}),
                ),
            )
        db.commit()
        log.info(
            "learning_handler forecast_accuracy signal emitted tenant=%s models=%s",
            tenant_id, list(accuracy_payload),
        )

    # ------------------------------------------------------------------
    # 5. Upsert data_fingerprint_cache — persist this run's fingerprints
    #    so the next run can classify cache/partial/full without re-reading
    #    demand history.
    # ------------------------------------------------------------------
    new_fps = ctx.preloaded.new_fingerprints
    if new_fps:
        from infrastructure.constants import ProcessingTier
        with db.cursor() as cur:
            for sku_id, entry in new_fps.items():
                # `tier` is NOT NULL in the schema and is what the next
                # run's preloader compares against to decide cache-vs-full.
                # Falls back to FULL if the SKU somehow isn't classified
                # (defensive — should never happen post-preloader).
                tier = ctx.preloaded.sku_tiers.get(sku_id, ProcessingTier.FULL)
                cur.execute(
                    _UPSERT_FINGERPRINT,
                    (
                        tenant_id, sku_id, entry["fingerprint"], tier,
                        entry.get("pattern_label"),
                        entry.get("demand_total"),
                    ),
                )
        db.commit()
        log.info(
            "learning_handler fingerprint_cache updated tenant=%s skus=%d",
            tenant_id, len(new_fps),
        )

    log.info(
        "learning_handler complete tenant=%s run=%s", tenant_id, run_id
    )
