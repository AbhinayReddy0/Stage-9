"""
PLANNING handler:

Two responsibilities:

1. Pre-fetch the two tenant-wide lookup caches that Sub-Stages 9.4 and 9.5
   need (replacing N per-SKU round-trips with two bulk SELECTs):

     calibrated_cache  — (pattern, model) → backtest window days
                         used by run_substage_94 to skip the per-SKU SELECT
     calibration_gaps  — (pattern, model) → calibration gap float
                         used by run_substage_95 in confidence step 4

2. For every FULL/PARTIAL SKU, run Sub-Stage 9.1 (model_initialisation) in
   the main process and assemble a `SkuPipelineInput` for the dual_pool
   worker. CACHE-tier SKUs go in a separate list — they're processed by
   the main-process micro-update path inside acting_handler and never
   reach dual_pool.

   Running 9.1 in the main process lets dual_pool route SKUs to the
   process pool vs thread pool by reading `assigned_model` (Prophet vs
   everything else) at submit time.

Both caches and the assembled inputs are stored on RunContext so
acting_handler picks them up without additional DB reads.
"""
from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from backtesting.backtesting import prefetch_calibrated_windows
from infrastructure.constants import MAX_DEMAND_HISTORY_DAYS, ProcessingTier
from infrastructure.db_utils import DBConnection
from forecasting.forecasting import prefetch_calibration_gaps
from handlers._context import fetch
from pipeline.model_initialization import run_model_initialisation

log = logging.getLogger(__name__)

# Number of threads used to run Sub-Stage 9.1 in parallel. Override via env
# var if your CPU count or DB connection ceiling demands a different value.
# 9.1 work is mostly Python/dict lookups (no per-SKU DB reads — preloader
# already loaded everything), so threading parallelism is real even under
# the GIL. BatchWriter is thread-safe via internal lock.
from infrastructure.config import PLANNING_THREADS as _PLANNING_THREADS  # noqa: E402
PLANNING_PARALLELISM: int = _PLANNING_THREADS


_SQL_DEMAND_HISTORY = f"""
    SELECT sku_id::text, sale_date, qty
      FROM stage8.demand_history
     WHERE tenant_id = %s
       AND sale_date >= CURRENT_DATE - INTERVAL '{MAX_DEMAND_HISTORY_DAYS} days'
       AND sale_date <= CURRENT_DATE
     ORDER BY sku_id, sale_date
"""


def _load_demand_history(tenant_id: str, db: DBConnection) -> dict[str, pd.DataFrame]:
    """Bulk SELECT of {MAX_DEMAND_HISTORY_DAYS} days demand → {{sku_id: DataFrame(date, qty)}}."""
    rows_by_sku: dict[str, list] = {}
    with db.cursor() as cur:
        cur.execute(_SQL_DEMAND_HISTORY, (tenant_id,))
        for sku_id, sale_date, qty in cur.fetchall():
            rows_by_sku.setdefault(sku_id, []).append(
                {"date": pd.Timestamp(sale_date), "qty": float(qty or 0.0)}
            )
    return {
        sku_id: pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        for sku_id, rows in rows_by_sku.items()
    }


def planning_handler(*, tenant_id: str, run_id: str, db: DBConnection) -> None:
    log.info("planning_handler starting tenant=%s run=%s", tenant_id, run_id)
    ctx = fetch(run_id)

    # ----- 1. Calibration caches -----------------------------------------
    ctx.calibrated_cache = prefetch_calibrated_windows(db, tenant_id)
    ctx.calibration_gaps = prefetch_calibration_gaps(db, tenant_id)

    # ----- 2. Run 9.1 + assemble SkuPipelineInput per SKU ----------------
    from handlers.acting import build_sku_pipeline_input

    # Load demand history ONCE here and stash on ctx so acting_handler
    # doesn't reload the same 400-day × N-SKU window. Cuts peak memory ~50%
    # and cuts run wall-clock by the cost of one bulk SELECT.
    demand_data = _load_demand_history(tenant_id, db)
    ctx.demand_data = demand_data

    # Pre-classify SKUs: cache-tier go to a separate list (handled in
    # acting_handler's main process); FULL/PARTIAL get the 9.1 + build
    # treatment in parallel below.
    cache_sku_ids: list[str] = []
    workable_skus: list[tuple[str, str]] = []  # (sku_id, tier)
    for _sid in ctx.sku_ids:
        _stier = ctx.preloaded.sku_tiers.get(_sid, ProcessingTier.FULL)
        if _stier == ProcessingTier.CACHE:
            cache_sku_ids.append(_sid)
            continue
        if demand_data.get(_sid) is None or demand_data[_sid].empty:
            log.warning("planning_handler sku=%s has no demand data — skipped", _sid)
            continue
        workable_skus.append((_sid, _stier))

    # Worker function — runs 9.1 + builds SkuPipelineInput for one SKU.
    # Returns the assembled SkuPipelineInput, or None on per-SKU failure.
    # BatchWriter is thread-safe via its internal Lock (see batch_writer.py).
    def _process_one(sku_tier: tuple[str, str]):
        sku_id, tier = sku_tier
        try:
            lctx = run_model_initialisation(
                sku_id=sku_id,
                preloaded=ctx.preloaded,
                params=ctx.params,
                batch_writer=ctx.batch_writer,
                consumer=ctx.signal_consumer,
                run_id=run_id,
            )
        except Exception:
            log.exception("planning_handler 9.1 failed sku=%s — fallback in acting", sku_id)
            return None

        try:
            return build_sku_pipeline_input(
                sku_id=sku_id,
                lctx=lctx,
                df=demand_data[sku_id],
                run_ctx=ctx,
                calibrated_window_days=int(
                    ctx.calibrated_cache.get(
                        (lctx.pattern_label, lctx.assigned_model), 30
                    )
                ),
                calibration_gap=ctx.calibration_gaps.get(
                    (lctx.pattern_label, lctx.assigned_model)
                ),
                tier=tier,
            )
        except Exception:
            log.exception("planning_handler build_input failed sku=%s", sku_id)
            return None

    # Parallel execution — at 5M SKUs serial 9.1 is the dominant bottleneck
    # (~10ms × 5M = 14h); ThreadPool brings that down to ~14h / N_THREADS.
    # GIL is not a problem here: 9.1 work is mostly dict / preloader lookups
    # plus one BatchWriter.queue per SKU (lock-protected, fast).
    pipeline_inputs: list = []
    if workable_skus:
        with ThreadPoolExecutor(max_workers=PLANNING_PARALLELISM) as ex:
            for result in ex.map(_process_one, workable_skus):
                if result is not None:
                    pipeline_inputs.append(result)

    ctx.pipeline_inputs = pipeline_inputs
    ctx.cache_sku_ids = cache_sku_ids

    # Counter is faster than the manual dict.get(...) + 1 pattern and is
    # the idiomatic way to count occurrences.
    tier_counts: Counter = Counter(ctx.preloaded.sku_tiers.values())

    log.info(
        "planning_handler complete tenant=%s run=%s threads=%d "
        "calibrated_windows=%d calibration_gaps=%d skus_in_scope=%d "
        "pipeline_inputs=%d cache_skus=%d "
        "tiers cache=%d partial=%d full=%d",
        tenant_id, run_id, PLANNING_PARALLELISM,
        len(ctx.calibrated_cache),
        len(ctx.calibration_gaps),
        len(ctx.sku_ids),
        len(pipeline_inputs),
        len(cache_sku_ids),
        tier_counts.get(ProcessingTier.CACHE, 0),
        tier_counts.get(ProcessingTier.PARTIAL, 0),
        tier_counts.get(ProcessingTier.FULL, 0),
    )
