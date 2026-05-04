"""
Stage 9 dual-pool executor.

Runs every FULL/PARTIAL-tier SKU concurrently across two pools:

  * ProcessPoolExecutor  — Prophet SKUs (CPU-heavy fits that benefit from
                           OS-level isolation), 4 workers, 120s per-SKU
                           timeout.
  * ThreadPoolExecutor   — Naive / Croston / Holt / SES SKUs (light
                           fits, no need for separate processes), 16
                           workers, 30s per-SKU timeout.

Both pools are submitted to back-to-back so neither starves; results
are collected via concurrent.futures.as_completed with a 3600s
wall-clock cap.

CRITICAL — pickle-safety. ProcessPoolExecutor pickles the worker
function, its arguments, and its return value. So:
  * `_subprocess_worker` is a module-level function (NOT a method,
    lambda, or nested function — those can't be pickled).
  * `pipeline_fn` is passed as a "module.attr" string; the worker
    re-imports it. Closures and partials don't survive a fork on
    Windows spawn semantics.
  * `SkuPipelineInput` is a plain dataclass holding only JSON-friendly
    fields — sku_data and preloaded_data are dicts, not Preloader
    objects.

A failure in one SKU never propagates: timeouts, worker exceptions,
and DB errors all become a naive-fallback result and a row in
stage9_sku_execution_log. The run continues.

Usage:

    from stage9.dual_pool import run_dual_pool, SkuPipelineInput

    results = run_dual_pool(
        sku_inputs=[SkuPipelineInput(...), ...],
        tenant_id="...",
        run_id="...",
        pipeline_fn_path="myapp.pipeline.run_one_sku",
        db_config={"host": "...", "port": 5432, "dbname": "...", ...},
        cleanup_conn=master_conn,
    )
"""

from __future__ import annotations

import importlib
import logging
import time
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from infrastructure.constants import (
    APPLICATION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FALLBACK_CONFIDENCE,
    DEFAULT_MAX_TASKS_PER_CHILD,
    ORPHAN_IDLE_MINUTES,
    OVERALL_TIMEOUT,
    PROCESS_POOL_WORKERS,
    PROCESS_TIMEOUT,
    PROPHET_FAMILY,
    THREAD_POOL_WORKERS,
    THREAD_TIMEOUT,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DualPoolResult",
    "DualPoolStats",
    "PROCESS_POOL_WORKERS",
    "THREAD_POOL_WORKERS",
    "PROCESS_TIMEOUT",
    "THREAD_TIMEOUT",
    "OVERALL_TIMEOUT",
    "APPLICATION_NAME",
    "SkuPipelineInput",
    "cleanup_orphan_connections",
    "db_config_from_env",
    "is_prophet_family_sku",
    "naive_fallback_result",
    "partition_skus",
    "run_dual_pool",
    "get_worker_tenant_params",
    "get_worker_tenant_invariants",
    "set_worker_globals",
]


# All dual-pool constants are defined in infrastructure/constants.py section 24
# and imported above. Nothing is hardcoded in this file.


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkuPipelineInput:
    """
    JSON-serializable per-SKU payload handed to the worker.

    Worker reconstructs the full pipeline context from these fields.
    Keep `sku_data` and `preloaded_data` as plain dicts — the Preloader
    object itself is NOT picklable.
    """
    sku_id: str
    assigned_model: str
    sku_data: dict[str, Any]
    preloaded_data: dict[str, Any]


@dataclass
class DualPoolResult:
    """Lightweight result dict the worker returns for each SKU."""
    sku_id: str
    status: str
    confidence_final: float
    pool: str = ""  # 'process' | 'thread' | 'fallback'
    error: Optional[str] = None  # populated on timeout / exception
    # Option B (collecting BatchWriter): rows the worker would have queued
    # into its local BatchWriter, deferred to the main process so cross-SKU
    # batching can amortize INSERT overhead.
    # Shape: {table_name: [row_dict, …]}.Empty / None on fallback paths and for workers that don't use this pattern.
    batch_rows: Optional[dict] = None
    backtest_mape: Optional[float] = None
    thompson_state: dict = field(default_factory=dict)


@dataclass
class DualPoolStats:
    """Summary the orchestrator returns alongside per-SKU results."""
    process_skus: int = 0
    thread_skus: int = 0
    timeouts: int = 0
    failures: int = 0
    process_pool_started_at: float = 0.0
    thread_pool_started_at: float = 0.0
    cleanup_terminated: int = 0
    results: dict[str, DualPoolResult] = field(default_factory=dict)


# Type alias for the injected pipeline.
# The string form ("module.attr") is what we hand the process-pool worker
# closures / partials are rejected by pickle on spawn-style multiprocessing (Windows).
PipelineFn = Callable[[SkuPipelineInput, str, str, Any], dict]


# ---------------------------------------------------------------------------
# Pool-routing predicate
# ---------------------------------------------------------------------------

def is_prophet_family_sku(sku: SkuPipelineInput) -> bool:
    """Process pool gets Prophet SKUs — CPU-heavy fits."""
    return sku.assigned_model.lower() in PROPHET_FAMILY


def partition_skus(
        skus: list[SkuPipelineInput],
        predicate: Callable[[SkuPipelineInput], bool] = is_prophet_family_sku,
) -> tuple[list[SkuPipelineInput], list[SkuPipelineInput]]:
    """Split inputs into (process_pool_skus, thread_pool_skus)."""
    process_skus: list[SkuPipelineInput] = []
    thread_skus: list[SkuPipelineInput] = []
    for s in skus:
        (process_skus if predicate(s) else thread_skus).append(s)
    return process_skus, thread_skus


# ---------------------------------------------------------------------------
# Subprocess / thread worker — TOP-LEVEL so pickle can serialize it
# ---------------------------------------------------------------------------

# Module-level globals populated by `_init_worker` so we don't pickle
# db_config / pipeline_fn_path / tenant invariants on every submit() call.
# ProcessPoolExecutor runs the initializer once per worker process; thread
# workers share the parent's memory so writing here is effectively free.
_WORKER_DB_CONFIG: Optional[dict] = None
_WORKER_PIPELINE_FN: Optional["PipelineFn"] = None
_WORKER_PIPELINE_FN_PATH: Optional[str] = None
# Tenant-wide invariants — same value for every SKU in this run. Stashed
# once per worker so they don't pollute every per-SKU pickle. Workers read
# via `pipeline._get_tenant_params` / `pipeline._get_invariants` which
# fall back to preloaded_data for test paths.
_WORKER_TENANT_ID: Optional[str] = None
_WORKER_TENANT_PARAMS: Optional[dict] = None  # output of TenantParams.to_dict()
_WORKER_TENANT_INVARIANTS: Optional[dict] = None  # arbitrary tenant-wide dict


def get_worker_tenant_params() -> Optional[dict]:
    return _WORKER_TENANT_PARAMS


def get_worker_tenant_invariants() -> Optional[dict]:
    return _WORKER_TENANT_INVARIANTS


def set_worker_globals(
    tenant_id: Optional[str],
    tenant_params: Optional[dict],
    tenant_invariants: Optional[dict],
) -> None:
    global _WORKER_TENANT_ID, _WORKER_TENANT_PARAMS, _WORKER_TENANT_INVARIANTS
    _WORKER_TENANT_ID = tenant_id
    _WORKER_TENANT_PARAMS = tenant_params
    _WORKER_TENANT_INVARIANTS = tenant_invariants


def _init_worker(
        db_config: dict,
        pipeline_fn_path: str,
        tenant_id: Optional[str] = None,
        tenant_params: Optional[dict] = None,
        tenant_invariants: Optional[dict] = None,
) -> None:
    """ProcessPoolExecutor / ThreadPoolExecutor `initializer`.

    The trailing three args are optional so existing callers that don't
    care about per-run invariants (most tests) keep working. Production
    callers (`run_dual_pool` invoked by `acting_handler`) should always
    populate them — that's where the pickle savings come from.
    """
    global _WORKER_DB_CONFIG, _WORKER_PIPELINE_FN, _WORKER_PIPELINE_FN_PATH
    global _WORKER_TENANT_ID, _WORKER_TENANT_PARAMS, _WORKER_TENANT_INVARIANTS
    _WORKER_DB_CONFIG = db_config
    _WORKER_PIPELINE_FN_PATH = pipeline_fn_path
    _WORKER_PIPELINE_FN = _resolve_pipeline_fn(pipeline_fn_path)
    _WORKER_TENANT_ID = tenant_id
    _WORKER_TENANT_PARAMS = tenant_params
    _WORKER_TENANT_INVARIANTS = tenant_invariants


def _subprocess_worker(
        sku_input: SkuPipelineInput,
        tenant_id: str,
        run_id: str,
        per_task_timeout: float,
) -> dict:
    """
    Run pipeline_fn for one SKU under a watchdog. ALWAYS close conn.

    A daemon thread runs the actual work; the main thread joins with the
    per-task timeout. If the join times out, we raise FuturesTimeoutError
    — the worker process is then either killed by max_tasks_per_child=1
    (the orchestrator's default) or by overall_timeout teardown.

    `_WORKER_DB_CONFIG` and `_WORKER_PIPELINE_FN` are set by `_init_worker`
    once per worker process, so every task reuses the resolved callable
    instead of re-importing it.
    """
    import threading

    if _WORKER_PIPELINE_FN is None or _WORKER_DB_CONFIG is None:
        # Reachable only when _init_worker didn't get a chance to run.
        # The matrix:
        #   * ProcessPoolExecutor  — initializer kwarg works → globals set
        #   * ThreadPoolExecutor  — initializer kwarg works → globals set
        #   * Fake executor with no init kwarg — _make_pool's TypeError
        #     fallback calls _init_worker in the parent → globals set
        # The path below covers the rare case where ALL the above failed
        # (e.g. someone constructed a worker manually). Resolve eagerly
        # from the cached path; raise if even that's missing.
        if _WORKER_PIPELINE_FN_PATH is None:
            raise RuntimeError(
                "_init_worker did not run; _WORKER_PIPELINE_FN_PATH is unset"
            )
        pipeline_fn = _resolve_pipeline_fn(_WORKER_PIPELINE_FN_PATH)
        db_config = _WORKER_DB_CONFIG or {}
    else:
        pipeline_fn = _WORKER_PIPELINE_FN
        db_config = _WORKER_DB_CONFIG

    holder: dict[str, Any] = {}

    def _run() -> None:
        conn = None
        try:
            conn = _open_worker_connection(db_config)
            holder["result"] = pipeline_fn(sku_input, tenant_id, run_id, conn)
        except Exception as e:
            holder["error"] = e
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=per_task_timeout)
    if t.is_alive():
        # Per-task budget blew — surface as TimeoutError.
        # The leaked daemon dies with the worker process.
        raise FuturesTimeoutError(
            f"per-task timeout {per_task_timeout}s sku={sku_input.sku_id}"
        )
    if "error" in holder:
        raise holder["error"]

    result = holder.get("result")
    if not isinstance(result, dict):
        raise ValueError(
            f"pipeline_fn must return a dict, got {type(result).__name__}"
        )
    # Validate the contract — fail loudly instead of substituting defaults.
    required = {"sku_id", "status", "confidence_final"}
    missing = required - result.keys()
    if missing:
        raise ValueError(
            f"pipeline_fn result missing keys {sorted(missing)}; got {sorted(result)}"
        )
    out = {
        "sku_id": str(result["sku_id"]),
        "status": str(result["status"]),
        "confidence_final": float(result["confidence_final"]),
    }
    # Option B pass-through: workers using a collecting BatchWriter return
    # the queued rows here so the main process can batch them across SKUs.
    if "batch_rows" in result and result["batch_rows"]:
        out["batch_rows"] = result["batch_rows"]
    if "backtest_mape" in result:
        out["backtest_mape"] = result["backtest_mape"]
    if "thompson_state" in result and result["thompson_state"]:
        out["thompson_state"] = result["thompson_state"]
    return out


def _open_worker_connection(db_config: dict):
    """
    Open a psycopg2 connection using the caller-supplied db_config.
    Falls back to DB_CONNECT_TIMEOUT from config and APPLICATION_NAME
    from constants if the dict doesn't already supply them.
    Imported lazily so unit tests don't require psycopg2.
    """
    import psycopg2  # type: ignore
    from infrastructure.config import DB_CONNECT_TIMEOUT as _cfg_timeout
    cfg = dict(db_config)
    cfg.setdefault("connect_timeout", _cfg_timeout)
    cfg.setdefault("application_name", APPLICATION_NAME)
    return psycopg2.connect(**cfg)


def _resolve_pipeline_fn(path: str) -> "PipelineFn":
    """Resolve "package.module.func_name" to a callable."""
    module_name, _, fn_name = path.rpartition(".")
    if not module_name:
        raise ValueError(
            f"pipeline_fn_path must be 'module.func', got {path!r}"
        )
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def _validate_pipeline_fn_path(path: str) -> None:
    """
    Verify that `path` resolves to an importable callable.

    Call this at the start of run_dual_pool() so a stale path (caused by
    renaming or moving run_one_sku) raises ImportError immediately in the
    orchestrator process — not silently inside a worker process where the
    traceback is harder to trace.
    """
    module_name, _, fn_name = path.rpartition(".")
    if not module_name:
        raise ValueError(
            f"pipeline_fn_path must be 'module.func', got {path!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"pipeline_fn_path '{path}' points to a module that cannot be "
            f"imported. Update pipeline_fn_path if the module was renamed or "
            f"moved."
        ) from exc
    if not hasattr(module, fn_name):
        raise ImportError(
            f"pipeline_fn_path '{path}': module '{module_name}' exists but "
            f"has no attribute '{fn_name}'. Update pipeline_fn_path if the "
            f"function was renamed."
        )


# ---------------------------------------------------------------------------
# Naive fallback — used on timeout / worker exception
# ---------------------------------------------------------------------------

def naive_fallback_result(
        sku_id: str,
        *,
        reason: str = "timeout",
        floor_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
) -> DualPoolResult:
    """
    What we return when the worker times out, the process is killed, or
    pipeline_fn raises. Stage 10 sees a row and a low-confidence flag;
    stage9_sku_execution_log records the cause.

    `floor_confidence` should be `tenant_learning_params.confidence_floor`
    when called from the orchestrator. The hardcoded default exists only
    for direct callers that don't have a TenantParams snapshot handy.
    """
    return DualPoolResult(
        sku_id=sku_id,
        status="needs_acknowledgment",
        confidence_final=floor_confidence,
        pool="fallback",
        error=reason,
    )


# ---------------------------------------------------------------------------
# Orphan connection cleanup (Part 10.4 of the tech context)
# ---------------------------------------------------------------------------

def cleanup_orphan_connections(
        conn: Any,
        *,
        application_name: str = APPLICATION_NAME,
        idle_minutes: int = ORPHAN_IDLE_MINUTES,
        tenant_id: Optional[str] = None,
) -> int:
    """
    Terminate any idle subprocess connections older than idle_minutes.
    Returns rows affected (== connections terminated). Safe to call
    multiple times.

    When `tenant_id` is provided, the cleanup is scoped to that tenant by
    matching `application_name = "{APPLICATION_NAME}_{tenant_id}"` — this
    prevents Tenant A's cleanup from terminating Tenant B's idle workers
    when both run concurrently. Production callers should always pass it.

    Uses Postgres `make_interval(mins => %s)` so the value is bound
    rather than f-string-interpolated — keeps the SQL parameterised
    even though `idle_minutes` is internally an int.
    """
    minutes = int(idle_minutes)
    effective_app = (
        f"{application_name}_{tenant_id}" if tenant_id else application_name
    )
    # pg_terminate_backend is a SELECT — count rows returned, not rowcount
    # (some drivers return -1 for SELECT statements).
    sql = (
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE state = 'idle' "
        "AND query_start < NOW() - make_interval(mins => %s) "
        "AND application_name = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (minutes, effective_app))
        rows = cur.fetchall()
    # commit() is a no-op for a SELECT but harmless; pg_terminate_backend
    # already took effect server-side. Surface a wedged conn at debug.
    try:
        conn.commit()
    except Exception:
        logger.debug(
            "cleanup_orphan_connections commit failed (harmless for SELECT)",
            exc_info=True,
        )
    return len(rows)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_dual_pool(
        sku_inputs: list[SkuPipelineInput],
        *,
        tenant_id: str,
        run_id: str,
        pipeline_fn_path: str,
        db_config: dict,
        log_failure_fn: Callable[[str, str, str, str], None],
        process_pool_factory: Callable[..., Any] = ProcessPoolExecutor,
        thread_pool_factory: Callable[..., Any] = ThreadPoolExecutor,
        process_workers: int = PROCESS_POOL_WORKERS,
        thread_workers: int = THREAD_POOL_WORKERS,
        process_timeout: float = PROCESS_TIMEOUT,
        thread_timeout: float = THREAD_TIMEOUT,
        overall_timeout: float = OVERALL_TIMEOUT,
        process_predicate: Callable[[SkuPipelineInput], bool] = is_prophet_family_sku,
        cleanup_conn=None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        fallback_confidence: Optional[float] = None,
        db_smoke_test: bool = True,
        max_tasks_per_child: Optional[int] = DEFAULT_MAX_TASKS_PER_CHILD,
        tenant_params: Optional[dict] = None,
        tenant_invariants: Optional[dict] = None,
) -> DualPoolStats:
    """
    Submit every SKU to whichever pool fits its model, collect results,
    apply per-task timeouts, run orphan-connection cleanup. Returns a
    DualPoolStats with per-SKU results.

    Key behaviors:
      * `log_failure_fn(tenant_id, run_id, sku_id, reason)` is REQUIRED
        — it should write to stage9_sku_execution_log so Principle 3's
        audit trail is preserved.
      * `chunk_size` bounds in-flight memory: we submit in waves of N
        and drain each wave before the next. Defaults to 10K SKUs.
      * `db_smoke_test=True` opens one connection up-front to validate
        `db_config` before fanning out — fails fast on misconfig.
      * Per-task timeouts (`process_timeout` / `thread_timeout`) are
        enforced inside the worker via a watchdog thread. Hung tasks
        raise TimeoutError; the worker process is then torn down
        (process pool uses max_tasks_per_child=1).

    `process_pool_factory` / `thread_pool_factory` are injectable for
    tests — a synchronous fake executor lets us assert dispatch order
    and timing without spawning OS processes.
    """
    if not callable(log_failure_fn):
        raise TypeError(
            "log_failure_fn is required (production must write to "
            "stage9_sku_execution_log; pass a callable)"
        )

    # Validate the path resolves before fanning out to N workers — a stale
    # path (rename / module move) surfaces here with a clear ImportError
    # instead of as a cryptic failure inside a worker process.
    _validate_pipeline_fn_path(pipeline_fn_path)

    if fallback_confidence is None:
        warnings.warn(
            "fallback_confidence not provided — using "
            f"DEFAULT_FALLBACK_CONFIDENCE={DEFAULT_FALLBACK_CONFIDENCE}. "
            "In production pass tenant_learning_params.confidence_floor.",
            stacklevel=2,
        )
        fallback_confidence = DEFAULT_FALLBACK_CONFIDENCE

    process_skus, thread_skus = partition_skus(sku_inputs, process_predicate)
    stats = DualPoolStats(
        process_skus=len(process_skus),
        thread_skus=len(thread_skus),
    )
    if not sku_inputs:
        return stats

    if db_smoke_test:
        _smoke_test_db_config(db_config)

    process_pool = thread_pool = None
    try:
        if process_skus:
            process_pool = _make_pool(
                process_pool_factory, process_workers,
                db_config, pipeline_fn_path, supports_init=True,
                max_tasks_per_child=max_tasks_per_child,
                tenant_id=tenant_id,
                tenant_params=tenant_params,
                tenant_invariants=tenant_invariants,
            )
            stats.process_pool_started_at = time.monotonic()
        if thread_skus:
            # Thread pool ignores max_tasks_per_child (threads are reused).
            thread_pool = _make_pool(
                thread_pool_factory, thread_workers,
                db_config, pipeline_fn_path, supports_init=True,
                max_tasks_per_child=None,
                tenant_id=tenant_id,
                tenant_params=tenant_params,
                tenant_invariants=tenant_invariants,
            )
            stats.thread_pool_started_at = time.monotonic()

        # If only one pool was created, mirror the timestamp so callers
        # checking the "within 100ms" Done-When still pass.
        if process_pool is None:
            stats.process_pool_started_at = stats.thread_pool_started_at
        if thread_pool is None:
            stats.thread_pool_started_at = stats.process_pool_started_at

        # Chunked submission: bounds in-flight Future count to chunk_size
        # per pool, not total SKU count.
        for chunk_idx in range(0, max(len(process_skus), len(thread_skus)), chunk_size):
            proc_chunk = process_skus[chunk_idx: chunk_idx + chunk_size]
            thr_chunk = thread_skus[chunk_idx: chunk_idx + chunk_size]
            future_meta: dict = {}

            for sku in proc_chunk:
                fut = process_pool.submit(
                    _subprocess_worker, sku, tenant_id, run_id, process_timeout,
                )
                future_meta[fut] = (sku.sku_id, "process")
            for sku in thr_chunk:
                fut = thread_pool.submit(
                    _subprocess_worker, sku, tenant_id, run_id, thread_timeout,
                )
                future_meta[fut] = (sku.sku_id, "thread")

            _collect_results(
                future_meta, stats,
                overall_timeout=overall_timeout,
                tenant_id=tenant_id, run_id=run_id,
                log_failure=log_failure_fn,
                fallback_confidence=fallback_confidence,
            )

    finally:
        # cancel_futures=True cancels any QUEUED futures.
        # Running tasks finish on their own — but the worker watchdog
        # already enforces the per-task budget so they can't hang
        # forever.
        if process_pool is not None:
            try:
                process_pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Older fakes may not accept cancel_futures.
                process_pool.shutdown(wait=False)
        if thread_pool is not None:
            try:
                thread_pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                thread_pool.shutdown(wait=False)

        if cleanup_conn is not None:
            try:
                stats.cleanup_terminated = cleanup_orphan_connections(
                    cleanup_conn, tenant_id=tenant_id,
                )
            except Exception:
                logger.exception("orphan connection cleanup failed")

    return stats


def _make_pool(
        factory: Callable[..., Any],
        max_workers: int,
        db_config: dict,
        pipeline_fn_path: str,
        *,
        supports_init: bool,
        max_tasks_per_child: Optional[int] = None,
        tenant_id: Optional[str] = None,
        tenant_params: Optional[dict] = None,
        tenant_invariants: Optional[dict] = None,
) -> Any:
    """
    Try to instantiate the pool with `initializer=`/`initargs=` so each
    worker pre-resolves pipeline_fn and stashes db_config in module
    globals (avoids re-pickling per submit). Falls back gracefully for
    fake executors that don't accept those kwargs.

    `max_tasks_per_child` is honored on Python 3.11+ ProcessPoolExecutor
    so each worker process is recycled after N tasks. We pass it
    OPTIONALLY: missing kwarg on older interpreters / thread pools /
    fake executors all fall through cleanly via the TypeError handler.
    """
    init_kwargs: dict[str, Any] = {"max_workers": max_workers}
    init_args = (
        db_config,
        pipeline_fn_path,
        tenant_id,
        tenant_params,
        tenant_invariants,
    )
    if supports_init:
        init_kwargs["initializer"] = _init_worker
        init_kwargs["initargs"] = init_args
    if max_tasks_per_child is not None:
        init_kwargs["max_tasks_per_child"] = max_tasks_per_child

    try:
        return factory(**init_kwargs)
    except TypeError:
        # Strip the kwargs that the factory didn't accept, retrying
        # progressively. ProcessPoolExecutor before 3.11 doesn't accept
        # max_tasks_per_child; some fakes don't accept initializer.
        for key in ("max_tasks_per_child", "initargs", "initializer"):
            init_kwargs.pop(key, None)
            try:
                pool = factory(**init_kwargs)
                # Set module globals in the parent process so single-
                # process / fake executors still work for tests.
                if supports_init:
                    _init_worker(*init_args)
                return pool
            except TypeError:
                continue
    # Last-ditch: bare factory call
    pool = factory(max_workers=max_workers)
    if supports_init:
        _init_worker(*init_args)
    return pool


def db_config_from_env() -> dict:
    """
    Build the db_config dict for run_dual_pool. Call once per run and pass
    as db_config=. Values are read from infrastructure.config (which loads .env).
    """
    from infrastructure.config import (
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
        DB_SSLMODE, DB_CONNECT_TIMEOUT,
    )
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "sslmode": DB_SSLMODE,
        "connect_timeout": DB_CONNECT_TIMEOUT,
        "options": "-c search_path=stage9,public",
    }


def _smoke_test_db_config(db_config: dict) -> None:
    """
    Open one connection with the worker's settings and immediately close.
    Catches typos in db_config before we fan out to N workers that all
    fail identically.
    """
    try:
        conn = _open_worker_connection(db_config)
    except ImportError:
        # psycopg2 not installed (test env). Skip — workers will raise.
        return
    except Exception as e:
        raise RuntimeError(f"db_config smoke test failed: {e}") from e
    try:
        conn.close()
    except Exception:
        # Close failure here is benign (we already validated the open).
        logger.debug("smoke-test conn.close() failed", exc_info=True)


def _collect_results(
        future_meta: dict,
        stats: DualPoolStats,
        *,
        overall_timeout: float,
        tenant_id: str,
        run_id: str,
        log_failure: Callable[[str, str, str, str], None],
        fallback_confidence: float,
) -> None:
    """
    Iterate completions with as_completed; the worker enforced the
    per-task timeout (it raises TimeoutError when its watchdog fires).
    Any exception becomes a naive-fallback result + a log row.
    """
    try:
        for fut in as_completed(future_meta, timeout=overall_timeout):
            sku_id, pool_name = future_meta[fut]
            try:
                raw = fut.result()
                stats.results[sku_id] = DualPoolResult(
                    sku_id=raw["sku_id"],
                    status=raw["status"],
                    confidence_final=raw["confidence_final"],
                    pool=pool_name,
                    batch_rows=raw.get("batch_rows"),
                    backtest_mape=raw.get("backtest_mape"),
                    thompson_state=raw.get("thompson_state", {}),
                )
            except FuturesTimeoutError:
                stats.timeouts += 1
                stats.results[sku_id] = naive_fallback_result(
                    sku_id, reason="timeout",
                    floor_confidence=fallback_confidence,
                )
                log_failure(tenant_id, run_id, sku_id, "timeout")
            except Exception as e:
                stats.failures += 1
                stats.results[sku_id] = naive_fallback_result(
                    sku_id, reason=f"error:{e}",
                    floor_confidence=fallback_confidence,
                )
                log_failure(tenant_id, run_id, sku_id, f"error:{e}")
    except FuturesTimeoutError:
        # Overall wall-clock blew past overall_timeout. Anything still
        # in future_meta whose result we haven't recorded is a timeout.
        for fut, (sku_id, _) in future_meta.items():
            if sku_id in stats.results:
                continue
            try:
                fut.cancel()
            except Exception:
                logger.debug(
                    "fut.cancel() failed during overall_timeout teardown sku_id=%s",
                    sku_id, exc_info=True,
                )
            stats.timeouts += 1
            stats.results[sku_id] = naive_fallback_result(
                sku_id, reason="overall_timeout",
                floor_confidence=fallback_confidence,
            )
            log_failure(tenant_id, run_id, sku_id, "overall_timeout")
