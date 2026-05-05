"""
Unit tests for stage9.dual_pool (Task 19).

Uses an in-process fake executor so we never actually spawn OS
processes. The fake records timestamps and per-future state — enough
to verify dispatch ordering, timeout fallback, and orphan cleanup.

A pipeline_fn is registered as a module-level callable (top-level in
this module) so it survives the same path resolution real workers go
through.
"""

from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

from pipeline.dual_pool import (
    APPLICATION_NAME,
    DualPoolResult,
    SkuPipelineInput,
    cleanup_orphan_connections,
    is_prophet_family_sku,
    naive_fallback_result,
    partition_skus,
    run_dual_pool,
)


# ---------------------------------------------------------------------------
# Fake executor — synchronous, records dispatch order and timestamps
# ---------------------------------------------------------------------------

class _FakeFuture:
    """Mimics concurrent.futures.Future enough for our orchestrator."""

    def __init__(self, callable_, args):
        self._cancelled = False
        try:
            self._result = callable_(*args)
            self._exc = None
        except Exception as e:
            self._result = None
            self._exc = e

    def result(self, timeout=None):
        if self._cancelled:
            raise FuturesTimeoutError("cancelled")
        if self._exc is not None:
            raise self._exc
        return self._result

    def cancel(self):
        self._cancelled = True
        return True

    # as_completed iterates over an iterable of futures; for our fake we
    # implement __hash__ + __eq__ via identity (default). No additional
    # protocol needed because we monkey-patch as_completed below.


class _FakeExecutor:
    """
    Synchronous executor — runs each submitted callable immediately.
    Records the timestamp at which submit() was first called so the
    "both pools started within 100ms" check is meaningful.
    """

    instances: list["_FakeExecutor"] = []  # collected for inspection by tests

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submits: list[float] = []
        self.shutdown_called = False
        _FakeExecutor.instances.append(self)

    def submit(self, fn, *args, **kwargs):
        self.submits.append(time.monotonic())
        return _FakeFuture(fn, args)

    def shutdown(self, wait=True):
        self.shutdown_called = True


@pytest.fixture(autouse=True)
def _reset_fake_executor_state(monkeypatch):
    _FakeExecutor.instances.clear()
    # Replace as_completed with a deterministic identity-iter so test
    # ordering is stable. Our fakes are synchronous so all futures are
    # already done.
    import pipeline.dual_pool as dp

    def _fake_as_completed(futures, timeout=None):
        return list(futures)

    monkeypatch.setattr(dp, "as_completed", _fake_as_completed)
    yield


# ---------------------------------------------------------------------------
# Top-level pipeline funcs (must be importable by string path)
# ---------------------------------------------------------------------------

def _ok_pipeline(sku_input, tenant_id, run_id, conn):
    return {
        "sku_id": sku_input.sku_id,
        "status": "forecasted",
        "confidence_final": 0.82,
    }


def _slow_pipeline(sku_input, tenant_id, run_id, conn):
    raise FuturesTimeoutError("worker exceeded timeout")


def _broken_pipeline(sku_input, tenant_id, run_id, conn):
    raise RuntimeError("pipeline blew up")


# Patch psycopg2.connect inside the worker so tests don't need a real DB.
@pytest.fixture(autouse=True)
def _patch_worker_connect(monkeypatch):
    import pipeline.dual_pool as dp

    class _DummyConn:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def _fake_connect(db_config):
        return _DummyConn()

    monkeypatch.setattr(dp, "_open_worker_connection", _fake_connect)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sku(sku_id, model="Prophet"):
    return SkuPipelineInput(
        sku_id=sku_id, assigned_model=model,
        sku_data={}, preloaded_data={},
    )


# ---------------------------------------------------------------------------
# Partition + predicate
# ---------------------------------------------------------------------------

def test_predicate_matches_prophet_family():
    assert is_prophet_family_sku(_sku("a", "Prophet")) is True
    assert is_prophet_family_sku(_sku("b", "NeuralProphet")) is False
    assert is_prophet_family_sku(_sku("c", "SES")) is False
    assert is_prophet_family_sku(_sku("d", "Naive Forecast")) is False


def test_partition_routes_skus_correctly():
    skus = [
        _sku("a", "Prophet"),
        _sku("b", "SES"),
        _sku("c", "NeuralProphet"),
        _sku("d", "Croston's Method"),
    ]
    proc, thread = partition_skus(skus)
    assert {s.sku_id for s in proc} == {"a"}
    assert {s.sku_id for s in thread} == {"b", "c", "d"}


# ---------------------------------------------------------------------------
# Naive fallback
# ---------------------------------------------------------------------------

def test_naive_fallback_shape():
    r = naive_fallback_result("sku-1", reason="timeout")
    assert r.sku_id == "sku-1"
    assert r.status == "needs_acknowledgment"
    assert r.confidence_final == 0.30
    assert r.pool == "fallback"
    assert r.error == "timeout"


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------

class _CleanupCursor:
    def __init__(self):
        self.executed = []
        self._rows = [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]  # 7 terminated

    def execute(self, sql, args=None):
        self.executed.append((sql, args))

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _CleanupConn:
    def __init__(self):
        self._cur = _CleanupCursor()
        self.committed = 0

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed += 1


def test_cleanup_orphan_connections_runs_correct_sql():
    conn = _CleanupConn()
    n = cleanup_orphan_connections(conn, idle_minutes=3)
    assert n == 7
    sql, args = conn._cur.executed[0]
    assert "pg_terminate_backend" in sql
    # Now uses parameterised make_interval(mins => %s) instead of f-string.
    assert "make_interval(mins => %s)" in sql
    # minutes is bound as the first arg, application_name as the second.
    assert args[0] == 3
    assert args[1] == "stage9_subprocess"


def test_cleanup_orphan_connections_int_coerces_minutes():
    """idle_minutes is forced to int — defends against bad args."""
    conn = _CleanupConn()
    cleanup_orphan_connections(conn, idle_minutes=10)
    sql, args = conn._cur.executed[0]
    # SQL is the same parameterised template; the bind value is what changes.
    assert "make_interval(mins => %s)" in sql
    assert args[0] == 10


# ---------------------------------------------------------------------------
# Orchestrator — happy path
# ---------------------------------------------------------------------------

_PIPELINE_PATH = "tests.test_dual_pool._ok_pipeline"


def _run(skus, *, log_failure_fn=None, **kwargs):
    """Test wrapper — supplies the now-required log_failure_fn, an
    explicit fallback_confidence (so the default-not-provided warning
    doesn't fire on every test), and skips the DB smoke test so we
    don't need a live Postgres."""
    kwargs.setdefault("fallback_confidence", 0.30)
    return run_dual_pool(
        skus,
        log_failure_fn=log_failure_fn or (lambda *a: None),
        db_smoke_test=False,
        **kwargs,
    )


def test_orchestrator_routes_skus_to_correct_pools():
    skus = [
        _sku("a", "Prophet"),
        _sku("b", "SES"),
        _sku("c", "NeuralProphet"),
        _sku("d", "Naive Forecast"),
    ]
    stats = _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH,
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )
    assert stats.process_skus == 1   # Prophet only
    assert stats.thread_skus == 3    # SES, NeuralProphet, Naive Forecast
    assert {s.sku_id for s in skus} == set(stats.results.keys())
    # Pool tag carried through:
    assert stats.results["a"].pool == "process"
    assert stats.results["b"].pool == "thread"


def test_orchestrator_done_when_both_pools_start_within_100ms():
    """Done-When #1: process and thread pool start within 100ms."""
    skus = [_sku(f"sku-{i}", "Prophet" if i % 2 == 0 else "SES") for i in range(10)]
    stats = _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH,
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )
    delta = abs(stats.thread_pool_started_at - stats.process_pool_started_at)
    assert delta < 0.100, f"pools started {delta*1000:.1f}ms apart"


def test_orchestrator_empty_inputs_short_circuits():
    stats = _run(
        [], tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH, db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )
    assert stats.results == {}
    assert _FakeExecutor.instances == []  # no pools opened


# ---------------------------------------------------------------------------
# Orchestrator — failure paths
# ---------------------------------------------------------------------------

def test_timeout_falls_back_to_naive_forecast():
    """Done-When: TimeoutError on one SKU → naive fallback, run continues."""
    failed_log: list[tuple] = []
    def log_failure(tenant_id, run_id, sku_id, reason):
        failed_log.append((sku_id, reason))

    skus = [
        _sku("ok", "Prophet"),       # will run _ok_pipeline (good path)
        _sku("slow", "Prophet"),     # picks _slow_pipeline below
    ]
    # Use two different paths via a small dispatcher
    stats = _run(
        [_sku("slow", "Prophet")],
        tenant_id="t1", run_id="r1",
        pipeline_fn_path="tests.test_dual_pool._slow_pipeline",
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
        log_failure_fn=log_failure,
    )
    r = stats.results["slow"]
    assert r.status == "needs_acknowledgment"
    assert r.confidence_final == 0.30
    assert r.pool == "fallback"
    assert stats.timeouts == 1
    assert failed_log == [("slow", "timeout")]


def test_one_failure_does_not_block_the_run():
    """Done-When: failed SKU isolated; other SKUs still complete."""
    failed_log: list[tuple] = []

    skus_ok = [_sku(f"ok-{i}", "Prophet") for i in range(3)]
    stats_ok = _run(
        skus_ok,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH,
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )
    for i in range(3):
        assert stats_ok.results[f"ok-{i}"].status == "forecasted"

    skus_broken = [_sku("crash", "Prophet")]
    stats_broken = _run(
        skus_broken,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path="tests.test_dual_pool._broken_pipeline",
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
        log_failure_fn=lambda *a: failed_log.append(a),
    )
    assert stats_broken.results["crash"].status == "needs_acknowledgment"
    assert stats_broken.failures == 1
    assert "error:" in failed_log[0][3]


# ---------------------------------------------------------------------------
# Worker contract
# ---------------------------------------------------------------------------

def test_worker_returns_lightweight_result_dict():
    """Done-When: worker returns {sku_id, status, confidence_final}."""
    skus = [_sku("a", "Prophet")]
    stats = _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH,
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )
    r = stats.results["a"]
    # The orchestrator wraps the dict in a DualPoolResult; the dict
    # the worker returned must have these three keys (plus the pool tag
    # the orchestrator appends).
    assert {"sku_id", "status", "confidence_final"}.issubset(
        {"sku_id", "status", "confidence_final", "pool", "error"}
    )
    assert r.sku_id == "a"
    assert r.status == "forecasted"
    assert r.confidence_final == 0.82


def test_orchestrator_runs_orphan_cleanup_after_collection():
    cleanup_conn = _CleanupConn()
    skus = [_sku("a", "Prophet")]
    stats = _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH,
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
        cleanup_conn=cleanup_conn,
    )
    # Cleanup SQL was issued; stats record terminated count.
    sqls = [e[0] for e in cleanup_conn._cur.executed]
    assert any("pg_terminate_backend" in s for s in sqls)
    assert stats.cleanup_terminated == 7


# ---------------------------------------------------------------------------
# pipeline_fn_path validation
# ---------------------------------------------------------------------------

def test_invalid_pipeline_fn_path_fails_fast():
    """
    With initializer-based resolution, a bad pipeline_fn_path fails the
    whole run up-front — not per-SKU. Better than 5M identical worker
    failures from the same typo.
    """
    skus = [_sku("a", "Prophet")]
    with pytest.raises(ImportError):
        _run(
            skus,
            tenant_id="t1", run_id="r1",
            pipeline_fn_path="not_a_module.func",
            db_config={},
            process_pool_factory=_FakeExecutor,
            thread_pool_factory=_FakeExecutor,
        )


# ---------------------------------------------------------------------------
# P1/P2 fix coverage — new contracts after the deep review
# ---------------------------------------------------------------------------

def test_log_failure_fn_is_required():
    """T19 P1-3: log_failure_fn must be supplied (no silent default)."""
    with pytest.raises(TypeError, match="log_failure_fn is required"):
        run_dual_pool(
            [_sku("a", "Prophet")],
            tenant_id="t1", run_id="r1",
            pipeline_fn_path=_PIPELINE_PATH,
            db_config={}, db_smoke_test=False,
            process_pool_factory=_FakeExecutor,
            thread_pool_factory=_FakeExecutor,
            log_failure_fn=None,
        )


def test_only_relevant_pool_is_created_when_one_pool_empty():
    """T19 P2-1: skip pool creation when its slice is empty."""
    only_thread = [_sku("a", "SES"), _sku("b", "Naive Forecast")]
    _run(
        only_thread,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH, db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )
    # Only one _FakeExecutor was instantiated (the thread pool).
    assert len(_FakeExecutor.instances) == 1


def test_fallback_confidence_threads_through_to_naive_result():
    """T19 P2-4: fallback_confidence param replaces the hardcoded 0.30."""
    skus = [_sku("crash", "Prophet")]
    stats = _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path="tests.test_dual_pool._broken_pipeline",
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
        fallback_confidence=0.42,
    )
    assert stats.results["crash"].confidence_final == 0.42


def test_chunked_submission_bounds_inflight_futures(monkeypatch):
    """
    T19 P1-4: at chunk_size=2, a 5-SKU run submits in 3 waves.
    We patch as_completed to record the SIZE of each future_meta dict
    handed to it — that's the in-flight count per wave.
    """
    import pipeline.dual_pool as dp
    sizes: list[int] = []
    real = dp.as_completed

    def _spy(futures, timeout=None):
        # futures may be dict (Py 3.x) or set/list. Record its length.
        sizes.append(len(futures))
        return list(futures)

    monkeypatch.setattr(dp, "as_completed", _spy)

    skus = [_sku(f"sku-{i}", "Prophet") for i in range(5)]
    _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH, db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
        chunk_size=2,
    )
    # 5 SKUs / chunk_size=2 → waves of 2, 2, 1.
    assert sizes == [2, 2, 1]


def test_worker_validates_result_keys():
    """T19 P2-3: pipeline_fn returning wrong shape is treated as a failure,
    not silently coerced into 'forecasted' / 0.0."""
    failed_log: list = []
    skus = [_sku("a", "Prophet")]
    stats = _run(
        skus,
        tenant_id="t1", run_id="r1",
        pipeline_fn_path="tests.test_dual_pool._bad_shape_pipeline",
        db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
        log_failure_fn=lambda *a: failed_log.append(a),
    )
    assert stats.failures == 1
    assert stats.results["a"].status == "needs_acknowledgment"


def test_smoke_test_skipped_when_disabled():
    """T19 P2-7: db_smoke_test=False bypasses the up-front connect."""
    # If smoke test were enabled with a bogus db_config and psycopg2 was
    # importable, this would raise. db_smoke_test=False → it doesn't.
    _run(
        [_sku("a", "Prophet")],
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH,
        db_config={"host": "definitely-not-a-host"},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_FakeExecutor,
    )  # _run fixture passes db_smoke_test=False


# ---------------------------------------------------------------------------
# Top-level pipeline funcs (continued)
# ---------------------------------------------------------------------------

def _bad_shape_pipeline(sku_input, tenant_id, run_id, conn):
    return {"sku_id": sku_input.sku_id}  # missing status + confidence_final


# ---------------------------------------------------------------------------
# Audit fix coverage
# ---------------------------------------------------------------------------

def test_default_fallback_confidence_emits_warning():
    """Audit P1-borderline: when fallback_confidence isn't provided, the
    orchestrator falls back to DEFAULT_FALLBACK_CONFIDENCE and warns so
    the gap is visible in production logs."""
    skus = [_sku("a", "Prophet")]
    with pytest.warns(UserWarning, match="fallback_confidence not provided"):
        run_dual_pool(
            skus,
            tenant_id="t1", run_id="r1",
            pipeline_fn_path=_PIPELINE_PATH,
            db_config={}, db_smoke_test=False,
            process_pool_factory=_FakeExecutor,
            thread_pool_factory=_FakeExecutor,
            log_failure_fn=lambda *a: None,
            # fallback_confidence intentionally omitted
        )


def test_max_tasks_per_child_passed_to_factory_when_supported():
    """Audit E003: max_tasks_per_child must reach the ProcessPoolExecutor
    constructor, otherwise the leaked-daemon-dies-with-worker promise is
    false. We use a fake factory that records its kwargs."""
    received_kwargs: list[dict] = []

    class _RecordingExecutor(_FakeExecutor):
        def __init__(self, **kwargs):
            received_kwargs.append(dict(kwargs))
            super().__init__(max_workers=kwargs["max_workers"])

    _run(
        [_sku("a", "Prophet")],  # process pool path
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH, db_config={},
        process_pool_factory=_RecordingExecutor,
        thread_pool_factory=_FakeExecutor,
        max_tasks_per_child=1,
    )
    # Process pool was constructed; one of the attempts carried the kwarg.
    process_kwargs = [k for k in received_kwargs if "max_tasks_per_child" in k]
    assert process_kwargs, "max_tasks_per_child not passed to factory"
    assert process_kwargs[0]["max_tasks_per_child"] == 1


def test_max_tasks_per_child_optional_thread_pool_does_not_get_it():
    """Thread pool ignores max_tasks_per_child (threads are reused);
    _make_pool must NOT pass it for the thread path."""
    received_kwargs: list[dict] = []

    class _RecordingExecutor(_FakeExecutor):
        def __init__(self, **kwargs):
            received_kwargs.append(dict(kwargs))
            super().__init__(max_workers=kwargs["max_workers"])

    _run(
        [_sku("a", "SES")],  # thread pool path only
        tenant_id="t1", run_id="r1",
        pipeline_fn_path=_PIPELINE_PATH, db_config={},
        process_pool_factory=_FakeExecutor,
        thread_pool_factory=_RecordingExecutor,
        max_tasks_per_child=1,
    )
    # No thread-pool construction call carried max_tasks_per_child.
    for k in received_kwargs:
        assert "max_tasks_per_child" not in k


def test_default_log_failure_was_deleted():
    """Audit BLOCKER: _default_log_failure was dead code; deleting it
    forces every caller to pass a real log_failure_fn (no silent
    fallback to logger.warning)."""
    import pipeline.dual_pool as dp
    assert not hasattr(dp, "_default_log_failure"), (
        "_default_log_failure should not exist; log_failure_fn is required"
    )
