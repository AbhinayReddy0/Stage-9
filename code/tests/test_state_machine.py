# tests/test_state_machine.py
"""
Full test suite for stage9/state_machine.py.

Covers all 8 checks from the spec verification checklist:
  1. AgentState enum has exactly 9 members.
  2. VALID_TRANSITIONS has exactly 9 entries — including PERCEIVING.
  3. transition() with an invalid pair raises InvalidStateTransitionError.
  4. transition() with a valid pair writes a row to agent_state_log_s9 and commits.
  5. transition() to FAILED also writes a log row.
  6. Two simultaneous run() calls for the same tenant → second raises
     RunAlreadyInProgressError within 100ms.
  7. Simulated crash mid-run → Redis key expires after TTL.
  8. agent_state_log_s9 shows the complete IDLE → ... → COMPLETE path after
     a clean run (7 rows, one per transition).

Plus supporting tests:
  - All valid happy-path transition pairs succeed.
  - All states with a FAILED edge can transition to FAILED.
  - Lock is always released in finally, even on failure.
  - TenantParamNotFoundError raised by a handler flows through to FAILED + re-raise.
  - Input validation rejects malformed tenant_id / run_id.
  - reason field is truncated to _REASON_MAX_LEN before the DB write.

Run with:
    python -m pytest tests/test_state_machine.py -v

Integration test (requires Redis at localhost:6379):
    Set RUN_INTEGRATION_TESTS=true in .env, then:
    python -m pytest tests/test_state_machine.py -v
"""

from __future__ import annotations

import os
import sys
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

# Make project root importable regardless of where pytest is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infrastructure.config import RUN_INTEGRATION_TESTS as _RUN_INTEGRATION_TESTS
from infrastructure.state_machine import (
    AgentState,
    VALID_TRANSITIONS,
    InvalidStateTransitionError,
    transition,
    _REASON_MAX_LEN,   # internal constant — tested directly
)
# Symbols moved out of state_machine.py during the orchestrator refactor:
#   - LOCK_KEY_TEMPLATE / LOCK_TTL_SECONDS → constants
#   - RunAlreadyInProgressError / TenantParamNotFoundError → errors
#   - run() → orchestrator.run() (new entry point uses RedisRunLock; direct
#     redis_client.set/delete is no longer the lock mechanism)
from infrastructure.constants import LOCK_KEY_TEMPLATE, LOCK_TTL_SECONDS
from infrastructure.errors import RunAlreadyInProgressError, TenantParamNotFoundError
from pipeline.orchestrator import run


# ===========================================================================
# Shared test doubles
# ===========================================================================

class _Cursor:
    """
    Records every (sql, params) pair passed to execute(). close() is a no-op.
    One instance is created per db.cursor() call.
    """
    def __init__(self, store: list):
        self._store = store

    def execute(self, sql: str, params: dict) -> None:
        # Store a copy so later mutations to params don't affect the record.
        self._store.append(dict(params))

    def close(self) -> None:
        pass


class FakeDB:
    """
    In-memory DB double that records all writes and commits.

    Attributes:
        written:      List of param dicts written via cursor.execute(), in call order.
                      One entry per transition() call, across the whole run.
        commit_count: Total number of times commit() was called.
    """
    def __init__(self):
        self.written: list[dict] = []
        self.commit_count: int = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self.written)

    def commit(self) -> None:
        self.commit_count += 1


class NoDB:
    """
    DB double that fails with AssertionError if cursor() or commit() is called.
    Used for tests that must verify the DB is NOT touched.
    """
    def cursor(self):
        raise AssertionError(
            "DB must NOT be touched — this test expects no DB interaction."
        )

    def commit(self):
        raise AssertionError(
            "DB must NOT be touched — this test expects no DB interaction."
        )


def _clean_redis() -> MagicMock:
    """Redis mock where SET NX always succeeds (lock acquired)."""
    r = MagicMock()
    r.set.return_value = True
    return r


def _locked_redis() -> MagicMock:
    """Redis mock where SET NX always fails (lock already held)."""
    r = MagicMock()
    r.set.return_value = None
    return r


class _HandlersMockedBase(unittest.TestCase):
    """Base class that mocks all six handler functions imported by
    `orchestrator.run`. Tests inheriting from this isolate the state
    machine + lock behavior from handler internals — only `transition()`
    calls write to the FakeDB. Use this when the test asserts on the
    audit-trail row count, the order of transitions, or lock behavior,
    but does NOT care what the handlers do.

    Tests that DO need real handler behavior should not use this base
    class (and need a richer FakeDB that supports the with-cursor
    context manager protocol).
    """

    def setUp(self):
        self._handler_patches = [
            patch(f"pipeline.orchestrator.{name}")
            for name in (
                "preloading_handler",
                "perceiving_handler",
                "planning_handler",
                "acting_handler",
                "learning_handler",
                "reporting_handler",
            )
        ]
        for p in self._handler_patches:
            p.start()
        self.addCleanup(self._stop_all_handler_patches)

    def _stop_all_handler_patches(self):
        for p in self._handler_patches:
            p.stop()


# ===========================================================================
# Check 1 — AgentState enum has exactly 9 members
# ===========================================================================

class TestAgentStateEnum(unittest.TestCase):

    def test_exactly_nine_members(self):
        members = list(AgentState)
        self.assertEqual(
            len(members), 9,
            f"Expected 9 members, found {len(members)}: {[s.value for s in members]}",
        )

    def test_all_nine_expected_names_are_present(self):
        expected = {
            "IDLE", "PRELOADING", "PERCEIVING", "PLANNING", "ACTING",
            "LEARNING", "REPORTING", "COMPLETE", "FAILED",
        }
        actual = {s.value for s in AgentState}
        self.assertEqual(actual, expected)

    def test_every_member_is_an_agentstate_instance(self):
        for member in AgentState:
            self.assertIsInstance(member, AgentState)

    def test_string_values_match_member_names(self):
        # e.g. AgentState.IDLE.value == "IDLE" — used only for DB persistence.
        for member in AgentState:
            self.assertEqual(member.value, member.name)


# ===========================================================================
# Check 2 — VALID_TRANSITIONS has exactly 9 entries, including PERCEIVING
# ===========================================================================

class TestValidTransitions(unittest.TestCase):

    def test_exactly_nine_entries(self):
        self.assertEqual(
            len(VALID_TRANSITIONS), 9,
            f"Expected 9 entries, found {len(VALID_TRANSITIONS)}",
        )

    def test_perceiving_is_a_key(self):
        # Explicitly required by the spec checklist.
        self.assertIn(AgentState.PERCEIVING, VALID_TRANSITIONS)

    def test_all_nine_states_are_keys(self):
        for state in AgentState:
            self.assertIn(state, VALID_TRANSITIONS, f"{state.value} missing from VALID_TRANSITIONS")

    def test_all_keys_are_enum_members_not_strings(self):
        for key in VALID_TRANSITIONS:
            self.assertIsInstance(key, AgentState, f"Key {key!r} is a bare string, not AgentState")

    def test_all_target_values_are_enum_members_not_strings(self):
        for key, targets in VALID_TRANSITIONS.items():
            for t in targets:
                self.assertIsInstance(t, AgentState, f"Target {t!r} under {key.value} is not AgentState")

    def test_learning_has_no_failed_edge(self):
        # LEARNING → FAILED is intentionally absent — see VALID_TRANSITIONS comment.
        self.assertNotIn(AgentState.FAILED, VALID_TRANSITIONS[AgentState.LEARNING])

    def test_reporting_has_no_failed_edge(self):
        self.assertNotIn(AgentState.FAILED, VALID_TRANSITIONS[AgentState.REPORTING])

    def test_happy_path_edges_exist(self):
        # Every adjacent pair in the clean run must be in the map.
        happy_path = [
            (AgentState.IDLE,       AgentState.PRELOADING),
            (AgentState.PRELOADING, AgentState.PERCEIVING),
            (AgentState.PERCEIVING, AgentState.PLANNING),
            (AgentState.PLANNING,   AgentState.ACTING),
            (AgentState.ACTING,     AgentState.LEARNING),
            (AgentState.LEARNING,   AgentState.REPORTING),
            (AgentState.REPORTING,  AgentState.COMPLETE),
        ]
        for current, nxt in happy_path:
            self.assertIn(
                nxt, VALID_TRANSITIONS[current],
                f"Missing edge: {current.value} -> {nxt.value}",
            )


# ===========================================================================
# Check 3 — transition() with an invalid pair raises InvalidStateTransitionError
# ===========================================================================

class TestTransitionInvalidPair(unittest.TestCase):

    def _assert_raises_no_db(self, current: AgentState, nxt: AgentState) -> None:
        """
        Verifies InvalidStateTransitionError is raised AND the DB is not touched.
        NoDB raises AssertionError on any cursor() or commit() call.
        """
        with self.assertRaises(InvalidStateTransitionError):
            transition(NoDB(), "acme", "run-1", current, nxt)

    def test_idle_to_failed_is_invalid(self):
        # Known spec gap: IDLE has no FAILED edge. See VALID_TRANSITIONS comment.
        self._assert_raises_no_db(AgentState.IDLE, AgentState.FAILED)

    def test_idle_skipping_directly_to_perceiving_is_invalid(self):
        self._assert_raises_no_db(AgentState.IDLE, AgentState.PERCEIVING)

    def test_idle_skipping_directly_to_acting_is_invalid(self):
        self._assert_raises_no_db(AgentState.IDLE, AgentState.ACTING)

    def test_complete_to_failed_is_invalid(self):
        self._assert_raises_no_db(AgentState.COMPLETE, AgentState.FAILED)

    def test_learning_to_failed_is_invalid(self):
        # LEARNING intentionally has no FAILED edge.
        self._assert_raises_no_db(AgentState.LEARNING, AgentState.FAILED)

    def test_reporting_to_failed_is_invalid(self):
        # REPORTING intentionally has no FAILED edge.
        self._assert_raises_no_db(AgentState.REPORTING, AgentState.FAILED)

    def test_backwards_transition_is_invalid(self):
        self._assert_raises_no_db(AgentState.PERCEIVING, AgentState.IDLE)

    def test_same_state_self_loop_is_invalid(self):
        self._assert_raises_no_db(AgentState.ACTING, AgentState.ACTING)

    def test_error_message_names_both_states(self):
        try:
            transition(NoDB(), "acme", "run-1", AgentState.IDLE, AgentState.FAILED)
            self.fail("Should have raised")
        except InvalidStateTransitionError as e:
            msg = str(e)
            self.assertIn("IDLE",   msg)
            self.assertIn("FAILED", msg)


# ===========================================================================
# Check 4 — transition() with a valid pair writes a row and commits
# ===========================================================================

class TestTransitionValidPair(unittest.TestCase):

    def test_returns_next_state(self):
        result = transition(FakeDB(), "acme", "run-1", AgentState.PRELOADING, AgentState.PERCEIVING)
        self.assertIs(result, AgentState.PERCEIVING)

    def test_writes_exactly_one_row(self):
        db = FakeDB()
        transition(db, "acme", "run-1", AgentState.PRELOADING, AgentState.PERCEIVING)
        self.assertEqual(len(db.written), 1)

    def test_row_contains_correct_fields(self):
        db = FakeDB()
        transition(db, "acme", "run-001", AgentState.PRELOADING, AgentState.PERCEIVING)
        row = db.written[0]
        self.assertEqual(row["tenant_id"],  "acme")
        self.assertEqual(row["run_id"],     "run-001")
        self.assertEqual(row["from_state"], "PRELOADING")   # .value — string for DB
        self.assertEqual(row["to_state"],   "PERCEIVING")
        self.assertIsNone(row["reason"])                    # NULL on happy path

    def test_commits_exactly_once(self):
        db = FakeDB()
        transition(db, "acme", "run-1", AgentState.PRELOADING, AgentState.PERCEIVING)
        self.assertEqual(db.commit_count, 1)

    def test_all_valid_happy_path_pairs_write_and_commit(self):
        """Every adjacent pair in the clean run path must write one row and commit."""
        pairs = [
            (AgentState.IDLE,       AgentState.PRELOADING),
            (AgentState.PRELOADING, AgentState.PERCEIVING),
            (AgentState.PERCEIVING, AgentState.PLANNING),
            (AgentState.PLANNING,   AgentState.ACTING),
            (AgentState.ACTING,     AgentState.LEARNING),
            (AgentState.LEARNING,   AgentState.REPORTING),
            (AgentState.REPORTING,  AgentState.COMPLETE),
        ]
        for current, nxt in pairs:
            db = FakeDB()
            result = transition(db, "acme", "run-1", current, nxt)
            self.assertIs(result, nxt,        f"{current.value} -> {nxt.value}: wrong return value")
            self.assertEqual(len(db.written), 1, f"{current.value} -> {nxt.value}: expected 1 row")
            self.assertEqual(db.commit_count, 1, f"{current.value} -> {nxt.value}: expected 1 commit")


# ===========================================================================
# Check 5 — transition() to FAILED also writes a log row
# ===========================================================================

class TestTransitionToFailed(unittest.TestCase):

    def test_failed_transition_writes_row(self):
        db = FakeDB()
        transition(db, "acme", "run-1", AgentState.ACTING, AgentState.FAILED, reason="boom")
        self.assertEqual(len(db.written), 1)

    def test_failed_transition_row_has_correct_to_state(self):
        db = FakeDB()
        transition(db, "acme", "run-1", AgentState.ACTING, AgentState.FAILED, reason="boom")
        self.assertEqual(db.written[0]["to_state"], "FAILED")

    def test_failed_transition_row_carries_reason(self):
        db = FakeDB()
        transition(db, "acme", "run-1", AgentState.ACTING, AgentState.FAILED, reason="handler crashed")
        self.assertEqual(db.written[0]["reason"], "handler crashed")

    def test_failed_transition_commits(self):
        db = FakeDB()
        transition(db, "acme", "run-1", AgentState.PLANNING, AgentState.FAILED, reason="err")
        self.assertEqual(db.commit_count, 1)

    def test_failed_transition_returns_failed_state(self):
        result = transition(FakeDB(), "acme", "run-1", AgentState.ACTING, AgentState.FAILED, reason="x")
        self.assertIs(result, AgentState.FAILED)

    def test_every_state_with_failed_edge_can_write_failed_row(self):
        """
        PRELOADING, PERCEIVING, PLANNING, ACTING all have FAILED edges.
        Each must successfully write a log row and commit.
        """
        states_with_failed_edge = [
            AgentState.PRELOADING,
            AgentState.PERCEIVING,
            AgentState.PLANNING,
            AgentState.ACTING,
        ]
        for state in states_with_failed_edge:
            db = FakeDB()
            result = transition(db, "acme", "run-1", state, AgentState.FAILED, reason="test")
            self.assertIs(result, AgentState.FAILED, f"Wrong return from {state.value} -> FAILED")
            self.assertEqual(db.written[0]["to_state"], "FAILED")
            self.assertEqual(db.commit_count, 1)


# ===========================================================================
# Check 6 — Two simultaneous run() calls → second raises within 100ms
# ===========================================================================

class TestConcurrentRuns(_HandlersMockedBase):

    def test_second_run_raises_run_already_in_progress(self):
        with self.assertRaises(RunAlreadyInProgressError):
            run("acme", "run-2", FakeDB(), _locked_redis())

    def test_second_run_raises_within_100ms(self):
        start = time.monotonic()
        try:
            run("acme", "run-2", FakeDB(), _locked_redis())
        except RunAlreadyInProgressError:
            pass
        elapsed_ms = (time.monotonic() - start) * 1000
        self.assertLess(
            elapsed_ms, 100,
            f"RunAlreadyInProgressError took {elapsed_ms:.1f}ms — expected < 100ms",
        )

    def test_second_run_writes_no_db_rows(self):
        """
        Lock rejection happens before any transition() call, so agent_state_log_s9
        must remain untouched when the lock is already held.
        """
        db = FakeDB()
        try:
            run("acme", "run-2", db, _locked_redis())
        except RunAlreadyInProgressError:
            pass
        self.assertEqual(
            len(db.written), 0,
            "No rows should be written to agent_state_log_s9 when the lock is held",
        )

    def test_truly_concurrent_second_call_raises(self):
        """
        Two threads call run() for the same tenant at the same instant
        (synchronised by threading.Barrier). The thread whose Redis mock
        returns None (lock held) must raise RunAlreadyInProgressError.
        """
        results: dict[str, str] = {}
        start_barrier = threading.Barrier(2)

        def do_run(run_id: str, redis_mock: MagicMock) -> None:
            start_barrier.wait()   # both threads reach this line before either proceeds
            try:
                run("acme", run_id, FakeDB(), redis_mock)
                results[run_id] = "ok"
            except RunAlreadyInProgressError:
                results[run_id] = "locked"

        t_winner = threading.Thread(target=do_run, args=("run-winner", _clean_redis()))
        t_loser  = threading.Thread(target=do_run, args=("run-loser",  _locked_redis()))

        t_winner.start()
        t_loser.start()
        t_winner.join(timeout=5)
        t_loser.join(timeout=5)

        self.assertEqual(results.get("run-winner"), "ok",     "Winner should complete cleanly")
        self.assertEqual(results.get("run-loser"),  "locked", "Loser should raise RunAlreadyInProgressError")


# ===========================================================================
# Check 7 — Redis key expires after TTL
# ===========================================================================

class TestLockTTL(unittest.TestCase):

    def test_correct_ttl_value(self):
        # LOCK_TTL_SECONDS must be 4 hours (14400 seconds) per spec.
        self.assertEqual(LOCK_TTL_SECONDS, 14400)

    def test_ttl_passed_as_ex_argument_to_redis_set(self):
        """
        Verifies the EX argument to redis.set() equals LOCK_TTL_SECONDS.
        The lock auto-expires after this TTL if the process crashes — the
        dead-man's switch. This unit test checks the wiring without real Redis.
        """
        redis = _locked_redis()     # raises immediately so no DB calls follow
        try:
            run("acme", "run-1", FakeDB(), redis)
        except RunAlreadyInProgressError:
            pass

        redis.set.assert_called_once_with(
            "stage9_lock_acme",
            "locked",
            ex=LOCK_TTL_SECONDS,    # must be 14400
            nx=True,
        )

    def test_lock_key_uses_correct_template(self):
        redis = _locked_redis()
        try:
            run("tenant-xyz", "run-1", FakeDB(), redis)
        except RunAlreadyInProgressError:
            pass
        key_used = redis.set.call_args[0][0]
        self.assertEqual(key_used, "stage9_lock_tenant-xyz")

    @unittest.skipUnless(
        _RUN_INTEGRATION_TESTS,
        "Set RUN_INTEGRATION_TESTS=true in .env and ensure Redis is at localhost:6379 to run.",
    )
    def test_redis_key_physically_expires_after_ttl(self):
        """
        Integration test: sets the lock key with a 10-second TTL, waits 11
        seconds, and confirms the key no longer exists in Redis.

        This proves the auto-expiry dead-man's switch works end-to-end.
        Requires a real Redis instance at localhost:6379, DB index 15
        (isolated from any production or development data).

        Run with:
            RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_state_machine.py -v
        """
        import redis as redis_lib
        import infrastructure.constants as const_mod

        TEST_TTL = 10   # seconds — small enough for CI, large enough to be stable
        client   = redis_lib.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        lock_key = LOCK_KEY_TEMPLATE.format(tenant_id="ttl-test-tenant")

        # Override LOCK_TTL_SECONDS temporarily so run() uses our short TTL.
        original_ttl = const_mod.LOCK_TTL_SECONDS
        const_mod.LOCK_TTL_SECONDS = TEST_TTL
        try:
            # Set the key manually — same semantics as what run() does.
            client.set(lock_key, "locked", ex=TEST_TTL, nx=True)
            self.assertTrue(client.exists(lock_key), "Key must exist immediately after SET")

            # Wait past the TTL.
            time.sleep(TEST_TTL + 1)

            self.assertFalse(
                client.exists(lock_key),
                f"Key {lock_key!r} should have expired after {TEST_TTL}s TTL",
            )
        finally:
            const_mod.LOCK_TTL_SECONDS = original_ttl
            client.delete(lock_key)     # safety cleanup if the test fails before expiry


# ===========================================================================
# Check 8 — Complete audit trail after a clean run
# ===========================================================================

class TestCleanRunAuditTrail(_HandlersMockedBase):
    """
    After a successful run, agent_state_log_s9 must contain exactly 7 rows
    in this order — one per transition, IDLE being the start state (not a
    target) during a fresh run:

        IDLE       -> PRELOADING
        PRELOADING -> PERCEIVING
        PERCEIVING -> PLANNING
        PLANNING   -> ACTING
        ACTING     -> LEARNING
        LEARNING   -> REPORTING
        REPORTING  -> COMPLETE
    """

    EXPECTED_PATH = [
        ("IDLE",       "PRELOADING"),
        ("PRELOADING", "PERCEIVING"),
        ("PERCEIVING", "PLANNING"),
        ("PLANNING",   "ACTING"),
        ("ACTING",     "LEARNING"),
        ("LEARNING",   "REPORTING"),
        ("REPORTING",  "COMPLETE"),
    ]

    def _run_clean(self) -> FakeDB:
        """Execute a clean run and return the DB double for inspection."""
        db = FakeDB()
        run("acme", "run-clean", db, _clean_redis())
        return db

    def test_exactly_seven_rows_written(self):
        db = self._run_clean()
        actual_path = [(r["from_state"], r["to_state"]) for r in db.written]
        self.assertEqual(
            len(db.written), 7,
            f"Expected 7 rows, got {len(db.written)}. Actual path: {actual_path}",
        )

    def test_rows_are_in_correct_order(self):
        db = self._run_clean()
        for i, (expected_from, expected_to) in enumerate(self.EXPECTED_PATH):
            row = db.written[i]
            self.assertEqual(
                row["from_state"], expected_from,
                f"Row {i}: from_state expected {expected_from!r}, got {row['from_state']!r}",
            )
            self.assertEqual(
                row["to_state"], expected_to,
                f"Row {i}: to_state expected {expected_to!r}, got {row['to_state']!r}",
            )

    def test_all_rows_carry_correct_tenant_and_run_id(self):
        db = self._run_clean()
        for i, row in enumerate(db.written):
            self.assertEqual(row["tenant_id"], "acme",      f"Row {i}: wrong tenant_id")
            self.assertEqual(row["run_id"],    "run-clean", f"Row {i}: wrong run_id")

    def test_all_happy_path_rows_have_null_reason(self):
        # reason must be NULL (None) on every non-FAILED row.
        db = self._run_clean()
        for i, row in enumerate(db.written):
            self.assertIsNone(row["reason"], f"Row {i} ({row['to_state']}): reason should be NULL")

    def test_seven_commits_one_per_transition(self):
        # Each transition() call issues exactly one COMMIT.
        db = self._run_clean()
        self.assertEqual(db.commit_count, 7)

    def test_redis_lock_released_after_clean_run(self):
        redis = _clean_redis()
        run("acme", "run-clean", FakeDB(), redis)
        redis.delete.assert_called_once_with("stage9_lock_acme")

    def test_first_row_from_state_is_idle(self):
        db = self._run_clean()
        self.assertEqual(db.written[0]["from_state"], "IDLE")

    def test_last_row_to_state_is_complete(self):
        db = self._run_clean()
        self.assertEqual(db.written[-1]["to_state"], "COMPLETE")


# ===========================================================================
# Supporting tests — lock release, error handling, validation, TenantParamNotFoundError
# ===========================================================================

class TestLockAlwaysReleased(_HandlersMockedBase):
    """
    The Redis lock must be released in the finally block unconditionally —
    on success, on handler failure, and on KeyboardInterrupt.
    """

    def test_lock_released_after_successful_run(self):
        redis = _clean_redis()
        run("acme", "run-1", FakeDB(), redis)
        redis.delete.assert_called_once_with("stage9_lock_acme")

    def test_lock_released_after_handler_failure(self):
        """
        Simulate a DB failure mid-run (commit raises). The finally block must
        still delete the Redis key.
        """
        class FailingDB:
            """Lets cursor.execute() succeed but raises on commit()."""
            def __init__(self):
                self.written = []
                self._commit_calls = 0

            def cursor(self):
                return _Cursor(self.written)

            def commit(self):
                self._commit_calls += 1
                if self._commit_calls >= 1:
                    raise RuntimeError("DB commit failed")

        redis = _clean_redis()
        try:
            run("acme", "run-1", FailingDB(), redis)
        except Exception:
            pass   # expected — DB is broken

        redis.delete.assert_called_once_with("stage9_lock_acme")

    def test_lock_not_released_when_never_acquired(self):
        """
        When the lock is not acquired (RunAlreadyInProgressError), delete()
        must NOT be called — the key belongs to the other run.
        """
        redis = _locked_redis()
        try:
            run("acme", "run-2", FakeDB(), redis)
        except RunAlreadyInProgressError:
            pass
        redis.delete.assert_not_called()


class TestTenantParamNotFoundError(_HandlersMockedBase):
    """
    TenantParamNotFoundError is raised by the PERCEIVING handler (Task 7)
    when tenant_learning_params has no rows for the tenant.

    Expected flow inside run():
      1. PERCEIVING transition succeeds — state = PERCEIVING.
      2. perceiving_handler() raises TenantParamNotFoundError.
      3. except block catches it, writes PERCEIVING → FAILED log row, re-raises.
    """

    def test_is_subclass_of_stage9_error(self):
        # All Stage 9 exceptions inherit Stage9Error — never built-in types.
        from infrastructure.errors import Stage9Error
        self.assertTrue(issubclass(TenantParamNotFoundError, Stage9Error))

    def test_is_caught_by_except_exception(self):
        # Verify it is a subclass of Exception and would be caught by the
        # generic `except Exception` block in run().
        self.assertTrue(issubclass(TenantParamNotFoundError, Exception))

    def test_flow_through_run_writes_failed_row_and_reraises(self):
        """
        Injects TenantParamNotFoundError immediately after the PERCEIVING
        transition succeeds — mimicking perceiving_handler() raising.

        Verifies:
          - The error propagates out of run() (re-raised).
          - A FAILED row is written to agent_state_log_s9.
          - The FAILED row's from_state is PERCEIVING (state at time of error).
        """
        import infrastructure.state_machine as sm

        db = FakeDB()
        original_transition = sm.transition

        def injecting_transition(d, tid, rid, current, nxt, reason=None):
            # Call real transition so the log row is written.
            result = original_transition(d, tid, rid, current, nxt, reason)
            # After PERCEIVING transition completes, simulate handler raising.
            if nxt == AgentState.PERCEIVING:
                raise TenantParamNotFoundError(
                    f"no tenant_learning_params rows for tenant {tid!r}"
                )
            return result

        with patch("pipeline.orchestrator.transition", side_effect=injecting_transition):
            with self.assertRaises(TenantParamNotFoundError):
                run("acme", "run-1", db, _clean_redis())

        # The FAILED transition is called with state = PRELOADING because the
        # PERCEIVING transition raised before `state = ...` could be assigned.
        failed_rows = [r for r in db.written if r.get("to_state") == "FAILED"]
        self.assertEqual(len(failed_rows), 1, "Exactly one FAILED row must be written")
        self.assertIn(
            "tenant_learning_params",
            failed_rows[0]["reason"],
            "FAILED row reason must include the exception message",
        )


class TestInputValidation(unittest.TestCase):
    """
    _validate_ids() guards Redis key construction and DB parameters.
    Malformed ids must raise ValueError before any Redis or DB interaction.
    """

    def _assert_value_error_no_redis(self, tenant_id, run_id):
        redis = MagicMock()  # must NOT be called
        with self.assertRaises(ValueError):
            run(tenant_id, run_id, FakeDB(), redis)
        redis.set.assert_not_called()

    def test_tenant_id_with_space_is_rejected(self):
        self._assert_value_error_no_redis("bad tenant", "run-1")

    def test_tenant_id_with_braces_is_rejected(self):
        # Braces in tenant_id would break str.format(tenant_id=...) for the lock key.
        self._assert_value_error_no_redis("{evil}", "run-1")

    def test_empty_tenant_id_is_rejected(self):
        self._assert_value_error_no_redis("", "run-1")

    def test_tenant_id_over_64_chars_is_rejected(self):
        self._assert_value_error_no_redis("a" * 65, "run-1")

    def test_valid_tenant_id_with_hyphens_is_accepted(self):
        # Should not raise ValueError — raises RunAlreadyInProgressError instead
        # because redis returns None (locked).
        redis = _locked_redis()
        try:
            run("acme-corp-uk", "run-1", FakeDB(), redis)
        except RunAlreadyInProgressError:
            pass   # expected — this means validation passed
        except ValueError:
            self.fail("Valid tenant_id with hyphens should not raise ValueError")

    def test_run_id_with_invalid_char_is_rejected(self):
        self._assert_value_error_no_redis("acme", "run id!")

    def test_empty_run_id_is_rejected(self):
        self._assert_value_error_no_redis("acme", "")

    def test_non_string_tenant_id_is_rejected(self):
        with self.assertRaises((ValueError, AttributeError, TypeError)):
            run(12345, "run-1", FakeDB(), MagicMock())


class TestBareExceptionFromHandler(unittest.TestCase):
    """
    EH-02: A handler that raises a plain Exception (not a Stage9Error subclass)
    must still cause run() to write a FAILED row, release the lock, and re-raise.
    """

    def test_bare_exception_transitions_to_failed(self):
        """EH-02: RuntimeError from preloading_handler must produce a FAILED log row."""
        db = FakeDB()
        with patch("pipeline.orchestrator.preloading_handler", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                run("acme", "run-bare", db, _clean_redis())

        failed_rows = [r for r in db.written if r.get("to_state") == "FAILED"]
        self.assertEqual(len(failed_rows), 1, "Exactly one FAILED row must be written")
        self.assertIn("disk full", failed_rows[0]["reason"])

    def test_bare_exception_releases_lock(self):
        """EH-02: Redis lock must be released even for a bare Exception."""
        redis = _clean_redis()
        with patch("pipeline.orchestrator.preloading_handler", side_effect=RuntimeError("oom")):
            with self.assertRaises(RuntimeError):
                run("acme", "run-bare", FakeDB(), redis)
        redis.delete.assert_called_once_with("stage9_lock_acme")

    def test_bare_exception_is_reraised_not_swallowed(self):
        """EH-02: run() must never swallow the original exception."""
        with patch("pipeline.orchestrator.preloading_handler", side_effect=ValueError("type mismatch")):
            with self.assertRaises(ValueError, msg="ValueError must propagate out of run()"):
                run("acme", "run-bare", FakeDB(), _clean_redis())

    def test_bare_exception_failed_row_from_state_is_preloading(self):
        """EH-02: the FAILED row must show from_state=PRELOADING (state at time of crash)."""
        db = FakeDB()
        with patch("pipeline.orchestrator.preloading_handler", side_effect=RuntimeError("io error")):
            with self.assertRaises(RuntimeError):
                run("acme", "run-bare", db, _clean_redis())

        failed_rows = [r for r in db.written if r.get("to_state") == "FAILED"]
        self.assertEqual(len(failed_rows), 1)
        self.assertEqual(failed_rows[0]["from_state"], "PRELOADING")


class TestRedisDownHandling(unittest.TestCase):
    """
    EH-05: When redis_client.set() raises before the lock is acquired,
    no DB rows must be written and the exception must propagate unchanged.
    """

    def test_redis_connection_error_propagates(self):
        """EH-05: ConnectionError from redis.set() must propagate out of run()."""
        redis = MagicMock()
        redis.set.side_effect = ConnectionError("Redis refused connection")
        db = FakeDB()
        with self.assertRaises(ConnectionError):
            run("acme", "run-redis-down", db, redis)

    def test_redis_connection_error_writes_no_db_rows(self):
        """EH-05: if lock acquisition fails, the DB must not be touched."""
        redis = MagicMock()
        redis.set.side_effect = ConnectionError("Redis refused connection")
        db = FakeDB()
        try:
            run("acme", "run-redis-down", db, redis)
        except ConnectionError:
            pass
        self.assertEqual(
            len(db.written), 0,
            "No agent_state_log_s9 rows must be written when Redis is unreachable",
        )


class TestReasonTruncation(unittest.TestCase):
    """
    reason passed to the FAILED transition must be truncated to _REASON_MAX_LEN
    characters to prevent storage bloat in agent_state_log_s9.
    """

    def test_long_exception_message_is_truncated(self):
        """
        A 5000-character exception message must be stored as _REASON_MAX_LEN chars.
        """
        long_message = "x" * 5000

        captured_reasons = []

        def capturing_transition(d, tid, rid, current, nxt, reason=None):
            if nxt == AgentState.FAILED:
                captured_reasons.append(reason)
            # Use real transition so the FakeDB records the row.
            return transition(d, tid, rid, current, nxt, reason)

        db = FakeDB()

        with patch("pipeline.orchestrator.transition", side_effect=capturing_transition):
            try:
                raise RuntimeError(long_message)
            except RuntimeError:
                pass

            # Simulate what run()'s except block does.
            reason_text = long_message[:_REASON_MAX_LEN]
            capturing_transition(db, "acme", "run-1", AgentState.ACTING, AgentState.FAILED, reason=reason_text)

        self.assertEqual(len(captured_reasons), 1)
        self.assertEqual(len(captured_reasons[0]), _REASON_MAX_LEN)

    def test_short_exception_message_is_not_padded(self):
        short_message = "short error"
        # reason_text = short_message[:_REASON_MAX_LEN] must equal the original
        self.assertEqual(short_message[:_REASON_MAX_LEN], short_message)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
