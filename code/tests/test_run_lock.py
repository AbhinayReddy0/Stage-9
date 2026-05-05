"""
unit/orchestration/test_run_lock.py — RedisRunLock coverage with a mocked
Redis client. No real Redis at localhost:6379 is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import redis as redis_lib

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from infrastructure.run_lock import (
    RedisRunLock,
    LockAcquisitionError,
    _RELEASE_SCRIPT,
    LOCK_TTL_SECONDS,
)


@pytest.fixture
def fake_client():
    """Mock redis.Redis client whose method returns we control."""
    return MagicMock(spec=redis_lib.Redis)


@pytest.fixture
def patched(fake_client, monkeypatch):
    """Patch _get_client so RedisRunLock methods use our mock."""
    monkeypatch.setattr(
        "infrastructure.run_lock._get_client", lambda: fake_client
    )
    return fake_client


# ---------------------------------------------------------------------------
# _validate_tenant_id
# ---------------------------------------------------------------------------

class TestValidateTenantId:

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RedisRunLock()._validate_tenant_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RedisRunLock()._validate_tenant_id("   ")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RedisRunLock()._validate_tenant_id(None)

    def test_int_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RedisRunLock()._validate_tenant_id(42)


# ---------------------------------------------------------------------------
# _key
# ---------------------------------------------------------------------------

class TestKey:

    def test_key_includes_tenant_id(self):
        assert "abc-123" in RedisRunLock._key("abc-123")


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------

class TestAcquire:

    def test_returns_true_and_token_on_first_acquire(self, patched):
        patched.set.return_value = True
        ok, token = RedisRunLock().acquire("t1")
        assert ok is True
        assert token is not None
        # Verify SET ... NX EX <ttl> contract
        args, kwargs = patched.set.call_args
        assert kwargs["nx"] is True
        assert kwargs["ex"] == LOCK_TTL_SECONDS

    def test_returns_false_when_key_already_held(self, patched):
        patched.set.return_value = None
        ok, token = RedisRunLock().acquire("t1")
        assert ok is False
        assert token is None

    def test_redis_error_raises_lock_acquisition_error(self, patched):
        patched.set.side_effect = redis_lib.RedisError("connection refused")
        with pytest.raises(LockAcquisitionError, match="Could not acquire"):
            RedisRunLock().acquire("t1")

    def test_acquire_validates_tenant_id_first(self, patched):
        with pytest.raises(ValueError):
            RedisRunLock().acquire("")
        # Redis client was NOT touched — guard fired before SET
        patched.set.assert_not_called()


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

class TestRelease:

    def test_returns_true_when_release_lua_returns_one(self, patched):
        patched.eval.return_value = 1
        assert RedisRunLock().release("t1", "the-token") is True
        # Lua script invoked with the right key + token
        args = patched.eval.call_args.args
        assert args[0] == _RELEASE_SCRIPT
        assert args[1] == 1     # numkeys
        assert "t1" in args[2]  # key includes tenant
        assert args[3] == "the-token"

    def test_returns_false_when_release_lua_returns_zero(self, patched):
        """Token mismatch / TTL fired between acquire and release."""
        patched.eval.return_value = 0
        assert RedisRunLock().release("t1", "the-token") is False

    def test_redis_error_returns_false_does_not_raise(self, patched):
        """Release failure must NOT crash the pipeline's finally block —
        the TTL will eventually clean the key up."""
        patched.eval.side_effect = redis_lib.RedisError("down")
        assert RedisRunLock().release("t1", "the-token") is False

    def test_release_validates_tenant_id(self):
        with pytest.raises(ValueError):
            RedisRunLock().release("", "tok")


# ---------------------------------------------------------------------------
# is_locked
# ---------------------------------------------------------------------------

class TestIsLocked:

    def test_returns_true_when_key_exists(self, patched):
        patched.exists.return_value = 1
        assert RedisRunLock().is_locked("t1") is True

    def test_returns_false_when_key_absent(self, patched):
        patched.exists.return_value = 0
        assert RedisRunLock().is_locked("t1") is False

    def test_redis_error_returns_false(self, patched):
        """Health checks must NOT raise on infra issues."""
        patched.exists.side_effect = redis_lib.RedisError("oops")
        assert RedisRunLock().is_locked("t1") is False


# ---------------------------------------------------------------------------
# ttl
# ---------------------------------------------------------------------------

class TestTtl:

    def test_returns_value_from_redis_ttl(self, patched):
        patched.ttl.return_value = 42
        assert RedisRunLock().ttl("t1") == 42

    def test_no_key_returns_minus_two(self, patched):
        """Convention: -2 means the key doesn't exist."""
        patched.ttl.return_value = -2
        assert RedisRunLock().ttl("t1") == -2

    def test_redis_error_returns_minus_two(self, patched):
        patched.ttl.side_effect = redis_lib.RedisError("boom")
        assert RedisRunLock().ttl("t1") == -2


# ---------------------------------------------------------------------------
# force_release
# ---------------------------------------------------------------------------

class TestForceRelease:

    def test_blocked_without_env_var_set(self, patched, monkeypatch):
        """The flag is captured at import time; ensure it's False here."""
        monkeypatch.setattr("infrastructure.run_lock._ALLOW_FORCE_RELEASE", False)
        with pytest.raises(PermissionError, match="STAGE9_ALLOW_FORCE_RELEASE"):
            RedisRunLock().force_release("t1")

    def test_force_release_with_env_var_deletes(self, patched, monkeypatch):
        monkeypatch.setattr("infrastructure.run_lock._ALLOW_FORCE_RELEASE", True)
        patched.delete.return_value = 1
        assert RedisRunLock().force_release("t1") is True

    def test_force_release_returns_false_when_no_lock(self, patched, monkeypatch):
        monkeypatch.setattr("infrastructure.run_lock._ALLOW_FORCE_RELEASE", True)
        patched.delete.return_value = 0
        assert RedisRunLock().force_release("t1") is False

    def test_force_release_redis_error_returns_false(self, patched, monkeypatch):
        monkeypatch.setattr("infrastructure.run_lock._ALLOW_FORCE_RELEASE", True)
        patched.delete.side_effect = redis_lib.RedisError("bad")
        assert RedisRunLock().force_release("t1") is False
