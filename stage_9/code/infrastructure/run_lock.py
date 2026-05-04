"""
run_lock.py

WHY THIS EXISTS
---------------
LangGraph triggers Stage 9 automatically the moment run.status becomes
'patterns_discovered'. Without a lock, a delayed first run and a fast retry
(triggered by a status re-check) can overlap for the same tenant — both
processes write forecast rows for the same run_id and Stage 10 reads
corrupted duplicates.

This module provides a single class — RedisRunLock — that Stage9Agent calls
at the start and end of every run to prevent concurrent execution per tenant.

LOCK DESIGN
-----------
  Key format  :  stage9_lock_{tenant_id}
  Value       :  UUID token (proves ownership on release)
  TTL         :  4 hours (LOCK_TTL_SECONDS)

  Acquire     :  Redis SET NX EX — atomic in one round-trip.
                 NX = only set if the key does not already exist.
                 EX = attach the TTL atomically in the same command.
                 This eliminates the SETNX + separate EXPIRE race condition
                 where a crash between the two commands leaves a key with
                 no TTL, blocking all future runs for that tenant forever.

  Release     :  Lua script (_RELEASE_SCRIPT) — atomic GET + compare + DEL.
                 A non-atomic Python GET → compare → DEL has a race:
                   1. Process A reads the token — matches.
                   2. Lock expires between read and delete (TTL was nearly gone).
                   3. Process B acquires the lock with a new token.
                   4. Process A deletes Process B's lock.
                 The Lua script runs as a single Redis command so steps 2-4
                 cannot happen.

  Token       :  UUID4 generated at acquire time. Only the process holding
                 the token can release the lock. If the TTL fires and another
                 process acquires the lock, the old process's release() sees
                 a token mismatch and returns False without deleting.

CONNECTION POOL
---------------
A module-level ConnectionPool is created once when the module is imported.
All RedisRunLock instances share this pool. Without pooling, every acquire()
and release() call opens a new TCP connection to Redis — at 10 tenants
running concurrently this caused the Stage-8-at-10-customers connection
exhaustion failure. Pool is capped at MAX_POOL_CONNECTIONS (10) with
socket timeouts to prevent hung connections from exhausting it silently.

USAGE IN Stage9Agent
--------------------
    from run_lock import RedisRunLock, LockAcquisitionError

    lock = RedisRunLock()
    acquired, token = lock.acquire(tenant_id)
    if not acquired:
        raise LockAcquisitionError(f"Run already active for tenant {tenant_id}")
    try:
        ... run pipeline ...
    finally:
        lock.release(tenant_id, token)

DEPENDENCIES
------------
  pip install redis
"""

import logging
import os
import threading
import uuid
from typing import Optional

import redis
from redis import ConnectionPool

from infrastructure.constants import LOCK_TTL_SECONDS, LOCK_KEY_TEMPLATE
from infrastructure.errors import LockAcquisitionError

# ---------------------------------------------------------------------------
# Logging — per-module logger so log lines are attributable to run_lock.py
# specifically, not just to 'stage9' as a whole.
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lock constants
# ---------------------------------------------------------------------------

# Connection pool ceiling. Caps the number of simultaneous Redis connections
# this process can hold. Defaults to 10 (enough for local dev and the first
# production tier). Override via REDIS_POOL_SIZE env var for environments
# running more than 10 concurrent tenant Stage 9 processes.
from infrastructure.config import REDIS_POOL_SIZE as _REDIS_POOL_SIZE, ALLOW_FORCE_RELEASE as _ALLOW_FORCE_RELEASE  # noqa: E402
MAX_POOL_CONNECTIONS: int = _REDIS_POOL_SIZE

# ---------------------------------------------------------------------------
# Lua release script — atomic GET + compare + DEL in a single Redis command.
#
# Why Lua: Redis executes Lua scripts atomically. The entire script runs
# as one unit — no other Redis command can interleave between the KEYS[1]
# GET and the DEL. This prevents the race condition where:
#   1. Python reads the stored token (matches).
#   2. Lock TTL expires between 'the read' and 'delete'.
#   3. Another process acquires the lock with a new token.
#   4. Python deletes the new process's lock.
#
# Returns 1 if the key was deleted (release succeeded).
# Returns 0 if the key did not exist or the token did not match.
# ---------------------------------------------------------------------------
_RELEASE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""

# ---------------------------------------------------------------------------
# Module-level connection pool.
#
# Created once on import. All RedisRunLock instances share this pool so
# the process never opens more than MAX_POOL_CONNECTIONS connections to Redis
# regardless of how many lock instances exist.
#
# REDIS_URL defaults to localhost for local development. In production this
# is set via environment variable, e.g.:
#   export REDIS_URL=redis://:password@redis-host:6379/0
#
# socket_connect_timeout=3  — fail fast if Redis is unreachable at connect time
# socket_timeout=10         — fail fast if a command hangs mid-flight
# ---------------------------------------------------------------------------
from infrastructure.config import REDIS_URL as _REDIS_URL_CFG  # noqa: E402
_REDIS_URL: str = _REDIS_URL_CFG

# Pool is created lazily on first use rather than at import time.
# This ensures REDIS_URL is fully resolved from the environment before the
# pool is constructed, regardless of import order or startup sequence.
# _pool_lock prevents two threads from simultaneously creating separate pools
# on the first call (double-checked locking pattern).
_pool: Optional[ConnectionPool] = None
_pool_lock: threading.Lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    """
    Return the module-level connection pool, creating it on first call.

    Thread-safe: uses a lock to ensure only one pool is ever created even
    when multiple threads call _get_pool() simultaneously at startup.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            # Re-check inside the lock — another thread may have created
            # the pool while we were waiting to acquire it.
            if _pool is None:
                _pool = ConnectionPool.from_url(
                    _REDIS_URL,
                    max_connections=MAX_POOL_CONNECTIONS,
                    socket_connect_timeout=3,
                    socket_timeout=10,
                    # decode_responses=False: keep bytes; we compare byte
                    # strings in the Lua script. Explicit decode avoids
                    # silent encoding mismatches across redis-py versions.
                    decode_responses=False,
                )
    return _pool


def _get_client() -> redis.Redis:
    """
    Return a Redis client backed by the module-level pool.

    Using connection_pool= means the client borrows a connection from the
    pool for each command and returns it immediately — no connection is held
    open between calls.
    """
    return redis.Redis(connection_pool=_get_pool())


# ---------------------------------------------------------------------------
# RedisRunLock
# ---------------------------------------------------------------------------

class RedisRunLock:
    """
    Distributed per-tenant run lock for Stage 9.

    All instances share the module-level connection pool. Callers do not
    need to pass a Redis client — instantiate with no arguments.
    """

    @staticmethod
    def _validate_tenant_id(tenant_id: str) -> None:
        """
        Guard against blank or non-string tenant_id values.

        A missing or empty tenant_id would produce key 'stage9_lock_' or
        'stage9_lock_None' — both silently wrong and shared across tenants.
        Catching this early produces a clear error rather than a confusing
        Redis or pipeline failure downstream.
        """
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError(
                f"tenant_id must be a non-empty string, got {tenant_id!r}"
            )

    @staticmethod
    def _key(tenant_id: str) -> str:
        """Return the Redis key for this tenant's lock."""
        return LOCK_KEY_TEMPLATE.format(tenant_id=tenant_id)

    def acquire(self, tenant_id: str) -> tuple[bool, Optional[str]]:
        """
        Attempt to acquire the run lock for this tenant.

        Returns (True, token) when the lock is acquired successfully.
        Returns (False, None) when another process already holds the lock.

        The token must be stored by the caller and passed to release() —
        it proves the caller is the process that originally acquired the lock.

        Raises LockAcquisitionError on Redis infrastructure failure
        (unreachable, timeout, pool exhausted). The caller should treat this
        as a hard failure and mark the run FAILED, not retry immediately.
        """
        token = str(uuid.uuid4())
        self._validate_tenant_id(tenant_id)
        key = self._key(tenant_id)
        try:
            client = _get_client()
            acquired = client.set(key, token, nx=True, ex=LOCK_TTL_SECONDS)
        except redis.RedisError as exc:
            # Redis infrastructure problem — not a normal "lock held" case.
            # Log at ERROR so on-call is alerted; do not silently swallow.
            log.error(
                "Redis error during lock acquire  tenant=%s  error=%s",
                tenant_id, exc,
            )
            raise LockAcquisitionError(
                f"Could not acquire run lock for tenant {tenant_id}: {exc}"
            ) from exc

        if acquired:
            # Log only the first 8 chars of the token — enough to correlate
            # acquire/release pairs in logs without exposing the full secret.
            # The full token proves lock ownership; leaking it via logs would
            # allow anyone with log access to forge a release() call.
            log.info(
                "Run lock acquired  tenant=%s  key=%s  ttl=%ds  token_prefix=%s",
                tenant_id, key, LOCK_TTL_SECONDS, token[:8],
            )
            return True, token

        log.warning("Run lock already held  tenant=%s  key=%s", tenant_id, key)
        return False, None

    def release(self, tenant_id: str, token: str) -> bool:
        """
        Release the run lock for this tenant using an atomic Lua script.

        The Lua script performs GET + compare + DEL as a single atomic Redis
        command. This prevents the race condition where the lock TTL fires
        between a Python-level GET and DEL, causing the new lock owner's key
        to be deleted.

        Returns True  — lock was owned by this token and successfully deleted.
        Returns False — lock had already expired, or is owned by a different
                        process (token mismatch). Neither case deletes the key.

        Redis errors are logged at ERROR level and return False — the lock
        will expire on its own via TTL so the pipeline is not permanently stuck,
        but the failure is surfaced for investigation.
        """
        self._validate_tenant_id(tenant_id)
        key = self._key(tenant_id)
        try:
            client = _get_client()
            result = client.eval(_RELEASE_SCRIPT, 1, key, token)
        except redis.RedisError as exc:
            # Log but do not raise — a release failure should not crash the
            # pipeline's finally block. The TTL will clean up the key.
            log.error(
                "Redis error during lock release  tenant=%s  error=%s",
                tenant_id, exc,
            )
            return False

        if result == 1:
            log.info(
                "Run lock released  tenant=%s  key=%s  token_prefix=%s",
                tenant_id, key, token[:8],
            )
            return True

        # result == 0 means key was gone (TTL fired) or token didn't match.
        # Both are logged at WARNING — unexpected but not pipeline-breaking.
        log.warning(
            "Run lock release no-op — key expired or token mismatch  tenant=%s  key=%s",
            tenant_id, key,
        )
        return False

    def is_locked(self, tenant_id: str) -> bool:
        """
        Return True if an active run lock exists for this tenant.

        Used by monitoring and health-check endpoints to inspect lock state
        without acquiring or releasing it. Not used in the hot pipeline path.
        """
        self._validate_tenant_id(tenant_id)
        try:
            return _get_client().exists(self._key(tenant_id)) == 1
        except redis.RedisError as exc:
            log.error("Redis error during is_locked check  tenant=%s  error=%s", tenant_id, exc)
            # Treat Redis unreachable as "unknown" — return False rather than raising,
            # so health-check callers don't blow up on infra issues.
            return False

    def ttl(self, tenant_id: str) -> int:
        """
        Return remaining lock TTL in seconds.

        -2 = key does not exist (no active lock).
        -1 = key exists but has no TTL (should never happen — indicates
             a key was set without EX, which this module never does).

        Used by monitoring to detect locks approaching expiry during long runs.
        """
        self._validate_tenant_id(tenant_id)
        try:
            return _get_client().ttl(self._key(tenant_id))
        except redis.RedisError as exc:
            log.error("Redis error during ttl check  tenant=%s  error=%s", tenant_id, exc)
            return -2

    def force_release(self, tenant_id: str) -> bool:
        """
        Delete the lock unconditionally without token verification.

        FOR MANUAL OPERATIONAL USE ONLY.

        Call this when a Stage 9 process is confirmed dead and its lock
        must be cleared before the lock TTL expires — for example, after
        a deployment that killed a mid-run worker.

        SAFETY GATE: this method requires the environment variable
        `STAGE9_ALLOW_FORCE_RELEASE=true` to be set. Otherwise it raises
        `PermissionError` immediately. The gate prevents accidental calls
        from pipeline code (where bypassing the token check could delete
        a healthy process's lock).
        """
        if not _ALLOW_FORCE_RELEASE:
            raise PermissionError(
                "RedisRunLock.force_release is gated behind "
                "STAGE9_ALLOW_FORCE_RELEASE=true to prevent accidental "
                "use from pipeline code. Set the env var explicitly "
                "(e.g. in an operator shell) before calling."
            )
        self._validate_tenant_id(tenant_id)
        try:
            deleted = _get_client().delete(self._key(tenant_id))
        except redis.RedisError as exc:
            log.error(
                "Redis error during force_release  tenant=%s  error=%s",
                tenant_id, exc,
            )
            return False

        if deleted:
            log.warning(
                "Run lock force-released  tenant=%s  key=%s",
                tenant_id, self._key(tenant_id),
            )
        else:
            log.warning(
                "force_release called but no lock existed  tenant=%s", tenant_id
            )
        return bool(deleted)
