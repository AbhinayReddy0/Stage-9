"""
Stage 9 run-lock verification.

Runs against fakeredis (no Redis server required).

Usage (from tests directory):
    python verify_run_lock.py
or (from project root):
    python -m stage_9.code.tests.verify_run_lock

Requires:
    pip install fakeredis
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import fakeredis

# Ensure the code directory is on sys.path (flat module layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import infrastructure.run_lock as run_lock
from infrastructure.run_lock import LOCK_TTL_SECONDS, RedisRunLock


def _make_fake_redis() -> fakeredis.FakeRedis:
    """
    Return a FakeRedis instance with a Python implementation of eval() that
    mirrors the Lua release script in run_lock.py.

    fakeredis[lua] requires lupa which is unreliable on Windows. Implementing
    the one script we use in Python avoids that dependency entirely.
    """
    fake = fakeredis.FakeRedis(decode_responses=False)

    def _eval(script: str, numkeys: int, *args):
        key   = args[0] if isinstance(args[0], bytes) else args[0].encode()
        token = args[1] if isinstance(args[1], bytes) else args[1].encode()
        stored = fake.get(key)
        if stored == token:
            return fake.delete(key)
        return 0

    fake.eval = _eval
    return fake


def main() -> int:
    # Single FakeRedis instance shared across all _get_client() calls so that
    # the lock's internal client and the verification client see the same state.
    fake = _make_fake_redis()

    with patch.object(run_lock, "_get_client", return_value=fake):
        tenant_id = f"verify-{uuid.uuid4()}"
        lock = RedisRunLock()
        key = lock._key(tenant_id)
        client = fake
        failures = 0

        def report(name: str, ok: bool, detail: str = "") -> None:
            nonlocal failures
            status = "PASS" if ok else "FAIL"
            print(f"{status}: {name}" + (f" ({detail})" if detail else ""))
            if not ok:
                failures += 1

        try:
            report("fakeredis reachable", client.ping())

            # Start clean.
            client.delete(key)

            # 1. Acquire succeeds when key is absent.
            acquired, token = lock.acquire(tenant_id)
            report("acquire returns token", acquired and token is not None)

            # 2. Key exists with matching value.
            # decode_responses=False — Redis returns bytes, so encode token for comparison.
            stored = client.get(key)
            report("key set in redis", stored == token.encode(), f"stored={stored!r}")

            # 3. TTL is set and within the expected window.
            ttl = client.ttl(key)
            report(
                "ttl set to ~4h",
                0 < ttl <= LOCK_TTL_SECONDS,
                f"ttl={ttl}s expected<={LOCK_TTL_SECONDS}s",
            )

            # 4. Second acquire fails while lock is held.
            acquired2, token2 = lock.acquire(tenant_id)
            report("second acquire blocked", not acquired2 and token2 is None)

            # 5. Release with wrong token is rejected.
            bad_release = lock.release(tenant_id, "not-the-real-token")
            report("wrong token rejected", bad_release is False)
            still_there = client.get(key)
            report("lock still held after bad release", still_there == token.encode())

            # 6. Release with correct token succeeds.
            ok_release = lock.release(tenant_id, token)
            report("correct token released", ok_release is True)
            report("key removed after release", client.get(key) is None)

            # 7. Re-acquire works after release and produces a new token.
            acquired3, token3 = lock.acquire(tenant_id)
            report(
                "re-acquire after release",
                acquired3 and token3 is not None and token3 != token,
            )

        finally:
            client.delete(key)

    print("----")
    print("ALL PASS" if failures == 0 else f"{failures} FAIL(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
