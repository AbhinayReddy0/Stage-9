"""
unit/orchestration/test_fingerprint.py — fingerprint.py coverage.

Production module is two pure functions:
    compute_fingerprint(sku_id, sales_last_30d, pattern_label, oos_pct, lifecycle_stage)
    classify_tier(sku_id, current_fingerprint, fingerprint_cache)

Both are deterministic — no DB, no I/O. Easy to unit test exhaustively.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from forecasting.fingerprint import (
    compute_fingerprint,
    classify_tier,
    _SALES_DECIMALS,
    _OOS_DECIMALS,
)


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------

class TestComputeFingerprint:

    def test_returns_64_char_hex_sha256(self):
        fp = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, None)
        assert isinstance(fp, str)
        assert len(fp) == 64
        # All chars are valid hex
        int(fp, 16)

    def test_identical_inputs_give_identical_fingerprint(self):
        a = compute_fingerprint("sku-1", [1.0, 2.0, 3.0], "stable", 0.05, "mature")
        b = compute_fingerprint("sku-1", [1.0, 2.0, 3.0], "stable", 0.05, "mature")
        assert a == b

    def test_different_sku_ids_give_different_fingerprints(self):
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, None)
        b = compute_fingerprint("sku-2", [1.0] * 30, "stable", 0.0, None)
        assert a != b

    def test_different_sales_give_different_fingerprints(self):
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [2.0] * 30, "stable", 0.0, None)
        assert a != b

    def test_different_pattern_label_gives_different_fingerprint(self):
        """Reclassification triggers a full-tier rerun even if numbers match."""
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [1.0] * 30, "trending", 0.0, None)
        assert a != b

    def test_different_oos_pct_gives_different_fingerprint(self):
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.00, None)
        b = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.10, None)
        assert a != b

    def test_different_lifecycle_stage_gives_different_fingerprint(self):
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, "launch")
        b = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, "mature")
        assert a != b

    # ----- numeric noise / rounding semantics --------------------------

    def test_sales_rounded_to_two_decimals(self):
        """Float-noise at the 3rd decimal must NOT change the fingerprint."""
        a = compute_fingerprint("sku-1", [1.001] * 30, "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [1.002] * 30, "stable", 0.0, None)
        # Both round to 1.00 → same fingerprint
        assert a == b

    def test_sales_rounded_at_boundary(self):
        """Rounding kicks in at _SALES_DECIMALS — beyond that, noise is invisible."""
        assert _SALES_DECIMALS == 2  # invariant: this test's epsilon depends on it
        a = compute_fingerprint("sku-1", [1.234567] * 30, "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [1.23] * 30,     "stable", 0.0, None)
        assert a == b

    def test_oos_rounded_to_three_decimals(self):
        """OOS at the 5th decimal must NOT change the fingerprint —
        both values round to 0.1234 at _OOS_DECIMALS=3 only when the 4th
        decimal is the same. Use two values that share the same 4th decimal
        (so the round-to-3 result matches) and differ at the 5th."""
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.12341, None)
        b = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.12349, None)
        # Both round to 0.123 (4th decimal = 4 < 5 → rounds down)
        assert a == b

    # ----- window truncation -------------------------------------------

    def test_only_last_30_days_hashed(self):
        """Sales array longer than 30 — only the trailing 30 affect the hash."""
        long_array = [99.0] * 100 + [1.0] * 30   # 130 elements
        a = compute_fingerprint("sku-1", long_array,            "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [1.0] * 30,            "stable", 0.0, None)
        assert a == b

    def test_short_series_uses_full_length(self):
        """Fewer than 30 elements: the whole series is hashed.
        New SKUs ramping up will keep changing fingerprints day-by-day
        until 30 days accumulate."""
        a = compute_fingerprint("sku-1", [1.0] * 5,  "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [1.0] * 10, "stable", 0.0, None)
        assert a != b

    def test_empty_sales_does_not_crash(self):
        fp = compute_fingerprint("sku-1", [], "stable", 0.0, None)
        assert len(fp) == 64

    # ----- None / empty lifecycle --------------------------------------

    def test_none_lifecycle_normalises_to_empty_string(self):
        """None → '' so payload is stable across runs missing the field."""
        a = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, None)
        b = compute_fingerprint("sku-1", [1.0] * 30, "stable", 0.0, "")
        assert a == b

    # ----- determinism across dict insertion orders --------------------

    def test_canonical_via_sort_keys_explicit(self):
        """Sanity: building the same payload by hand and serialising with
        sort_keys=True must match `compute_fingerprint`."""
        sku, sales, pat, oos, lc = "sku-X", [1.0, 2.0, 3.0], "stable", 0.05, "mature"
        manual_payload = {
            "sku": sku,
            "sales": [round(v, _SALES_DECIMALS) for v in sales],
            "pattern": pat,
            "oos": round(oos, _OOS_DECIMALS),
            "lifecycle": lc,
        }
        manual = hashlib.sha256(
            json.dumps(manual_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        actual = compute_fingerprint(sku, sales, pat, oos, lc)
        assert manual == actual

    # ----- types / negative inputs -------------------------------------

    def test_negative_sales_handled(self):
        """Returns may produce negative qty in clean_orders — fingerprint
        must still be computable, not crash."""
        fp = compute_fingerprint("sku-1", [-1.0, -2.0, 3.0], "stable", 0.0, None)
        assert len(fp) == 64

    def test_very_large_sales_values(self):
        fp = compute_fingerprint("sku-1", [1e9] * 30, "stable", 0.0, None)
        assert len(fp) == 64


# ---------------------------------------------------------------------------
# classify_tier
# ---------------------------------------------------------------------------

class TestClassifyTier:

    def test_unseen_sku_returns_full(self):
        """Decision rule #1: sku_id not in cache → full."""
        assert classify_tier("sku-1", "any-fp", {}) == "full"

    def test_cached_match_returns_cache(self):
        """Decision rule #2: cached fingerprint == current → cache."""
        cache = {"sku-1": {"fingerprint": "abc"}}
        assert classify_tier("sku-1", "abc", cache) == "cache"

    def test_cached_mismatch_returns_full(self):
        """Decision rule #3: cached fingerprint != current → full."""
        cache = {"sku-1": {"fingerprint": "abc"}}
        assert classify_tier("sku-1", "different", cache) == "full"

    def test_first_match_wins_other_skus_irrelevant(self):
        """Other SKUs in the cache must not affect this SKU's decision."""
        cache = {
            "sku-other-1": {"fingerprint": "fp-other-1"},
            "sku-1":       {"fingerprint": "abc"},
            "sku-other-2": {"fingerprint": "fp-other-2"},
        }
        assert classify_tier("sku-1", "abc", cache) == "cache"

    def test_returns_only_cache_or_full(self):
        """Without partial-tier logic in this function, output is binary."""
        cache = {"sku-1": {"fingerprint": "abc"}}
        for current in ("abc", "different", ""):
            t = classify_tier("sku-1", current, cache)
            assert t in {"cache", "full"}

    def test_empty_string_fingerprint_handled(self):
        cache = {"sku-1": {"fingerprint": ""}}
        assert classify_tier("sku-1", "",    cache) == "cache"
        assert classify_tier("sku-1", "abc", cache) == "full"
