"""
Sub-Stage 9.0 PRELOADING — Data Fingerprinting and Processing Tier Classification.

Spec: STAGE_9_TECHNICAL_CONTEXT.md Part 4.3 | Linear ticket: ATH-32

Each SKU-tenant pair is assigned to one of three processing tiers before the
forecasting pipeline runs:

  cache   — inputs are byte-identical to the previous run; reuse stored forecast
  partial — demand signal changed modestly; run lightweight refresh only
  full    — significant change or no prior run; execute full model pipeline

Target split at scale: ~70 % cache / 20 % partial / 10 % full.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Literal, TypedDict

from infrastructure.constants import PARTIAL_TIER_CHANGE_PCT

Tier = Literal["cache", "partial", "full"]

# Rounding precision for the two numeric fingerprint components.
# Without explicit rounding, pandas/numpy float accumulation noise across runs
# produces microscopically different values on identical inputs, which would
# force every SKU to the 'full' tier on re-runs and defeat the cache entirely.
_SALES_DECIMALS: int = 2
_OOS_DECIMALS: int = 3

# usedforsecurity was added in Python 3.9. Build kwargs once at import time so
# the call site stays clean and the fallback is handled in a single place.
_SHA256_KWARGS: dict[str, bool] = (
    {"usedforsecurity": False} if sys.version_info >= (3, 9) else {}
)




class _CacheEntry(TypedDict):
    fingerprint: str


def compute_fingerprint(
    sku_id: str,
    sales_last_30d: list[float],
    pattern_label: str,
    oos_pct: float,
    lifecycle_stage: str | None,
) -> str:
    """Compute a stable SHA-256 fingerprint for a SKU's input snapshot.

    The fingerprint captures every input that influences the forecast so that
    the tier-classification logic can detect byte-identical re-runs (cache),
    modest demand shifts (partial), or significant changes (full) without
    re-executing the full model pipeline.

    Args:
        sku_id: Unique identifier for the SKU.  Used as the primary grouping
            key; two different SKUs with the same numeric inputs still produce
            different fingerprints.
        sales_last_30d: Raw daily sales figures for the trailing 30-day window.
            Callers may pass a longer series; only the final 30 elements are
            hashed.  Values are rounded to ``_SALES_DECIMALS`` places before
            serialisation to suppress float-accumulation noise.  If fewer than
            30 elements are supplied (e.g. a new SKU still ramping up), the
            fingerprint is computed over the shorter series — the hash will
            change as more days accumulate even if existing values are
            identical, which forces ``"full"`` tier runs until the series
            stabilises at 30 elements.
        pattern_label: Demand-pattern classification label (e.g. "steady",
            "seasonal", "erratic") assigned upstream.  Included verbatim so
            that a reclassification triggers a full-tier re-run even when the
            raw sales figures are unchanged.
        oos_pct: Out-of-stock rate for the window, expressed as a fraction in
            [0, 1].  Rounded to ``_OOS_DECIMALS`` places before serialisation.
        lifecycle_stage: Product lifecycle label (e.g. "launch", "mature",
            "eol"), or ``None`` if not yet classified.  ``None`` is normalised
            to an empty string so the JSON payload remains stable across runs
            where the field is absent.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest that uniquely
        represents the combination of inputs at the precision levels defined by
        ``_SALES_DECIMALS`` and ``_OOS_DECIMALS``.

    Notes:
        **Why ``sort_keys=True`` is mandatory**: Python dicts preserve insertion
        order since 3.7, but ``json.dumps`` without ``sort_keys`` would produce
        different byte strings if the payload were ever constructed in a
        different order (e.g. after a refactor or across CPython versions).
        ``sort_keys=True`` is the single guarantee that the serialised bytes are
        canonical regardless of insertion order or interpreter internals.

        **Why floats are rounded**: pandas and numpy accumulate floating-point
        arithmetic noise differently depending on chunking, dtype promotion, and
        BLAS implementation.  Two runs over identical source data can produce
        values that differ at the 15th decimal place.  Rounding to
        ``_SALES_DECIMALS`` / ``_OOS_DECIMALS`` collapses that noise so the
        fingerprint is stable on true re-runs and only changes when the signal
        itself changes.
    """
    payload = {
        "sku": sku_id,
        "sales": [round(v, _SALES_DECIMALS) for v in sales_last_30d[-30:]],
        "pattern": pattern_label,
        "oos": round(oos_pct, _OOS_DECIMALS),
        "lifecycle": lifecycle_stage if lifecycle_stage is not None else "",
    }

    # sort_keys=True is what guarantees determinism across Python versions and
    # dict insertion orders — do not remove.
    serialised = json.dumps(payload, sort_keys=True).encode("utf-8")

    return hashlib.sha256(serialised, **_SHA256_KWARGS).hexdigest()


def classify_tier(
    sku_id: str,
    current_fingerprint: str,
    fingerprint_cache: dict[str, _CacheEntry],
    *,
    current_pattern_label: str = "",
    current_demand_total: float = 0.0,
) -> Tier:
    """Classify a SKU into a processing tier based on its fingerprint.

    Decision table (first match wins):

        1. ``sku_id`` not in ``fingerprint_cache``                       -> ``"full"``
        2. ``current_fingerprint`` == cached fingerprint                 -> ``"cache"``
        3. pattern_label changed vs cached                               -> ``"full"``
        4. cached demand_total > 0 and change < _PARTIAL_CHANGE_PCT      -> ``"partial"``
        5. otherwise                                                     -> ``"full"``
    """
    if sku_id not in fingerprint_cache:
        return "full"

    cached = fingerprint_cache[sku_id]
    if current_fingerprint == cached["fingerprint"]:
        return "cache"

    cached_pattern = cached.get("pattern_label")
    if cached_pattern is not None and cached_pattern != current_pattern_label:
        return "full"

    cached_total = cached.get("demand_total") or 0.0
    if cached_total > 0:
        change = abs(current_demand_total - cached_total) / cached_total
        if change < PARTIAL_TIER_CHANGE_PCT:
            return "partial"

    return "full"
