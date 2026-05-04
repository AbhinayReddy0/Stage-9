"""
bootstrap_quantiles() — converts a point forecast into {mean, p50, p80, p90}.

Called by Sub-Stage 9.5 ONCE per SKU per horizon, immediately after
predict_all_horizons() returns. Shared across all 5 model classes — not a
method on any individual model.

Algorithm — 5 steps, always in order:
    Step 1  len(residuals) < 3  → log-normal proxy (E004 fix)
    Step 2  Bootstrap: resample residuals n=1000 times, shift by point
    Step 3  Percentiles: p50, p80, p90 from sampled distribution
    Step 4  Enforce ordering: p50 ≤ p80 ≤ p90  (ALWAYS — CRITICAL)
    Step 5  Return: {'mean': point, 'p50': p50, 'p80': p80, 'p90': p90}

"""

from __future__ import annotations

import numpy as np

from infrastructure.constants import BOOTSTRAP_SAMPLE_COUNT, BOOTSTRAP_SEED

__all__ = ["bootstrap_quantiles", "BOOTSTRAP_UNCERTAINTY"]


# ===========================================================================
# Pattern uncertainty factors for log-normal proxy (Step 1)
# ===========================================================================
# Controls the spread of the log-normal distribution used when fewer than 3
# residuals are available (new SKUs, model failures). Higher factor → wider
# quantile spread → more conservatism in the face of uncertainty.
#
# These values are FIXED algorithm constants — not in tenant_learning_params.
# Do not change them. They encode calibrated uncertainty by demand archetype.

BOOTSTRAP_UNCERTAINTY: dict[str, float] = {
    "cold_start":   0.60,  # no history at all — very widespread
    "intermittent": 0.50,  # sporadic demand — moderate-high uncertainty
    "seasonal":     0.40,  # cycle direction known, magnitude uncertain
    "trending":     0.35,  # direction known, rate-of-change uncertain
    "stable":       0.25,  # most predictable archetype — tightest spread

}


def bootstrap_quantiles(
    point: float,
    residuals: np.ndarray,
    pattern: str,
    n: int = BOOTSTRAP_SAMPLE_COUNT,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Convert a single point forecast into {mean, p50, p80, p90} quantiles.

    The bootstrap simulates the distribution of plausible future outcomes by
    resampling the model's recent residuals. p50/p80/p90 represent the
    50th/80th/90th percentiles of that distribution.

    Args:
        point:     Model's point forecast for the horizon (cumulative demand,
                   already multiplied by oos_factor). Must be ≥ 0.
        residuals: Array of (actual − fitted) values from compute_residuals().
                   compute_residuals() already returns only the last 30 rows.
                   If len < 3: log-normal proxy is used (E004 fix).
        pattern:   Demand pattern string. Selects log-normal uncertainty factor
                   from BOOTSTRAP_UNCERTAINTY. Valid: keys of BOOTSTRAP_UNCERTAINTY.
        n:         Bootstrap sample count. Default 1000.
                   Higher n → smoother quantiles but slower; 1000 is calibrated.
        seed:      RNG seed for reproducibility. Fixed at 42. Not a security seed.

    Returns:
        dict with exactly 4 keys:
            'mean': point  (the raw point forecast — display only, not for ordering)
            'p50':  float  (50th percentile — median scenario)
            'p80':  float  (80th percentile — used for trending/stable ordering)
            'p90':  float  (90th percentile — conservative order; used for seasonal/
                            intermittent/cold_start and criticality_tier='A' override)
        All values ≥ 0. Invariant: p50 ≤ p80 ≤ p90 (always enforced in Step 4).

    Edge cases handled:
        - point == 0: returns all-zero quantiles immediately.
        - len(residuals) < 3: log-normal proxy (E004).
        - unknown pattern: defaults to factor 0.50 (intermittent-level uncertainty).

    Security note:
        seed=42 is for numerical reproducibility, not cryptographic security.
        The RNG is process-local and has no external side effects.
    """
    # Defensive clamp — point must be non-negative before bootstrap
    point = float(max(0.0, point))

    # ---------------------------------------------------------------------------
    # Fast path: zero-demand forecast → all quantiles are zero
    # ---------------------------------------------------------------------------
    if point == 0.0:
        return {"mean": 0.0, "p50": 0.0, "p80": 0.0, "p90": 0.0}

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------------
    # Step 1: Choose sampling strategy based on residual count
    # ---------------------------------------------------------------------------
    if len(residuals) < 3:
        # not enough residual history to bootstrap reliably.
        # Generate samples from a log-normal distribution parameterised by the
        # pattern's uncertainty factor (coefficient of variation).
        #
        # Log-normal parameterization (Build Plan §04, BOOTSTRAP_UNCERTAINTY table):
        #   mean of log  = log(point)
        #   sigma of log = log(1 + CoV)   where CoV = pattern_factor
        #
        # Example (cold_start, point=100, factor=0.60):
        #   sigma_param = log(1 + 0.60) = log(1.6) ≈ 0.47
        #   Distribution median ≈ 100, mean slightly higher due to log-normal skew.
        sigma_factor = BOOTSTRAP_UNCERTAINTY.get(pattern, 0.50)

        if point <= 0.0:
            return {"mean": 0.0, "p50": 0.0, "p80": 0.0, "p90": 0.0}

        log_sigma = np.log(1.0 + sigma_factor)
        samples = rng.lognormal(
            mean=np.log(point),
            sigma=log_sigma,
            size=n,
        )

    else:
        # ---------------------------------------------------------------------------
        # Step 2: Bootstrap resampling — resample residuals with replacement
        # ---------------------------------------------------------------------------
        # Each sample = point forecast + a randomly drawn historical deviation.
        # This simulates "what would the actual demand be if future error patterns
        # match historical error patterns?"
        resampled = rng.choice(residuals, size=n, replace=True)
        samples   = resampled + point

    # Clamp to non-negative — demand cannot go below zero
    samples = np.maximum(samples, 0.0)

    # ---------------------------------------------------------------------------
    # Step 3: Compute percentiles from the simulated distribution
    # ---------------------------------------------------------------------------
    p50 = float(np.percentile(samples, 50))
    p80 = float(np.percentile(samples, 80))
    p90 = float(np.percentile(samples, 90))

    # ---------------------------------------------------------------------------
    # Step 4: Enforce monotone ordering  (ALWAYS — CRITICAL)
    # ---------------------------------------------------------------------------
    # Floating-point arithmetic can produce p50 > p80 when all samples are nearly
    # identical (e.g. a very tight distribution from a stable SKU). Stage 10 reads
    # p90 as the conservative order quantity — a violated ordering causes silent
    # under-ordering. Always sort to guarantee the invariant.
    # Done Criterion D5: p50 ≤ p80 ≤ p90 must hold for every call, zero violations.
    vals = sorted([p50, p80, p90])
    p50, p80, p90 = vals[0], vals[1], vals[2]

    # ---------------------------------------------------------------------------
    # Step 5: Return
    # ---------------------------------------------------------------------------
    return {
        "mean": point,      # raw point forecast — display only, never used for ordering
        "p50":  p50,
        "p80":  p80,
        "p90":  p90,
    }
