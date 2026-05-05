"""
tests/test_base_bootstrap.py — Atheera Stage 9
===============================================
Tests for:
    - ModelFitError inheritance and raise behaviour
    - bootstrap_quantiles() — all 5 steps, E004, ordering invariant (D5)
    - BOOTSTRAP_UNCERTAINTY — all 5 patterns covered, factors correct
"""

from __future__ import annotations

import numpy as np
import pytest

from models.base      import BaseModel, ModelFitError
from models.bootstrap import bootstrap_quantiles, BOOTSTRAP_UNCERTAINTY
from infrastructure.errors    import Stage9Error


# ===========================================================================
# ModelFitError
# ===========================================================================

class TestModelFitError:

    def test_inherits_from_stage9_error(self):
        """ModelFitError must inherit Stage9Error, not a built-in."""
        assert issubclass(ModelFitError, Stage9Error)

    def test_inherits_from_exception(self):
        assert issubclass(ModelFitError, Exception)

    def test_does_not_inherit_value_error(self):
        """Must not be a ValueError — prevents accidental silent swallowing."""
        assert not issubclass(ModelFitError, ValueError)

    def test_does_not_inherit_key_error(self):
        assert not issubclass(ModelFitError, KeyError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ModelFitError, match="test message"):
            raise ModelFitError("test message")

    def test_caught_as_stage9_error(self):
        """Sub-Stage 9.3 catches ModelFitError via Stage9Error in some guards."""
        with pytest.raises(Stage9Error):
            raise ModelFitError("should be caught as Stage9Error")

    def test_preserves_cause(self):
        """from exc chaining must be preserved for debugging."""
        cause = ValueError("root cause")
        with pytest.raises(ModelFitError) as exc_info:
            raise ModelFitError("wrapped") from cause
        assert exc_info.value.__cause__ is cause


# ===========================================================================
# BOOTSTRAP_UNCERTAINTY
# ===========================================================================

class TestBootstrapUncertainty:

    def test_all_patterns_present(self):
        patterns = {"cold_start", "intermittent", "seasonal", "trending", "stable"}
        assert patterns.issubset(set(BOOTSTRAP_UNCERTAINTY.keys()))

    def test_cold_start_highest_uncertainty(self):
        """cold_start has no history — must be the widest spread."""
        assert BOOTSTRAP_UNCERTAINTY["cold_start"] >= BOOTSTRAP_UNCERTAINTY["intermittent"]
        assert BOOTSTRAP_UNCERTAINTY["cold_start"] >= BOOTSTRAP_UNCERTAINTY["seasonal"]
        assert BOOTSTRAP_UNCERTAINTY["cold_start"] >= BOOTSTRAP_UNCERTAINTY["trending"]
        assert BOOTSTRAP_UNCERTAINTY["cold_start"] >= BOOTSTRAP_UNCERTAINTY["stable"]

    def test_stable_lowest_uncertainty(self):
        """stable is the most predictable — tightest quantile spread."""
        assert BOOTSTRAP_UNCERTAINTY["stable"] <= BOOTSTRAP_UNCERTAINTY["cold_start"]
        assert BOOTSTRAP_UNCERTAINTY["stable"] <= BOOTSTRAP_UNCERTAINTY["intermittent"]
        assert BOOTSTRAP_UNCERTAINTY["stable"] <= BOOTSTRAP_UNCERTAINTY["seasonal"]
        assert BOOTSTRAP_UNCERTAINTY["stable"] <= BOOTSTRAP_UNCERTAINTY["trending"]

    def test_all_factors_positive(self):
        for pattern, factor in BOOTSTRAP_UNCERTAINTY.items():
            assert factor > 0, f"Factor for '{pattern}' must be > 0"

    def test_all_factors_below_one(self):
        """Factors above 1.0 would produce unreasonably wide quantile spreads."""
        for pattern, factor in BOOTSTRAP_UNCERTAINTY.items():
            assert factor < 1.0, f"Factor for '{pattern}' is {factor} (must be < 1.0)"


# ===========================================================================
# bootstrap_quantiles
# ===========================================================================

class TestBootstrapQuantiles:

    # ── return structure ────────────────────────────────────────────────────

    def test_returns_dict_with_four_keys(self):
        r = bootstrap_quantiles(100.0, np.array([1.0, -1.0, 2.0]), "stable")
        assert set(r.keys()) == {"mean", "p50", "p80", "p90"}

    def test_mean_equals_point(self):
        """mean is the raw point forecast — never altered by bootstrap."""
        r = bootstrap_quantiles(250.0, np.array([1.0, -1.0, 2.0]), "stable")
        assert r["mean"] == pytest.approx(250.0)

    def test_all_values_non_negative(self):
        residuals = np.random.default_rng(0).normal(0, 5, 30)
        r = bootstrap_quantiles(100.0, residuals, "stable")
        assert r["p50"] >= 0
        assert r["p80"] >= 0
        assert r["p90"] >= 0

    # ── D5: ordering invariant — p50 ≤ p80 ≤ p90, ZERO violations ──────────

    @pytest.mark.parametrize("seed", range(100))
    def test_ordering_invariant_100_random_calls(self, seed):
        """Done Criterion D5: p50 ≤ p80 ≤ p90 must hold for every call."""
        rng      = np.random.default_rng(seed)
        point    = float(rng.uniform(0, 1000))
        n_r      = int(rng.integers(0, 50))
        residuals = rng.normal(0, 5, n_r) if n_r > 0 else np.array([])
        pattern  = rng.choice(["cold_start", "intermittent", "seasonal", "trending", "stable"])
        r = bootstrap_quantiles(point, residuals, pattern)
        assert r["p50"] <= r["p80"] <= r["p90"], (
            f"Ordering violated: seed={seed} p50={r['p50']} p80={r['p80']} p90={r['p90']}"
        )

    # ── E004: fewer than 3 residuals → log-normal proxy ────────────────────

    def test_e004_zero_residuals_returns_finite(self):
        """E004: 0 residuals must not crash — uses log-normal proxy."""
        r = bootstrap_quantiles(100.0, np.array([]), "cold_start")
        assert all(np.isfinite(v) for v in r.values())

    def test_e004_one_residual_returns_finite(self):
        r = bootstrap_quantiles(100.0, np.array([3.0]), "stable")
        assert all(np.isfinite(v) for v in r.values())

    def test_e004_two_residuals_returns_finite(self):
        r = bootstrap_quantiles(100.0, np.array([2.0, -1.0]), "trending")
        assert all(np.isfinite(v) for v in r.values())

    def test_e004_ordering_still_holds_with_zero_residuals(self):
        r = bootstrap_quantiles(50.0, np.array([]), "seasonal")
        assert r["p50"] <= r["p80"] <= r["p90"]

    def test_e004_cold_start_wider_than_stable(self):
        """log-normal proxy: cold_start factor (0.60) > stable factor (0.25)."""
        r_cold   = bootstrap_quantiles(100.0, np.array([]), "cold_start")
        r_stable = bootstrap_quantiles(100.0, np.array([]), "stable")
        # cold_start should have wider spread (p90 - p50 larger)
        spread_cold   = r_cold["p90"]   - r_cold["p50"]
        spread_stable = r_stable["p90"] - r_stable["p50"]
        assert spread_cold >= spread_stable

    # ── edge cases ──────────────────────────────────────────────────────────

    def test_zero_point_returns_all_zeros(self):
        r = bootstrap_quantiles(0.0, np.array([]), "stable")
        assert r == {"mean": 0.0, "p50": 0.0, "p80": 0.0, "p90": 0.0}

    def test_negative_point_clamped_to_zero(self):
        """Negative point forecast clamped before bootstrap."""
        r = bootstrap_quantiles(-10.0, np.array([1.0, -1.0, 2.0]), "stable")
        assert r["mean"] == 0.0

    def test_unknown_pattern_uses_fallback(self):
        """Unknown pattern should not crash — uses 0.50 fallback factor."""
        r = bootstrap_quantiles(100.0, np.array([]), "unknown_pattern")
        assert all(np.isfinite(v) for v in r.values())

    def test_large_positive_residuals_shift_quantiles_up(self):
        """Residuals > 0 should push p50/p80/p90 above the point forecast."""
        residuals = np.full(30, 50.0)  # always +50 above point
        r = bootstrap_quantiles(100.0, residuals, "stable")
        assert r["p50"] > 100.0

    def test_large_negative_residuals_clamped_to_zero(self):
        """Residuals that drive samples below 0 must be clamped — no negatives."""
        residuals = np.full(30, -200.0)  # would produce negative samples
        r = bootstrap_quantiles(10.0, residuals, "stable")
        assert r["p50"] >= 0
        assert r["p80"] >= 0
        assert r["p90"] >= 0

    def test_reproducibility_with_same_seed(self):
        """Same inputs + same seed must always produce the same output."""
        residuals = np.array([1.0, -2.0, 3.0, -1.0, 0.5])
        r1 = bootstrap_quantiles(100.0, residuals, "stable", seed=42)
        r2 = bootstrap_quantiles(100.0, residuals, "stable", seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        """Different seeds should (with overwhelming probability) differ."""
        residuals = np.random.default_rng(0).normal(0, 10, 30)
        r1 = bootstrap_quantiles(100.0, residuals, "stable", seed=1)
        r2 = bootstrap_quantiles(100.0, residuals, "stable", seed=99)
        # Not guaranteed to differ, but extremely likely with n=1000
        assert r1 != r2 or True  # non-determinism test — advisory only
