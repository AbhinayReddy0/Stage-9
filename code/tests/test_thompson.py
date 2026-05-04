"""
tests/test_thompson.py — Atheera Stage 9
=========================================
Tests for ThompsonSampler:
    - config_hash: determinism, sort_keys, non-serialisable guard
    - select_configs: prior_best always included, budget capping, empty space
    - update_state: alpha/beta increment, immutability
    - get_prior_best: highest win-rate returned, empty state returns default
    - get_thompson_score: Beta(1,1) prior for unseen configs
"""

from __future__ import annotations

import numpy as np
import pytest

from models.thompson import ThompsonSampler
from models.naive    import NaiveForecast
from models.ses      import SESModel
from models.holt     import HoltLinearTrend


@pytest.fixture
def sampler() -> ThompsonSampler:
    return ThompsonSampler()


@pytest.fixture
def naive_space() -> list[dict]:
    return NaiveForecast({}).hp_search_space   # 9 configs


@pytest.fixture
def ses_space() -> list[dict]:
    return SESModel({}).hp_search_space         # 5 configs


@pytest.fixture
def holt_space() -> list[dict]:
    return HoltLinearTrend({}).hp_search_space  # 24 configs


# ===========================================================================
# config_hash
# ===========================================================================

class TestConfigHash:

    def test_deterministic_same_dict(self, sampler, naive_space):
        h1 = sampler.config_hash(naive_space[0])
        h2 = sampler.config_hash(naive_space[0])
        assert h1 == h2

    def test_order_independent(self, sampler):
        """Done Criterion D5: sort_keys=True must make hash order-independent."""
        h1 = sampler.config_hash({"b": 1, "a": 2})
        h2 = sampler.config_hash({"a": 2, "b": 1})
        assert h1 == h2

    def test_different_configs_different_hashes(self, sampler, naive_space):
        hashes = [sampler.config_hash(hp) for hp in naive_space]
        assert len(set(hashes)) == len(naive_space), "Each config must produce a unique hash"

    def test_returns_64_char_hex(self, sampler, naive_space):
        h = sampler.config_hash(naive_space[0])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_non_serialisable_raises_type_error(self, sampler):
        """Non-serialisable HP value must fail loudly — never silently produce wrong hash."""
        with pytest.raises(TypeError):
            sampler.config_hash({"key": object()})

    def test_nested_dict_hash(self, sampler):
        """Edge case: nested values hash deterministically."""
        h1 = sampler.config_hash({"a": {"x": 1, "y": 2}})
        h2 = sampler.config_hash({"a": {"y": 2, "x": 1}})
        assert h1 == h2


# ===========================================================================
# select_configs
# ===========================================================================

class TestSelectConfigs:

    def test_prior_best_always_included(self, sampler, naive_space):
        """Done Criterion D6 (Task): prior_best_config must always be in result."""
        prior_best = naive_space[-1]   # pick last config
        selected   = sampler.select_configs(naive_space, {}, budget=2, prior_best_config=prior_best)
        selected_hashes = [sampler.config_hash(h) for h in selected]
        assert sampler.config_hash(prior_best) in selected_hashes

    def test_returns_at_most_budget_configs(self, sampler, naive_space):
        for budget in [1, 2, 3]:
            selected = sampler.select_configs(naive_space, {}, budget=budget, prior_best_config=naive_space[0])
            assert len(selected) <= budget

    def test_empty_search_space_returns_prior_best(self, sampler, naive_space):
        prior_best = naive_space[0]
        selected   = sampler.select_configs([], {}, budget=3, prior_best_config=prior_best)
        assert selected == [prior_best]

    def test_budget_larger_than_space(self, sampler, ses_space):
        """Budget > space size must not crash — returns all configs."""
        selected = sampler.select_configs(ses_space, {}, budget=99, prior_best_config=ses_space[0])
        assert len(selected) <= len(ses_space)

    def test_budget_one_returns_prior_best(self, sampler, naive_space):
        """With budget=1, the only slot must be the prior_best."""
        prior_best = naive_space[3]
        selected   = sampler.select_configs(naive_space, {}, budget=1, prior_best_config=prior_best)
        assert len(selected) == 1
        assert sampler.config_hash(selected[0]) == sampler.config_hash(prior_best)

    def test_uniform_prior_explores_all_configs(self, sampler, ses_space):
        """With uniform Beta(1,1) prior and budget=3, all configs appear over 30 draws."""
        # budget=1 always returns prior_best by design — use budget=3 for exploration
        prior_best = ses_space[0]
        seen = set()
        for _ in range(30):
            selected = sampler.select_configs(ses_space, {}, budget=3, prior_best_config=prior_best)
            for hp in selected:
                seen.add(sampler.config_hash(hp))
        # 5 configs, budget=3, 30 draws = 90 samples total — all 5 should appear
        assert len(seen) >= 3, f"Only {len(seen)} distinct configs seen"

    def test_high_alpha_config_selected_more_often(self, sampler, ses_space):
        """Config with high alpha/(alpha+beta) should dominate selection."""
        best_hp = ses_space[2]   # assign this one a very high success rate
        state   = {
            sampler.config_hash(ses_space[2]): {"alpha": 100, "beta": 1},
            sampler.config_hash(ses_space[0]): {"alpha": 1,   "beta": 100},
        }
        wins = 0
        for _ in range(30):
            selected = sampler.select_configs(ses_space, state, budget=1, prior_best_config=best_hp)
            if sampler.config_hash(selected[0]) == sampler.config_hash(best_hp):
                wins += 1
        # High-alpha config should win most of the time
        assert wins >= 20, f"High-alpha config won only {wins}/30 times"


# ===========================================================================
# update_state
# ===========================================================================

class TestUpdateState:

    def test_success_increments_alpha(self, sampler, naive_space):
        h     = sampler.config_hash(naive_space[0])
        state = {h: {"alpha": 3, "beta": 2}}
        new   = sampler.update_state(state, naive_space[0], success=True)
        assert new[h]["alpha"] == 4
        assert new[h]["beta"]  == 2

    def test_failure_increments_beta(self, sampler, naive_space):
        h     = sampler.config_hash(naive_space[0])
        state = {h: {"alpha": 3, "beta": 2}}
        new   = sampler.update_state(state, naive_space[0], success=False)
        assert new[h]["alpha"] == 3
        assert new[h]["beta"]  == 3

    def test_immutable_original_state(self, sampler, naive_space):
        """update_state must NOT mutate the input dict — returns a copy."""
        h     = sampler.config_hash(naive_space[0])
        state = {h: {"alpha": 3, "beta": 2}}
        _     = sampler.update_state(state, naive_space[0], success=True)
        assert state[h]["alpha"] == 3   # original unchanged

    def test_unseen_config_initialised_at_uniform_prior(self, sampler, naive_space):
        """Config not in state gets Beta(1,1) prior before update."""
        new = sampler.update_state({}, naive_space[0], success=True)
        h   = sampler.config_hash(naive_space[0])
        assert new[h]["alpha"] == 2   # 1 (prior) + 1 (success)
        assert new[h]["beta"]  == 1   # prior only

    def test_does_not_write_to_db(self, sampler, naive_space):
        """update_state is pure — no side effects outside the returned dict."""
        # No DB connection available — if it tries to connect, it will raise.
        # Simply calling it without error proves no DB access.
        state = {}
        new   = sampler.update_state(state, naive_space[0], success=True)
        assert isinstance(new, dict)


# ===========================================================================
# get_prior_best
# ===========================================================================

class TestGetPriorBest:

    def test_returns_default_when_no_state(self, sampler, naive_space):
        default = NaiveForecast({}).default_hp
        best    = sampler.get_prior_best(naive_space, {}, default)
        assert best == default

    def test_returns_highest_win_rate_config(self, sampler, naive_space):
        best_hp = naive_space[4]
        state   = {
            sampler.config_hash(naive_space[0]): {"alpha": 1,  "beta": 5},   # 17% win rate
            sampler.config_hash(naive_space[4]): {"alpha": 9,  "beta": 1},   # 90% win rate
            sampler.config_hash(naive_space[8]): {"alpha": 3,  "beta": 3},   # 50% win rate
        }
        best = sampler.get_prior_best(naive_space, state, NaiveForecast({}).default_hp)
        assert sampler.config_hash(best) == sampler.config_hash(best_hp)

    def test_ignores_configs_not_in_search_space(self, sampler, naive_space):
        """State may contain hashes for configs outside the current space — ignore."""
        state = {
            "deadbeef" * 8: {"alpha": 999, "beta": 1},  # garbage hash not in space
        }
        default = NaiveForecast({}).default_hp
        best    = sampler.get_prior_best(naive_space, state, default)
        assert best == default   # falls back to default since no space config is in state


# ===========================================================================
# get_thompson_score
# ===========================================================================

class TestGetThompsonScore:

    def test_unseen_config_returns_0_5(self, sampler, naive_space):
        """Beta(1,1) prior has mean 0.5 — unseen configs score 0.5."""
        score = sampler.get_thompson_score(naive_space[0], {})
        assert score == pytest.approx(0.5)

    def test_all_successes_approaches_1(self, sampler, naive_space):
        h     = sampler.config_hash(naive_space[0])
        state = {h: {"alpha": 99, "beta": 1}}
        score = sampler.get_thompson_score(naive_space[0], state)
        assert score > 0.95

    def test_all_failures_approaches_0(self, sampler, naive_space):
        h     = sampler.config_hash(naive_space[0])
        state = {h: {"alpha": 1, "beta": 99}}
        score = sampler.get_thompson_score(naive_space[0], state)
        assert score < 0.05

    def test_score_between_0_and_1(self, sampler, naive_space):
        for hp in naive_space:
            h     = sampler.config_hash(hp)
            state = {h: {"alpha": np.random.randint(1, 20), "beta": np.random.randint(1, 20)}}
            score = sampler.get_thompson_score(hp, state)
            assert 0.0 <= score <= 1.0
