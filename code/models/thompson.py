"""
thompson.py — Atheera Stage 9 Forecasting Agent
================================================
ThompsonSampler: Bayesian bandit for hyperparameter selection.

Thompson Sampling maintains a Beta(alpha, beta) distribution for each
HP configuration. Each run, it samples from those distributions and tests
the highest-sampled configs. Configs that outperform the baseline earn
alpha += 1 (success); configs that don't earn beta += 1 (failure).
Over many runs, Thompson converges to the best-performing config.

Used by: Sub-Stage 9.3 (HP Tuning). Instantiated once per SKU.
Written to DB: NOT here. The LEARNING state flushes thompson_sampling_state
in bulk after all SKUs complete. Per-SKU writes here would produce ~40K
inserts per run — prohibited by architecture.

Critical Rules:
    - sort_keys=True in ALL config_hash() calls — deterministic hashing
    - Prior best config is ALWAYS included in the test set (even if not top-N)
    - Thompson state updated in memory only inside Sub-Stage 9.3

"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from infrastructure.constants import THOMPSON_ALPHA_INIT, THOMPSON_BETA_INIT

__all__ = ["ThompsonSampler"]

_ALPHA_INIT = int(THOMPSON_ALPHA_INIT)
_BETA_INIT  = int(THOMPSON_BETA_INIT)

log = logging.getLogger(__name__)


class ThompsonSampler:
    """
    Bayesian bandit that selects HP configurations to test per run.

    State format (thompson_sampling_state table, loaded in PRELOADING):
        {config_hash: {'alpha': int, 'beta': int}}

    Each config starts with alpha=1, beta=1 (uniform Beta prior) when no
    prior evidence exists. This ensures every config has a non-zero chance
    of being selected on the first run — exploration is guaranteed.

    Thread-safe: all methods are pure functions on the state dict (no
    class-level mutable state). Re-instantiate per SKU in Sub-Stage 9.3.
    ProcessPool safe: top-level class, no closures, no lambdas.
    """

    # -----------------------------------------------------------------------
    # Core interface used by Sub-Stage 9.3
    # -----------------------------------------------------------------------

    def select_configs(
        self,
        hp_search_space: list[dict],
        thompson_state:  dict,
        budget:          int,
        prior_best_config: dict,
    ) -> list[dict]:
        """
        Select up to `budget` HP configurations to test in this run.

        Algorithm:
            1. For each config, sample theta ~ Beta(alpha, beta).
               Configs never seen before get Beta(1, 1) — uniform, so theta ~ U(0,1).
            2. Sort configs by sampled theta descending (highest UCB first).
            3. Take top `budget` configs.
            4. ALWAYS include prior_best_config even if it didn't rank in the top budget.
               If already present: no change. If not: replace the last slot.

        Args:
            hp_search_space:   Complete list of HP dicts for the assigned model.
            thompson_state:    {config_hash: {'alpha': int, 'beta': int}} from preloaded.
                               Missing configs get (alpha=1, beta=1) uniform prior.
            budget:            Max configs to test (from TenantParams 'thompson_exploration_budget').
            prior_best_config: Config with the highest alpha/(alpha+beta) ratio from prior runs.
                               Used as the "safe bet" — always included in the test set.

        Returns:
            List of HP config dicts, length ≤ budget.
            Always contains prior_best_config.
            Order: highest sampled theta first.
        """
        if not hp_search_space:
            log.warning("ThompsonSampler.select_configs: empty hp_search_space — returning default")
            return [prior_best_config]

        budget = max(1, int(budget))

        # Sample theta ~ Beta(alpha, beta) for every config in the search space.
        # Reproducibility note: we use numpy global RNG here (not a seeded Generator)
        # because Thompson Sampling benefits from stochastic exploration across runs.
        # Using a fixed seed would cause the same configs to always be selected first.
        sampled: list[tuple[dict, float]] = []
        for hp in hp_search_space:
            h     = self.config_hash(hp)
            state = thompson_state.get(h, {"alpha": _ALPHA_INIT, "beta": _BETA_INIT})
            a     = max(1, int(state.get("alpha", _ALPHA_INIT)))   # guard: alpha ≥ 1
            b     = max(1, int(state.get("beta",  _BETA_INIT)))   # guard: beta  ≥ 1
            theta = float(np.random.beta(a, b))
            sampled.append((hp, theta))

        # Sort descending by sampled theta — highest expected performance first
        sampled.sort(key=lambda x: x[1], reverse=True)
        selected = [hp for hp, _ in sampled[:budget]]

        # Guarantee prior_best_config is always included (Task §9.3 PLAN step)
        prior_hash = self.config_hash(prior_best_config)
        selected_hashes = [self.config_hash(hp) for hp in selected]

        if prior_hash not in selected_hashes:
            if len(selected) >= budget:
                # Replace the last (lowest-theta) slot with the prior best
                selected[-1] = prior_best_config
                log.debug(
                    "ThompsonSampler: prior_best_config not in top-%d — replaced last slot",
                    budget,
                )
            else:
                selected.append(prior_best_config)

        return selected

    def update_state(
        self,
        thompson_state: dict,
        hp_config:      dict,
        success:        bool,
    ) -> dict:
        """
        Update Beta distribution for one HP config after evaluation.

        Success  (mape ≤ baseline × 0.98):  alpha += 1
        Failure  (mape >  baseline × 0.98):  beta  += 1

        This is a pure function: returns a new dict copy. Does NOT modify
        thompson_state in-place. Caller replaces preloaded['thompson_state']
        with the returned value.

        Does NOT write to DB. The LEARNING state flushes all updates in bulk.

        Args:
            thompson_state: Current state dict {config_hash: {alpha, beta}}.
            hp_config:      The HP dict that was just evaluated.
            success:        True if mape ≤ baseline × 0.98 (Task §9.3 LEARN).

        Returns:
            Updated state dict (shallow copy with one entry modified).
        """
        h           = self.config_hash(hp_config)
        state_copy  = dict(thompson_state)          # shallow copy — top-level keys
        prior_entry = state_copy.get(h, {"alpha": _ALPHA_INIT, "beta": _BETA_INIT})

        # Defensive copy of the entry itself before mutating
        updated_entry = {
            "alpha": int(prior_entry.get("alpha", _ALPHA_INIT)),
            "beta":  int(prior_entry.get("beta",  _BETA_INIT)),
        }

        if success:
            updated_entry["alpha"] += 1
        else:
            updated_entry["beta"]  += 1

        state_copy[h] = updated_entry

        log.debug(
            "ThompsonSampler.update_state hash=%s... success=%s "
            "alpha=%d beta=%d",
            h[:8], success,
            updated_entry["alpha"], updated_entry["beta"],
        )
        return state_copy

    # -----------------------------------------------------------------------
    # Utility — used throughout Sub-Stage 9.3
    # -----------------------------------------------------------------------

    @staticmethod
    def config_hash(hp: dict) -> str:
        """
        Compute a deterministic SHA-256 hex digest for an HP configuration dict.

        CRITICAL: sort_keys=True is MANDATORY (Task §9.3 Critical Rules, D5).
        Python dict insertion order is non-deterministic across runs and
        interpreter versions. Without sort_keys, {'a': 1, 'b': 2} and
        {'b': 2, 'a': 1} produce different hashes, breaking Thompson convergence.

        Args:
            hp: Hyperparameter dict. All values must be JSON-serialisable
                (int, float, str, bool). Never contains None.

        Returns:
            64-character hexadecimal SHA-256 digest.
        """
        # json.dumps with sort_keys=True produces a canonical string regardless
        # of the order in which keys were inserted into the dict.
        canonical = json.dumps(hp, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -----------------------------------------------------------------------
    # Helper: find the prior best config from an existing Thompson state
    # -----------------------------------------------------------------------

    def get_prior_best(
        self,
        hp_search_space: list[dict],
        thompson_state:  dict,
        default_hp:      dict,
    ) -> dict:
        """
        Return the config with the highest alpha/(alpha+beta) win rate.

        Called at the start of Sub-Stage 9.3 (PERCEIVE step) to identify the
        incumbent best config before sampling. If no prior state exists for any
        config, returns default_hp.

        Args:
            hp_search_space: All candidate HP dicts for the model.
            thompson_state:  {config_hash: {alpha, beta}} from preloaded data.
            default_hp:      Model's default config — used when no state exists.

        Returns:
            HP dict with the highest observed success rate, or default_hp.
        """
        best_hp    = default_hp
        best_rate  = -1.0

        for hp in hp_search_space:
            h     = self.config_hash(hp)
            entry = thompson_state.get(h)
            if entry is None:
                continue
            a     = int(entry.get("alpha", _ALPHA_INIT))
            b     = int(entry.get("beta",  _BETA_INIT))
            rate  = a / (a + b)
            if rate > best_rate:
                best_rate = rate
                best_hp   = hp

        if best_rate < 0:
            log.debug(
                "ThompsonSampler.get_prior_best: no prior state found — "
                "returning default_hp"
            )
        return best_hp

    def get_thompson_score(
        self,
        hp_config:      dict,
        thompson_state: dict,
    ) -> float:
        """
        Return the current alpha/(alpha+beta) win-rate score for one HP config.

        Written into hyperparameter_decisions.thompson_score (Task §9.3 Output).
        Returns 0.5 for configs with no prior state (Beta(1,1) prior has mean 0.5).
        """
        h     = self.config_hash(hp_config)
        entry = thompson_state.get(h, {"alpha": _ALPHA_INIT, "beta": _BETA_INIT})
        a     = int(entry.get("alpha", _ALPHA_INIT))
        b     = int(entry.get("beta",  _BETA_INIT))
        return a / (a + b)
