"""Symmetric reward-flip noise model for the MaxRL imperfect-reward study.

Motivation
----------
The Countdown verifier (``evaluation.countdown.compute_score``) is, by
construction, an *exact* checker: it returns ``1.0`` only when the predicted
equation evaluates to the target using each provided number exactly once. Real
reward models / verifiers are not exact -- they mislabel a fraction of
trajectories. To characterise how MaxRL behaves under such imperfect rewards we
inject *controlled* label noise into the otherwise-exact verifier output.

We model the verifier as a binary symmetric channel acting on the *success
indicator* ``s = 1[reward > success_threshold]``:

    with probability ``p``   ->  flip the success bit (s -> 1 - s)
    with probability ``1-p`` ->  keep the bit

A flipped success becomes a (false) failure and a flipped failure becomes a
(false) success. The flip is *symmetric*: the same per-sample probability ``p``
governs both 0->1 and 1->0 transitions, matching the proposal's "symmetric
reward-flip noise" specification with ``p in {0, 0.001, 0.01, 0.1}``.

Why this matters for MaxRL
--------------------------
MaxRL places gradient weight ``1 / S`` on every successful rollout in a group
(``S`` = number of successes). A single *false positive* on a genuinely hard
prompt (true ``S = 0``) therefore receives weight ``1.0`` -- the maximum
possible -- so the estimator is structurally fragile to ``0 -> 1`` flips. This
module lets us quantify that fragility and test mitigations (RLAIF, curriculum).

The model is deterministic given its seed so experiments are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def binarize_rewards(rewards: np.ndarray, success_threshold: float = 0.5) -> np.ndarray:
    """Return the binary success indicator ``1[reward > success_threshold]``.

    Countdown rewards live in ``{0.0, 0.1, 1.0}``; the default threshold of
    ``0.5`` keeps only fully-correct rollouts as successes (matching the MaxRL
    update worker's ``success_threshold`` convention).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    return (rewards > float(success_threshold)).astype(np.float64)


@dataclass
class SymmetricRewardFlipNoise:
    """Binary-symmetric-channel corruption of a verifier's success labels.

    Parameters
    ----------
    flip_prob:
        Per-sample probability ``p`` of flipping the success bit. ``p = 0``
        is a no-op (the model returns the clean rewards unchanged).
    success_threshold:
        Threshold used to binarise rewards into success/failure before
        flipping.
    success_value, failure_value:
        Reward values written back after a flip. A bit flipped *to* success
        becomes ``success_value`` (default ``1.0``); a bit flipped *to* failure
        becomes ``failure_value`` (default ``0.0``). Non-flipped rewards are
        passed through unchanged so that partial-format rewards (``0.1``) are
        preserved when the success bit does not change.
    seed:
        Base seed for the internal :class:`numpy.random.Generator`.
    """

    flip_prob: float = 0.0
    success_threshold: float = 0.5
    success_value: float = 1.0
    failure_value: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.flip_prob) <= 1.0):
            raise ValueError(f"flip_prob must be in [0, 1], got {self.flip_prob}")
        self._rng = np.random.default_rng(int(self.seed))

    def reseed(self, seed: int) -> None:
        """Reset the RNG (e.g. to make a particular training step reproducible)."""
        self._rng = np.random.default_rng(int(seed))

    def corrupt(
        self,
        rewards: np.ndarray,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply symmetric flip noise to a flat array of rewards.

        Parameters
        ----------
        rewards:
            1-D array of scalar rewards (already computed by the verifier).
        step:
            Optional training-step index. When provided, the RNG is reseeded
            deterministically as ``seed * 1_000_003 + step`` so that re-running
            a given step reproduces the same flips while different steps see
            independent noise.

        Returns
        -------
        noisy_rewards:
            Array of the same shape as ``rewards`` with corrupted values.
        flip_mask:
            Boolean array marking which entries were flipped.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        if step is not None:
            self.reseed(int(self.seed) * 1_000_003 + int(step))

        if self.flip_prob <= 0.0:
            return rewards.copy(), np.zeros_like(rewards, dtype=bool)

        clean_success = rewards > float(self.success_threshold)
        flip_mask = self._rng.random(rewards.shape) < float(self.flip_prob)
        noisy_success = np.logical_xor(clean_success, flip_mask)

        noisy_rewards = rewards.copy()
        # Only rewrite entries whose success bit actually changed; this keeps
        # the partial-format reward (0.1) intact whenever it is not flipped.
        became_success = flip_mask & ~clean_success
        became_failure = flip_mask & clean_success
        noisy_rewards[became_success] = float(self.success_value)
        noisy_rewards[became_failure] = float(self.failure_value)
        return noisy_rewards, flip_mask

    def corrupt_groups(
        self,
        grouped_rewards: Sequence[Sequence[float]],
        step: int | None = None,
    ) -> tuple[list[list[float]], list[list[bool]]]:
        """Convenience wrapper for prompt-major nested reward lists.

        ``grouped_rewards`` is a list (per prompt) of lists (per rollout). The
        flip noise is applied jointly across the flattened batch so that the
        per-sample flip probability is exactly ``flip_prob`` regardless of how
        the rollouts are grouped.
        """
        lengths = [len(g) for g in grouped_rewards]
        flat = np.asarray([r for g in grouped_rewards for r in g], dtype=np.float64)
        noisy_flat, mask_flat = self.corrupt(flat, step=step)

        noisy_groups: list[list[float]] = []
        mask_groups: list[list[bool]] = []
        idx = 0
        for n in lengths:
            noisy_groups.append([float(v) for v in noisy_flat[idx : idx + n]])
            mask_groups.append([bool(v) for v in mask_flat[idx : idx + n]])
            idx += n
        return noisy_groups, mask_groups
