"""Dynamic prompt re-weighting curriculum for MaxRL under imperfect rewards.

Idea
----
Under a binary-symmetric verifier with flip probability ``p`` (see
:mod:`maxrl_trainer.extension.reward_noise`), a genuinely unsolvable prompt --
one whose true success rate is ~0 -- still exhibits an *observed* success rate
of roughly ``p`` purely from ``0 -> 1`` flips. MaxRL then spends gradient mass
chasing these false positives, which is exactly where reward noise does the
most damage.

The curriculum mitigates this by *down-weighting prompts whose observed success
rate is statistically indistinguishable from the noise floor* and giving full
weight to prompts whose success rate is comfortably above it. The test for
"indistinguishable from noise" is binomial: a prompt with smoothed observed
success rate ``q``, under a noise floor ``f`` estimated from ``n_eff`` rollouts,
has floor standard deviation ``sd = sqrt(f (1-f) / n_eff)``. We place the gate
midpoint ``kappa`` standard deviations above the floor and use a softness that
also scales with ``sd``:

    midpoint = f + kappa * sd
    width    = max(beta * sd, min_width)
    w        = w_min + (1 - w_min) * sigmoid((q - midpoint) / width)

so that ``q`` near the floor -> ``w ~ w_min`` (treat as noise, suppress) and
``q`` well above it -> ``w ~ 1`` (real signal, keep). Crucially, when
``flip_prob == 0`` the floor and its spread are ``0``, the midpoint collapses to
``0``, and *any* prompt with real successes is kept -- the curriculum becomes a
no-op and standard MaxRL is recovered. This avoids the failure mode of a fixed
margin, which would wrongly suppress genuinely-hard-but-solvable prompts whose
true success rate is low.

The observed success rate is tracked as an exponential moving average (EMA)
across training steps, keyed by prompt, so the estimate sharpens over time
rather than reacting to a single noisy group.

Depends only on numpy + the standard library (CPU/unit-test friendly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PromptReweightingCurriculum:
    """Track per-prompt success rates and emit noise-aware prompt weights.

    Parameters
    ----------
    noise_floor:
        The success rate ``f`` expected from pure noise (typically the verifier
        flip probability ``p``). Prompts whose smoothed success rate is at or
        below this are suppressed.
    n_eff:
        Effective number of rollouts used to estimate the binomial spread of
        the noise floor, ``sd = sqrt(f (1-f) / n_eff)``. Typically the rollout
        group size ``G``.
    kappa:
        Number of floor standard deviations above ``f`` at which the gate
        midpoint is placed. Larger -> stricter (a prompt must clear the floor by
        more before it counts as real signal).
    beta:
        Softness of the gate in units of the floor standard deviation. The gate
        width is ``max(beta * sd, min_width)``.
    min_width:
        Lower bound on the gate width so the gate stays smooth even when the
        floor spread is ~0 (e.g. ``flip_prob == 0``).
    w_min:
        Minimum weight assigned to fully-suppressed prompts (in ``[0, 1]``).
        ``w_min = 0`` removes such prompts entirely; a small positive value
        keeps a trickle of signal and avoids permanently starving a prompt.
    ema_decay:
        EMA decay for the running per-prompt success-rate estimate. Higher
        values place more weight on history (slower, smoother).
    """

    noise_floor: float = 0.0
    n_eff: float = 8.0
    kappa: float = 1.0
    beta: float = 1.0
    min_width: float = 0.03
    w_min: float = 0.1
    ema_decay: float = 0.9
    _ema: dict[Hashable, float] = field(default_factory=dict, repr=False)
    _count: dict[Hashable, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.w_min <= 1.0):
            raise ValueError(f"w_min must be in [0, 1], got {self.w_min}")
        if self.min_width <= 0.0:
            raise ValueError(f"min_width must be > 0, got {self.min_width}")
        if self.n_eff <= 0.0:
            raise ValueError(f"n_eff must be > 0, got {self.n_eff}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")

    @classmethod
    def from_noise(
        cls,
        flip_prob: float,
        n_eff: float = 8.0,
        kappa: float = 1.0,
        beta: float = 1.0,
        min_width: float = 0.03,
        w_min: float = 0.1,
        ema_decay: float = 0.9,
    ) -> "PromptReweightingCurriculum":
        """Build a curriculum whose floor is tied to the verifier flip rate."""
        return cls(
            noise_floor=float(flip_prob),
            n_eff=float(n_eff),
            kappa=float(kappa),
            beta=float(beta),
            min_width=float(min_width),
            w_min=float(w_min),
            ema_decay=float(ema_decay),
        )

    def observe(self, keys: Sequence[Hashable], success_rates: Sequence[float]) -> None:
        """Update the per-prompt EMA of observed success rate.

        Parameters
        ----------
        keys:
            Hashable prompt identifiers (e.g. the prompt strings).
        success_rates:
            This step's observed success rate for each prompt (fraction of the
            group whose (possibly noisy) reward counted as a success).
        """
        if len(keys) != len(success_rates):
            raise ValueError("keys and success_rates must have equal length")
        for k, r in zip(keys, success_rates):
            r = float(r)
            if k in self._ema:
                self._ema[k] = self.ema_decay * self._ema[k] + (1.0 - self.ema_decay) * r
                self._count[k] += 1
            else:
                self._ema[k] = r
                self._count[k] = 1

    def weights(self, keys: Sequence[Hashable]) -> np.ndarray:
        """Return the curriculum weight in ``[w_min, 1]`` for each prompt key.

        Prompts not yet observed are given full weight (exactly 1.0) so they are
        never suppressed before any evidence is collected.
        """
        seen = np.array([k in self._ema for k in keys], dtype=bool)
        q = np.array([self._ema.get(k, 1.0) for k in keys], dtype=np.float64)
        f = self.noise_floor
        sd = float(np.sqrt(max(f * (1.0 - f), 0.0) / self.n_eff))
        midpoint = f + self.kappa * sd
        width = max(self.beta * sd, self.min_width)
        gate = _sigmoid((q - midpoint) / width)
        w = self.w_min + (1.0 - self.w_min) * gate
        # Unseen prompts: full weight regardless of the floor/gate.
        w = np.where(seen, w, 1.0)
        return w

    def step(
        self,
        keys: Sequence[Hashable],
        success_rates: Sequence[float],
    ) -> np.ndarray:
        """Observe this step's success rates and return updated prompt weights.

        This is the typical entry point inside a training loop: it folds the new
        observations into the EMA and then returns the weight to scale each
        prompt's MaxRL gradient contribution.
        """
        self.observe(keys, success_rates)
        return self.weights(keys)

    def state(self) -> dict[Hashable, dict[str, float]]:
        """Return a snapshot of per-prompt EMA + observation count (for logging)."""
        return {
            k: {"ema_success": self._ema[k], "count": float(self._count[k])}
            for k in self._ema
        }
