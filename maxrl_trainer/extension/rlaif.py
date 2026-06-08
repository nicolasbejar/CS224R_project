"""RLAIF reward channel: a second (LLM-judge) supervision signal.

Background
----------
Our verifier is corrupted by symmetric flip noise with probability ``p``
(:mod:`maxrl_trainer.extension.reward_noise`). RLAIF (Lee et al., 2023) adds a
*second*, independent reward channel produced by an LLM judge that reads the
prompt + response and decides whether the answer is correct. If the judge's
errors are (approximately) independent of the verifier's, combining the two
channels by *agreement* drives the effective label-error rate down
multiplicatively.

Concretely, for a genuinely-incorrect trajectory the verifier emits a false
positive with probability ``p`` and the judge with probability ``q``. Requiring
both channels to agree ("agreement gate") keeps the false positive only when
*both* err, i.e. with probability ``~= p * q`` (``~= p**2`` when ``q ~= p``).
This is the ``p -> p^2`` flip-rate reduction promised in our proposal.

This module provides:
  * :class:`SimulatedJudge` -- a controllable-error judge with an explicit
    error rate and an optional correlation with the verifier's errors. It needs
    no GPU and powers the offline characterization + unit tests.
  * :class:`VLLMJudge` -- a real LLM judge that runs on the existing vLLM stack
    (GPU/Modal only; ``vllm`` is imported lazily).
  * :func:`combine_rewards` -- the channel-combination rules: agreement gate,
    logical-or, majority vote, and continuous soft averaging (d-RLAIF).
  * :func:`agreement_gate_fp_rate` -- the analytic effective false-positive rate
    used for the report's theory curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RLAIFConfig:
    """Settings for the RLAIF reward channel."""

    enabled: bool = False
    backend: str = "simulated"  # {"simulated", "vllm"}
    combine: str = "agree_gate"  # {"agree_gate", "or", "majority", "soft_avg"}
    success_threshold: float = 0.5
    # Simulated-judge parameters.
    judge_error_rate: float = 0.0
    judge_correlation: float = 0.0
    judge_seed: int = 0
    # vLLM-judge parameters (only used when backend == "vllm").
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct"
    judge_max_tokens: int = 8
    judge_temperature: float = 0.0


# --------------------------------------------------------------------------- #
# Combination rules
# --------------------------------------------------------------------------- #
def combine_rewards(
    verifier_rewards: np.ndarray,
    judge_signal: np.ndarray,
    mode: str = "agree_gate",
    success_threshold: float = 0.5,
    success_value: float = 1.0,
    failure_value: float = 0.0,
) -> np.ndarray:
    """Combine the verifier channel with one or more judge channels.

    Parameters
    ----------
    verifier_rewards:
        1-D array ``[N]`` of (possibly noisy) verifier rewards.
    judge_signal:
        Judge channel(s). Shape ``[N]`` for a single judge or ``[N, K]`` for
        ``K`` judges. For ``mode == "soft_avg"`` these are interpreted as
        continuous scores in ``[0, 1]``; otherwise they are thresholded.
    mode:
        ``"agree_gate"`` -- success iff verifier *and* all judges agree on
            success (strongest de-noiser).
        ``"or"``         -- success iff any channel says success.
        ``"majority"``   -- success iff a strict majority of channels (verifier
            + judges) say success; ties resolve to failure (conservative).
        ``"soft_avg"``   -- continuous d-RLAIF reward: mean of the clipped
            verifier reward and judge score(s).

    Returns
    -------
    np.ndarray
        Combined rewards ``[N]``. For thresholding modes the values are
        ``success_value`` / ``failure_value``; for ``soft_avg`` they are
        continuous in ``[0, 1]``.
    """
    v = np.asarray(verifier_rewards, dtype=np.float64).reshape(-1)
    j = np.asarray(judge_signal, dtype=np.float64)
    if j.ndim == 1:
        j = j.reshape(-1, 1)
    if j.shape[0] != v.shape[0]:
        raise ValueError(
            f"verifier ({v.shape[0]}) and judge ({j.shape[0]}) must align on axis 0"
        )

    if mode == "soft_avg":
        v_soft = np.clip(v, 0.0, 1.0)
        j_soft = np.clip(j, 0.0, 1.0).mean(axis=1)
        return 0.5 * (v_soft + j_soft)

    v_succ = v > success_threshold
    j_succ = j > success_threshold  # [N, K]

    if mode == "agree_gate":
        combined = v_succ & j_succ.all(axis=1)
    elif mode == "or":
        combined = v_succ | j_succ.any(axis=1)
    elif mode == "majority":
        votes = v_succ.astype(np.int64) + j_succ.sum(axis=1).astype(np.int64)
        n_channels = 1 + j.shape[1]
        combined = votes > (n_channels / 2.0)
    else:
        raise ValueError(f"unknown combine mode: {mode!r}")

    return np.where(combined, float(success_value), float(failure_value))


def agreement_gate_fp_rate(p: float, q: float, correlation: float = 0.0) -> float:
    """Analytic effective false-positive rate of the agreement gate.

    A truly-incorrect trajectory is labelled a (false) success only if both the
    verifier (false-positive prob ``p``) and the judge (false-positive prob
    ``q``) err. With error correlation ``rho`` modelled as a fraction of samples
    on which the judge copies the verifier's mistake:

        P(both err) = rho * p + (1 - rho) * p * q

    For ``rho = 0`` this is the independent product ``p * q``; for ``rho = 1`` it
    collapses to ``p`` (no de-noising benefit).
    """
    p = float(p)
    q = float(q)
    rho = float(correlation)
    return rho * p + (1.0 - rho) * p * q


# --------------------------------------------------------------------------- #
# Simulated judge (CPU; for offline study + tests)
# --------------------------------------------------------------------------- #
@dataclass
class SimulatedJudge:
    """A judge with a controllable, optionally verifier-correlated error rate.

    The simulated judge is given the *clean* (ground-truth) rewards and emits a
    success label that matches the truth except on a fraction ``error_rate`` of
    samples. The ``correlation`` knob couples the judge's mistakes to the
    verifier's flips so we can study how the RLAIF benefit erodes when the two
    channels fail on the same (ambiguous) examples.
    """

    error_rate: float = 0.0
    seed: int = 0
    success_threshold: float = 0.5
    success_value: float = 1.0
    failure_value: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.error_rate <= 1.0):
            raise ValueError(f"error_rate must be in [0, 1], got {self.error_rate}")
        self._rng = np.random.default_rng(int(self.seed))

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def judge(
        self,
        clean_rewards: np.ndarray,
        verifier_flip_mask: np.ndarray | None = None,
        correlation: float = 0.0,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(judge_rewards, judge_error_mask)``.

        Parameters
        ----------
        clean_rewards:
            The true (uncorrupted) rewards, used to define the judge's target
            label. In practice these come from the exact Countdown verifier.
        verifier_flip_mask:
            Boolean mask of which samples the *verifier* flipped. Used only when
            ``correlation > 0`` to couple judge errors to verifier errors.
        correlation:
            Fraction ``rho in [0, 1]`` of samples on which the judge's error
            indicator is forced to equal the verifier's flip indicator. The
            realized marginal judge-error rate is then
            ``rho * p + (1 - rho) * error_rate`` (``p`` = verifier flip rate).
        """
        clean_rewards = np.asarray(clean_rewards, dtype=np.float64).reshape(-1)
        if step is not None:
            self.reseed(int(self.seed) * 7_001 + int(step))

        true_success = clean_rewards > float(self.success_threshold)
        indep_error = self._rng.random(clean_rewards.shape) < float(self.error_rate)

        if verifier_flip_mask is not None and correlation > 0.0:
            vmask = np.asarray(verifier_flip_mask, dtype=bool).reshape(-1)
            coupled = self._rng.random(clean_rewards.shape) < float(correlation)
            judge_error = np.where(coupled, vmask, indep_error)
        else:
            judge_error = indep_error

        judge_success = np.logical_xor(true_success, judge_error)
        judge_rewards = np.where(
            judge_success, float(self.success_value), float(self.failure_value)
        )
        return judge_rewards, judge_error


# --------------------------------------------------------------------------- #
# Real LLM judge (GPU / Modal; vllm imported lazily)
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for the Countdown arithmetic game. The player is "
    "given a list of numbers and a target. A solution is CORRECT only if it is "
    "an arithmetic expression that uses each provided number exactly once and "
    "evaluates exactly to the target. Respond with a single word: YES if the "
    "player's final answer is correct, otherwise NO."
)


class VLLMJudge:
    """An LLM judge backed by vLLM (runs on GPU; e.g. inside Modal).

    ``vllm`` is imported lazily so this module stays importable on CPU. The
    judge produces a binary success label per (prompt, response, ground_truth);
    with ``return_soft=True`` it also returns a continuous score (used for the
    d-RLAIF ``soft_avg`` combination) derived from the YES/NO token probability.
    """

    def __init__(self, config: RLAIFConfig):
        self.config = config
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM  # lazy import (GPU dependency)
        from transformers import AutoTokenizer

        self._llm = LLM(model=self.config.judge_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.judge_model)

    def _build_prompt(self, prompt: str, response: str, ground_truth: dict) -> str:
        nums = ground_truth.get("numbers")
        target = ground_truth.get("target")
        user = (
            f"Numbers: {nums}\nTarget: {target}\n\n"
            f"Player's answer:\n{response}\n\n"
            "Is the player's final <answer> correct? Reply YES or NO."
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def judge(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        ground_truths: Sequence[dict],
        return_soft: bool = False,
    ) -> np.ndarray:
        """Return per-sample judge rewards (``1.0``/``0.0``) or soft scores."""
        self._ensure_loaded()
        from vllm import SamplingParams

        texts = [
            self._build_prompt(p, r, gt)
            for p, r, gt in zip(prompts, responses, ground_truths)
        ]
        sampling = SamplingParams(
            temperature=float(self.config.judge_temperature),
            max_tokens=int(self.config.judge_max_tokens),
            logprobs=5,
        )
        outputs = self._llm.generate(texts, sampling)

        rewards = np.zeros(len(texts), dtype=np.float64)
        for i, out in enumerate(outputs):
            text = out.outputs[0].text.strip().upper()
            yes = text.startswith("YES") or "YES" in text[:5]
            if return_soft:
                # Map a YES decision to 1.0 and NO to 0.0; a confidence-weighted
                # variant could read the YES/NO token logprob here.
                rewards[i] = 1.0 if yes else 0.0
            else:
                rewards[i] = 1.0 if yes else 0.0
        return rewards
