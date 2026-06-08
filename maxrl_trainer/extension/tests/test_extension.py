"""Unit + statistical tests for the MaxRL imperfect-reward extension.

Runnable two ways:

    pytest maxrl_trainer/extension/tests/test_extension.py        # if pytest present
    python  maxrl_trainer/extension/tests/test_extension.py       # standalone runner

Only depends on numpy so it can be validated on CPU without torch/vllm/pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the project root importable when run as a standalone script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maxrl_trainer.extension.reward_noise import (  # noqa: E402
    SymmetricRewardFlipNoise,
    binarize_rewards,
)
from maxrl_trainer.extension.curriculum import PromptReweightingCurriculum  # noqa: E402
from maxrl_trainer.extension.rlaif import (  # noqa: E402
    SimulatedJudge,
    combine_rewards,
    agreement_gate_fp_rate,
)


# --------------------------------------------------------------------------- #
# reward_noise
# --------------------------------------------------------------------------- #
def test_zero_noise_is_identity():
    rewards = np.array([0.0, 0.1, 1.0, 1.0, 0.0])
    noise = SymmetricRewardFlipNoise(flip_prob=0.0, seed=0)
    noisy, mask = noise.corrupt(rewards)
    assert np.allclose(noisy, rewards)
    assert not mask.any()


def test_binarize():
    rewards = np.array([0.0, 0.1, 0.5, 1.0])
    b = binarize_rewards(rewards, success_threshold=0.5)
    assert b.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_flip_rate_matches_p_statistically():
    rng_rewards = np.zeros(200_000)  # all true-failures
    noise = SymmetricRewardFlipNoise(flip_prob=0.1, seed=123)
    noisy, mask = noise.corrupt(rng_rewards)
    observed = mask.mean()
    assert abs(observed - 0.1) < 0.005, observed
    # All flips on true-failures must become the success value (1.0).
    assert np.allclose(noisy[mask], 1.0)
    assert np.allclose(noisy[~mask], 0.0)


def test_flip_is_symmetric_and_reproducible():
    rewards = np.concatenate([np.ones(100_000), np.zeros(100_000)])
    n1 = SymmetricRewardFlipNoise(flip_prob=0.05, seed=7)
    n2 = SymmetricRewardFlipNoise(flip_prob=0.05, seed=7)
    a, ma = n1.corrupt(rewards, step=3)
    b, mb = n2.corrupt(rewards, step=3)
    assert np.array_equal(a, b) and np.array_equal(ma, mb)  # reproducible
    # symmetric: flip rate on successes ~= flip rate on failures ~= p
    succ, fail = rewards > 0.5, rewards <= 0.5
    assert abs(ma[succ].mean() - 0.05) < 0.01
    assert abs(ma[fail].mean() - 0.05) < 0.01


def test_corrupt_groups_preserves_shape():
    groups = [[0.0, 1.0, 0.1], [1.0, 1.0], [0.0]]
    noise = SymmetricRewardFlipNoise(flip_prob=0.5, seed=1)
    noisy, mask = noise.corrupt_groups(groups)
    assert [len(g) for g in noisy] == [3, 2, 1]
    assert [len(m) for m in mask] == [3, 2, 1]


# --------------------------------------------------------------------------- #
# curriculum
# --------------------------------------------------------------------------- #
def test_curriculum_suppresses_noise_floor_prompts():
    # Floor p=0.1. Prompt A sits at the floor (pure noise); prompt B well above.
    cur = PromptReweightingCurriculum.from_noise(flip_prob=0.1, n_eff=8, w_min=0.1)
    for _ in range(20):
        cur.observe(["A", "B"], [0.1, 0.6])
    w = cur.weights(["A", "B"])
    assert w[0] < 0.4, w  # noise-floor prompt strongly suppressed
    assert w[1] > 0.9, w  # clearly-solvable prompt kept
    assert w[1] - w[0] > 0.5, w  # large gap between noise and signal prompts


def test_curriculum_unseen_prompt_full_weight():
    cur = PromptReweightingCurriculum.from_noise(flip_prob=0.1)
    w = cur.weights(["never_seen"])
    assert w[0] == 1.0


def test_curriculum_zero_noise_keeps_solvable():
    # With no noise the floor and its spread vanish: a hard-but-real prompt
    # (low success rate) must still be kept -> curriculum is a no-op.
    cur = PromptReweightingCurriculum.from_noise(flip_prob=0.0, w_min=0.0)
    for _ in range(10):
        cur.observe(["S"], [0.3])
    assert cur.weights(["S"])[0] > 0.9
    # even a very-low-success real prompt is not suppressed when p=0
    for _ in range(10):
        cur.observe(["hard"], [0.08])
    assert cur.weights(["hard"])[0] > 0.9


# --------------------------------------------------------------------------- #
# rlaif
# --------------------------------------------------------------------------- #
def test_agree_gate_reduces_false_positives():
    # 100k genuine failures; verifier flips 10% to false-positive successes.
    n = 100_000
    clean = np.zeros(n)
    verifier = SymmetricRewardFlipNoise(flip_prob=0.1, seed=11)
    v_rewards, v_flip = verifier.corrupt(clean)
    judge = SimulatedJudge(error_rate=0.1, seed=22)
    j_rewards, _ = judge.judge(clean)

    fp_verifier_only = (v_rewards > 0.5).mean()
    combined = combine_rewards(v_rewards, j_rewards, mode="agree_gate")
    fp_combined = (combined > 0.5).mean()

    assert abs(fp_verifier_only - 0.1) < 0.01
    # Independent channels: agreement FP ~= p*q = 0.01.
    assert fp_combined < 0.02, fp_combined
    assert fp_combined < fp_verifier_only / 3.0


def test_correlation_erodes_denoising():
    n = 100_000
    clean = np.zeros(n)
    verifier = SymmetricRewardFlipNoise(flip_prob=0.1, seed=33)
    v_rewards, v_flip = verifier.corrupt(clean)

    judge_indep = SimulatedJudge(error_rate=0.1, seed=44)
    j_indep, _ = judge_indep.judge(clean, verifier_flip_mask=v_flip, correlation=0.0)
    fp_indep = (combine_rewards(v_rewards, j_indep, mode="agree_gate") > 0.5).mean()

    judge_corr = SimulatedJudge(error_rate=0.1, seed=44)
    j_corr, _ = judge_corr.judge(clean, verifier_flip_mask=v_flip, correlation=0.8)
    fp_corr = (combine_rewards(v_rewards, j_corr, mode="agree_gate") > 0.5).mean()

    # Correlated judge errors let more false positives through.
    assert fp_corr > fp_indep, (fp_indep, fp_corr)


def test_soft_avg_is_continuous_mean():
    v = np.array([1.0, 0.0, 0.1])
    j = np.array([0.0, 1.0, 1.0])
    out = combine_rewards(v, j, mode="soft_avg")
    assert np.allclose(out, [0.5, 0.5, 0.55])


def test_majority_vote_multi_channel():
    v = np.array([1.0, 1.0, 0.0])
    judges = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])  # 2 judges
    out = combine_rewards(v, judges, mode="majority")
    # row0: votes=2/3 -> success; row1: 1/3 -> fail; row2: 2/3 -> success
    assert out.tolist() == [1.0, 0.0, 1.0]


def test_agreement_gate_fp_rate_theory():
    assert abs(agreement_gate_fp_rate(0.1, 0.1, 0.0) - 0.01) < 1e-12
    assert abs(agreement_gate_fp_rate(0.1, 0.1, 1.0) - 0.1) < 1e-12
    mid = agreement_gate_fp_rate(0.1, 0.1, 0.5)
    assert 0.01 < mid < 0.1


# --------------------------------------------------------------------------- #
# standalone runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
