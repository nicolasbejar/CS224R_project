"""Integration tests for the extension wiring inside ``maxrl_trainer/maxrl.py``.

These tests exercise the *real* ``MaxRLTrainer._apply_reward_pipeline`` method
(the corrupt -> RLAIF de-noise -> curriculum-reweight pipeline) on CPU. The
trainer module pulls in GPU/cluster dependencies (torch, ray, wandb,
transformers, the sampling/update workers) at import time, so we install
light-weight stand-ins in ``sys.modules`` *before* importing it. The pipeline
itself is pure-numpy and uses only the extension classes, so the stubs never
affect the code under test.

We construct the trainer via ``__new__`` (bypassing the heavy ``__init__`` that
loads datasets / tokenizer / wandb) and set just the attributes the pipeline
reads. This keeps the test fast and dependency-free while still calling the
exact production method.

Run directly (``python test_trainer_integration.py``) or via pytest.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Install minimal stubs for the heavy imports in maxrl.py so it imports on CPU.
# --------------------------------------------------------------------------- #
def _install_stubs() -> None:
    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    if "ray" not in sys.modules:
        ray = _mod("ray")
        ray.init = lambda *a, **k: None
        ray.get = lambda x: x
        ray.kill = lambda x: None
        ray.shutdown = lambda *a, **k: None

        def _remote(*d_args, **d_kwargs):
            # Support both @ray.remote and @ray.remote(num_gpus=1) usage.
            def wrap(cls):
                return cls
            if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
                return d_args[0]
            return wrap

        ray.remote = _remote

    if "torch" not in sys.modules:
        torch = _mod("torch")
        torch.Tensor = object

    if "wandb" not in sys.modules:
        wandb = _mod("wandb")
        wandb.init = lambda *a, **k: types.SimpleNamespace(
            config=types.SimpleNamespace(update=lambda *a, **k: None),
            log=lambda *a, **k: None,
            finish=lambda *a, **k: None,
        )
        wandb.Table = object

    if "transformers" not in sys.modules:
        transformers = _mod("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: None
        )
        transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: None
        )

    # Stub the sibling packages imported by maxrl.py.
    if "evaluation" not in sys.modules:
        evaluation = _mod("evaluation")
        countdown = _mod("evaluation.countdown")
        countdown.compute_score = lambda *a, **k: 0.0
        evaluation.countdown = countdown
    if "rloo_trainer" not in sys.modules:
        rloo = _mod("rloo_trainer")
        sw = _mod("rloo_trainer.sampling_worker")
        sw.SamplingWorker = object
        ds = _mod("rloo_trainer.rloo_dataset")
        ds.get_dataloaders = lambda *a, **k: {"train": [], "test": []}
        rloo.sampling_worker = sw
        rloo.rloo_dataset = ds
    # The update worker imports torch at module top; stub it too.
    if "maxrl_trainer.maxrl_update_worker" not in sys.modules:
        uw = _mod("maxrl_trainer.maxrl_update_worker")
        uw.MaxRLUpdateWorker = object


_install_stubs()

# Make the project root importable, then import the real trainer class.
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maxrl_trainer.maxrl import MaxRLTrainer  # noqa: E402
from maxrl_trainer.extension.reward_noise import SymmetricRewardFlipNoise  # noqa: E402
from maxrl_trainer.extension.rlaif import SimulatedJudge  # noqa: E402
from maxrl_trainer.extension.curriculum import PromptReweightingCurriculum  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_trainer(
    *,
    reward_noise=None,
    use_rlaif=False,
    judge=None,
    judge_backend="simulated",
    rlaif_combine="agree_gate",
    judge_correlation=0.0,
    curriculum=None,
    group_size=4,
    success_threshold=0.5,
) -> MaxRLTrainer:
    """Build a trainer instance without running the heavy __init__."""
    t = MaxRLTrainer.__new__(MaxRLTrainer)
    t.success_threshold = float(success_threshold)
    t.group_size = int(group_size)
    t.reward_noise = reward_noise
    t.reward_noise_p = 0.0 if reward_noise is None else reward_noise.flip_prob
    t.use_rlaif = use_rlaif
    t.judge = judge
    t.judge_backend = judge_backend
    t.rlaif_combine = rlaif_combine
    t.judge_correlation = judge_correlation
    t.judge_error_rate = getattr(judge, "error_rate", 0.0) if judge is not None else 0.0
    t.curriculum = curriculum
    t.use_curriculum = curriculum is not None
    t._extension_active = bool(reward_noise is not None or use_rlaif or curriculum is not None)
    return t


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_pipeline_noop_when_all_off():
    """With every extension feature off the rewards must be returned unchanged."""
    t = _make_trainer(group_size=4)
    rewards = [[1.0, 0.0, 0.1, 1.0], [0.0, 0.0, 1.0, 0.1]]
    out, pw, ext = t._apply_reward_pipeline(
        rewards, ["pA", "pB"], [["r"] * 4, ["r"] * 4], [{}, {}], step=0
    )
    assert out == rewards
    assert pw is None
    assert ext == {}


def test_pipeline_noise_introduces_flips():
    """A high flip probability must change some success bits and report frac."""
    noise = SymmetricRewardFlipNoise(flip_prob=0.5, seed=1)
    t = _make_trainer(reward_noise=noise, group_size=8)
    rewards = [[1.0] * 8, [0.0] * 8]
    out, pw, ext = t._apply_reward_pipeline(
        rewards, ["pA", "pB"], [["r"] * 8, ["r"] * 8], [{}, {}], step=3
    )
    flat_in = [r for g in rewards for r in g]
    flat_out = [r for g in out for r in g]
    assert flat_in != flat_out  # at p=0.5 flips are overwhelmingly likely
    assert "ext/flip_frac" in ext and 0.0 < ext["ext/flip_frac"] <= 1.0
    assert pw is None


def test_pipeline_rlaif_gate_removes_false_positive():
    """A verifier false positive on a true-fail prompt is gated out by RLAIF.

    We force a 0->1 verifier flip on a genuinely-failing prompt and use a
    perfect judge (error_rate=0). The agreement gate requires both channels to
    agree on success, so the false positive must be suppressed back to failure.
    """
    noise = SymmetricRewardFlipNoise(flip_prob=1.0, seed=0)  # flip every bit
    judge = SimulatedJudge(error_rate=0.0, seed=0, success_threshold=0.5)
    t = _make_trainer(
        reward_noise=noise, use_rlaif=True, judge=judge, rlaif_combine="agree_gate",
        group_size=4,
    )
    # All truly failing -> verifier flips all to (false) success, judge says fail.
    rewards = [[0.0, 0.0, 0.0, 0.0]]
    out, _, ext = t._apply_reward_pipeline(
        rewards, ["pA"], [["r"] * 4], [{}], step=7
    )
    # Agreement gate must wipe out the false successes.
    assert all(r <= 0.5 for r in out[0]), out
    assert "ext/judge_err_frac" in ext


def test_pipeline_curriculum_weights_in_range_and_suppress_noise_floor():
    """Curriculum returns per-prompt weights in [w_min, 1]; floor-level prompts
    are pushed toward w_min after enough observations, real prompts stay high."""
    curr = PromptReweightingCurriculum.from_noise(
        flip_prob=0.1, n_eff=40.0, kappa=1.0, w_min=0.1, ema_decay=0.9
    )
    t = _make_trainer(curriculum=curr, group_size=8, reward_noise=None)
    # pA: real, high success; pB: noise-floor-level success (~0.1).
    real = [1.0] * 1 + [1.0] * 5 + [0.0] * 2  # 0.75 success
    floor = [1.0] * 1 + [0.0] * 7             # 0.125 success ~ floor
    pw = None
    for step in range(40):
        _, pw, ext = t._apply_reward_pipeline(
            [real, floor], ["pA", "pB"], [["r"] * 8, ["r"] * 8], [{}, {}], step=step
        )
    assert pw is not None
    assert pw.shape == (2,)
    assert np.all(pw >= 0.1 - 1e-6) and np.all(pw <= 1.0 + 1e-6)
    # Real prompt keeps near-full weight; floor prompt is suppressed below it.
    assert pw[0] > 0.8, pw
    assert pw[1] < pw[0], pw
    assert "ext/curriculum_weight_mean" in ext


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
