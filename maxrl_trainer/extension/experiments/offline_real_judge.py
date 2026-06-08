"""End-to-end RLAIF with a *real* LLM judge on real Countdown rollouts.

Motivation
----------
The offline signal-fidelity study (``offline_signal_fidelity.py``) and the toy
bandit both use a :class:`SimulatedJudge` whose error rate ``q`` and
verifier-correlation ``rho`` are free parameters (we fixed ``q = 0.1``). The
poster's future-work item asks the obvious next question: *what does a real LLM
judge actually do here?* This script answers it.

It takes the recorded eval rollouts (prompt, 16 responses, ground truth, 16
verifier scores) from our trained SFT/IPO checkpoints, runs a real
:class:`VLLMJudge` over every response, and measures the two quantities that the
simulation could only assume:

  1. ``q_fp`` -- the judge's empirical false-positive rate, i.e. how often it
     calls a *truly-incorrect* equation correct. This is the only judge error
     the agreement gate cares about, and it replaces the assumed ``q = 0.1``.
  2. The real-judge agreement-gate contamination curves (false-positive
     gradient-weight fraction and spurious-success prompt rate vs the verifier
     flip probability ``p``), computed with the judge's *actual* per-response
     decisions instead of a simulated channel.

Because the Countdown verifier on these clean eval rollouts is exact, the clean
verifier label *is* ground truth, so the judge's errors are measured directly
against it. The simulated verifier flips are then injected on top, and the gate
keeps a false positive only when the real judge *also* (wrongly) accepts that
response -- which happens with the measured rate ``q_fp``. This is the
independent-error regime (``rho = 0``) with a *measured* ``q`` rather than an
assumed one; we say so explicitly in the report.

Backends
--------
``--judge-backend vllm`` runs the real judge (GPU/Modal only).
``--judge-backend stub`` fabricates judge labels by flipping the clean label
with a fixed probability; it needs no GPU and exists only to validate the
plumbing/metrics on CPU before paying for a GPU run.

Run (GPU/Modal):
    python maxrl_trainer/extension/experiments/offline_real_judge.py \
        --judge-backend vllm --judge-model Qwen/Qwen2.5-7B-Instruct

Run (CPU smoke test):
    python maxrl_trainer/extension/experiments/offline_real_judge.py \
        --judge-backend stub --stub-q 0.1 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maxrl_trainer.extension.reward_noise import SymmetricRewardFlipNoise, binarize_rewards
from maxrl_trainer.extension.rlaif import (
    RLAIFConfig,
    SimulatedJudge,
    VLLMJudge,
    combine_rewards,
)

DEFAULT_EVAL_FILES = {
    "IPO": os.path.join(_PROJECT_ROOT, "evaluation", "eval_results", "ipo_e1.json"),
    "SFT": os.path.join(_PROJECT_ROOT, "evaluation", "eval_results", "sft_full_e6.json"),
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
class PromptGroup:
    """One eval prompt: clean labels, response texts, and ground truth."""

    __slots__ = ("prompt", "responses", "clean", "ground_truth")

    def __init__(self, prompt, responses, clean, ground_truth):
        self.prompt = prompt
        self.responses = responses
        self.clean = clean              # binary verifier success (== ground truth)
        self.ground_truth = ground_truth


def load_groups(path: str, success_threshold: float = 0.5) -> list[PromptGroup]:
    groups: list[PromptGroup] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores = np.asarray(rec["scores"], dtype=np.float64)
            clean = binarize_rewards(scores, success_threshold)
            responses = list(rec["response"])
            gt = rec.get("ground_truth") or {
                "numbers": rec.get("nums"),
                "target": rec.get("target"),
            }
            groups.append(PromptGroup(rec.get("prompt", ""), responses, clean, gt))
    return groups


# --------------------------------------------------------------------------- #
# Judge label collection
# --------------------------------------------------------------------------- #
def collect_judge_labels(
    groups: list[PromptGroup],
    backend: str,
    judge_model: str,
    stub_q: float,
    stub_seed: int,
    success_threshold: float = 0.5,
) -> list[np.ndarray]:
    """Return, per group, the judge's binary success label for each response.

    ``vllm``  -- real judge decision per response.
    ``stub``  -- clean label flipped independently with probability ``stub_q``
                 (CPU-only plumbing check; NOT a real measurement).
    """
    if backend == "stub":
        rng = np.random.default_rng(int(stub_seed))
        labels = []
        for g in groups:
            flip = rng.random(g.clean.shape[0]) < stub_q
            labels.append(np.where(flip, 1.0 - g.clean, g.clean).astype(np.float64))
        return labels

    if backend != "vllm":
        raise ValueError(f"unknown judge backend: {backend!r}")

    cfg = RLAIFConfig(
        enabled=True,
        backend="vllm",
        judge_model=judge_model,
        success_threshold=success_threshold,
    )
    judge = VLLMJudge(cfg)

    # Flatten across all groups for one batched vLLM call, then re-split.
    flat_prompts, flat_responses, flat_gts, offsets = [], [], [], []
    cursor = 0
    for g in groups:
        n = len(g.responses)
        offsets.append((cursor, cursor + n))
        cursor += n
        flat_prompts.extend([g.prompt] * n)
        flat_responses.extend(g.responses)
        flat_gts.extend([g.ground_truth] * n)

    flat_labels = judge.judge(flat_prompts, flat_responses, flat_gts)
    flat_labels = np.asarray(flat_labels, dtype=np.float64).reshape(-1)
    return [flat_labels[a:b] for (a, b) in offsets]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def judge_error_stats(
    groups: list[PromptGroup], judge_labels: list[np.ndarray], thr: float = 0.5
) -> dict:
    """Empirical judge error rates relative to the (exact) clean verifier."""
    n = fp = fn = tot_fail = tot_succ = 0
    disagree = 0
    for g, jl in zip(groups, judge_labels):
        true_succ = g.clean > thr
        judge_succ = jl > thr
        n += true_succ.size
        disagree += int(np.sum(true_succ != judge_succ))
        tot_fail += int(np.sum(~true_succ))
        tot_succ += int(np.sum(true_succ))
        fp += int(np.sum(judge_succ & ~true_succ))   # judge accepts a wrong answer
        fn += int(np.sum(~judge_succ & true_succ))   # judge rejects a correct answer
    return {
        "n_samples": n,
        "q_overall": disagree / n if n else 0.0,
        "q_fp": fp / tot_fail if tot_fail else 0.0,   # P(judge=1 | truly fail)
        "q_fn": fn / tot_succ if tot_succ else 0.0,   # P(judge=0 | truly succeed)
        "n_true_fail": tot_fail,
        "n_true_success": tot_succ,
    }


def maxrl_weights(success: np.ndarray) -> np.ndarray:
    S = success.sum()
    if S <= 0:
        return np.zeros_like(success)
    return success / S


def _contamination(clean: np.ndarray, signal: np.ndarray, thr: float) -> tuple:
    """Return (fp_weight, total_weight, zero_prompt, spurious_prompt)."""
    true_success = clean > thr
    obs_success = (signal > thr).astype(np.float64)
    w = maxrl_weights(obs_success)
    fp_mask = obs_success.astype(bool) & (~true_success)
    zero = not true_success.any()
    spurious = zero and obs_success.any()
    return w[fp_mask].sum(), w.sum(), zero, spurious


def gate_curves(
    groups: list[PromptGroup],
    judge_labels: list[np.ndarray],
    flip_grid: list[float],
    n_seeds: int,
    thr: float = 0.5,
) -> dict:
    """Contamination curves vs flip rate for verifier-only and the real-judge gate.

    The verifier flips are simulated (the eval rollouts are clean); the judge
    labels are fixed real decisions, gated against the noisy verifier.
    """
    out = {"flip_grid": flip_grid, "maxrl": [], "maxrl_real_judge": []}
    for p in flip_grid:
        mr = {"fp": 0.0, "tot": 0.0, "zero": 0, "spur": 0.0}
        rj = {"fp": 0.0, "tot": 0.0, "zero": 0, "spur": 0.0}
        for seed in range(n_seeds):
            noise = SymmetricRewardFlipNoise(flip_prob=p, seed=1000 + seed)
            for g, jl in zip(groups, judge_labels):
                noisy, _ = noise.corrupt(g.clean, step=seed)
                # verifier only
                fpw, tw, z, sp = _contamination(g.clean, noisy, thr)
                mr["fp"] += fpw; mr["tot"] += tw
                mr["zero"] += int(z); mr["spur"] += float(sp)
                # agreement gate against the real judge
                gated = combine_rewards(noisy, jl, mode="agree_gate", success_threshold=thr)
                fpw, tw, z, sp = _contamination(g.clean, gated, thr)
                rj["fp"] += fpw; rj["tot"] += tw
                rj["zero"] += int(z); rj["spur"] += float(sp)
        out["maxrl"].append(_summ(mr))
        out["maxrl_real_judge"].append(_summ(rj))
    return out


def _summ(a: dict) -> dict:
    return {
        "false_positive_weight_fraction": a["fp"] / a["tot"] if a["tot"] > 0 else 0.0,
        "spurious_success_prompt_rate": a["spur"] / a["zero"] if a["zero"] > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def make_plots(payload: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plots skipped: matplotlib unavailable: {exc}]")
        return

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    for metric, ylabel, fname in [
        ("false_positive_weight_fraction",
         "false-positive gradient-weight fraction", "real_judge_fp_weight_fraction.png"),
        ("spurious_success_prompt_rate",
         "spurious-success prompt rate", "real_judge_spurious_prompt_rate.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        name, mdl = next(iter(payload["models"].items()))
        res = mdl["curves"]
        xs = res["flip_grid"]
        ax.plot(xs, [d[metric] for d in res["maxrl"]], "-o", color="#d62728",
                label="MaxRL (verifier only)")
        ax.plot(xs, [d[metric] for d in res["maxrl_real_judge"]], "--s", color="#1f77b4",
                label="MaxRL + RLAIF gate (real judge)")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("verifier flip probability $p$")
        ax.set_ylabel(ylabel)
        q_fp = mdl["judge_stats"]["q_fp"]
        ax.set_title(f"Real Countdown rollouts \u00b7 real LLM judge "
                     f"($q_{{fp}}{{=}}{q_fp:.2f}$, {name})")
        ax.legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)
    print(f"Wrote 2 figures to {out_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-backend", choices=("vllm", "stub"), default="vllm")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--stub-q", type=float, default=0.1,
                        help="stub backend only: fabricated judge error rate")
    parser.add_argument("--stub-seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(_HERE, "..", "figures"))
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.quick:
        flip_grid = [0.0, 0.01, 0.1]
        n_seeds = 20
    else:
        flip_grid = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        n_seeds = 100

    payload: dict = {
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model if args.judge_backend == "vllm" else "stub",
        "stub_q": args.stub_q if args.judge_backend == "stub" else None,
        "n_seeds": n_seeds,
        "models": {},
    }

    for name, path in DEFAULT_EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"[skip {name}: {path} not found]")
            continue
        groups = load_groups(path)
        labels = collect_judge_labels(
            groups, args.judge_backend, args.judge_model,
            args.stub_q, args.stub_seed,
        )
        stats = judge_error_stats(groups, labels)
        curves = gate_curves(groups, labels, flip_grid, n_seeds)
        payload["models"][name] = {
            "n_prompts": len(groups),
            "judge_stats": stats,
            "curves": curves,
        }
        print(f"{name}: {len(groups)} prompts | judge q_fp={stats['q_fp']:.3f} "
              f"(accepts {stats['q_fp']:.1%} of wrong answers), "
              f"q_fn={stats['q_fn']:.3f}, q_overall={stats['q_overall']:.3f}")
        for p in flip_grid:
            i = flip_grid.index(p)
            mr = curves["maxrl"][i]["false_positive_weight_fraction"]
            rj = curves["maxrl_real_judge"][i]["false_positive_weight_fraction"]
            print(f"  p={p:<6} FP-weight: MaxRL={mr:.3f}  real-judge gate={rj:.3f}")

    json_path = os.path.join(out_dir, "offline_real_judge_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    make_plots(payload, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
