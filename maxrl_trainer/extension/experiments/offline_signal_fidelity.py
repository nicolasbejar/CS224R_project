"""Offline signal-fidelity analysis on *real* Countdown eval rollouts.

Unlike the toy bandit (which trains a synthetic policy), this script takes the
actual grouped rollouts produced by our trained checkpoints -- the eval-result
files under ``evaluation/eval_results/`` -- and characterises, as a function of
the verifier flip probability ``p``, exactly how reward noise contaminates the
MaxRL gradient. No GPU and no training required: it replays the recorded
rewards through the imperfect-reward pipeline and measures the resulting
gradient-weight statistics in closed form.

Each eval record holds 16 sampled responses for one prompt, each scored on the
Countdown rubric ``{0.0 (no answer), 0.1 (answer, wrong), 1.0 (correct)}``. We
binarise success as ``score >= 0.5`` (only fully-correct responses), which is
the signal MaxRL conditions on.

Quantities measured vs ``p`` (averaged over noise seeds):
  * false_positive_weight_fraction -- the share of total MaxRL gradient weight
    (``sum_i s_i / S`` per prompt) that lands on responses that were *truly*
    failures but were flipped to "success". This is the contamination MaxRL
    injects, and it is amplified because a false positive on a low-``S`` prompt
    receives weight up to ``1.0``.
  * spurious_success_prompt_rate -- the fraction of prompts with *zero* true
    successes that acquire at least one (false) success after noise. These
    prompts contribute pure noise to the MaxRL objective.
  * mean_weight_on_false_positives -- average per-sample MaxRL weight assigned
    to a false-positive response (illustrates the ``1/S`` amplification).

Each metric is reported for three pipelines:
    MaxRL (verifier only) | MaxRL + RLAIF agreement gate | + curriculum

Run:
    python maxrl_trainer/extension/experiments/offline_signal_fidelity.py
    python maxrl_trainer/extension/experiments/offline_signal_fidelity.py --quick
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
from maxrl_trainer.extension.curriculum import PromptReweightingCurriculum
from maxrl_trainer.extension.rlaif import SimulatedJudge, combine_rewards

DEFAULT_EVAL_FILES = {
    "IPO": os.path.join(_PROJECT_ROOT, "evaluation", "eval_results", "ipo_e1.json"),
    "SFT": os.path.join(_PROJECT_ROOT, "evaluation", "eval_results", "sft_full_e6.json"),
}


def load_grouped_rewards(path: str, success_threshold: float = 0.5) -> list[np.ndarray]:
    """Load one eval-result JSONL file into a list of binarised reward groups."""
    groups: list[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores = np.asarray(rec["scores"], dtype=np.float64)
            groups.append(binarize_rewards(scores, success_threshold))
    return groups


def maxrl_weights(success: np.ndarray) -> np.ndarray:
    """Per-sample MaxRL weight ``s_i / S`` (zeros if the group has no success)."""
    S = success.sum()
    if S <= 0:
        return np.zeros_like(success)
    return success / S


def analyze(
    groups: list[np.ndarray],
    flip_grid: list[float],
    n_seeds: int,
    judge_error: float,
    success_threshold: float = 0.5,
) -> dict:
    """Replay grouped rewards through the pipeline; return metrics vs flip rate."""
    results = {
        "flip_grid": flip_grid,
        "maxrl": [],
        "maxrl_rlaif": [],
        "maxrl_rlaif_curriculum": [],
    }

    for p in flip_grid:
        acc = {
            "maxrl": _Accum(),
            "maxrl_rlaif": _Accum(),
            "maxrl_rlaif_curriculum": _Accum(),
        }
        for seed in range(n_seeds):
            noise = SymmetricRewardFlipNoise(flip_prob=p, seed=1000 + seed)
            judge = SimulatedJudge(error_rate=judge_error, seed=2000 + seed)
            # curriculum keyed by prompt index, floor = effective post-RLAIF rate
            eff_floor = p * judge_error
            curr = PromptReweightingCurriculum.from_noise(
                flip_prob=eff_floor, n_eff=len(groups[0]), kappa=1.0, w_min=0.05
            )

            for gi, clean in enumerate(groups):
                noisy, flip_mask = noise.corrupt(clean, step=seed)

                # ---- MaxRL (verifier only) ----
                _accumulate(acc["maxrl"], clean, noisy, success_threshold)

                # ---- MaxRL + RLAIF agreement gate ----
                judge_sig, _ = judge.judge(clean, verifier_flip_mask=flip_mask, step=seed)
                gated = combine_rewards(noisy, judge_sig, mode="agree_gate",
                                        success_threshold=success_threshold)
                _accumulate(acc["maxrl_rlaif"], clean, gated, success_threshold)

                # ---- + curriculum (per-prompt weight on the gated signal) ----
                succ = (gated > success_threshold).astype(np.float64)
                cw = float(curr.step([gi], [succ.mean()])[0])
                _accumulate(acc["maxrl_rlaif_curriculum"], clean, gated,
                            success_threshold, prompt_weight=cw)

        for key in acc:
            results[key].append(acc[key].summary())

    return results


class _Accum:
    def __init__(self) -> None:
        self.fp_weight = 0.0          # total MaxRL weight on false positives
        self.total_weight = 0.0       # total MaxRL weight (sum of prompt weights w/ S>0)
        self.fp_samples = 0           # count of false-positive samples (weighted nonzero)
        self.fp_weight_samples = 0.0  # sum of per-sample weights on FPs (for mean)
        self.zero_prompts = 0         # prompts with zero true success
        self.spurious_prompts = 0.0   # of those, how many got a false success

    def summary(self) -> dict:
        return {
            "false_positive_weight_fraction":
                self.fp_weight / self.total_weight if self.total_weight > 0 else 0.0,
            "spurious_success_prompt_rate":
                self.spurious_prompts / self.zero_prompts if self.zero_prompts > 0 else 0.0,
            "mean_weight_on_false_positives":
                self.fp_weight_samples / self.fp_samples if self.fp_samples > 0 else 0.0,
        }


def _accumulate(acc: _Accum, clean: np.ndarray, signal: np.ndarray,
                thr: float, prompt_weight: float = 1.0) -> None:
    true_success = clean > thr
    obs_success = (signal > thr).astype(np.float64)
    w = maxrl_weights(obs_success) * prompt_weight
    fp_mask = obs_success.astype(bool) & (~true_success)   # counted success but truly failure

    acc.total_weight += w.sum()
    acc.fp_weight += w[fp_mask].sum()
    acc.fp_samples += int(fp_mask.sum())
    acc.fp_weight_samples += w[fp_mask].sum()

    if not true_success.any():
        acc.zero_prompts += 1
        if obs_success.any():
            acc.spurious_prompts += 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--judge_error", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str,
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

    payload: dict = {"judge_error": args.judge_error, "n_seeds": n_seeds, "models": {}}
    for name, path in DEFAULT_EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"[skip {name}: {path} not found]")
            continue
        groups = load_grouped_rewards(path)
        base_succ = float(np.mean([g.mean() for g in groups]))
        zero_frac = float(np.mean([1.0 if not g.any() else 0.0 for g in groups]))
        print(f"{name}: {len(groups)} prompts, mean success={base_succ:.3f}, "
              f"zero-success prompts={zero_frac:.2%}")
        res = analyze(groups, flip_grid, n_seeds, args.judge_error)
        payload["models"][name] = {
            "n_prompts": len(groups),
            "mean_success": base_succ,
            "zero_success_prompt_fraction": zero_frac,
            "metrics": res,
        }
        for p, m in zip(flip_grid, res["maxrl"]):
            mr = res["maxrl"][flip_grid.index(p)]
            rl = res["maxrl_rlaif"][flip_grid.index(p)]
            print(f"  p={p:<6} FP-weight: MaxRL={mr['false_positive_weight_fraction']:.3f} "
                  f"RLAIF={rl['false_positive_weight_fraction']:.3f} | "
                  f"spurious-prompt: MaxRL={mr['spurious_success_prompt_rate']:.3f} "
                  f"RLAIF={rl['spurious_success_prompt_rate']:.3f}")

    json_path = os.path.join(out_dir, "offline_signal_fidelity_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    _make_plots(payload, out_dir)
    return 0


def _make_plots(payload: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plots skipped: matplotlib unavailable: {exc}]")
        return

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    cond_labels = {
        "maxrl": "MaxRL (verifier only)",
        "maxrl_rlaif": "MaxRL + RLAIF gate",
        "maxrl_rlaif_curriculum": "MaxRL + RLAIF + curriculum",
    }

    for metric, ylabel, fname in [
        ("false_positive_weight_fraction",
         "false-positive gradient-weight fraction", "offline_fp_weight_fraction.png"),
        ("spurious_success_prompt_rate",
         "spurious-success prompt rate", "offline_spurious_prompt_rate.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        for name, mdl in payload["models"].items():
            res = mdl["metrics"]
            xs = res["flip_grid"]
            for cond, style in zip(
                ["maxrl", "maxrl_rlaif", "maxrl_rlaif_curriculum"],
                ["-o", "--s", ":^"],
            ):
                ys = [d[metric] for d in res[cond]]
                ax.plot(xs, ys, style, label=f"{name}: {cond_labels[cond]}", alpha=0.9)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("verifier flip probability $p$")
        ax.set_ylabel(ylabel)
        ax.set_title("MaxRL contamination on real Countdown rollouts")
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)

    print(f"Wrote 2 figures to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
