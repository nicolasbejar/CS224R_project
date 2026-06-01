"""Controlled toy study of MaxRL under imperfect rewards.

This is a small, fully-CPU experiment that reproduces -- in a controlled
setting where we know the ground truth exactly -- the central claims of our
extension before committing GPU hours to the full Countdown runs:

  1. MaxRL's ``1/S`` (successful-conditional) weighting learns hard, low-success
     prompts far faster than RLOO, but is *structurally fragile* to reward-flip
     noise: a single false-positive success on a truly-hard prompt receives the
     maximum gradient weight.
  2. An RLAIF agreement gate (a second, independent judge channel) reduces the
     effective false-positive rate from ``p`` to ``~p*q`` and largely restores
     MaxRL's accuracy under noise.
  3. A dynamic prompt re-weighting curriculum, which suppresses prompts whose
     observed success rate sits at the noise floor, provides a complementary
     gain, and the two combine.

Environment
-----------
A tabular contextual bandit that mimics the structure of Countdown: each
"prompt" ``x`` exposes ``K`` candidate answers, exactly one of which is correct.
The policy ``pi_theta(.|x) = softmax(theta[x])`` is tabular, so we can run many
prompts x groups x steps in milliseconds. Prompt difficulty is varied through
the initial logit gap, giving a realistic mix of easy and hard prompts.

Reward pipeline per step (mirrors the real trainer):
    sample G answers ~ pi  ->  exact reward r in {0,1}
        ->  symmetric flip noise (SymmetricRewardFlipNoise)
        ->  optional RLAIF judge + agreement gate (SimulatedJudge, combine_rewards)
        ->  optional curriculum prompt weight (PromptReweightingCurriculum)
        ->  MaxRL (1/S) or RLOO (leave-one-out) advantage  ->  tabular PG step

Run:
    python maxrl_trainer/extension/experiments/toy_maxrl_noise.py            # full sweep + plots
    python maxrl_trainer/extension/experiments/toy_maxrl_noise.py --quick    # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from maxrl_trainer.extension.reward_noise import SymmetricRewardFlipNoise
from maxrl_trainer.extension.curriculum import PromptReweightingCurriculum
from maxrl_trainer.extension.rlaif import SimulatedJudge, combine_rewards


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class ToyConfig:
    n_prompts: int = 48
    n_actions: int = 12          # K candidate answers; 1 is correct
    group_size: int = 8          # G rollouts per prompt per step
    n_steps: int = 140
    lr: float = 0.5
    seed: int = 0
    # init logit gap range: smaller -> harder (lower base success prob)
    min_init_gap: float = -1.8
    max_init_gap: float = 0.4


def _init_policy(cfg: ToyConfig, rng: np.random.Generator) -> np.ndarray:
    """Return logits [n_prompts, n_actions]; action 0 is the correct answer."""
    logits = rng.normal(0.0, 0.3, size=(cfg.n_prompts, cfg.n_actions))
    gaps = np.linspace(cfg.min_init_gap, cfg.max_init_gap, cfg.n_prompts)
    logits[:, 0] = gaps  # spread of difficulties via correct-action init logit
    return logits


def _passk(prob_correct: np.ndarray, k: int) -> float:
    """Analytic pass@k under i.i.d. sampling: 1 - (1 - p_correct)^k, mean over prompts."""
    return float((1.0 - (1.0 - prob_correct) ** k).mean())


def run_condition(
    cfg: ToyConfig,
    flip_prob: float,
    algorithm: str = "maxrl",          # {"maxrl", "rloo"}
    use_rlaif: bool = False,
    judge_error_rate: float = 0.0,
    judge_correlation: float = 0.0,
    use_curriculum: bool = False,
    curriculum_kappa: float = 1.0,
    success_threshold: float = 0.5,
) -> dict:
    """Train the toy policy under one configuration; return metrics + curve."""
    rng = np.random.default_rng(cfg.seed)
    logits = _init_policy(cfg, rng)
    G = cfg.group_size

    noise = SymmetricRewardFlipNoise(flip_prob=flip_prob, seed=cfg.seed + 101)
    judge = SimulatedJudge(error_rate=judge_error_rate, seed=cfg.seed + 202)
    # The EMA accumulates evidence over steps; its effective sample size is
    # ~ G / (1 - ema_decay), which sharpens the floor-vs-signal gate.
    ema_decay = 0.9
    n_eff = G / (1.0 - ema_decay)
    # When RLAIF's agreement gate is active the *effective* false-positive floor
    # that the curriculum sees is ~ p * q (independent channels), not the raw
    # verifier rate p. Configuring the curriculum with the post-RLAIF floor
    # stops it from fighting RLAIF by over-suppressing.
    effective_floor = flip_prob * judge_error_rate if use_rlaif else flip_prob
    curriculum = PromptReweightingCurriculum.from_noise(
        flip_prob=effective_floor, n_eff=n_eff, kappa=curriculum_kappa,
        ema_decay=ema_decay, w_min=0.05,
    )

    pass1_curve: list[float] = []
    for step in range(cfg.n_steps):
        probs = _softmax(logits)                       # [P, K]
        p_correct = probs[:, 0]
        pass1_curve.append(float(p_correct.mean()))

        # ---- sample G actions per prompt ----
        # actions[p, g] ~ pi(.|p)
        actions = np.array(
            [rng.choice(cfg.n_actions, size=G, p=probs[p]) for p in range(cfg.n_prompts)]
        )                                              # [P, G]
        clean = (actions == 0).astype(np.float64)      # exact verifier reward

        # ---- imperfect-reward pipeline ----
        clean_flat = clean.reshape(-1)
        noisy_flat, flip_mask = noise.corrupt(clean_flat, step=step)
        if use_rlaif:
            judge_flat, _ = judge.judge(
                clean_flat, verifier_flip_mask=flip_mask,
                correlation=judge_correlation, step=step,
            )
            reward_flat = combine_rewards(noisy_flat, judge_flat, mode="agree_gate",
                                          success_threshold=success_threshold)
        else:
            reward_flat = noisy_flat
        rewards = reward_flat.reshape(cfg.n_prompts, G)

        success = (rewards > success_threshold).astype(np.float64)   # [P, G]

        # ---- curriculum prompt weights from observed (noisy) success rate ----
        if use_curriculum:
            obs_rate = success.mean(axis=1)
            keys = [f"p{p}" for p in range(cfg.n_prompts)]
            prompt_w = curriculum.step(keys, obs_rate.tolist())       # [P]
        else:
            prompt_w = np.ones(cfg.n_prompts)

        # ---- advantage / weighting ----
        if algorithm == "maxrl":
            S = success.sum(axis=1, keepdims=True)                    # [P,1]
            safe_S = np.clip(S, 1.0, None)
            sample_w = (success / safe_S) * (S > 0)                   # 1/S on successes
            adv = sample_w                                            # weight is the coefficient
        elif algorithm == "rloo":
            r_sum = rewards.sum(axis=1, keepdims=True)
            adv = (G * rewards - r_sum) / (G - 1)
        else:
            raise ValueError(algorithm)

        # ---- tabular policy-gradient step ----
        # grad_theta log pi(a|x) = onehot(a) - pi(.|x)
        grad = np.zeros_like(logits)
        for p in range(cfg.n_prompts):
            pi_p = probs[p]
            for g in range(G):
                a = actions[p, g]
                onehot = np.zeros(cfg.n_actions)
                onehot[a] = 1.0
                grad[p] += prompt_w[p] * adv[p, g] * (onehot - pi_p)
        logits += cfg.lr * grad / G

    final_probs = _softmax(logits)
    pc = final_probs[:, 0]
    return {
        "flip_prob": flip_prob,
        "algorithm": algorithm,
        "use_rlaif": use_rlaif,
        "use_curriculum": use_curriculum,
        "judge_error_rate": judge_error_rate,
        "judge_correlation": judge_correlation,
        "final_pass@1": float(pc.mean()),
        "final_pass@4": _passk(pc, 4),
        "final_pass@8": _passk(pc, 8),
        "hard_prompt_pass@1": float(pc[: cfg.n_prompts // 2].mean()),
        "pass1_curve": pass1_curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_HERE, "..", "figures"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.quick:
        cfg = ToyConfig(n_prompts=24, n_steps=120, seed=args.seed)
        flip_grid = [0.0, 0.01, 0.1]
    else:
        cfg = ToyConfig(seed=args.seed)
        flip_grid = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]

    judge_error = 0.1  # RLAIF judge error rate q (independent of verifier)

    conditions = [
        ("RLOO", dict(algorithm="rloo")),
        ("MaxRL", dict(algorithm="maxrl")),
        ("MaxRL + RLAIF", dict(algorithm="maxrl", use_rlaif=True,
                               judge_error_rate=judge_error)),
        ("MaxRL + curriculum", dict(algorithm="maxrl", use_curriculum=True)),
        ("MaxRL + RLAIF + curriculum",
         dict(algorithm="maxrl", use_rlaif=True, judge_error_rate=judge_error,
              use_curriculum=True)),
    ]

    results: dict[str, list[dict]] = {}
    for label, kw in conditions:
        results[label] = []
        for p in flip_grid:
            res = run_condition(cfg, flip_prob=p, **kw)
            results[label].append(res)
            print(f"{label:30s} p={p:<6} pass@1={res['final_pass@1']:.3f} "
                  f"hard={res['hard_prompt_pass@1']:.3f} pass@8={res['final_pass@8']:.3f}")

    # correlation sweep (RLAIF benefit vs verifier-judge error correlation)
    corr_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    corr_results = []
    for rho in corr_grid:
        res = run_condition(cfg, flip_prob=0.1, algorithm="maxrl", use_rlaif=True,
                            judge_error_rate=judge_error, judge_correlation=rho)
        corr_results.append(res)
        print(f"corr rho={rho:<5} pass@1={res['final_pass@1']:.3f}")

    payload = {
        "config": vars(cfg),
        "flip_grid": flip_grid,
        "judge_error": judge_error,
        "conditions": results,
        "correlation_sweep": {"rho": corr_grid, "results": corr_results},
    }
    json_path = os.path.join(out_dir, "toy_maxrl_noise_results.json")
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
    flip_grid = payload["flip_grid"]
    conditions = payload["conditions"]

    # Figure 1: pass@1 vs flip rate, all conditions.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, runs in conditions.items():
        xs = [r["flip_prob"] for r in runs]
        ys = [r["final_pass@1"] for r in runs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("verifier flip probability $p$")
    ax.set_ylabel("final pass@1")
    ax.set_title("MaxRL under reward-flip noise (toy Countdown bandit)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toy_pass1_vs_flip.png"), dpi=160)
    plt.close(fig)

    # Figure 2: hard-prompt pass@1 (the regime MaxRL targets).
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, runs in conditions.items():
        xs = [r["flip_prob"] for r in runs]
        ys = [r["hard_prompt_pass@1"] for r in runs]
        ax.plot(xs, ys, marker="s", label=label)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("verifier flip probability $p$")
    ax.set_ylabel("hard-prompt pass@1")
    ax.set_title("Hard prompts: where noise hurts MaxRL most")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toy_hard_pass1_vs_flip.png"), dpi=160)
    plt.close(fig)

    # Figure 3: RLAIF benefit vs verifier-judge error correlation.
    corr = payload["correlation_sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(corr["rho"], [r["final_pass@1"] for r in corr["results"]],
            marker="d", color="C3")
    ax.set_xlabel(r"verifier-judge error correlation $\rho$")
    ax.set_ylabel("final pass@1 (MaxRL + RLAIF, $p=0.1$)")
    ax.set_title("RLAIF de-noising erodes as channel errors correlate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toy_rlaif_correlation.png"), dpi=160)
    plt.close(fig)

    # Figure 4: example learning curves at p=0.1.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    target_p = 0.1 if 0.1 in flip_grid else flip_grid[-1]
    for label, runs in conditions.items():
        run = next(r for r in runs if r["flip_prob"] == target_p)
        ax.plot(run["pass1_curve"], label=label, alpha=0.9)
    ax.set_xlabel("training step")
    ax.set_ylabel("pass@1")
    ax.set_title(f"Learning curves at $p={target_p}$")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "toy_learning_curves_p010.png"), dpi=160)
    plt.close(fig)

    print(f"Wrote 4 figures to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
