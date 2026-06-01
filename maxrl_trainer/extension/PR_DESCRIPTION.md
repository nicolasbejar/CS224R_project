# MaxRL with Imperfect Reward Signals — Extension

**Base:** `milestone-ipo` · **Compare:** `extension-maxrl-noise-rlaif`

Open PR: https://github.com/nicolasbejar/CS224R_project/compare/milestone-ipo...extension-maxrl-noise-rlaif?expand=1

## Summary

This PR implements the research contributions promised in the project proposal:
robustifying **MaxRL** (success-conditional maximum-likelihood RL) against
**imperfect / noisy reward signals** on Countdown. MaxRL's per-prompt
`weight = s_i / S` makes it structurally fragile to *false positives*: a single
spurious success on a truly-unsolved prompt receives the maximum weight `1.0`.
We quantify this fragility and add two opt-in mitigations.

All new hands-on code lives under `maxrl_trainer/extension/` (the extension
method-under-study), plus opt-in flags in the MaxRL trainer. **No default-project
component** (RLOO / IPO / SFT / `evaluation/countdown.py`) is modified — consistent
with honor-code §3.4.

## What's included

| Module | Purpose |
| --- | --- |
| `extension/reward_noise.py` | `SymmetricRewardFlipNoise`, `binarize_rewards` — reproducible per-step flip corruption of the success indicator. |
| `extension/rlaif.py` | `SimulatedJudge`, `VLLMJudge`, `combine_rewards` (agree-gate / OR / majority / soft-avg). Agreement gate drives effective FP rate `p → ~p·q`. |
| `extension/curriculum.py` | `PromptReweightingCurriculum` — binomial gate down-weights prompts whose EMA success rate is statistically indistinguishable from the noise floor; **no-op at p=0**. |
| `extension/experiments/toy_maxrl_noise.py` | Synthetic CPU bandit comparing MaxRL vs RLOO under noise + mitigations. |
| `extension/experiments/offline_signal_fidelity.py` | Replays **real** Countdown eval rollouts to measure gradient-weight routed to false positives. |
| `maxrl.py`, `maxrl_update_worker.py` | Opt-in flags wiring noise → RLAIF → curriculum into the reward pipeline & per-prompt weighting. Defaults OFF ⇒ identical to base MaxRL. |
| `train_maxrl_noise_modal.sh` | Modal launcher threading the extension knobs. |

## Key findings

- MaxRL is **~3.6× more noise-fragile** than RLOO (toy pass@1 collapse).
- On real rollouts, false-positive gradient-weight fraction: **p=0.01 → 4.5%**,
  **p=0.1 → 30.8%**, **p=0.2 → 42%**.
- RLAIF agreement gate caps this at **~10%** even at p=0.2 (realizes `p → p²`),
  recovering toy pass@1 from **0.253 → 0.536** at p=0.2.
- RLAIF benefit erodes as verifier/judge errors correlate (reported honestly).

## Testing

17 CPU-only tests pass (`extension/tests/test_extension.py` 13/13,
`test_trainer_integration.py` 4/4), including `test_pipeline_noop_when_all_off`
which guarantees the trainer is unchanged when all flags are off.

```
python maxrl_trainer/extension/tests/test_extension.py
python maxrl_trainer/extension/tests/test_trainer_integration.py
```
