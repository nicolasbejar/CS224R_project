"""High-level MaxRL training orchestration.

This is structurally identical to `rloo_trainer/rloo.py` (alternating vLLM
sampling and PyTorch updates). The only methodological change is that the
update worker implements the MaxRL (log p_theta) objective instead of REINFORCE
with a leave-one-out baseline. Sampling, reward computation, tokenization, and
checkpointing are reused as-is.
"""

import os
import shutil
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import random
import ray
import torch
import wandb
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# Make sibling packages (e.g., evaluation/) importable when this file is run as
# `python maxrl_trainer/maxrl.py`.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.countdown import compute_score
# Sampling + dataset are identical to RLOO, so reuse directly.
from rloo_trainer.sampling_worker import SamplingWorker
from rloo_trainer.rloo_dataset import get_dataloaders
from maxrl_trainer.maxrl_update_worker import MaxRLUpdateWorker
# Extension: imperfect-reward modelling, RLAIF de-noising, and a noise-aware
# curriculum. All default OFF so standard MaxRL behaviour is unchanged.
from maxrl_trainer.extension.reward_noise import SymmetricRewardFlipNoise
from maxrl_trainer.extension.rlaif import (
    RLAIFConfig,
    SimulatedJudge,
    VLLMJudge,
    combine_rewards,
)
from maxrl_trainer.extension.curriculum import PromptReweightingCurriculum


class MaxRLTrainer:
    """Coordinates online sampling, reward computation, and MaxRL updates."""

    def __init__(
        self,
        model_name='asingh15/qwen-sft-countdown-defaultproj',
        ref_model_name=None,
        tokenizer_name=None,
        dataset_name='asingh15/countdown_tasks_3to4',
        wandb_project='maxrl_default_project',
        wandb_name='test',
        lr_schedule='constant',
        learning_rate=1e-5,
        warmup_ratio=0.0,
        weight_decay=0.01,
        batch_size=4,
        group_size=8,
        entropy_coefficient=0.01,
        kl_divergence_coefficient=0.0,
        num_epochs=10,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=1024,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        max_num_seqs=64,
        num_training_steps=250,
        max_table_rows=20,
        save_every_n_steps=-1,
        save_dir='checkpoints/maxrl_checkpoints',
        ppo_epochs=1,
        importance_weight_clip=5.0,
        success_threshold=0.5,
        # ---- Extension flags (all default OFF -> standard MaxRL) ----
        reward_noise_p=0.0,
        reward_noise_seed=0,
        use_rlaif=False,
        judge_backend='simulated',
        rlaif_combine='agree_gate',
        judge_error_rate=0.05,
        judge_correlation=0.0,
        judge_seed=0,
        judge_model='Qwen/Qwen2.5-7B-Instruct',
        use_curriculum=False,
        curriculum_kappa=1.0,
        curriculum_w_min=0.1,
        curriculum_ema_decay=0.9,
    ):
        self.model_name = model_name
        self.ref_model_name = self.model_name if ref_model_name is None else ref_model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.dataset_name = dataset_name
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps
        self.group_size = group_size
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_seqs = max_num_seqs
        self.batch_size = batch_size
        self.max_prompt_length = 512
        self.max_response_length = max_tokens
        self.max_table_rows = max_table_rows
        self.save_every_n_steps = save_every_n_steps
        self.save_dir = save_dir
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.importance_weight_clip = float(importance_weight_clip)
        self.success_threshold = float(success_threshold)

        # ---- Extension: imperfect-reward pipeline ----
        # 1) Symmetric reward-flip noise (verifier as a binary symmetric channel).
        self.reward_noise_p = float(reward_noise_p)
        self.reward_noise_seed = int(reward_noise_seed)
        self.reward_noise = (
            SymmetricRewardFlipNoise(
                flip_prob=self.reward_noise_p,
                success_threshold=self.success_threshold,
                seed=self.reward_noise_seed,
            )
            if self.reward_noise_p > 0.0
            else None
        )
        # 2) RLAIF judge channel (agreement gate de-noises the verifier).
        self.use_rlaif = bool(use_rlaif)
        self.judge_backend = str(judge_backend)
        self.rlaif_combine = str(rlaif_combine)
        self.judge_error_rate = float(judge_error_rate)
        self.judge_correlation = float(judge_correlation)
        self.judge = None
        if self.use_rlaif:
            if self.judge_backend == 'simulated':
                self.judge = SimulatedJudge(
                    error_rate=self.judge_error_rate,
                    seed=int(judge_seed),
                    success_threshold=self.success_threshold,
                )
            elif self.judge_backend == 'vllm':
                self.judge = VLLMJudge(RLAIFConfig(
                    enabled=True,
                    backend='vllm',
                    combine=self.rlaif_combine,
                    success_threshold=self.success_threshold,
                    judge_model=str(judge_model),
                ))
            else:
                raise ValueError(f"unknown judge_backend: {self.judge_backend!r}")
        # 3) Noise-aware prompt-reweighting curriculum.
        self.use_curriculum = bool(use_curriculum)
        self.curriculum = None
        if self.use_curriculum:
            # The effective post-gate noise floor is p when RLAIF is off, and the
            # agreement-gate floor p * q when RLAIF is on (so the curriculum does
            # not fight the judge). n_eff reflects the EMA window ~ G / (1 - decay).
            effective_floor = (
                self.reward_noise_p * self.judge_error_rate
                if self.use_rlaif
                else self.reward_noise_p
            )
            n_eff = float(self.group_size) / max(1.0 - float(curriculum_ema_decay), 1e-6)
            self.curriculum = PromptReweightingCurriculum.from_noise(
                flip_prob=effective_floor,
                n_eff=n_eff,
                kappa=float(curriculum_kappa),
                w_min=float(curriculum_w_min),
                ema_decay=float(curriculum_ema_decay),
            )
        # Master switch: when no extension feature is on, the reward pipeline is
        # skipped entirely so the rewards passed downstream are byte-identical to
        # vanilla MaxRL.
        self._extension_active = bool(
            self.reward_noise is not None or self.use_rlaif or self.use_curriculum
        )

        dataloaders = get_dataloaders(
            self.dataset_name,
            splits=['train', 'test'],
            batch_size=self.batch_size,
            num_proc=4,
        )
        self.train_dataloader, self.test_dataloader = dataloaders['train'], dataloaders['test']

        self.sampling_worker = None
        self.update_worker = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.wandb = wandb.init(project=self.wandb_project, name=self.wandb_name)
        self.wandb.config.update(vars(self))

    def _create_sampling_worker(self, model_path):
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None

        self.sampling_worker = SamplingWorker.remote(
            model_path=model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_num_seqs=self.max_num_seqs,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            group_size=self.group_size,
        )
        ray.get(self.sampling_worker.load_checkpoint.remote())
        return self.sampling_worker

    def _create_update_worker(self, model_path, optimizer_path, scheduler_path):
        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None

        self.update_worker = MaxRLUpdateWorker.remote(
            model_path=model_path,
            ref_model_path=self.ref_model_name,
            optimizer_path=optimizer_path,
            scheduler_path=scheduler_path,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_clipping=self.gradient_clipping,
            group_size=self.group_size,
            entropy_coefficient=self.entropy_coefficient,
            kl_divergence_coefficient=self.kl_divergence_coefficient,
            lr_schedule=self.lr_schedule,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            num_training_steps=self.num_training_steps,
            importance_weight_clip=self.importance_weight_clip,
            success_threshold=self.success_threshold,
        )
        ray.get(self.update_worker.load_checkpoint.remote())
        return self.update_worker

    def _build_generation_table(self, prompts, responses, rewards):
        if self.max_table_rows <= 0:
            return None

        flat_rows = []
        for prompt, prompt_responses, prompt_rewards in zip(prompts, responses, rewards):
            for response, reward in zip(prompt_responses, prompt_rewards):
                flat_rows.append((prompt, response, float(np.array(reward).item())))

        if not flat_rows:
            return None

        random.shuffle(flat_rows)
        flat_rows = flat_rows[: self.max_table_rows]

        table = wandb.Table(columns=["prompt", "response", "reward"])
        for prompt, response, reward in flat_rows:
            table.add_data(prompt, response, reward)
        return table

    def tokenize_batch(self, batch):
        all_prompts = batch['prompt']
        all_responses = batch['response']
        all_rewards = batch['rewards']
        all_sample_log_probs = batch['sample_log_probs']

        all_prompts_repeated = [item for item in all_prompts for _ in range(self.group_size)]
        all_responses_flattened = [item for sublist in all_responses for item in sublist]
        all_rewards_flattened = [item for sublist in all_rewards for item in sublist]
        all_sample_log_probs_flattened = [item for sublist in all_sample_log_probs for item in sublist]
        assert (
            len(all_prompts_repeated)
            == len(all_responses_flattened)
            == len(all_rewards_flattened)
            == len(all_sample_log_probs_flattened)
        )

        self.tokenizer.padding_side = "left"
        tokenized_prompts = self.tokenizer(
            all_prompts_repeated,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="np",
        )
        self.tokenizer.padding_side = "right"
        tokenized_responses = self.tokenizer(
            all_responses_flattened,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_response_length,
            return_tensors="np",
        )

        prompt_input_ids = tokenized_prompts["input_ids"]
        prompt_attention_mask = tokenized_prompts["attention_mask"]
        response_input_ids = tokenized_responses["input_ids"]
        response_attention_mask = tokenized_responses["attention_mask"]
        is_response_token = np.concatenate(
            [np.zeros_like(prompt_input_ids), np.ones_like(response_input_ids)], axis=1
        )
        input_ids = np.concatenate([prompt_input_ids, response_input_ids], axis=1)
        attention_mask = np.concatenate([prompt_attention_mask, response_attention_mask], axis=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_response_token": is_response_token,
            "rewards": np.array(all_rewards_flattened, dtype=np.float32),
            "sample_log_probs": np.array(all_sample_log_probs_flattened, dtype=np.float32),
        }

    def _apply_reward_pipeline(
        self,
        all_rewards,
        all_prompts,
        all_responses,
        all_ground_truth,
        step,
    ):
        """Extension: corrupt -> de-noise -> curriculum-reweight the rewards.

        Operates on prompt-major nested reward lists ``all_rewards[B][G]``.
        Returns ``(processed_rewards, prompt_weights, ext_metrics)`` where
        ``prompt_weights`` is a ``[B]`` array (or ``None`` if no curriculum) and
        ``ext_metrics`` is a dict of diagnostics for logging.

        When no extension feature is enabled this method is never called, so
        vanilla MaxRL rewards are left untouched.
        """
        lengths = [len(g) for g in all_rewards]
        clean_flat = np.asarray([r for g in all_rewards for r in g], dtype=np.float64)
        ext_metrics = {}

        # 1) Symmetric reward-flip noise on the verifier success bit.
        if self.reward_noise is not None:
            noisy_flat, flip_mask = self.reward_noise.corrupt(clean_flat, step=step)
            ext_metrics['ext/flip_frac'] = float(np.mean(flip_mask))
        else:
            noisy_flat = clean_flat.copy()
            flip_mask = np.zeros_like(clean_flat, dtype=bool)

        # 2) RLAIF judge channel + combination rule.
        final_flat = noisy_flat
        if self.use_rlaif and self.judge is not None:
            if self.judge_backend == 'simulated':
                judge_flat, judge_err = self.judge.judge(
                    clean_flat,
                    verifier_flip_mask=flip_mask,
                    correlation=self.judge_correlation,
                    step=step,
                )
                ext_metrics['ext/judge_err_frac'] = float(np.mean(judge_err))
            else:  # vllm
                prompts_repeated = [p for p, g in zip(all_prompts, all_rewards) for _ in g]
                responses_flat = [r for sub in all_responses for r in sub]
                gt_repeated = [gt for gt, g in zip(all_ground_truth, all_rewards) for _ in g]
                judge_flat = self.judge.judge(prompts_repeated, responses_flat, gt_repeated)
            final_flat = combine_rewards(
                noisy_flat,
                judge_flat,
                mode=self.rlaif_combine,
                success_threshold=self.success_threshold,
            )

        # Reshape back to prompt-major groups.
        processed = []
        idx = 0
        for n in lengths:
            processed.append([float(v) for v in final_flat[idx:idx + n]])
            idx += n

        # 3) Noise-aware curriculum: observe this step's (post-pipeline) success
        # rates and emit a per-prompt weight in [w_min, 1].
        prompt_weights = None
        if self.curriculum is not None:
            success_rates = [
                float(np.mean([1.0 if r > self.success_threshold else 0.0 for r in rs]))
                for rs in processed
            ]
            prompt_weights = self.curriculum.step(all_prompts, success_rates)
            ext_metrics['ext/curriculum_weight_mean'] = float(np.mean(prompt_weights))
            ext_metrics['ext/curriculum_weight_min'] = float(np.min(prompt_weights))
            prompt_weights = np.asarray(prompt_weights, dtype=np.float32)

        return processed, prompt_weights, ext_metrics

    def train(self):
        last_checkpoint_dir = None
        global_step = 0
        for epoch in range(self.num_epochs):
            if global_step > 0 and global_step == self.num_training_steps:
                break
            for train_iter, batch in enumerate(self.train_dataloader):
                if global_step > 0 and global_step == self.num_training_steps:
                    break

                # 1) Sample group_size responses per prompt with current policy.
                print(f"Sampling, Epoch {epoch}, Global Step {global_step}")
                model_path = (
                    self.model_name if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "model")
                )
                self._create_sampling_worker(model_path)

                all_prompts = batch['prompt']
                all_ground_truth = batch['ground_truth']
                assert len(all_prompts) == len(all_ground_truth) == self.batch_size
                all_responses, all_sample_log_probs = ray.get(
                    self.sampling_worker.generate.remote(all_prompts)
                )

                # 2) Score sampled responses against task ground truth.
                print(f"Computing rewards, Epoch {epoch}, Global Step {global_step}")
                all_rewards = []
                for curr_responses, curr_ground_truth in zip(all_responses, all_ground_truth):
                    all_rewards.append([compute_score(x, curr_ground_truth) for x in curr_responses])

                # 2b) Extension: corrupt -> RLAIF de-noise -> curriculum reweight.
                # Skipped entirely (no-op) when no extension feature is enabled.
                prompt_weights = None
                ext_metrics = {}
                clean_reward_mean = float(np.mean(all_rewards).item())
                if self._extension_active:
                    all_rewards, prompt_weights, ext_metrics = self._apply_reward_pipeline(
                        all_rewards,
                        all_prompts,
                        all_responses,
                        all_ground_truth,
                        step=global_step,
                    )

                reward_mean = float(np.mean(all_rewards).item())
                # Empirical per-prompt success rate p_hat = mean(1[r > thr]) over the group.
                p_hat_per_prompt = [
                    float(np.mean([1.0 if r > self.success_threshold else 0.0 for r in rs]))
                    for rs in all_rewards
                ]
                active_group_frac = float(np.mean([1.0 if p > 0 else 0.0 for p in p_hat_per_prompt]))
                print(f"Reward mean: {reward_mean:.4f}, active groups: {active_group_frac:.3f}")

                generation_table = self._build_generation_table(all_prompts, all_responses, all_rewards)

                # 3) Tokenize.
                tokenized_batch = self.tokenize_batch({
                    'prompt': all_prompts,
                    'response': all_responses,
                    'rewards': all_rewards,
                    'sample_log_probs': all_sample_log_probs,
                })

                # 4) Update worker with latest checkpoint state.
                optimizer_path = (
                    None if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "optimizer.pt")
                )
                scheduler_path = (
                    None if last_checkpoint_dir is None
                    else os.path.join(last_checkpoint_dir, "scheduler.pt")
                )
                self._create_update_worker(model_path, optimizer_path, scheduler_path)

                # 5) MaxRL update(s).
                all_metrics = None
                for inner_epoch in range(self.ppo_epochs):
                    inner_metrics = ray.get(self.update_worker.update_gradient_accumulation.remote(
                        input_ids=tokenized_batch["input_ids"],
                        attention_mask=tokenized_batch["attention_mask"],
                        is_response_token=tokenized_batch["is_response_token"],
                        rewards=tokenized_batch["rewards"],
                        sample_log_probs=tokenized_batch["sample_log_probs"],
                        prompt_weights=prompt_weights,
                    ))
                    inner_metrics['inner_epoch'] = inner_epoch
                    all_metrics = inner_metrics
                    if self.ppo_epochs > 1:
                        wandb.log(
                            {f'inner/{k}': v for k, v in inner_metrics.items()
                             if isinstance(v, (int, float, np.floating, np.integer))},
                            step=global_step,
                        )

                # Save checkpoint (persistent or scratch).
                if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                    save_dir = os.path.join(
                        self.save_dir, self.wandb_project, self.wandb_name,
                        f"epoch_{epoch}_step_{global_step}",
                    )
                else:
                    save_dir = os.path.join(
                        self.save_dir, self.wandb_project, self.wandb_name, "latest_checkpoint",
                    )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)

                save_model_path = os.path.join(save_dir, "model")
                save_optimizer_path = os.path.join(save_dir, "optimizer.pt")
                save_scheduler_path = os.path.join(save_dir, "scheduler.pt")
                ray.get(self.update_worker.update_checkpoint_paths.remote(
                    model_path=save_model_path,
                    optimizer_path=save_optimizer_path,
                    scheduler_path=save_scheduler_path,
                    load_checkpoint=False,
                ))
                ray.get(self.update_worker.save_checkpoint.remote())
                last_checkpoint_dir = save_dir

                print("-" * 80)
                print(f"Epoch {epoch}, Global Step {global_step}")
                scientific_metric_names = {"lr", "kl_loss"}
                for k, v in all_metrics.items():
                    if isinstance(v, (float, np.floating)):
                        v = float(v)
                        if k in scientific_metric_names or (0 < abs(v) < 1e-4):
                            print(f"{k}: {v:.6e}")
                        else:
                            print(f"{k}: {v:.4f}")
                    else:
                        print(f"{k}: {v}")
                print("-" * 80)

                metrics_logged = {'train/' + k: v for k, v in all_metrics.items()}
                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    "sampling/active_group_frac": active_group_frac,
                    **metrics_logged,
                }
                if self._extension_active:
                    log_dict["sampling/clean_reward_mean"] = clean_reward_mean
                    log_dict.update(ext_metrics)
                if generation_table is not None:
                    log_dict["samples/generations"] = generation_table

                self.wandb.log(log_dict, step=global_step)
                global_step += 1

        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None

        ray.shutdown()
        self.wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='asingh15/qwen-sft-countdown-defaultproj')
    parser.add_argument('--ref_model_name', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4')
    parser.add_argument('--wandb_project', type=str, default='maxrl_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=8)
    parser.add_argument('--entropy_coefficient', type=float, default=0.01)
    parser.add_argument('--kl_divergence_coefficient', type=float, default=0.0)
    parser.add_argument('--num_training_steps', type=int, default=250)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_batched_tokens', type=int, default=8192)
    parser.add_argument('--enable_chunked_prefill', action='store_true')
    parser.add_argument('--disable_chunked_prefill', action='store_true')
    parser.add_argument('--max_num_seqs', type=int, default=64)
    parser.add_argument('--max_table_rows', type=int, default=20)
    parser.add_argument('--save_every_n_steps', type=int, default=-1)
    parser.add_argument('--save_dir', type=str, default='checkpoints/maxrl_checkpoints')
    parser.add_argument('--ppo_epochs', type=int, default=1,
                        help='K: number of inner update epochs per sampled batch (off-policy reuse)')
    parser.add_argument('--importance_weight_clip', type=float, default=5.0,
                        help='Per-sequence importance weight clip (0 disables)')
    parser.add_argument('--success_threshold', type=float, default=0.5,
                        help='Reward threshold above which a rollout counts as a success for the MaxRL weighting.')
    # ---- Extension flags (all default OFF -> standard MaxRL) ----
    parser.add_argument('--reward_noise_p', type=float, default=0.0,
                        help='Symmetric reward-flip probability p on the verifier success bit (0 disables).')
    parser.add_argument('--reward_noise_seed', type=int, default=0,
                        help='Base seed for the reward-flip noise RNG.')
    parser.add_argument('--use_rlaif', action='store_true',
                        help='Enable the RLAIF judge channel to de-noise the verifier.')
    parser.add_argument('--judge_backend', type=str, default='simulated',
                        choices=['simulated', 'vllm'],
                        help='RLAIF judge backend: a controllable simulated judge or a real vLLM LLM judge.')
    parser.add_argument('--rlaif_combine', type=str, default='agree_gate',
                        choices=['agree_gate', 'or', 'majority', 'soft_avg'],
                        help='Rule for combining the verifier and judge channels.')
    parser.add_argument('--judge_error_rate', type=float, default=0.05,
                        help='Simulated-judge marginal error rate q.')
    parser.add_argument('--judge_correlation', type=float, default=0.0,
                        help='Fraction of judge errors coupled to verifier flips (erodes the RLAIF benefit).')
    parser.add_argument('--judge_seed', type=int, default=0,
                        help='Base seed for the simulated judge RNG.')
    parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name for the vLLM judge backend.')
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Enable the noise-aware prompt-reweighting curriculum.')
    parser.add_argument('--curriculum_kappa', type=float, default=1.0,
                        help='Strictness of the curriculum gate in floor standard deviations.')
    parser.add_argument('--curriculum_w_min', type=float, default=0.1,
                        help='Minimum weight assigned to fully-suppressed prompts.')
    parser.add_argument('--curriculum_ema_decay', type=float, default=0.9,
                        help='EMA decay for the per-prompt success-rate estimate.')
    args = parser.parse_args()
    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError("Cannot set both --enable_chunked_prefill and --disable_chunked_prefill.")
    if args.enable_chunked_prefill:
        args.enable_chunked_prefill = True
    elif args.disable_chunked_prefill:
        args.enable_chunked_prefill = False
    else:
        args.enable_chunked_prefill = True
    del args.disable_chunked_prefill

    ray.init()

    trainer = MaxRLTrainer(**vars(args))
    trainer.train()
