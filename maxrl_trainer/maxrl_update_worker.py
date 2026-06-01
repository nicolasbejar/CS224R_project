"""Ray actor that applies MaxRL policy-gradient updates.

MaxRL (Tajwar et al., 2026) optimizes the maximum-likelihood objective
    J_ML(x) = log p_theta(success | x)
instead of the standard RL objective J_RL(x) = p_theta(success | x).

Concretely, the MaxRL gradient is an importance-weighted version of REINFORCE
in which each rollout is reweighted by 1 / p_theta(success | x):

    grad_theta log p   =   grad_theta E[r]   /   E[r]
                       =   E[ r * grad_theta log pi(y|x) ]   /   E[r]
                       ~   ( 1 / S ) * sum_{i : r_i = 1} grad_theta log pi(y_i | x)

where S is the number of successful rollouts in the group of G samples for a
given prompt. This is the "successful-conditional" estimator described in our
proposal (see Section 3): per-prompt, gradients are averaged only over the
rollouts whose binary success indicator is 1, and prompts with zero successes
in the group contribute no signal (weight = 0).

This file mirrors `rloo_trainer/rloo_update_worker.py` so the rest of the
training pipeline (sampling worker, tokenizer, orchestrator, checkpointing) is
shared. The only methodological change is in the per-sample advantage /
weighting computation inside `update(...)`.
"""

import os
import warnings
import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")


@ray.remote(num_gpus=1)
class MaxRLUpdateWorker:
    """Owns policy/ref models and optimizer state for MaxRL updates."""

    def __init__(
        self,
        model_path,
        optimizer_path,
        scheduler_path,
        tokenizer_path=None,
        ref_model_path=None,
        batch_size=64,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        group_size=16,
        entropy_coefficient=0.01,
        kl_divergence_coefficient=0.0,
        lr_schedule='constant',
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.0,
        num_training_steps=250,
        importance_weight_clip=5.0,
        success_threshold=0.5,
    ):
        self.model_path = model_path
        self.ref_model_path = ref_model_path if ref_model_path is not None else model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.group_size = group_size
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for MaxRL, got {self.group_size}")
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if warmup_ratio > 0:
            raise NotImplementedError("Warmup ratio > 0 is not supported for constant learning rate schedule")
        self.num_training_steps = num_training_steps
        self.importance_weight_clip = float(importance_weight_clip)
        # Threshold above which a (possibly soft) reward is treated as a success.
        # Countdown rewards are in {0.0, 0.1, 1.0}; 0.5 selects fully-correct only.
        self.success_threshold = float(success_threshold)

    def tear_down(self):
        """Release model/optimizer objects and clear GPU memory."""
        import gc
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ref_model'):
            del self.ref_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_checkpoint_paths(self, model_path, optimizer_path, scheduler_path, load_checkpoint=False):
        """Update output paths (and optionally reload state immediately)."""
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        if load_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load policy model, optional reference model, and optimizer/scheduler."""
        self.tear_down()

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")
        self.model.gradient_checkpointing_enable()

        if self.kl_divergence_coefficient > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device="cuda")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        if (
            self.optimizer_path
            and self.scheduler_path
            and os.path.exists(self.optimizer_path)
            and os.path.exists(self.scheduler_path)
        ):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")
            self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")

        self.model.train()

    def save_checkpoint(self):
        """Persist optimizer/scheduler state plus model+tokenizer weights."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        torch.save(self.scheduler.state_dict(), self.scheduler_path)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def update_gradient_accumulation(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        prompt_weights: Optional[np.ndarray] = None,
        device='cuda',
    ):
        """Split incoming batch into microbatches and call `update(...)`.

        ``prompt_weights`` (extension): an optional ``[B]`` array of per-prompt
        curriculum weights in ``[0, 1]`` that scale each prompt's MaxRL gradient
        contribution. ``None`` (default) leaves the standard MaxRL behaviour
        unchanged.
        """
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            # Ensure each microbatch still contains full groups so the per-prompt
            # success counts (which set the MaxRL weights) are well defined.
            assert group_per_gradient_accumulation_step % self.group_size == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size "
                f"{self.group_size} when using gradient_accumulation_steps={self.gradient_accumulation_steps}."
            )
            prompts_per_step = group_per_gradient_accumulation_step // self.group_size
            all_metrics = []
            for i in range(self.gradient_accumulation_steps):
                lo = i * group_per_gradient_accumulation_step
                hi = (i + 1) * group_per_gradient_accumulation_step
                curr_sample_log_probs = sample_log_probs[lo:hi] if sample_log_probs is not None else None
                if prompt_weights is not None:
                    p_lo = i * prompts_per_step
                    p_hi = (i + 1) * prompts_per_step
                    curr_prompt_weights = prompt_weights[p_lo:p_hi]
                else:
                    curr_prompt_weights = None
                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    input_ids[lo:hi],
                    attention_mask[lo:hi],
                    is_response_token[lo:hi],
                    rewards[lo:hi],
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                    prompt_weights=curr_prompt_weights,
                )
                all_metrics.append(curr_update_metrics)
            update_metrics = {}
            for metric_name in all_metrics[0].keys():
                update_metrics[metric_name] = np.mean(
                    [metric[metric_name] for metric in all_metrics]
                ).item()
        else:
            update_metrics = self.update(
                input_ids,
                attention_mask,
                is_response_token,
                rewards,
                sample_log_probs,
                True,
                device,
                prompt_weights=prompt_weights,
            )

        return update_metrics

    def update(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        is_update_step: bool = True,
        device='cuda',
        importance_weight_clip: Optional[float] = None,
        prompt_weights: Optional[np.ndarray] = None,
    ):
        """One MaxRL policy gradient update with successful-conditional weights.

        Inputs are flattened over (batch, group): rows i*G..(i+1)*G belong to
        the i-th prompt's group.
        """
        if not hasattr(self, 'model'):
            self.load_checkpoint()

        if importance_weight_clip is None:
            importance_weight_clip = self.importance_weight_clip

        # ---- Tensor setup ----
        input_ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
        is_response_token_t = torch.as_tensor(is_response_token, dtype=torch.long, device=device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        sample_log_probs_t = (
            torch.as_tensor(sample_log_probs, dtype=torch.float32, device=device)
            if sample_log_probs is not None else None
        )

        N = input_ids_t.shape[0]
        G = self.group_size
        assert N % G == 0, f"flattened batch {N} must be divisible by group_size {G}"
        B = N // G

        # ---- Forward pass under current policy ----
        self.model.train()
        outputs = self.model(input_ids=input_ids_t, attention_mask=attention_mask_t)
        logits = outputs.logits  # [N, T, V]

        # Predict token t+1 from position t.
        shift_logits = logits[..., :-1, :].float()
        shift_labels = input_ids_t[..., 1:]
        shift_mask = is_response_token_t[..., 1:].to(shift_logits.dtype)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [N, T-1]

        # Per-token entropy of policy (used for entropy bonus).
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)  # [N, T-1]

        masked_token_logp = token_logp * shift_mask
        seq_logp = masked_token_logp.sum(dim=-1)  # [N]
        token_count = shift_mask.sum(dim=-1).clamp(min=1.0)  # [N]
        seq_entropy = (token_entropy * shift_mask).sum(dim=-1) / token_count  # [N]

        # ---- MaxRL successful-conditional weighting ----
        # success_i in {0,1}; per-prompt S = sum of successes within group.
        success = (rewards_t > self.success_threshold).float()  # [N]
        success_grp = success.view(B, G)
        S_grp = success_grp.sum(dim=1, keepdim=True)  # [B, 1]
        # Avoid divide-by-zero: prompts with S=0 contribute 0 (weights stay 0).
        safe_S = S_grp.clamp(min=1.0)
        # Weight is the importance-style 1/p_hat = G/S divided by G samples-per-prompt,
        # giving 1/S per successful sample. Failed samples contribute 0.
        weights_grp = success_grp / safe_S  # [B, G], sums to <=1 per group
        # Zero out groups with no successes explicitly (keeps shape, kills signal).
        active = (S_grp > 0).float()  # [B, 1]
        weights_grp = weights_grp * active
        # ---- Extension: curriculum per-prompt re-weighting ----
        # Scale each prompt's MaxRL contribution by an optional curriculum weight
        # in [0, 1]. Default (None) -> all-ones, leaving standard MaxRL unchanged.
        if prompt_weights is not None:
            pw = torch.as_tensor(prompt_weights, dtype=torch.float32, device=device).view(B, 1)
            weights_grp = weights_grp * pw
            curriculum_weight_mean = float(pw.mean().item())
        else:
            curriculum_weight_mean = 1.0
        weights = weights_grp.reshape(N).detach()  # [N]

        # ---- Importance weighting (per-sequence; for off-policy reuse) ----
        if sample_log_probs_t is not None:
            log_iw = (seq_logp.detach() - sample_log_probs_t)
            iw = torch.exp(log_iw)
            if importance_weight_clip is not None and importance_weight_clip > 0:
                iw = iw.clamp(max=float(importance_weight_clip))
            iw_mean = iw.mean().item()
            iw_max = iw.max().item()
        else:
            iw = torch.ones_like(weights)
            iw_mean = 1.0
            iw_max = 1.0

        # ---- MaxRL policy gradient ----
        # Average gradient is sum_i w_i * grad log pi(y_i|x_i) / B (per-prompt mean).
        # Using per-token mean log-prob keeps the loss on a length-normalized scale,
        # matching the RLOO worker's convention.
        per_seq_logp_mean = seq_logp / token_count
        # Divide by B so the loss magnitude is independent of group_size B*G.
        pg_loss = -(iw.detach() * weights * per_seq_logp_mean).sum() / float(max(B, 1))

        # ---- Entropy bonus (encourage exploration on noisy reward tasks) ----
        entropy_bonus = seq_entropy.mean()
        entropy_loss = -self.entropy_coefficient * entropy_bonus

        # ---- Optional KL penalty to reference model ----
        kl_value = 0.0
        kl_loss = torch.zeros((), device=device)
        if self.kl_divergence_coefficient > 0 and hasattr(self, 'ref_model'):
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids_t, attention_mask=attention_mask_t)
                ref_logits = ref_outputs.logits[..., :-1, :].float()
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_token_logp = ref_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # k3 estimator of KL(pi || pi_ref) at the realized tokens.
            log_ratio = (token_logp - ref_token_logp) * shift_mask
            kl_per_token = (log_ratio.exp() - 1.0) - log_ratio
            kl_per_seq = kl_per_token.sum(dim=-1) / token_count
            kl_loss = self.kl_divergence_coefficient * kl_per_seq.mean()
            kl_value = kl_per_seq.mean().item()

        loss = pg_loss + entropy_loss + kl_loss

        # ---- Backward (scaled for grad accumulation) ----
        scaled_loss = loss / float(self.gradient_accumulation_steps)
        scaled_loss.backward()

        if is_update_step:
            if self.gradient_clipping is not None and self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        # ---- Diagnostics ----
        n_active_groups = float(active.sum().item())
        active_group_frac = n_active_groups / float(max(B, 1))
        # Per-prompt empirical success rate p_hat = S/G.
        p_hat = (S_grp.squeeze(1) / float(G))
        metrics = {
            'loss': float(loss.detach().item()),
            'pg_loss': float(pg_loss.detach().item()),
            'entropy': float(entropy_bonus.detach().item()),
            'kl_loss': float(kl_value),
            'iw_mean': float(iw_mean),
            'iw_max': float(iw_max),
            'reward_mean': float(rewards_t.mean().item()),
            'success_rate': float(success.mean().item()),
            'p_hat_mean': float(p_hat.mean().item()),
            'p_hat_min': float(p_hat.min().item()),
            'p_hat_max': float(p_hat.max().item()),
            'active_group_frac': float(active_group_frac),
            'curriculum_weight_mean': float(curriculum_weight_mean),
            'weight_mean_active': float(
                (weights.sum() / max(float(success.sum().item()), 1.0)).item()
            ),
            'seq_logp_mean': float(seq_logp.detach().mean().item()),
            'lr': float(self.scheduler.get_last_lr()[0]) if hasattr(self, 'scheduler') else 0.0,
        }
        return metrics
