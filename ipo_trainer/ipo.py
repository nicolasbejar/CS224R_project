"""Starter IPO training entrypoint for the class project.

This script wires model loading, data loading, and optimizer setup.
Students are expected to implement `train(...)` for the IPO objective.
"""

import sys
from pathlib import Path

# Allow `python ipo_trainer/ipo.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from ipo_trainer.ipo_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
import copy
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device, use_gradient_checkpointing=True):
    """Load trainable policy model and frozen reference model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()

    # IPO compares policy preferences to a fixed baseline policy.
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return model, tokenizer, reference_model

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def _sequence_logprobs(model, input_ids, attention_mask, is_response_token,
                       average_logps=False, no_grad=False):
    """Compute summed (or averaged) log-prob of response tokens for each example.

    Returns a tensor of shape [B] (per-sequence log-prob).
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Predict token t+1 from position t.
        shift_logits = logits[..., :-1, :].float()
        shift_labels = input_ids[..., 1:]
        shift_mask = is_response_token[..., 1:].to(shift_logits.dtype)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * shift_mask  # zero out non-response positions

        seq_logp_sum = token_logp.sum(dim=-1)
        if average_logps:
            counts = shift_mask.sum(dim=-1).clamp(min=1.0)
            return seq_logp_sum / counts
        return seq_logp_sum


@torch.no_grad()
def evaluate_ipo(model, reference_model, dataloader, device, beta,
                 average_logps, loss_type, max_batches=None):
    """Evaluate IPO/DPO loss + reward margin on a held-out split."""
    model.eval()
    total_loss = 0.0
    total_margin = 0.0
    total_chosen_rw = 0.0
    total_rejected_rw = 0.0
    total_acc = 0.0
    n = 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        ids_w = batch['input_ids_w'].to(device)
        am_w = batch['attention_mask_w'].to(device)
        rt_w = batch['is_response_token_w'].to(device)
        ids_l = batch['input_ids_l'].to(device)
        am_l = batch['attention_mask_l'].to(device)
        rt_l = batch['is_response_token_l'].to(device)

        pol_w = _sequence_logprobs(model, ids_w, am_w, rt_w, average_logps, no_grad=True)
        pol_l = _sequence_logprobs(model, ids_l, am_l, rt_l, average_logps, no_grad=True)
        ref_w = _sequence_logprobs(reference_model, ids_w, am_w, rt_w, average_logps, no_grad=True)
        ref_l = _sequence_logprobs(reference_model, ids_l, am_l, rt_l, average_logps, no_grad=True)

        h = (pol_w - ref_w) - (pol_l - ref_l)
        chosen_rw = beta * (pol_w - ref_w)
        rejected_rw = beta * (pol_l - ref_l)
        margin = chosen_rw - rejected_rw
        if loss_type == 'ipo':
            loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
        else:  # dpo
            loss = -F.logsigmoid(beta * h).mean()
        acc = (margin > 0).float().mean()

        bs = ids_w.size(0)
        total_loss += loss.item() * bs
        total_margin += margin.mean().item() * bs
        total_chosen_rw += chosen_rw.mean().item() * bs
        total_rejected_rw += rejected_rw.mean().item() * bs
        total_acc += acc.item() * bs
        n += bs
    model.train()
    if n == 0:
        return {}
    return {
        'loss': total_loss / n,
        'reward_margin': total_margin / n,
        'chosen_reward': total_chosen_rw / n,
        'rejected_reward': total_rejected_rw / n,
        'pref_acc': total_acc / n,
    }


def train(
    model, 
    tokenizer, 
    reference_model,
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0,
    beta=0.1,
    average_logps=False,
    loss_type='ipo',
    eval_every=50,
):
    """Pairwise preference optimization loop (DPO or IPO)."""
    model.train()
    reference_model.eval()
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        pbar = tqdm.tqdm(train_dataloader, desc=f"epoch {epoch}")
        for batch in pbar:
            ids_w = batch['input_ids_w'].to(device)
            am_w = batch['attention_mask_w'].to(device)
            rt_w = batch['is_response_token_w'].to(device)
            ids_l = batch['input_ids_l'].to(device)
            am_l = batch['attention_mask_l'].to(device)
            rt_l = batch['is_response_token_l'].to(device)

            # Policy log-probs (gradient).
            pol_w = _sequence_logprobs(model, ids_w, am_w, rt_w, average_logps)
            pol_l = _sequence_logprobs(model, ids_l, am_l, rt_l, average_logps)
            # Reference log-probs (no gradient, frozen reference).
            with torch.no_grad():
                ref_w = _sequence_logprobs(reference_model, ids_w, am_w, rt_w, average_logps, no_grad=True)
                ref_l = _sequence_logprobs(reference_model, ids_l, am_l, rt_l, average_logps, no_grad=True)

            # h = log pi/ref(y_w|x) - log pi/ref(y_l|x)
            h = (pol_w - ref_w) - (pol_l - ref_l)
            chosen_rw = beta * (pol_w - ref_w)
            rejected_rw = beta * (pol_l - ref_l)
            margin = (chosen_rw - rejected_rw).detach()

            if loss_type == 'ipo':
                # IPO: squared deviation from 1/(2 beta).
                loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
            elif loss_type == 'dpo':
                loss = -F.logsigmoid(beta * h).mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            (loss / gradient_accumulation_steps).backward()
            micro_step += 1

            if micro_step % gradient_accumulation_steps == 0:
                if gradient_clipping is not None and gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                wandb.log({
                    'train/loss': loss.detach().float().item(),
                    'train/reward_margin': margin.mean().item(),
                    'train/chosen_reward': chosen_rw.detach().mean().float().item(),
                    'train/rejected_reward': rejected_rw.detach().mean().float().item(),
                    'train/pref_acc': (margin > 0).float().mean().item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                }, step=global_step)

                if eval_every > 0 and global_step % eval_every == 0:
                    eval_metrics = evaluate_ipo(
                        model, reference_model, test_dataloader, device,
                        beta, average_logps, loss_type, max_batches=10,
                    )
                    wandb.log({f'test/{k}': v for k, v in eval_metrics.items()}, step=global_step)
                    pbar.set_postfix(loss=loss.item(), margin=margin.mean().item())

        full_metrics = evaluate_ipo(
            model, reference_model, test_dataloader, device,
            beta, average_logps, loss_type,
        )
        wandb.log({f'test/epoch_{k}': v for k, v in full_metrics.items()}, step=global_step)
        print(f"[epoch {epoch}] full test {full_metrics}")

    if save_model:
        save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)
    clear_cache(model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4-dpo')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--average_logps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dpo')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        reference_model,
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping,
        args.beta,
        args.average_logps,
        args.loss_type
    )

if __name__ == "__main__":
    main()
