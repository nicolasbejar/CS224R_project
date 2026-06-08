#!/bin/bash
# Modal launcher for the MaxRL *imperfect-reward* extension experiments.
#
# This wraps the same `modal_train.py maxrl` entrypoint as
# `train_maxrl_modal.sh` but additionally threads through the extension flags
# (reward-flip noise, RLAIF agreement-gate de-noising, and the noise-aware
# prompt-reweighting curriculum). With all extension env-vars left at their
# defaults this script trains *vanilla* MaxRL, byte-identical to the base
# launcher; set the EXT_* variables below to enable the extension.
#
# Required before running:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#   export MODEL_NAME=path-or-hf-repo-of-initial-policy
#   export DATASET_NAME=your-countdown-dataset
#
# Example: train MaxRL under p=0.1 flip noise, RLAIF on, curriculum on:
#   REWARD_NOISE_P=0.1 USE_RLAIF=1 JUDGE_ERROR_RATE=0.1 USE_CURRICULUM=1 \
#     ./maxrl_trainer/train_maxrl_noise_modal.sh

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

read -r -a lrs <<< "${LRS:-1e-5}"
num_lrs=${#lrs[@]}

which_exp=${1:-0}
if (( which_exp < 0 || which_exp >= num_lrs )); then
    echo "Error: which_exp must be between 0 and $((num_lrs - 1))"
    exit 1
fi

curr_lr="${lrs[$which_exp]}"

# ---- Base MaxRL hyper-parameters (match train_maxrl_modal.sh) ----
batch_size="${BATCH_SIZE:-128}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-128}"
gradient_clipping="${GRADIENT_CLIPPING:-0.0}"
group_size="${GROUP_SIZE:-8}"
num_training_steps="${NUM_TRAINING_STEPS:-100}"
kl_divergence_coefficient="${KL_DIVERGENCE_COEFFICIENT:-0.001}"
entropy_coefficient="${ENTROPY_COEFFICIENT:-0.001}"
save_every_n_steps="${SAVE_EVERY_N_STEPS:-10}"
lr_schedule="${LR_SCHEDULE:-constant}"
warmup_ratio="${WARMUP_RATIO:-0.0}"
weight_decay="${WEIGHT_DECAY:-1e-4}"
temperature="${TEMPERATURE:-1.0}"
top_k="${TOP_K:--1}"
top_p="${TOP_P:-1.0}"
min_p="${MIN_P:-0.0}"
success_threshold="${SUCCESS_THRESHOLD:-0.5}"

tokenizer_name="${TOKENIZER_NAME:-Qwen/Qwen2.5-0.5B}"
model_name="${MODEL_NAME:-your-org/your-model}"
dataset_name="${DATASET_NAME:-your-org/your-dataset}"
wandb_project="${WANDB_PROJECT:-maxrl_noise_extension}"
save_dir="${SAVE_DIR:-/vol/checkpoints/maxrl_noise_checkpoints}"

# ---- Extension knobs (defaults -> vanilla MaxRL) ----
reward_noise_p="${REWARD_NOISE_P:-0.0}"
reward_noise_seed="${REWARD_NOISE_SEED:-0}"
use_rlaif="${USE_RLAIF:-0}"
judge_backend="${JUDGE_BACKEND:-simulated}"
rlaif_combine="${RLAIF_COMBINE:-agree_gate}"
judge_error_rate="${JUDGE_ERROR_RATE:-0.05}"
judge_correlation="${JUDGE_CORRELATION:-0.0}"
judge_seed="${JUDGE_SEED:-0}"
judge_model="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
use_curriculum="${USE_CURRICULUM:-0}"
curriculum_kappa="${CURRICULUM_KAPPA:-1.0}"
curriculum_w_min="${CURRICULUM_W_MIN:-0.1}"
curriculum_ema_decay="${CURRICULUM_EMA_DECAY:-0.9}"

ext_tag="p${reward_noise_p}"
if [[ "$use_rlaif" == "1" ]]; then ext_tag="${ext_tag}_rlaif${judge_error_rate}_${rlaif_combine}"; fi
if [[ "$use_curriculum" == "1" ]]; then ext_tag="${ext_tag}_curr${curriculum_kappa}"; fi

wandb_name="${WANDB_NAME:-maxrl_neb_lr${curr_lr}_bs${batch_size}_gs${group_size}_thr${success_threshold}_${ext_tag}}"

command=(
    modal run "$PROJECT_ROOT/modal_train.py"
    maxrl
    --model_name "$model_name"
    --ref_model_name "$model_name"
    --tokenizer_name "$tokenizer_name"
    --dataset_name "$dataset_name"
    --wandb_project "$wandb_project"
    --wandb_name "$wandb_name"
    --learning_rate "$curr_lr"
    --batch_size "$batch_size"
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --gradient_clipping "$gradient_clipping"
    --group_size "$group_size"
    --entropy_coefficient "$entropy_coefficient"
    --kl_divergence_coefficient "$kl_divergence_coefficient"
    --num_training_steps "$num_training_steps"
    --lr_schedule "$lr_schedule"
    --save_every_n_steps "$save_every_n_steps"
    --save_dir "$save_dir"
    --warmup_ratio "$warmup_ratio"
    --weight_decay "$weight_decay"
    --temperature "$temperature"
    --top_p "$top_p"
    --top_k "$top_k"
    --min_p "$min_p"
    --success_threshold "$success_threshold"
    --reward_noise_p "$reward_noise_p"
    --reward_noise_seed "$reward_noise_seed"
    --judge_backend "$judge_backend"
    --rlaif_combine "$rlaif_combine"
    --judge_error_rate "$judge_error_rate"
    --judge_correlation "$judge_correlation"
    --judge_seed "$judge_seed"
    --judge_model "$judge_model"
    --curriculum_kappa "$curriculum_kappa"
    --curriculum_w_min "$curriculum_w_min"
    --curriculum_ema_decay "$curriculum_ema_decay"
)

# store_true flags: only pass them when enabled.
if [[ "$use_rlaif" == "1" ]]; then command+=( --use_rlaif ); fi
if [[ "$use_curriculum" == "1" ]]; then command+=( --use_curriculum ); fi

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
