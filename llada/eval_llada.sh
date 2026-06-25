#!/bin/bash

# Launch a nemo_skills benchmark evaluation for a LLaDA + ProSeCo diffusion model.
#
# This drives the nemo_skills-based evaluation harness (eval_llada.py).
#
# It is driven by environment variables (see eval_wrapper.sh for a sweep
# launcher) and falls back to sensible defaults so it can also be run directly.

# Run this script from the repo root (the parent of the llada folder), e.g.:
#     bash llada/eval_llada.sh

source ./setup_env.sh || exit
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# nemo_skills benchmark name: gsm8k, human-eval, mbpp, hendrycks_math, ...
benchmark=${BENCHMARK:-human-eval}
gen_length=${LENGTH:-1024}
block_length=${BLOCK_LENGTH:-32}
steps=${STEPS:-1024}
apply_corrector_every_n_steps=${APPLY_CORRECTOR_EVERY_N_STEPS:-2}
max_corrector_steps_per_loop=${MAX_CORRECTOR_STEPS_PER_LOOP:-4}
early_eos_stopping=${EARLY_EOS_STOPPING:-True}
threshold=${THRESHOLD:-None}
tokenizer_path=${TOKENIZER_PATH:-'GSAI-ML/LLaDA-8B-Instruct'}
model_path=${MODEL_PATH:-'kuleshov-group/proseco-llada-sft'}
prompt_config=${PROMPT_CONFIG:-"${PWD}/llada/prompt_configs/code.yaml"}
BASE_SAVE_DIR=${BASE_SAVE_DIR:-"${PWD}/llada/outputs"}
max_samples_arg=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  max_samples_arg=(--max_samples "${MAX_SAMPLES}")
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
  num_gpus="${NUM_GPUS}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  num_gpus="${SLURM_GPUS_ON_NODE%%(*}"
elif [[ -n "${SLURM_NTASKS_PER_NODE:-}" ]]; then
  num_gpus="${SLURM_NTASKS_PER_NODE%%(*}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  IFS=',' read -r -a cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
  num_gpus="${#cuda_devices[@]}"
else
  num_gpus=1
fi

save_dir="${BASE_SAVE_DIR}/${benchmark}/length-${gen_length}--block_length-${block_length}--steps-${steps}--early_eos_stopping-${early_eos_stopping}--apply_corrector_every_n_steps-${apply_corrector_every_n_steps}--max_corrector_steps_per_loop-${max_corrector_steps_per_loop}"

# Data-parallel: accelerate launches one process per GPU and eval_llada.py
# shards the dataset across them (>1 process auto-enables multi-GPU).
accelerate launch --num_processes "${num_gpus}" llada/eval_llada.py \
  --benchmark "${benchmark}" \
  --model_path "${model_path}" \
  --tokenizer_path "${tokenizer_path}" \
  --gen_length "${gen_length}" \
  --block_length "${block_length}" \
  --steps "${steps}" \
  --threshold "${threshold}" \
  --apply_corrector_every_n_steps "${apply_corrector_every_n_steps}" \
  --max_corrector_steps_per_loop "${max_corrector_steps_per_loop}" \
  --early_eos_stopping "${early_eos_stopping}" \
  --output_dir "${save_dir}" \
  --prompt_config "${prompt_config}" \
  "${max_samples_arg[@]}"
