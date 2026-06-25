#!/bin/bash

# Sweep launcher for the nemo_skills-based evaluation (eval_llada.sh).
#
# Run this from the repo root (the parent of the llada folder).
#
# Sweeps over the two corrector hyperparameters:
#     apply_corrector_every_n_steps
#     max_corrector_steps_per_loop
# submitting one slurm job per configuration.

WATCH_FOLDER=$(realpath "./watch_folder")
mkdir -p "${WATCH_FOLDER}"
NUM_VISIBLE_DEVICES=1

BASE_SAVE_DIR="${PWD}/llada/outputs"
MODEL="corrector"  # "corrector" or "instruct"
BENCHMARK="human-eval"  # nemo_skills name: "gsm8k", "human-eval", "mbpp", "hendrycks_math"
PROMPT_CONFIG="${PWD}/llada/prompt_configs/code.yaml"   # optional nemo_skills prompt-config name or yaml path
BLOCK_LENGTH=32
EARLY_EOS_STOPPING=True
LENGTH=1024
THRESHOLD=None

if [[ ${MODEL} == "corrector" ]]; then
  MODEL_PATH='kuleshov-group/proseco-llada-sft'
elif [[ ${MODEL} == "instruct" ]]; then
  MODEL_PATH='GSAI-ML/LLaDA-8B-Instruct'
fi
BASE_SAVE_DIR="${BASE_SAVE_DIR}/${MODEL}"

for STEPS in 128 256 512 1024; do
  for APPLY_CORRECTOR_EVERY_N_STEPS in 1 2 4 8; do
    for MAX_CORRECTOR_STEPS_PER_LOOP in 1 2 4 8; do

    EXPORT_STR="ALL,BENCHMARK=${BENCHMARK},PROMPT_CONFIG=${PROMPT_CONFIG},LENGTH=${LENGTH},BLOCK_LENGTH=${BLOCK_LENGTH},EARLY_EOS_STOPPING=${EARLY_EOS_STOPPING},THRESHOLD=${THRESHOLD},NUM_GPUS=${NUM_VISIBLE_DEVICES}"
    EXPORT_STR="${EXPORT_STR},BASE_SAVE_DIR=${BASE_SAVE_DIR},MODEL_PATH=${MODEL_PATH},STEPS=${STEPS},APPLY_CORRECTOR_EVERY_N_STEPS=${APPLY_CORRECTOR_EVERY_N_STEPS},MAX_CORRECTOR_STEPS_PER_LOOP=${MAX_CORRECTOR_STEPS_PER_LOOP}"
    JOB_NAME="${MODEL}-${BENCHMARK}_L-${LENGTH}_T-${STEPS}_F-${APPLY_CORRECTOR_EVERY_N_STEPS}_S-${MAX_CORRECTOR_STEPS_PER_LOOP}"

    sbatch \
      --job-name="${JOB_NAME}" \
      --output="${WATCH_FOLDER}/%x_%j.log" \
      --open-mode=append \
      --get-user-env \
      --constraint="[h200|h100|a100|a6000]" \
      --time=960:00:00 \
      --mem=128000 \
      --nodes=1 \
      --ntasks-per-node=${NUM_VISIBLE_DEVICES} \
      --gres=gpu:${NUM_VISIBLE_DEVICES} \
      --mail-type=ALL \
      --requeue \
      --export="${EXPORT_STR}" \
      "$(realpath "./llada/eval_llada.sh")"

  #  # Interactive
  #  BENCHMARK=${BENCHMARK} \
  #  PROMPT_CONFIG=${PROMPT_CONFIG} \
  #  LENGTH=${LENGTH} \
  #  BLOCK_LENGTH=${BLOCK_LENGTH} \
  #  EARLY_EOS_STOPPING=${EARLY_EOS_STOPPING} \
  #  THRESHOLD=${THRESHOLD} \
  #  BASE_SAVE_DIR=${BASE_SAVE_DIR} \
  #  MODEL_PATH=${MODEL_PATH} \
  #  STEPS=${STEPS} \
  #  APPLY_CORRECTOR_EVERY_N_STEPS=${APPLY_CORRECTOR_EVERY_N_STEPS} \
  #  MAX_CORRECTOR_STEPS_PER_LOOP=${MAX_CORRECTOR_STEPS_PER_LOOP} \
  #  "$(realpath "./llada/eval_llada.sh")"
    done
  done
done
