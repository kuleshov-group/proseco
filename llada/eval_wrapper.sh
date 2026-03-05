#!/bin/bash

WATCH_FOLDER=$(realpath "../watch_folder")
mkdir -p ${WATCH_FOLDER}
NUM_VISIBLE_DEVICES=1

BASE_SAVE_DIR="${PWD}/outputs"
MODEL="corrector"  # "corrector" or "instruct"
TASK="gsm8k"  # "gsm8k", "humaneval", "mbpp"
BLOCK_LENGTH=32
EARLY_EOS_STOPPING=True
LENGTH=256
NUM_FEWSHOT=5


if [[ ${MODEL} == "corrector" ]]; then
  MODEL_PATH='kuleshov-group/proseco-llada-sft'
elif [[ ${MODEL} == "instruct" ]]; then
  MODEL_PATH='GSAI-ML/LLaDA-8B-Instruct'
fi
BASE_SAVE_DIR="${BASE_SAVE_DIR}/${MODEL}"

for STEPS in 32 64 128 256; do
  for APPLY_CORRECTOR_EVERY_N_STEPS in 1 2 4 8; do
    for MAX_CORRECTOR_STEPS_PER_LOOP in 1 2 4 8 16 32; do

    EXPORT_STR="ALL,TASK=${TASK},NUM_FEWSHOT=${NUM_FEWSHOT},LENGTH=${LENGTH},BLOCK_LENGTH=${BLOCK_LENGTH},EARLY_EOS_STOPPING=${EARLY_EOS_STOPPING}"

    EXPORT_STR="${EXPORT_STR},BASE_SAVE_DIR=${BASE_SAVE_DIR},MODEL_PATH=${MODEL_PATH},STEPS=${STEPS},APPLY_CORRECTOR_EVERY_N_STEPS=${APPLY_CORRECTOR_EVERY_N_STEPS},MAX_CORRECTOR_STEPS_PER_LOOP=${MAX_CORRECTOR_STEPS_PER_LOOP}"
    JOB_NAME="${MODEL}-${TASK}-${NUM_FEWSHOT}shot_L-${LENGTH}_T-${STEPS}_F-${APPLY_CORRECTOR_EVERY_N_STEPS}_S-${MAX_CORRECTOR_STEPS_PER_LOOP}"

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
      "$(realpath "./eval_llada.sh")"

#    # Interactive
#    TASK=${TASK} \
#    NUM_FEWSHOT=${NUM_FEWSHOT} \
#    LENGTH=${LENGTH} \
#    BLOCK_LENGTH=${BLOCK_LENGTH} \
#    EARLY_EOS_STOPPING=${EARLY_EOS_STOPPING} \
#    BASE_SAVE_DIR=${BASE_SAVE_DIR} \
#    MODEL_PATH=${MODEL_PATH} \
#    STEPS=${STEPS} \
#    APPLY_CORRECTOR_EVERY_N_STEPS=${APPLY_CORRECTOR_EVERY_N_STEPS} \
#    MAX_CORRECTOR_STEPS_PER_LOOP=${MAX_CORRECTOR_STEPS_PER_LOOP} \
#    "$(realpath "./eval_llada.sh")"
    done
  done
done
