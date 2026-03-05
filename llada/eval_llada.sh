#!/bin/bash

# Set the environment variables first before running the command.
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
cd ./llada || exit  # Go to the root directory of the repo

task=${TASK}
length=${LENGTH}
block_length=${BLOCK_LENGTH}
num_fewshot=${NUM_FEWSHOT}
steps=${STEPS}
remasking_only_masked=True
apply_corrector_every_n_steps=${APPLY_CORRECTOR_EVERY_N_STEPS}
max_corrector_steps_per_loop=${MAX_CORRECTOR_STEPS_PER_LOOP}
early_eos_stopping=${EARLY_EOS_STOPPING}
tokenizer_path='GSAI-ML/LLaDA-8B-Instruct'
model_path=${MODEL_PATH}
save_dir="${BASE_SAVE_DIR}/${task}/num_fewshot-${num_fewshot}/length-${length}--block_length-${block_length}--steps-${steps}--early_eos_stopping-${early_eos_stopping}--remasking_only_masked-${remasking_only_masked}--apply_corrector_every_n_steps-${apply_corrector_every_n_steps}--max_corrector_steps_per_loop-${max_corrector_steps_per_loop}"

accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--output_path ${save_dir} --log_samples \
--model_args model_path=${model_path},tokenizer_path=${tokenizer_path},early_eos_stopping=${early_eos_stopping},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,remasking_only_masked=${remasking_only_masked},apply_corrector_every_n_steps=${apply_corrector_every_n_steps},max_corrector_steps_per_loop=${max_corrector_steps_per_loop},save_dir=${save_dir}
