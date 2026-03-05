#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 96:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu      #
#SBATCH --constraint="[h200|h100|a100|a6000|a5000]"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

# Expecting:
#  - USE_HF (whether to use HF model)
#  - CKPT (path to ckpt dir; not needed if using HF model)
#  - CKPT_FILE (name of ckpt file; not needed if using HF model)
#  - BATCH_SIZE (optional: default = 1)
#  - NUM_SAMPLES (optional: default = 5000)
#  - SAMPLING_STEPS (optional: default = 1024)
#  - NUCLEUS_P (optional: default = 0.9)
#  - SEED (optional: default = 1)
#  - USE_FLOAT64 (optional: default = True)
#  - CORRECTOR_SAMPLING (optional: default = argmax)
#  - CORRECTOR_EVERY_N_STEPS (optional: default = 1)
#  - CORRECTOR_STEPS (optional: default = 0)
#  - CORRECTOR_TOP_K (optional: default = 0)

if [ -z ${BATCH_SIZE} ]; then
  BATCH_SIZE=1
fi
if [ -z ${NUM_SAMPLES} ]; then
  NUM_SAMPLES=5000
fi
if [ -z "${SAMPLING_STEPS}" ]; then
  SAMPLING_STEPS=1024
fi
if [ -z "${NUCLEUS_P}" ]; then
  NUCLEUS_P=0.9
fi
if [ -z "${SEED}" ]; then
  SEED=1
fi
if [ -z "${USE_FLOAT64}" ]; then
  USE_FLOAT64=True
fi
if [ -z "${CORRECTOR_SAMPLING}" ]; then
  CORRECTOR_SAMPLING="argmax"
fi
if [ -z "${CORRECTOR_START_ITER}" ]; then
  CORRECTOR_START_ITER=0
fi
if [ -z "${CORRECTOR_EVERY_N_STEPS}" ]; then
  CORRECTOR_EVERY_N_STEPS=1
fi
if [ -z "${CORRECTOR_STEPS}" ]; then
  CORRECTOR_STEPS=0
fi
if [ -z "${CORRECTOR_TOP_K}" ]; then
  CORRECTOR_TOP_K=0
fi

BACKBONE="dit"
parameterization="subs"
subs_masking=False
diffusion="absorbing_state"
TRAIN_T=0
time_conditioning=False
sampling_use_cache=True
if [[ "${USE_HF}" == "True" ]]; then
  CKPT="kuleshov-group/proseco-owt"
  generated_seqs_path="./outputs/owt/corrector/eval_samples/num_samples-${NUM_SAMPLES}--eval_float64--${USE_FLOAT64}--T-${SAMPLING_STEPS}--corrector_start_iter-${CORRECTOR_START_ITER}--corrector_sampling-${CORRECTOR_SAMPLING}--corrector_steps-${CORRECTOR_STEPS}--corrector_every_n_steps-${CORRECTOR_EVERY_N_STEPS}--corrector_top_k-${CORRECTOR_TOP_K}--nucleus_p-${NUCLEUS_P}--seed-${SEED}"
  BACKBONE="hf_dit"  # override to use HF model
else
  if [ -z "${CKPT}" ]; then
    CKPT="<CKPT_DIR>"  # TODO change this
  fi
  if [ -z "${CKPT_FILE}" ]; then
    CKPT_FILE="checkpoints/<CKPT_FILE>"  # TODO: change this
  fi
  generated_seqs_path="${CKPT}/eval_samples/num_samples-${NUM_SAMPLES}--eval_float64--${USE_FLOAT64}--T-${SAMPLING_STEPS}--corrector_start_iter-${CORRECTOR_START_ITER}--corrector_sampling-${CORRECTOR_SAMPLING}--corrector_steps-${CORRECTOR_STEPS}--corrector_every_n_steps-${CORRECTOR_EVERY_N_STEPS}--corrector_top_k-${CORRECTOR_TOP_K}--nucleus_p-${NUCLEUS_P}--seed-${SEED}"
fi
mkdir -p "${generated_seqs_path}"
if [ -v CKPT_FILE ]; then
  CKPT="${CKPT}/${CKPT_FILE}"
fi

NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# shellcheck disable=SC2086
PORT=29504
torchrun --nproc_per_node ${NUM_DEVICES} --master_port=${PORT} main.py \
    hydra.output_subdir=null \
    hydra.run.dir="${PWD}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    seed=${SEED} \
    mode="sample_eval" \
    eval.checkpoint_path="${CKPT}" \
    data=openwebtext-split \
    backbone="${BACKBONE}" \
    model=small \
    model.length=1024 \
    training.guidance=null \
    parameterization=${parameterization} \
    subs_masking=${subs_masking} \
    diffusion=${diffusion} \
    time_conditioning=${time_conditioning} \
    T=${TRAIN_T} \
    loader.global_batch_size=$(( BATCH_SIZE * NUM_DEVICES )) \
    loader.eval_batch_size=1 \
    sampling.num_sample_batches=$(( NUM_SAMPLES / BATCH_SIZE / NUM_DEVICES )) \
    sampling.steps=${SAMPLING_STEPS} \
    sampling.corrector_sampling=${CORRECTOR_SAMPLING} \
    sampling.corrector_start_iter=${CORRECTOR_START_ITER} \
    sampling.corrector_every_n_steps=${CORRECTOR_EVERY_N_STEPS} \
    sampling.corrector_steps=${CORRECTOR_STEPS} \
    sampling.corrector_top_k=${CORRECTOR_TOP_K} \
    sampling.use_cache=${sampling_use_cache} \
    sampling.use_float64=${USE_FLOAT64} \
    sampling.nucleus_p=${NUCLEUS_P} \
    eval.max_samples=${NUM_SAMPLES} \
    eval.generated_samples_path=${generated_seqs_path} \
    +eval.generative_ppl_model_name_or_path="gpt2-large" \
    +eval.mauve_p_features_path="/share/kuleshov/yzs2/nvidia-collab/human_reference_mauve_featurized.npy" \
    wandb.group="gen_eval" \
    wandb.id=null \
    wandb.tags=null
