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

# Network settings
export NCCL_IB_SL="${NCCL_IB_SL:-1}"

# export NCCL_DEBUG="${NCCL_DEBUG:-OFF}"
export NCCL_DEBUG=OFF
export NCCL_P2P_LEVEL=NVL

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-25001}"

# Node settings
export NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
export CURRENT_RANK="${SLURM_PROCID:-0}"
export NPROC="${NPROC:-${SLURM_JOB_NUM_NODES}}"

RUN_NAME="owt_mdlm_corrector"
DATA_CACHE_DIR="<DATA_CACHE_DIR>"  # TODO: Set this
RUN_DIR="${PWD}/outputs/owt/${RUN_NAME}"
mkdir -p "${RUN_DIR}/checkpoints"

DIFFUSION="absorbing_state"
PARAMETERIZATION="d3pm"
SUBS_MASKING=True
T=0
TIME_COND=False
ZERO_RECON_LOSS=False
USE_MODEL_OUTPUTS_AS_CORRECTOR_INPUT=True
USE_MODEL_ARGMAX_OUTPUT=True
USE_WEIGHTED_CORRECTOR_LOSS=True
CORRECTOR_TRAINING_START_STEP=0
MDLM_LOSS_WEIGHT=1.0
CORRECTOR_LOSS_WEIGHT=1.0
CORRECTOR_LOSS_ERRORS_UPWEIGHTED=True
SAMPLING_EPS_TRAINING=1e-1


# To enable preemption re-loading, set `hydra.run.dir` or
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NPROC --master_port=$MASTER_PORT --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK main.py \
  corrector_training=True \
  use_model_outputs_as_corrector_input=${USE_MODEL_OUTPUTS_AS_CORRECTOR_INPUT} \
  use_model_argmax_output=${USE_MODEL_ARGMAX_OUTPUT} \
  corrector_training_start_step=${CORRECTOR_TRAINING_START_STEP} \
  use_weighted_corrector_loss=${USE_WEIGHTED_CORRECTOR_LOSS} \
  mdlm_loss_weight=${MDLM_LOSS_WEIGHT} \
  corrector_loss_weight=${CORRECTOR_LOSS_WEIGHT} \
  corrector_loss_errors_upweighted=${CORRECTOR_LOSS_ERRORS_UPWEIGHTED} \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  subs_masking=${SUBS_MASKING} \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=openwebtext-split \
  data.cache_dir=${DATA_CACHE_DIR} \
  backbone="dit" \
  model=small \
  model.length=1024 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  callbacks.checkpoint_every_n_steps.save_last=False \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=512 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  training.sampling_eps_training=${SAMPLING_EPS_TRAINING} \
  training.guidance=null \
  eval.generate_samples=False \
  trainer.num_nodes=${NUM_NODES} \
  trainer.devices=${NPROC} \
  wandb.name=${RUN_NAME} \
  hydra.run.dir=${RUN_DIR}
