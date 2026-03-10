#!/bin/bash
set -euo pipefail

ENV_NAME="proseco"
PYTHON_VERSION="3.12.3"

echo "=== Creating conda environment: ${ENV_NAME} ==="
# Swapped cuda-nvcc for cuda-toolkit to get the full runtime libraries
#conda create -y -n "${ENV_NAME}" \
#    -c nvidia -c conda-forge -c defaults \
#    python="${PYTHON_VERSION}" \
#    cuda-toolkit=12.9 \
#    ipykernel=6.29.5 \
#    pip

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Configuring persistent environment variables ==="
# This ensures CUDA_HOME and LD_LIBRARY_PATH are set every time you activate this env
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export OLD_CUDA_HOME="${CUDA_HOME:-}"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$OLD_LD_LIBRARY_PATH"
EOF

cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
#!/bin/bash
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
export CUDA_HOME="$OLD_CUDA_HOME"
unset OLD_LD_LIBRARY_PATH
unset OLD_CUDA_HOME
EOF

# Source the newly created activation script for the current session
source "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"

echo "=== Installing pip packages ==="
pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchmetrics==1.8.1 \
    transformers==4.54.1 \
    tokenizers==0.21.4 \
    datasets==4.0.0 \
    huggingface-hub==0.34.4 \
    accelerate==1.10.1 \
    safetensors==0.5.3 \
    peft==0.17.1 \
    lightning==2.4.0 \
    lightning-utilities==0.14.3 \
    pytorch-lightning==2.6.1 \
    lm-eval==0.4.8 \
    wandb==0.21.1 \
    mauve-text \
    timm==1.0.16 \
    einops==0.8.1 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    rich==14.0.0 \
    numpy==1.26.4 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    pandas==2.2.3 \
    h5py==3.14.0 \
    rdkit \
    matplotlib==3.10.3 \
    seaborn==0.13.2 \
    tqdm==4.67.1 \
    regex==2024.11.6 \
    typing_extensions==4.15.0 \
    fsspec==2024.12.0 \
    ipdb \
    ipython==9.2.0 \
    jupyterlab==4.4.2 \
    notebook==7.4.2 \
    requests==2.32.5 \
    pyyaml==6.0.3 \
    filelock==3.25.0 \
    networkx==3.6.1 \
    sympy==1.14.0 \
    ninja==1.13.0 \
    protobuf==6.33.5 \
    pydantic==2.12.5 \
    jinja2==3.1.6 \
    pyparsing==3.3.2 \
    pytz==2026.1.post1

echo "=== Installing CUDA extension packages (require torch at build time) ==="

# Create a temporary directory inside your conda environment on the shared drive
mkdir -p "${CONDA_PREFIX}/tmp"
export TMPDIR="${CONDA_PREFIX}/tmp"

# Install FlashAttention
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

# Install compatible causal-conv1d (v1.4.0) and Mamba (v2.2.4) versions
pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" --no-build-isolation --no-cache-dir
pip install git+https://github.com/state-spaces/mamba.git@v2.2.4 --no-build-isolation --no-cache-dir

echo "=== Cleaning up temporary build files ==="
rm -rf "${CONDA_PREFIX}/tmp"
unset TMPDIR

echo "=== Environment '${ENV_NAME}' is ready ==="
echo "Activate with: conda activate ${ENV_NAME}"

