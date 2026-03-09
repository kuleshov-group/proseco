#!/bin/bash
set -euo pipefail

ENV_NAME="proseco"
PYTHON_VERSION="3.12.3"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -y -n "${ENV_NAME}" \
    -c pytorch -c nvidia -c conda-forge -c defaults \
    python="${PYTHON_VERSION}" \
    cuda-nvcc=12.9 \
    ipykernel=6.29.5 \
    pip

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

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
pip install flash-attn==2.7.3 --no-build-isolation
pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1" --no-build-isolation
pip install git+https://github.com/state-spaces/mamba.git@v2.2.4 --no-build-isolation

echo "=== Environment '${ENV_NAME}' is ready ==="
echo "Activate with: conda activate ${ENV_NAME}"
