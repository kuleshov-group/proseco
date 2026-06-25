#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-proseco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8}"
RECREATE_ENV="${RECREATE_ENV:-0}"
REUSE_EXISTING_ENV="${REUSE_EXISTING_ENV:-1}"
MAX_JOBS="${MAX_JOBS:-4}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"

echo "=== Creating conda environment: ${ENV_NAME} ==="
eval "$(conda shell.bash hook)"

ENV_EXISTS=0
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    ENV_EXISTS=1
    if [[ "${RECREATE_ENV}" == "1" ]]; then
        echo "=== Removing existing environment: ${ENV_NAME} ==="
        conda env remove -y -n "${ENV_NAME}"
        ENV_EXISTS=0
    elif [[ "${REUSE_EXISTING_ENV}" == "1" ]]; then
        echo "=== Reusing existing environment: ${ENV_NAME} ==="
        echo "Set RECREATE_ENV=1 to remove and recreate it."
    else
        echo "Environment '${ENV_NAME}' already exists."
        echo "Set RECREATE_ENV=1 to remove/recreate it, or REUSE_EXISTING_ENV=1 to install into it."
        exit 1
    fi
fi

if [[ "${ENV_EXISTS}" == "0" ]]; then
    # Keep the conda CUDA toolkit aligned with the CUDA runtime used by the torch
    # 2.8 Linux wheels. FlashAttention, causal-conv1d, and mamba-ssm build native
    # extensions and need nvcc plus runtime libraries available at build and import
    # time.
    conda create -y -n "${ENV_NAME}" \
        -c nvidia -c conda-forge -c defaults \
        python="${PYTHON_VERSION}" \
        "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" \
        ipykernel=6.29.5 \
        pip
fi

echo "=== Activating environment ==="
# NVIDIA's cuda-nvcc activation hook references optional NVCC_* variables.
# Temporarily disable nounset so conda package hooks can run normally.
set +u
conda activate "${ENV_NAME}"
set -u

echo "=== Configuring persistent environment variables ==="
# This ensures CUDA_HOME, CUDACXX, and LD_LIBRARY_PATH are set every time this
# env is activated. They are needed when building and importing CUDA extensions.
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export OLD_CUDA_HOME="${CUDA_HOME:-}"
export OLD_CUDACXX="${CUDACXX:-}"
export OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-}"
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$OLD_LD_LIBRARY_PATH"
EOF

cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
#!/bin/bash
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
export CUDA_HOME="$OLD_CUDA_HOME"
export CUDACXX="$OLD_CUDACXX"
export PYTHONNOUSERSITE="$OLD_PYTHONNOUSERSITE"
unset OLD_LD_LIBRARY_PATH
unset OLD_CUDA_HOME
unset OLD_CUDACXX
unset OLD_PYTHONNOUSERSITE
EOF

# Source the newly created activation script for the current session.
source "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"

python - <<'PYSITE'
import site
import sys
print(f"python={sys.executable}")
print(f"user_site_enabled={site.ENABLE_USER_SITE}")
PYSITE

echo "=== Upgrading pip build tools ==="
python -m pip install --upgrade pip setuptools wheel
python -m pip install --ignore-installed --no-deps packaging==24.2
python - <<'PYPACKAGING'
from pathlib import Path
import shutil
import site

for site_dir in site.getsitepackages():
    for dist_info in Path(site_dir).glob('packaging-*.dist-info'):
        if not (dist_info / 'METADATA').exists():
            shutil.rmtree(dist_info)
PYPACKAGING

echo "=== Installing pip packages ==="
python -m pip install \
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
    requests==2.34.2 \
    chardet==5.2.0 \
    pyyaml==6.0.3 \
    filelock==3.25.0 \
    networkx==3.6.1 \
    sympy==1.14.0 \
    ninja==1.13.0 \
    protobuf==5.29.6 \
    pydantic==2.12.5 \
    jinja2==3.1.6 \
    pyparsing==3.3.2 \
    pytz==2026.1.post1 \
    packaging==24.2 \
    click==8.1.8

# nemo_skills is required for llada/eval_llada.py.  Its full dependency
# tree currently has an unsatisfiable httpx conflict between litellm and
# leptonai, so install the pinned code without deps and add the local eval
# dependencies imported by llada/eval_llada.py.
python -m pip install --no-deps "git+https://github.com/NVIDIA-NeMo/Skills.git@da85a881d972e6fec847b90cf553a0bf9bf10638"
python -m pip install \
    "latex2sympy2-extended==1.11.0" \
    "math-verify[antlr4_9_3]==0.9.0" \
    evalplus==0.3.1 \
    iso639-lang==2.6.3 \
    langdetect==1.0.9 \
    langcodes==3.5.1 \
    language-data==1.4.0

echo "=== Installing CUDA extension packages (require torch at build time) ==="

# Put extension build scratch under the conda env on the shared drive.
mkdir -p "${CONDA_PREFIX}/tmp"
export TMPDIR="${CONDA_PREFIX}/tmp"
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST

python - <<'PYINFO'
import torch
print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
PYINFO

# Build isolation must stay disabled so setup.py sees the already-installed
# torch package and its CUDA ABI.
python -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
python -m pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" --no-build-isolation --no-cache-dir

# mamba-ssm 2.2.4 hardcodes several legacy CUDA architectures in setup.py and
# ignores TORCH_CUDA_ARCH_LIST. Patch the local clone so cluster rebuilds do not
# spend most of their time compiling unused sm_5x/sm_6x/sm_7x kernels.
MAMBA_BUILD_DIR="${CONDA_PREFIX}/tmp/mamba-ssm"
rm -rf "${MAMBA_BUILD_DIR}"
git clone --depth 1 --branch v2.2.4 https://github.com/state-spaces/mamba.git "${MAMBA_BUILD_DIR}"
python - <<'PYMAMBA'
import os
from pathlib import Path

setup_py = Path(os.environ['MAMBA_BUILD_DIR']) / 'setup.py'
text = setup_py.read_text()
old = """        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_53,code=sm_53")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_62,code=sm_62")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_70,code=sm_70")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_72,code=sm_72")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")

        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
"""
new = """        arch_list = os.getenv("MAMBA_CUDA_ARCH_LIST", os.getenv("TORCH_CUDA_ARCH_LIST", "8.0;8.6;9.0"))
        for arch in re.split(r"[;, ]+", arch_list):
            arch = arch.strip()
            if not arch:
                continue
            arch = arch.replace("+PTX", "")
            arch_digits = arch.replace(".", "")
            cc_flag.append("-gencode")
            cc_flag.append(f"arch=compute_{arch_digits},code=sm_{arch_digits}")
"""
if old not in text:
    raise SystemExit('mamba setup.py architecture block not found')
setup_py.write_text(text.replace(old, new, 1))
PYMAMBA
MAMBA_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python -m pip install "${MAMBA_BUILD_DIR}" --no-build-isolation --no-cache-dir

echo "=== Running import sanity checks ==="
python - <<'PYCHECK'
import accelerate
import causal_conv1d
import flash_attn
import mamba_ssm
import nemo_skills
import torch

print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"flash_attn={flash_attn.__version__}")
print("causal_conv1d import ok")
print("mamba_ssm import ok")
print("nemo_skills import ok")
print(f"accelerate={accelerate.__version__}")
PYCHECK

echo "=== Cleaning up temporary build files ==="
rm -rf "${CONDA_PREFIX}/tmp"
unset TMPDIR

echo "=== Environment '${ENV_NAME}' is ready ==="
echo "Activate with: conda activate ${ENV_NAME}"
