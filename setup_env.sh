#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

if ! command -v conda >/dev/null 2>&1; then
  if [[ -n "${CONDA_SHELL:-}" && -f "${CONDA_SHELL}" ]]; then
    # shellcheck source=/dev/null disable=SC1091
    source "${CONDA_SHELL}"
  else
    echo "conda command not found. Load conda first, then source setup_env.sh."
    return 1 2>/dev/null || exit 1
  fi
fi

eval "$(conda shell.bash hook)"

if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX}")" != "proseco" ]]; then
  conda activate proseco
fi

# Keep this env isolated from ~/.local Python packages on the cluster.
export PYTHONNOUSERSITE=1

# Setup HF cache.
export HF_HOME="${HF_HOME:-${PWD}/.hf_cache}"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules:${PYTHONPATH}"
else
  export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"
fi
