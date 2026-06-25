### LLaDA + ProSeCo SFT Eval

This directory contains scripts to reproduce the evaluation for our LLaDA + ProSeCo SFT
experiments.

The two parameters for corrector sampling are:
```python
apply_corrector_every_n_steps = ...
max_corrector_steps_per_loop = ...
```

[`eval_llada.py`](./eval_llada.py) is the evaluation harness. It uses
[nemo_skills](https://github.com/NVIDIA-NeMo/Skills) for dataset loading, prompt
formatting, and benchmark evaluation, and the corrector-based diffusion sampler
(`generate.py`) for generation, reporting NFE and throughput statistics.

It requires the `proseco` env from [`create_env.sh`](../create_env.sh). The script
installs the pinned `nemo_skills` code without its full remote-launcher dependency
stack because the upstream `litellm`/`nemo_run` requirements currently disagree on
`httpx`; the local dependencies used by this harness (`math-verify`, `evalplus`,
prompt/data utilities, etc.) are installed explicitly.

Generation is sharded across all `accelerate` processes and gathered on rank 0 for
evaluation. Run these from the repo root. Launch
it via [`eval_llada.sh`](./eval_llada.sh), or directly:
```bash
accelerate launch llada/eval_llada.py \
    --benchmark human-eval \
    --model_path kuleshov-group/proseco-llada-sft \
    --tokenizer_path GSAI-ML/LLaDA-8B-Instruct \
    --gen_length 1024 --block_length 32 --steps 1024 \
    --apply_corrector_every_n_steps 2 --max_corrector_steps_per_loop 4 \
    --output_dir ./llada/outputs/humaneval \
    --prompt_config llada/prompt_configs/code.yaml
```
Use [`eval_wrapper.sh`](./eval_wrapper.sh) (also from the repo root)
to sweep over the corrector hyperparameters. `eval_llada.sh` automatically detects
`NUM_GPUS` from Slurm/CUDA environment variables; the wrapper exports it explicitly
so a one-GPU job does not accidentally request eight accelerate processes.

For a short smoke run on an interactive GPU, cap the dataset and redirect outputs
outside the repo:

```bash
NUM_GPUS=1 MAX_SAMPLES=1 LENGTH=16 BLOCK_LENGTH=16 STEPS=1 BASE_SAVE_DIR=/tmp/proseco_llada_smoke bash llada/eval_llada.sh
```
