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

It requires `nemo_skills` (see the block in [`create_env.sh`](../create_env.sh)):
```bash
pip install "git+https://github.com/NVIDIA-NeMo/Skills.git@da85a881d972e6fec847b90cf553a0bf9bf10638"
```

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
to sweep over the corrector hyperparameters.
