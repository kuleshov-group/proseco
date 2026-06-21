# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a LLaDA + ProSeCo masked-diffusion model on nemo_skills benchmarks.

This harness is an official implementation of ProSeCo that supports corrector-based
diffusion sampling on gsm8k / human-eval / mbpp / hendrycks_math
(and other nemo_skills benchmarks) with NFE and throughput tracking.

Launch it with ``accelerate launch`` (see ``eval_llada.sh``); data is sharded
across ranks and gathered on rank 0 for evaluation.

Example:
    accelerate launch llada/eval_llada.py \\
        --benchmark human-eval \\
        --prompt_config llada/prompt_configs/code.yaml \\
        --model_path kuleshov-group/proseco-llada-sft \\
        --tokenizer_path GSAI-ML/LLaDA-8B-Instruct \\
        --gen_length 1024 --block_length 32 --steps 1024 \\
        --apply_corrector_every_n_steps 2 --max_corrector_steps_per_loop 4 \\
        --output_dir ./llada/outputs/humaneval
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import accelerate
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from nemo_skills.dataset.prepare import prepare_datasets
from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.evaluator import evaluate, get_evaluator_class, supports_single_eval
from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.prompt.utils import get_prompt

from generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_generation_args(gen_args_str):
    """Extract default prompt_config and eval_type from a benchmark's GENERATION_ARGS.

    Dotted keys like ``eval_config.evalplus.dataset`` are expanded into nested
    dicts so that downstream consumers (OmegaConf, dataclass configs) receive
    properly structured config trees.
    """
    result = {}
    for token in gen_args_str.split():
        if token.startswith("++") and "=" in token:
            key, value = token[2:].split("=", 1)
            parts = key.split(".")
            d = result
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
    return result


def _load_benchmark_data(benchmark, split="test", data_dir=None):
    """Load a nemo_skills benchmark dataset, preparing it on the fly if needed.

    Returns (data_list, benchmark_defaults_dict, metrics_type_str).
    """
    dataset_module, _ = get_dataset_module(benchmark, data_dir=data_dir)

    pkg_data_dir = Path(dataset_module.__file__).parent
    data_file = pkg_data_dir / f"{split}.jsonl"

    if not data_file.exists():
        print(f"Dataset file {data_file} not found, preparing {benchmark}...")
        prepare_datasets([benchmark])

    if not data_file.exists():
        raise FileNotFoundError(f"Could not find or prepare dataset: {data_file}")

    data = []
    with open(data_file, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    # Shuffling here to make sure all ranks get the same data so they will finish approximately at the same time
    random.Random(42).shuffle(data)

    defaults = _parse_generation_args(getattr(dataset_module, "GENERATION_ARGS", ""))
    metrics_type = getattr(dataset_module, "METRICS_TYPE", None)

    return data, defaults, metrics_type


def _load_model(model_path, tokenizer_path, device, world_size, torch_dtype=torch.bfloat16):
    """Load a LLaDA-style HF model and tokenizer.

    The model is a ``trust_remote_code`` checkpoint loaded with
    ``AutoModel.from_pretrained``.  When running on more than one process, the
    weights are placed directly on the local device via ``device_map`` so each
    rank holds its own full copy of the model.
    """

    model_kwargs = {}
    if world_size > 1:
        model_kwargs["device_map"] = {"": device}

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        **model_kwargs,
    )
    model.eval()
    if world_size <= 1:
        model = model.to(device)

    tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _generate_batch(batch, prompt_builder, model, tokenizer, sampler_args, device, disable_pbar):
    """Format prompts, run the proseco diffusion sampler, and decode a batch.

    ``sampler_args`` are forwarded to ``generate`` (e.g. ``steps``,
    ``gen_length``, ``block_length``, ``max_corrector_steps_per_loop``,
    ``apply_corrector_every_n_steps``, ``threshold`` ...).  Prompts are
    left-padded so the generated region starts at the same offset for every
    sequence in the batch.
    """
    batched_input_ids = []
    max_len = 0

    for sample in batch:
        prompt_str = prompt_builder.fill(sample, format_as_string=True)
        input_ids = tokenizer(prompt_str)["input_ids"]
        batched_input_ids.append(input_ids)
        max_len = max(max_len, len(input_ids))

    padded = []
    for ids in batched_input_ids:
        pad_len = max_len - len(ids)
        padded_ids = torch.cat(
            [
                torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device),
                torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0),
            ],
            dim=1,
        )
        padded.append(padded_ids)
    input_tensor = torch.cat(padded, dim=0)

    generated, nfe, _ = generate(
        model,
        input_tensor,
        tokenizer=tokenizer,
        disable_pbar=disable_pbar,
        **sampler_args,
    )

    gen_ids = generated[:, input_tensor.shape[1]:]
    texts = [tokenizer.decode(gen_ids[j], skip_special_tokens=True) for j in range(len(batch))]
    n_tokens = [int((gen_ids[j] != tokenizer.eos_token_id).sum().item()) for j in range(len(batch))]

    return {"texts": texts, "nfe": nfe, "n_tokens": n_tokens}


def _evaluate_results(all_results, eval_type, eval_config, samples_file):
    """Run nemo_skills evaluation on all results and rewrite the merged file."""
    eval_cfg = {**eval_config, "input_file": samples_file}

    if supports_single_eval(eval_type, eval_cfg):
        evaluator = get_evaluator_class(eval_type, eval_cfg)

        async def _run():
            for result in all_results:
                updates = await evaluator.eval_single(result)
                result.update(updates)
        asyncio.run(_run())

        with open(samples_file, "wt", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
    else:
        evaluate(eval_type, OmegaConf.create(eval_cfg))


def _compute_and_print_metrics(benchmark, metrics_type, samples_file):
    """Compute metrics using nemo_skills."""
    compute = ComputeMetrics(benchmark=benchmark, metric_type=metrics_type)
    return compute.compute_metrics([samples_file])


def _print_final_summary(title, metrics, extra_sections=None):
    """Print a clear, formatted summary of evaluation results.

    Args:
        title: Header string displayed at the top of the summary.
        metrics: Nested dict of ``{subset: {eval_mode: {metric: value}}}``.
        extra_sections: Optional dict of ``{section_name: {key: value}}`` printed
            after the metrics.  Values may be scalars or nested dicts (whose
            entries are expanded with a ``key/subkey`` label).
    """
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS: {title}")
    print(sep)

    for subset, subset_metrics in metrics.items():
        if subset.startswith("_") and subset != "_all_":
            continue
        label = "All samples" if subset == "_all_" else subset
        print(f"\n  [{label}]")

        for eval_mode, eval_metrics in subset_metrics.items():
            print(f"    {eval_mode}:")
            numeric = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))}
            non_numeric = {k: v for k, v in eval_metrics.items() if not isinstance(v, (int, float))}
            if numeric:
                max_key_len = max(len(k) for k in numeric)
                for key, val in numeric.items():
                    formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                    print(f"      {key:<{max_key_len}}  {formatted}")
            for key, val in non_numeric.items():
                print(f"      {key}: {val}")

    if extra_sections:
        for section_name, entries in extra_sections.items():
            print(f"\n  [{section_name}]")
            flat_entries = []
            for key, val in entries.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        flat_entries.append((f"{key}/{sub_key}", sub_val))
                else:
                    flat_entries.append((key, val))
            if flat_entries:
                max_key_len = max(len(k) for k, _ in flat_entries)
                for key, val in flat_entries:
                    formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                    print(f"      {key:<{max_key_len}}  {formatted}")

    print(f"\n{sep}\n")


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "t", "yes", "y")


def _optional_float(value):
    if value is None:
        return None
    if str(value).lower() in ("none", "null", ""):
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a LLaDA + ProSeCo diffusion model on nemo_skills benchmarks.",
    )

    # dataset / evaluation
    parser.add_argument("--benchmark", default="human-eval", help="nemo_skills benchmark name (e.g. gsm8k, human-eval, mbpp).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_dir", default=None, help="Optional nemo_skills dataset directory.")
    parser.add_argument("--output_dir", default=None, help="Where to write samples.jsonl / metrics.json. Eval runs only when set.")
    parser.add_argument("--prompt_config", default=None, help="Override the benchmark's default nemo_skills prompt_config (name or yaml path).")
    parser.add_argument("--eval_type", default=None, help="Override the benchmark's default nemo_skills eval_type.")

    # model
    parser.add_argument("--model_path", required=True, help="HF model id / path (e.g. kuleshov-group/proseco-llada-sft).")
    parser.add_argument("--tokenizer_path", default=None, help="HF tokenizer id / path (defaults to --model_path).")

    # runtime
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nccl_timeout", type=int, default=30, help="NCCL timeout in minutes.")

    # sampler (proseco generate) hyperparameters
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--threshold", type=_optional_float, default=None, help="Confidence-threshold unmasking (fast-dLLM). None = fixed schedule.")
    parser.add_argument("--max_corrector_steps_per_loop", type=int, default=0, help="Max corrector (fixed-point) iterations per step. 0 disables the corrector.")
    parser.add_argument("--apply_corrector_every_n_steps", type=int, default=1)
    parser.add_argument("--early_eos_stopping", type=_str2bool, default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    state = accelerate.PartialState(timeout=timedelta(minutes=args.nccl_timeout))
    rank = state.process_index
    world_size = state.num_processes
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # ── dataset & defaults – prepare on rank 0, then let others load ─────
    if rank == 0:
        data, benchmark_defaults, metrics_type = _load_benchmark_data(
            args.benchmark, split=args.split, data_dir=args.data_dir,
        )
    if world_size > 1:
        state.wait_for_everyone()
    if rank != 0:
        data, benchmark_defaults, metrics_type = _load_benchmark_data(
            args.benchmark, split=args.split, data_dir=args.data_dir,
        )

    if args.prompt_config is not None:
        benchmark_defaults["prompt_config"] = args.prompt_config
    if args.eval_type is not None:
        benchmark_defaults["eval_type"] = args.eval_type
    prompt_config = benchmark_defaults["prompt_config"]
    eval_type = benchmark_defaults["eval_type"]
    eval_config = benchmark_defaults.get("eval_config", {})

    # ── model & prompt builder ───────────────────────────────────────────
    model, tokenizer = _load_model(args.model_path, args.tokenizer_path, device, world_size)
    prompt_builder = get_prompt(prompt_config=prompt_config, tokenizer=tokenizer)

    sampler_args = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=args.mask_id,
        threshold=args.threshold,
        max_corrector_steps_per_loop=args.max_corrector_steps_per_loop,
        apply_corrector_every_n_steps=args.apply_corrector_every_n_steps,
        early_eos_stopping=args.early_eos_stopping,
    )

    # ── shard data across ranks ──────────────────────────────────────────
    chunk_size = (len(data) + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, len(data))
    local_data = data[start_idx:end_idx]
    print(f"[rank {rank}] generating for {len(local_data)} samples ({start_idx}:{end_idx})")

    results = []

    # Synchronize all ranks and flush pending CUDA work so every rank starts
    # timing from the same wall-clock moment.
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    if world_size > 1:
        state.wait_for_everyone()
    start_time = time.time()

    for i in tqdm(range(0, len(local_data), args.batch_size), desc=f"Generating (rank {rank})", disable=(rank != 0)):
        batch = local_data[i : i + args.batch_size]

        gen_output = _generate_batch(
            batch, prompt_builder, model, tokenizer, sampler_args, device, disable_pbar=(rank != 0),
        )
        nfe = gen_output["nfe"]

        for j, text in enumerate(gen_output["texts"]):
            results.append({
                **batch[j],
                "generation": text,
                "nfe": nfe,
                "n_tokens": gen_output["n_tokens"][j],
                "_idx": start_idx + i + j,
            })

        if rank == 0:
            print(f"\n[{start_idx + i}] nfe={nfe}")
            print(f"{'=' * 60}")
            print(gen_output["texts"][0])
            print(f"{'=' * 60}\n")

    # Wall-clock end measured after the slowest rank finishes all GPU work.
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    if world_size > 1:
        state.wait_for_everyone()
    end_time = time.time()
    gen_time = end_time - start_time

    # ── gather results in memory via distributed collectives ─────────────
    if world_size > 1:
        gathered_results = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(results, gathered_results, dst=0)

        if rank == 0:
            all_results = [r for rank_results in gathered_results for r in rank_results]
        else:
            all_results = []
    else:
        all_results = results

    # ── rank 0: save, evaluate, metrics ──────────────────────────────────
    if rank == 0:
        all_results.sort(key=lambda x: x.get("_idx", 0))
        for r in all_results:
            r.pop("_idx", None)

        all_nfes = [r["nfe"] for r in all_results]
        if isinstance(all_nfes[0], dict):
            num_nfes = {k: sum(n[k] for n in all_nfes) for k in all_nfes[0]}
        else:
            num_nfes = int(sum(all_nfes))
        num_tokens = sum(r["n_tokens"] for r in all_results)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            samples_file = os.path.join(args.output_dir, "samples.jsonl")
            with open(samples_file, "wt", encoding="utf-8") as f:
                for r in all_results:
                    f.write(json.dumps(r) + "\n")

            print(f"Running {eval_type} evaluation on {len(all_results)} samples...")
            _evaluate_results(all_results, eval_type, eval_config, samples_file)

            metrics = _compute_and_print_metrics(args.benchmark, metrics_type, samples_file)
            stats_info = {"num_tokens": num_tokens, "num_nfes": num_nfes, "world_size": world_size}
            metrics["_stats"] = stats_info

            speed_info = {}
            if isinstance(all_nfes[0], dict):
                speed_info["avg_nfe"] = {k: float(np.mean([n[k] for n in all_nfes])) for k in all_nfes[0]}
                speed_info["tokens_per_step"] = {
                    k: num_tokens / num_nfes[k] if num_nfes[k] > 0 else 0
                    for k in num_nfes
                }
            else:
                speed_info["avg_nfe"] = float(np.mean(all_nfes))
                total_nfes = sum(all_nfes)
                speed_info["tokens_per_step"] = num_tokens / total_nfes if total_nfes > 0 else 0
            if gen_time > 0:
                speed_info["tokens_per_second"] = num_tokens / gen_time

            metrics["_speed"] = speed_info

            extra_sections = {"Stats": stats_info, "Speed": speed_info}
            _print_final_summary(args.benchmark, metrics, extra_sections)

            with open(os.path.join(args.output_dir, "metrics.json"), "wt", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)

            OmegaConf.save(OmegaConf.create(vars(args)), os.path.join(args.output_dir, "config.yaml"))
        else:
            print("No --output_dir provided; skipping evaluation and metric computation.")

    if world_size > 1:
        state.wait_for_everyone()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
