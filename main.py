import datetime
import json
import math
import os
import random

import fsspec
import hydra
import lightning as L
import mauve
import numpy as np
import omegaconf
import rich.syntax
import rich.tree
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_info
from tqdm import tqdm

import classifier
import dataloader
import diffusion
import eval_utils
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else',
  lambda condition, x, y: x if condition else y
)


# Helper function for parallel processing (must be at module level for pickling)
def _collect_and_decode_sample(args):
  """Collect input_ids from dataset and decode.

  Args:
    args: Tuple of (idx, valid_set, tokenizer)

  Returns:
    Decoded text string
  """
  idx, valid_set, tokenizer = args
  sample = valid_set[idx]
  input_ids = sample['input_ids']
  return tokenizer.decode(input_ids)


def _load_from_checkpoint(config, tokenizer, device='cuda'):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to(device)

  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False, map_location=device)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _train(config, logger, tokenizer,
           train_classifier=False):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  if not config.is_vision:
    _print_batch(train_ds, valid_ds, tokenizer)

  if train_classifier:
    # This param indicates classifier will be used for
    #   PPLM / NOS-style guidance
    #  (see: https://arxiv.org/abs/2305.20009).
    if getattr(config, 'is_pplm_classifier', False):
      pretrained_model = _load_from_checkpoint(
        config, tokenizer)
      if (getattr(config.classifier_model, 'use_encoder_ema', True)
          and pretrained_model.ema):
        pretrained_model.load_ema_params()
      pretrained_backbone = pretrained_model.backbone
      # Remove the last layer for the classifier
      if hasattr(pretrained_backbone, 'output_layer'):  #DiT
        delattr(pretrained_backbone, 'output_layer')
      if hasattr(pretrained_backbone, 'model.lm_head'):  #DiMamba
        delattr(pretrained_backbone, 'model.lm_head')
      if getattr(config.classifier_model, 'freeze_encoder', True):
        for param in pretrained_backbone.parameters():
          param.requires_grad = False
    else:
      pretrained_backbone = None

    model = classifier.Classifier(
      config,
      tokenizer=valid_ds.tokenizer,
      pretrained_backbone=pretrained_backbone)
  else:
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _gen_eval(config, tokenizer, rank=0, device='cuda'):
  world_size = dist.get_world_size()
  pretrained = _load_from_checkpoint(
    config=config, tokenizer=tokenizer, device='cpu'
  )
  pretrained = pretrained.to(device=f"cuda:{rank}")
  pretrained.eval()
  generated_samples_path_rank = os.path.join(
    config.eval.generated_samples_path, f"samples_rank{rank}.jsonl")
  entropies_path_rank = os.path.join(
    config.eval.generated_samples_path, f"entropies_rank{rank}.jsonl")
  nfes_path_rank = os.path.join(
    config.eval.generated_samples_path, f"nfes_rank{rank}.jsonl")
  rngs_path_rank = os.path.join(
    config.eval.generated_samples_path, f"rng_state_rank{rank}.ckpt"
  )
  agg_NFEs_dict = {}
  if utils.fsspec_exists(generated_samples_path_rank):
    print(f"Loading generated sequences from {generated_samples_path_rank}")
    with fsspec.open(generated_samples_path_rank, 'r', encoding='utf-8') as f:
      samples = [json.loads(line) for line in f]
      generated_samples_count = len(samples)
  else:
    samples= []
    generated_samples_count, generated_batches_count = 0, 0
  if utils.fsspec_exists(entropies_path_rank):
    print(f"Loading entropies from {entropies_path_rank}")
    with fsspec.open(entropies_path_rank, 'r', encoding='utf-8') as f:
      entropies = [float(json.loads(line)) for line in f]
  else:
    entropies= []
  NFEs_dicts = []
  if utils.fsspec_exists(nfes_path_rank):
    print(f"Loading nfes from {nfes_path_rank}")
    with fsspec.open(nfes_path_rank, 'r', encoding='utf-8') as f:
      for line in f:
        NFEs_dict = json.loads(line)
        NFEs_dicts.append(NFEs_dict)
        for k, v in NFEs_dict.items():
          if k in agg_NFEs_dict:
            agg_NFEs_dict[k].append(v)
          else:
            agg_NFEs_dict[k] = [v]
  if utils.fsspec_exists(rngs_path_rank):
    print(f"Loading rngs from {rngs_path_rank}")
    rng_state_dict = torch.load(rngs_path_rank, map_location='cpu', weights_only=False)
    torch.set_rng_state(rng_state_dict['cpu_rng_state'])
    torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
    np.random.set_state(rng_state_dict['numpy_rng_state'])
    random.setstate(rng_state_dict['py_rng_state'])

  generated_samples_count = torch.tensor(generated_samples_count, device=device)
  dist.all_reduce(generated_samples_count)
  generated_samples_count = generated_samples_count.item()
  remaining_samples_count = config.eval.max_samples - generated_samples_count
  total_batches = math.ceil(remaining_samples_count / config.sampling.batch_size)
  rank_num_sample_batches = total_batches // world_size
  rank_num_sample_batches += int(rank < total_batches % world_size)
  if rank == 0:
    print(f"Already generated {generated_samples_count} samples.")

  for _ in tqdm(range(rank_num_sample_batches),
                desc="Gen. batches", leave=False, disable=(rank != 0)):
    sample, NFEs_dict = pretrained.sample(disable_pbar=(rank != 0))
    with fsspec.open(generated_samples_path_rank, "a", encoding="utf-8") as f:
      for decoded_sample in pretrained.tokenizer.batch_decode(sample):
        f.write(json.dumps(decoded_sample, ensure_ascii=False) + "\n")
    for i in range(sample.shape[0]):
      row = sample[i]
      counts = torch.unique(row, return_counts=True, sorted=True)[1]
      row_entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
      entropies.append(row_entropy)
      with fsspec.open(entropies_path_rank, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_entropy, ensure_ascii=False) + "\n")

    with fsspec.open(nfes_path_rank, "a", encoding="utf-8") as f:
      f.write(json.dumps(NFEs_dict, ensure_ascii=False) + "\n")
    samples.extend(pretrained.tokenizer.batch_decode(sample))
    NFEs_dicts.append(NFEs_dict)
    if rank == 0:
      print("\nNFEs:", {k: v for k, v in NFEs_dict.items()})
    for k, v in NFEs_dict.items():
      if k in agg_NFEs_dict:
        agg_NFEs_dict[k].append(v)
      else:
        agg_NFEs_dict[k] = [v]
    if rank == 0:
      print("Running avg NFEs:", {k: np.mean(v) for k, v in agg_NFEs_dict.items()})

    rng_state_dict = {
      "cpu_rng_state": torch.get_rng_state(),
      "gpu_rng_state": torch.cuda.get_rng_state(),
      "numpy_rng_state": np.random.get_state(),
      "py_rng_state": random.getstate(),
    }
    torch.save(rng_state_dict, rngs_path_rank)

  print(f"RANK{rank}: Generated {len(samples)} samples.")
  dist.barrier()
  del pretrained  # free up space for eval

  samples = _gather_results(samples, world_size)
  entropies = _gather_results(entropies, world_size)
  NFEs_dicts = _gather_results(NFEs_dicts, world_size)
  dist.barrier()

  # Check if metrics have already been computed (rank 0 checks, then broadcasts)
  if utils.fsspec_exists(os.path.join(config.eval.generated_samples_path, 'results.json')):
    if rank == 0:
      with fsspec.open(os.path.join(config.eval.generated_samples_path, 'results.json'), 'r') as f:
        print(json.load(f))
    dist.barrier()
    dist.destroy_process_group()
    return

  # Prepare data on rank 0
  if rank == 0:
    agg_NFEs_dict = {k: [] for k in agg_NFEs_dict.keys()}
    for NFEs_dict in NFEs_dicts:
      for k, v in NFEs_dict.items():
        agg_NFEs_dict[k].append(v)
    samples = samples[:config.eval.max_samples]
  else:
    samples = None

  # Broadcast samples to all ranks for distributed computation
  samples_list = [samples]
  dist.broadcast_object_list(samples_list, src=0)
  samples = samples_list[0]

  # ========== Distributed Generative PPL Computation ==========
  if rank == 0:
    print(f"Computing generative PPL across {world_size} GPUs for {len(samples)} samples.")

  # Split samples across GPUs for parallel PPL computation
  num_samples = len(samples)
  samples_per_rank = num_samples // world_size
  start_idx = rank * samples_per_rank
  end_idx = start_idx + samples_per_rank if rank < world_size - 1 else num_samples

  local_samples = samples[start_idx:end_idx]

  if rank == 0:
    print(f"Each rank processing ~{samples_per_rank} samples for PPL")

  # Compute PPL locally on each GPU - returns Perplexity metric object
  local_ppl_metric = eval_utils.compute_generative_ppl(
    local_samples,
    eval_model_name_or_path=config.eval.generative_ppl_model_name_or_path,
    gen_ppl_eval_batch_size=1,
    max_length=config.model.length,
    device=f"cuda:{rank}",
    return_metric=True)  # Modified to return metric object

  # Gather mean_value and weight from all ranks (attributes of MeanMetric)
  local_mean_value = local_ppl_metric.mean_value.cpu()
  local_weight = local_ppl_metric.weight.cpu()

  gathered_mean_values = [None for _ in range(world_size)]
  gathered_weights = [None for _ in range(world_size)]
  dist.all_gather_object(gathered_mean_values, local_mean_value)
  dist.all_gather_object(gathered_weights, local_weight)

  # ========== Distributed Entropy Computation ==========
  if rank == 0:
      # Filter out any None or zero-weight entries
    valid_pairs = [
      (mv, w)
      for mv, w in zip(gathered_mean_values, gathered_weights)
      if w is not None and w > 0
    ]

    if not valid_pairs:
      print("ERROR: No valid metric data from any GPU!")
      generative_ppl = float('nan')
    else:
      # Combine metrics from all GPUs
      # Note: mean_value in MeanMetric stores the sum (numerator), not the mean
      # So we sum all mean_values and divide by sum of weights
      # PPL = exp(sum(mean_values) / sum(weights))
      total_nll_sum = sum(mv for mv, _ in valid_pairs)
      total_weight = sum(w for _, w in valid_pairs)

      if total_weight == 0:
        print("ERROR: Total weight is zero!")
        generative_ppl = float('nan')
      else:
        avg_nll = total_nll_sum / total_weight
        generative_ppl = torch.exp(avg_nll).item()

      print(f"Generative PPL: {generative_ppl}")

    print(f"Computing entropy.")
    if len(entropies) != len(samples):
      entropies = []
      for sample in tqdm(samples, desc="(Re)Computing entropy."):
        counts = torch.unique(
        tokenizer(
          sample, return_tensors='pt')['input_ids'], return_counts=True, sorted=True)[1]
        entropies.append(torch.special.entr(counts.float() / counts.sum()).sum().item())
    entropy = sum(entropies) / len(entropies)
    print(f"Entropy: {entropy}")

    if not config.eval.skip_mauve:
      # ========== MAUVE Computation (rank 0 only) ==========
      print(f"Computing MAUVE score on device 0 for {len(samples)} samples...")
      p_features, p_text = None, None
      if utils.fsspec_exists(config.eval.mauve_p_features_path):
        p_features = np.load(config.eval.mauve_p_features_path)
        p_features = p_features[:min(len(samples), p_features.shape[0])]
        assert len(samples) == p_features.shape[0]
      else:
        assert config.loader.eval_batch_size == 1
        human_references = []
        _, valid_loader = dataloader.get_dataloaders(
          config, tokenizer, valid_seed=config.seed, skip_train=True)
        for _ in range(config.sampling.num_sample_batches):
          batch = next(iter(valid_loader))
          input_ids = batch['input_ids']
          human_references.extend(tokenizer.batch_decode(input_ids))
        assert len(samples) == len(human_references)
        p_text = human_references
      mauve_results = mauve.compute_mauve(
        p_text=p_text,
        p_features=p_features,
        q_text=samples,
        device_id=0,
        max_text_length=1024,
        verbose=True,
      )
      mauve_score = mauve_results.mauve
      print(f"MAUVE score: {mauve_score}")
    else:
      print("MAUVE score skipped. Setting to placeholder value of -1.0")
      mauve_score = -1.0

    with fsspec.open(os.path.join(config.eval.generated_samples_path, 'results.json'), 'w') as f:
      json.dump({
        'avg nfes': {k: np.mean(v) for k, v in agg_NFEs_dict.items()},
        'mauve': mauve_score,
        'generative_ppl': generative_ppl,
        'entropy': entropy,
        'generated_seqs': samples,
        'nfes': agg_NFEs_dict,
      },
        f, indent=4) # type: ignore
    print('Avg NFEs:', {k: f"{np.mean(v):0.3f}" for k, v in agg_NFEs_dict.items()})
    print(f"MAUVE score: {mauve_score:0.3f}")
    print(f"Entropy: {entropy:0.3f}")
    print(f"Gen. PPL: {generative_ppl:0.3f}")

    if config.get('wandb', None) is not None:
      wandb_logger = L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config),
        ** config.wandb)

      wandb_logger.log_metrics({
        "eval/mauve": float(mauve_score),
        "eval/generative_ppl": float(generative_ppl),
        "eval/entropy": float(entropy),
        "eval/num_samples": int(len(samples)),
      } | {f"eval/avg_{k}": np.mean(v) for k, v in agg_NFEs_dict.items()})
      wandb_logger.experiment.finish()

  # Synchronize all ranks and destroy process group after evaluation is complete
  dist.barrier()
  dist.destroy_process_group()


def _ppl_eval(config, tokenizer):
  print(f"Evaluating perplexity on {config.data.valid}.")
  pretrained = _load_from_checkpoint(
    config=config, tokenizer=tokenizer)
  pretrained.eval()
  if not config.eval.disable_ema:
    pretrained.load_ema_params()

  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  ppl = eval_utils.compute_ppl(pretrained, valid_ds)
  print(f"PPL: {ppl:0.3f}")


def _gather_results(results, world_size):
  # Each GPU has local 'results' (any pickle-able object)
  gathered_results = [None for _ in range(world_size)]
  dist.all_gather_object(gathered_results, results)

  # gathered_results is now a list of lists (one per rank)
  all_results = []
  for partial in gathered_results:
    all_results.extend(partial)  # type: ignore

  return all_results


def _setup_ddp_and_return_local_rank() -> int:
  """Sets up torch.distributed and selects GPU.

  Returns:
      (int) local_rank
  """
  dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=500))
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  return local_rank


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'ppl_eval':
    _ppl_eval(config, tokenizer)
  elif config.mode == 'sample_eval':
    local_rank = _setup_ddp_and_return_local_rank()
    # Reseed by rank:
    L.seed_everything(config.seed + local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)
    _gen_eval(config, tokenizer, rank=local_rank, device=device)
  elif 'train' in config.mode:
    _train(config, logger, tokenizer,
           train_classifier='classifier' in config.mode)
  else:
    raise NotImplementedError(f"Mode {config.mode} not implemented.")


if __name__ == '__main__':
  main()
