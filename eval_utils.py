import os

import torch
import transformers
from tqdm import tqdm

import diffusion


def compute_ppl(
    pretrained_model,
    val_ds,
    device='cuda',
):
  ppl_metrics = diffusion.Perplexity().to(device)
  pbar = tqdm(val_ds, desc='PPL')
  for batch in pbar:
    input_ids = batch['input_ids'].to(device)
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask'].to(device)
    else:
      attention_mask = None
    losses = pretrained_model._loss(input_ids, attention_mask)
    ppl_metrics.update(losses.nlls, losses.token_mask)
    pbar.set_postfix({'ppl': ppl_metrics.compute().item()})
  return ppl_metrics.compute().item()


def compute_generative_ppl(
    sentences,
    eval_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=128,
    device='cuda',
    return_metric=False,
  ):
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  gen_ppl_metric = diffusion.Perplexity().to(device)
  eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
    eval_model_name_or_path, trust_remote_code=True)
  if eval_model_tokenizer.pad_token is None:
    eval_model_tokenizer.pad_token = \
      eval_model_tokenizer.eos_token
    eval_model_tokenizer.pad_token_id = \
      eval_model_tokenizer.eos_token_id
  eval_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name_or_path, trust_remote_code=True).eval()
  if max_length is None:
    max_length = max_length
  eval_model = eval_model.to(device)
  # Re-tokenize using eval model's tokenizer
  tokenizer_kwargs = {
    'return_tensors': 'pt',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'truncation': True,
    'padding': True,
    'max_length': max_length,
  }
  eval_context_size = 1024
  samples = eval_model_tokenizer(
    sentences, **tokenizer_kwargs)
  attn_mask = samples['attention_mask']
  samples = samples['input_ids']
  gen_ppl_eval_batch_size = min(gen_ppl_eval_batch_size, samples.shape[0])
  attn_mask = attn_mask.to(device)
  samples = samples.to(device)
  num_batches = samples.shape[0] // gen_ppl_eval_batch_size
  for i in tqdm(range(num_batches),
                desc=f'Gen. PPL', leave=False, disable=(device != 'cuda:0')):
    _samples = torch.split(
      samples[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    _attn_mask = torch.split(
      attn_mask[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
      logits = eval_model(
        sample_chunk, attention_mask=attn_mask_chunk)[0]
      logits = logits.transpose(-1, -2)

      nlls = torch.nn.functional.cross_entropy(
        logits[..., :-1],
        sample_chunk[..., 1:],
        reduction='none')
      first_eos = (sample_chunk == eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
      token_mask = (sample_chunk != eval_model_tokenizer.eos_token_id)
      gen_ppl_metric.update(
        nlls, first_eos[..., 1:] + token_mask[..., 1:])
      # gen_ppl_metric.update(
      #   nlls, attn_mask_chunk[..., 1:])

  if return_metric:
    return gen_ppl_metric
  return gen_ppl_metric.compute().item()
