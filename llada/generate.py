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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA
import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity
    score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode="floor")  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens


@ torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    remasking_only_masked=True,
    apply_corrector_every_n_steps=1,
    max_corrector_steps_per_loop=0,
    early_eos_stopping=False,
    tokenizer=None,
    disable_pbar=True,
    save_intermediate_outputs=False,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
          If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        threshold: (Optional) float value for confidence-based threshold used in
          fast-dllm decoding.
        tokenizer: Tokenizer (used for doing early stopping on EOS).
        remasking_only_masked: Whether to remasking only masked tokens.
            When using the corrector we could potentially remask already decoded tokens
            as well.
        apply_corrector_every_n_steps: Frequency for applying corrector steps.
        max_corrector_steps_per_loop: Maximum number of corrector steps per loop.
        early_eos_stopping: Whether to stop early on EOS tokens.
        disable_pbar: Disable progress bar.
        save_intermediate_outputs: Whether to save intermediate outputs after each step.
    """
    x = torch.full(
      (prompt.shape[0], prompt.shape[1] + gen_length),
      mask_id,
      dtype=torch.long
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        print(f"ERROR: steps {steps} is not divisible by num_blocks {num_blocks}")
        exit(0)

    steps_per_block = steps // num_blocks

    nfe = 0
    corrector_nfe = 0
    total_nfe = 0
    saved_intermediate_outputs = []
    block_pbar = tqdm(range(num_blocks), leave=False, disable=disable_pbar)
    for num_block in block_pbar:
        block_start_index = prompt.shape[1] + num_block * block_length
        block_end_index = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start_index : block_end_index] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        if not remasking_only_masked:
            block_index = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
            block_index[:, block_start_index : block_end_index] = True
            # This makes sure that num_transfer_tokens is correct for the next step
            # while allowing for the selection of non-masked tokens as well
            num_transfer_tokens = num_transfer_tokens.cumsum(dim=-1)
        else:
            block_index = None
        i = 0
        is_corrector = False
        while True:
            nfe += 1
            total_nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, block_end_index :] = 0
            block_mask_index = mask_index[:, prompt.shape[1]:block_end_index]
            # Correct tokens within a diffusion step
            if (i+1) % apply_corrector_every_n_steps == 0:
                ci_step = 0
                if max_corrector_steps_per_loop > 0:
                    corrector_x = x.clone()
                    corrector_x[:, prompt.shape[1]:] = torch.argmax(logits, dim=-1)[:, prompt.shape[1]:]
                    corrector_x[:, prompt.shape[1]:block_end_index][~block_mask_index] = x[:, prompt.shape[1]:block_end_index][~block_mask_index]
                    is_corrector = True
                else:
                    corrector_x, corrector_logits = None, None
                while ci_step < max_corrector_steps_per_loop:
                    ci_step += 1
                    corrector_nfe += 1
                    total_nfe += 1
                    block_pbar.set_postfix(
                        nfe=nfe,
                        corrector_nfe=corrector_nfe,
                        total=total_nfe,
                    )
                    corrector_logits = model(corrector_x).logits[:, prompt.shape[1]:block_end_index]
                    corrected_output = torch.argmax(corrector_logits, dim=-1)
                    if torch.allclose(corrector_x[:, prompt.shape[1]:block_end_index], corrected_output):
                        break
                    corrector_x[:, prompt.shape[1]:block_end_index] = corrected_output
                if max_corrector_steps_per_loop > 0:
                    logits[:, prompt.shape[1]:block_end_index] = corrector_logits
                    block_mask_index = mask_index[:, prompt.shape[1]:block_end_index]
                    x[:, prompt.shape[1]:block_end_index][~block_mask_index] = corrected_output[~block_mask_index]

            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                # `block_index` remasks all positions in the block, change to
                # `mask_index` for selecting to remask only masked positions and make
                # sure that `num_transfer_tokens` is not with `cumsum`
                mask_index if remasking_only_masked else block_index,
                x,
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
            )
            x[transfer_index] = x0[transfer_index]
            i += 1
            block_pbar.set_postfix(
                nfe=nfe,
                corrector_nfe=corrector_nfe,
                total=total_nfe,
            )
            if save_intermediate_outputs:
                saved_intermediate_outputs.append(
                    {
                        "step": i,
                        "is_corrector": is_corrector,
                        "output": x[:, prompt.shape[1]:block_end_index].detach().cpu(),
                    }
                )
            if (
                x[
                    :,
                    prompt.shape[1] + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break
        if early_eos_stopping and tokenizer is not None and all(x[:, block_end_index-1] == tokenizer.eos_token_id):
            x[x == mask_id] = tokenizer.eos_token_id
            break

    return x, {
        "nfe": nfe,
        "corrector_nfe": corrector_nfe,
        "total_nfe": total_nfe,
    }, saved_intermediate_outputs



def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(
        torch.finfo(x0_p.dtype).min,
        device=x0_p.device,
        dtype=x0_p.dtype
    )
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index


def main():
    device = "cuda"

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    prompt = ("Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. "
              "How many kilometers can she run in 8 hours?")

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate(
            model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking="low_confidence")

        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
