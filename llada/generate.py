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


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=256,
    gen_length=256,
    block_length=32,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    max_corrector_steps_per_loop=4,
    apply_corrector_every_n_steps=2,
    early_eos_stopping=True,
    tokenizer=None,
    disable_pbar=True,
    save_intermediate_outputs=False,
):
    """Semi-autoregressive masked diffusion generation with corrector refinement.

    Generates tokens by iteratively unmasking positions in blocks. Within each
    block, a predictor pass proposes tokens for all masked positions, then a
    subset is committed based on confidence ranking (or a threshold). An optional
    corrector refines predictions via fixed-point iterations over all tokens
    generated so far.

    Args:
        model: Masked diffusion language model whose forward pass returns an
            object with a ``.logits`` attribute of shape ``(B, L, V)``.
        prompt: Input token ids, shape ``(B, prompt_len)``.
        steps: Total unmasking steps across all blocks.
        gen_length: Number of tokens to generate. Must be divisible by
            ``block_length``.
        block_length: Tokens per semi-autoregressive block.
        temperature: Gumbel-noise temperature for sampling (0 = greedy argmax).
        remasking: Remasking strategy: ``'low_confidence'`` or ``'random'``.
        mask_id: Token id used for ``[MASK]``.
        threshold: If set, use confidence-threshold unmasking (as in fast-DLLM)
            instead of the fixed-schedule top-k strategy.
        max_corrector_steps_per_loop: Maximum corrector (fixed-point) iterations
            per generation step. Set to 0 to disable the corrector.
        apply_corrector_every_n_steps: Run the corrector every N predictor steps.
        early_eos_stopping: If True, halt generation when the last position in a
            block is EOS for every batch element, and fill remaining masks with
            EOS.
        tokenizer: Required when ``early_eos_stopping`` is True.
        disable_pbar: If True, suppress the progress bar.
        save_intermediate_outputs: If True, record the sequence state after each
            step.

    Returns:
        Tuple of ``(x, metrics, intermediate_outputs)`` where

        - **x**: Token ids, shape ``(B, prompt_len + gen_length)``.
        - **metrics**: ``dict`` with keys ``'predictor_nfe'``,
          ``'corrector_nfe'``, ``'total_nfe'``.
        - **intermediate_outputs**: ``list[dict]`` (empty when
          ``save_intermediate_outputs`` is False).
    """
    prompt_len = prompt.shape[1]
    batch_size = prompt.shape[0]

    # Initialise sequence: [prompt tokens | MASK ... MASK]
    x = torch.full(
        (batch_size, prompt_len + gen_length), mask_id, dtype=torch.long,
    ).to(model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        )
    steps_per_block = steps // num_blocks

    predictor_nfe = 0
    corrector_nfe = 0
    total_nfe = 0
    intermediate_outputs = []

    block_pbar = tqdm(range(num_blocks), leave=False, disable=disable_pbar)
    for block_idx in block_pbar:
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)

        # The corrector refines all generated tokens (across past blocks),
        # not just the current block, so the active region starts at the
        # prompt boundary rather than at block_start.
        active_region_start = prompt_len

        step = 0
        applied_corrector = False

        while True:
            # --- Predictor step ---
            predictor_nfe += 1
            total_nfe += 1

            global_mask = (x == mask_id)
            logits = model(x).logits

            # Ignore mask positions beyond the current block boundary
            global_mask[:, block_end:] = False
            active_mask = global_mask[:, active_region_start:block_end]

            # --- Corrector steps (fixed-point iteration) ---
            if (step + 1) % apply_corrector_every_n_steps == 0:
                corrector_step = 0

                if max_corrector_steps_per_loop > 0:
                    corrector_x = x.clone()
                    # Fill all positions from active_region_start onward with
                    # greedy predictions so the corrector never sees mask tokens.
                    corrector_x[:, active_region_start:] = torch.argmax(
                        logits, dim=-1,
                    )[:, active_region_start:]
                    # Restore already-committed (non-masked) tokens
                    corrector_x[:, active_region_start:block_end][~active_mask] = (
                        x[:, active_region_start:block_end][~active_mask]
                    )
                    applied_corrector = True
                else:
                    corrector_x, corrector_logits = None, None

                while corrector_step < max_corrector_steps_per_loop:
                    corrector_step += 1
                    corrector_nfe += 1
                    total_nfe += 1
                    block_pbar.set_postfix(
                        predictor_nfe=predictor_nfe,
                        corrector_nfe=corrector_nfe,
                        total_nfe=total_nfe,
                    )
                    corrector_logits = model(corrector_x).logits[
                        :, active_region_start:block_end
                    ]
                    corrected_tokens = torch.argmax(corrector_logits, dim=-1)

                    if torch.allclose(
                        corrector_x[:, active_region_start:block_end],
                        corrected_tokens,
                    ):
                        break
                    corrector_x[:, active_region_start:block_end] = corrected_tokens

                # Merge corrector outputs back into the main sequence
                if max_corrector_steps_per_loop > 0:
                    logits[:, active_region_start:block_end] = corrector_logits
                    active_mask = global_mask[:, active_region_start:block_end]
                    x[:, active_region_start:block_end][~active_mask] = (
                        corrected_tokens[~active_mask]
                    )

            # --- Select which tokens to unmask this step ---
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                global_mask,
                x,
                num_transfer_tokens[:, step] if threshold is None else None,
                threshold,
            )
            x[transfer_index] = x0[transfer_index]
            step += 1

            block_pbar.set_postfix(
                predictor_nfe=predictor_nfe,
                corrector_nfe=corrector_nfe,
                total_nfe=total_nfe,
            )

            if save_intermediate_outputs:
                intermediate_outputs.append({
                    "step": step,
                    "applied_corrector": applied_corrector,
                    "output": x[:, prompt_len:block_end].detach().cpu(),
                })

            # All mask tokens in the current block have been revealed
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

        # Stop early when every batch element ends the block with EOS
        if (
            early_eos_stopping
            and tokenizer is not None
            and (x[:, block_end - 1] == tokenizer.eos_token_id).all()
        ):
            x[x == mask_id] = tokenizer.eos_token_id
            break

    return x, {
        "predictor_nfe": predictor_nfe,
        "corrector_nfe": corrector_nfe,
        "total_nfe": total_nfe,
    }, intermediate_outputs



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
