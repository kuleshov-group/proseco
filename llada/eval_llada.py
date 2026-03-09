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

"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""
import json
import os
import random
import time
from datetime import timedelta
from typing import List, Tuple

import accelerate
import numpy as np
import torch
from accelerate.utils import InitProcessGroupKwargs
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        tokenizer_path=None,
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        max_corrector_steps_per_loop=0,
        apply_corrector_every_n_steps=1,
        early_eos_stopping=True,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        **kwargs,
    ):
        """
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires
                the model to verify whether the answer is generated through greedy
                sampling conditioned on the prompt (note that this differs from
                conditional generation).
                We implement this verification through the suffix_greedy_prediction()
                function, which returns a True/False judgment used for accuracy
                calculation.
                When is_check_greedy is set to True, the lm-evaluation-harness library
                automatically invokes this function.
                However, since none of the metrics in the LLaDA paper
                (https://arxiv.org/abs/2502.09992) require this functionality, we
                recommend setting is_check_greedy to False. This configuration causes
                suffix_greedy_prediction() to return False by default, significantly
                accelerating the evaluation process.
        """
        super().__init__()

        timeout_duration = timedelta(seconds=7200)
        kwargs_handler = [InitProcessGroupKwargs(timeout=timeout_duration)]
        accelerator = accelerate.Accelerator(kwargs_handler)  # type: ignore
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
        self.model = AutoModel.from_pretrained(
          model_path,
          trust_remote_code=True,
          torch_dtype=torch.bfloat16,
          **model_kwargs
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
          tokenizer_path,
          trust_remote_code=True
        )

        self.batch_size = int(batch_size)
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.max_corrector_steps_per_loop = max_corrector_steps_per_loop
        self.apply_corrector_every_n_steps = apply_corrector_every_n_steps
        self.is_instruct = True
        self.early_eos_stopping = early_eos_stopping
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood_rolling(self, requests) -> List[float]:
        pass

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pass

    def generate_until(self, requests):
        output = []
        nfes = []
        num_tokens = 0
        num_nfe = {
            "predictor_nfe": 0,
            "corrector_nfe": 0,
            "total_nfe": 0,
        }
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_output_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_output_path: {save_output_path}")
            if os.path.exists(save_output_path):
                print(f"load from {save_output_path}")
                with open(save_output_path, "r", encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        else:
          save_output_path = None

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])

        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating...", disable=(self.rank != 0)):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(
                      m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)["input_ids"]
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)["input_ids"]
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))

            # pad batched_input_ids to the same length
            batched_input_ids = [
                torch.cat([
                    torch.full(
                      (1, max_len - len(input_ids)),
                      self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
                    ),
                    torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1)
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros(
                    (
                        batched_input_ids.shape[0],
                        1,
                        max_len + self.gen_length,
                        max_len + self.gen_length,
                    ),
                    device=self.device,
                    dtype=torch.bool,
                )
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = req.args[1]["until"]
            input_ids = batched_input_ids
            generated_answer, nfe, intermediate_outputs = generate(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                remasking=self.remasking,
                mask_id=self.mask_id,
                threshold=self.threshold,
                max_corrector_steps_per_loop=self.max_corrector_steps_per_loop,
                apply_corrector_every_n_steps=self.apply_corrector_every_n_steps,
                early_eos_stopping=self.early_eos_stopping,
                tokenizer=self.tokenizer,
                disable_pbar=(self.rank != 0),
                save_intermediate_outputs=False
            )

            if self.is_instruct and "task_id" in req.doc and str(req.doc["task_id"]).lower().startswith("humaneval"):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    # num_nfe += nfe
                    num_nfe = {k: v + nfe[k] for k, v in num_nfe.items()}
                batched_generated_answer = [
                    self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True)
                    for i in range(len(generated_answer_ids))
                ]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(
                        generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False
                    )
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        # num_nfe += nfe
                        num_nfe = {k: v + nfe[k] for k, v in num_nfe.items()}
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)
            nfes.append(nfe)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_output_path, "a", encoding="utf-8") as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")

            if self.rank == 0:
              for i in range(len(batched_generated_answer)):
                  print("=" * 20)
                  print("question:\n", question)
                  print("answer:\n", batched_generated_answer[i])
                  print("nfe: ", nfe)
                  print({f"avg {k}: {v / len(output)}" for k, v in num_nfe.items()})
                  print("=" * 20, end="\n\n")
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        if self.show_speed and self.rank == 0:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_nfe_path = os.path.join(self.save_dir, f"rank_{rank}_nfe.json")
            print(f"save_nfe_path: {save_nfe_path}")
            with open(save_nfe_path, "w") as f:
                json.dump(
                    {
                        "cumulative_nfes": num_nfe,
                        "average_nfes": {
                            k: v / len(output) for k, v in num_nfe.items()
                        },
                        "nfes_per_sequence": nfes,
                    },
                    f,
                    indent=4,
                )

        return output


if __name__ == "__main__":
    set_seed(42)
    cli_evaluate()
