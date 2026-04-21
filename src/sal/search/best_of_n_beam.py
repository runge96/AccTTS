#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import copy
import time
import torch
import torch.cuda.nvtx as nvtx

import numpy as np
from vllm import LLM, SamplingParams
from vllm.attention.ops.attn_timing import (get_and_reset_stats,
                                            is_enabled as attn_timing_enabled,
                                            reset_stats)

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores


# REQUEST_GEN_TOKENS = {
#     (0, 0): 592,
#     (0, 1): 353,
#     (0, 2): 375,
#     (0, 3): 354,
#     (1, 0): 357,
#     (1, 1): 509,
#     (1, 2): 725,
#     (1, 3): 443,
# }
REQUEST_GEN_TOKENS = {
    (0, 0): 20,
    (0, 1): 20,
    (0, 2): 20,
    (0, 3): 20,
    (1, 0): 20,
    (1, 1): 20,
    (1, 2): 20,
    (1, 3): 20,
}


def _request_time_ms(start_time: float | None,
                     end_time: float | None,
                     metrics) -> float:
    if end_time is not None and start_time is not None:
        return max(0.0, (end_time - start_time) * 1000.0)
    if metrics is None:
        return 0.0
    start = (metrics.first_scheduled_time or metrics.first_token_time
             or metrics.arrival_time)
    end = metrics.finished_time or metrics.last_token_time
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start) * 1000.0)

def best_of_n_beam(examples, indices, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(indices)}
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    base_sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        n=1,
        max_tokens=config.max_tokens,
        min_tokens=1,
    )

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in problems
    ]
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError("LLMEngine has unfinished requests.")

    request_meta = {}
    request_start_time = {}
    request_end_time = {}
    for idx, prompt in enumerate(templated_convs):
        problem_idx = indices[idx // config.n]
        beam_idx = idx % config.n
        req_gen_tokens = REQUEST_GEN_TOKENS.get(
            (problem_idx, beam_idx), config.max_tokens)
        request_sampling_params = copy.deepcopy(base_sampling_params)
        request_sampling_params.max_tokens = req_gen_tokens
        request_sampling_params.min_tokens = req_gen_tokens
        request_id = f"best-of-n-{idx}"
        engine.add_request(request_id, prompt, request_sampling_params)
        request_meta[request_id] = (problem_idx, beam_idx)
        request_start_time[request_id] = time.perf_counter()
    
    # ---------------------- Profiler Starting ------------------------
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("E2E Timer")
    
    if attn_timing_enabled():
        reset_stats()

    aggregated = {}
    total_attn_time_ms = 0.0
    total_attn_tokens = 0
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        if attn_timing_enabled():
            step_attn_time_ms, step_attn_tokens = get_and_reset_stats()
            total_attn_time_ms += step_attn_time_ms
            total_attn_tokens += step_attn_tokens
        for output in step_outputs:
            aggregated[output.request_id] = output
            if output.finished:
                request_end_time[output.request_id] = time.perf_counter()
    
    # ---------------------- Profiler Ending ------------------------
    nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    completions = [[""] * config.n for _ in problems]
    completion_tokens = [[0] * config.n for _ in problems]
    request_timings = [[0.0] * config.n for _ in problems]
    for request_id, output in aggregated.items():
        problem_idx, beam_idx = request_meta[request_id]
        local_idx = index_map[problem_idx]
        completion = output.outputs[0]
        completions[local_idx][beam_idx] = completion.text
        completion_tokens[local_idx][beam_idx] = len(completion.token_ids)
        request_timings[local_idx][beam_idx] = _request_time_ms(
            request_start_time.get(request_id),
            request_end_time.get(request_id),
            output.metrics,
        )

    scores = prm.score(problems, completions)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score]
        for score in scores
    ]
    preds = [
        completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)
    ]

    examples["completions"] = completions
    examples["scores"] = scores
    examples["pred"] = preds
    examples["completion_tokens"] = completion_tokens
    examples["request_timings"] = request_timings
    examples["overall_attn_time_ms"] = [total_attn_time_ms] * len(problems)
    examples["overall_attn_tokens"] = [total_attn_tokens] * len(problems)

    return examples
