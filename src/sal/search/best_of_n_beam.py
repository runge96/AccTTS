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
import logging
import runpy
import time
from pathlib import Path

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams
from vllm.attention.ops import chunked_prefill_paged_decode as cpd
from vllm.inputs import TokensPrompt

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils import xformers_attn
from sal.utils.attn_lut import ENV_VAR as _ATTN_LUT_ENV_VAR
from sal.utils.attn_lut import AttnLUT, lut_path_for_mode
from sal.utils.score import aggregate_scores

logger = logging.getLogger(__name__)

# Loaded lazily on first call (config.model_path is needed to resolve the
# per-model default path). Cached by source path so we don't reparse on every
# batch. None if no LUT JSON is on disk; the search loop then leaves
# cpd.PROFILE_CHUNK_SIZE_PAGE alone (vanilla vLLM behavior).
_ATTN_LUT_CACHE: dict[str, AttnLUT] = {}
_ATTN_LUT_LOGGED: set[str] = set()


def _configure_attention(config: Config) -> None:
    """Switch vLLM's paged attention kernel based on config.chunk_size.

    "heuristic" -> patch vLLM's Triton backend to call our xformers-style
                   kernel (sal.utils.xformers_attn) where CHUNK_Size_Page
                   is tl.constexpr. Chunk size is computed per call from
                   get_split_k (continuous in Mk), so each unique value
                   triggers a Triton compile — reproducing xformers'
                   behavior, including its cold-start cost. The LUT is
                   unused.
    "dynamic"   -> standard vLLM kernel (plain-int CHUNK_Size_Page) driven
                   by PROFILE_CHUNK_SIZE_PAGE from the profile-tuned LUT.
    "none"      -> standard vLLM kernel, FA path always.
    """
    if getattr(config, "chunk_size", "none") == "heuristic":
        xformers_attn.enable()
    else:
        xformers_attn.disable()


def _get_attn_lut(config: Config) -> "AttnLUT | None":
    mode = getattr(config, "chunk_size", "none")
    # Only "dynamic" mode uses an actual LUT lookup at runtime. "heuristic"
    # is handled by sal.utils.xformers_attn (separate kernel); "none" uses FA.
    if mode != "dynamic":
        return None
    # SAL_ATTN_LUT_PATH wins over the per-mode default if set.
    import os
    env_path = os.environ.get(_ATTN_LUT_ENV_VAR)
    path = Path(env_path) if env_path else lut_path_for_mode(config.model_path, mode)
    if path is None or not path.exists():
        logger.warning(
            "best_of_n_beam: chunk_size=%s but LUT not found at %s; "
            "falling back to FA path.", mode, path,
        )
        return None
    key = str(path)
    cached = _ATTN_LUT_CACHE.get(key)
    if cached is None:
        cached = AttnLUT.load(path)
        _ATTN_LUT_CACHE[key] = cached
    if key not in _ATTN_LUT_LOGGED:
        logger.info(
            "best_of_n_beam: loaded attention LUT (%s) from %s", mode, path,
        )
        _ATTN_LUT_LOGGED.add(key)
    return cached


# Trace file produced by scripts/best_of_n_task_trace.py. Maps
# (problem_idx, beam_idx) -> generated token length so the search loop can
# reproduce per-beam generation lengths exactly across runs.
TASK_TRACE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "best_of_n_task_trace.py"
)


def _load_request_gen_tokens(path: Path) -> dict[tuple[int, int], int]:
    if not path.exists():
        return {}
    ns = runpy.run_path(str(path))
    return dict(ns.get("REQUEST_GEN_TOKENS", {}))


REQUEST_GEN_TOKENS = _load_request_gen_tokens(TASK_TRACE_PATH)


def trace_path_for_config(config: Config) -> Path:
    """Per-config trace location, matching scripts/best_of_n_task_trace.py and
    scripts/extract_smaller_beams.py: data/{model_path}/{dataset_short}/beams={n}.
    """
    output_dir = Path(config.output_dir or f"data/{config.model_path}")
    dataset_short_name = config.dataset_name.split("/")[-1]
    return output_dir / dataset_short_name / f"beams={config.n}"


def load_trace_for_config(config: Config) -> dict[tuple[int, int], int]:
    """Load the per-(model, dataset, n) REQUEST_GEN_TOKENS produced by
    best_of_n_task_trace.py or extract_smaller_beams.py. Returns an empty
    dict if the file is missing — caller can decide whether that's an error
    or just "use free generation".
    """
    return _load_request_gen_tokens(trace_path_for_config(config))


# Hardcoded fallbacks for quick local testing — uncomment one to override the
# trace loaded above.
# REQUEST_GEN_TOKENS = {
#     (0, 0): 573,
#     (0, 1): 536,
#     (0, 2): 600,
#     (0, 3): 4096,
#     (0, 4): 1197,
#     (0, 5): 712,
#     (0, 6): 736,
#     (0, 7): 2123,
#     (0, 8): 1040,
#     (0, 9): 4096,
#     (0, 10): 583,
#     (0, 11): 714,
#     (0, 12): 738,
#     (0, 13): 260,
#     (0, 14): 711,
#     (0, 15): 786,
# }

# problem_num_ctrl = 1
# beam_num_ctrl = 128
# gen_num_ctrl = 20
# REQUEST_GEN_TOKENS = {
#     (p, b): gen_num_ctrl
#     for p in range(problem_num_ctrl)
#     for b in range(beam_num_ctrl)
# }


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

    # Tokenize so we can optionally left-pad to a target prompt length.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    prompt_token_ids_list = [
        tokenizer.encode(c, add_special_tokens=False) for c in templated_convs
    ]
    if getattr(config, "padded_prompt_len", 0) and config.padded_prompt_len > 0:
        target = config.padded_prompt_len
        prompt_token_ids_list = [
            ([pad_token_id] * (target - len(ids)) + ids) if len(ids) < target else ids
            for ids in prompt_token_ids_list
        ]

    # Duplicate to config.n requests per problem (matches templated_convs above).
    prompt_token_ids_list = [
        ids for ids in prompt_token_ids_list for _ in range(config.n)
    ]

    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError("LLMEngine has unfinished requests.")

    request_meta = {}
    request_start_time = {}
    request_end_time = {}
    for idx, token_ids in enumerate(prompt_token_ids_list):
        problem_idx = indices[idx // config.n]
        beam_idx = idx % config.n
        request_sampling_params = copy.deepcopy(base_sampling_params)
        forced_tokens = REQUEST_GEN_TOKENS.get((problem_idx, beam_idx))
        if forced_tokens is not None:
            request_sampling_params.max_tokens = forced_tokens
            request_sampling_params.min_tokens = forced_tokens
        request_id = f"best-of-n-{idx}"
        engine.add_request(
            request_id,
            TokensPrompt(prompt_token_ids=token_ids),
            request_sampling_params,
        )
        request_meta[request_id] = (problem_idx, beam_idx)
        request_start_time[request_id] = time.perf_counter()

    # Attention-LUT lazy state. In steady state the per-step cost is two int
    # compares; the LUT lookup only fires when the live beam count drops or
    # max_kv_len crosses the next prompt_len axis value.
    _configure_attention(config)
    attn_lut = _get_attn_lut(config)
    lut_active = attn_lut is not None
    if lut_active:
        # Use the (already padded + duplicated) token-id list so the LUT sees
        # the actual prompt length the engine receives.
        max_prompt_len = max((len(p) for p in prompt_token_ids_list), default=0)
        pl_axis = attn_lut.prompt_len_axis
        live_n_beams = len(prompt_token_ids_list)
        lut_step_idx = 0
        last_n_for_csp = -1
        next_kv_threshold = 0  # forces first update on loop entry

    # # ---------------------- Profiler Starting ------------------------
    # torch.cuda.synchronize()
    # torch.cuda.cudart().cudaProfilerStart()
    # nvtx.range_push("E2E Timer")


    aggregated = {}
    while engine.has_unfinished_requests():
        if lut_active:
            current_kv = max_prompt_len + lut_step_idx
            if live_n_beams != last_n_for_csp or current_kv >= next_kv_threshold:
                cpd.PROFILE_CHUNK_SIZE_PAGE = attn_lut.pick_csp(
                    live_n_beams, current_kv
                )
                print(f"[LUT] step={lut_step_idx} n_beams={live_n_beams} kv~={current_kv} -> CSP={cpd.PROFILE_CHUNK_SIZE_PAGE}", flush=True)
                last_n_for_csp = live_n_beams
                nxt = int(np.searchsorted(pl_axis, current_kv, side="right"))
                next_kv_threshold = (
                    int(pl_axis[nxt]) if nxt < len(pl_axis) else 10**9
                )

        step_outputs = engine.step()

        if lut_active:
            lut_step_idx += 1

        for output in step_outputs:
            aggregated[output.request_id] = output
            if output.finished:
                request_end_time[output.request_id] = time.perf_counter()
                if lut_active:
                    live_n_beams -= 1

    if lut_active:
        cpd.PROFILE_CHUNK_SIZE_PAGE = None
                
    # # ---------------------- Profiler Ending ------------------------
    # nvtx.range_pop()
    # torch.cuda.synchronize()
    # torch.cuda.cudart().cudaProfilerStop()  

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

    if prm is not None:
        scores = prm.score(problems, completions)
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]
        preds = [
            completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)
        ]
    else:
        # PRM disabled — placeholder scores; pred = beam 0 (deterministic).
        scores = [[[0.0]] * config.n for _ in problems]
        preds = [completion[0] for completion in completions]

    examples["completions"] = completions
    examples["scores"] = scores
    examples["pred"] = preds
    examples["completion_tokens"] = completion_tokens
    examples["request_timings"] = request_timings

    return examples
