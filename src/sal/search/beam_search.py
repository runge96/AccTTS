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
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.attention.ops.attn_timing import (get_and_reset_stats,
                                            is_enabled as attn_timing_enabled,
                                            reset_stats)

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last

logger = logging.getLogger()
from sal.utils.score import aggregate_scores


def _beam_search(
    batch_of_prompts, config: Config, llm: LLM, prm: PRM
) -> tuple[list[Beam], dict[str, list[list[int]]], dict[str, list[dict[str, object]]]]:
    # sampling_params = SamplingParams(
    #     temperature=config.temperature,
    #     max_tokens=config.max_tokens,
    #     top_p=config.top_p,
    #     stop=["\n\n"],
    #     include_stop_str_in_output=True,
    #     n=1,
    # )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=20,
        min_tokens=20,
        top_p=config.top_p,
        stop=[],
        include_stop_str_in_output=True,
        ignore_eos = True,
        n=1,
    )

    # Initialize beams - Creates n beams per prompt
    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    completed_beams: list[Beam] = []
    beam_step_token_lengths: dict[str, list[list[int]]] = defaultdict(list)
    attn_step_timings: dict[str, list[dict[str, object]]] = defaultdict(list)
    wall_step_timings: dict[str, list[dict[str, object]]] = defaultdict(list)
    wall_total_timings: dict[str, float] = defaultdict(float)

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            # sampling_params = SamplingParams(
            #     temperature=config.temperature,
            #     max_tokens=config.max_tokens,
            #     top_p=config.top_p,
            #     n=1,
            # )
            sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=20,
            min_tokens=20,
            top_p=config.top_p,
            stop=[],
            include_stop_str_in_output=True,
            ignore_eos = True,
            n=1,
        )
        
        convs = [ # Build conversations & generate for each beam
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        if attn_timing_enabled():
            reset_stats()
        wall_start = time.perf_counter()
        gen_results = generate_k_steps(  # Generate one step
            templated_convs, lookahead, llm, sampling_params, 1
        )
        wall_end = time.perf_counter()
        step_wall_time_ms = (wall_end - wall_start) * 1000.0
        step_attn_time_ms = 0.0
        step_attn_tokens = 0
        if attn_timing_enabled():
            step_attn_time_ms, step_attn_tokens = get_and_reset_stats()

        prompts, completions = [], []
        step_token_lengths: dict[str, list[int]] = defaultdict(list)
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            step_len = len(
                tokenizer.encode(gen_result.next_texts[0], add_special_tokens=False)
            )
            step_token_lengths[beam.prompt].append(step_len)
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])
        for prompt, lengths in step_token_lengths.items():
            beam_step_token_lengths[prompt].append(lengths)
        for prompt in step_token_lengths.keys():
            wall_step_timings[prompt].append({
                "step_idx": i,
                "wall_time_ms": step_wall_time_ms,
            })
            wall_total_timings[prompt] += step_wall_time_ms
        if attn_timing_enabled():
            total_step_tokens = sum(sum(lengths)
                                    for lengths in step_token_lengths.values())
            if total_step_tokens > 0:
                for prompt, lengths in step_token_lengths.items():
                    prompt_tokens = sum(lengths)
                    attn_step_timings[prompt].append({
                        "step_idx": i,
                        "attn_time_ms": step_attn_time_ms,
                        "attn_tokens": prompt_tokens,
                        "attn_ms_per_token":
                        (step_attn_time_ms / total_step_tokens),
                    })

        scores = prm.score(prompts, completions)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return (completed_beams, beam_step_token_lengths, attn_step_timings,
            wall_step_timings, wall_total_timings)


def beam_search(examples, indices=None, config: Config = None, llm: LLM = None,
                prm: PRM = None):
    problems = examples["problem"]
    (beam_results, beam_step_token_lengths, attn_step_timings,
     wall_step_timings, wall_total_timings) = _beam_search(
        problems, config, llm, prm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {
        "completions": [],
        "pred": [],
        "completion_tokens": [],
        "scores": [],
        "beam_step_token_lengths": [],
        "attn_step_timings": [],
        "wall_step_timings": [],
        "wall_total_timings": [],
    }

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["beam_step_token_lengths"].append(
            beam_step_token_lengths.get(p, [])
        )
        results["attn_step_timings"].append(attn_step_timings.get(p, []))
        results["wall_step_timings"].append(wall_step_timings.get(p, []))
        results["wall_total_timings"].append(wall_total_timings.get(p, 0.0))

    return results
