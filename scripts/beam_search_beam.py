#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Beam search experiment driver — online vLLM engine + per-beam timer.

Why this script exists:
  * `sal.search.beam_search` calls `llm.generate(...)` inside `generate_k_steps`.
    That's the offline path; it ignores the online engine that
    `best_of_n_beam` uses for continuous-batched serving.
  * `sal.search.beam_search` only records per-step wall times. Beam search
    interleaves PRM scoring with generation, so a per-beam end-to-end timer
    has to *accumulate* the gen-only chunks across iterations, skipping the
    PRM gaps. That accumulator did not exist.

What this script does:
  * Drives `engine.add_request` / `engine.step()` directly, one beam-search
    iteration at a time. Each active beam slot gets its own perf_counter
    timing for that iteration's 20-token chunk; the result is added to a
    `gen_time_ms` field on the beam.
  * On `copy.deepcopy(active * repeats)[:n]` (the "fill back to n by
    duplicating survivors" step that beam_search.py also does), the copy
    inherits the parent's accumulated `gen_time_ms` — so a final completed
    beam carries the cost of every iteration its lineage spent generating.
  * Emits CSVs in the same schema as `best_of_n_request_timings_*.csv`, so
    `scripts/e2e_results_analyze.ipynb` reads beam_search results with no
    notebook changes.

Usage:
    python scripts/beam_search_beam.py recipes/Qwen2.5-1.5B-Instruct/beam_search.yaml \\
        n=4 chunk_size=heuristic gemm_opt=false
"""

import copy
import csv
import logging
import os
import runpy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.attention.ops import chunked_prefill_paged_decode as cpd
from vllm.config import CompilationConfig, CompilationLevel

from sal.config import Config
from sal.search.utils import build_conv
from sal.utils import xformers_attn
from sal.utils.attn_lut import (
    ENV_VAR as _ATTN_LUT_ENV_VAR,
)
from sal.utils.attn_lut import (
    AttnLUT,
    lut_path_for_mode,
)
from sal.utils.data import get_dataset
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class BeamState:
    """One beam slot. `gen_time_ms` accumulates across iterations and
    survives `copy.deepcopy` when survivors are duplicated to refill n.
    """
    prompt: str
    current_text: str = ""
    completed: bool = False
    pruned: bool = False
    completion_tokens: int = 0
    gen_time_ms: float = 0.0
    stop_reason: object = None


# ---------------------------------------------------------------------------
# vLLM / kernel plumbing — same knobs as scripts/test_time_compute.py so the
# (gemm_opt, chunk_size) combinations name the same vllm/FD/ours methods.
# ---------------------------------------------------------------------------

def _configure_attention(config: Config) -> None:
    if getattr(config, "chunk_size", "none") == "heuristic":
        xformers_attn.enable()
    else:
        xformers_attn.disable()


def _maybe_load_attn_lut(config: Config) -> "AttnLUT | None":
    mode = getattr(config, "chunk_size", "none")
    if mode != "dynamic":
        return None
    env_path = os.environ.get(_ATTN_LUT_ENV_VAR)
    path = Path(env_path) if env_path else lut_path_for_mode(config.model_path, mode)
    if path is None or not path.exists():
        logger.warning("chunk_size=%s but LUT not found at %s; FA fallback.", mode, path)
        return None
    return AttnLUT.load(path)


def _warmup_attn_kernels(llm: LLM) -> None:
    sp = SamplingParams(temperature=0.0, max_tokens=20, min_tokens=20)
    prev = cpd.PROFILE_CHUNK_SIZE_PAGE
    try:
        cpd.PROFILE_CHUNK_SIZE_PAGE = None
        llm.generate(["warmup"], sp, use_tqdm=False)
        cpd.PROFILE_CHUNK_SIZE_PAGE = 4
        llm.generate(["warmup"], sp, use_tqdm=False)
    finally:
        cpd.PROFILE_CHUNK_SIZE_PAGE = prev


# ---------------------------------------------------------------------------
# Per-(problem, step, slot) forced-length trace for reproduction.
#
# Mirrors best_of_n_beam.REQUEST_GEN_TOKENS: scripts/beam_search_task_trace.py
# writes a Python literal mapping (problem_idx, iter_idx, slot_idx) -> gen_len;
# main() reassigns this dict from disk so subsequent FD/ours runs reproduce
# the exact per-step lengths of the vllm-baseline trace. Empty when no trace
# is loaded — the search loop then generates naturally with stop=["\n\n"].
# ---------------------------------------------------------------------------

REQUEST_STEP_GEN_TOKENS: dict[tuple[int, int, int], int] = {}
# Set of (problem_idx, iter_idx, slot_idx) marking which slots had their
# beam graduate to `completed` AT THAT iteration in the trace. Replay must
# reproduce these completions; otherwise was_forced=True keeps every beam
# alive past EOS and the lineage accumulates more tokens than the trace.
REQUEST_STEP_COMPLETED: set[tuple[int, int, int]] = set()


def trace_path_for_config(config: Config) -> Path:
    output_dir = Path(config.output_dir or f"data/{config.model_path}")
    dataset_short_name = config.dataset_name.split("/")[-1]
    return output_dir / dataset_short_name / f"beam_search_n{config.n}"


def _load_trace_dicts(path: Path
                      ) -> tuple[dict[tuple[int, int, int], int],
                                 set[tuple[int, int, int]]]:
    if not path.exists():
        return {}, set()
    ns = runpy.run_path(str(path))
    return (
        dict(ns.get("REQUEST_STEP_GEN_TOKENS", {})),
        set(ns.get("REQUEST_STEP_COMPLETED", ())),
    )


def load_trace_for_config(config: Config
                          ) -> tuple[dict[tuple[int, int, int], int],
                                     set[tuple[int, int, int]]]:
    return _load_trace_dicts(trace_path_for_config(config))


def _short_problem(problem_idx: int, problem: str, n_words: int = 6) -> str:
    """Compact identifier for the CSV `problem` column: `p{idx}: first words...`.

    The full problem text is irrelevant to the timing analysis and bloats
    the CSV; this keeps rows greppable while staying short.
    """
    words = problem.split()[:n_words]
    return f"p{problem_idx}: {' '.join(words)}"


def _method_suffix(config: Config) -> str:
    chunk_size = getattr(config, "chunk_size", "none")
    gemm_opt = bool(getattr(config, "gemm_opt", False))
    if not gemm_opt and chunk_size == "none":
        return "vllm"
    if not gemm_opt and chunk_size == "heuristic":
        return "FD"
    if gemm_opt and chunk_size == "dynamic":
        return "ours"
    if not gemm_opt and chunk_size == "dynamic":
        return "TurnOffGEMM"
    logger.warning("(gemm_opt=%s, chunk_size=%s) is not a named method.",
                   gemm_opt, chunk_size)
    return ""


# ---------------------------------------------------------------------------
# Beam search core — one problem at a time.
# ---------------------------------------------------------------------------

def _step_prompt(tokenizer, config: Config, problem: str, current_text: str,
                 is_first: bool) -> str:
    conv = build_conv(problem, current_text, config.system_prompt)
    return tokenizer.apply_chat_template(
        [conv],
        add_generation_prompt=is_first,
        continue_final_message=not is_first,
        tokenize=False,
    )[0]


def _run_one_problem(problem: str, problem_idx: int, config: Config, llm: LLM,
                     attn_lut
                     ) -> tuple[list[BeamState], list[tuple[int, int, int, float]]]:
    """Run beam search for one problem.

    Returns:
        completed_beams: final n beams (sorted/padded as in beam_search.py).
        step_trace:      list of (iter_idx, slot_idx, gen_len, gen_time_ms,
                         completed_this_step) covering every (iteration, beam
                         slot) that actually ran a step. A slot that was
                         short-circuited by the cumulative cap appears with
                         gen_len=0, gen_time_ms=0, completed=True.
    """
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    beams = [BeamState(prompt=problem) for _ in range(config.n)]
    completed_beams: list[BeamState] = []
    step_trace: list[tuple[int, int, int, float, bool]] = []
    trace_loaded = bool(REQUEST_STEP_GEN_TOKENS)

    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError("LLMEngine has unfinished requests at iteration start.")

    pl_axis = attn_lut.prompt_len_axis if attn_lut is not None else None

    for iter_idx in range(config.num_iterations):
        active = [b for b in beams if not b.pruned and not b.completed]
        if not active:
            break
        # Refill to n by duplicating survivors. deepcopy carries gen_time_ms
        # so the duplicate's lineage cost is preserved.
        if len(active) != config.n:
            repeats = (config.n // len(active)) + 1
            active = [copy.deepcopy(b) for b in (active * repeats)[: config.n]]

        # Submit each active beam with a per-beam max_tokens equal to the
        # remaining global budget. No min_tokens / ignore_eos forcing — beams
        # stop naturally on EOS or on the "\n\n" step delimiter, and we also
        # cap their cumulative completion_tokens at config.max_tokens.
        is_first = (iter_idx == 0)
        request_meta: dict[str, BeamState] = {}
        request_start: dict[str, float] = {}
        request_end: dict[str, float] = {}
        submitted: list[BeamState] = []
        submitted_slots: list[int] = []
        submitted_reqs: list[str] = []
        for slot_idx, beam in enumerate(active):
            forced = REQUEST_STEP_GEN_TOKENS.get(
                (problem_idx, iter_idx, slot_idx))
            if forced is not None:
                # Trace replay path. forced==0 means the trace short-circuited
                # this slot (cap hit) — replicate that. Otherwise, force
                # min==max==forced with stop=[] / ignore_eos=True so the
                # engine emits exactly `forced` tokens, no more, no less.
                # Clamp to the max-summed remaining budget so the cap
                # `beam.completion_tokens >= config.max_tokens` is respected
                # even when the trace was produced under the old per-lineage
                # rule that allowed iter_max sums beyond config.max_tokens.
                if forced == 0:
                    beam.completed = True
                    completed_beams.append(beam)
                    step_trace.append((iter_idx, slot_idx, 0, 0.0, True))
                    continue
                remaining = config.max_tokens - beam.completion_tokens
                if remaining <= 0:
                    beam.completed = True
                    completed_beams.append(beam)
                    step_trace.append((iter_idx, slot_idx, 0, 0.0, True))
                    continue
                forced_clamped = min(int(forced), remaining)
                sp = SamplingParams(
                    temperature=config.temperature, top_p=config.top_p,
                    max_tokens=forced_clamped, min_tokens=forced_clamped,
                    stop=[], include_stop_str_in_output=True,
                    ignore_eos=True, n=1,
                )
            else:
                # Natural generation path (no trace entry for this slot).
                remaining = config.max_tokens - beam.completion_tokens
                if remaining <= 0:
                    # Already at the cumulative cap — skip generation, mark done.
                    beam.completed = True
                    completed_beams.append(beam)
                    step_trace.append((iter_idx, slot_idx, 0, 0.0, True))
                    continue
                sp = SamplingParams(
                    temperature=config.temperature, top_p=config.top_p,
                    max_tokens=remaining,
                    min_tokens=min(20, remaining),
                    stop=["\n\n##", "\n\n\n"], include_stop_str_in_output=True,
                    n=1,
                )
            prompt_text = _step_prompt(
                tokenizer, config, beam.prompt, beam.current_text, is_first)
            req_id = f"bs-p{problem_idx}-it{iter_idx}-s{slot_idx}"
            engine.add_request(req_id, prompt_text, sp)
            request_meta[req_id] = beam
            request_start[req_id] = time.perf_counter()
            submitted.append(beam)
            submitted_slots.append(slot_idx)
            submitted_reqs.append(req_id)
        if not submitted:
            break

        # Drive the engine to completion. LUT-tuned CHUNK_Size_Page is
        # refreshed at threshold crossings, mirroring best_of_n_beam.
        _configure_attention(config)
        if attn_lut is not None:
            prompt_ids = tokenizer(
                [_step_prompt(tokenizer, config, b.prompt, b.current_text, is_first)
                 for b in active],
                add_special_tokens=False,
            )["input_ids"]
            max_prompt_len = max((len(p) for p in prompt_ids), default=0)
            live_n = len(active)
            lut_step = 0
            last_n = -1
            next_kv = 0

        outputs: dict[str, "vllm.RequestOutput"] = {}
        while engine.has_unfinished_requests():
            if attn_lut is not None:
                cur_kv = max_prompt_len + lut_step
                if live_n != last_n or cur_kv >= next_kv:
                    cpd.PROFILE_CHUNK_SIZE_PAGE = attn_lut.pick_csp(live_n, cur_kv)
                    last_n = live_n
                    nxt = int(np.searchsorted(pl_axis, cur_kv, side="right"))
                    next_kv = int(pl_axis[nxt]) if nxt < len(pl_axis) else 10**9

            step_outputs = engine.step()
            if attn_lut is not None:
                lut_step += 1
            for output in step_outputs:
                outputs[output.request_id] = output
                if output.finished:
                    request_end[output.request_id] = time.perf_counter()
                    if attn_lut is not None:
                        live_n -= 1
        if attn_lut is not None:
            cpd.PROFILE_CHUNK_SIZE_PAGE = None

        # Per-beam (own-delta) accumulators. Each beam carries the
        # cumulative `gen_len` / `gen_time_ms` along its own lineage —
        # NOT the per-iter max across siblings. step_trace still records
        # each slot's actual per-iter values.

        # Attribute time + tokens, decide which beams just finished.
        for slot_idx, req_id, beam in zip(
                submitted_slots, submitted_reqs, submitted):
            output = outputs[req_id]
            completion = output.outputs[0]
            gen_len = len(completion.token_ids)
            gen_dt_ms = (request_end[req_id] - request_start[req_id]) * 1000.0
            beam.current_text += completion.text
            beam.completion_tokens += gen_len
            beam.gen_time_ms += gen_dt_ms
            beam.stop_reason = completion.stop_reason

            # Decide whether this slot's beam graduates this iteration.
            # Replay path: forced-length step (ignore_eos=True, stop=[]) makes
            # vLLM always finish via finish_reason="length" + stop_reason=None,
            # so the natural EOS/length checks below would mis-fire. We must
            # instead replay the trace's recorded completion at this slot.
            was_forced = (problem_idx, iter_idx, slot_idx) in REQUEST_STEP_GEN_TOKENS
            if was_forced:
                completed_this_step = (
                    (problem_idx, iter_idx, slot_idx) in REQUEST_STEP_COMPLETED
                    or beam.completion_tokens >= config.max_tokens
                )
            else:
                # Natural-generation completion signals (mirrors
                # sal/search/beam_search.py:170 plus the cumulative cap).
                sr = completion.stop_reason
                sr_str = sr if sr is not None else "EOS"
                completed_this_step = (
                    sr_str == "EOS"
                    or sr_str == "length"
                    or completion.text == ""
                    or beam.completion_tokens >= config.max_tokens
                )

            if completed_this_step:
                beam.completed = True
                completed_beams.append(beam)
            step_trace.append(
                (iter_idx, slot_idx, gen_len, gen_dt_ms, completed_this_step)
            )

        # Early-stop ONLY in REPLAY mode (a trace was loaded). Trace mode —
        # `task_trace` clears the globals before invoking — runs the search
        # to natural exit so the trace captures every iter the model would
        # have actually executed. Replay mode knows the answer is the n
        # earliest completions, so it stops as soon as the threshold is
        # crossed.
        if trace_loaded and len(completed_beams) >= config.n:
            break

        # No PRM scoring — survivor selection is deterministic (first k).
        # The user's goal is reproducible compute load across methods, not
        # search-quality, so we skip PRM entirely.
        candidates = [b for b in submitted if not b.completed]
        if not candidates:
            break

        # Last iteration *or* the trace is exhausted (no entries for the
        # next iter): graduate the remaining candidates to completed and
        # stop. Without the trace-exhaustion check, the loop would keep
        # iterating with natural generation past the baseline's end —
        # all 4 active beams are deepcopies of the same parent, share an
        # engine RNG, and vLLM emits identical content for them, so the
        # final 4 beams collapse to one.
        trace_exhausted = trace_loaded and not any(
            (problem_idx, iter_idx + 1, s) in REQUEST_STEP_GEN_TOKENS
            for s in range(config.n)
        )
        if iter_idx == config.num_iterations - 1 or trace_exhausted:
            for beam in candidates:
                beam.completed = True
                completed_beams.append(beam)
            break

        # Filter duplicate active beams (same as sal/search/beam_search.py:222).
        if config.filter_duplicates:
            seen = {}
            for i, b in enumerate(candidates):
                seen.setdefault(b.current_text, i)
            keep_pos = list(seen.values())
            candidates = [candidates[i] for i in keep_pos]

        # Pick the first n/beam_width as survivors (deterministic, no PRM).
        # Same rule across trace and replay so the search-tree shape is
        # identical for every method.
        keep = config.n // config.beam_width
        beams = []
        for idx, beam in enumerate(candidates):
            if idx < keep:
                beams.append(beam)
            else:
                beam.pruned = True

    # Pad to exactly n entries (FIFO first n). Each beam reports its own
    # per-lineage cumulative completion_tokens and gen_time_ms.
    if not completed_beams:
        completed_beams = beams
    if len(completed_beams) == 0:
        return [], step_trace
    while len(completed_beams) < config.n:
        completed_beams.append(copy.deepcopy(completed_beams[-1]))
    return completed_beams[: config.n], step_trace


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    if config.approach != "beam_search":
        raise ValueError(
            f"beam_search_beam only supports approach=beam_search, got {config.approach}"
        )

    os.environ["VLLM_TRITON_DECODE_KERNEL"] = config.triton_decode_kernel

    num_gpus = torch.cuda.device_count()
    llm_kwargs = dict(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        load_format="safetensors",
        task="generate",
        dtype="auto",
        disable_cascade_attn=True,
    )
    if config.gemm_opt:
        _LADDER = [1, 2, 4] + list(range(8, 513, 8))
        upper = next((s for s in _LADDER if s >= config.n), _LADDER[-1])
        compile_sizes = [s for s in _LADDER if s <= upper]
        llm_kwargs["compilation_config"] = CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            compile_sizes=compile_sizes,
            inductor_compile_config={
                "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
                "epilogue_fusion": True,
            },
        )
    llm = LLM(**llm_kwargs)

    logger.info("Warming up FA and FD-path Triton kernels...")
    _warmup_attn_kernels(llm)
    logger.info("Warmup done.")

    attn_lut = _maybe_load_attn_lut(config)

    # Load per-(problem, step, slot) forced-length trace if one exists. With
    # a trace, FD/ours runs reproduce the exact gen lengths AND completion
    # timing captured by a vllm trace run; without a trace, we generate
    # naturally.
    global REQUEST_STEP_GEN_TOKENS, REQUEST_STEP_COMPLETED
    REQUEST_STEP_GEN_TOKENS, REQUEST_STEP_COMPLETED = load_trace_for_config(config)
    _trace_path = trace_path_for_config(config)
    logger.info(
        "beam_search trace: %d forced (problem, step, slot) entries (%d completed) from %s",
        len(REQUEST_STEP_GEN_TOKENS), len(REQUEST_STEP_COMPLETED), _trace_path,
    )

    dataset = get_dataset(config)

    output_dir = Path(config.output_dir or f"data/{config.model_path}")
    dataset_short = config.dataset_name.split("/")[-1]
    out_dir = output_dir / dataset_short
    out_dir.mkdir(parents=True, exist_ok=True)
    method = _method_suffix(config)
    if not method:
        raise ValueError(
            f"(gemm_opt={config.gemm_opt}, chunk_size={config.chunk_size}) "
            "does not map to a named method. Valid combos: "
            "gemm_opt=true chunk_size=dynamic   -> ours; "
            "gemm_opt=false chunk_size=heuristic -> FD; "
            "gemm_opt=false chunk_size=none      -> vllm; "
            "gemm_opt=false chunk_size=dynamic  -> TurnOffGEMM."
        )
    suffix = f"_{method}"
    timing_csv   = out_dir / f"beam_search_request_timings{suffix}_n{config.n}.csv"
    # Per-(problem, iter, slot) gen_len + gen_time_ms + completed flag, so
    # any future change to the per-beam aggregation rule (or the slicing
    # convention) can be re-derived from these files without another GPU
    # run. Mirrors the per-step CSVs that scripts/beam_search_task_trace.py
    # writes for the vllm baseline.
    per_step_csv = out_dir / f"beam_search_per_step{suffix}_n{config.n}.csv"

    # Stream both CSVs to disk as we finish each problem so a crash mid-sweep
    # still leaves usable partial results.
    with timing_csv.open("w", newline="") as f_t, \
         per_step_csv.open("w", newline="") as f_s:
        w_t = csv.writer(f_t)
        w_t.writerow(["problem", "beam", "token_length", "gen_time_ms"])
        w_s = csv.writer(f_s)
        w_s.writerow([
            "problem", "step_idx", "beam_idx", "gen_len", "gen_time_ms", "completed"
        ])
        for problem_idx, ex in enumerate(dataset):
            problem = ex["problem"]
            t0 = time.perf_counter()
            completed, step_trace = _run_one_problem(
                problem, problem_idx, config, llm, attn_lut)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            problem_short = _short_problem(problem_idx, problem)
            for beam_idx, beam in enumerate(completed):
                w_t.writerow([
                    problem_short, beam_idx,
                    beam.completion_tokens, beam.gen_time_ms,
                ])
            for iter_idx, slot_idx, gen_len, gen_dt_ms, was_completed in step_trace:
                w_s.writerow([
                    problem_short, iter_idx, slot_idx,
                    gen_len, gen_dt_ms, int(was_completed),
                ])
            f_t.flush(); f_s.flush()

            max_t = max((b.gen_time_ms for b in completed), default=0.0)
            logger.info(
                "[%d/%d] beams=%d max_gen_time=%.1fms wall=%.1fms",
                problem_idx + 1, len(dataset), len(completed), max_t, wall_ms,
            )

    logger.info("Saved timings to %s", timing_csv)
    logger.info("Saved per-step trace to %s", per_step_csv)


if __name__ == "__main__":
    main()
