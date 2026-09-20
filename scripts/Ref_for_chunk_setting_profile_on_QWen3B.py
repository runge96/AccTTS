#!/usr/bin/env python
"""Minimal FA vs FD-8 timing sanity check.

Usage:
    python scripts/test.py recipes/best-of-n.yaml
"""

import os
import sys
import time

import torch
from vllm import LLM, SamplingParams
from vllm.attention.ops import chunked_prefill_paged_decode as cpd
from vllm.inputs import TokensPrompt

from sal.config import Config
from sal.utils.parser import H4ArgumentParser

GEN_TOKENS = 102


# NOTE !!!
# This file has a different timing trends from chunk_setting_profile.py.
# We found QWen2.5-3B-Instruct model is very sensitive to the way how you write chunk_setting_profile.py.
# 1. Reduce unnecessary lib import;
# 2. Remove all unnecessary CONSTANT variables;
# 3. Remove all unnecessary functions;
# 4. logger has impacts on timing results, remove.


def _load_config():
    yaml_path = next((a for a in sys.argv[1:] if a.endswith(".yaml")), None)
    if yaml_path is None:
        raise SystemExit("Pass a yaml path, e.g. recipes/best-of-n.yaml")
    parser = H4ArgumentParser(Config)
    out = parser.parse_yaml_file(os.path.abspath(yaml_path))
    return out[0] if isinstance(out, (list, tuple)) else out


def run_variant(llm, n_beams, token_ids, label, csp):
    cpd.PROFILE_CHUNK_SIZE_PAGE = csp
    engine = llm.llm_engine
    sp = SamplingParams(
        temperature=0.8, top_p=1.0, seed=0, n=1,
        max_tokens=GEN_TOKENS, min_tokens=GEN_TOKENS,
    )
    rids = [f"{label}-{i}" for i in range(n_beams)]
    starts, ends = {}, {}
    for rid in rids:
        engine.add_request(rid, TokensPrompt(prompt_token_ids=token_ids), sp)
        starts[rid] = time.perf_counter()
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                ends[o.request_id] = time.perf_counter()
    return [(rid, (ends[rid] - starts[rid]) * 1000.0) for rid in rids]


def main():
    config = _load_config()
    os.environ["VLLM_TRITON_DECODE_KERNEL"] = config.triton_decode_kernel

    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        load_format="safetensors",
        task="generate",
        dtype="auto",
        max_num_seqs=config.n,
        disable_cascade_attn=True,
    )

    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    base_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": "Solve: 2+2=?"},
        ],
        tokenize=True, add_generation_prompt=True,
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    target = max(config.padded_prompt_len, len(base_ids))
    token_ids = [pad_id] * (target - len(base_ids)) + list(base_ids)

    # Untimed pre-warm at the measurement shape: fire each variant once so the
    # subsequent timed runs all start from steady state (CUDA-graph captured,
    # prefix cache populated, kernel state primed).
    for label, csp in [("FA", None), ("FD-8", 8)]:
        run_variant(llm, config.n, token_ids, f"warmup-{label}", csp)

    for label, csp in [("FA", None), ("FD-8", 8)]:
        # results = run_variant(llm, config.n, token_ids, label, csp)
        results = run_variant(llm, config.n, token_ids, label, csp)
        print(f"=== {label}  n_beams={config.n}  prompt_len={len(token_ids)}  gen={GEN_TOKENS} ===")
        for rid, ms in results:
            print(f"  {rid}: {ms:.3f} ms")
        times = [ms for _, ms in results]
        print(f"  min={min(times):.3f}  max={max(times):.3f}  mean={sum(times) / len(times):.3f}")


if __name__ == "__main__":
    main()
