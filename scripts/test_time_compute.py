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

import logging
import os

# os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
from vllm import LLM, SamplingParams
from vllm.attention.ops import chunked_prefill_paged_decode as cpd
from vllm.attention.ops.attn_timing import set_enabled as set_attn_timing_enabled
from vllm.config import CompilationConfig, CompilationLevel

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n_beam, dvts
from sal.utils.attn_lut import AttnLUT
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n_beam,
}


def _warmup_attn_kernels(llm):
    """JIT-compile both FA and FD-path Triton kernels before measurement.

    FA fires when max_num_chunks==1 (PROFILE_CHUNK_SIZE_PAGE=None uses the
    ceil() formula). FD fires when seq_len > CHUNK_Size_Page * BLOCK_SIZE
    (setting PROFILE_CHUNK_SIZE_PAGE=1 → threshold of one block).
    """
    sp = SamplingParams(temperature=0.0, max_tokens=20, min_tokens=20)
    prev = cpd.PROFILE_CHUNK_SIZE_PAGE
    try:
        cpd.PROFILE_CHUNK_SIZE_PAGE = None  # FA: _paged_attn_
        llm.generate(["warmup"], sp, use_tqdm=False)
        cpd.PROFILE_CHUNK_SIZE_PAGE = 4     # FD: _paged_attn_stage1/stage2
        llm.generate(["warmup"], sp, use_tqdm=False)
    finally:
        cpd.PROFILE_CHUNK_SIZE_PAGE = prev


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    os.environ["VLLM_TRITON_DECODE_KERNEL"] = config.triton_decode_kernel
    set_attn_timing_enabled(config.timing_enabled)

    approach_fn = APPROACHES[config.approach]

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
        # Custom triton_attn backend asserts attn_metadata.use_cascade is
        # False; cascade triggers when many requests share a long prefix
        # (n identical beams with prefix caching). Disable engine-wide.
        disable_cascade_attn=True,
    )
    if config.gemm_opt:
        # Derive compile_sizes from config.n. Best-of-N's active beam count
        # walks N → 1, and pad_for_cudagraph rounds each step up to the next
        # entry in vLLM's default cudagraph_capture_sizes ladder. So we need
        # every ladder entry up to and including the smallest entry >= N.
        _LADDER = [1, 2, 4] + list(range(8, 513, 8))
        _upper = next((s for s in _LADDER if s >= config.n), _LADDER[-1])
        compile_sizes = [s for s in _LADDER if s <= _upper]
        llm_kwargs["compilation_config"] = CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            compile_sizes=compile_sizes,
            inductor_compile_config={
                # max_autotune_gemm OFF → general-shape compile uses cuBLAS via
                # aten.mm. Prefill at sizes not in compile_sizes falls here and
                # benefits from cuBLAS heuristics without per-shape autotune cost.
                # Shape-specific compiles still autotune via vLLM's auto-enable
                # of max_autotune=True for runtime_shape=int.
                "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
                "epilogue_fusion": True,
            },
        )
    llm = LLM(**llm_kwargs)
    # llm = LLM(
    #     model=config.model_path,
    #     gpu_memory_utilization=config.gpu_memory_utilization,
    #     enable_prefix_caching=True,
    #     seed=config.seed,
    #     tensor_parallel_size=num_gpus,
    #     load_format="safetensors",
    #     block_size=32,  # choose 8/16/32/64/128
    # )
    if config.disable_prm:
        logger.info("PRM disabled via config.disable_prm — placeholder pred from beam 0.")
        prm = None
    else:
        prm = load_prm(config)

    logger.info("Warming up FA and FD-path Triton kernels...")
    _warmup_attn_kernels(llm)
    logger.info("Warmup done.")

    # Load the per-(model, dataset, n) gen-token trace if one exists. Forces
    # min==max tokens for each beam so per-step KV/timing match the recorded
    # run (no sampling-driven length variance). Missing file -> free decode.
    # NOTE: sal/search/__init__.py does `from .best_of_n_beam import
    # best_of_n_beam`, which shadows the submodule attribute with the
    # function. So `import sal.search.best_of_n_beam as _bn_mod` resolves
    # to the function, not the module. Use importlib to get the real module
    # from sys.modules.
    if config.approach == "best_of_n":
        import importlib
        _bn_mod = importlib.import_module("sal.search.best_of_n_beam")
        #########################################
        _bn_mod.REQUEST_GEN_TOKENS = _bn_mod.load_trace_for_config(config)
        #########################################
        _trace_path = _bn_mod.trace_path_for_config(config)
        logger.info(
            "best_of_n trace: %d forced (problem, beam) entries from %s",
            len(_bn_mod.REQUEST_GEN_TOKENS), _trace_path,
        )

    _lut = AttnLUT.from_env(model_path=config.model_path)
    if _lut is not None and config.chunk_size != "none":
        logger.info(
            "Attention LUT loaded from %s; chunk_size=%s, "
            "chunked-prefill paged decode will switch FA/FD-x at runtime.",
            _lut.source, config.chunk_size,
        )
    elif _lut is not None:
        logger.info(
            "Attention LUT loaded from %s but chunk_size=none; "
            "set chunk_size: dynamic or heuristic to enable runtime switching.",
            _lut.source,
        )
    else:
        logger.info(
            "No attention LUT found (set %s or place profile_results_lut.json "
            "under data/%s/); using vLLM default CHUNK_Size_Page formula.",
            "SAL_ATTN_LUT_PATH", config.model_path,
        )

    dataset = get_dataset(config)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        with_indices=True,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done ??!")


if __name__ == "__main__":
    main()
