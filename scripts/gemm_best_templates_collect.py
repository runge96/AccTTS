#!/usr/bin/env python
# Pre-compile a dense ladder of shape-specific GEMM kernels for vLLM and
# persist them to the on-disk compile cache. After running this once, any
# subsequent bench run with the same model and the same inductor_compile_config
# (regardless of its own compile_sizes value) will hit the cache for any size
# in this ladder, skipping the autotune warmup entirely.
#
# Usage:
#   python scripts/gemm_best_templates_collect.py recipes/best-of-n.yaml
#
# Cache lands at ~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/...

import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel

from sal.config import Config
from sal.utils.parser import H4ArgumentParser

# Match vLLM v1's default cudagraph_capture_sizes ladder so every runtime
# batch size that the engine will ever see (after pad_for_cudagraph rounds up)
# has a pre-compiled shape-specific kernel.
COMPILE_LADDER = [1, 2, 4] + list(range(8, 65, 8))

# Must mirror the inductor_compile_config used by the bench (gemm_opt=True
# branch in scripts/test_time_compute.py). The parent cache hash is derived
# from this dict, so any divergence here invalidates cache reuse.
INDUCTOR_COMPILE_CONFIG = {
    # max_autotune_gemm intentionally OFF: general-shape compile (used for prefill
    # sizes that miss the ladder, e.g. 600) falls back to aten.mm → cuBLAS heuristic.
    # Shape-specific compiles (entries in compile_sizes) still autotune because
    # vLLM auto-enables max_autotune=True for runtime_shape=int.
    "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
    "epilogue_fusion": True,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        load_format="safetensors",
        task="generate",
        dtype="auto",
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            compile_sizes=COMPILE_LADDER,
            inductor_compile_config=INDUCTOR_COMPILE_CONFIG,
        ),
    )

    # Force one trivial generation so any deferred kernel loads finalize
    # and the cache is fully written to disk.
    llm.generate(["hello"], SamplingParams(temperature=0.0, max_tokens=1))
    print(
        f"Pre-compile finished for {len(COMPILE_LADDER)} shapes: "
        f"{COMPILE_LADDER[0]}..{COMPILE_LADDER[-1]}. "
        f"Cache populated under ~/.cache/vllm/torch_compile_cache/."
    )


if __name__ == "__main__":
    main()
