# Minimal Triton demo for Nsight Systems
# Usage:
#   python scripts/triton_nsys_demo.py
#   nsys profile --trace=cuda,nvtx --stats=true -o triton_demo \
#       python scripts/triton_nsys_demo.py
#
# This launches one Triton kernel (vector add) so you can validate Nsight setup.

import torch
import triton
import triton.language as tl
import torch.cuda.nvtx as nvtx


@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    # Grid settings
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    
    # Warmup
    vec_add_kernel[grid](x, y, out, n_elements=n, BLOCK=256)
    torch.cuda.synchronize()
    
    # NVTX tag + Profiler tag 
    torch.cuda.cudart().cudaProfilerStart()
    
    nvtx.range_push("triton_label")
    vec_add_kernel[grid](x, y, out, n_elements=n, BLOCK=256)
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    torch.cuda.cudart().cudaProfilerStop()

    # Simple correctness check
    torch.testing.assert_close(out, x + y)
    print("OK: kernel ran and output matches.")


if __name__ == "__main__":
    main()
