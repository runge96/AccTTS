"""xformers-style chunked-prefill paged-decode kernel."""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from vllm.attention.ops.chunked_prefill_paged_decode import (
    _paged_attn_,
    cdiv_fn,
)
from vllm.attention.ops.prefix_prefill import context_attention_fwd

logger = logging.getLogger(__name__)


def get_split_k(B: int, G: int, H: int, Mk: int, Mq: int = 1) -> int:
    """Port of xformers ``FwOp.get_split_k`` (CUDA branch, no quantization).

    Reference: xformers/ops/fmha/triton_splitk.py:337-380.
    """
    bh = max(B * H, 1)
    if Mq > 1 and B * G * H > 64:
        return 1
    split_k = Mk // bh                   
    max_chunk_size = 64 if (Mk <= 512 and bh <= 64) else 128
    while split_k > Mk / max_chunk_size:
        split_k //= 2
    split_k = min(split_k, 64)
    split_k = max(split_k, 1)
    return split_k


# -- Kernels ---------------------------------------------------------------
# Same bodies as vLLM's _paged_attn_stage1 / _paged_attn_stage2


@triton.jit
def _paged_attn_stage1_xformers(
    tmp_output_ptr,
    logsumexp_ptr,
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    k_scale,
    v_scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    tmp_output_stride_0: tl.int64,
    tmp_output_stride_1: tl.int64,
    tmp_output_stride_2: tl.int64,
    tmp_output_stride_3: tl.int64,
    tmp_output_stride_4: tl.int64,
    logsumexp_stride_0: tl.int64,
    logsumexp_stride_1: tl.int64,
    logsumexp_stride_2: tl.int64,
    logsumexp_stride_3: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_Size_Page: tl.constexpr, 
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    chunk_idx = tl.program_id(2)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    q_padded = tl.arange(0, num_queries_per_kv_padded)

    query_head_idx = kv_head_idx * num_queries_per_kv + q_padded
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)
    Q = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    for block_in_chunk in range(0, CHUNK_Size_Page):
        block_id = chunk_idx * CHUNK_Size_Page + block_in_chunk

        offs_n = tl.arange(0, BLOCK_SIZE)
        seq_offset = block_id * BLOCK_SIZE + offs_n
        seq_mask = (seq_offset < seq_len)

        block_table_offset = seq_idx * block_table_stride + block_id
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[None, :] * stride_v_cache_2 +
                    offs_n[:, None] * stride_v_cache_3)

        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)
        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)
        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        S = tl.where(head_mask[:, None] & seq_mask[None, :], 0.0, float("-inf")).to(tl.float32)
        S += scale * tl.dot(Q, K)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)

    L_safe = tl.where(head_mask, L, 1.0)
    acc = acc / L_safe[:, None]
    tmp_out_ptrs = (
        tmp_output_ptr + seq_idx * tmp_output_stride_0 +
        kv_head_idx * tmp_output_stride_1 +
        chunk_idx * tmp_output_stride_2 +
        q_padded[:, None] * tmp_output_stride_3 +
        offs_d[None, :] * tmp_output_stride_4)
    tl.store(tmp_out_ptrs, acc, mask=dim_mask[None, :] & head_mask[:, None])

    logsumexp = M + tl.log(L_safe)
    logsumexp_ptrs = (logsumexp_ptr + seq_idx * logsumexp_stride_0 +
                      kv_head_idx * logsumexp_stride_1 +
                      chunk_idx * logsumexp_stride_2 +
                      q_padded * logsumexp_stride_3)
    tl.store(logsumexp_ptrs, logsumexp, mask=head_mask)


@triton.jit
def _paged_attn_stage2_xformers(
    output_ptr,
    tmp_output_ptr,
    logsumexp_ptr,
    seq_lens_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    tmp_output_stride_0: tl.int64,
    tmp_output_stride_1: tl.int64,
    tmp_output_stride_2: tl.int64,
    tmp_output_stride_3: tl.int64,
    tmp_output_stride_4: tl.int64,
    logsumexp_stride_0: tl.int64,
    logsumexp_stride_1: tl.int64,
    logsumexp_stride_2: tl.int64,
    logsumexp_stride_3: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_Size_Page: tl.constexpr, 
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_chunks = cdiv_fn(seq_len, CHUNK_Size_Page * BLOCK_SIZE)

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded)
    q_local = tl.arange(0, num_queries_per_kv_padded)
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    logsumexp_total = tl.where(head_mask, float("-inf"), 0.0)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    for chunk_idx in range(0, num_chunks):
        logsumexp_ptrs = (logsumexp_ptr + seq_idx * logsumexp_stride_0 +
                          kv_head_idx * logsumexp_stride_1 +
                          chunk_idx * logsumexp_stride_2 +
                          q_local * logsumexp_stride_3)
        logsumexp_chunk = tl.load(logsumexp_ptrs, mask=head_mask, other=float("-inf"))

        tmp_ptrs = (
            tmp_output_ptr + seq_idx * tmp_output_stride_0 +
            kv_head_idx * tmp_output_stride_1 +
            chunk_idx * tmp_output_stride_2 +
            q_local[:, None] * tmp_output_stride_3 +
            offs_d[None, :] * tmp_output_stride_4)
        acc_chunk = tl.load(tmp_ptrs,
                            mask=dim_mask[None, :] & head_mask[:, None],
                            other=0.0)

        m_j = tl.maximum(logsumexp_total, logsumexp_chunk)
        alpha = tl.exp(logsumexp_total - m_j)
        beta = tl.exp(logsumexp_chunk - m_j)
        denom = alpha + beta
        acc = (acc * alpha[:, None] + acc_chunk * beta[:, None]) / denom[:, None]
        logsumexp_total = m_j + tl.log(denom)
    out_ptrs = (output_ptr +
                cur_batch_in_all_start_index * output_stride_0 +
                query_head_idx[:, None] * output_stride_1 +
                offs_d[None, :])
    tl.store(out_ptrs, acc, mask=dim_mask[None, :] & head_mask[:, None])


# -- Host dispatcher -------------------------------------------------------


def chunked_prefill_paged_decode_xformers(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
):
    """xformers-style replacement for vLLM's chunked_prefill_paged_decode.

    The FA path (max_num_chunks == 1) delegates to vLLM's standard
    ``_paged_attn_``
    """
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[1] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query, k=key, v=value, o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache, v_cache=value_cache,
            b_loc=block_table, b_start_loc=query_start_loc,
            b_seq_len=seq_lens, max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale, v_scale=v_scale,
            alibi_slopes=alibi_slopes, sliding_window=sliding_window,
            sm_scale=sm_scale, skip_decode=True,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)
    head_size_padded = triton.next_power_of_2(head_size)

    max_seq_len = int(max_seq_len)

    # xformers parity: chunk size from get_split_k, continuous in max_seq_len.
    split_k = get_split_k(num_seqs, num_kv_heads, num_queries_per_kv, max_seq_len, Mq=1)
    if split_k <= 1:
        CHUNK_Size_Page = (max_seq_len + block_size - 1) // block_size
    else:
        tokens_per_split = math.ceil(max_seq_len / split_k)
        CHUNK_Size_Page = max(1, math.ceil(tokens_per_split / block_size))
    CHUNK_Size_Tokens = CHUNK_Size_Page * block_size
    max_num_chunks = (max_seq_len + CHUNK_Size_Tokens - 1) // CHUNK_Size_Tokens

    if max_num_chunks == 1:
        # FA path
        grid = (num_seqs, num_kv_heads)
        _paged_attn_[grid](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=block_size,
            CHUNK_Size_Page=CHUNK_Size_Page,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_ALIBI_SLOPES=use_alibi_slopes,
            SLIDING_WINDOW=sliding_window,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
        )
        return

    # FD path 
    grid_stage1 = (num_seqs, num_kv_heads, max_num_chunks)
    tmp_output = torch.empty(
        size=(num_seqs, num_kv_heads, max_num_chunks,
              num_queries_per_kv_padded, head_size_padded),
        dtype=torch.float32,
        device=output.device,
    )
    logsumexp = torch.empty(
        size=(num_seqs, num_kv_heads, max_num_chunks, num_queries_per_kv_padded),
        dtype=torch.float32,
        device=output.device,
    )
    _paged_attn_stage1_xformers[grid_stage1](
        tmp_output_ptr=tmp_output,
        logsumexp_ptr=logsumexp,
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=alibi_slopes,
        scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        tmp_output_stride_0=tmp_output.stride(0),
        tmp_output_stride_1=tmp_output.stride(1),
        tmp_output_stride_2=tmp_output.stride(2),
        tmp_output_stride_3=tmp_output.stride(3),
        tmp_output_stride_4=tmp_output.stride(4),
        logsumexp_stride_0=logsumexp.stride(0),
        logsumexp_stride_1=logsumexp.stride(1),
        logsumexp_stride_2=logsumexp.stride(2),
        logsumexp_stride_3=logsumexp.stride(3),
        BLOCK_SIZE=block_size,
        CHUNK_Size_Page=CHUNK_Size_Page,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        SLIDING_WINDOW=sliding_window,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
    )

    grid_stage2 = (num_seqs, num_kv_heads)
    _paged_attn_stage2_xformers[grid_stage2](
        output_ptr=output,
        tmp_output_ptr=tmp_output,
        logsumexp_ptr=logsumexp,
        seq_lens_ptr=seq_lens,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        tmp_output_stride_0=tmp_output.stride(0),
        tmp_output_stride_1=tmp_output.stride(1),
        tmp_output_stride_2=tmp_output.stride(2),
        tmp_output_stride_3=tmp_output.stride(3),
        tmp_output_stride_4=tmp_output.stride(4),
        logsumexp_stride_0=logsumexp.stride(0),
        logsumexp_stride_1=logsumexp.stride(1),
        logsumexp_stride_2=logsumexp.stride(2),
        logsumexp_stride_3=logsumexp.stride(3),
        BLOCK_SIZE=block_size,
        CHUNK_Size_Page=CHUNK_Size_Page,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
    )


# -- Patch / unpatch -------------------------------------------------------

# Backends that import chunked_prefill_paged_decode at module load time.
# Patching cpd's module attribute would not propagate (`from X import Y`
# captures the binding); we have to update each importing module's binding.
_BACKEND_MODULES = (
    "vllm.v1.attention.backends.triton_attn",
)

_original_refs: dict[str, object] = {}
_patched: bool = False


def enable() -> None:
    """Replace ``chunked_prefill_paged_decode`` in every importing backend
    with the xformers-style version. Idempotent."""
    global _patched
    if _patched:
        return
    import importlib

    for mod_name in _BACKEND_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            logger.warning("xformers_attn.enable: cannot import %s (%s)", mod_name, e)
            continue
        if hasattr(mod, "chunked_prefill_paged_decode"):
            _original_refs[mod_name] = mod.chunked_prefill_paged_decode
            mod.chunked_prefill_paged_decode = chunked_prefill_paged_decode_xformers
            logger.info(
                "xformers_attn: patched %s.chunked_prefill_paged_decode", mod_name,
            )
    _patched = True


def disable() -> None:
    """Restore the original ``chunked_prefill_paged_decode`` references."""
    global _patched
    if not _patched:
        return
    import importlib

    for mod_name, orig in _original_refs.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        mod.chunked_prefill_paged_decode = orig
    _original_refs.clear()
    _patched = False


def is_enabled() -> bool:
    return _patched
