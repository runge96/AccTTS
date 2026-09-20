#!/usr/bin/env python
"""Profile CHUNK_Size_Page settings across (n_beams, kv_token_len) combinations.

Usage:
    python scripts/chunk_setting_profile.py recipes/best-of-n.yaml

For each (n_beams, prompt_len) combination, fan out n_beams identical requests
and measure first-beam end-to-end latency under several attention configs:

    FA           -- _paged_attn_  (CHUNK_Size_Page = ceil(seq_len/block_size))
    FD-{4,6,8,16,32,64}
                 -- _paged_attn_stage1 + _paged_attn_stage2 with CHUNK_Size_Page=N

Selection between FA and FD is driven by the module-level override
PROFILE_CHUNK_SIZE_PAGE in vllm.attention.ops.chunked_prefill_paged_decode:
None forces the original ceil() formula -> max_num_chunks==1 -> FA path; an
int N forces CHUNK_Size_Page=N, which routes to FD when N*block_size<seq_len.

CHUNK_Size_Page in both _paged_attn_stage1 and _paged_attn_stage2 is plain int
(non-constexpr), so changing the value does not trigger Triton recompilation.
That removes the per-csp warmup; we still take min() over a few measurements
to absorb one-time costs (Triton first-compile of FD kernels, vLLM piecewise
CUDA-graph capture of new shapes).

Sweep:
  - n_beams:    [1, 2, 4, 8, 16, 32, 64, 128, 256]
  - prompt_len: nearest mult of 128 >= initial prompt len; step 128 up to 1024,
                then step 256 up to 4096
  - skip combos with n_beams * prompt_len > 256 * 2048  (4090 memory ceiling)
  - variants:   FA + FD-{4,6,8,16,32,64} (skip FD-N when prompt_len < N * 16)
  - gen_tokens: max(20, 5% of prompt_len)

Output is a long-format CSV at scripts/chunk_setting_profile_results.csv,
written line-buffered + flushed so it can be opened/read while the run is in
progress. OOM and other runtime errors are written as `time_ms=nan` with the
error string, and the sweep continues.
"""

import csv
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.attention.ops import chunked_prefill_paged_decode as cpd
from vllm.inputs import TokensPrompt

from sal.config import Config
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SEED = 0
# Match vLLM's default cudagraph_capture_sizes ladder
N_BEAMS_LIST = [1, 2, 4] + list(range(8, 65, 8))
# N_BEAMS_LIST = [4]
PROMPT_LEN_FINE_STEP = 64   # used until prompt_len reaches PROMPT_LEN_BREAK
PROMPT_LEN_COARSE_STEP = 256  # used after PROMPT_LEN_BREAK
PROMPT_LEN_BREAK = 1024
PROMPT_LEN_MAX = 4096
NSEQ_KVLEN_MAX = 256 * 2048  # 4090 memory budget: skip combos beyond this
CHUNK_SIZE_PAGE_LIST = [4, 6, 8, 16, 32, 64]
# CHUNK_SIZE_PAGE_LIST = [4, 6, 8]
GEN_RATIO = 0.05
GEN_MIN = 20
BLOCK_SIZE = 16
WARMUP_PROMPT_LEN = 256  # > FD-4 threshold (4 * BLOCK_SIZE = 64) so FD actually fires
WARMUP_GEN_TOKENS = 20


def _find_single_token_pad(tokenizer):
    for s in [" x", " 0", " a", "x", "0", "a"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise RuntimeError("Could not find a single-token pad string for this tokenizer")


def _build_prompt_lens(start_len: int, max_len: int) -> list[int]:
    """Step PROMPT_LEN_FINE_STEP up to PROMPT_LEN_BREAK inclusive, then PROMPT_LEN_COARSE_STEP."""
    lens = []
    cur = start_len
    while cur <= max_len:
        lens.append(cur)
        step = (
            PROMPT_LEN_FINE_STEP if cur < PROMPT_LEN_BREAK else PROMPT_LEN_COARSE_STEP
        )
        cur += step
    return lens


def _build_token_ids(tokenizer, system_prompt, user_msg, target_len, pad_id):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    base_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True
    )
    base_len = len(base_ids)
    if target_len < base_len:
        return None
    return list(base_ids) + [pad_id] * (target_len - base_len)


def _drain(engine):
    while engine.has_unfinished_requests():
        engine.step()


def _load_existing_results(path):
    """Return set of (n_beams, prompt_len, variant) tuples to skip on resume.

    Reads the CSV, finds the FIRST row whose time_ms is NaN (OOM / ERR), and
    returns every combo recorded BEFORE the row just above that error. The row
    immediately above the first error is intentionally NOT in the returned
    set, so the resumed sweep re-runs that combo first — its result serves as
    a fresh in-process warmup before re-attempting the row that failed.

    Side effect: if an error is found, the CSV is truncated in place to drop
    the row above the first error, the error row, and everything after it.
    Without this, repeated resumes would keep stacking duplicate rows on top
    of the original error.

    If no error row is present, every parsed combo is returned and the file
    is left untouched.
    """
    if not path.exists():
        return set()

    # Read all rows so we can both parse and (if needed) rewrite.
    with open(path, newline="") as f:
        all_rows = list(csv.reader(f))

    # Header section: leading "#" rows, blanks, and one column-header row.
    header_end = None
    header_seen = False
    for i, row in enumerate(all_rows):
        if not row or row[0].startswith("#"):
            continue
        if not header_seen:
            header_seen = True
            header_end = i + 1
            continue
        break
    if header_end is None:
        return set()

    data_rows = all_rows[header_end:]
    parsed = []  # list of ((n, p, variant) | None, is_error)
    for row in data_rows:
        try:
            key = (int(row[0]), int(row[1]), row[2])
        except (ValueError, IndexError):
            parsed.append((None, False))
            continue
        try:
            tv = float(row[4]) if len(row) > 4 else float("nan")
            is_err = tv != tv  # NaN self-inequality
        except ValueError:
            is_err = True
        parsed.append((key, is_err))

    first_err = next((i for i, (_, e) in enumerate(parsed) if e), None)
    if first_err is None:
        return {k for k, _ in parsed if k is not None}

    cutoff = max(0, first_err - 1)

    # Truncate file: keep headers + first `cutoff` data rows.
    kept = all_rows[:header_end] + data_rows[:cutoff]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in kept:
            writer.writerow(r)

    return {k for k, _ in parsed[:cutoff] if k is not None}


# [warmup] for both FA and FD is very important
# Otherwise, JIT compilation time will obviously affect profiling results, which is contrary to seperate run of test_time_compute.py
# Meanwhile, in practice, we switch between FA/FDs dynamically. Pre-compilation or warmup is necessary.
def _global_warmup(llm, tokenizer, system_prompt, sample_user, pad_id):
    """Compile both FA and FD-path Triton kernels before the sweep.

    Two short generations against the same padded prompt:
      - FA pass (PROFILE_CHUNK_SIZE_PAGE=None): compiles `_paged_attn_`.
      - FD pass (PROFILE_CHUNK_SIZE_PAGE=4):    compiles `_paged_attn_stage1`
                                                and `_paged_attn_stage2`.

    vLLM's startup dummy inference typically already compiles FA (its short
    sequences hit `max_num_chunks==1`), but we force it here too so the warmup
    does not depend on that assumption. The FD path only fires when
    `max_seq_len > CHUNK_Size_Page * BLOCK_SIZE`, which the startup dummy never
    hits, so FD must be warmed explicitly.
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample_user},
    ]
    base_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True
    )
    if len(base_ids) < WARMUP_PROMPT_LEN:
        token_ids = list(base_ids) + [pad_id] * (WARMUP_PROMPT_LEN - len(base_ids))
    else:
        token_ids = list(base_ids)

    prev = cpd.PROFILE_CHUNK_SIZE_PAGE
    try:
        cpd.PROFILE_CHUNK_SIZE_PAGE = None  # FA: max_num_chunks==1 → _paged_attn_
        _measure_first_beam_ms(llm, 1, token_ids, WARMUP_GEN_TOKENS)
        cpd.PROFILE_CHUNK_SIZE_PAGE = 4     # FD: max_num_chunks>1 → stage1/stage2
        _measure_first_beam_ms(llm, 1, token_ids, WARMUP_GEN_TOKENS)
    finally:
        cpd.PROFILE_CHUNK_SIZE_PAGE = prev


def _measure_first_beam_ms(llm, n_beams, token_ids, gen_tokens):
    """Submit n_beams identical requests; return ms from submit to profile-0 done.

    Aborts the remaining beams as soon as profile-0 finishes to keep the sweep
    tractable for large n_beams.
    """
    engine = llm.llm_engine
    sp = SamplingParams(
        temperature=0.8,
        top_p=1.0,
        seed=SEED,
        n=1,
        max_tokens=gen_tokens,
        min_tokens=gen_tokens,
    )

    _drain(engine)

    rids = [f"profile-{i}" for i in range(n_beams)]
    start_times = {}
    for rid in rids:
        engine.add_request(rid, TokensPrompt(prompt_token_ids=token_ids), sp)
        start_times[rid] = time.perf_counter()

    target = rids[0]
    end_t = None
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for o in outputs:
            if o.finished and o.request_id == target:
                end_t = time.perf_counter()
                break
        if end_t is not None:
            for other in rids[1:]:
                try:
                    engine.abort_request(other)
                except Exception:
                    pass
            _drain(engine)
            break

    if end_t is None:
        raise RuntimeError(f"profile-0 never reported finished (n_beams={n_beams})")
    return (end_t - start_times[target]) * 1000.0


def _load_config():
    """Load Config from a yaml path on the command line.

    Bypasses H4ArgumentParser.parse(): that helper takes the elif branch
    (parse_yaml_and_args) whenever sys.argv has more than two entries, then
    crashes splitting any non-`key=value` token on '='. We only need the yaml
    here, so locate it and call parse_yaml_file directly.
    """
    yaml_path = next((a for a in sys.argv[1:] if a.endswith(".yaml")), None)
    if yaml_path is None:
        raise SystemExit(
            "Pass a yaml config path, e.g. "
            "python scripts/chunk_setting_profile.py recipes/best-of-n.yaml"
        )
    parser = H4ArgumentParser(Config)
    output = parser.parse_yaml_file(os.path.abspath(yaml_path))
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def main():
    config = _load_config()
    os.environ["VLLM_TRITON_DECODE_KERNEL"] = config.triton_decode_kernel

    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=SEED,
        load_format="safetensors",
        task="generate",
        dtype="auto",
        max_num_seqs=max(N_BEAMS_LIST),
        # The custom triton_attn backend asserts attn_metadata.use_cascade is
        # False (see vllm/v1/attention/backends/triton_attn.py). vLLM enables
        # cascade attention automatically when many requests share a long
        # prefix — exactly the regime this profiler runs in (n_beams identical
        # prompts with prefix caching on). Disable it engine-wide so the
        # scheduler never builds metadata that triggers the assertion.
        disable_cascade_attn=True,
    )
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    pad_id = _find_single_token_pad(tokenizer)
    sample_user = "Solve: 2+2=?"

    base_msgs = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": sample_user},
    ]
    base_ids = tokenizer.apply_chat_template(
        base_msgs, tokenize=True, add_generation_prompt=True
    )
    initial_prompt_len = len(base_ids)
    logger.info("Initial prompt length: %d", initial_prompt_len)

    logger.info("Warming up FD path Triton kernels...")
    _global_warmup(llm, tokenizer, config.system_prompt, sample_user, pad_id)
    logger.info("Warmup done; starting sweep.")

    start_len = (
        (initial_prompt_len + PROMPT_LEN_FINE_STEP - 1) // PROMPT_LEN_FINE_STEP
    ) * PROMPT_LEN_FINE_STEP
    prompt_lens = _build_prompt_lens(start_len, PROMPT_LEN_MAX)
    logger.info("Prompt lengths to profile (%d): %s", len(prompt_lens), prompt_lens)

    out_dir = Path(config.output_dir) if config.output_dir else Path("data") / config.model_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunk_setting_profile_results.csv"

    # Resume: if the CSV already exists, parse it. _load_existing_results
    # finds the first row with a NaN time_ms (OOM / ERR), excludes the row
    # immediately above it (so it gets re-run as fresh in-process warmup),
    # and truncates the CSV in place to drop the error row + everything
    # after the warmup row.
    done_set = _load_existing_results(out_path)
    file_exists = out_path.exists()
    if file_exists:
        logger.info(
            "Resuming from %s: %d entries already recorded; skipping them.",
            out_path, len(done_set),
        )

    with open(out_path, "a" if file_exists else "w", buffering=1, newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["# model", config.model_path])
            writer.writerow(["# initial_prompt_len", initial_prompt_len])
            writer.writerow(["# block_size", BLOCK_SIZE])
            writer.writerow(["# seed", SEED])
            writer.writerow(["# nseq_kvlen_max", NSEQ_KVLEN_MAX])
            writer.writerow([])
            writer.writerow(
                ["n_beams", "prompt_len", "variant", "gen_tokens", "time_ms", "note"]
            )
            f.flush()

        for n_beams in N_BEAMS_LIST:
            for prompt_len in prompt_lens:
                if n_beams * prompt_len > NSEQ_KVLEN_MAX:
                    continue

                token_ids = _build_token_ids(
                    tokenizer, config.system_prompt, sample_user, prompt_len, pad_id
                )
                if token_ids is None:
                    continue

                gen_tokens = max(GEN_MIN, int(prompt_len * GEN_RATIO))

                variants = [("FA", None)] + [
                    (f"FD-{c}", c)
                    for c in CHUNK_SIZE_PAGE_LIST
                    if prompt_len >= c * BLOCK_SIZE
                ]
                # variants = [("FA", None)] + [("FD-8", 8)] + [
                #     (f"FD-{c}", c)
                #     for c in CHUNK_SIZE_PAGE_LIST
                #     if prompt_len >= c * BLOCK_SIZE
                # ] + [("FA", None)]
                # variants = [(f"FD-{8}", 8)]
                # variants = [("FA", None)]

                for variant_name, csp in variants:
                    if (n_beams, prompt_len, variant_name) in done_set:
                        logger.info(
                            "Skip (already done): n_beams=%d prompt_len=%d %s",
                            n_beams, prompt_len, variant_name,
                        )
                        continue

                    cpd.PROFILE_CHUNK_SIZE_PAGE = csp

                    # # Reset prefix cache so each variant pays the same prefill
                    # # cost. Without this, variants after the first find their
                    # # entire prompt already cached (from the previous variant's
                    # # run) and pay zero prefill — biasing FD variants to look
                    # # faster than FA which ran first.
                    # llm.llm_engine.reset_prefix_cache()

                    note = ""
                    try:
                        # Triton FA + FD kernels are compiled in the global
                        # warmup, and vLLM pre-captures cudagraphs for default
                        # ladder sizes (which cover N_BEAMS_LIST). One run is
                        # enough.
                        time_ms = _measure_first_beam_ms(
                            llm, n_beams, token_ids, gen_tokens
                        )
                    except torch.cuda.OutOfMemoryError as e:
                        time_ms = float("nan")
                        note = f"OOM: {e}"
                        torch.cuda.empty_cache()
                        try:
                            _drain(llm.llm_engine)
                        except Exception:
                            pass
                    except Exception as e:
                        time_ms = float("nan")
                        note = f"ERR: {type(e).__name__}: {e}"
                        # Log full traceback for diagnosis (the assertion
                        # message itself is often empty for bare `assert`s).
                        logger.error(
                            "Exception at n_beams=%d prompt_len=%d %s:\n%s",
                            n_beams, prompt_len, variant_name,
                            traceback.format_exc(),
                        )

                    writer.writerow(
                        [n_beams, prompt_len, variant_name, gen_tokens, time_ms, note]
                    )
                    f.flush()
                    logger.info(
                        "n_beams=%d prompt_len=%d %s gen=%d -> %.3f ms %s",
                        n_beams, prompt_len, variant_name, gen_tokens, time_ms, note,
                    )

        cpd.PROFILE_CHUNK_SIZE_PAGE = None


if __name__ == "__main__":
    main()
