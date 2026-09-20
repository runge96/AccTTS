# accuracy_verify_QWen_MATH500.md

**Task for Claude Code:** run the **best-of-n** test-time-compute method on the
**MATH-500** dataset with the **Qwen2.5-1.5B-Instruct** reasoning model, score
every beam with the PRM, and save the results as JSON.

- Run **ours** at beam counts `n ∈ [2, 4, 8, 16, 32, 64]`
- Run **vllm** at beam counts `n ∈ [2, 4, 8]` (the higher-`n` vllm runs already
  exist under `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/best_of_n/`)

## Working rules

1. Create one log file, `logs/accuracy_verify_QWen_MATH500.log`, in the repo.
2. Record every command you run and its status (success/fail) in the log file.
3. If you find a problem you may modify code — but record every modification in
   the log file.
4. Whenever you modify code under `src/` or `scripts/`, `git add` + `git commit`.
   Per-run edits to `recipes/best-of-n.yaml` are *configuration*, not code, and
   do **not** need commits.
5. If a particular `(n, problem)` keeps OOM'ing or hanging, skip it and record
   the problem index + `n` in the log rather than blocking the rest of the sweep.
6. Use the correct conda env and set the variables before running anything.

---

## Step 0 — Environment

```bash
cd $REPO_ROOT
conda activate sal_new_vllm
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

`HF_TOKEN` is **not** needed — Qwen2.5-1.5B-Instruct and HuggingFaceH4/MATH-500
are public (unlike the gated Llama model).

---

## Step 1 — Configure `recipes/best-of-n.yaml` (once, before the sweep)

Edit these fields:

| field | value | note |
|---|---|---|
| `dataset_name` | `HuggingFaceH4/MATH-500` | MATH-500 |
| `dataset_split` | `test` | MATH-500's split (500 problems) |
| `model_path` | `Qwen/QWen2.5-1.5B-Instruct` | reasoning model — **keep this exact casing** (`QWen`); it determines the output dir `data/Qwen/QWen2.5-1.5B-Instruct/`, which must match the existing data folder |
| `prm_path` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | the PRM (same as the AIME task) |
| `custom_chat_template` | `null` | use Qwen's built-in chat template |
| `max_tokens` | **`2048`** | |
| `disable_prm` | **`false`** | PRM must be loaded and score every beam |
| `gpu_memory_utilization` | **`0.6`** | with `disable_prm: false` the Skywork-1.5B PRM shares the 24 GB RTX 4090 with vLLM; drop to `0.5` if the PRM load OOMs |
| `approach` | `best_of_n` | |
| `n` | per-run (see Step 4) | |
| `gemm_opt` | per-method (see Step 4) | |
| `chunk_size` | per-method (see Step 4) | |
| `num_samples` | keep commented out | runs all 500 MATH-500 problems |

Leave `agg_strategy` unset — it defaults to `"last"` (best-of-n picks the beam
with the highest last-step PRM score).

---

## Step 2 — Set the MATH-500 system prompt (code change → commit)

`src/sal/config.py` has a module-level preset that selects the system prompt.
It is currently set to `"AIME"`; for MATH-500 it must be `"MATH500"`, otherwise
every problem is prefixed with the wrong (AIME systematic-thinking) prompt.

Change the single line in `src/sal/config.py`:

```python
_SYSTEM_PROMPT_PRESET = "MATH500"      # was "AIME"
```

Keep the assignment on one physical line (a regex-based driver depends on that).
Then `git add src/sal/config.py` + `git commit`, and record it in the log.

---

## Step 3 — Free decoding, NO trace forcing

Every beam must generate freely and stop on its **own** criteria (EOS token or
`max_tokens`), for **both** vllm and ours. Do **not** force per-beam token
counts from a trace file.

How the trace mechanism works (`src/sal/search/best_of_n_beam.py`):

- `scripts/test_time_compute.py` calls `load_trace_for_config(config)`, which
  reads `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/beams={n}`.
- If that file **exists**, every beam's `min_tokens == max_tokens` is forced to
  the recorded length (replay mode) — **not what we want**.
- If that file is **absent**, `REQUEST_GEN_TOKENS` stays empty → every beam
  decodes freely. **This is what we want.**

Therefore:

1. **Do NOT run `scripts/best_of_n_task_trace.py`** — that script creates the
   `beams={n}` trace files. Skipping it keeps decoding free.
2. **Run both vllm and ours through `scripts/test_time_compute.py`.**
3. Before starting, verify no `beams=*` trace file sits **directly** under the
   MATH-500 dir (old trace files live in the `best_of_n/` subfolder and are
   *not* on the lookup path, so they are harmless):
   ```bash
   ls data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/beams* 2>/dev/null
   ```
   Expect "no such file". If any exist, move them into a `trace_bak/` folder and
   record it in the log.

---

## Step 4 — The sweep

Method labels are auto-derived by `_method_suffix` from `(gemm_opt, chunk_size)`:
`(false, none) → vllm`, `(true, dynamic) → ours`. If you see an output file with
no `_vllm`/`_ours` suffix, the recipe was in the wrong state — fix it and rerun.

### 4a. ours — `n ∈ [2, 4, 8, 16, 32, 64]`

For each `n`, edit `recipes/best-of-n.yaml`: `n: {n}`, `gemm_opt: true`,
`chunk_size: dynamic`, then:

```bash
python scripts/test_time_compute.py recipes/best-of-n.yaml
```

### 4b. vllm — `n ∈ [2, 4, 8]`

For each `n`, edit `recipes/best-of-n.yaml`: `n: {n}`, `gemm_opt: false`,
`chunk_size: none`, then:

```bash
python scripts/test_time_compute.py recipes/best-of-n.yaml
```

**What happens inside each run:** vLLM generates all `n` completions per problem
(free decode), then — because `disable_prm: false` — `load_prm` loads the
Skywork PRM and `best_of_n_beam` calls `prm.score(...)` on **all** `n` beams,
aggregates each beam's per-step scores with `agg_strategy="last"`, and selects
the argmax beam as `pred`. `score()` then adds `pred_weighted@k` / `pred_maj@k`
/ `pred_naive@k` columns. `save_dataset` writes everything to JSONL.

That is 9 runs total (6 ours + 3 vllm).

---

## Step 5 — Expected outputs

Written directly under `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/`:

- `best_of_n_completions_ours_n{2,4,8,16,32,64}.jsonl`
- `best_of_n_completions_vllm_n{2,4,8}.jsonl`
- `best_of_n_request_timings_{ours,vllm}_n{n}.csv`

> These land directly in the `MATH-500/` folder — **not** in the existing
> `MATH-500/best_of_n/` subdirectory, which holds the earlier vllm n16/32/64/128
> runs. Don't confuse the two.

Each JSONL has **500 rows** (one per MATH-500 problem). Relevant keys:
`problem`, `answer` (gold), `completions`, `scores`, `pred` (best-of-n winner),
`pred_weighted@{k}`, `pred_maj@{k}`, `pred_naive@{k}`.

After every run, confirm 500 rows and that `scores` is non-empty (proof the PRM
ran). Record the row count and any failures in the log.

---

## Notes & pitfalls

- **GPU:** RTX 4090, 24 GB, shared by vLLM + the Skywork-1.5B PRM. Keep
  `gpu_memory_utilization ≤ 0.6`.
- **First run is slow:** the first `test_time_compute.py` invocation JIT-compiles
  Triton attention kernels for Qwen2.5-1.5B's head config (~30 s extra).
- **`n=64` is the heaviest** ours run (500 problems × 64 beams × up to 2048
  tokens). If KV-cache pressure causes OOM, lower `gpu_memory_utilization`
  further; `search_batch_size` is already `1`.
- **`chunk_size: dynamic` (ours)** needs the LUT at
  `data/Qwen/QWen2.5-1.5B-Instruct/profile_results_lut.json`. If it is missing,
  the ours runs silently fall back to the FA path — verify the file exists
  before the ours legs and note it in the log (accuracy is unaffected either
  way; only the kernel path differs).
- Kernel optimizations change floating-point results, so `ours` completions will
  not be row-by-row identical to `vllm` — that is expected sampling noise, not a
  bug.
