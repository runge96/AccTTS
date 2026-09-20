Create one file in current folder.

If you run any command or change code, record your action and status (success/fail) in the log file.

If you find problems, you can modify if necessary.

But do remember to record your modification in the log file.

Whenever you modify my code, please git.

Finally, if you find that some test cases always fail, you can skip them and record the information.

Use correct conda env and set it correctly before running experiments.
Go to /search-and-learn
conda activate sal_new_vllm
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0




## Per-#beams sweep: replay both vllm and ours from existing trace files

Dataset is MATH500. Method is beam_search.

Recipe defaults assumed: `gpu_memory_utilization: 0.9` and `max_tokens: 4096`. The 0.9 figure is safe because we no longer load the PRM model — that ~3 GB slot is free. `max_tokens=4096` doubles the prior 2048 cap; on a 24 GB 4090 the worst-case KV at `n=128, max_tokens=4096` is ~14.7 GB (Qwen-2.5-1.5B uses GQA with `num_kv_heads=2`), which fits inside the 0.9 budget.

Trace literals (`beam_search_n{n}`) already exist for `n ∈ {2, 4, 8, 16, 32, 64}` in
`$REPO_ROOT/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/`. For `n=128` you'll need to generate one first via Step 0 below.

### Step 0 (only if the trace is missing for some `n`)

Edit `n` in `recipes/beam-search.yaml` (and `beam_width=2` when `n=2`). Then run

```
python scripts/beam_search_task_trace.py recipes/beam-search.yaml
```

This runs beam_search natively (forces `gemm_opt=false, chunk_size=none` internally), writes the per-step trace + the cumulative per-beam vllm timing CSV. Trace mode does **not** apply the `len(completed_beams) ≥ n` early-stop — the search runs to natural exit so the trace captures every iter the model actually executed.

**Note for `n=128`**: this is the only `n` whose trace doesn't exist yet, so Step 0 must run for it. Because Step 0 also writes `beam_search_request_timings_vllm_n128.csv` as a side effect (the trace itself is the vllm-method experiment), you do **not** need to repeat the vllm replay in Step 1 for `n=128` — the file is already there. Only the ours replay is needed at Step 1 for `n=128`.

### Step 1: replay each `n` for both vllm and ours

For each `n` in [2, 4, 8, 16, 32, 64, 128]:

1. Edit `n` in `$REPO_ROOT/recipes/beam-search.yaml`. When `n=2`, also set `beam_width=2` (the default `beam_width=4` would give `n // beam_width = 0` survivors and the search dies after iter 0). Other `n` values can keep the default.

2. **Replay with vllm settings** *(skip for `n=128`, already produced by Step 0)*. Set `gemm_opt: false`, `chunk_size: none` in the recipe and run
   ```
   python scripts/beam_search_beam.py recipes/beam-search.yaml
   ```
   On startup `beam_search_beam.py` loads the trace literal at `data/{model_path}/MATH-500/beam_search_n{n}` (must exist; see Step 0). It forces per-step gen lengths + completion timing through the vanilla FA decode kernel.

3. **Replay with ours settings** *(needed for every `n`, including `n=128`)*. Set `gemm_opt: true`, `chunk_size: dynamic` in the recipe (keep the same `n`) and run
   ```
   python scripts/beam_search_beam.py recipes/beam-search.yaml
   ```
   Same trace, same forced workload, but the LUT-tuned chunked decode kernel + GEMM autotune are active.

Then move on to the next `n`. Both replays use deterministic survivor selection (first `n//beam_width` candidates), so the search-tree shape is identical across methods. Replay mode applies the `len(completed_beams) ≥ n` early-stop, so the loop exits as soon as `n` beams have graduated.

The output filename suffix (`_vllm` / `_ours` / `_FD` / `_TurnOffGEMM`) is auto-derived from `(gemm_opt, chunk_size)`. If the two flags do not match one of the named combos, `beam_search_beam.py` raises rather than writing un-suffixed files.

After all sweeps you will have, for each `{n}`, in
`$REPO_ROOT/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500`:
- `beam_search_n{n}` — Python literal with `REQUEST_STEP_GEN_TOKENS` and `REQUEST_STEP_COMPLETED` (the trace; consumed by replay)
- `beam_search_n{n}.csv` — columns: problem, step_idx, beam_idx, gen_len (per-step vllm trace, for inspection)
- `beam_search_n{n}_timing.csv` — columns: problem, step_idx, beam_idx, gen_time_ms (per-step vllm trace, for inspection)
- `beam_search_request_timings_vllm_n{n}.csv` — columns: problem, beam, token_length, gen_time_ms (replay-vllm result)
- `beam_search_request_timings_ours_n{n}.csv` — columns: problem, beam, token_length, gen_time_ms (replay-ours result)
- `beam_search_per_step_vllm_n{n}.csv` — per-(problem, iter, slot) raw per-step data of the vllm replay (for re-deriving any aggregation later)
- `beam_search_per_step_ours_n{n}.csv` — per-(problem, iter, slot) raw per-step data of the ours replay
