Create one log file in current folder.

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


## Dataset and recipes

Dataset is AIME (`AI-MO/aimo-validation-aime`, split `train`, 90 problems). Both
`recipes/best-of-n.yaml` and `recipes/beam-search.yaml` already declare
`dataset_name` and `dataset_split` for this dataset, so no YAML edit is needed
for the dataset itself.

Outputs land under
`$REPO_ROOT/data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/`
(driven by `model_path`/`dataset_name` in the recipes). The dataset's
short-name segment is `aimo-validation-aime`.

Sweep `n ∈ [2, 4, 8, 16, 32, 64, 128]` for both methods. Order: best_of_n
first (all `n`), then beam_search (all `n`).

Recipe defaults assumed: `gpu_memory_utilization: 0.9`, `max_tokens: 4096`.
PRM is not loaded in either flow (best-of-n has `disable_prm: true` in the
YAML; beam_search's trace+replay scripts never call `load_prm`), so the 0.9
budget leaves headroom for KV.


## Part A — best_of_n sweep

For each `n ∈ [2, 4, 8, 16, 32, 64, 128]`:

1. Edit `n` in `recipes/best-of-n.yaml`. No `beam_width` exists for best_of_n,
   so no other field needs changing for the sweep.

2. **Trace + vllm replay** (one command does both).
   Set `gemm_opt: false`, `chunk_size: none` in the recipe and run

   ```
   python scripts/best_of_n_task_trace.py recipes/best-of-n.yaml
   ```

   `best_of_n_task_trace.py` internally forces `gemm_opt=False,
   chunk_size="none"` regardless of the YAML, then writes both the per-(problem,
   beam) trace and (because the trace run *is* the vllm-method experiment) the
   `_vllm` cumulative timing CSV via `save_dataset`.

3. **Ours replay** (uses the trace from step 2).
   Set `gemm_opt: true`, `chunk_size: dynamic` in the recipe (keep the same
   `n`) and run

   ```
   python scripts/test_time_compute.py recipes/best-of-n.yaml
   ```

   `test_time_compute.py` auto-loads the trace literal at
   `data/{model_path}/aimo-validation-aime/beams={n}` (forcing per-beam token
   lengths so the workload matches the vllm run exactly) and runs best_of_n
   through the LUT-tuned chunked decode kernel + GEMM autotune. `save_dataset`
   sees `(gemm_opt=True, chunk_size="dynamic")` and writes the `_ours` CSV.

After each `n` you should have, in
`data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/`:

- `beams={n}` — Python literal `REQUEST_GEN_TOKENS = {(problem_idx, beam_idx): tokens}` (the trace; consumed by replay)
- `beams={n}.csv` — columns: problem_idx, beam_idx, gen_len (per-beam trace, for inspection)
- `beams={n}_timing.csv` — columns: problem_idx, beam_idx, gen_time_ms (per-beam trace, for inspection)
- `best_of_n_completions_vllm_n{n}.jsonl` — full dataset rows from the trace run
- `best_of_n_request_timings_vllm_n{n}.csv` — columns: problem, beam, token_length, gen_time_ms (vllm replay result)
- `best_of_n_completions_ours_n{n}.jsonl` — full dataset rows from the ours replay
- `best_of_n_request_timings_ours_n{n}.csv` — columns: problem, beam, token_length, gen_time_ms (ours replay result)


## Part B — beam_search sweep

For each `n ∈ [2, 4, 8, 16, 32, 64, 128]`:

1. Edit `n` in `recipes/beam-search.yaml`. **When `n=2`, also set
   `beam_width: 2`** (the default `beam_width: 4` would give
   `n // beam_width = 0` survivors and the search dies after iter 0). Other
   `n` values can keep the default `beam_width: 4`.

2. **Trace + vllm replay** (one command does both).
   Set `gemm_opt: false`, `chunk_size: none` in the recipe and run

   ```
   python scripts/beam_search_task_trace.py recipes/beam-search.yaml
   ```

   The script internally forces `gemm_opt=False, chunk_size="none"` and writes
   the per-(problem, step, beam) trace plus the cumulative per-beam
   `_vllm` timing CSV. Trace mode does **not** apply the
   `len(completed_beams) ≥ n` early-stop — the search runs to natural exit so
   the trace captures every iteration the model actually executed.

3. **Ours replay** (uses the trace from step 2).
   Set `gemm_opt: true`, `chunk_size: dynamic` in the recipe (keep the same
   `n` and matching `beam_width`) and run

   ```
   python scripts/beam_search_beam.py recipes/beam-search.yaml
   ```

   `beam_search_beam.py` loads the trace literal at
   `data/{model_path}/aimo-validation-aime/beam_search_n{n}` (must exist; see
   step 2), forces per-step gen lengths + completion timing, and runs through
   the LUT-tuned chunked decode kernel + GEMM autotune. Replay mode *does*
   apply the `len(completed_beams) ≥ n` early-stop, so the loop exits as soon
   as `n` beams have graduated. Output suffix (`_vllm` / `_ours` / `_FD` /
   `_TurnOffGEMM`) is auto-derived from `(gemm_opt, chunk_size)`; the script
   raises if the combo is unnamed.

After each `n` you should have, in
`data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/`:

- `beam_search_n{n}` — Python literal with `REQUEST_STEP_GEN_TOKENS` and `REQUEST_STEP_COMPLETED`
- `beam_search_n{n}.csv` — columns: problem, step_idx, beam_idx, gen_len
- `beam_search_n{n}_timing.csv` — columns: problem, step_idx, beam_idx, gen_time_ms
- `beam_search_request_timings_vllm_n{n}.csv` — columns: problem, beam, token_length, gen_time_ms (vllm replay result, written by the trace script)
- `beam_search_request_timings_ours_n{n}.csv` — columns: problem, beam, token_length, gen_time_ms (ours replay result)
- `beam_search_per_step_vllm_n{n}.csv` — per-(problem, iter, slot) raw per-step data of the vllm replay
- `beam_search_per_step_ours_n{n}.csv` — per-(problem, iter, slot) raw per-step data of the ours replay


## Notes

- The two replays share the same trace per `n`, so the per-beam workload is
  identical between `_vllm` and `_ours`. The only difference is the attention
  kernel + GEMM autotune behavior.
- If a particular (n, problem) keeps OOM'ing or hanging, skip it and record
  the problem index and the `n` in the log file rather than blocking the rest
  of the sweep.
- If you change any code, `git add` + `git commit` with a one-line message
  describing the change. Don't squash unrelated edits.
