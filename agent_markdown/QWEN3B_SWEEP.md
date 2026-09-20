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


## Model and constants

New model: `Qwen/QWen2.5-3B-Instruct` (note the capital W — matches the 1.5B
folder convention `QWen2.5-1.5B-Instruct`, the actual HF id is case-insensitive).

For this model:
- Max #beams = **64**. The full sweep grid is `BEAMS = [2, 4, 8, 16, 32, 64]`.
- `max_tokens = 4096` always (already set in both recipe YAMLs).
- All artifacts land under `$REPO_ROOT/data/Qwen/QWen2.5-3B-Instruct/`
  (the directory has been pre-created and is currently empty).

Both `recipes/best-of-n.yaml` and `recipes/beam-search.yaml` already declare
`model_path: Qwen/QWen2.5-3B-Instruct`.


## Step 1 — Collect GEMM templates for the new model

The compile ladder in `scripts/gemm_best_templates_collect.py` currently goes
up to 512. Since this sweep caps `n` at 64, trim the ladder first to avoid
wasting compile time on shapes we never use.

1. Edit [scripts/gemm_best_templates_collect.py:24](scripts/gemm_best_templates_collect.py#L24):

   ```python
   COMPILE_LADDER = [1, 2, 4] + list(range(8, 65, 8))
   # yields: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
   ```

   `git add` + `git commit` the change with a one-line message.

2. Run the collector. The YAML's `gemm_opt`/`chunk_size`/`n`/`num_samples` are
   ignored by this script — it hardcodes the optimized PIECEWISE compile config
   and runs one trivial generation.

   ```
   python scripts/gemm_best_templates_collect.py recipes/best-of-n.yaml
   ```

3. **Verify the cache exists.** vLLM logs the cache directory at startup
   ("Using cache directory: `~/.cache/vllm/torch_compile_cache/<hash>/...`").
   Note that `<hash>` — it's the parent dir keyed on this model + the
   `INDUCTOR_COMPILE_CONFIG` dict. Verify with:

   ```
   ls -la ~/.cache/vllm/torch_compile_cache/
   ```

   Expect a new `<hash>/rank_0_0/` subdirectory whose mtime is from this run.
   The script's stdout should end with `Pre-compile finished for 11 shapes: 1..64.`


## Step 2 — Profile chunk settings + build the LUT

The chunked-decode profiler also needs trimming to stop at n=64.

1. Edit [scripts/chunk_setting_profile.py:62](scripts/chunk_setting_profile.py#L62):

   ```python
   N_BEAMS_LIST = [1, 2, 4] + list(range(8, 65, 8))
   # yields: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
   ```

   `git add` + `git commit` the change.

2. Run the profiler. It writes raw timing data to
   `data/Qwen/QWen2.5-3B-Instruct/chunk_setting_profile_results.csv`.

   ```
   python scripts/chunk_setting_profile.py recipes/best-of-n.yaml
   ```

   The script resumes if interrupted (it parses partial CSVs and restarts from
   the first NaN row), so you can re-launch it cleanly on OOM/error.

3. **Build the LUT from the CSV.** The LUT JSON is produced by
   `scripts/profile_results_analyze.ipynb`, not by the profiler script itself.

   a. Edit the notebook at line 29 to change `MODEL_PATH`:

      ```python
      MODEL_PATH = 'Qwen/QWen2.5-3B-Instruct'
      ```

      `git add` + `git commit` the notebook change.

   b. Run all cells (e.g. `jupyter nbconvert --to notebook --execute scripts/profile_results_analyze.ipynb --inplace`,
      or open in JupyterLab and "Run All").

4. **Verify the LUT files exist** in the correct location:

   ```
   ls $REPO_ROOT/data/Qwen/QWen2.5-3B-Instruct/
   ```

   Expected files:
   - `chunk_setting_profile_results.csv` (from step 2.2)
   - `profile_results_best_method.csv` (from the notebook)
   - `profile_results_lut.json` (from the notebook — **this is the file the bench loads at runtime**)
   - `profile_results_lut.npz` (from the notebook)

   At runtime, `AttnLUT.from_env(model_path=config.model_path)` reads
   `data/Qwen/QWen2.5-3B-Instruct/profile_results_lut.json`. If this file is
   missing, the bench logs "No attention LUT found … using vLLM default
   CHUNK_Size_Page formula" and silently falls back to the heuristic, defeating
   the purpose of the profile — so this check is mandatory before proceeding.


## Step 3 — Best-of-n sweep on MATH-500

Set the dataset to MATH-500 by **commenting out** the dataset lines in
`recipes/best-of-n.yaml` so the Config defaults take over (`HuggingFaceH4/MATH-500`
+ split `test`):

```yaml
# dataset_name: AI-MO/aimo-validation-aime
# dataset_split: train
```

For each `n ∈ [2, 4, 8, 16, 32, 64]`:

1. Edit `n` in `recipes/best-of-n.yaml`.

2. **Trace + vllm replay.** Set `gemm_opt: false`, `chunk_size: none` in the
   recipe and run:

   ```
   python scripts/best_of_n_task_trace.py recipes/best-of-n.yaml
   ```

   The script internally forces `gemm_opt=False, chunk_size="none"` and writes
   both the trace literal and the `_vllm` cumulative timing CSV.

3. **Ours replay.** Set `gemm_opt: true`, `chunk_size: dynamic` (keep the same
   `n`) and run:

   ```
   python scripts/test_time_compute.py recipes/best-of-n.yaml
   ```

   This auto-loads the trace at `data/Qwen/QWen2.5-3B-Instruct/MATH-500/beams={n}`,
   hits the gemm cache from Step 1, hits the LUT from Step 2, and writes the
   `_ours` CSV via `save_dataset`.

After each `n`, expect under `data/Qwen/QWen2.5-3B-Instruct/MATH-500/`:
- `beams={n}`, `beams={n}.csv`, `beams={n}_timing.csv`
- `best_of_n_completions_vllm_n{n}.jsonl`, `best_of_n_request_timings_vllm_n{n}.csv`
- `best_of_n_completions_ours_n{n}.jsonl`, `best_of_n_request_timings_ours_n{n}.csv`


## Step 4 — Beam-search sweep on MATH-500

Set the dataset to MATH-500 in `recipes/beam-search.yaml` by commenting out
the AIME lines:

```yaml
# dataset_name: AI-MO/aimo-validation-aime
# dataset_split: train
```

For each `n ∈ [2, 4, 8, 16, 32, 64]`:

1. Edit `n` in `recipes/beam-search.yaml`. **When `n=2`, also set
   `beam_width: 2`** (default `beam_width: 4` would give `n // beam_width = 0`
   survivors). Other `n` keep `beam_width: 4`.

2. **Trace + vllm replay.** Set `gemm_opt: false`, `chunk_size: none` and run:

   ```
   python scripts/beam_search_task_trace.py recipes/beam-search.yaml
   ```

3. **Ours replay.** Set `gemm_opt: true`, `chunk_size: dynamic` (keep the same
   `n` and matching `beam_width`) and run:

   ```
   python scripts/beam_search_beam.py recipes/beam-search.yaml
   ```

After each `n`, expect under `data/Qwen/QWen2.5-3B-Instruct/MATH-500/`:
- `beam_search_n{n}`, `beam_search_n{n}.csv`, `beam_search_n{n}_timing.csv`
- `beam_search_request_timings_vllm_n{n}.csv`
- `beam_search_request_timings_ours_n{n}.csv`
- `beam_search_per_step_vllm_n{n}.csv`, `beam_search_per_step_ours_n{n}.csv`


## Step 5 — Best-of-n sweep on AIME

Switch the dataset to AIME by **uncommenting** the AIME lines in
`recipes/best-of-n.yaml`:

```yaml
dataset_name: AI-MO/aimo-validation-aime
dataset_split: train
```

Then run the same per-`n` flow as Step 3 (same `n` grid `[2, 4, 8, 16, 32, 64]`,
same trace/replay commands). Outputs land under
`data/Qwen/QWen2.5-3B-Instruct/aimo-validation-aime/` with the same filename
schema as Step 3.


## Step 6 — Beam-search sweep on AIME

Switch the dataset to AIME by uncommenting (or re-adding) the AIME lines in
`recipes/beam-search.yaml`:

```yaml
dataset_name: AI-MO/aimo-validation-aime
dataset_split: train
```

Then run the same per-`n` flow as Step 4 (same `n` grid, same `beam_width: 2`
rule for `n=2`, same trace/replay commands). Outputs land under
`data/Qwen/QWen2.5-3B-Instruct/aimo-validation-aime/`.


## Notes

- **Method labels are auto-derived from `(gemm_opt, chunk_size)` by `_method_suffix`**:
  `(False, "none") → vllm`, `(True, "dynamic") → ours`. Any other combination
  writes outputs without a suffix and logs a warning — if you see un-suffixed
  files, the recipe was in a wrong state when you ran the script.
- The gemm cache (Step 1) is only consulted when `gemm_opt: true`; the "vllm"
  replays bypass it and run on vLLM defaults. That's expected — the vllm
  baseline must NOT use the optimized compile path.
- The LUT (Step 2) is only consulted when `chunk_size: dynamic`. The vllm
  baseline (`chunk_size: none`) uses the FA path via vLLM's default formula.
- Per-`n` YAML edits during sweeps are configuration changes, not code
  modifications, so they don't need git commits. The script edits in Steps 1
  and 2, and the notebook edit in Step 2, DO need commits.
- If a particular `(n, problem)` keeps OOM'ing or hanging, skip it and record
  the problem index + `n` in the log file rather than blocking the rest of the
  sweep.
- If `num_samples` is set in either YAML (e.g. `num_samples: 1` for debug),
  remove or comment it before running the actual sweep — otherwise only one
  problem will be timed.
