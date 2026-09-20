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
export HF_TOKEN=YOUR_HUGGING_FACE_TOKEN   # Llama 3.2 is gated; the token must be on an
                               # HF account that has accepted the license at
                               # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct


## Model and constants

New model: `meta-llama/Llama-3.2-1B-Instruct`. Unlike the Qwen folders this is
the exact HF id (no case mangling).

For this model:
- Max #beams = **64**. The full sweep grid is `BEAMS = [2, 4, 8, 16, 32, 64]`.
- `max_tokens = 4096` always (already set in both recipe YAMLs).
- All artifacts land under `$REPO_ROOT/data/meta-llama/Llama-3.2-1B-Instruct/`.

Both `recipes/best-of-n.yaml` and `recipes/beam-search.yaml` already declare
`model_path: meta-llama/Llama-3.2-1B-Instruct`.


## Step 1 — Collect GEMM templates for the new model

The compile ladder in `scripts/gemm_best_templates_collect.py` should already
be trimmed to `[1, 2, 4] + list(range(8, 65, 8))` from the Qwen-3B sweep. If
not, set it now and commit.

1. Verify [scripts/gemm_best_templates_collect.py:24](scripts/gemm_best_templates_collect.py#L24):

   ```python
   COMPILE_LADDER = [1, 2, 4] + list(range(8, 65, 8))
   # yields: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
   ```

   `git add` + `git commit` only if you had to change it.

2. Run the collector. The YAML's `gemm_opt`/`chunk_size`/`n`/`num_samples` are
   ignored by this script — it hardcodes the optimized PIECEWISE compile config
   and runs one trivial generation.

   ```
   python scripts/gemm_best_templates_collect.py recipes/best-of-n.yaml
   ```

3. **Verify the cache exists.** vLLM logs the cache directory at startup
   ("Using cache directory: `~/.cache/vllm/torch_compile_cache/<hash>/...`").
   The `<hash>` is keyed on (model + INDUCTOR_COMPILE_CONFIG) so it will be a
   **different hash than the Qwen-3B run** — note it down. Verify with:

   ```
   ls -la ~/.cache/vllm/torch_compile_cache/
   ```

   Expect a new `<hash>/rank_0_0/` subdirectory whose mtime is from this run.
   The script's stdout should end with `Pre-compile finished for 11 shapes: 1..64.`


## Step 2 — Chunk-setting profile + LUT (ALREADY DONE)

This step is **skipped**. The chunk_setting profile was already collected and
the LUT was already built for Llama-3.2-1B-Instruct. Verify the four expected
files exist before proceeding — `AttnLUT.from_env` reads
`profile_results_lut.json` at runtime, and missing files would silently fall
back to vLLM's default chunk formula.

```
ls $REPO_ROOT/data/meta-llama/Llama-3.2-1B-Instruct/
```

Expected files (all four must be present):
- `chunk_setting_profile_results.csv`
- `profile_results_best_method.csv`
- `profile_results_lut.json`     ← **mandatory** for `chunk_size: dynamic`
- `profile_results_lut.npz`

If any are missing, run the Qwen-3B Step 2 procedure from `QWEN3B_SWEEP.md`
against this model first (edit the notebook's `MODEL_PATH` to
`'meta-llama/Llama-3.2-1B-Instruct'` and re-execute).


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

   This auto-loads the trace at
   `data/meta-llama/Llama-3.2-1B-Instruct/MATH-500/beams={n}`, hits the gemm
   cache from Step 1, hits the LUT from the already-done Step 2, and writes
   the `_ours` CSV via `save_dataset`.

After each `n`, expect under `data/meta-llama/Llama-3.2-1B-Instruct/MATH-500/`:
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

After each `n`, expect under `data/meta-llama/Llama-3.2-1B-Instruct/MATH-500/`:
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
`data/meta-llama/Llama-3.2-1B-Instruct/aimo-validation-aime/` with the same
filename schema as Step 3.


## Step 6 — Beam-search sweep on AIME

Switch the dataset to AIME by uncommenting (or re-adding) the AIME lines in
`recipes/beam-search.yaml`:

```yaml
dataset_name: AI-MO/aimo-validation-aime
dataset_split: train
```

Then run the same per-`n` flow as Step 4 (same `n` grid, same `beam_width: 2`
rule for `n=2`, same trace/replay commands). Outputs land under
`data/meta-llama/Llama-3.2-1B-Instruct/aimo-validation-aime/`.


## Notes

- **Method labels are auto-derived from `(gemm_opt, chunk_size)` by `_method_suffix`**:
  `(False, "none") → vllm`, `(True, "dynamic") → ours`. Any other combination
  writes outputs without a suffix and logs a warning — if you see un-suffixed
  files, the recipe was in a wrong state when you ran the script.
- The gemm cache (Step 1) is only consulted when `gemm_opt: true`; the "vllm"
  replays bypass it and run on vLLM defaults. That's expected — the vllm
  baseline must NOT use the optimized compile path. The Llama gemm cache lives
  under a different `<hash>` than the Qwen-3B cache; both can coexist.
- The LUT (Step 2, already done) is only consulted when `chunk_size: dynamic`.
  The vllm baseline (`chunk_size: none`) uses the FA path via vLLM's default
  formula.
- Per-`n` YAML edits during sweeps are configuration changes, not code
  modifications, so they don't need git commits. The model_path change in
  `recipes/beam-search.yaml` (Qwen → Llama), and any script edits in Step 1,
  DO need commits.
- Llama-3.2-1B has 8 KV heads / 4 queries-per-KV / head_dim=64 (vs Qwen-3B's
  2 / 8 / 128). The Triton attention kernels will JIT-compile a new
  specialization for these constexprs on first call — expect the first
  trace/replay invocation per script to take ~30 s longer than steady state.
- If a particular `(n, problem)` keeps OOM'ing or hanging, skip it and record
  the problem index + `n` in the log file rather than blocking the rest of the
  sweep. Llama-1B is ~3× smaller than Qwen-3B so OOM is unlikely, but the
  policy is the same.
- If `num_samples` is set in either YAML (e.g. `num_samples: 1` for debug),
  remove or comment it before running the actual sweep — otherwise only one
  problem will be timed.
