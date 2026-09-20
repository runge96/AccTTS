# Ablation Study — `onlyGEMM` variant (best_of_n, QWen2.5-1.5B)

Create one log file in the current folder (`ablation_study_LOG.md`).

If you run any command or change code, record your action and status
(success/fail) in the log file.

If you find problems, you can modify if necessary. But do remember to record
your modification in the log file.

Whenever you modify my code, please `git add` + `git commit` with a one-line
message describing the change. Don't squash unrelated edits.

Finally, if you find that some test cases always fail, you can skip them and
record the information (dataset, `n`, problem index if applicable) in the log.


## Environment

Use the correct conda env and set it before running experiments.

```
cd $REPO_ROOT
conda activate sal_new_vllm
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```


## What this ablation runs

- **Model:** `Qwen/QWen2.5-1.5B-Instruct`
- **Algorithm:** `best_of_n`
- **Method variant:** `onlyGEMM` only — `gemm_opt: true`, `chunk_size: none`
  (GEMM autotune on, vanilla FA attention, no LUT). `save_dataset` derives the
  `onlyGEMM` file suffix from this `(gemm_opt, chunk_size)` pair via
  `_method_suffix()` in `src/sal/utils/data.py`.
- **Datasets:** MATH-500 first, then AIME.
- **Beam counts:** `n ∈ [2, 4, 8, 16, 32, 64, 128]`.
- **PRM:** not used (`disable_prm: true`).
- **GPU memory:** `gpu_memory_utilization: 0.9`.
- **force_beam_gen: true** — per-beam generation lengths are forced from the
  pre-recorded trace files (see next section). No sampling-driven length
  variance; the workload is fixed.


## force_beam_gen — trace files

`best_of_n` runs through `scripts/test_time_compute.py`, which auto-loads the
per-`(model, dataset, n)` trace literal at
`data/{model_path}/{dataset_short}/beams={n}` and forces `min==max` tokens for
each `(problem, beam)`. This *is* the "force_beam_gen" mechanism — there is no
YAML key; forcing is automatic whenever the trace file exists.

The traces already exist for every `n` in the sweep:

- MATH-500: `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/beams={2,4,8,16,32,64,128}`
- AIME:     `data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/beams={2,4,8,16,32,64,128}`

**Before each run, confirm the matching `beams={n}` trace file exists.** If a
trace file is missing, do NOT run free-decode — stop and record it in the log
(generating a trace is out of scope for this ablation).


## Recipe — common settings

Edit `recipes/best-of-n.yaml` so these fields hold for the whole sweep:

```yaml
model_path: Qwen/QWen2.5-1.5B-Instruct
approach: best_of_n
gpu_memory_utilization: 0.9
disable_prm: true
gemm_opt: true
chunk_size: none
```

`n` and the dataset fields change per phase below.


## Part 1 — MATH-500

1. Set the system prompt preset. In `src/sal/config.py`, the single line
   ```
   _SYSTEM_PROMPT_PRESET = "MATH500"
   ```
   must read `"MATH500"`. If you change it, `git commit` the edit.

2. In `recipes/best-of-n.yaml` set:
   ```yaml
   dataset_name: HuggingFaceH4/MATH-500
   dataset_split: test
   max_tokens: 2048
   ```

3. For each `n ∈ [2, 4, 8, 16, 32, 64, 128]`:
   - Edit `n` in `recipes/best-of-n.yaml`.
   - Confirm `data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/beams={n}` exists.
   - Run:
     ```
     python scripts/test_time_compute.py recipes/best-of-n.yaml
     ```
   - Record the command and success/fail in the log.

After each `n`, expect in
`data/Qwen/QWen2.5-1.5B-Instruct/MATH-500/`:

- `best_of_n_completions_onlyGEMM_n{n}.jsonl` — full dataset rows
- `best_of_n_request_timings_onlyGEMM_n{n}.csv` — columns:
  `problem, beam, token_length, gen_time_ms`


## Part 2 — AIME

1. Set the system prompt preset. In `src/sal/config.py`, change the single line
   to:
   ```
   _SYSTEM_PROMPT_PRESET = "AIME"
   ```
   Then `git commit` the edit. This swaps `Config.system_prompt` to the long
   systematic-thinking prompt AIME needs.

2. In `recipes/best-of-n.yaml` set:
   ```yaml
   dataset_name: AI-MO/aimo-validation-aime
   dataset_split: train
   max_tokens: 4096
   ```

3. For each `n ∈ [2, 4, 8, 16, 32, 64, 128]`:
   - Edit `n` in `recipes/best-of-n.yaml`.
   - Confirm
     `data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/beams={n}` exists.
   - Run:
     ```
     python scripts/test_time_compute.py recipes/best-of-n.yaml
     ```
   - Record the command and success/fail in the log.

After each `n`, expect in
`data/Qwen/QWen2.5-1.5B-Instruct/aimo-validation-aime/`:

- `best_of_n_completions_onlyGEMM_n{n}.jsonl`
- `best_of_n_request_timings_onlyGEMM_n{n}.csv`

4. When the AIME phase is done, restore `_SYSTEM_PROMPT_PRESET = "MATH500"` in
   `src/sal/config.py` (and `git commit`) so the repo default is unchanged.


## Notes

- The `onlyGEMM` suffix only appears if `_method_suffix()` recognizes
  `(gemm_opt=true, chunk_size=none)`. If `save_dataset` logs a "does not match
  any of the named methods" warning, the `_method_suffix()` branch for
  `onlyGEMM` is missing — add it and `git commit`.
- The first run after changing `gemm_opt`/`n` pays a one-time inductor
  GEMM-autotune compile cost (compile_sizes ladder derived from `n`). That is
  expected; only steady-state timing matters.
- If a particular `(n, problem)` keeps OOM'ing or hanging, skip it and record
  the `n` and problem index in the log rather than blocking the rest of the
  sweep.
- Always finish a phase before switching `_SYSTEM_PROMPT_PRESET` — MATH-500 and
  AIME need different prompts and the wrong one silently degrades accuracy.
