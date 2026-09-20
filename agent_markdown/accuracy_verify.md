# accuracy_verify.md

**Task for Claude Code:** run the **best-of-n** test-time-compute method on the
**AIME** dataset with the **Llama-3.2-1B-Instruct** reasoning model, score every
beam with the PRM, save the results as JSON, and compare the **vllm** baseline
against **ours** at beam counts `n ∈ [2, 4, 8, 16, 32, 64]`.

## Working rules

1. Create one log file, `logs/accuracy_verify.log`, in the repo.
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

Run these first (the `export` block, including the gated-Llama `HF_TOKEN`, is
copied verbatim from `LLAMA1B_SWEEP.md` lines 14-21):

```bash
cd $REPO_ROOT
conda activate sal_new_vllm
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_TOKEN=YOUR_HUGGING_FACE_TOKEN   # must be on an HF account
                                                        # that accepted the
                                                        # Llama-3.2-1B license
```

---

## Step 1 — Configure `recipes/best-of-n.yaml` (once, before the sweep)

Edit these fields. Three of them are **mandatory changes** from the file's
current state:

| field | value | note |
|---|---|---|
| `dataset_name` | `AI-MO/aimo-validation-aime` | AIME — already set |
| `dataset_split` | `train` | AIME's only split — already set |
| `model_path` | `meta-llama/Llama-3.2-1B-Instruct` | reasoning model — already set |
| `prm_path` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | the PRM — already set |
| `max_tokens` | **`3072`** | **CHANGE** (currently 4096) |
| `disable_prm` | **`false`** | **CHANGE** (currently `true`). Without this the PRM is never loaded and beams are not scored. |
| `gpu_memory_utilization` | **`0.6`** | **CHANGE** (currently 0.9). With `disable_prm: false` the Skywork-1.5B PRM now shares the 24 GB RTX 4090 with vLLM; 0.9 leaves too little and the PRM load will OOM. Drop to `0.5` if it still OOMs. |
| `n` | per-run (see Step 3) | |
| `gemm_opt` | per-method (see Step 3) | |
| `chunk_size` | per-method (see Step 3) | |
| `num_samples` | keep commented out | runs all 90 AIME problems |

Leave `agg_strategy` unset — it defaults to `"last"` (best-of-n picks the beam
with the highest last-step PRM score).

Confirm the `# num_samples:` line is still commented before each run.

---

## Step 2 — The crucial rule: free decoding, NO trace forcing

Every beam must generate freely and stop on its **own** criteria (EOS token or
`max_tokens`), for **both** vllm and ours. Do **not** force per-beam token
counts from a trace file.

How the trace mechanism works (`src/sal/search/best_of_n_beam.py`):

- `scripts/test_time_compute.py` calls `load_trace_for_config(config)`, which
  reads `data/meta-llama/Llama-3.2-1B-Instruct/aimo-validation-aime/beams={n}`.
- If that file **exists**, every beam's `min_tokens == max_tokens` is forced to
  the recorded length (replay mode) — **not what we want**.
- If that file is **absent**, `REQUEST_GEN_TOKENS` stays empty → every beam
  decodes freely. **This is what we want.**

Therefore:

1. **Do NOT run `scripts/best_of_n_task_trace.py`** — that script is what
   creates the `beams={n}` trace files. Skipping it keeps decoding free.
2. **Run both vllm and ours through `scripts/test_time_compute.py`.** This
   deviates from `LLAMA1B_SWEEP.md` Step 3, which used the trace script for the
   vllm leg — here the vllm leg must free-decode too.
3. Before starting, verify no `beams=*` trace file exists in the output dir:
   ```bash
   ls data/meta-llama/Llama-3.2-1B-Instruct/aimo-validation-aime/beams* 2>/dev/null
   ```
   Expect "no such file". If any exist, move them into a `trace_bak/` folder and
   record it in the log.

---

## Step 3 — The sweep

For each `n ∈ [2, 4, 8, 16, 32, 64]`, run two configs.

**vllm baseline** — in `recipes/best-of-n.yaml` set `n: {n}`, `gemm_opt: false`,
`chunk_size: none`, then:

```bash
python scripts/test_time_compute.py recipes/best-of-n.yaml
```

**ours** — keep `n: {n}`, set `gemm_opt: true`, `chunk_size: dynamic`, then:

```bash
python scripts/test_time_compute.py recipes/best-of-n.yaml
```

Method labels are auto-derived by `_method_suffix` from `(gemm_opt, chunk_size)`:
`(false, none) → vllm`, `(true, dynamic) → ours`. If you see an output file with
no `_vllm`/`_ours` suffix, the recipe was in the wrong state — fix it and rerun.

**What happens inside each run:** vLLM generates all `n` completions per problem
(free decode), then — because `disable_prm: false` — `load_prm` loads the
Skywork PRM and `best_of_n_beam` calls `prm.score(...)` on **all** `n` beams,
aggregates each beam's per-step scores with `agg_strategy="last"`, and selects
the argmax beam as `pred`. `score()` then adds `pred_weighted@k` / `pred_maj@k`
/ `pred_naive@k` columns. `save_dataset` writes everything to JSONL.

---

## Step 4 — Expected outputs

Per `n`, written directly under
`data/meta-llama/Llama-3.2-1B-Instruct/aimo-validation-aime/`:

- `best_of_n_completions_vllm_n{n}.jsonl`
- `best_of_n_completions_ours_n{n}.jsonl`
- `best_of_n_request_timings_{vllm,ours}_n{n}.csv`

> These land directly in the `aimo-validation-aime/` folder — **not** in the
> existing `best_of_n/` subdirectory, which holds older data. Don't confuse them.

Each JSONL has **90 rows** (one per AIME problem). Relevant keys: `problem`,
`answer` (gold), `completions`, `scores`, `pred` (best-of-n winner),
`pred_weighted@{k}`, `pred_maj@{k}`, `pred_naive@{k}`.

After every run, confirm 90 rows and that `scores` is non-empty (proof the PRM
ran). Record the row count and any failures in the log.

---

## Step 5 — Accuracy comparison (vllm vs ours)

Grade each JSONL: extract the answer from a prediction with `extract_answer`,
normalize both the prediction and the gold `answer` to a sympy canonical form,
and compare. Run this from the repo root in the `sal_new_vllm` env:

```python
import sys, os, json, glob, re, csv
sys.path.insert(0, "src")
from sal.utils.qwen_math_parser import extract_answer
from sal.utils.math import memoized_canonical_form as canon

AIME_DIR = "data/meta-llama/Llama-3.2-1B-Instruct/aimo-validation-aime"
os.makedirs("figures", exist_ok=True)


def correct(pred_text, gold_canon):
    if not pred_text:
        return False
    a = extract_answer(str(pred_text), "math")
    return bool(a) and canon(a) == gold_canon


rows = []
for path in sorted(glob.glob(f"{AIME_DIR}/best_of_n_completions_*_n*.jsonl")):
    m = re.search(r"_(vllm|ours)_n(\d+)\.jsonl$", path)
    if not m:
        continue
    method, n = m.group(1), int(m.group(2))
    data = [json.loads(l) for l in open(path)]
    gold = [canon(d["answer"]) for d in data]
    N = len(data)
    bon = sum(correct(d["pred"], g) for d, g in zip(data, gold))
    wt = sum(correct(d.get(f"pred_weighted@{n}"), g) for d, g in zip(data, gold))
    mj = sum(correct(d.get(f"pred_maj@{n}"), g) for d, g in zip(data, gold))
    rows.append((method, n, N, bon, wt, mj))
    print(f"{method:5s} n={n:<3d}  best-of-n={bon}/{N}  "
          f"weighted={wt}/{N}  majority={mj}/{N}")

rows.sort(key=lambda r: (r[1], r[0]))
with open("figures/accuracy_verify_aime.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "n", "problems", "best_of_n", "weighted", "majority"])
    for method, n, N, bon, wt, mj in rows:
        w.writerow([method, n, N, bon / N, wt / N, mj / N])
print("saved figures/accuracy_verify_aime.csv")
```

Put the resulting vllm-vs-ours table into `logs/accuracy_verify.log`.

---

## Notes & pitfalls

- **GPU:** RTX 4090, 24 GB, shared by vLLM + the Skywork-1.5B PRM. Keep
  `gpu_memory_utilization ≤ 0.6`; the recipe's old `0.9` was for PRM-disabled
  timing runs only.
- **AIME is hard and small** — only 90 problems, and a 1B model will solve very
  few (expect single-digit %). With so few correct answers, the vllm-vs-ours gap
  is dominated by sampling noise. The correctness criterion is **"ours tracks
  vllm within noise"**, not exact equality: kernel optimizations change
  floating-point results, so completions will *not* be row-by-row identical.
- **First run is slow:** the first `test_time_compute.py` invocation JIT-compiles
  Triton attention kernels for Llama-1B's head config (~30 s extra).
- **`n=64` is the heaviest** run (90 × 64 beams × up to 3072 tokens). If KV-cache
  pressure causes OOM, lower `gpu_memory_utilization` further; `search_batch_size`
  is already `1`.
- **`chunk_size: dynamic` (ours)** needs the LUT at
  `data/meta-llama/Llama-3.2-1B-Instruct/profile_results_lut.json` (built in
  `LLAMA1B_SWEEP.md` Step 2). If it is missing, the ours runs silently fall back
  to the FA path — verify the file exists before the ours legs and note it in
  the log.
