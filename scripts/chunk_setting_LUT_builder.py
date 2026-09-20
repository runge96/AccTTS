#!/usr/bin/env python
"""Build runtime attention LUT files from chunk_setting_profile results.

Usage:
    python scripts/chunk_setting_LUT_builder.py recipes/best-of-n.yaml

The model path is read from the YAML config. The script expects:

    data/{model_path}/chunk_setting_profile_results.csv

and writes:

    data/{model_path}/profile_results_best_method.csv
    data/{model_path}/profile_results_lut.json
    data/{model_path}/profile_results_lut.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sal.config import Config
from sal.utils.parser import H4ArgumentParser

DEFAULT_TIE_THRESHOLD_MS = 2.0
DEFAULT_BLOCK_SIZE = 16
DEFAULT_N_HEADS_ASSUMED = 2


def load_config(yaml_path: Path) -> Config:
    parser = H4ArgumentParser(Config)
    output = parser.parse_yaml_file(os.path.abspath(yaml_path))
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def label_to_csp(label) -> int:
    """0 -> FA; N -> FD-N."""
    if not isinstance(label, str) or label == "FA":
        return 0
    if label.startswith("FD-"):
        return int(label.split("-", 1)[1])
    return 0


def load_profile_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing profile CSV: {csv_path}\n"
            "Run `python scripts/chunk_setting_profile.py <recipe.yaml>` first."
        )

    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    required = {"n_beams", "prompt_len", "variant", "time_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df[["n_beams", "prompt_len", "variant", "time_ms"]].copy()
    df = df.dropna(subset=["n_beams", "prompt_len", "variant"])
    df["n_beams"] = df["n_beams"].astype(int)
    df["prompt_len"] = df["prompt_len"].astype(int)
    df["variant"] = df["variant"].astype(str)
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    return df


def build_winner_table(df: pd.DataFrame, tie_threshold_ms: float) -> pd.DataFrame:
    # If a combo was measured more than once, keep the fastest valid time.
    fa = (
        df[df["variant"] == "FA"]
        .dropna(subset=["time_ms"])
        .groupby(["n_beams", "prompt_len"], as_index=False)["time_ms"]
        .min()
        .rename(columns={"time_ms": "fa_time_ms"})
    )

    fd = df[df["variant"].str.startswith("FD-")].dropna(subset=["time_ms"]).copy()
    if fd.empty:
        fd_best = pd.DataFrame(
            columns=["n_beams", "prompt_len", "best_fd_variant", "best_fd_time_ms"]
        )
    else:
        fd_best_idx = fd.groupby(["n_beams", "prompt_len"])["time_ms"].idxmin()
        fd_best = fd.loc[fd_best_idx].rename(
            columns={"variant": "best_fd_variant", "time_ms": "best_fd_time_ms"}
        )[["n_beams", "prompt_len", "best_fd_variant", "best_fd_time_ms"]]

    merged = fa.merge(fd_best, on=["n_beams", "prompt_len"], how="outer")

    def pick_winner(row):
        fa_t = row.get("fa_time_ms")
        fd_t = row.get("best_fd_time_ms")
        fd_v = row.get("best_fd_variant")
        if pd.isna(fd_t):
            return "FA"
        if pd.isna(fa_t):
            return fd_v
        diff = fa_t - fd_t  # positive means FD is faster.
        return fd_v if diff >= tie_threshold_ms else "FA"

    merged["winner"] = merged.apply(pick_winner, axis=1)
    winner_table = merged.pivot(
        index="n_beams", columns="prompt_len", values="winner"
    ).sort_index()
    winner_table = winner_table.reindex(sorted(winner_table.columns), axis=1)
    return winner_table


def build_csp_grid(winner_table: pd.DataFrame):
    n_beams_axis = sorted(int(v) for v in winner_table.index.tolist())
    prompt_len_axis = sorted(int(v) for v in winner_table.columns.tolist())

    csp_grid = np.zeros((len(n_beams_axis), len(prompt_len_axis)), dtype=np.int16)
    for i, b in enumerate(n_beams_axis):
        last_good = 0
        for j, p in enumerate(prompt_len_axis):
            label = winner_table.loc[b, p]
            if pd.isna(label):
                csp_grid[i, j] = last_good
            else:
                csp_grid[i, j] = label_to_csp(label)
                last_good = int(csp_grid[i, j])

    return n_beams_axis, prompt_len_axis, csp_grid


def write_lut_files(
    data_dir,
    winner_table,
    n_beams_axis,
    prompt_len_axis,
    csp_grid,
    tie_threshold_ms,
    block_size,
    n_heads_assumed,
) -> None:
    best_method_path = data_dir / "profile_results_best_method.csv"
    winner_table.to_csv(best_method_path)

    lut_payload = {
        "description": (
            "Best CHUNK_Size_Page per (n_beams, prompt_len) for vLLM "
            "chunked-prefill paged decode. 0 = FA "
            "(PROFILE_CHUNK_SIZE_PAGE = None); N = FD-N."
        ),
        "tie_threshold_ms": tie_threshold_ms,
        "block_size": block_size,
        "n_heads_assumed": n_heads_assumed,
        "n_beams_axis": [int(b) for b in n_beams_axis],
        "prompt_len_axis": [int(p) for p in prompt_len_axis],
        "csp_grid": csp_grid.astype(int).tolist(),
    }

    json_path = data_dir / "profile_results_lut.json"
    json_path.write_text(json.dumps(lut_payload, indent=2))

    npz_path = data_dir / "profile_results_lut.npz"
    np.savez(
        npz_path,
        n_beams_axis=np.asarray(n_beams_axis, dtype=np.int32),
        prompt_len_axis=np.asarray(prompt_len_axis, dtype=np.int32),
        csp_grid=csp_grid.astype(np.int16),
        tie_threshold_ms=np.float32(tie_threshold_ms),
        block_size=np.int32(block_size),
        n_heads_assumed=np.int32(n_heads_assumed),
    )

    print(f"Wrote {best_method_path} ({best_method_path.stat().st_size} bytes)")
    print(f"Wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(f"Wrote {npz_path} ({npz_path.stat().st_size} bytes)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        nargs="?",
        default="recipes/best-of-n.yaml",
        help="YAML recipe to read model_path from.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional explicit chunk_setting_profile_results.csv path.",
    )
    parser.add_argument(
        "--tie-threshold-ms",
        type=float,
        default=DEFAULT_TIE_THRESHOLD_MS,
        help="FD must beat FA by at least this many ms to win.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Block size metadata to store in the LUT.",
    )
    parser.add_argument(
        "--n-heads-assumed",
        type=int,
        default=DEFAULT_N_HEADS_ASSUMED,
        help="Metadata only; matches the original notebook export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe_path = Path(args.recipe)
    if not recipe_path.is_absolute():
        recipe_path = REPO_ROOT / recipe_path

    config = load_config(recipe_path)
    data_dir = REPO_ROOT / "data" / config.model_path
    csv_path = args.csv or (data_dir / "chunk_setting_profile_results.csv")

    print(f"recipe: {recipe_path}")
    print(f"model : {config.model_path}")
    print(f"csv   : {csv_path}")

    df = load_profile_csv(csv_path)
    winner_table = build_winner_table(df, args.tie_threshold_ms)
    n_beams_axis, prompt_len_axis, csp_grid = build_csp_grid(winner_table)

    print(f"winner_table shape: {winner_table.shape}")
    print(f"csp_grid shape    : {csp_grid.shape}")
    print(f"CSP values        : {sorted(np.unique(csp_grid).astype(int).tolist())}")

    data_dir.mkdir(parents=True, exist_ok=True)
    write_lut_files(
        data_dir=data_dir,
        winner_table=winner_table,
        n_beams_axis=n_beams_axis,
        prompt_len_axis=prompt_len_axis,
        csp_grid=csp_grid,
        tie_threshold_ms=args.tie_threshold_ms,
        block_size=args.block_size,
        n_heads_assumed=args.n_heads_assumed,
    )


if __name__ == "__main__":
    main()
