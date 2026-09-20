#!/usr/bin/env python
"""Derive smaller-n trace files from an existing larger-n trace.

Each beam in ``best_of_n_beam.py`` is an independent vLLM request, so
``REQUEST_GEN_TOKENS`` is just a per-(problem, beam) length table that
forces ``min_tokens == max_tokens`` for that request. Taking the first
``n`` beams from a ``beams={N}`` trace therefore gives a valid trace for
any ``n <= N`` — no re-running needed.

For each target ``n`` the script writes three files in the same directory
as the source:

* ``beams={n}``             — Python literal trace loaded by best_of_n_beam
* ``beams={n}.csv``         — ``problem_idx, beam_idx, gen_len``
* ``beams={n}_timing.csv``  — ``problem_idx, beam_idx, gen_time_ms``

The path layout matches what ``scripts/best_of_n_task_trace.py`` writes,
i.e. ``data/{model_path}/{dataset_short_name}/``.

Usage:
    python scripts/extract_smaller_beams.py recipes/best-of-n.yaml \\
        --source-n 128 --target-ns 2 4 8 16 32 64
"""

import argparse
import csv
import logging
import os
import runpy
from pathlib import Path

from sal.config import Config
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_config(yaml_path: str) -> Config:
    parser = H4ArgumentParser(Config)
    output = parser.parse_yaml_file(os.path.abspath(yaml_path))
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def resolve_dir(config: Config) -> Path:
    output_dir = config.output_dir or f"data/{config.model_path}"
    dataset_short_name = config.dataset_name.split("/")[-1]
    return Path(output_dir) / dataset_short_name


def filter_csv(src: Path, dst: Path, target_n: int) -> int:
    """Copy `src` -> `dst`, keeping only rows with beam_idx < target_n."""
    with src.open("r", newline="") as f_in, dst.open("w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        kept = 0
        for row in reader:
            try:
                beam_idx = int(row[1])
            except (ValueError, IndexError):
                continue
            if beam_idx < target_n:
                writer.writerow(row)
                kept += 1
    return kept


def filter_trace(src: Path, dst: Path, target_n: int) -> int:
    """Read REQUEST_GEN_TOKENS from `src`, write filtered version to `dst`."""
    ns = runpy.run_path(str(src))
    src_dict = dict(ns.get("REQUEST_GEN_TOKENS", {}))
    filtered = {(p, b): v for (p, b), v in src_dict.items() if b < target_n}

    lines = [
        f"# Derived from {src.name} by scripts/extract_smaller_beams.py.",
        f"# Keeping only beam_idx < {target_n} from {src.name}.",
        "REQUEST_GEN_TOKENS = {",
    ]
    for problem_idx, beam_idx in sorted(filtered.keys()):
        lines.append(
            f"    ({problem_idx}, {beam_idx}): {filtered[(problem_idx, beam_idx)]},"
        )
    lines.append("}")
    lines.append("")
    dst.write_text("\n".join(lines))
    return len(filtered)


def main():
    argp = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argp.add_argument("yaml_path", help="Recipe YAML (e.g. recipes/best-of-n.yaml)")
    argp.add_argument(
        "--source-n", type=int, required=True,
        help="Source beam count (must already exist in the data dir)",
    )
    argp.add_argument(
        "--target-ns", type=int, nargs="+", required=True,
        help="Beam counts to derive; each must be <= source-n",
    )
    args = argp.parse_args()

    config = load_config(args.yaml_path)
    data_dir = resolve_dir(config)
    logger.info("Working directory: %s", data_dir)

    src_trace = data_dir / f"beams={args.source_n}"
    src_csv = data_dir / f"beams={args.source_n}.csv"
    src_timing = data_dir / f"beams={args.source_n}_timing.csv"
    for path, label in (
        (src_trace, "Python literal trace"),
        (src_csv, "gen_len CSV"),
        (src_timing, "timing CSV"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Source {label} not found: {path}")

    for n in args.target_ns:
        if n > args.source_n:
            logger.warning(
                "Skipping target n=%d (> source-n=%d); cannot fabricate beams.",
                n, args.source_n,
            )
            continue
        if n == args.source_n:
            logger.info("Skipping target n=%d (same as source-n; nothing to do)", n)
            continue

        kept_trace = filter_trace(src_trace, data_dir / f"beams={n}", n)
        kept_csv = filter_csv(src_csv, data_dir / f"beams={n}.csv", n)
        kept_timing = filter_csv(src_timing, data_dir / f"beams={n}_timing.csv", n)
        logger.info(
            "n=%d -> trace=%d entries, gen_len.csv=%d rows, timing.csv=%d rows",
            n, kept_trace, kept_csv, kept_timing,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
