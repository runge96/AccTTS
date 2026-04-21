# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import logging
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config) -> Dataset:
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.shuffle(seed=config.seed)
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            config.output_dir = f"data/{config.model_path}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        dataset.to_json(
            f"{config.output_dir}/{config.approach}_completions.jsonl", lines=True
        )
        logger.info(
            f"Saved completions to {config.output_dir}/{config.approach}_completions.jsonl"
        )
        if (
            config.approach == "beam_search"
            and "beam_step_token_lengths" in dataset.column_names
        ):
            csv_path = Path(config.output_dir) / "beam_search_step_token_lengths.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                include_timing = "attn_step_timings" in dataset.column_names
                if include_timing:
                    writer.writerow([
                        "problem", "step", "beam", "token_length",
                        "attn_time_ms", "attn_tokens", "attn_ms_per_token"
                    ])
                else:
                    writer.writerow(["problem", "step", "beam", "token_length"])
                for problem, steps, timings in zip(
                    dataset["problem"],
                    dataset["beam_step_token_lengths"],
                    dataset["attn_step_timings"]
                    if include_timing else [None] * len(dataset),
                ):
                    step_timing_map = {}
                    if include_timing and timings:
                        step_timing_map = {
                            t.get("step_idx", idx): t
                            for idx, t in enumerate(timings)
                        }
                    for step_idx, step_lengths in enumerate(steps):
                        timing = step_timing_map.get(step_idx, {})
                        attn_time_ms = timing.get("attn_time_ms", 0.0)
                        attn_tokens = timing.get("attn_tokens", 0)
                        attn_ms_per_token = timing.get("attn_ms_per_token", 0.0)
                        for beam_idx, token_len in enumerate(step_lengths):
                            row = [problem, step_idx, beam_idx, token_len]
                            if include_timing:
                                row.extend([
                                    attn_time_ms, attn_tokens, attn_ms_per_token
                                ])
                            writer.writerow(row)
            logger.info(f"Saved beam step token lengths to {csv_path}")
        if (config.approach == "best_of_n"
                and "completion_tokens" in dataset.column_names
                and "request_timings" in dataset.column_names):
            csv_path = Path(config.output_dir) / "best_of_n_request_timings.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                include_overall_attn = "overall_attn_time_ms" in dataset.column_names
                header = ["problem", "beam", "token_length", "gen_time_ms"]
                if include_overall_attn:
                    header.extend(["overall_attn_time_ms", "overall_attn_tokens"])
                writer.writerow(header)
                for problem, tokens, timings in zip(
                    dataset["problem"],
                    dataset["completion_tokens"],
                    dataset["request_timings"],
                ):
                    for beam_idx, (token_len, time_ms) in enumerate(
                            zip(tokens, timings)):
                        row = [problem, beam_idx, token_len, time_ms]
                        if include_overall_attn:
                            row.extend([
                                dataset["overall_attn_time_ms"][0],
                                dataset["overall_attn_tokens"][0]
                                if "overall_attn_tokens" in dataset.column_names else 0,
                            ])
                        writer.writerow(row)
            logger.info(f"Saved best-of-n request timings to {csv_path}")
