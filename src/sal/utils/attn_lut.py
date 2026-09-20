"""Runtime LUT for vLLM chunked-prefill paged-decode CHUNK_Size_Page.

Loaded from JSON produced by scripts/profile_results_analyze.ipynb. Best-of-n
callers consult `pick_csp(live_n_beams, live_kv_len)` before each engine.step()
and assign the result to `vllm.attention.ops.chunked_prefill_paged_decode
.PROFILE_CHUNK_SIZE_PAGE` (None == FA path; int N == FD-N path).

Lookup uses floor-nearest on both axes: live workloads outside the measured
grid clamp to the nearest validated grid point rather than extrapolating.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ENV_VAR = "SAL_ATTN_LUT_PATH"
# Repo root, used to resolve the default per-model LUT location.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_lut_path(model_path: str) -> Path:
    """Per-model default LUT location, matching where the analysis notebook writes it."""
    return _REPO_ROOT / "data" / model_path / "profile_results_lut.json"


def lut_path_for_mode(model_path: str, mode: str) -> Path | None:
    """Resolve LUT path for the `chunk_size` config value.

    Only "dynamic" loads a LUT at runtime; "heuristic" mode is handled by
    sal.utils.xformers_attn (a separate kernel that computes its own chunk
    size per call), and "none" stays on the FA path.
    """
    if mode == "dynamic":
        return _REPO_ROOT / "data" / model_path / "profile_results_lut.json"
    return None


class AttnLUT:
    def __init__(self, n_beams_axis, prompt_len_axis, csp_grid, source: str = ""):
        self.n_beams_axis = np.asarray(n_beams_axis, dtype=np.int32)
        self.prompt_len_axis = np.asarray(prompt_len_axis, dtype=np.int32)
        self.csp_grid = np.asarray(csp_grid, dtype=np.int16)
        self.source = source
        expected = (len(self.n_beams_axis), len(self.prompt_len_axis))
        if self.csp_grid.shape != expected:
            raise ValueError(
                f"csp_grid shape {self.csp_grid.shape} does not match axes {expected}"
            )

    @classmethod
    def load(cls, path) -> "AttnLUT":
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            n_beams_axis=data["n_beams_axis"],
            prompt_len_axis=data["prompt_len_axis"],
            csp_grid=data["csp_grid"],
            source=str(path),
        )

    @classmethod
    def from_env(cls, model_path: str | None = None) -> "AttnLUT | None":
        """Resolve a LUT path, in order: ``SAL_ATTN_LUT_PATH`` env var, then
        ``data/{model_path}/profile_results_lut.json`` if ``model_path`` is given."""
        path = os.environ.get(ENV_VAR)
        if path:
            return cls.load(path)
        if model_path is not None:
            cand = default_lut_path(model_path)
            if cand.exists():
                return cls.load(cand)
        return None

    def pick_csp(self, live_n_beams: int, live_kv_len: int) -> int | None:
        """Return None for FA path, int N for FD-N path."""
        i = int(np.searchsorted(self.n_beams_axis, live_n_beams, side="right") - 1)
        j = int(np.searchsorted(self.prompt_len_axis, live_kv_len, side="right") - 1)
        i = max(i, 0)
        j = max(j, 0)
        csp = int(self.csp_grid[i, j])
        return None if csp == 0 else csp
