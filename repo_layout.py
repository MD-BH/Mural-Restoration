from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "dataset"
DHMURALS_ROOT = DATASET_ROOT / "DhMurals-inpainting-dataset"
DEFAULT_REALISTIC_DATA_ROOT = DHMURALS_ROOT / "test"
DEFAULT_TRAIN_DATA_ROOT = DHMURALS_ROOT / "train"
DEFAULT_TRAIN_SUB_DATA_ROOT = DHMURALS_ROOT / "train_sub"
DEFAULT_VAE_MASKS_ROOT = DATASET_ROOT / "VAE_generated_masks"

DATA_SUBDIR_ALIASES: dict[str, tuple[str, ...]] = {
    "images": ("images", "original_images"),
    "original_images": ("original_images", "images"),
    "masks": ("masks",),
    "edges": ("edges",),
    "damaged_albedo": ("damaged_albedo", "damaged_albedo_dark"),
    "damaged_albedo_dark": ("damaged_albedo_dark", "damaged_albedo"),
    "white_block_damaged": ("white_block_damaged",),
}


def get_realistic_data_root(env_var: str = "MURAL_DATA_ROOT") -> Path:
    """Return the shared inference/evaluation dataset root for restoration pipelines."""

    raw_value = os.environ.get(env_var)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return DEFAULT_REALISTIC_DATA_ROOT.resolve()


def resolve_data_subdir(data_root: str | Path, logical_name: str) -> Path:
    """Resolve a logical dataset directory across old and new naming schemes."""

    root = Path(data_root)
    candidates = DATA_SUBDIR_ALIASES.get(logical_name, (logical_name,))
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / candidates[0]


def pipeline_result_root(pipeline_root: str | Path) -> Path:
    """Return the standard result directory for a pipeline."""

    return Path(pipeline_root) / "result"
