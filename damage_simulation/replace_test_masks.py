from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "dataset" / "VAE_generated_masks"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "dataset" / "DhMurals-inpainting-dataset" / "train_sub" / "masks"


def list_image_files(directory: Path) -> List[Path]:
    """Return sorted image files under a directory."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {directory}")

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No image files were found in: {directory}")
    return files


def _copy_image_with_target_name(source_path: Path, target_path: Path) -> None:
    """Copy image content into target_path while preserving the target filename."""

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() == target_path.suffix.lower():
        shutil.copyfile(source_path, target_path)
        return

    with Image.open(source_path) as image:
        image.save(target_path)


def replace_masks_with_unique_samples(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    target_dir: Path | str = DEFAULT_TARGET_DIR,
    seed: int | None = 42,
    dry_run: bool = False,
) -> List[Tuple[Path, Path]]:
    """
    Sample source masks without replacement and overwrite target masks in order.

    The target filenames stay unchanged. The returned list contains
    (sampled_source_path, overwritten_target_path) pairs.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    source_files = list_image_files(source_dir)
    target_files = list_image_files(target_dir)

    if len(source_files) < len(target_files):
        raise ValueError(
            "Source directory does not contain enough images for non-repeating sampling: "
            f"{len(source_files)} available, {len(target_files)} required."
        )

    rng = random.Random(seed)
    sampled_sources: Sequence[Path] = rng.sample(source_files, k=len(target_files))
    replacements = list(zip(sampled_sources, target_files))

    if dry_run:
        return replacements

    for source_path, target_path in replacements:
        _copy_image_with_target_name(source_path, target_path)

    return replacements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample images from dataset/VAE_generated_masks without replacement "
            "and overwrite dataset/DhMurals-inpainting-dataset/train_sub/masks while "
            "keeping target filenames unchanged."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing sampled source masks.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory whose mask files will be overwritten in place.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible non-repeating sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview replacements without modifying files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    replacements = replace_masks_with_unique_samples(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    mode = "Planned" if args.dry_run else "Completed"
    print(f"{mode} {len(replacements)} mask replacements.")
    for source_path, target_path in replacements[:10]:
        print(f"{source_path.name} -> {target_path.name}")
    if len(replacements) > 10:
        print(f"... and {len(replacements) - 10} more")


if __name__ == "__main__":
    main()
