#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample paired DhMurals training samples and copy them into "
            "a train_sub directory while preserving the directory structure."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "dataset" / "DhMurals-inpainting-dataset" / "train",
        help="Source train directory.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT / "dataset" / "DhMurals-inpainting-dataset" / "train_sub",
        help="Target train_sub directory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of paired samples to copy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing target directory before copying.",
    )
    return parser.parse_args()


def find_leaf_directories(root: Path) -> list[Path]:
    leaf_dirs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if "backup" in path.name.lower():
            continue
        if not any(child.is_dir() for child in path.iterdir()):
            leaf_dirs.append(path)
    return leaf_dirs


def build_stem_maps(source_root: Path, leaf_dirs: list[Path]) -> dict[Path, dict[str, Path]]:
    stem_maps: dict[Path, dict[str, Path]] = {}
    for directory in leaf_dirs:
        mapping: dict[str, Path] = {}
        for file_path in sorted(directory.iterdir()):
            if not file_path.is_file():
                continue
            stem = file_path.stem
            if stem in mapping:
                raise ValueError(
                    f"Duplicate sample stem '{stem}' found in {directory}."
                )
            mapping[stem] = file_path
        if not mapping:
            raise ValueError(f"No files found in leaf directory: {directory}")
        stem_maps[directory.relative_to(source_root)] = mapping
    return stem_maps


def copy_subset(
    source_root: Path,
    target_root: Path,
    selected_stems: list[str],
    stem_maps: dict[Path, dict[str, Path]],
) -> None:
    for relative_dir, mapping in stem_maps.items():
        destination_dir = target_root / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        for stem in selected_stems:
            source_path = mapping[stem]
            shutil.copy2(source_path, destination_dir / source_path.name)


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    target_root = args.target.resolve()

    if not source_root.is_dir():
        print(f"Source directory does not exist: {source_root}", file=sys.stderr)
        return 1

    leaf_dirs = find_leaf_directories(source_root)
    if not leaf_dirs:
        print(f"No leaf directories found under: {source_root}", file=sys.stderr)
        return 1

    try:
        stem_maps = build_stem_maps(source_root, leaf_dirs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    common_stems = sorted(set.intersection(*(set(m.keys()) for m in stem_maps.values())))
    if len(common_stems) < args.num_samples:
        print(
            (
                f"Only found {len(common_stems)} paired samples shared by all leaf "
                f"directories, which is less than the requested {args.num_samples}.\n"
                "Please verify that the files use the same sample stems across all "
                "subdirectories under train."
            ),
            file=sys.stderr,
        )
        print("Leaf directories checked:", file=sys.stderr)
        for relative_dir in sorted(stem_maps):
            print(f"  - {relative_dir.as_posix()}", file=sys.stderr)
        return 1

    if target_root.exists():
        if not args.overwrite:
            print(
                (
                    f"Target directory already exists: {target_root}\n"
                    "Use --overwrite to recreate it."
                ),
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(target_root)

    rng = random.Random(args.seed)
    selected_stems = sorted(rng.sample(common_stems, args.num_samples))
    copy_subset(source_root, target_root, selected_stems, stem_maps)

    print(f"Copied {len(selected_stems)} paired samples to: {target_root}")
    print(f"Leaf directories copied: {len(stem_maps)}")
    print(f"Random seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
