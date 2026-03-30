#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Set

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, resolve_data_subdir


PIPELINES_ROOT = REPO_ROOT / "pipelines"


def png_names(folder: Path) -> Set[str]:
    return {p.name for p in folder.glob("*.png") if p.is_file()}


def load_as_rgb(path: Path, target_size: tuple[int, int], is_mask: bool = False) -> Image.Image:
    img = Image.open(path)
    if img.size != target_size:
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize(target_size, resample)
    return img.convert("RGB")


def make_four_panel(mask_path: Path, input_path: Path, output_path: Path, original_path: Path, save_path: Path) -> None:
    original = Image.open(original_path).convert("RGB")
    base_size = original.size

    mask = load_as_rgb(mask_path, base_size, is_mask=True)
    damaged = load_as_rgb(input_path, base_size)
    repaired = load_as_rgb(output_path, base_size)

    canvas = Image.new("RGB", (base_size[0] * 4, base_size[1]))
    canvas.paste(mask, (0, 0))
    canvas.paste(damaged, (base_size[0], 0))
    canvas.paste(repaired, (base_size[0] * 2, 0))
    canvas.paste(original, (base_size[0] * 3, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create comparison PNGs for the standard restoration pipeline outputs.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mask_dir = resolve_data_subdir(args.data_root, "masks")
    original_dir = resolve_data_subdir(args.data_root, "original_images")
    input_dirs = {
        "result-white": resolve_data_subdir(args.data_root, "white_block_damaged"),
        "result-damage": resolve_data_subdir(args.data_root, "damaged_albedo"),
    }
    model_outputs: Dict[str, Dict[str, Path]] = {
        "lama": {
            "result-white": PIPELINES_ROOT / "lama" / "result" / "result-white",
            "result-damage": PIPELINES_ROOT / "lama" / "result" / "result-damaged",
        },
        "deepfill": {
            "result-white": PIPELINES_ROOT / "deepfill" / "result" / "result-white",
            "result-damage": PIPELINES_ROOT / "deepfill" / "result" / "result-damaged",
        },
        "muralnet": {
            "result-white": PIPELINES_ROOT / "muralnet" / "result" / "result-white",
            "result-damage": PIPELINES_ROOT / "muralnet" / "result" / "result-damaged",
        },
        "powerpaint": {
            "result-white": PIPELINES_ROOT / "powerpaint" / "result" / "result-white",
            "result-damage": PIPELINES_ROOT / "powerpaint" / "result" / "result-damaged",
        },
    }

    mask_names = png_names(mask_dir)
    original_names = png_names(original_dir)

    for model_name, groups in model_outputs.items():
        compare_root = PIPELINES_ROOT / model_name / "compare_png"

        for group_name, output_dir in groups.items():
            input_dir = input_dirs[group_name]
            if not output_dir.exists():
                print(f"Skipping {model_name}/{group_name}: missing output directory {output_dir}")
                continue
            group_out = compare_root / group_name
            group_out.mkdir(parents=True, exist_ok=True)

            names = sorted(mask_names & original_names & png_names(input_dir) & png_names(output_dir))
            for name in names:
                make_four_panel(
                    mask_path=mask_dir / name,
                    input_path=input_dir / name,
                    output_path=output_dir / name,
                    original_path=original_dir / name,
                    save_path=group_out / name,
                )

            print(f"{model_name}/{group_name}: saved {len(names)} PNG files")


if __name__ == "__main__":
    main()
