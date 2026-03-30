from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, resolve_data_subdir


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_images(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def prepare_data(input_images_dir: Path, input_masks_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    prepared = 0
    for img_path in iter_images(input_images_dir):
        filename = img_path.stem
        mask_path = input_masks_dir / f"{filename}.png"
        if not mask_path.exists():
            mask_path = input_masks_dir / f"{filename}.jpg"
        if not mask_path.exists():
            print(f"Warning: Mask for {filename} not found, skipping.")
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            print(f"Warning: Could not read {img_path} or {mask_path}, skipping.")
            continue

        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        dest_img_path = output_dir / f"{filename}{img_path.suffix.lower()}"
        dest_mask_path = output_dir / f"{filename}_mask.png"
        cv2.imwrite(str(dest_img_path), img)
        cv2.imwrite(str(dest_mask_path), mask)
        prepared += 1

    print(f"Prepared {prepared} images in {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LaMa inference inputs from the shared mural dataset.")
    default_output = Path(__file__).resolve().parents[1] / "mural_realistic_white_input"
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    parser.add_argument("--input-subdir", type=str, default="white_block_damaged")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare_data(
        resolve_data_subdir(args.data_root, args.input_subdir),
        resolve_data_subdir(args.data_root, "masks"),
        args.output_dir,
    )
