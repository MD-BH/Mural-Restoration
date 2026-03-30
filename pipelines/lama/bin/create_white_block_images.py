from __future__ import annotations

import argparse
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


def create_white_block_images(original_images_dir: Path, masks_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for img_path in iter_images(original_images_dir):
        filename = img_path.stem
        mask_path = masks_dir / f"{filename}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{filename}.jpg"
        if not mask_path.exists():
            print(f"Warning: Mask for {filename} not found, skipping.")
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            print(f"Warning: Could not read {img_path} or {mask_path}, skipping.")
            continue

        h, w = img.shape[:2]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        white_block_img = img.copy()
        white_block_img[mask > 0] = [255, 255, 255]

        output_path = output_dir / f"{filename}{img_path.suffix.lower()}"
        cv2.imwrite(str(output_path), white_block_img)
        processed_count += 1

    print(f"Successfully processed {processed_count} images. Saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create white-block mural inputs from images and binary masks.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    parser.add_argument("--output-subdir", type=str, default="white_block_damaged")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    create_white_block_images(
        resolve_data_subdir(args.data_root, "original_images"),
        resolve_data_subdir(args.data_root, "masks"),
        resolve_data_subdir(args.data_root, args.output_subdir),
    )
