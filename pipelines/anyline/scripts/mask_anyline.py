from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_RESULT_ROOT = pipeline_result_root(PIPELINE_ROOT)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply binary masks to Anyline guidance images.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    parser.add_argument("--lines-dir", type=Path, default=PROJECT_RESULT_ROOT / "result-damaged")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_RESULT_ROOT / "result-white")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    masks_dir = resolve_data_subdir(args.data_root, "masks")

    if not args.lines_dir.exists():
        print(f"Error: Source directory does not exist: {args.lines_dir}")
        return
    if not masks_dir.exists():
        print(f"Error: Masks directory does not exist: {masks_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing images from {args.lines_dir} with masks from {masks_dir}")
    print(f"Writing masked outputs to {args.output_dir}")

    for line_path in sorted(args.lines_dir.iterdir()):
        if not line_path.is_file() or line_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        stem = line_path.stem
        mask_path = masks_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{stem}.jpg"
        if not mask_path.exists():
            print(f"Warning: No mask found for {stem}; copying original line image.")
            shutil.copy2(line_path, args.output_dir / line_path.name)
            continue

        line_img = cv2.imread(str(line_path))
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if line_img is None:
            print(f"Error reading {line_path}")
            continue
        if mask_img is None:
            print(f"Error reading {mask_path}")
            continue

        if line_img.shape[:2] != mask_img.shape[:2]:
            mask_img = cv2.resize(mask_img, (line_img.shape[1], line_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        mask_3c = cv2.merge([mask_binary, mask_binary, mask_binary])
        result = cv2.bitwise_or(line_img, mask_3c)

        out_path = args.output_dir / line_path.name
        cv2.imwrite(str(out_path), result)
        print(f"Processed {line_path.name}")


if __name__ == "__main__":
    main()
