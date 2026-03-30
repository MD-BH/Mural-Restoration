import argparse
import os
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, resolve_data_subdir


TARGET_SIZE = (256, 256)
DIRECTORIES = [
    "damaged_albedo",
    "masks",
    "original_images",
    "white_block_damaged",
    "edges",
]


def batch_resize(base_dir: Path) -> None:
    for folder in DIRECTORIES:
        folder_path = resolve_data_subdir(base_dir, folder)
        if not folder_path.exists():
            print(f"Skipping {folder}, does not exist.")
            continue

        print(f"Processing {folder}...")
        for root, _, files in os.walk(folder_path):
            root = Path(root)
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    continue
                file_path = root / file
                try:
                    with Image.open(file_path) as img:
                        if img.size == TARGET_SIZE:
                            continue
                        print(f"Resizing {file_path} from {img.size} to {TARGET_SIZE}")
                        try:
                            resample_filter = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample_filter = Image.LANCZOS
                        resized_img = img.resize(TARGET_SIZE, resample_filter)
                        resized_img.save(file_path)
                except Exception as exc:
                    print(f"Failed to process {file_path}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resize shared mural dataset assets for DeepFill.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.data_root.exists():
        print(f"Starting resize operation in {args.data_root}")
        batch_resize(args.data_root)
        print("Resize operation completed.")
    else:
        print(f"Error: Directory {args.data_root} not found.")
