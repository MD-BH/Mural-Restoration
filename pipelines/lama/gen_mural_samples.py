from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PIPELINE_ROOT = Path(__file__).resolve().parent


def iter_image_names(directory: Path) -> list[str]:
    return [
        path.name
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LaMa prediction and save a small mural sample set.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    parser.add_argument("--input-subdir", type=str, default="white_block_damaged")
    parser.add_argument("--output-subdir", type=str, default="result/result-white")
    parser.add_argument("--sample-subdir", type=str, default="result/samples-white")
    parser.add_argument("--model-path", type=Path, default=PIPELINE_ROOT / "big-lama")
    parser.add_argument("--predict-script", type=Path, default=PIPELINE_ROOT / "bin" / "predict.py")
    parser.add_argument("--sample-count", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_root = pipeline_result_root(PIPELINE_ROOT)

    input_dir = resolve_data_subdir(args.data_root, args.input_subdir)
    mask_dir = resolve_data_subdir(args.data_root, "masks")
    original_dir = resolve_data_subdir(args.data_root, "original_images")
    if args.output_subdir.startswith("result/"):
        out_dir = result_root / Path(*Path(args.output_subdir).parts[1:])
    else:
        out_dir = Path(args.output_subdir)
    if args.sample_subdir.startswith("result/"):
        sample_dir = result_root / Path(*Path(args.sample_subdir).parts[1:])
    else:
        sample_dir = Path(args.sample_subdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(args.predict_script),
            f"model.path={args.model_path}",
            f"indir={input_dir}",
            f"outdir={out_dir}",
        ],
        check=True,
    )

    all_images = iter_image_names(input_dir)
    if not all_images:
        print(f"No images found in {input_dir}")
        return

    sample_images = random.sample(all_images, min(args.sample_count, len(all_images)))
    for name in sample_images:
        stem = Path(name).stem
        damaged = input_dir / name
        mask = next((mask_dir / f"{stem}{suffix}" for suffix in [".png", ".jpg"] if (mask_dir / f"{stem}{suffix}").exists()), None)
        restored = next((out_dir / f"{stem}{suffix}" for suffix in [".png", ".jpg"] if (out_dir / f"{stem}{suffix}").exists()), None)
        original = next((original_dir / f"{stem}{suffix}" for suffix in [".png", ".jpg"] if (original_dir / f"{stem}{suffix}").exists()), None)

        if mask is None or restored is None or original is None:
            print(f"Skipping incomplete sample bundle for {name}")
            continue

        shutil.copy2(damaged, sample_dir / f"{stem}_damaged{damaged.suffix.lower()}")
        shutil.copy2(mask, sample_dir / f"{stem}_mask{mask.suffix.lower()}")
        shutil.copy2(restored, sample_dir / f"{stem}_restored{restored.suffix.lower()}")
        shutil.copy2(original, sample_dir / f"{stem}_original{original.suffix.lower()}")


if __name__ == "__main__":
    main()
