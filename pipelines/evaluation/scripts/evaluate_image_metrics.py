#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from lpips import LPIPS
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image groups against references using SSIM/PSNR/MSE (full-image), "
            "LPIPS, and FID."
        )
    )
    parser.add_argument("--reference-dir", type=Path, required=True, help="Reference image directory")
    parser.add_argument(
        "--test-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more test image directories",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save results")
    parser.add_argument("--suffix", type=str, default=".png", help="Image suffix to include, default .png")
    parser.add_argument(
        "--resize-if-needed",
        action="store_true",
        help="Resize test image to reference size when dimensions differ",
    )
    parser.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="Backbone for LPIPS model",
    )
    parser.add_argument("--fid-batch-size", type=int, default=16, help="Batch size for FID")
    parser.add_argument("--fid-num-workers", type=int, default=0, help="Num workers for FID")
    parser.add_argument(
        "--csv-decimals",
        type=int,
        default=None,
        help="Optional fixed decimal places for float values in output CSV files",
    )
    return parser.parse_args()


def list_images(image_dir: Path, suffix: str) -> Dict[str, Path]:
    return {p.name: p for p in sorted(image_dir.glob(f"*{suffix}")) if p.is_file()}


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_uint8_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)


def pil_to_lpips_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def evaluate_pair_metrics(
    ref_map: Dict[str, Path],
    test_map: Dict[str, Path],
    resize_if_needed: bool,
    lpips_model: LPIPS,
    device: torch.device,
) -> pd.DataFrame:
    common_names = sorted(set(ref_map.keys()) & set(test_map.keys()))
    if not common_names:
        raise ValueError("No matching filenames between reference and test directories.")

    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for name in common_names:
            ref_img = load_rgb(ref_map[name])
            test_img = load_rgb(test_map[name])

            if ref_img.size != test_img.size:
                if not resize_if_needed:
                    raise ValueError(
                        f"Image size mismatch for {name}: ref={ref_img.size}, test={test_img.size}. "
                        "Use --resize-if-needed to auto-resize."
                    )
                test_img = test_img.resize(ref_img.size, Image.BILINEAR)

            ref_np = pil_to_uint8_array(ref_img)
            test_np = pil_to_uint8_array(test_img)

            ssim = structural_similarity(ref_np, test_np, channel_axis=2, data_range=255)
            psnr = peak_signal_noise_ratio(ref_np, test_np, data_range=255)
            mse = mean_squared_error(ref_np, test_np)

            ref_tensor = pil_to_lpips_tensor(ref_img, device)
            test_tensor = pil_to_lpips_tensor(test_img, device)
            lpips_value = float(lpips_model(ref_tensor, test_tensor).item())

            rows.append(
                {
                    "filename": name,
                    "ssim": float(ssim),
                    "psnr": float(psnr),
                    "mse": float(mse),
                    "lpips": lpips_value,
                }
            )

    return pd.DataFrame(rows)


def evaluate_fid(
    reference_dir: Path,
    test_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    return float(
        calculate_fid_given_paths(
            [str(reference_dir), str(test_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=num_workers,
        )
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    reference_dir = args.reference_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_map = list_images(reference_dir, args.suffix)
    if not ref_map:
        raise ValueError(f"No images found in reference directory: {reference_dir}")

    device = choose_device()
    print(f"Using device: {device}")
    lpips_model = LPIPS(net=args.lpips_net).to(device)
    lpips_model.eval()

    summary_rows: List[Dict[str, float]] = []
    float_format = None if args.csv_decimals is None else f"%.{args.csv_decimals}f"

    for test_dir in args.test_dirs:
        test_map = list_images(test_dir, args.suffix)
        if not test_map:
            raise ValueError(f"No images found in test directory: {test_dir}")

        missing_in_test = sorted(set(ref_map.keys()) - set(test_map.keys()))
        extra_in_test = sorted(set(test_map.keys()) - set(ref_map.keys()))

        if missing_in_test:
            print(f"Warning: {test_dir.name} is missing {len(missing_in_test)} reference files.")
        if extra_in_test:
            print(f"Warning: {test_dir.name} has {len(extra_in_test)} extra files.")

        per_image_df = evaluate_pair_metrics(
            ref_map=ref_map,
            test_map=test_map,
            resize_if_needed=args.resize_if_needed,
            lpips_model=lpips_model,
            device=device,
        )

        fid_value = evaluate_fid(
            reference_dir=reference_dir,
            test_dir=test_dir,
            device=device,
            batch_size=args.fid_batch_size,
            num_workers=args.fid_num_workers,
        )

        per_image_path = output_dir / f"per_image_metrics_{test_dir.name}.csv"
        per_image_df.to_csv(per_image_path, index=False, float_format=float_format)

        summary_rows.append(
            {
                "test_group": test_dir.name,
                "num_pairs": int(len(per_image_df)),
                "mean_ssim": float(per_image_df["ssim"].mean()),
                "mean_psnr": float(per_image_df["psnr"].mean()),
                "mean_mse": float(per_image_df["mse"].mean()),
                "mean_lpips": float(per_image_df["lpips"].mean()),
                "fid": fid_value,
                "per_image_csv": str(per_image_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False, float_format=float_format)

    print("\n===== Summary =====")
    with pd.option_context("display.max_colwidth", None):
        print(summary_df[["test_group", "num_pairs", "mean_ssim", "mean_psnr", "mean_mse", "mean_lpips", "fid"]])
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
