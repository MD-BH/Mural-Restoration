from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir


def calculate_metrics(gt_path: Path, pred_path: Path, mask_path: Path):
    gt_img = cv2.imread(str(gt_path))
    pred_img = cv2.imread(str(pred_path))
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if gt_img is None:
        print(f"Error reading GT: {gt_path}")
        return None
    if pred_img is None:
        print(f"Error reading prediction: {pred_path}")
        return None
    if mask_img is None:
        print(f"Error reading mask: {mask_path}")
        return None

    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
    if mask_img.shape != gt_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    mask_bool = mask_binary > 0
    if np.sum(mask_bool) == 0:
        print(f"Warning: Empty mask for {mask_path}")
        return None

    diff = np.abs(gt_img.astype(np.float32) - pred_img.astype(np.float32))
    l1_loss = np.mean(diff[mask_bool])

    sq_diff = (gt_img.astype(np.float32) - pred_img.astype(np.float32)) ** 2
    mse = np.mean(sq_diff[mask_bool])
    psnr_val = 100.0 if mse == 0 else 10 * np.log10(255**2 / mse)

    try:
        _, ssim_img = ssim(gt_img, pred_img, channel_axis=2, data_range=255, full=True)
        ssim_val = np.mean(ssim_img[mask_bool])
    except Exception as exc:
        print(f"SSIM error for {pred_path.name}: {exc}")
        ssim_val = 0.0

    return l1_loss, psnr_val, ssim_val


def evaluate_folder(dataset_name: str, output_dir: Path, gt_dir: Path, mask_dir: Path) -> None:
    print(f"\nEvaluating {dataset_name} (masked region)")
    if not output_dir.exists():
        print(f"Skipping missing output directory: {output_dir}")
        return

    l1_scores = []
    psnr_scores = []
    ssim_scores = []

    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        basename = path.stem.replace("_mask", "")

        gt_path = gt_dir / f"{basename}.jpg"
        if not gt_path.exists():
            gt_path = gt_dir / f"{basename}.png"
        if not gt_path.exists():
            print(f"GT not found for {path.name}")
            continue

        mask_path = mask_dir / f"{basename}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{basename}_mask.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{basename}.jpg"
        if not mask_path.exists():
            print(f"Mask not found for {path.name}")
            continue

        metrics = calculate_metrics(gt_path, path, mask_path)
        if metrics is None:
            continue

        l1_loss, psnr_val, ssim_val = metrics
        l1_scores.append(l1_loss)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    if not l1_scores:
        print(f"No valid image pairs found for {dataset_name}")
        return

    print(f"Images processed: {len(l1_scores)}")
    print(f"Average L1 Loss: {np.mean(l1_scores):.4f}")
    print(f"Average PSNR:    {np.mean(psnr_scores):.4f}")
    print(f"Average SSIM:    {np.mean(ssim_scores):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LaMa outputs against masked mural ground truth.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    result_root = pipeline_result_root(Path(__file__).resolve().parent)
    parser.add_argument("--damaged-output-dir", type=Path, default=result_root / "result-damaged")
    parser.add_argument("--white-output-dir", type=Path, default=result_root / "result-white")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gt_dir = resolve_data_subdir(args.data_root, "original_images")
    mask_dir = resolve_data_subdir(args.data_root, "masks")

    evaluate_folder("Mural Realistic Damaged", args.damaged_output_dir, gt_dir, mask_dir)
    evaluate_folder("Mural Realistic White", args.white_output_dir, gt_dir, mask_dir)


if __name__ == "__main__":
    main()
