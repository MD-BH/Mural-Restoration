import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 256x256 if needed
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256))
    return img

def ssim_map_calc(img1, img2, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=255.0):
    """
    Calculate SSIM map for a single channel.
    """
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (window_size, window_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (window_size, window_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), sigma) - mu1_mu2
    
    ssim_map_res = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map_res

def evaluate_folder(result_dir, original_dir, masks_dir, label):
    print(f"\nEvaluating: {label} (Masked Region Only)")
    print(f"Result Dir: {result_dir}")
    print(f"Original Dir: {original_dir}")
    print(f"Masks Dir: {masks_dir}")
    
    if not os.path.exists(result_dir):
        print(f"Error: Directory {result_dir} does not exist.")
        return

    file_names = sorted([f for f in os.listdir(result_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    masked_psnr_values = []
    masked_l1_values = []
    masked_ssim_values = []
    
    count = 0
    
    for fname in file_names:
        res_path = os.path.join(result_dir, fname)
        
        # Try both extensions for original
        base_name = os.path.splitext(fname)[0]
        orig_path = os.path.join(original_dir, base_name + '.jpg')
        if not os.path.exists(orig_path):
            orig_path = os.path.join(original_dir, base_name + '.png')
            
        # Try both extensions for mask
        mask_path = os.path.join(masks_dir, base_name + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, base_name + '.jpg')

        if not os.path.exists(orig_path):
            continue
        if not os.path.exists(mask_path):
            print(f"Mask not found for {fname}")
            continue
            
        img_res = load_image(res_path)
        img_orig = load_image(orig_path)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        if mask.shape != (256, 256):
            mask = cv2.resize(mask, (256, 256))
        
        if img_res is None or img_orig is None:
            continue
            
        # Convert to float32
        img_res_f = img_res.astype(np.float32)
        img_orig_f = img_orig.astype(np.float32)
        
        # Normalize mask: 1 for hole (sample/damaged area), 0 for valid
        mask_f = (mask > 127.5).astype(np.float32) 
        # mask_f is (256, 256)
        
        # Calculate number of pixels in the hole
        valid_pixels = np.sum(mask_f)
        if valid_pixels == 0:
            continue
            
        mask_f_3d = np.expand_dims(mask_f, axis=-1) # (256, 256, 1)

        # Difference
        diff = np.abs(img_res_f - img_orig_f) * mask_f_3d
        
        # L1 (MAE)
        l1 = np.sum(diff) / (valid_pixels * 3)
        
        # MSE
        diff_sq = (diff ** 2)
        mse = np.sum(diff_sq) / (valid_pixels * 3)
        
        # PSNR
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10.0 * np.log10((255.0 ** 2) / mse)
            
        # SSIM (Masked)
        ssim_accum = np.zeros_like(img_res_f[:,:,0], dtype=np.float64)
        for i in range(3):
            ssim_accum += ssim_map_calc(img_res_f[:,:,i], img_orig_f[:,:,i])
        
        ssim_map_avg = ssim_accum / 3.0
        # Average SSIM only inside the mask
        ssim_val = np.sum(ssim_map_avg * mask_f) / valid_pixels
        
        masked_psnr_values.append(psnr)
        masked_l1_values.append(l1)
        masked_ssim_values.append(ssim_val)
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} images...", end='\r')
            
    if count == 0:
        print("No matching file pairs found.")
        return

    print(f"Processed {count} images.          ")
    
    avg_psnr = np.mean(masked_psnr_values)
    avg_l1 = np.mean(masked_l1_values)
    avg_ssim = np.mean(masked_ssim_values)
    
    print("-" * 40)
    print(f"Masked Metrics for {label}:")
    print(f"L1 Loss (MAE) ↓: {avg_l1:.4f}")
    print(f"PSNR          ↑: {avg_psnr:.4f}")
    print(f"SSIM          ↑: {avg_ssim:.4f}")
    print("-" * 40)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DeepFill outputs against the shared mural dataset.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    result_root = pipeline_result_root(Path(__file__).resolve().parents[1])
    parser.add_argument("--white-output-dir", type=Path, default=result_root / "result-white")
    parser.add_argument("--damaged-output-dir", type=Path, default=result_root / "result-damaged")
    return parser


def main():
    args = build_parser().parse_args()
    original_dir = str(resolve_data_subdir(args.data_root, 'original_images'))
    masks_dir = str(resolve_data_subdir(args.data_root, 'masks'))

    evaluate_folder(str(args.white_output_dir), original_dir, masks_dir, "Result-White (White Block)")
    evaluate_folder(str(args.damaged_output_dir), original_dir, masks_dir, "Result-Damaged (Damaged Albedo)")

if __name__ == "__main__":
    main()
