import cv2
import glob
import os
import shutil
from pathlib import Path

def create_samples(input_dir, output_dir, samples_dir, count=3):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    samples_dir = Path(samples_dir)
    
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)
    
    mask_files = sorted(list(output_dir.glob("*_mask.png")))
    
    selected_files = mask_files[:count]
    
    print(f"Generating {len(selected_files)} samples...")
    
    for mask_path in selected_files:
        filename = mask_path.stem.replace("_mask", "") # e.g. 000098
        
        # Paths
        input_img_path = input_dir / (filename + ".jpg")
        input_mask_path = input_dir / (filename + "_mask.png")
        output_img_path = mask_path

        if not input_img_path.exists():
             print(f"Warning: Input image for {filename} not found at {input_img_path}, using placeholder.")
             continue
             
        # Read images
        img_in = cv2.imread(str(input_img_path))
        mask_in = cv2.imread(str(input_mask_path))
        img_out = cv2.imread(str(output_img_path))
        
        if img_in is None or mask_in is None or img_out is None:
             print(f"Error reading images for {filename}, skipping.")
             continue

        # Resize for consistent viewing if needed (they should be same size now due to prev steps)
        h, w = img_in.shape[:2]
        if mask_in.shape[:2] != (h,w):
             mask_in = cv2.resize(mask_in, (w, h))
        if img_out.shape[:2] != (h,w):
             img_out = cv2.resize(img_out, (w, h))

        # Concatenate horizontally
        # Add labels maybe? Naah just raw stack
        combined = cv2.hconcat([img_in, mask_in, img_out])
        
        # Save
        save_path = samples_dir / f"sample_{filename}.png"
        cv2.imwrite(str(save_path), combined)
        print(f"Saved sample: {save_path}")

if __name__ == "__main__":
    create_samples(
        "/Users/zyy/workspace/GNN-Lama/lama/mural_realistic_white_input",
        "/Users/zyy/workspace/GNN-Lama/lama/mural_realistic_white_output",
        "/Users/zyy/workspace/GNN-Lama/lama/mural_realistic_white_samples",
        count=5
    )
