import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from batch_inference import PowerPaintBatchInference

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir


def make_sample_strip(damaged, mask, restored, original):
    # Keep all panels the same size for clean side-by-side comparison.
    w, h = damaged.size
    mask_r = mask.resize((w, h), Image.NEAREST)
    restored_r = restored.resize((w, h), Image.LANCZOS)
    original_r = original.resize((w, h), Image.LANCZOS)

    strip = Image.new("RGB", (w * 4, h), (255, 255, 255))
    strip.paste(damaged, (0, 0))
    strip.paste(mask_r, (w, 0))
    strip.paste(restored_r, (w * 2, 0))
    strip.paste(original_r, (w * 3, 0))
    return strip


def run_dataset(
    inference,
    dataset_name,
    input_dir,
    mask_dir,
    original_dir,
    output_root,
    ddim_steps,
    enable_control,
    max_images=0,
):
    result_dir = output_root
    sample_dir = os.path.join(output_root, "samples")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Clean old sample files so each run keeps latest 10 sample strips.
    for name in os.listdir(sample_dir):
        p = os.path.join(sample_dir, name)
        if os.path.isfile(p):
            os.remove(p)

    # Force PNG-only input to avoid accidentally using converted JPG folders.
    all_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    all_imgs.sort()
    if max_images > 0:
        all_imgs = all_imgs[:max_images]

    sample_count = 0
    sample_max = 10

    print(f"\n===== Running dataset: {dataset_name} ({len(all_imgs)} images) =====")
    for img_name in all_imgs:
        img_path = os.path.join(input_dir, img_name)
        stem, _ = os.path.splitext(img_name)
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        orig_path = os.path.join(original_dir, f"{stem}.png")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Skip {img_name}: missing image or mask.")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")

            result = inference.predict(
                input_image=img,
                input_mask=mask,
                prompt="restored mural painting, high quality",
                task="context-aware",
                fitting_degree=1.0,
                ddim_steps=ddim_steps,
                scale=7.5,
                seed=42,
                enable_control=enable_control,
                control_type="hed",
            )

            if result:
                out_path = os.path.join(result_dir, f"{stem}.png")
                result.save(out_path)
                print(f"Saved: {out_path}")

                if sample_count < sample_max and os.path.exists(orig_path):
                    orig = Image.open(orig_path).convert("RGB")
                    sample_strip = make_sample_strip(img, mask, result.convert("RGB"), orig)
                    sample_out = os.path.join(sample_dir, f"{sample_count + 1:02d}_sample_compare.png")
                    sample_strip.save(sample_out)
                    sample_count += 1
            else:
                print(f"Failed to process {img_name}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print(f"Completed {dataset_name}. Results: {result_dir}, Samples: {sample_dir} ({sample_count} files)")

def main():
    parser = argparse.ArgumentParser(description="Run PowerPaint on two mural PNG datasets with optional fast mode")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer steps (still uses PNG input folders)")
    parser.add_argument("--max-images", type=int, default=0, help="Process only first N images per dataset (0 = all)")
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).resolve().parent
    result_root = pipeline_result_root(base_dir)
    data_root = args.data_root

    mask_dir = str(resolve_data_subdir(data_root, "masks"))
    original_dir = str(resolve_data_subdir(data_root, "original_images"))

    # Fast mode only reduces steps and can disable ControlNet on white set for speed.
    if args.fast:
        damaged_input_dir = str(resolve_data_subdir(data_root, "damaged_albedo"))
        white_input_dir = str(resolve_data_subdir(data_root, "white_block_damaged"))
        damaged_steps = 12
        white_steps = 10
        # White-block set is usually faster and more stable without ControlNet.
        damaged_use_control = True
        white_use_control = False
        print("Fast mode enabled: PNG inputs, fewer steps, white set without ControlNet")
    else:
        damaged_input_dir = str(resolve_data_subdir(data_root, "damaged_albedo"))
        white_input_dir = str(resolve_data_subdir(data_root, "white_block_damaged"))
        damaged_steps = 20
        white_steps = 20
        damaged_use_control = True
        white_use_control = True

    damaged_output_root = str(result_root / "result-damaged")
    white_output_root = str(result_root / "result-white")

    checkpoint_path = str(base_dir / "checkpoints" / "ppt-v1")
    print(f"Initializing PowerPaint model with checkpoints at {checkpoint_path}...")
    inference = PowerPaintBatchInference(checkpoint_dir=checkpoint_path, version="ppt-v1")

    run_dataset(
        inference=inference,
        dataset_name="damaged_albedo_dark",
        input_dir=damaged_input_dir,
        mask_dir=mask_dir,
        original_dir=original_dir,
        output_root=damaged_output_root,
        ddim_steps=damaged_steps,
        enable_control=damaged_use_control,
        max_images=args.max_images,
    )

    run_dataset(
        inference=inference,
        dataset_name="white_block_damaged",
        input_dir=white_input_dir,
        mask_dir=mask_dir,
        original_dir=original_dir,
        output_root=white_output_root,
        ddim_steps=white_steps,
        enable_control=white_use_control,
        max_images=args.max_images,
    )

if __name__ == "__main__":
    main()
