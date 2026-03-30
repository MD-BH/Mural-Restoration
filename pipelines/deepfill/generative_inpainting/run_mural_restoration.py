import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
import neuralgym as ng
from inpaint_model import InpaintCAModel

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir


def parse_args():
    default_data_root = get_realistic_data_root()
    default_checkpoint_dir = Path(__file__).resolve().parent / "model_logs"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=str(default_data_root))
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--masks-dir', type=str, default=None)
    parser.add_argument('--original-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--samples-dir', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=str(default_checkpoint_dir))
    parser.add_argument('--sample-count', type=int, default=10)
    parser.add_argument('--mask-dilate', type=int, default=0,
                        help='Dilate binary mask by N pixels before inpainting. Set >0 for stronger visible restoration.')
    return parser.parse_args()


def load_mask(mask_path, target_size=(256, 256), dilate_px=0):
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return None, None

    if mask_gray.shape != target_size:
        mask_gray = cv2.resize(mask_gray, target_size, interpolation=cv2.INTER_NEAREST)

    binary = (mask_gray > 127).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel, iterations=1)

    mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return mask_bgr, binary


def main():
    args = parse_args()
    FLAGS = ng.Config(str(Path(__file__).resolve().parent / 'inpaint.yml'))
    data_root = Path(args.data_root)
    project_root = Path(__file__).resolve().parents[1]
    result_root = pipeline_result_root(project_root)
    input_dir = Path(args.input_dir) if args.input_dir else resolve_data_subdir(data_root, 'damaged_albedo')
    masks_dir = Path(args.masks_dir) if args.masks_dir else resolve_data_subdir(data_root, 'masks')
    original_dir = Path(args.original_dir) if args.original_dir else resolve_data_subdir(data_root, 'original_images')
    output_dir = Path(args.output_dir) if args.output_dir else result_root / 'result-damaged'
    samples_dir = Path(args.samples_dir) if args.samples_dir else result_root / 'samples-damaged'
    
    # Checkpoint logic
    checkpoint_dir = Path(args.checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir))
    if ckpt and ckpt.model_checkpoint_path:
        checkpoint_path = ckpt.model_checkpoint_path
    else:
        checkpoint_path = str(checkpoint_dir / 'snap-0')

    # Create directories if not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    input_images = sorted([f.name for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.png'}])

    if not input_images:
        print(f"No images found in {input_dir}")
        return

    # Build graph
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()

    # Network input: [1, H, W*2, 3]
    input_image_ph = tf.placeholder(tf.float32, shape=[1, 256, 512, 3])
    
    # Build Inpaint Model Graph
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)

    # Load weights
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    for var in vars_list:
        vname = var.name
        from_name = vname
        try:
            var_value = tf.train.load_variable(checkpoint_path, from_name)
            assign_ops.append(tf.assign(var, var_value))
        except Exception as e:
            # print(f"Warning: could not load variable {vname}: {e}")
            pass
    
    sess.run(assign_ops)
    print('Model loaded.')

    # Process images
    print(f"Starting processing of {len(input_images)} images from {input_dir}.")
    print(f"Mask dilation radius: {args.mask_dilate}px")
    for idx, img_name in enumerate(input_images):
        basename = os.path.splitext(img_name)[0]
        img_path = input_dir / img_name
        
        # Determine mask path 
        mask_path = masks_dir / f'{basename}.png'
        if not mask_path.exists():
             mask_path = masks_dir / f'{basename}.jpg'
        if not mask_path.exists():
            print(f"Mask not found for {img_name}, skipping.")
            continue
            
        # Determine original path
        orig_path = original_dir / f'{basename}.jpg'
        if not orig_path.exists():
             orig_path = original_dir / f'{basename}.png'
        
        # Load images
        img = cv2.imread(str(img_path))
        mask = None
        
        if img is None:
            print(f"Failed to load image or mask for {img_name}")
            continue

        orig_h, orig_w = img.shape[:2]

        # Align mask to original image resolution first, then resize both together.
        mask, mask_gray = load_mask(str(mask_path), target_size=(orig_h, orig_w), dilate_px=args.mask_dilate)
        if mask is None:
            print(f"Failed to load mask for {img_name}")
            continue

        if orig_h != 256 or orig_w != 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Prepare input
        input_data = np.concatenate([img, mask], axis=1)
        input_data = np.expand_dims(input_data, 0)
        
        # Run inference
        result = sess.run(output, feed_dict={input_image_ph: input_data})
        
        # Save result
        out_path = output_dir / img_name
        inpainted_bgr = result[0][:, :, ::-1]
        cv2.imwrite(str(out_path), inpainted_bgr)
        
        # Process samples (first 10)
        if idx < args.sample_count:
            if orig_path.exists():
                orig_img = cv2.imread(str(orig_path))
                if orig_img.shape[:2] != (256, 256):
                    orig_img = cv2.resize(orig_img, (256, 256))
            else:
                 orig_img = np.zeros((256, 256, 3), dtype=np.uint8)

            # Combine: Damage | Mask | Inpainted | Original
            sample_path = samples_dir / f'{basename}_sample_damaged.jpg'
            sample_img = np.concatenate([img, mask, inpainted_bgr, orig_img], axis=1)
            cv2.imwrite(str(sample_path), sample_img)

    print("Processing complete.")

if __name__ == "__main__":
    main()
