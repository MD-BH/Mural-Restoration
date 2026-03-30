import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
import neuralgym as ng
import matplotlib.pyplot as plt
from inpaint_model import InpaintCAModel

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, resolve_data_subdir

def visualize_gates(gate_maps, save_path):
    """
    Visualizes gate maps.
    gate_maps: list of (H, W) arrays (averaged over channels)
    """
    num_layers = len(gate_maps)
    cols = 4
    rows = (num_layers + cols - 1) // cols
    
    plt.figure(figsize=(15, 3 * rows))
    for i, gmap in enumerate(gate_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(gmap, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Layer {i} Gate')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_sample(image_path, mask_path, checkpoint_dir, output_prefix):
    print(f"Analyzing {image_path}...")
    
    # Load Image & Mask
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 0-255
    
    if img is None or mask is None:
        print("Error loading image/mask")
        return

    # Resize to 256x256
    img = cv2.resize(img, (256, 256))
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Expand dims
    mask_3d = np.expand_dims(mask, axis=-1) # (256, 256, 1)
    # Ensure mask is 0 or 255. Threshold just in case.
    _, mask_3d = cv2.threshold(mask_3d, 127, 255, cv2.THRESH_BINARY)
    
    # Mask format for model: 255 is hole/mask? 
    # Usually in DeepFill: 255 is mask (hole). 
    # Let's check `test.py`: Yes.
    
    # Model input: image + mask concatenated
    # image: BGR -> maybe need to check if model handles BGR/RGB?
    # NeuralGym usually expects RGB if trained on RGB.
    # cv2 loads BGR.
    # BUT `test.py` does `cv2.imread` then passes to model directly. 
    # If trained on BGR (default cv2), then fine.
    # Standard practice is to check data loader.
    # Assuming inputs are fine as BGR since test.py does that.
    
    img_input = img
    mask_input = np.expand_dims(mask, axis=-1)
    
    input_data = np.concatenate([img_input, mask_input], axis=2) # (256, 256, 4)
    input_data = np.expand_dims(input_data, 0) # (1, 256, 256, 4)

    # Clean Graph
    tf.reset_default_graph()
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    
    FLAGS = ng.Config(str(Path(__file__).resolve().parent / 'inpaint.yml'))
    model = InpaintCAModel()

    input_ph = tf.placeholder(tf.float32, shape=[1, 256, 256, 4])
    
    # Build Graph
    # The `build_server_graph` assumes input is (1, H, W, C) where C is correct split.
    # Wait, `build_server_graph` splits input by `input_image.shape[2]`.
    # Let's inspect `inpaint_model.py`:
    #   if FLAGS.guided: ... split(3) ...
    #   else: batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
    # It splits in half!
    # So input must be [Image, Mask] concatenated but Mask must have same channels as Image? NO.
    # Wait, `test.py` does: `np.concatenate([image, mask], axis=2)`. Mask is 3 channels there!
    # My simple analysis code: mask is 1 channel.
    # I MUST MAKE MASK 3 CHANNELS to match `test.py` logic if using `build_server_graph` directly from unmodified model.
    # Let's check `build_server_graph` again (I read it earlier).
    #   batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
    # Yes, it splits in half. So mask must have 3 channels if image has 3.
    
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Concatenate along WIDTH (Axis 1 for HWC)
    input_data = np.concatenate([img_input, mask_3c], axis=1) # (256, 512, 3)
    input_data = np.expand_dims(input_data, 0) # (1, 256, 512, 3)

    # Placeholder must be [1, 256, 512, 3]
    input_ph = tf.placeholder(tf.float32, shape=[1, 256, 512, 3])
    
    output = model.build_server_graph(FLAGS, input_ph)
    
    # Get Gating Feature Maps
    gate_tensors = tf.get_collection('gate_maps')
    
    # Load Checkpoint
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    checkpoint_path = ckpt.model_checkpoint_path if ckpt else os.path.join(checkpoint_dir, 'snap-0')
    
    for var in vars_list:
        try:
            val = tf.train.load_variable(checkpoint_path, var.name)
            assign_ops.append(tf.assign(var, val))
        except:
            pass
    sess.run(assign_ops)
    
    # Run Inference
    gates_val = sess.run(gate_tensors, feed_dict={input_ph: input_data})
    
    # Process Gates
    # gates_val is a list of [1, H, W, C].
    # We want mean across C -> [H, W]
    
    vis_gates = []
    stats = []

    # Get Mask as binary (0, 1) for stats
    mask_binary = (mask > 127).astype(np.float32)
    
    for i, g in enumerate(gates_val):
        # g: (1, H, W, C)
        g_mean_c = np.mean(g[0], axis=-1) # (H, W)
        
        # Resize mask to current gate size for stats
        h, w = g_mean_c.shape
        mask_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Stats
        val_in_hole = g_mean_c[mask_resized == 1]
        val_outside = g_mean_c[mask_resized == 0]
        
        mean_hole = np.mean(val_in_hole) if val_in_hole.size > 0 else 0
        mean_out = np.mean(val_outside) if val_outside.size > 0 else 0
        
        stats.append((i, mean_hole, mean_out))
        vis_gates.append(g_mean_c)

    # Save visualization
    visualize_gates(vis_gates, output_prefix + "_gates_vis.png")
    
    # Print stats
    print(f"\nGate Statistics (Mean Activation) for {output_prefix}:")
    print(f"{'Layer':<5} | {'Hole Mean':<10} | {'Valid Mean':<10} | {'Ratio (H/V)':<10}")
    print("-" * 45)
    for i, mh, mo in stats:
        ratio = mh / (mo + 1e-6)
        print(f"{i:<5} | {mh:.4f}     | {mo:.4f}      | {ratio:.4f}")

    return stats

def main():
    base_dir = get_realistic_data_root()
    sample_name = '000098'
    
    # Paths
    white_root = resolve_data_subdir(base_dir, 'white_block_damaged')
    damaged_root = resolve_data_subdir(base_dir, 'damaged_albedo')
    masks_root = resolve_data_subdir(base_dir, 'masks')
    white_img = str((white_root / f'{sample_name}.png') if (white_root / f'{sample_name}.png').exists() else (white_root / f'{sample_name}.jpg'))
    damaged_img = str((damaged_root / f'{sample_name}.png') if (damaged_root / f'{sample_name}.png').exists() else (damaged_root / f'{sample_name}.jpg'))
    mask_img = str(masks_root / f'{sample_name}.png')
    checkpoint_dir = str(Path(__file__).resolve().parent / 'model_logs')
    
    if not os.path.exists(mask_img):
         mask_img = str(masks_root / f'{sample_name}.jpg')
    
    output_dir = str(Path(__file__).resolve().parent / 'gate_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Analyzing White Block ---")
    analyze_sample(white_img, mask_img, checkpoint_dir, os.path.join(output_dir, 'white_block'))
    
    print("\n--- Analyzing Damaged Albedo ---")
    analyze_sample(damaged_img, mask_img, checkpoint_dir, os.path.join(output_dir, 'damaged_albedo'))

if __name__ == "__main__":
    main()
