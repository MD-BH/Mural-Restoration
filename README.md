# Mural-Restoration

Utilities for synthetic mural-damage generation, mask preparation, and DhMurals-style dataset curation.

## Overview

This repository centers on a reusable mural damage simulator plus a small set of helper scripts and notebooks:

- `mural_damage_simulator.py`: core library for loading mural images and masks, synthesizing damage textures, and exporting batch results with metadata.
- `replace_test_masks.py`: samples masks from `dataset/VAE_generated_masks` and overwrites the paired filenames in `dataset/DhMurals-inpainting-dataset/train_sub/masks`.
- `sample_dhmurals_train_subset.py`: creates a reproducible subset from the DhMurals training split while preserving directory structure.
- `demo.ipynb`: interactive notebook for single-image testing, multi-mode comparison, and DhMurals batch experiments.
- `VAE_learn_mask.ipynb`: notebook for training a VAE on mask data and generating synthetic masks.

The large `dataset/` directory is intentionally ignored by Git, so the repository stays lightweight while still documenting the expected local layout.

## Damage Modes

The simulator currently supports three damage styles:

- `base_fill`
- `rough_plaster`
- `powdered_loss`

`apply_damage(...)` returns both the synthesized image and per-sample metadata such as base color, alpha map, texture noise, and sampling parameters. `batch_apply_damage(...)` writes output images plus `metadata.json` and `metadata.csv`.

## Expected Local Structure

The code assumes a local directory layout similar to:

```text
dataset/
├── DhMurals-inpainting-dataset/
│   ├── train/
│   ├── train_sub/
│   └── test/
├── MuralDH/
│   ├── Mural512/
│   ├── Mural_SR/
│   └── Mural_seg/
├── VAE_generated_masks/
└── decomp_test/
    ├── albedo/
    ├── mask/
    └── ...
```

Because `dataset/` is gitignored, these assets stay local and are not pushed to GitHub.

## Installation

Create an environment and install the libraries used by the scripts and notebooks:

```bash
pip install numpy opencv-python matplotlib pillow torch torchvision tqdm jupyter ipywidgets
```

## Quick Start

Use the simulator as a Python module:

```python
from mural_damage_simulator import apply_damage, load_image, load_mask

image = load_image("path/to/albedo.png")
mask = load_mask("path/to/mask.png", target_shape=image.shape[:2])
result, metadata = apply_damage(
    image=image,
    mask=mask,
    damage_type="rough_plaster",
    noise_strength=0.45,
    noise_scale=28.0,
    opacity=0.9,
    edge_sharpness=0.75,
    seed=42,
)
```

For batch synthesis:

```python
from mural_damage_simulator import batch_apply_damage

records = batch_apply_damage(
    images_dir="dataset/decomp_test/albedo",
    masks_dir="dataset/decomp_test/mask",
    output_dir="outputs/demo_batch",
    n_samples_per_image=3,
    seed=42,
)
```

## Utility Scripts

Sample a reproducible DhMurals subset:

```bash
python sample_dhmurals_train_subset.py --num-samples 1000 --overwrite
```

Replace subset masks with non-repeating VAE-generated masks:

```bash
python replace_test_masks.py --seed 42
```

## Notes

- The helper scripts now resolve default paths relative to the repository root, so they can be run after cloning without editing absolute local paths.
- `demo.ipynb` is the best entry point for visually inspecting the three damage modes on example murals.
- `VAE_learn_mask.ipynb` contains the experimental mask-generation workflow used to produce the `VAE_generated_masks` directory.
