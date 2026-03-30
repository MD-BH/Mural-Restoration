# Mural-Restoration

Workspace for mural damage simulation, mask generation experiments, and optional intrinsic-editing side projects.

## Overview

The main tracked part of this repository focuses on synthetic mural-damage generation and dataset preparation:

- `damage_simulation/`: the reusable mural damage simulator, dataset helper scripts, and an interactive demo notebook.
- `VAE_learn_mask.ipynb`: an experimental notebook for training a VAE on mask data and generating synthetic masks.
- `dataset/`: large local datasets used by the scripts and notebooks. This directory stays local and is not pushed to GitHub.
- `rgbx/`: a separate local side project for RGB-to-intrinsic style processing. It is treated as its own project and environment.
- `intrinsic_edit/`: a separate local side project for intrinsic editing. It is also managed with its own environment.

To keep the main repository lightweight and reproducible, `dataset/`, `rgbx/`, and `intrinsic_edit/` are not tracked here as regular source content.

## Damage Modes

The damage simulator currently supports three damage styles:

- `base_fill`
- `rough_plaster`
- `powdered_loss`

`damage_simulation/mural_damage_simulator.py` exposes `apply_damage(...)` and `batch_apply_damage(...)` for generating damaged murals and saving metadata such as base color, alpha masks, texture noise, and sampling parameters.

## Project Structure

The workspace is organized at a high level like this:

```text
.
├── damage_simulation/
├── VAE_learn_mask.ipynb
├── dataset/
├── rgbx/
└── intrinsic_edit/
```

The internal structures of `rgbx/` and `intrinsic_edit/` are intentionally omitted here because they are maintained as separate side projects.

## Installation

For the tracked damage-simulation workflow, create an environment and install the libraries used by the scripts and notebooks:

```bash
pip install numpy opencv-python matplotlib pillow torch torchvision tqdm jupyter ipywidgets
```

The side projects use their own environments:

- `rgbx/` should be installed with the environment definition shipped inside that project.
- `intrinsic_edit/` should be installed with the environment definition shipped inside that project.

## Quick Start

The quickest way to explore the main workflow is the demo notebook:

```bash
jupyter notebook damage_simulation/demo.ipynb
```

If you want to call the simulator from Python, run from inside `damage_simulation/`:

```bash
cd damage_simulation
python - <<'PY'
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
print(metadata["damage_type"], metadata["nonzero_mask_pixels"])
PY
```

## Utility Scripts

Sample a reproducible DhMurals subset:

```bash
python damage_simulation/sample_dhmurals_train_subset.py --num-samples 1000 --overwrite
```

Replace subset masks with non-repeating VAE-generated masks:

```bash
python damage_simulation/replace_test_masks.py --seed 42
```

## Notes

- The helper scripts resolve their default dataset paths relative to the repository root.
- `VAE_learn_mask.ipynb` contains the mask-generation experiment that complements the main damage-simulation workflow.
- `rgbx/` and `intrinsic_edit/` are currently treated as local companion projects rather than content tracked by this repository.
