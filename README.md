# Mural Restoration

`mural-restoration` is the canonical repository for this workspace. It combines the original damage-simulation tooling with the imported restoration pipelines from `GNN_Final_Project_Mural_Restoration` under one cleaner structure.

## Overview

The repository now has two main layers:

- `damage_simulation/`: reusable mural-damage generation, subset sampling, and mask replacement utilities.
- `pipelines/`: imported restoration and evaluation methods, organized by model instead of keeping the source repo's top-level `GNN-*` layout.

Large datasets remain local under `dataset/` and are intentionally ignored by Git.

## Repository Layout

```text
.
├── damage_simulation/
├── pipelines/
│   ├── anyline/
│   ├── deepfill/
│   ├── evaluation/
│   ├── lama/
│   ├── muralnet/
│   ├── powerpaint/
│   └── vae/
├── requirements-damage-simulation.txt
├── dataset/
├── rgbx/
└── intrinsic_edit/
```

## Shared Data Convention

The restoration pipelines default to a shared inference/evaluation dataset root at:

```text
dataset/DhMurals-inpainting-dataset/test
```

Expected subdirectories:

- `images/`
- `masks/`
- `edges/`
- `damaged_albedo/`
- `white_block_damaged/` (optional, can be generated with `pipelines/lama/bin/create_white_block_images.py`)

Set `MURAL_DATA_ROOT` to point the imported pipelines at a different compatible data root. The path resolver also accepts older names like `original_images/` and `damaged_albedo_dark/` for backward compatibility.

## Installation

For the core damage-simulation workflow:

```bash
pip install -r requirements-damage-simulation.txt
```

The imported restoration methods keep separate environments because their dependency stacks conflict. See `pipelines/README.md` for the per-method requirement files and entry points.

## Quick Start

Open the simulator notebook:

```bash
jupyter notebook damage_simulation/demo.ipynb
```

Call the simulator from Python:

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

Sample a reproducible training subset:

```bash
python damage_simulation/sample_dhmurals_train_subset.py --num-samples 1000 --overwrite
```

Replace subset masks with generated VAE masks:

```bash
python damage_simulation/replace_test_masks.py --seed 42
```

Generate white-block inputs for restoration pipelines:

```bash
python pipelines/lama/bin/create_white_block_images.py
```

Run a restoration method:

```bash
python pipelines/deepfill/main.py
python pipelines/powerpaint/process_powerpaint.py --fast
```

The imported VAE notebook now lives at `pipelines/vae/VAE_learn_mask.ipynb` instead of the repository root.
