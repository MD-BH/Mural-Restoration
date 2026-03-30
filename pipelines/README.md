# Restoration Pipelines

The imported restoration methods now live under `pipelines/` so the repository keeps one root structure while still allowing per-method environments.

Common conventions:

- Shared inference/evaluation data defaults to `dataset/DhMurals-inpainting-dataset/test`.
- Set `MURAL_DATA_ROOT=/path/to/compatible/data/root` to override that location.
- The expected dataset layout is `images/`, `masks/`, `edges/`, `damaged_albedo/`, and optional `white_block_damaged/`.
- Each method writes generated outputs under its own local `result/` directory, which is ignored by Git.

Entry points:

- `pipelines/anyline/scripts/process_anyline.py`: generate Anyline guidance for damaged and white-block inputs.
- `pipelines/anyline/scripts/mask_anyline.py`: apply binary masks to Anyline outputs.
- `pipelines/deepfill/main.py`: run the DeepFill mural restoration wrapper.
- `pipelines/lama/bin/create_white_block_images.py`: generate white-block inputs from the shared dataset.
- `pipelines/lama/bin/prepare_inference_data.py`: build LaMa-style inference inputs.
- `pipelines/lama/gen_mural_samples.py`: run LaMa prediction and copy a small comparison sample set.
- `pipelines/muralnet/main.py`: run MuralNet with defaults aligned to the shared dataset layout.
- `pipelines/powerpaint/process_powerpaint.py`: batch-run PowerPaint on the damaged and white-block sets.
- `pipelines/evaluation/scripts/evaluate_image_metrics.py`: compute SSIM/PSNR/MSE/LPIPS/FID for one or more result folders.
- `pipelines/evaluation/scripts/create_compare_png.py`: create side-by-side comparison PNGs from the standard pipeline outputs.
- `pipelines/vae/VAE_learn_mask.ipynb`: mask-generation notebook adapted to the merged repository layout.

Environment files:

- `pipelines/deepfill/requirements.txt`
- `pipelines/lama/requirements.txt`
- `pipelines/muralnet/requirements.txt`
- `pipelines/powerpaint/requirements/requirements.txt`
- `pipelines/anyline/requirements.txt`
- `pipelines/evaluation/requirements.txt`
