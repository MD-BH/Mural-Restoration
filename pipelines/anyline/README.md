# Anyline Pipeline

This pipeline keeps the mural-specific Anyline preprocessing scripts and a slim vendored copy of `custom_controlnet_aux` instead of the full ComfyUI workspace from the source repository.

Use:

- `python pipelines/anyline/scripts/process_anyline.py`
- `python pipelines/anyline/scripts/mask_anyline.py`

Both scripts read the shared dataset root from `MURAL_DATA_ROOT` and default to `dataset/DhMurals-inpainting-dataset/test`.
