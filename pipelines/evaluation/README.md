# Evaluation Scripts

Use `evaluate_image_metrics.py` for metric tables and `create_compare_png.py` for visual four-panel comparisons against the standard pipeline outputs.

Examples:

```bash
python pipelines/evaluation/scripts/evaluate_image_metrics.py \
  --reference-dir dataset/DhMurals-inpainting-dataset/test/images \
  --test-dirs pipelines/deepfill/result/result-damaged pipelines/powerpaint/result/result-damaged \
  --output-dir pipelines/evaluation/result/metrics
```

```bash
python pipelines/evaluation/scripts/create_compare_png.py
```
