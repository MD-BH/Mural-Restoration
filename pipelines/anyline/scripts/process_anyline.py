from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from skimage import morphology

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PIPELINE_ROOT / "vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir

from custom_controlnet_aux.lineart_standard import LineartStandardDetector
from custom_controlnet_aux.teed import TEDDetector
from custom_controlnet_aux.teed.ted import TED


PROJECT_RESULT_ROOT = pipeline_result_root(PIPELINE_ROOT)
MODEL_CACHE_DIR = PIPELINE_ROOT / "models" / "Anyline"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def choose_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AnyLineProcessor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.ted_model: TED | None = None

    def load_ted_model(self) -> TED:
        if self.ted_model is not None:
            return self.ted_model

        checkpoint_filename = "MTEED.pth"
        checkpoint_path = MODEL_CACHE_DIR / checkpoint_filename
        if not checkpoint_path.is_file():
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id="TheMistoAI/MistoLine",
                filename=checkpoint_filename,
                subfolder="Anyline",
                local_dir=MODEL_CACHE_DIR,
            )
            checkpoint_path = Path(downloaded_path)

        self.ted_model = TED()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.ted_model.load_state_dict(state_dict)
        self.ted_model.eval()
        self.ted_model.to(self.device)
        return self.ted_model

    @staticmethod
    def get_intensity_mask(image_array: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
        mask = image_array[:, :, 0]
        mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
        return np.expand_dims(mask, 2).repeat(3, axis=2)

    @staticmethod
    def combine_layers(base_layer: np.ndarray, top_layer: np.ndarray) -> np.ndarray:
        mask = top_layer.astype(bool)
        temp = 1 - (1 - top_layer) * (1 - base_layer)
        return base_layer * (~mask) + temp * mask

    def process(self, image_path: Path) -> np.ndarray | None:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"Warning: Could not open {image_path}: {exc}")
            return None

        image_np = np.array(image)
        detector = TEDDetector(model=self.load_ted_model())
        detector.device = self.device

        mteed_np = np.array(detector(image_np, detect_resolution=1280, output_type="pil")).astype(np.float32) / 255.0
        if mteed_np.ndim == 2:
            mteed_np = np.stack([mteed_np] * 3, axis=-1)
        elif mteed_np.shape[2] == 1:
            mteed_np = np.repeat(mteed_np, 3, axis=2)

        lineart_detector = LineartStandardDetector()
        lineart_np = np.array(
            lineart_detector(
                image_np,
                guassian_sigma=2.0,
                intensity_threshold=3,
                detect_resolution=1280,
                output_type="pil",
            )
        ).astype(np.float32) / 255.0
        if lineart_np.ndim == 2:
            lineart_np = np.stack([lineart_np] * 3, axis=-1)
        elif lineart_np.shape[2] == 1:
            lineart_np = np.repeat(lineart_np, 3, axis=2)

        lineart_processed = self.get_intensity_mask(lineart_np, lower_bound=0, upper_bound=1)
        cleaned = morphology.remove_small_objects(lineart_processed.astype(bool), min_size=36, connectivity=1)
        lineart_processed = lineart_processed * cleaned
        final_result = self.combine_layers(mteed_np, lineart_processed)
        return 1.0 - final_result


def process_directory(processor: AnyLineProcessor, input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        print(f"Skipping missing directory: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {input_dir} -> {output_dir}")
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        print(f"  {path.name}")
        try:
            result = processor.process(path)
            if result is None:
                continue
            result_image = (result * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(result_image).save(output_dir / path.name)
        except Exception as exc:
            print(f"Failed to process {path.name}: {exc}")
            traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Anyline guidance images for mural restoration datasets.")
    parser.add_argument("--data-root", type=Path, default=get_realistic_data_root())
    parser.add_argument("--device", type=str, default=None, help="Torch device override")
    parser.add_argument("--damaged-subdir", type=str, default="damaged_albedo")
    parser.add_argument("--white-subdir", type=str, default="white_block_damaged")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    processor = AnyLineProcessor(device=choose_device(args.device))

    damaged_dir = resolve_data_subdir(args.data_root, args.damaged_subdir)
    white_dir = resolve_data_subdir(args.data_root, args.white_subdir)

    process_directory(processor, damaged_dir, PROJECT_RESULT_ROOT / "result-damaged")
    process_directory(processor, white_dir, PROJECT_RESULT_ROOT / "result-white")


if __name__ == "__main__":
    main()
