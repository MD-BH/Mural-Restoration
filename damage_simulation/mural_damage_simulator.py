from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np


PathLike = str | Path
Array = np.ndarray

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DAMAGE_TYPES = ("base_fill", "rough_plaster", "powdered_loss")


def _clip01(array: Array) -> Array:
    """Clamp an array to the [0, 1] interval."""

    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _ensure_float01(array: Array) -> Array:
    """Convert an array to float32 and normalize to [0, 1] when possible."""

    array = np.asarray(array)
    if array.size == 0:
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / max(max_value, 1.0)

    array = array.astype(np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(array.min())
    max_value = float(array.max())
    if min_value >= 0.0 and max_value <= 1.0:
        return array
    if max_value <= 255.0 and min_value >= 0.0:
        return array / 255.0
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32) if max_value <= 0 else np.ones_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def _ensure_rgb_image(image: Array) -> Array:
    """Validate and normalize an RGB image."""

    image = _ensure_float01(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, but got shape {image.shape}.")
    return _clip01(image)


def _normalize_mask(mask: Array) -> Array:
    """Normalize a mask to float32 in [0, 1]."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected an HxW mask, but got shape {mask.shape}.")
    mask = _ensure_float01(mask)
    return _clip01(mask)


def _is_binary_like(mask: Array, tolerance: float = 1e-3) -> bool:
    """Heuristically check whether a mask is nearly binary."""

    mask = _normalize_mask(mask)
    if mask.size == 0:
        return True
    mid_values = np.logical_and(mask > tolerance, mask < 1.0 - tolerance)
    return float(mid_values.mean()) < 0.01


def _resize_mask(mask: Array, target_shape: tuple[int, int]) -> Array:
    """Resize a mask to a target height and width."""

    mask = _normalize_mask(mask)
    target_h, target_w = target_shape
    if mask.shape == (target_h, target_w):
        return mask.astype(np.float32)
    interpolation = cv2.INTER_NEAREST if _is_binary_like(mask) else cv2.INTER_LINEAR
    resized = cv2.resize(mask, (target_w, target_h), interpolation=interpolation)
    return _normalize_mask(resized)


def _ellipse_kernel(radius: int) -> Array:
    """Create an elliptical morphology kernel."""

    radius = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def _smoothstep(edge0: float, edge1: float, x: Array) -> Array:
    """Smoothly map x from edge0-edge1 into [0, 1]."""

    if edge1 <= edge0 + 1e-8:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def _distance_inside(mask: Array) -> Array:
    """Compute normalized inside distance for a binary support mask."""

    binary = (_normalize_mask(mask) > 1e-6).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(binary, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    max_value = float(dist.max())
    if max_value <= 1e-8:
        return np.zeros_like(dist, dtype=np.float32)
    return (dist / max_value).astype(np.float32)


def _alpha_blend(image: Array, layer: Array, alpha: Array) -> Array:
    """Composite a layer over an image using an alpha mask."""

    image = _ensure_rgb_image(image)
    layer = _ensure_rgb_image(layer)
    alpha = _normalize_mask(alpha)
    return _clip01(image * (1.0 - alpha[..., None]) + layer * alpha[..., None])


def _save_rgb_image(path: PathLike, image: Array) -> None:
    """Save an RGB float image to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _ensure_rgb_image(image)
    bgr = cv2.cvtColor((image * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"Failed to save image to {path}")


def _collect_supported_files(directory: PathLike) -> list[Path]:
    """Collect supported image files from a directory."""

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No supported image files were found in {directory}")
    return paths


def load_image(path: PathLike) -> Array:
    """
    Load an RGB image as float32 in [0, 1].

    Alpha channels are composited over white. Grayscale images are replicated
    into three channels.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Unable to read image: {path}")

    if image.ndim == 2:
        image = _ensure_float01(image)
        image = np.repeat(image[..., None], 3, axis=2)
        return _clip01(image)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")

    image = _ensure_float01(image)
    if image.shape[2] == 4:
        alpha = image[..., 3:4]
        rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        image = rgb * alpha + (1.0 - alpha)
        return _clip01(image)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return _clip01(image)
    raise ValueError(f"Unsupported channel count {image.shape[2]} for {path}")


def load_mask(path: PathLike, target_shape: tuple[int, int] | None = None) -> Array:
    """
    Load a single-channel mask as float32 in [0, 1].

    When target_shape is given, the mask is resized to match the image height
    and width.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"Unable to read mask: {path}")

    if mask.ndim == 3:
        mask = _ensure_float01(mask)
        if mask.shape[2] == 4:
            mask = mask[..., :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = _normalize_mask(mask)
    if target_shape is not None:
        mask = _resize_mask(mask, target_shape)
    return mask.astype(np.float32)


def estimate_base_color(image: Array, mask: Array, ring_width: int = 9) -> Array:
    """
    Estimate mural substrate color from pixels around the mask boundary.

    The preferred reference is a ring outside the mask. If that ring is empty,
    the function falls back to all pixels outside the mask, then finally to the
    global image median.
    """

    image = _ensure_rgb_image(image)
    mask = _resize_mask(mask, image.shape[:2])
    binary = (mask > 1e-6).astype(np.uint8)

    if binary.max() == 0:
        return np.median(image.reshape(-1, 3), axis=0).astype(np.float32)

    ring_width = max(1, int(round(ring_width)))
    for multiplier in (1, 2, 4):
        radius = max(1, ring_width * multiplier)
        dilated = cv2.dilate(binary, _ellipse_kernel(radius))
        ring = np.logical_and(dilated > 0, binary == 0)
        if int(ring.sum()) >= 12:
            return np.median(image[ring], axis=0).astype(np.float32)

    outside = binary == 0
    if int(outside.sum()) >= 1:
        return np.median(image[outside], axis=0).astype(np.float32)

    return np.median(image.reshape(-1, 3), axis=0).astype(np.float32)


def build_soft_mask(mask: Array, edge_sharpness: float = 0.7) -> Array:
    """
    Convert a binary or grayscale mask into a soft mask.

    Larger edge_sharpness values keep the boundary crisper. Lower values create
    a wider transition band.
    """

    mask = _normalize_mask(mask)
    edge_sharpness = float(np.clip(edge_sharpness, 0.0, 1.0))
    if float(mask.max()) <= 1e-8:
        return np.zeros_like(mask, dtype=np.float32)

    binary = (mask > max(0.05, 0.5 * float(mask.max()))).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5).astype(np.float32)
    signed_distance = inside - outside

    softness_px = 0.75 + (1.0 - edge_sharpness) * max(2.0, 0.02 * min(mask.shape))
    signed_scaled = np.clip(signed_distance / max(softness_px, 1e-6), -30.0, 30.0)
    soft = 1.0 / (1.0 + np.exp(-signed_scaled))

    if not _is_binary_like(mask):
        soft = 0.5 * soft + 0.5 * mask

    blur_sigma = max(0.0, (1.0 - edge_sharpness) * 1.2)
    if blur_sigma > 0.0:
        soft = cv2.GaussianBlur(soft.astype(np.float32), (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    return _normalize_mask(soft)


def generate_texture_noise(
    shape: tuple[int, int] | Sequence[int],
    noise_strength: float = 1.0,
    noise_scale: float = 32.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    octaves: int = 4,
) -> Array:
    """
    Generate smooth zero-centered texture noise.

    Returns a float32 map roughly in [-noise_strength, noise_strength].
    Larger noise_scale values produce coarser patterns.
    """

    if len(shape) < 2:
        raise ValueError(f"shape must have at least two dimensions, got {shape}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid shape for noise generation: {shape}")

    rng = np.random.default_rng(seed) if rng is None else rng
    noise_strength = max(0.0, float(noise_strength))
    noise_scale = max(2.0, float(noise_scale))
    octaves = max(1, int(octaves))

    field = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    current_scale = noise_scale

    for _ in range(octaves):
        grid_h = max(2, int(math.ceil(height / current_scale)) + 1)
        grid_w = max(2, int(math.ceil(width / current_scale)) + 1)
        grid = rng.random((grid_h, grid_w)).astype(np.float32)
        octave = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        sigma = max(0.0, current_scale * 0.05)
        if sigma > 0.0:
            octave = cv2.GaussianBlur(octave, (0, 0), sigmaX=sigma, sigmaY=sigma)
        field += amplitude * octave
        total_amplitude += amplitude
        amplitude *= 0.5
        current_scale = max(2.0, current_scale / 2.0)

    field /= max(total_amplitude, 1e-8)
    min_value = float(field.min())
    max_value = float(field.max())
    if max_value - min_value < 1e-8:
        centered = np.zeros_like(field, dtype=np.float32)
    else:
        centered = ((field - min_value) / (max_value - min_value) - 0.5) * 2.0
    return (centered * noise_strength).astype(np.float32)


def apply_damage(
    image: Array,
    mask: Array,
    damage_type: str,
    noise_strength: float = 0.35,
    noise_scale: float = 32.0,
    opacity: float = 1.0,
    edge_sharpness: float = 0.7,
    seed: int | None = None,
    ring_width: int = 9,
) -> tuple[Array, dict[str, Any]]:
    """
    Simulate mural damage on an intrinsic-color mural image.

    Parameters
    ----------
    image:
        RGB image in HxWx3 format, uint8 or float.
    mask:
        Single-channel damage mask. Nonzero regions indicate the damaged area.
    damage_type:
        One of `base_fill`, `rough_plaster`, `powdered_loss`.
    noise_strength:
        Controls texture intensity.
    noise_scale:
        Controls texture granularity. Larger values give coarser variation.
    opacity:
        Blend strength for the damaged layer.
    edge_sharpness:
        Controls mask softness. Larger means sharper edges.
    seed:
        Random seed for reproducibility.
    ring_width:
        Width of the outer reference ring used to estimate the base color.
    """

    damage_type = str(damage_type)
    if damage_type not in DAMAGE_TYPES:
        raise ValueError(f"Unsupported damage_type '{damage_type}'. Valid values: {DAMAGE_TYPES}")

    image = _ensure_rgb_image(image)
    mask = _resize_mask(mask, image.shape[:2])
    opacity = float(np.clip(opacity, 0.0, 1.0))
    noise_strength = max(0.0, float(noise_strength))
    noise_scale = max(2.0, float(noise_scale))
    edge_sharpness = float(np.clip(edge_sharpness, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    if float(mask.max()) <= 1e-8 or opacity <= 1e-8:
        empty_result = image.copy()
        return empty_result, {
            "status": "empty_mask_or_zero_opacity",
            "damage_type": damage_type,
            "base_color": np.median(image.reshape(-1, 3), axis=0).astype(np.float32),
            "soft_mask": np.zeros(image.shape[:2], dtype=np.float32),
            "effective_alpha": np.zeros(image.shape[:2], dtype=np.float32),
        }

    soft_mask = build_soft_mask(mask, edge_sharpness=edge_sharpness)
    if float(soft_mask.max()) <= 1e-8:
        return image.copy(), {
            "status": "empty_soft_mask",
            "damage_type": damage_type,
            "base_color": np.median(image.reshape(-1, 3), axis=0).astype(np.float32),
            "soft_mask": soft_mask,
            "effective_alpha": np.zeros(image.shape[:2], dtype=np.float32),
        }

    base_color = estimate_base_color(image, mask, ring_width=ring_width)
    base_layer = np.ones_like(image, dtype=np.float32) * base_color[None, None, :]
    inside_distance = _distance_inside(soft_mask)
    edge_factor = (1.0 - inside_distance).astype(np.float32)

    coarse = generate_texture_noise(mask.shape, noise_strength=1.0, noise_scale=noise_scale, rng=rng, octaves=4)
    mid = generate_texture_noise(mask.shape, noise_strength=1.0, noise_scale=max(2.0, noise_scale / 2.2), rng=rng, octaves=4)
    fine = generate_texture_noise(mask.shape, noise_strength=1.0, noise_scale=max(2.0, noise_scale / 5.0), rng=rng, octaves=3)

    off_white = np.array([0.93, 0.91, 0.86], dtype=np.float32)

    if damage_type == "base_fill":
        variation = noise_strength * (0.07 * coarse + 0.03 * fine)
        layer = _clip01(base_layer + variation[..., None] * np.array([1.0, 0.9, 0.8], dtype=np.float32))
        alpha = _clip01(soft_mask * opacity)

    elif damage_type == "rough_plaster":
        warm_tint = np.array([1.04, 1.00, 0.96], dtype=np.float32)
        plaster = base_layer * warm_tint[None, None, :]
        brightness = noise_strength * (0.10 * coarse + 0.06 * mid)
        pores = np.clip(-(0.35 * mid + 0.65 * fine), 0.0, 1.0)
        highlights = np.clip(0.65 * coarse + 0.35 * fine, 0.0, 1.0)
        plaster = plaster + brightness[..., None]
        plaster = plaster * (1.0 - 0.18 * noise_strength * pores[..., None])
        plaster = plaster + 0.05 * noise_strength * highlights[..., None] * off_white[None, None, :]
        plaster = plaster * (1.0 - 0.04 * noise_strength * edge_factor[..., None])
        layer = _clip01(plaster)
        alpha = _clip01(soft_mask * opacity)

    elif damage_type == "powdered_loss":
        powder_density = _clip01(0.55 + 0.40 * coarse + 0.25 * np.clip(fine, 0.0, 1.0))
        substrate = _clip01(base_layer + 0.05 * noise_strength * (0.7 * coarse + 0.3 * mid)[..., None])
        chalk = _clip01(substrate * 0.82 + off_white[None, None, :] * 0.18)
        powder = _clip01(chalk + 0.08 * noise_strength * np.clip(fine, 0.0, 1.0)[..., None])
        mix = 0.30 + 0.45 * powder_density
        layer = _clip01(substrate * (1.0 - mix[..., None]) + powder * mix[..., None])
        alpha = _clip01(soft_mask * opacity * (0.55 + 0.45 * powder_density * (0.65 + 0.35 * edge_factor)))

    else:
        substrate = _clip01(base_layer * np.array([1.03, 1.00, 0.95], dtype=np.float32)[None, None, :])
        substrate = _clip01(substrate + 0.08 * noise_strength * (0.6 * coarse + 0.4 * fine)[..., None])
        gray = np.mean(image, axis=2, keepdims=True)
        worn_original = _clip01(image * 0.82 + gray * 0.18)
        worn_original = _clip01(worn_original * (1.0 - 0.10 * edge_factor[..., None]))
        loss_driver = _clip01(0.55 * ((coarse + 1.0) * 0.5) + 0.35 * edge_factor + 0.10 * ((mid + 1.0) * 0.5))
        transition = _smoothstep(0.35 - 0.20 * noise_strength, 0.78, loss_driver + 0.12 * fine)
        layer = _clip01(worn_original * (1.0 - transition[..., None]) + substrate * transition[..., None])
        alpha = _clip01(soft_mask * opacity * (0.65 + 0.35 * transition))

    result = _alpha_blend(image, layer, alpha)
    metadata: dict[str, Any] = {
        "status": "ok",
        "damage_type": damage_type,
        "base_color": base_color.astype(np.float32),
        "soft_mask": soft_mask.astype(np.float32),
        "effective_alpha": alpha.astype(np.float32),
        "texture_noise": coarse.astype(np.float32),
        "base_layer": layer.astype(np.float32),
        "nonzero_mask_pixels": int((mask > 1e-6).sum()),
        "seed": seed,
        "noise_strength": noise_strength,
        "noise_scale": noise_scale,
        "opacity": opacity,
        "edge_sharpness": edge_sharpness,
        "ring_width": int(ring_width),
    }
    return result.astype(np.float32), metadata


def batch_apply_damage(
    images_dir: PathLike,
    masks_dir: PathLike,
    output_dir: PathLike,
    n_samples_per_image: int = 2,
    damage_types: Sequence[str] | None = None,
    noise_strength_range: tuple[float, float] = (0.15, 0.85),
    noise_scale_range: tuple[float, float] = (12.0, 64.0),
    opacity_range: tuple[float, float] = (0.65, 1.0),
    edge_sharpness_range: tuple[float, float] = (0.35, 0.95),
    seed: int | None = None,
    sample_with_replacement: bool = True,
    ring_width: int = 9,
) -> list[dict[str, Any]]:
    """
    Batch-apply randomly sampled damage masks and damage types to images.

    Metadata is saved to `metadata.json` and `metadata.csv` in output_dir.
    """

    if n_samples_per_image <= 0:
        raise ValueError("n_samples_per_image must be a positive integer.")

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_supported_files(images_dir)
    mask_paths = _collect_supported_files(masks_dir)
    valid_damage_types = list(damage_types or DAMAGE_TYPES)
    valid_damage_types = [damage_type for damage_type in valid_damage_types if damage_type in DAMAGE_TYPES]
    if not valid_damage_types:
        raise ValueError(f"damage_types must contain at least one valid type from {DAMAGE_TYPES}")

    rng = np.random.default_rng(seed)
    combinations = [(mask_path, damage_type) for mask_path in mask_paths for damage_type in valid_damage_types]
    records: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = load_image(image_path)
        replace = sample_with_replacement or n_samples_per_image > len(combinations)
        selected_indices = rng.choice(len(combinations), size=n_samples_per_image, replace=replace)
        for sample_idx, combo_index in enumerate(np.atleast_1d(selected_indices)):
            mask_path, damage_type = combinations[int(combo_index)]
            local_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            mask = load_mask(mask_path, target_shape=image.shape[:2])

            noise_strength = float(rng.uniform(*sorted(noise_strength_range)))
            noise_scale = float(rng.uniform(*sorted(noise_scale_range)))
            opacity = float(rng.uniform(*sorted(opacity_range)))
            edge_sharpness = float(rng.uniform(*sorted(edge_sharpness_range)))

            result, info = apply_damage(
                image=image,
                mask=mask,
                damage_type=damage_type,
                noise_strength=noise_strength,
                noise_scale=noise_scale,
                opacity=opacity,
                edge_sharpness=edge_sharpness,
                seed=local_seed,
                ring_width=ring_width,
            )

            output_name = (
                f"{image_path.stem}__mask-{mask_path.stem}__mode-{damage_type}"
                f"__idx-{sample_idx:03d}__seed-{local_seed}.png"
            )
            output_path = output_dir / output_name
            _save_rgb_image(output_path, result)

            base_color = np.asarray(info["base_color"], dtype=np.float32)
            record = {
                "source_image_path": str(image_path),
                "mask_path": str(mask_path),
                "output_path": str(output_path),
                "damage_type": damage_type,
                "seed": local_seed,
                "noise_strength": noise_strength,
                "noise_scale": noise_scale,
                "opacity": opacity,
                "edge_sharpness": edge_sharpness,
                "ring_width": int(ring_width),
                "status": str(info["status"]),
                "nonzero_mask_pixels": int(info.get("nonzero_mask_pixels", 0)),
                "base_color_r": float(base_color[0]),
                "base_color_g": float(base_color[1]),
                "base_color_b": float(base_color[2]),
            }
            records.append(record)

    json_path = output_dir / "metadata.json"
    csv_path = output_dir / "metadata.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return records


def show_comparison(
    image: Array,
    mask: Array,
    results: Mapping[str, Array] | Sequence[tuple[str, Array]],
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Visualize the original image, mask, and one or more damaged results.

    Parameters
    ----------
    image:
        Original RGB image.
    mask:
        Damage mask.
    results:
        Either a mapping from title to image or a sequence of (title, image).
    """

    image = _ensure_rgb_image(image)
    mask = _resize_mask(mask, image.shape[:2])

    if isinstance(results, Mapping):
        items = list(results.items())
    else:
        items = list(results)

    ncols = 2 + len(items)
    if figsize is None:
        figsize = (4.2 * ncols, 4.5)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = np.array([axes])

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    for axis, (title, result) in zip(axes[2:], items):
        axis.imshow(_ensure_rgb_image(result))
        axis.set_title(str(title))
        axis.axis("off")

    fig.tight_layout()
    return fig, np.asarray(axes)


__all__ = [
    "DAMAGE_TYPES",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "apply_damage",
    "batch_apply_damage",
    "build_soft_mask",
    "estimate_base_color",
    "generate_texture_noise",
    "load_image",
    "load_mask",
    "show_comparison",
]
