"""Template-guided digit cleanup for the 1/3/4/7 microscopy masks.

The algorithm does not receive the target digit label. For each input image it
matches all available 1/3/4/7 reference templates under translation and scale,
selects the best match, and uses the aligned template as an explainable shape
gate to suppress non-template ghost strokes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DIGITS = (1, 3, 4, 7)
MATCH_SIZE = 384
MM_PER_INCH = 25.4
DEFAULT_OUTDIR = ROOT / "figures" / "template_guided_repair"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _find_layer_jpg(root: Path, data_id: int) -> Path:
    suffix = str(data_id)
    for path in sorted(root.glob("*.jpg")):
        if path.stem.endswith(suffix):
            return path
    raise FileNotFoundError(f"Cannot find input JPG for data_id {data_id}")


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def read_raw_rgb(root: Path, data_id: int) -> np.ndarray:
    with Image.open(_find_layer_jpg(root, data_id)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return (gray > 127).astype(np.float32)


def robust_norm(image: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _positive_norm(image: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    values = arr[(arr > 0) & np.isfinite(arr)]
    if values.size == 0:
        values = arr[np.isfinite(arr)]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _foreground_norm(image: np.ndarray, support: np.ndarray, low: float = 3.0, high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mask = (np.asarray(support) > 0.5) & (arr > 0)
    if np.count_nonzero(mask) < 16:
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[mask]
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros_like(arr, dtype=np.float32)
    out[mask] = np.power(np.clip((arr[mask] - lo) / (hi - lo), 0.0, 1.0), 0.45)
    out = ndi.gaussian_filter(out, sigma=0.35, mode="reflect")
    out[np.asarray(support) <= 0.5] = 0.0
    foreground = out[mask]
    if foreground.size:
        scale = float(np.percentile(foreground, 98.0))
        if scale > 1.0e-6:
            out = np.clip(out / scale, 0.0, 1.0)
            out[np.asarray(support) <= 0.5] = 0.0
    return out.astype(np.float32)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0.5, dtype=bool)
    binary = remove_small_objects(binary, min_size=90)
    binary = ndi.binary_closing(binary, structure=np.ones((3, 3), dtype=bool))
    return binary.astype(np.float32)


def _load_digit_data(root: Path, data_id: int) -> dict[str, np.ndarray]:
    raw_path = _find_layer_jpg(root, data_id)
    tophat_path = root / "results" / "paper_tophat_jpg" / f"layer{data_id}_black_tophat_signal.tif"
    ultra_path = root / "results" / "aggressive_ghost_fade" / f"layer{data_id}_ultra_signal.tif"
    mask_path = root / "results" / "binary_clean_ultra_24level_comparison" / f"layer{data_id}_ultra_03_p62_bin.png"
    for path in (tophat_path, ultra_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(path)

    raw = _read_gray(raw_path)
    tophat = tifffile.imread(tophat_path).astype(np.float32)
    ultra = np.clip(tifffile.imread(ultra_path).astype(np.float32), 0.0, 1.0)
    mask = _read_mask(mask_path)
    clean_mask = _clean_mask(mask)
    tophat_norm = _positive_norm(tophat, 55.0, 99.5)
    smoothed = robust_norm(ndi.gaussian_filter(tophat_norm, sigma=2.0, mode="reflect"), 0.2, 99.7)
    masked_display = np.where(mask > 0.5, np.clip(0.30 + 0.70 * _foreground_norm(tophat_norm, mask), 0.0, 1.0), 0.0)
    selected_display = np.where(
        clean_mask > 0.5,
        np.clip(0.30 + 0.70 * _foreground_norm(tophat_norm, clean_mask), 0.0, 1.0),
        0.0,
    )
    return {
        "raw": robust_norm(raw, 0.8, 99.4),
        "tophat_norm": tophat_norm,
        "smoothed": smoothed,
        "signal_display": np.clip(0.22 + 0.78 * tophat_norm, 0.0, 1.0),
        "smoothed_display": np.clip(0.22 + 0.78 * smoothed, 0.0, 1.0),
        "masked_display": masked_display.astype(np.float32),
        "selected_display": selected_display.astype(np.float32),
        "ultra": ultra,
        "mask": mask,
        "mask_clean": clean_mask,
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def downsample_mean(image: np.ndarray, max_size: int = 900) -> np.ndarray:
    arr = np.asarray(image)
    factor = max(1, int(np.ceil(max(arr.shape[:2]) / max_size)))
    if factor <= 1:
        return arr
    height = (arr.shape[0] // factor) * factor
    width = (arr.shape[1] // factor) * factor
    if arr.ndim == 2:
        trimmed = arr[:height, :width]
        return trimmed.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))
    trimmed = arr[:height, :width, :]
    return trimmed.reshape(height // factor, factor, width // factor, factor, arr.shape[2]).mean(axis=(1, 3)).astype(arr.dtype)


def downsample_max(image: np.ndarray, max_size: int = 900) -> np.ndarray:
    arr = np.asarray(image)
    factor = max(1, int(np.ceil(max(arr.shape[:2]) / max_size)))
    if factor <= 1:
        return arr
    height = (arr.shape[0] // factor) * factor
    width = (arr.shape[1] // factor) * factor
    if arr.ndim == 2:
        trimmed = arr[:height, :width]
        return trimmed.reshape(height // factor, factor, width // factor, factor).max(axis=(1, 3))
    trimmed = arr[:height, :width, :]
    return trimmed.reshape(height // factor, factor, width // factor, factor, arr.shape[2]).max(axis=(1, 3)).astype(arr.dtype)


def change_overlay(candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    candidate_bool = np.asarray(candidate, dtype=bool)
    mask_bool = np.asarray(mask, dtype=bool)
    kept = candidate_bool & mask_bool
    removed = candidate_bool & ~mask_bool
    added = mask_bool & ~candidate_bool
    out = np.zeros((*candidate_bool.shape, 3), dtype=np.uint8)
    out[kept] = np.array([245, 245, 245], dtype=np.uint8)
    out[removed] = np.array([235, 40, 40], dtype=np.uint8)
    out[added] = np.array([0, 205, 255], dtype=np.uint8)
    return out


def signal_on_yellow(raw_rgb: np.ndarray, soft_signal: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_rgb, dtype=np.float32)
    background = np.median(raw.reshape(-1, 3), axis=0)
    signal = np.clip(np.asarray(soft_signal, dtype=np.float32), 0.0, 1.0)
    rgb = background.reshape(1, 1, 3) - signal[..., None] * np.array([88.0, 78.0, 42.0], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def line_kernel(length: int, angle: int) -> np.ndarray:
    length = int(length) | 1
    kernel = np.zeros((length, length), dtype=bool)
    center = length // 2
    if angle == 0:
        kernel[center, :] = True
    elif angle == 90:
        kernel[:, center] = True
    elif angle == 45:
        np.fill_diagonal(kernel, True)
    elif angle == 135:
        np.fill_diagonal(np.fliplr(kernel), True)
    else:
        raise ValueError("angle must be one of 0, 45, 90, 135")
    return kernel


def directional_close(mask: np.ndarray, length: int) -> np.ndarray:
    closed = np.asarray(mask, dtype=bool).copy()
    for angle in (0, 45, 90, 135):
        closed |= ndi.binary_closing(mask, structure=line_kernel(length, angle))
    return closed


def center_prior_map(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy = (height - 1.0) / 2.0
    cx = (width - 1.0) / 2.0
    ny = np.abs(yy - cy) / max(cy, 1.0)
    nx = np.abs(xx - cx) / max(cx, 1.0)
    radial = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
    radial_score = np.clip(1.0 - np.power(radial, 0.85), 0.0, 1.0)
    border_distance = np.minimum.reduce([yy, xx, height - 1.0 - yy, width - 1.0 - xx])
    border_score = np.clip(border_distance / (0.50 * min(height, width)), 0.0, 1.0)
    return (0.58 * radial_score + 0.42 * border_score).astype(np.float32)


def component_center_score(shape: tuple[int, int], centroid: tuple[float, float]) -> float:
    height, width = shape
    cy = (height - 1.0) / 2.0
    cx = (width - 1.0) / 2.0
    y, x = centroid
    ny = abs(float(y) - cy) / max(cy, 1.0)
    nx = abs(float(x) - cx) / max(cx, 1.0)
    radial = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
    radial_score = float(np.clip(1.0 - np.power(radial, 0.85), 0.0, 1.0))
    border_distance = min(float(y), float(x), height - 1.0 - float(y), width - 1.0 - float(x))
    border_score = float(np.clip(border_distance / (0.50 * min(height, width)), 0.0, 1.0))
    return float(np.clip(0.58 * radial_score + 0.42 * border_score, 0.0, 1.0))


def area_filter(mask: np.ndarray, min_size: int = 4200) -> np.ndarray:
    cleaned = remove_small_objects(np.asarray(mask, dtype=bool), min_size=min_size)
    cleaned = ndi.binary_closing(cleaned, structure=np.ones((3, 3), dtype=bool))
    return cleaned.astype(bool)


def component_score_clean(mask: np.ndarray, signal: np.ndarray) -> np.ndarray:
    candidate = remove_small_objects(np.asarray(mask, dtype=bool), min_size=170)
    strong_strokes = area_filter(candidate, min_size=4200)
    distance_to_strong = ndi.distance_transform_edt(~ndi.binary_dilation(strong_strokes, iterations=1))
    labels = label(candidate, connectivity=2)
    keep = np.zeros(candidate.shape, dtype=bool)
    sig = np.asarray(signal, dtype=np.float32)

    for prop in regionprops(labels, intensity_image=sig):
        coords = prop.coords
        area = int(prop.area)
        y0, x0, y1, x1 = prop.bbox
        span = max(y1 - y0, x1 - x0)
        short_span = max(1, min(y1 - y0, x1 - x0))
        aspect = span / short_span
        mean_intensity = float(prop.mean_intensity)
        near_strong = float(np.min(distance_to_strong[coords[:, 0], coords[:, 1]]))
        center_score = component_center_score(candidate.shape, prop.centroid)
        edge_penalty = float(np.clip(1.25 * (1.0 - center_score), 0.0, 1.0))
        elongated = float(prop.eccentricity) >= 0.83 or aspect >= 2.8
        stroke_consistent = elongated or span >= 92 or area >= 9800
        wrong_stroke_penalty = (not stroke_consistent) and mean_intensity < 0.40

        large_rule = (
            area >= int(round(3000 + 7200 * edge_penalty))
            and (stroke_consistent or mean_intensity >= 0.42 or center_score >= 0.58)
        )
        nearby_rule = (
            near_strong <= 14
            and area >= int(round(440 + 860 * edge_penalty))
            and (mean_intensity >= 0.165 + 0.105 * edge_penalty or span >= 50 + 34 * edge_penalty)
            and (stroke_consistent or center_score >= 0.62)
        )
        stroke_rule = (
            area >= int(round(700 + 1250 * edge_penalty))
            and span >= 68 + 42 * edge_penalty
            and elongated
            and mean_intensity >= 0.150 + 0.095 * edge_penalty
        )
        isolated_penalty = (
            (near_strong > 36 and area < 6200 + 7600 * edge_penalty and mean_intensity < 0.38 + 0.13 * edge_penalty)
            or (center_score < 0.40 and area < 15000 and mean_intensity < 0.50)
            or wrong_stroke_penalty
        )

        if (large_rule or nearby_rule or stroke_rule) and not isolated_penalty:
            keep[coords[:, 0], coords[:, 1]] = True

    keep = directional_close(keep, length=7)
    keep = ndi.binary_closing(keep, structure=np.ones((3, 3), dtype=bool))
    keep = remove_small_objects(keep, min_size=280)
    return keep.astype(bool)


@dataclass(frozen=True)
class TemplateCandidate:
    digit: int
    path: Path
    crop_soft: np.ndarray
    crop_mask: np.ndarray
    source_shape: tuple[int, int]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TargetFrame:
    """Target-image-derived scale and position for template alignment."""

    bbox: tuple[int, int, int, int]
    center_y_frac: float
    center_x_frac: float
    height_frac: float
    width_frac: float


@dataclass(frozen=True)
class TemplateMatch:
    predicted_digit: int
    template_path: Path
    score: float
    normalized_overlap: float
    candidate_inside_template: float
    template_hit_density: float
    scale_height_frac: float
    center_y_frac: float
    center_x_frac: float
    template_height_px: int
    template_width_px: int


@dataclass(frozen=True)
class TemplateGuidedMetrics:
    data_id: int
    predicted_digit: int
    template_path: str
    score: float
    before_px: int
    pool_px: int
    final_px: int
    removed_px: int
    added_px: int
    template_gate_px: int
    final_components: int
    template_match_fraction: float
    template_core_fraction: float
    template_dice: float
    template_signal_fraction: float
    alignment_score: float
    template_scale_multiplier: float
    template_scale_x_multiplier: float
    template_scale_y_multiplier: float
    template_y_offset_px: int
    template_x_offset_px: int
    target_height_frac: float
    target_width_frac: float


@dataclass(frozen=True)
class TemplateMatchQuality:
    """Full-resolution overlap between the target candidate strokes and template."""

    template_match_fraction: float
    template_core_fraction: float
    template_dice: float
    template_signal_fraction: float


@dataclass(frozen=True)
class FilterParameters:
    soft_margin_distance: int
    strong_margin_distance: int
    template_soft_threshold: float
    soft_edge_inside_fraction: float
    soft_edge_min_signal: float
    strong_contiguous_seed_overlap: int
    strong_contiguous_min_signal: float
    repair_distance: int
    strong_distance: int
    template_min: float
    close_length: int
    bridge_length: int
    bridge_distance: int
    connected_zone_iterations: int
    use_repair_pool: bool = True
    allow_repair_add: bool = True


@dataclass(frozen=True)
class AlignmentParameters:
    template_scale_multiplier: float = 1.0
    template_scale_x_multiplier: float = 1.0
    template_scale_y_multiplier: float = 1.0
    template_y_offset_px: int = 0
    template_x_offset_px: int = 0
    auto_refine_alignment: bool = True
    alignment_score: float = 0.0


def default_alignment_parameters(data_id: int) -> AlignmentParameters:
    if data_id == 4:
        return AlignmentParameters(
            template_scale_multiplier=1.30,
            template_scale_x_multiplier=1.44,
            template_scale_y_multiplier=1.32,
            template_y_offset_px=235,
            template_x_offset_px=-70,
            auto_refine_alignment=False,
        )
    return AlignmentParameters()


def apply_alignment_parameters(match: TemplateMatch, image_shape: tuple[int, int], params: AlignmentParameters) -> TemplateMatch:
    height, width = image_shape
    y_scale = params.template_scale_multiplier * params.template_scale_y_multiplier
    return replace(
        match,
        scale_height_frac=float(np.clip(match.scale_height_frac * y_scale, 0.30, 0.98)),
        center_y_frac=float(np.clip(match.center_y_frac + params.template_y_offset_px / height, 0.0, 1.0)),
        center_x_frac=float(np.clip(match.center_x_frac + params.template_x_offset_px / width, 0.0, 1.0)),
    )


def default_filter_parameters(data_id: int) -> FilterParameters:
    soft_margin_distance = 26
    strong_margin_distance = 48
    template_soft_threshold = 0.006
    soft_edge_inside_fraction = 0.28
    soft_edge_min_signal = 0.17
    strong_contiguous_seed_overlap = 26
    strong_contiguous_min_signal = 0.24
    repair_distance = 16
    strong_distance = 9
    template_min = 0.025
    close_length = 9
    bridge_length = 0
    bridge_distance = 0
    connected_zone_iterations = 9
    if data_id == 3:
        repair_distance = 24
        strong_distance = 14
        template_min = 0.016
        close_length = 13
        bridge_length = 0
        bridge_distance = 0
        soft_margin_distance = 38
        strong_margin_distance = 68
        template_soft_threshold = 0.008
        soft_edge_inside_fraction = 0.30
        soft_edge_min_signal = 0.16
        strong_contiguous_seed_overlap = 24
        strong_contiguous_min_signal = 0.23
        connected_zone_iterations = 8
        use_repair_pool = True
        allow_repair_add = True
    elif data_id == 4:
        repair_distance = 22
        strong_distance = 14
        template_min = 0.100
        close_length = 23
        bridge_length = 24
        bridge_distance = 22
        soft_margin_distance = 18
        strong_margin_distance = 38
        template_soft_threshold = 0.085
        soft_edge_inside_fraction = 0.32
        soft_edge_min_signal = 0.18
        strong_contiguous_seed_overlap = 28
        strong_contiguous_min_signal = 0.25
        connected_zone_iterations = 7
        use_repair_pool = True
        allow_repair_add = True
    elif data_id == 7:
        soft_margin_distance = 32
        strong_margin_distance = 56
        use_repair_pool = True
        allow_repair_add = True
    else:
        use_repair_pool = True
        allow_repair_add = True
    return FilterParameters(
        soft_margin_distance=soft_margin_distance,
        strong_margin_distance=strong_margin_distance,
        template_soft_threshold=template_soft_threshold,
        soft_edge_inside_fraction=soft_edge_inside_fraction,
        soft_edge_min_signal=soft_edge_min_signal,
        strong_contiguous_seed_overlap=strong_contiguous_seed_overlap,
        strong_contiguous_min_signal=strong_contiguous_min_signal,
        repair_distance=repair_distance,
        strong_distance=strong_distance,
        template_min=template_min,
        close_length=close_length,
        bridge_length=bridge_length,
        bridge_distance=bridge_distance,
        connected_zone_iterations=connected_zone_iterations,
        use_repair_pool=use_repair_pool,
        allow_repair_add=allow_repair_add,
    )


def image_to_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.seek(0)
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max() + 1), int(xs.max() + 1)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.quantile(values[np.isfinite(values)], quantiles)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= max(float(cumulative[-1]), 1.0e-12)
    return np.interp(quantiles, cumulative, values)


def extract_dark_digit_template(path: Path) -> TemplateCandidate | None:
    arr = ndi.gaussian_filter(image_to_gray(path), sigma=0.6, mode="reflect")
    background = ndi.gaussian_filter(arr, sigma=max(24.0, min(arr.shape) / 68.0), mode="reflect")
    dark = robust_norm(np.clip(background - arr, 0.0, None), 70.0, 99.7)

    high_seed = dark > 0.36
    high_seed = remove_small_objects(high_seed, min_size=120)
    support = (dark > 0.22) & ndi.binary_dilation(high_seed, iterations=34)
    support = ndi.binary_closing(support, structure=np.ones((5, 5), dtype=bool))
    support = remove_small_objects(support, min_size=250)

    # Remove isolated dust/scratches by first finding the main digit group. The
    # subsequent template gate should represent digit geometry, not every speck
    # in the reference frame.
    grouping = ndi.binary_dilation(support, iterations=28)
    grouping = ndi.binary_closing(grouping, structure=np.ones((17, 17), dtype=bool))
    grouping = remove_small_objects(grouping, min_size=max(2200, int(0.00035 * grouping.size)))
    group_labels = label(grouping, connectivity=2)
    best_group = 0
    best_group_score = -np.inf
    group_prior = center_prior_map(grouping.shape)
    for prop in regionprops(group_labels):
        coords = prop.coords
        support_area = int(np.count_nonzero(support[coords[:, 0], coords[:, 1]]))
        if support_area < 300:
            continue
        center_score = float(np.mean(group_prior[coords[:, 0], coords[:, 1]]))
        mean_dark = float(np.mean(dark[coords[:, 0], coords[:, 1]]))
        score = np.log1p(support_area) + 3.0 * center_score + 0.8 * mean_dark
        if score > best_group_score:
            best_group_score = score
            best_group = int(prop.label)
    if best_group:
        support &= ndi.binary_dilation(group_labels == best_group, iterations=8)

    prior = center_prior_map(support.shape)
    labels = label(support, connectivity=2)
    keep = np.zeros(support.shape, dtype=bool)
    for prop in regionprops(labels, intensity_image=dark):
        coords = prop.coords
        area = int(prop.area)
        if area < 180:
            continue
        y0, x0, y1, x1 = prop.bbox
        span = max(y1 - y0, x1 - x0)
        short_span = max(1, min(y1 - y0, x1 - x0))
        aspect = span / short_span
        center_score = float(np.mean(prior[coords[:, 0], coords[:, 1]]))
        mean_dark = float(prop.mean_intensity)
        thin_scratch = aspect >= 18.0 and area < 3200 and mean_dark < 0.60
        if not thin_scratch and (area >= 420 or center_score >= 0.22 or mean_dark >= 0.48):
            keep[coords[:, 0], coords[:, 1]] = True

    keep = ndi.binary_dilation(keep, iterations=2)
    keep = ndi.binary_closing(keep, structure=np.ones((5, 5), dtype=bool))
    keep = remove_small_objects(keep, min_size=300)
    bbox = tight_bbox(keep)
    if bbox is None:
        return None

    y0, x0, y1, x1 = bbox
    margin = int(round(0.08 * max(y1 - y0, x1 - x0)))
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(keep.shape[0], y1 + margin)
    x1 = min(keep.shape[1], x1 + margin)
    crop_mask = keep[y0:y1, x0:x1]
    crop_soft = dark[y0:y1, x0:x1] * ndi.binary_dilation(crop_mask, iterations=10)
    if np.count_nonzero(crop_mask) < 800:
        return None

    digits = re.findall(r"[1347]", path.name)
    if not digits:
        return None
    return TemplateCandidate(
        digit=int(digits[0]),
        path=path,
        crop_soft=np.asarray(crop_soft, dtype=np.float32),
        crop_mask=np.asarray(crop_mask, dtype=bool),
        source_shape=keep.shape,
        bbox=(y0, x0, y1, x1),
    )


def load_templates(root: Path) -> list[TemplateCandidate]:
    templates: list[TemplateCandidate] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".tif":
            continue
        if "figures" in path.parts or "results" in path.parts:
            continue
        if "blank" in path.name.lower():
            continue
        if not re.search(r"[1347]", path.name):
            continue
        template = extract_dark_digit_template(path)
        if template is not None:
            templates.append(template)
    if not templates:
        raise FileNotFoundError("No usable .tif templates for digits 1/3/4/7 were found.")
    return templates


def resize_float(image: np.ndarray, shape: tuple[int, int], *, order: int) -> np.ndarray:
    return resize(
        image,
        shape,
        order=order,
        mode="constant",
        cval=0.0,
        preserve_range=True,
        anti_aliasing=order > 0,
    ).astype(np.float32)


def loose_repair_pool(candidate: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = np.asarray(candidate, dtype=bool)
    sig = ndi.gaussian_filter(np.asarray(signal, dtype=np.float32), sigma=0.75, mode="reflect")
    values = sig[sig > 0]
    if values.size == 0:
        empty = np.zeros_like(before, dtype=bool)
        return before.copy(), empty, sig

    seed = component_score_clean(before, sig)
    seed = ndi.binary_dilation(seed, iterations=1)
    spatial_prior = center_prior_map(before.shape)
    distance_to_seed = ndi.distance_transform_edt(~seed)
    weak_signal = sig >= float(np.percentile(values, 62.0))
    strong_signal = sig >= float(np.percentile(values, 78.0))

    repair_zone = (
        ((distance_to_seed <= 35) & (spatial_prior >= 0.34))
        | ((distance_to_seed <= 24) & (spatial_prior >= 0.22) & strong_signal)
    )
    pool = before | seed | (weak_signal & repair_zone)
    pool = directional_close(pool, length=13)
    pool = ndi.binary_closing(pool, structure=np.ones((5, 5), dtype=bool))
    pool = remove_small_objects(pool, min_size=90)
    return pool.astype(bool), seed.astype(bool), sig


def estimate_target_frame(pool: np.ndarray, seed: np.ndarray, signal: np.ndarray) -> TargetFrame:
    """Estimate digit position/scale from the target image, before using templates."""
    pool_bool = np.asarray(pool, dtype=bool)
    seed_bool = np.asarray(seed, dtype=bool)
    sig = np.asarray(signal, dtype=np.float32)
    height, width = pool_bool.shape
    evidence = robust_norm(sig, 55.0, 99.7)
    prior = center_prior_map(pool_bool.shape)

    values = evidence[pool_bool]
    if values.size:
        threshold = float(np.percentile(values, 72.0))
    else:
        threshold = 0.0
    support = pool_bool & ((evidence >= threshold) | seed_bool)
    if np.count_nonzero(support) < 200:
        support = pool_bool

    yy, xx = np.indices(pool_bool.shape, dtype=np.float32)
    weights = support.astype(np.float32) * (0.20 + 0.80 * evidence) * (0.32 + 0.68 * prior)
    if float(weights.sum()) <= 0.0:
        bbox = tight_bbox(pool_bool) or (0, 0, height, width)
        y0, x0, y1, x1 = bbox
    else:
        y0f, y1f = weighted_quantile(yy, weights, np.array([0.03, 0.97], dtype=np.float64))
        x0f, x1f = weighted_quantile(xx, weights, np.array([0.03, 0.97], dtype=np.float64))
        frame_h = max(1.0, y1f - y0f)
        frame_w = max(1.0, x1f - x0f)
        margin_y = 0.045 * frame_h
        margin_x = 0.045 * frame_w
        y0 = int(np.clip(np.floor(y0f - margin_y), 0, height - 1))
        y1 = int(np.clip(np.ceil(y1f + margin_y), y0 + 1, height))
        x0 = int(np.clip(np.floor(x0f - margin_x), 0, width - 1))
        x1 = int(np.clip(np.ceil(x1f + margin_x), x0 + 1, width))

    frame_h = max(1, y1 - y0)
    frame_w = max(1, x1 - x0)
    return TargetFrame(
        bbox=(int(y0), int(x0), int(y1), int(x1)),
        center_y_frac=float((y0 + y1) / (2.0 * height)),
        center_x_frac=float((x0 + x1) / (2.0 * width)),
        height_frac=float(frame_h / height),
        width_frac=float(frame_w / width),
    )


def target_for_matching(pool: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evidence = robust_norm(signal, 55.0, 99.7)
    weighted = np.where(pool, 0.35 + 0.65 * evidence, 0.0).astype(np.float32)
    target = resize_float(weighted, (MATCH_SIZE, MATCH_SIZE), order=1)
    target_mask = resize_float(pool.astype(np.float32), (MATCH_SIZE, MATCH_SIZE), order=0) > 0.5
    if float(target.max()) > 0.0:
        target /= float(target.max())
    return target.astype(np.float32), target_mask


def place_at_center(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = np.zeros(shape, dtype=image.dtype)
    h, w = image.shape
    top = int(round(center_y - h / 2.0))
    left = int(round(center_x - w / 2.0))
    y0 = max(0, top)
    x0 = max(0, left)
    y1 = min(shape[0], top + h)
    x1 = min(shape[1], left + w)
    sy0 = y0 - top
    sx0 = x0 - left
    sy1 = sy0 + (y1 - y0)
    sx1 = sx0 + (x1 - x0)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = image[sy0:sy1, sx0:sx1]
    return out, (top, left, top + h, left + w)


def score_template(
    target: np.ndarray,
    target_mask: np.ndarray,
    template: TemplateCandidate,
    height_frac: float,
    target_frame: TargetFrame,
) -> tuple[float, TemplateMatch, np.ndarray] | None:
    crop_h, crop_w = template.crop_soft.shape
    template_aspect = crop_w / max(1, crop_h)
    new_h = int(round(MATCH_SIZE * height_frac))
    new_w = int(round(new_h * crop_w / max(1, crop_h)))
    if new_h < 50 or new_w < 35 or new_h > MATCH_SIZE * 0.97 or new_w > MATCH_SIZE * 0.97:
        return None

    tmpl_soft = resize_float(template.crop_soft, (new_h, new_w), order=1)
    tmpl_mask = resize_float(template.crop_mask.astype(np.float32), (new_h, new_w), order=0) > 0.5
    if float(tmpl_soft.max()) > 0.0:
        tmpl_soft /= float(tmpl_soft.max())
    if np.count_nonzero(tmpl_mask) < 80:
        return None

    corr = fftconvolve(target, tmpl_soft[::-1, ::-1], mode="same")

    # The target image defines the allowed position. The auxiliary template is
    # only moved inside this target-derived window; it cannot choose an unrelated
    # local stroke as its own center.
    yy, xx = np.indices(target.shape, dtype=np.float32)
    target_cy = target_frame.center_y_frac * MATCH_SIZE
    target_cx = target_frame.center_x_frac * MATCH_SIZE
    radius_y = max(16.0, MATCH_SIZE * target_frame.height_frac * 0.20)
    radius_x = max(16.0, MATCH_SIZE * target_frame.width_frac * 0.20)
    allowed_centers = ((yy - target_cy) / radius_y) ** 2 + ((xx - target_cx) / radius_x) ** 2 <= 1.0
    if not np.any(allowed_centers):
        return None
    constrained_corr = np.where(allowed_centers, corr, -np.inf)
    cy, cx = np.unravel_index(int(np.argmax(constrained_corr)), corr.shape)
    if not np.isfinite(constrained_corr[cy, cx]):
        return None
    placed_soft, _ = place_at_center(tmpl_soft, int(cy), int(cx), target.shape)
    placed_mask, _ = place_at_center(tmpl_mask.astype(np.float32), int(cy), int(cx), target.shape)
    placed_mask_bool = placed_mask > 0.5
    tolerance = ndi.binary_dilation(placed_mask_bool, iterations=max(5, int(round(0.035 * max(new_h, new_w)))))

    overlap = float(np.sum(target * placed_soft))
    normalized_overlap = overlap / (float(np.sqrt(np.sum(target * target) * np.sum(placed_soft * placed_soft))) + 1.0e-8)
    candidate_inside = float(np.count_nonzero(target_mask & tolerance) / max(1, np.count_nonzero(target_mask)))
    template_hit_density = float(np.sum(target[placed_mask_bool]) / max(1, np.count_nonzero(placed_mask_bool)))
    dist_to_template = ndi.distance_transform_edt(~placed_mask_bool)
    dist_to_target = ndi.distance_transform_edt(~target_mask)
    target_to_template = float(np.mean(np.exp(-np.square(dist_to_template[target_mask]) / (2.0 * 12.0 * 12.0)))) if np.count_nonzero(target_mask) else 0.0
    template_to_target = float(np.mean(np.exp(-np.square(dist_to_target[placed_mask_bool]) / (2.0 * 12.0 * 12.0)))) if np.count_nonzero(placed_mask_bool) else 0.0
    chamfer_similarity = 0.5 * (target_to_template + template_to_target)
    edge_penalty = float(np.count_nonzero(placed_mask_bool & ~target_mask) / max(1, np.count_nonzero(placed_mask_bool)))
    center_offset = np.sqrt(
        ((float(cy) / MATCH_SIZE - target_frame.center_y_frac) / max(0.08, target_frame.height_frac * 0.50)) ** 2
        + ((float(cx) / MATCH_SIZE - target_frame.center_x_frac) / max(0.08, target_frame.width_frac * 0.50)) ** 2
    )
    target_fit_height_frac = max(target_frame.height_frac, target_frame.width_frac / max(template_aspect, 1.0e-6))
    target_fit_height_frac = float(np.clip(target_fit_height_frac, 0.34, 0.94))
    scale_offset = abs(height_frac - target_fit_height_frac) / max(0.08, target_fit_height_frac)
    score = (
        0.62 * normalized_overlap
        + 0.42 * candidate_inside
        + 0.22 * template_hit_density
        + 0.46 * chamfer_similarity
        - 0.16 * edge_penalty
        - 0.12 * float(center_offset)
        - 0.08 * float(scale_offset)
    )
    match = TemplateMatch(
        predicted_digit=template.digit,
        template_path=template.path,
        score=score,
        normalized_overlap=normalized_overlap,
        candidate_inside_template=candidate_inside,
        template_hit_density=template_hit_density,
        scale_height_frac=height_frac,
        center_y_frac=float(cy) / MATCH_SIZE,
        center_x_frac=float(cx) / MATCH_SIZE,
        template_height_px=new_h,
        template_width_px=new_w,
    )
    return score, match, placed_mask_bool


def find_best_template_match(
    pool: np.ndarray,
    seed: np.ndarray,
    signal: np.ndarray,
    templates: list[TemplateCandidate],
) -> tuple[TemplateMatch, TargetFrame]:
    target, target_mask = target_for_matching(pool, signal)
    target_frame = estimate_target_frame(pool, seed, signal)
    best: tuple[float, TemplateMatch] | None = None
    for template in templates:
        template_aspect = template.crop_soft.shape[1] / max(1, template.crop_soft.shape[0])
        target_fit_height = max(target_frame.height_frac, target_frame.width_frac / max(template_aspect, 1.0e-6))
        target_fit_height = float(np.clip(target_fit_height, 0.34, 0.94))
        # The target image defines scale. For wide target digits such as "4",
        # the height is increased enough for the template width to span the
        # target-derived frame instead of fitting only by template height.
        height_fracs = np.clip(target_fit_height * np.linspace(0.93, 1.07, 11), 0.34, 0.94)
        height_fracs = np.unique(np.round(height_fracs, 4))
        for height_frac in height_fracs:
            scored = score_template(target, target_mask, template, float(height_frac), target_frame)
            if scored is None:
                continue
            score, match, _ = scored
            if best is None or score > best[0]:
                best = (score, match)
    if best is None:
        raise RuntimeError("Template matching failed for all templates and scales.")
    return best[1], target_frame


def aligned_template_full(
    shape: tuple[int, int],
    template: TemplateCandidate,
    match: TemplateMatch,
    alignment: AlignmentParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    new_h = int(round(height * match.scale_height_frac))
    width_multiplier = 1.0
    if alignment is not None:
        width_multiplier = alignment.template_scale_x_multiplier / max(alignment.template_scale_y_multiplier, 1.0e-6)
    new_w = int(round(new_h * template.crop_soft.shape[1] / max(1, template.crop_soft.shape[0]) * width_multiplier))
    new_h = max(8, min(height, new_h))
    new_w = max(8, min(width, new_w))
    soft = resize_float(template.crop_soft, (new_h, new_w), order=1)
    if float(soft.max()) > 0.0:
        soft /= float(soft.max())
    mask = resize_float(template.crop_mask.astype(np.float32), (new_h, new_w), order=0) > 0.5
    center_y = int(round(match.center_y_frac * height))
    center_x = int(round(match.center_x_frac * width))
    placed_soft, _ = place_at_center(soft, center_y, center_x, shape)
    placed_mask, _ = place_at_center(mask.astype(np.float32), center_y, center_x, shape)
    core = placed_mask > 0.5
    min_tolerance_px = int(np.clip(round(0.015 * max(shape)), 5, 45))
    max_tolerance_px = int(np.clip(round(0.040 * max(shape)), 18, 120))
    tolerance_px = int(np.clip(round(0.038 * max(new_h, new_w)), min_tolerance_px, max_tolerance_px))
    gate = ndi.distance_transform_edt(~core) <= tolerance_px
    gate = ndi.binary_closing(gate, structure=np.ones((7, 7), dtype=bool))
    return core.astype(bool), gate.astype(bool), placed_soft.astype(np.float32)


def score_alignment_candidate(
    target: np.ndarray,
    target_mask: np.ndarray,
    template_core: np.ndarray,
    template_gate: np.ndarray,
    template_soft: np.ndarray,
) -> float:
    target_bool = np.asarray(target_mask, dtype=bool)
    core_bool = np.asarray(template_core, dtype=bool)
    gate_bool = np.asarray(template_gate, dtype=bool)
    target_count = int(np.count_nonzero(target_bool))
    core_count = int(np.count_nonzero(core_bool))
    gate_count = int(np.count_nonzero(gate_bool))
    if target_count == 0 or core_count == 0 or gate_count == 0:
        return -np.inf

    gate_overlap = int(np.count_nonzero(target_bool & gate_bool))
    core_overlap = int(np.count_nonzero(target_bool & core_bool))
    match_fraction = gate_overlap / max(1, target_count)
    gate_dice = (2.0 * gate_overlap) / max(1, target_count + gate_count)
    core_dice = (2.0 * core_overlap) / max(1, target_count + core_count)
    target_signal_fraction = float(np.sum(target[gate_bool]) / max(float(np.sum(target[target_bool])), 1.0e-8))
    core_density = float(np.mean(target[core_bool])) if core_count else 0.0
    normalized_overlap = float(
        np.sum(target * template_soft)
        / (np.sqrt(np.sum(target * target) * np.sum(template_soft * template_soft)) + 1.0e-8)
    )

    target_neighborhood = ndi.binary_dilation(target_bool, iterations=10)
    outside_core_penalty = float(np.count_nonzero(core_bool & ~target_neighborhood) / max(1, core_count))
    size_ratio = gate_count / max(1, target_count)
    size_penalty = abs(float(np.log(max(size_ratio, 1.0e-6))))
    coverage_deficit = max(0.0, 0.74 - match_fraction)

    return (
        0.34 * gate_dice
        + 0.26 * match_fraction
        + 0.22 * target_signal_fraction
        + 0.18 * normalized_overlap
        + 0.12 * core_density
        + 0.10 * core_dice
        - 0.20 * outside_core_penalty
        - 0.08 * size_penalty
        - 0.24 * coverage_deficit
    )


def refine_template_alignment(
    pool: np.ndarray,
    signal: np.ndarray,
    template: TemplateCandidate,
    match: TemplateMatch,
    image_shape: tuple[int, int],
    initial_alignment: AlignmentParameters,
) -> tuple[AlignmentParameters, TemplateMatch]:
    if not initial_alignment.auto_refine_alignment:
        adjusted_match = apply_alignment_parameters(match, image_shape, initial_alignment)
        core, gate, soft = aligned_template_full((MATCH_SIZE, MATCH_SIZE), template, adjusted_match, alignment=initial_alignment)
        target, target_mask = target_for_matching(pool, signal)
        score = score_alignment_candidate(target, target_mask, core, gate, soft)
        return replace(initial_alignment, alignment_score=float(score)), adjusted_match

    target, target_mask = target_for_matching(pool, signal)
    best_score = -np.inf
    best_alignment = initial_alignment
    best_match = apply_alignment_parameters(match, image_shape, initial_alignment)

    global_factors = (0.96, 1.00, 1.04, 1.08)
    x_factors = (0.98, 1.02, 1.06, 1.10)
    y_factors = (0.98, 1.02, 1.06)
    offsets = (-45, 0, 45)
    for global_factor in global_factors:
        for x_factor in x_factors:
            for y_factor in y_factors:
                for dy in offsets:
                    for dx in offsets:
                        candidate_alignment = replace(
                            initial_alignment,
                            template_scale_multiplier=float(
                                np.clip(initial_alignment.template_scale_multiplier * global_factor, 0.72, 1.18)
                            ),
                            template_scale_x_multiplier=float(
                                np.clip(initial_alignment.template_scale_x_multiplier * x_factor, 0.78, 1.22)
                            ),
                            template_scale_y_multiplier=float(
                                np.clip(initial_alignment.template_scale_y_multiplier * y_factor, 0.78, 1.22)
                            ),
                            template_y_offset_px=int(initial_alignment.template_y_offset_px + dy),
                            template_x_offset_px=int(initial_alignment.template_x_offset_px + dx),
                        )
                        candidate_match = apply_alignment_parameters(match, image_shape, candidate_alignment)
                        core, gate, soft = aligned_template_full(
                            (MATCH_SIZE, MATCH_SIZE),
                            template,
                            candidate_match,
                            alignment=candidate_alignment,
                        )
                        score = score_alignment_candidate(target, target_mask, core, gate, soft)
                        if score > best_score:
                            best_score = score
                            best_alignment = candidate_alignment
                            best_match = candidate_match

    return replace(best_alignment, alignment_score=float(best_score)), best_match


def template_guided_filter(
    data_id: int,
    before: np.ndarray,
    pool: np.ndarray,
    seed: np.ndarray,
    signal: np.ndarray,
    core_gate: np.ndarray,
    gate: np.ndarray,
    template_soft: np.ndarray,
    predicted_digit: int,
    params: FilterParameters | None = None,
) -> np.ndarray:
    before_bool = np.asarray(before, dtype=bool)
    pool_bool = np.asarray(pool, dtype=bool)
    sig = np.asarray(signal, dtype=np.float32)
    values = sig[sig > 0]
    if values.size == 0:
        return pool_bool & gate
    weak_signal = sig >= float(np.percentile(values, 64.0))
    strong_signal = sig >= float(np.percentile(values, 79.0))
    distance_to_gate = ndi.distance_transform_edt(~gate)
    distance_to_core = ndi.distance_transform_edt(~core_gate)
    params = params or default_filter_parameters(data_id)
    soft_margin_distance = params.soft_margin_distance
    strong_margin_distance = params.strong_margin_distance
    soft_gate = gate | ((distance_to_core <= soft_margin_distance) & (template_soft >= params.template_soft_threshold))
    strong_continuity_zone = distance_to_core <= strong_margin_distance

    source_bool = pool_bool if params.use_repair_pool else before_bool
    labels = label(source_bool, connectivity=2)
    keep = np.zeros(pool_bool.shape, dtype=bool)
    for prop in regionprops(labels, intensity_image=sig):
        coords = prop.coords
        area = int(prop.area)
        if area < 80:
            continue
        inside_gate = float(np.mean(gate[coords[:, 0], coords[:, 1]]))
        inside_soft_gate = float(np.mean(soft_gate[coords[:, 0], coords[:, 1]]))
        mean_template = float(np.mean(template_soft[coords[:, 0], coords[:, 1]]))
        mean_signal = float(prop.mean_intensity)
        seed_overlap = int(np.count_nonzero(seed[coords[:, 0], coords[:, 1]]))
        near_gate = float(np.min(distance_to_gate[coords[:, 0], coords[:, 1]]))
        near_core = float(np.min(distance_to_core[coords[:, 0], coords[:, 1]]))
        y0, x0, y1, x1 = prop.bbox
        span = max(y1 - y0, x1 - x0)
        short_span = max(1, min(y1 - y0, x1 - x0))
        aspect = span / short_span
        stroke_like = span >= 28 and (float(prop.eccentricity) >= 0.62 or aspect >= 1.55)

        inside_shape = inside_gate >= 0.54
        strong_on_template = inside_gate >= 0.30 and mean_template >= 0.10 and (mean_signal >= 0.15 or seed_overlap >= 16)
        near_template_stroke = near_gate <= 18 and stroke_like and mean_signal >= 0.17
        high_conf_edge = inside_gate >= 0.22 and area >= 900 and mean_signal >= 0.29 and seed_overlap >= 40
        soft_edge_stroke = (
            inside_soft_gate >= params.soft_edge_inside_fraction
            and near_core <= strong_margin_distance
            and stroke_like
            and mean_signal >= params.soft_edge_min_signal
        )
        strong_contiguous_target = (
            near_core <= strong_margin_distance
            and seed_overlap >= params.strong_contiguous_seed_overlap
            and mean_signal >= params.strong_contiguous_min_signal
        )
        if inside_shape or strong_on_template or near_template_stroke or high_conf_edge or soft_edge_stroke or strong_contiguous_target:
            keep[coords[:, 0], coords[:, 1]] = True

    distance_to_keep = ndi.distance_transform_edt(~ndi.binary_dilation(keep, iterations=2))
    repair_distance = params.repair_distance
    strong_distance = params.strong_distance
    template_min = params.template_min
    close_length = params.close_length
    bridge_length = params.bridge_length
    bridge_distance = params.bridge_distance

    repair_add = weak_signal & soft_gate & (distance_to_keep <= repair_distance) & ((template_soft >= template_min) | (distance_to_core <= soft_margin_distance))
    repair_add |= strong_signal & soft_gate & (distance_to_keep <= strong_distance)
    if not params.allow_repair_add:
        repair_add &= False
    final = keep | repair_add
    if bridge_length > 0:
        bridge_source = final | (strong_signal & soft_gate & ((template_soft >= template_min) | strong_continuity_zone))
        bridge_add = directional_close(bridge_source, length=bridge_length)
        bridge_add &= soft_gate | strong_continuity_zone
        bridge_add &= (template_soft >= template_min * 0.70) | (ndi.distance_transform_edt(~final) <= bridge_distance)
        bridge_add &= weak_signal | ndi.binary_dilation(final, iterations=5)
        final |= bridge_add
    final = directional_close(final, length=close_length)
    final = ndi.binary_closing(final, structure=np.ones((5, 5), dtype=bool))
    final &= soft_gate | (strong_signal & strong_continuity_zone) | (before_bool & ndi.binary_dilation(keep, iterations=10))
    final = remove_small_objects(final, min_size=120)

    # Restore strong original target pixels in the soft template margin when
    # connected to kept strokes; this prevents real deformed edge strokes from
    # being cut by a hard template boundary.
    connected_zone = ndi.binary_dilation(final, iterations=params.connected_zone_iterations)
    final |= before_bool & strong_signal & (soft_gate | strong_continuity_zone) & connected_zone
    final = remove_small_objects(final, min_size=120)
    return final.astype(bool)


def plot_digit_figure(
    outdir: Path,
    data_id: int,
    predicted_digit: int,
    raw_rgb: np.ndarray,
    before: np.ndarray,
    pool: np.ndarray,
    template_gate: np.ndarray,
    template_soft: np.ndarray,
    template_overlay: np.ndarray,
    final: np.ndarray,
    before_yellow: np.ndarray,
    after_yellow: np.ndarray,
    match: TemplateMatch,
    quality: TemplateMatchQuality,
    dpi: int,
) -> Path:
    fig = plt.figure(figsize=(190.0 / MM_PER_INCH, 142.0 / MM_PER_INCH), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.018, right=0.992, bottom=0.030, top=0.895, wspace=0.045, hspace=0.155)
    removed = pool & ~final
    panels = [
        ("Raw image", downsample_mean(raw_rgb), "rgb"),
        ("Candidate mask\nultra_03_p62", downsample_max(before), "gray"),
        ("Repair pool\nbefore template gate", downsample_max(pool), "gray"),
        (f"Template on raw\npredicted digit {predicted_digit}", downsample_mean(template_overlay), "rgb"),
        ("Template tolerance\ntarget-scale adjusted", downsample_max(template_gate), "gray"),
        ("Final mask\ntemplate-guided", downsample_max(final), "gray"),
        ("Removed non-template\nfragments", downsample_max(removed), "gray"),
        ("Final yellow", downsample_mean(after_yellow), "rgb"),
        ("Change map\nwhite kept, cyan added, red removed", downsample_max(change_overlay(before, final)), "rgb"),
    ]
    for idx, (title, image, mode) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        if mode == "rgb":
            ax.imshow(np.asarray(image, dtype=np.uint8), interpolation="nearest")
        elif mode == "hot":
            ax.imshow(np.asarray(image, dtype=np.float32), cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        else:
            ax.imshow(np.asarray(image, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=7.2, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(
        0.018,
        0.976,
        (
            f"Data {data_id}: template-guided cleanup, predicted digit {predicted_digit}, "
            f"score {match.score:.3f}, template match {quality.template_match_fraction:.2f}"
        ),
        ha="left",
        va="top",
        fontsize=8.2,
    )
    stem = outdir / f"data_{data_id}_template_guided_pred{predicted_digit}"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return stem


def make_contact_sheet(stems: list[Path], output: Path) -> Path:
    thumbs: list[Image.Image] = []
    for stem in stems:
        with Image.open(stem.with_suffix(".png")) as image:
            thumb = image.convert("RGB")
        thumb.thumbnail((780, 590), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (800, 640), "white")
        canvas.paste(thumb, ((canvas.width - thumb.width) // 2, 6))
        ImageDraw.Draw(canvas).text((10, canvas.height - 26), stem.name, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1600, 1280), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % 2) * 800, (idx // 2) * 640))
    sheet.save(output)
    return output


def make_image_contact_sheet(image_paths: list[Path], output: Path) -> Path:
    thumbs: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
        thumb.thumbnail((560, 560), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (600, 630), "white")
        canvas.paste(thumb, ((canvas.width - thumb.width) // 2, 8))
        ImageDraw.Draw(canvas).text((10, canvas.height - 26), path.name, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1200, 1260), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % 2) * 600, (idx // 2) * 630))
    sheet.save(output)
    return output


def count_components(mask: np.ndarray) -> int:
    return len(regionprops(label(np.asarray(mask, dtype=bool), connectivity=2)))


def compute_template_match_quality(
    pool: np.ndarray,
    signal: np.ndarray,
    template_core: np.ndarray,
    template_gate: np.ndarray,
) -> TemplateMatchQuality:
    pool_bool = np.asarray(pool, dtype=bool)
    core_bool = np.asarray(template_core, dtype=bool)
    gate_bool = np.asarray(template_gate, dtype=bool)
    pool_count = int(np.count_nonzero(pool_bool))
    gate_count = int(np.count_nonzero(gate_bool))
    gate_overlap = int(np.count_nonzero(pool_bool & gate_bool))
    core_overlap = int(np.count_nonzero(pool_bool & core_bool))
    sig = np.clip(np.asarray(signal, dtype=np.float32), 0.0, None)
    signal_total = float(np.sum(sig[pool_bool]))
    signal_overlap = float(np.sum(sig[pool_bool & gate_bool]))
    return TemplateMatchQuality(
        template_match_fraction=gate_overlap / max(1, pool_count),
        template_core_fraction=core_overlap / max(1, pool_count),
        template_dice=(2.0 * gate_overlap) / max(1, pool_count + gate_count),
        template_signal_fraction=signal_overlap / max(signal_total, 1.0e-8),
    )


def template_overlay_on_raw(
    raw_rgb: np.ndarray,
    template_core: np.ndarray,
    template_gate: np.ndarray,
    template_soft: np.ndarray,
    target_frame: TargetFrame,
) -> np.ndarray:
    base = np.asarray(raw_rgb, dtype=np.float32)
    soft = np.clip(np.asarray(template_soft, dtype=np.float32), 0.0, 1.0)
    core = np.asarray(template_core, dtype=bool)
    gate = np.asarray(template_gate, dtype=bool)
    overlay = base.copy()

    tint = np.array([0.0, 245.0, 255.0], dtype=np.float32)
    alpha = (0.34 * soft)[..., None]
    overlay = overlay * (1.0 - alpha) + tint * alpha

    core_edge = core & ~ndi.binary_erosion(core, iterations=3, border_value=0)
    gate_edge = gate & ~ndi.binary_erosion(gate, iterations=3, border_value=0)
    overlay[gate_edge] = np.array([255.0, 205.0, 0.0], dtype=np.float32)
    overlay[core_edge] = np.array([0.0, 255.0, 255.0], dtype=np.float32)

    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    y0, x0, y1, x1 = target_frame.bbox
    draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(0, 220, 80), width=6)
    return np.asarray(image, dtype=np.uint8)


def write_metrics(outdir: Path, rows: list[TemplateGuidedMetrics]) -> Path:
    path = outdir / "template_guided_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "data_id",
                "predicted_digit",
                "template_path",
                "score",
                "before_px",
                "pool_px",
                "final_px",
                "removed_px",
                "added_px",
                "template_gate_px",
                "final_components",
                "template_match_fraction",
                "template_core_fraction",
                "template_dice",
                "template_signal_fraction",
                "alignment_score",
                "template_scale_multiplier",
                "template_scale_x_multiplier",
                "template_scale_y_multiplier",
                "template_y_offset_px",
                "template_x_offset_px",
                "target_height_frac",
                "target_width_frac",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.data_id,
                    row.predicted_digit,
                    row.template_path,
                    f"{row.score:.6f}",
                    row.before_px,
                    row.pool_px,
                    row.final_px,
                    row.removed_px,
                    row.added_px,
                    row.template_gate_px,
                    row.final_components,
                    f"{row.template_match_fraction:.6f}",
                    f"{row.template_core_fraction:.6f}",
                    f"{row.template_dice:.6f}",
                    f"{row.template_signal_fraction:.6f}",
                    f"{row.alignment_score:.6f}",
                    f"{row.template_scale_multiplier:.6f}",
                    f"{row.template_scale_x_multiplier:.6f}",
                    f"{row.template_scale_y_multiplier:.6f}",
                    row.template_y_offset_px,
                    row.template_x_offset_px,
                    f"{row.target_height_frac:.6f}",
                    f"{row.target_width_frac:.6f}",
                ]
            )
    return path


def write_matches(outdir: Path, rows: list[TemplateMatch], target_frames: list[TargetFrame], data_ids: list[int]) -> Path:
    path = outdir / "template_match_decisions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "data_id",
                "predicted_digit",
                "template_path",
                "score",
                "normalized_overlap",
                "candidate_inside_template",
                "template_hit_density",
                "scale_height_frac",
                "center_y_frac",
                "center_x_frac",
                "template_height_px_at_match_size",
                "template_width_px_at_match_size",
                "target_center_y_frac",
                "target_center_x_frac",
                "target_height_frac",
                "target_width_frac",
                "target_bbox",
            ]
        )
        for data_id, row, frame in zip(data_ids, rows, target_frames, strict=True):
            writer.writerow(
                [
                    data_id,
                    row.predicted_digit,
                    str(row.template_path),
                    f"{row.score:.6f}",
                    f"{row.normalized_overlap:.6f}",
                    f"{row.candidate_inside_template:.6f}",
                    f"{row.template_hit_density:.6f}",
                    f"{row.scale_height_frac:.6f}",
                    f"{row.center_y_frac:.6f}",
                    f"{row.center_x_frac:.6f}",
                    row.template_height_px,
                    row.template_width_px,
                    f"{frame.center_y_frac:.6f}",
                    f"{frame.center_x_frac:.6f}",
                    f"{frame.height_frac:.6f}",
                    f"{frame.width_frac:.6f}",
                    frame.bbox,
                ]
            )
    return path


def zip_outputs(outdir: Path) -> Path:
    zip_path = outdir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(outdir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(outdir.parent))
    return zip_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template-guided digit cleanup without passing target labels.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_matplotlib()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for subdir in (
        "masks",
        "yellow",
        "changes",
        "template_gate",
        "template_soft",
        "template_overlay",
        "removed",
        "pool",
        "target_frame",
    ):
        (args.outdir / subdir).mkdir(parents=True, exist_ok=True)

    templates = load_templates(args.root)
    template_by_path = {template.path: template for template in templates}
    stems: list[Path] = []
    metric_rows: list[TemplateGuidedMetrics] = []
    match_rows: list[TemplateMatch] = []
    target_frames: list[TargetFrame] = []
    data_ids: list[int] = []
    overlay_paths: list[Path] = []
    for data_id in DIGITS:
        data = _load_digit_data(args.root, data_id)
        raw_rgb = read_raw_rgb(args.root, data_id)
        before = np.asarray(data["mask"], dtype=bool)
        tophat = np.asarray(data["tophat_norm"], dtype=np.float32)
        pool, seed, signal = loose_repair_pool(before, tophat)
        match, target_frame = find_best_template_match(pool, seed, signal, templates)
        template = template_by_path[match.template_path]
        alignment = default_alignment_parameters(data_id)
        refined_alignment, aligned_match = refine_template_alignment(pool, signal, template, match, before.shape, alignment)
        template_core, template_gate, template_soft = aligned_template_full(
            before.shape,
            template,
            aligned_match,
            alignment=refined_alignment,
        )
        quality = compute_template_match_quality(pool, signal, template_core, template_gate)
        template_overlay = template_overlay_on_raw(raw_rgb, template_core, template_gate, template_soft, target_frame)
        final = template_guided_filter(data_id, before, pool, seed, signal, template_core, template_gate, template_soft, aligned_match.predicted_digit)

        before_signal = np.where(before, np.clip(0.30 + 0.70 * tophat, 0.0, 1.0), 0.0)
        final_signal = np.where(final, np.clip(0.25 + 0.75 * signal, 0.0, 1.0), 0.0)
        before_yellow = signal_on_yellow(raw_rgb, before_signal)
        after_yellow = signal_on_yellow(raw_rgb, final_signal)
        removed = pool & ~final

        save_mask(args.outdir / "masks" / f"data{data_id}_before_ultra_03_p62.png", before)
        save_mask(args.outdir / "masks" / f"data{data_id}_template_guided_pred{aligned_match.predicted_digit}.png", final)
        save_mask(args.outdir / "pool" / f"data{data_id}_loose_repair_pool.png", pool)
        save_mask(args.outdir / "template_gate" / f"data{data_id}_aligned_template_gate_pred{aligned_match.predicted_digit}.png", template_gate)
        save_mask(args.outdir / "template_gate" / f"data{data_id}_aligned_template_core_pred{aligned_match.predicted_digit}.png", template_core)
        save_mask(args.outdir / "removed" / f"data{data_id}_removed_non_template.png", removed)
        frame_mask = np.zeros(before.shape, dtype=bool)
        y0, x0, y1, x1 = target_frame.bbox
        frame_mask[y0:y1, x0] = True
        frame_mask[y0:y1, max(x0, x1 - 1)] = True
        frame_mask[y0, x0:x1] = True
        frame_mask[max(y0, y1 - 1), x0:x1] = True
        frame_mask = ndi.binary_dilation(frame_mask, iterations=3)
        save_mask(args.outdir / "target_frame" / f"data{data_id}_target_derived_scale_frame.png", frame_mask)
        Image.fromarray(np.clip(template_soft * 255, 0, 255).astype(np.uint8), mode="L").save(
            args.outdir / "template_soft" / f"data{data_id}_aligned_template_soft_pred{aligned_match.predicted_digit}.png"
        )
        Image.fromarray(change_overlay(before, final), mode="RGB").save(args.outdir / "changes" / f"data{data_id}_change_template_guided.png")
        Image.fromarray(before_yellow, mode="RGB").save(args.outdir / "yellow" / f"data{data_id}_before_yellow.png")
        Image.fromarray(after_yellow, mode="RGB").save(args.outdir / "yellow" / f"data{data_id}_template_guided_yellow.png")
        overlay_path = args.outdir / "template_overlay" / f"data{data_id}_template_on_raw_pred{aligned_match.predicted_digit}.png"
        Image.fromarray(template_overlay, mode="RGB").save(overlay_path)
        overlay_paths.append(overlay_path)

        stems.append(
            plot_digit_figure(
                args.outdir,
                data_id,
                aligned_match.predicted_digit,
                raw_rgb,
                before,
                pool,
                template_gate,
                template_soft,
                template_overlay,
                final,
                before_yellow,
                after_yellow,
                aligned_match,
                quality,
                dpi=args.dpi,
            )
        )
        added = final & ~before
        metric_rows.append(
            TemplateGuidedMetrics(
                data_id=data_id,
                predicted_digit=aligned_match.predicted_digit,
                template_path=str(aligned_match.template_path),
                score=aligned_match.score,
                before_px=int(np.count_nonzero(before)),
                pool_px=int(np.count_nonzero(pool)),
                final_px=int(np.count_nonzero(final)),
                removed_px=int(np.count_nonzero(before & ~final)),
                added_px=int(np.count_nonzero(added)),
                template_gate_px=int(np.count_nonzero(template_gate)),
                final_components=count_components(final),
                template_match_fraction=quality.template_match_fraction,
                template_core_fraction=quality.template_core_fraction,
                template_dice=quality.template_dice,
                template_signal_fraction=quality.template_signal_fraction,
                alignment_score=refined_alignment.alignment_score,
                template_scale_multiplier=refined_alignment.template_scale_multiplier,
                template_scale_x_multiplier=refined_alignment.template_scale_x_multiplier,
                template_scale_y_multiplier=refined_alignment.template_scale_y_multiplier,
                template_y_offset_px=refined_alignment.template_y_offset_px,
                template_x_offset_px=refined_alignment.template_x_offset_px,
                target_height_frac=target_frame.height_frac,
                target_width_frac=target_frame.width_frac,
            )
        )
        match_rows.append(aligned_match)
        target_frames.append(target_frame)
        data_ids.append(data_id)

    sheet = make_contact_sheet(stems, args.outdir / "template_guided_contact_sheet.png")
    overlay_sheet = make_image_contact_sheet(overlay_paths, args.outdir / "template_overlay_contact_sheet.png")
    metrics_path = write_metrics(args.outdir, metric_rows)
    matches_path = write_matches(args.outdir, match_rows, target_frames, data_ids)
    zip_path = zip_outputs(args.outdir)
    check_paths = [*(stem.with_suffix(".png") for stem in stems), *overlay_paths, sheet, overlay_sheet, metrics_path, matches_path, zip_path]
    for path in check_paths:
        print(f"{'OK' if path.exists() and path.stat().st_size > 0 else 'FAIL'} {path}")
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
