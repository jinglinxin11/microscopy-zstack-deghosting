"""Shared image-processing primitives for auxiliary-guided matching.

This module intentionally contains only preprocessing and geometry helpers
used by the current unified registration and matched-only binary export. It
does not contain a matcher, batch assignment, or executable experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize


ANALYSIS_WIDTH = 900
ROI_X = (0.11, 0.89)
ROI_Y = (0.04, 0.88)


@dataclass
class Structure:
    image: np.ndarray
    response: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    points_xy: np.ndarray
    bbox: tuple[int, int, int, int]


def read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.clip(image, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def resize_for_analysis(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = ANALYSIS_WIDTH / float(width)
    return cv2.resize(
        image,
        (ANALYSIS_WIDTH, int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def roi_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    x0, x1 = int(round(width * ROI_X[0])), int(round(width * ROI_X[1]))
    y0, y1 = int(round(height * ROI_Y[0])), int(round(height * ROI_Y[1]))
    mask[y0:y1, x0:x1] = True
    return mask


def robust_unit(values: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    q_low, q_high = np.percentile(values[valid], (low, high))
    return np.clip((values - q_low) / max(q_high - q_low, 1e-6), 0.0, 1.0)


def dark_response(image: np.ndarray) -> np.ndarray:
    """Extract local dark/brown evidence without candidate-label input."""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[..., 0] / 255.0
    red_axis = (lab[..., 1] - 128.0) / 127.0
    local_luminance = cv2.GaussianBlur(luminance, (0, 0), 1.15)
    local_red = cv2.GaussianBlur(red_axis, (0, 0), 1.15)
    sigma = max(12.0, 0.045 * min(image.shape[:2]))
    background_luminance = cv2.GaussianBlur(local_luminance, (0, 0), sigma)
    background_red = cv2.GaussianBlur(local_red, (0, 0), sigma)
    darkness = np.maximum(background_luminance - local_luminance, 0.0)
    red_excess = np.maximum(local_red - background_red, 0.0)
    valid = roi_mask(image.shape[:2])
    dark = robust_unit(darkness, valid, 72.0, 99.55)
    red = robust_unit(red_excess, valid, 72.0, 99.55)
    response = np.maximum(0.74 * dark, 0.58 * dark + 0.42 * red)
    response = cv2.GaussianBlur(response.astype(np.float32), (0, 0), 1.35)
    response[~valid] = 0.0
    return np.clip(response, 0.0, 1.0).astype(np.float32)


def foreground_mask(response: np.ndarray) -> np.ndarray:
    valid = roi_mask(response.shape)
    smooth = cv2.GaussianBlur(response, (0, 0), 1.6)
    threshold = max(0.14, float(np.percentile(smooth[valid], 94.3)))
    mask = (smooth >= threshold) & valid
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return np.asarray(remove_small_objects(mask, min_size=8, connectivity=2), dtype=bool)


def build_structure(image: np.ndarray) -> Structure:
    response = dark_response(image)
    mask = foreground_mask(response)
    skeleton = skeletonize(mask).astype(bool)
    points_yx = np.argwhere(skeleton)
    if len(points_yx) < 20:
        raise ValueError("Foreground framework has too few skeleton pixels.")
    if len(points_yx) > 1000:
        indices = np.linspace(0, len(points_yx) - 1, 1000).astype(int)
        points_yx = points_yx[indices]
    points_xy = points_yx[:, ::-1].astype(np.float32)
    ys, xs = np.where(mask)
    return Structure(
        image=image,
        response=response,
        mask=mask,
        skeleton=skeleton,
        points_xy=points_xy,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    )


def orientation_fields(response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tangent orientation and structure-tensor coherence."""

    gx = cv2.Sobel(response, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(response, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + np.pi / 2.0
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / (jxx + jyy + 1e-6)
    return np.mod(orientation, np.pi).astype(np.float32), np.clip(coherence, 0.0, 1.0).astype(np.float32)


def corridor_from_points(shape: tuple[int, int], points_xy: np.ndarray, radius: int = 8) -> np.ndarray:
    """Build a hard corridor around valid point coordinates."""

    canvas = np.zeros(shape, dtype=np.uint8)
    points = np.rint(points_xy).astype(np.int32)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < shape[0])
    )
    canvas[points[inside, 1], points[inside, 0]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(canvas, kernel) > 0
