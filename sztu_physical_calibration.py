"""Auditable physical-scale calibration for microscopy images.

The detector deliberately identifies only a white horizontal scale-bar graphic.
It does not OCR the adjacent label or assume a physical length from pixels.  A
caller must supply the physical bar length (for example, ``500.0`` for a
validated ``500 um`` label) before a pixels-per-micrometre value is returned.
This separation keeps calibration assumptions explicit in downstream reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


CalibrationSource = Literal[
    "explicit_pixels_per_um",
    "explicit_scale_bar_pixels",
    "white_scale_bar",
    "white_scale_bar_unscaled",
    "failed",
]


@dataclass(frozen=True)
class PhysicalScaleEstimate:
    """Result of a non-destructive physical-scale calibration attempt.

    ``pixels_per_um`` is populated only when both a pixel length and a
    validated physical length are available.  ``failure_reason`` is retained
    for incomplete detections as well, so callers can distinguish an absent
    bar from a detected-but-unlabelled one.
    """

    pixels_per_um: float | None
    scale_bar_length_um: float | None
    scale_bar_pixels: float | None
    confidence: float
    source: CalibrationSource
    failure_reason: str | None
    image_shape: tuple[int, int] | None
    bar_bbox_xyxy: tuple[int, int, int, int] | None
    candidates_considered: int

    @property
    def success(self) -> bool:
        """Whether a usable pixels-per-micrometre value was estimated."""
        return self.pixels_per_um is not None and self.failure_reason is None


@dataclass(frozen=True)
class _BarCandidate:
    width_px: float
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


def estimate_pixels_per_um(
    image: np.ndarray | None,
    *,
    pixels_per_um: float | None = None,
    scale_bar_length_um: float | None = None,
    scale_bar_pixels: float | None = None,
    search_region: Literal["lower_right", "lower", "full"] = "lower_right",
) -> PhysicalScaleEstimate:
    """Estimate image calibration without changing ``image``.

    Parameters
    ----------
    image:
        A grayscale, RGB/BGR, or RGBA/BGRA NumPy array.  It is read only.  It
        may be ``None`` when ``pixels_per_um`` is supplied explicitly.
    pixels_per_um:
        A pre-validated calibration value.  This is the strongest source and
        takes precedence over image detection.
    scale_bar_length_um:
        Physical length represented by the scale bar.  The function never
        infers this from OCR; pass ``500.0`` only after validating the image
        annotation or acquisition metadata.
    scale_bar_pixels:
        Optional explicitly measured length of the scale bar.  It must be
        paired with ``scale_bar_length_um``.
    search_region:
        ``"lower_right"`` is the microscopy default.  ``"lower"`` and
        ``"full"`` are available for images with a different annotation
        layout.
    """
    image_shape = _image_shape(image)

    if pixels_per_um is not None:
        if not _is_positive_finite(pixels_per_um):
            return _failure(
                "invalid_explicit_pixels_per_um", image_shape=image_shape
            )
        return PhysicalScaleEstimate(
            pixels_per_um=float(pixels_per_um),
            scale_bar_length_um=None,
            scale_bar_pixels=None,
            confidence=1.0,
            source="explicit_pixels_per_um",
            failure_reason=None,
            image_shape=image_shape,
            bar_bbox_xyxy=None,
            candidates_considered=0,
        )

    if scale_bar_pixels is not None:
        if not _is_positive_finite(scale_bar_pixels):
            return _failure("invalid_explicit_scale_bar_pixels", image_shape=image_shape)
        if not _is_positive_finite(scale_bar_length_um):
            return _failure(
                "missing_or_invalid_scale_bar_length_um", image_shape=image_shape
            )
        return PhysicalScaleEstimate(
            pixels_per_um=float(scale_bar_pixels) / float(scale_bar_length_um),
            scale_bar_length_um=float(scale_bar_length_um),
            scale_bar_pixels=float(scale_bar_pixels),
            confidence=0.98,
            source="explicit_scale_bar_pixels",
            failure_reason=None,
            image_shape=image_shape,
            bar_bbox_xyxy=None,
            candidates_considered=0,
        )

    if scale_bar_length_um is not None and not _is_positive_finite(scale_bar_length_um):
        return _failure("invalid_scale_bar_length_um", image_shape=image_shape)
    if search_region not in {"lower_right", "lower", "full"}:
        return _failure("invalid_search_region", image_shape=image_shape)

    prepared = _prepare_image(image)
    if prepared is None:
        return _failure("invalid_image", image_shape=image_shape)
    gray, whiteness, height, width = prepared
    candidates = _detect_white_scale_bars(gray, whiteness, search_region)
    if not candidates:
        return _failure("scale_bar_not_found", image_shape=(height, width))

    best = max(candidates, key=lambda candidate: candidate.confidence)
    if best.confidence < 0.62:
        return PhysicalScaleEstimate(
            pixels_per_um=None,
            scale_bar_length_um=(
                float(scale_bar_length_um) if scale_bar_length_um is not None else None
            ),
            scale_bar_pixels=best.width_px,
            confidence=best.confidence,
            source="failed",
            failure_reason="low_confidence_scale_bar_detection",
            image_shape=(height, width),
            bar_bbox_xyxy=best.bbox_xyxy,
            candidates_considered=len(candidates),
        )

    if scale_bar_length_um is None:
        return PhysicalScaleEstimate(
            pixels_per_um=None,
            scale_bar_length_um=None,
            scale_bar_pixels=best.width_px,
            confidence=best.confidence,
            source="white_scale_bar_unscaled",
            failure_reason="missing_scale_bar_length_um",
            image_shape=(height, width),
            bar_bbox_xyxy=best.bbox_xyxy,
            candidates_considered=len(candidates),
        )

    return PhysicalScaleEstimate(
        pixels_per_um=best.width_px / float(scale_bar_length_um),
        scale_bar_length_um=float(scale_bar_length_um),
        scale_bar_pixels=best.width_px,
        confidence=best.confidence,
        source="white_scale_bar",
        failure_reason=None,
        image_shape=(height, width),
        bar_bbox_xyxy=best.bbox_xyxy,
        candidates_considered=len(candidates),
    )


def _image_shape(image: np.ndarray | None) -> tuple[int, int] | None:
    if image is None:
        return None
    try:
        array = np.asarray(image)
    except (TypeError, ValueError):
        return None
    if array.ndim < 2:
        return None
    return int(array.shape[0]), int(array.shape[1])


def _failure(
    reason: str, *, image_shape: tuple[int, int] | None
) -> PhysicalScaleEstimate:
    return PhysicalScaleEstimate(
        pixels_per_um=None,
        scale_bar_length_um=None,
        scale_bar_pixels=None,
        confidence=0.0,
        source="failed",
        failure_reason=reason,
        image_shape=image_shape,
        bar_bbox_xyxy=None,
        candidates_considered=0,
    )


def _is_positive_finite(value: float | None) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric)) and numeric > 0.0


def _prepare_image(
    image: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Return independent float arrays so input pixels can never be mutated."""
    if image is None:
        return None
    try:
        array = np.asarray(image)
    except (TypeError, ValueError):
        return None
    if array.ndim not in {2, 3} or array.shape[0] < 8 or array.shape[1] < 8:
        return None
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        return None

    # astype(copy=True) prevents in-place changes even for writable inputs.
    working = array.astype(np.float32, copy=True)
    if not np.isfinite(working).all():
        return None
    if np.issubdtype(array.dtype, np.bool_):
        working *= 255.0
    elif np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        if maximum > 0.0:
            working *= 255.0 / maximum
    elif float(np.max(working)) <= 1.0:
        working *= 255.0
    working = np.clip(working, 0.0, 255.0)

    height, width = int(working.shape[0]), int(working.shape[1])
    if working.ndim == 2:
        return working, np.ones_like(working, dtype=np.float32), height, width
    if working.shape[2] == 1:
        gray = working[:, :, 0]
        return gray, np.ones_like(gray, dtype=np.float32), height, width

    colors = working[:, :, :3]
    # White is channel-order invariant; the weighted luminance only affects
    # brightness ranking and works for both RGB and BGR input in this use case.
    gray = np.mean(colors, axis=2, dtype=np.float32)
    whiteness = np.min(colors, axis=2) - 0.35 * (np.max(colors, axis=2) - np.min(colors, axis=2))
    return gray, whiteness, height, width


def _detect_white_scale_bars(
    gray: np.ndarray,
    whiteness: np.ndarray,
    search_region: Literal["lower_right", "lower", "full"],
) -> list[_BarCandidate]:
    height, width = gray.shape
    candidates: list[_BarCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x0, y0, x1, y1 in _search_rectangles(height, width, search_region):
        local_gray = gray[y0:y1, x0:x1]
        local_white = whiteness[y0:y1, x0:x1]
        if local_gray.size == 0:
            continue
        bright_threshold = max(160.0, 0.86 * float(np.max(local_gray)))
        white_threshold = max(150.0, 0.82 * float(np.max(local_white)))
        mask = ((local_gray >= bright_threshold) & (local_white >= white_threshold)).astype(
            np.uint8
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, count):
            local_x, local_y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            candidate = _score_component(
                x0 + local_x,
                y0 + local_y,
                component_width,
                component_height,
                area,
                width,
                height,
                search_region,
            )
            if candidate is None or candidate.bbox_xyxy in seen:
                continue
            seen.add(candidate.bbox_xyxy)
            candidates.append(candidate)
    return candidates


def _search_rectangles(
    height: int,
    width: int,
    search_region: Literal["lower_right", "lower", "full"],
) -> tuple[tuple[int, int, int, int], ...]:
    if search_region == "lower_right":
        # The fallback covers lower-left annotations without letting a bright
        # central specimen dominate the primary lower-right search.
        return (
            (int(width * 0.50), int(height * 0.55), width, height),
            (0, int(height * 0.70), width, height),
        )
    if search_region == "lower":
        return ((0, int(height * 0.55), width, height),)
    return ((0, 0, width, height),)


def _score_component(
    x: int,
    y: int,
    component_width: int,
    component_height: int,
    area: int,
    image_width: int,
    image_height: int,
    search_region: Literal["lower_right", "lower", "full"],
) -> _BarCandidate | None:
    if component_height <= 0 or component_width <= 0:
        return None
    width_fraction = component_width / float(image_width)
    height_fraction = component_height / float(image_height)
    aspect_ratio = component_width / float(component_height)
    fill_ratio = area / float(component_width * component_height)
    if width_fraction < 0.018 or width_fraction > 0.55:
        return None
    # Some acquisition systems render a white rectangular scale-label panel
    # around the bar.  Its aspect ratio can be near 4.4 while text glyphs
    # remain far below 4.0, so retain this panel as a valid bar proxy.
    if height_fraction > 0.045 or aspect_ratio < 4.0 or fill_ratio < 0.70:
        return None

    x1, y1 = x + component_width, y + component_height
    center_x = (x + x1) * 0.5 / float(image_width)
    center_y = (y + y1) * 0.5 / float(image_height)
    # Rectangularity distinguishes a bar from the adjacent text glyphs.
    aspect_score = min(1.0, aspect_ratio / 12.0)
    width_score = min(1.0, width_fraction / 0.10)
    position_score = 0.5 * center_y + 0.5 * center_x
    if search_region == "lower":
        position_score = center_y
    elif search_region == "full":
        position_score = 0.5 * center_y + 0.25 * center_x + 0.25
    confidence = float(
        np.clip(
            0.42 * fill_ratio
            + 0.25 * aspect_score
            + 0.15 * width_score
            + 0.18 * position_score,
            0.0,
            1.0,
        )
    )
    return _BarCandidate(
        width_px=float(component_width),
        bbox_xyxy=(x, y, x1, y1),
        confidence=confidence,
    )
