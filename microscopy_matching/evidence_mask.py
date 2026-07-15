"""Non-fabricating binary export helpers for selected registrations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_processing import corridor_from_points


@dataclass(frozen=True)
class MatchedBinary:
    """Binary target evidence and its selected spatial corridor."""

    mask: np.ndarray
    corridor: np.ndarray


def matched_only_mask(
    target_mask: np.ndarray,
    transformed_auxiliary_points: np.ndarray,
    *,
    corridor_radius_px: int = 10,
) -> MatchedBinary:
    """Keep target foreground only where the selected transform supports it.

    The auxiliary points build the corridor but are never copied into the
    result. Consequently every foreground output pixel is an original target
    mask pixel and every target-mask pixel outside the corridor is removed.
    """

    target = np.asarray(target_mask, dtype=bool)
    points = np.asarray(transformed_auxiliary_points, dtype=np.float32)
    if target.ndim != 2:
        raise ValueError("target_mask must be a two-dimensional array.")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("transformed_auxiliary_points must have shape (N, 2).")
    if corridor_radius_px < 1:
        raise ValueError("corridor_radius_px must be at least one pixel.")
    corridor = corridor_from_points(target.shape, points, radius=corridor_radius_px)
    return MatchedBinary(mask=target & corridor, corridor=np.asarray(corridor, dtype=bool))


def binary_image(mask: np.ndarray) -> np.ndarray:
    """Return the conventional black-background, white-foreground image."""

    return np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)


def native_binary_image(mask: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample of an analysis binary without new evidence."""

    height, width = native_shape
    if height < 1 or width < 1:
        raise ValueError("native_shape must contain positive dimensions.")
    return cv2.resize(binary_image(mask), (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
