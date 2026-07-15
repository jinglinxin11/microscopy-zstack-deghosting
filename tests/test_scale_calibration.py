from __future__ import annotations

import cv2
import numpy as np
import pytest

from microscopy_matching.scale_calibration import estimate_pixels_per_um


def _synthetic_scale_image(
    height: int,
    width: int,
    bar_width: int,
    *,
    bar_height: int = 6,
) -> np.ndarray:
    """Make a yellow microscopy-like image with a right-lower white scale bar."""
    image = np.full((height, width, 3), (72, 154, 190), dtype=np.uint8)
    x1 = width - max(18, width // 40)
    x0 = x1 - bar_width
    y0 = height - max(30, height // 18)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y0 + bar_height - 1), (255, 255, 255), -1)
    # The label is a distractor: bar detection must select the long rectangle,
    # not individual text glyphs.
    cv2.putText(
        image,
        "500 um",
        (x0, min(height - 8, y0 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


@pytest.mark.parametrize(
    ("shape", "bar_width", "length_um"),
    [((480, 640), 160, 500.0), ((1080, 1470), 294, 500.0)],
)
def test_detects_scale_bar_across_native_image_sizes(
    shape: tuple[int, int], bar_width: int, length_um: float
) -> None:
    image = _synthetic_scale_image(*shape, bar_width)
    original = image.copy()

    result = estimate_pixels_per_um(image, scale_bar_length_um=length_um)

    assert result.success, result.failure_reason
    assert result.source == "white_scale_bar"
    assert result.scale_bar_pixels == pytest.approx(bar_width, abs=1.0)
    assert result.pixels_per_um == pytest.approx(bar_width / length_um, rel=0.02)
    assert result.confidence >= 0.62
    assert result.image_shape == shape
    assert result.bar_bbox_xyxy is not None
    assert np.array_equal(image, original), "calibration must not mutate its input image"


def test_explicit_calibration_does_not_need_scale_bar_detection() -> None:
    image = np.zeros((77, 123), dtype=np.uint8)

    result = estimate_pixels_per_um(image, pixels_per_um=0.284)

    assert result.success
    assert result.source == "explicit_pixels_per_um"
    assert result.pixels_per_um == pytest.approx(0.284)
    assert result.confidence == 1.0
    assert result.image_shape == (77, 123)


def test_explicit_bar_pixels_require_a_valid_physical_length() -> None:
    result = estimate_pixels_per_um(
        None, scale_bar_pixels=200.0, scale_bar_length_um=500.0
    )

    assert result.success
    assert result.source == "explicit_scale_bar_pixels"
    assert result.pixels_per_um == pytest.approx(0.4)

    missing_length = estimate_pixels_per_um(None, scale_bar_pixels=200.0)
    assert not missing_length.success
    assert missing_length.failure_reason == "missing_or_invalid_scale_bar_length_um"


def test_detected_unlabelled_bar_exposes_audit_failure_reason() -> None:
    image = _synthetic_scale_image(600, 800, 200)

    result = estimate_pixels_per_um(image)

    assert not result.success
    assert result.source == "white_scale_bar_unscaled"
    assert result.failure_reason == "missing_scale_bar_length_um"
    assert result.scale_bar_pixels == pytest.approx(200, abs=1.0)
    assert result.confidence >= 0.62


def test_detects_a_white_scale_label_panel_with_text() -> None:
    # This matches target images whose 200 um annotation is a white panel with
    # a black border and text, rather than a thin solid white line.
    image = np.full((900, 1500, 3), (72, 154, 190), dtype=np.uint8)
    panel_width, panel_height = 180, 42  # Aspect ratio 4.29.
    x0, y0 = 1500 - panel_width - 28, 900 - panel_height - 25
    cv2.rectangle(image, (x0, y0), (x0 + panel_width - 1, y0 + panel_height - 1), (255, 255, 255), -1)
    cv2.rectangle(image, (x0, y0), (x0 + panel_width - 1, y0 + panel_height - 1), (0, 0, 0), 2)
    cv2.putText(
        image,
        "200 um",
        (x0 + 43, y0 + 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    result = estimate_pixels_per_um(image, scale_bar_length_um=200.0)

    # The black border reduces the detected white rectangle by two pixels per
    # side; calibration intentionally reports that measured inner-panel width.
    assert result.success, result.failure_reason
    assert result.scale_bar_pixels == pytest.approx(panel_width - 4, abs=1.0)
    assert result.pixels_per_um == pytest.approx((panel_width - 4) / 200.0, rel=0.02)
    assert result.confidence >= 0.62


def test_absent_scale_bar_fails_instead_of_fabricating_calibration() -> None:
    image = np.full((360, 540, 3), (72, 154, 190), dtype=np.uint8)

    result = estimate_pixels_per_um(image, scale_bar_length_um=500.0)

    assert not result.success
    assert result.source == "failed"
    assert result.failure_reason == "scale_bar_not_found"
    assert result.pixels_per_um is None


def test_invalid_explicit_calibration_returns_a_failure_record() -> None:
    result = estimate_pixels_per_um(None, pixels_per_um="not-a-number")  # type: ignore[arg-type]

    assert not result.success
    assert result.source == "failed"
    assert result.failure_reason == "invalid_explicit_pixels_per_um"
