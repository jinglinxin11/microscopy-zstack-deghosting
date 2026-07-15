from __future__ import annotations

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

from microscopy_matching.image_processing import Structure
from microscopy_matching.registration import (
    UnifiedSearchConfig,
    analysis_scale_prior,
    native_affine,
    refine_candidate,
    select_central_auxiliary_support,
    similarity_matrix,
)


def _structure(mask: np.ndarray) -> Structure:
    mask = np.asarray(mask, dtype=bool)
    skeleton = skeletonize(mask)
    response = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.4)
    response = np.clip(response / max(float(response.max()), 1e-6), 0.0, 1.0)
    points = np.argwhere(skeleton)[:, ::-1].astype(np.float32)
    ys, xs = np.where(mask)
    image = cv2.cvtColor((response * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return Structure(
        image=image,
        response=response,
        mask=mask,
        skeleton=skeleton,
        points_xy=points,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    )


def _u_mask(size: int = 128) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.line(image, (38, 26), (38, 83), 1, 3)
    cv2.line(image, (90, 26), (90, 83), 1, 3)
    cv2.ellipse(image, (64, 82), (26, 25), 0, 0, 180, 1, 3)
    return image.astype(bool)


def test_analysis_scale_prior_converts_native_to_analysis_coordinates() -> None:
    prior = analysis_scale_prior(
        source_pixels_per_um=0.25,
        target_pixels_per_um=1.0,
        source_native_shape=(100, 200),
        target_native_shape=(400, 800),
        source_analysis_shape=(200, 400),
        target_analysis_shape=(200, 400),
    )

    assert prior == 1.0
    assert (
        analysis_scale_prior(
            source_pixels_per_um=None,
            target_pixels_per_um=1.0,
            source_native_shape=(100, 200),
            target_native_shape=(400, 800),
            source_analysis_shape=(200, 400),
            target_analysis_shape=(200, 400),
        )
        is None
    )


def test_refinement_recovers_a_synthetic_similarity_transform() -> None:
    auxiliary = _structure(_u_mask())
    expected_scale, expected_angle, expected_dx, expected_dy = 1.08, 3.0, 8.0, -6.0
    matrix = similarity_matrix(auxiliary.bbox, expected_scale, expected_angle, expected_dx, expected_dy)
    transformed = cv2.warpAffine(
        auxiliary.mask.astype(np.uint8),
        matrix,
        (128, 128),
        flags=cv2.INTER_NEAREST,
    )
    target = _structure(transformed.astype(bool))
    config = UnifiedSearchConfig(
        generic_scale_min=0.88,
        generic_scale_max=1.22,
        generic_scale_count=5,
        coarse_angles_deg=(-5.0, 0.0, 5.0),
        coarse_translation_offsets=(-16.0, 0.0, 16.0),
        maximum_translation_px=32.0,
        top_k_peaks=3,
        fine_scale_half_width=0.12,
        fine_angle_half_width_deg=4.0,
        fine_translation_radius_px=18.0,
        topology_tolerance_px=3.0,
    )

    result = refine_candidate(target, auxiliary, config=config)

    assert result.score > 0.65
    assert result.topology_score > 0.75
    assert result.scale == pytest.approx(expected_scale, abs=0.10)
    assert result.angle_deg == pytest.approx(expected_angle, abs=4.0)
    assert result.dx == pytest.approx(expected_dx, abs=5.0)
    assert result.dy == pytest.approx(expected_dy, abs=5.0)


def test_native_affine_is_derived_from_the_selected_match() -> None:
    auxiliary = _structure(_u_mask())
    target = _structure(_u_mask())
    result = refine_candidate(
        target,
        auxiliary,
        config=UnifiedSearchConfig(
            generic_scale_min=0.95,
            generic_scale_max=1.05,
            generic_scale_count=3,
            coarse_angles_deg=(0.0,),
            coarse_translation_offsets=(0.0,),
            maximum_translation_px=12.0,
            top_k_peaks=1,
            fine_translation_radius_px=4.0,
        ),
    )

    native = native_affine(
        auxiliary,
        result,
        source_shape=(128, 128),
        target_shape=(128, 128),
        analysis_shape=(128, 128),
    )
    selected = similarity_matrix(auxiliary.bbox, result.scale, result.angle_deg, result.dx, result.dy)

    assert np.allclose(native, selected, atol=1e-5)


def test_central_auxiliary_support_removes_remote_observed_outlier_without_adding_pixels() -> None:
    mask = _u_mask()
    mask[8:12, 8:12] = True
    source = _structure(mask)

    selected = select_central_auxiliary_support(source, grouping_radius_px=8)

    assert selected.mask.sum() < source.mask.sum()
    assert not selected.mask[9, 9]
    assert np.all(selected.mask <= source.mask)
    assert selected.bbox[0] > 20


def test_report_only_calibration_is_not_marked_as_unavailable() -> None:
    auxiliary = _structure(_u_mask())
    target = _structure(_u_mask())

    result = refine_candidate(
        target,
        auxiliary,
        physical_scale_available=True,
        config=UnifiedSearchConfig(
            generic_scale_min=0.95,
            generic_scale_max=1.05,
            generic_scale_count=3,
            coarse_angles_deg=(0.0,),
            coarse_translation_offsets=(0.0,),
            maximum_translation_px=12.0,
            top_k_peaks=1,
            fine_translation_radius_px=4.0,
        ),
    )

    assert result.physical_scale_available
    assert "physical_scale_report_only" in result.status_flags
    assert "physical_scale_unavailable" not in result.status_flags
