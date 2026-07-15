from __future__ import annotations

import cv2
import numpy as np
import pytest

from microscopy_matching.topology_metrics import extract_skeleton_topology, score_aligned_skeletons


def _canvas() -> np.ndarray:
    return np.zeros((96, 96), dtype=np.uint8)


def _draw_s() -> np.ndarray:
    image = _canvas()
    cv2.ellipse(image, (48, 34), (22, 18), 0, 205, 355, 1, 1)
    cv2.ellipse(image, (48, 62), (22, 18), 0, 25, 175, 1, 1)
    cv2.line(image, (29, 45), (67, 52), 1, 1)
    return image.astype(bool)


def _draw_t() -> np.ndarray:
    image = _canvas()
    cv2.line(image, (22, 22), (74, 22), 1, 1)
    cv2.line(image, (48, 22), (48, 74), 1, 1)
    return image.astype(bool)


def _draw_u() -> np.ndarray:
    image = _canvas()
    cv2.line(image, (27, 20), (27, 61), 1, 1)
    cv2.line(image, (69, 20), (69, 61), 1, 1)
    cv2.ellipse(image, (48, 60), (21, 20), 0, 0, 180, 1, 1)
    return image.astype(bool)


def _draw_z() -> np.ndarray:
    image = _canvas()
    cv2.line(image, (23, 23), (73, 23), 1, 1)
    cv2.line(image, (73, 23), (23, 73), 1, 1)
    cv2.line(image, (23, 73), (73, 73), 1, 1)
    return image.astype(bool)


def _orientations(topology) -> list[float]:
    return [segment.orientation_rad for segment in topology.segments if segment.length_px >= 12.0]


def _near_angle(value: float, expected: float, tolerance: float = 0.20) -> bool:
    delta = abs((value - expected + np.pi / 2.0) % np.pi - np.pi / 2.0)
    return bool(delta <= tolerance)


@pytest.mark.parametrize(
    ("glyph", "expected_endpoints", "required_angles"),
    [
        (_draw_t, 3, (0.0, np.pi / 2.0)),
        (_draw_u, 2, (np.pi / 2.0,)),
        (_draw_z, 2, (0.0, 3.0 * np.pi / 4.0)),
    ],
)
def test_extract_topology_detects_endpoints_and_directional_strokes(
    glyph,
    expected_endpoints: int,
    required_angles: tuple[float, ...],
) -> None:
    topology = extract_skeleton_topology(glyph())

    assert len(topology.endpoint_yx) == expected_endpoints
    assert topology.segments
    angles = _orientations(topology)
    for expected in required_angles:
        assert any(_near_angle(value, expected) for value in angles)


def test_matching_pairs_outrank_cross_glyphs() -> None:
    correct_u = score_aligned_skeletons(_draw_u(), _draw_u())
    wrong_t_for_u = score_aligned_skeletons(_draw_t(), _draw_u())
    correct_z = score_aligned_skeletons(_draw_z(), _draw_z())
    wrong_s_for_z = score_aligned_skeletons(_draw_s(), _draw_z())

    assert correct_u.score > wrong_t_for_u.score
    assert correct_z.score > wrong_s_for_z.score
    assert correct_u.missing_stroke_penalty < wrong_t_for_u.missing_stroke_penalty
    assert correct_z.unexplained_target_evidence_penalty < wrong_s_for_z.unexplained_target_evidence_penalty


def test_missing_stroke_and_unexplained_evidence_are_separate_penalties() -> None:
    template = _draw_t()
    intact = score_aligned_skeletons(template, template)
    missing_vertical = _canvas()
    cv2.line(missing_vertical, (22, 22), (74, 22), 1, 1)
    missing = score_aligned_skeletons(template, missing_vertical.astype(bool))
    # A detached long residual is target evidence that this template cannot explain.
    with_spur = template.astype(np.uint8)
    cv2.line(with_spur, (8, 78), (35, 78), 1, 1)
    residual = score_aligned_skeletons(template, with_spur.astype(bool))

    assert missing.missing_stroke_penalty > intact.missing_stroke_penalty
    assert residual.unexplained_target_evidence_penalty > intact.unexplained_target_evidence_penalty
    assert residual.coverage_score >= 0.99


def test_repeated_calls_are_deterministic_and_label_free() -> None:
    first = score_aligned_skeletons(_draw_z(), _draw_z())
    second = score_aligned_skeletons(_draw_z(), _draw_z())

    assert first.as_dict() == second.as_dict()
    assert "label" not in first.as_dict()


def test_score_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        score_aligned_skeletons(np.zeros((16, 16), dtype=bool), np.zeros((15, 16), dtype=bool))
