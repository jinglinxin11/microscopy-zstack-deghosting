"""Label-independent topology evidence for aligned microscopy skeletons.

This module deliberately has no filename, class-label, or drawing interface.
It consumes two already aligned 2-D masks/skeletons and reports how well the
target contains the template's directional strokes, plus penalties for missing
strokes and target evidence not explained by the template.  It is intended as a
post-registration score component, not as a replacement for geometric
registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


__all__ = [
    "DirectionalSegment",
    "SegmentCoverage",
    "SkeletonTopology",
    "TopologyScore",
    "extract_skeleton_topology",
    "score_aligned_skeletons",
    "skeletonize_input",
]


_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)
_EIGHT_CONNECTED[1, 1] = 0
_FULL_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class DirectionalSegment:
    """One ordered directional portion of a skeleton path."""

    segment_id: int
    path_id: int
    points_yx: np.ndarray
    length_px: float
    orientation_rad: float
    start_yx: tuple[int, int]
    end_yx: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        """Return compact, JSON-safe geometry without duplicating all points."""

        return {
            "segment_id": self.segment_id,
            "path_id": self.path_id,
            "length_px": self.length_px,
            "orientation_rad": self.orientation_rad,
            "start_yx": list(self.start_yx),
            "end_yx": list(self.end_yx),
            "point_count": int(len(self.points_yx)),
        }


@dataclass(frozen=True)
class SkeletonTopology:
    """Skeleton nodes and directionally split paths derived from a 2-D input."""

    skeleton: np.ndarray
    endpoint_yx: tuple[tuple[int, int], ...]
    junction_yx: tuple[tuple[int, int], ...]
    segments: tuple[DirectionalSegment, ...]
    graph_path_count: int

    @property
    def skeleton_length_px(self) -> float:
        return float(self.skeleton.sum())

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint_count": len(self.endpoint_yx),
            "junction_count": len(self.junction_yx),
            "graph_path_count": self.graph_path_count,
            "skeleton_length_px": self.skeleton_length_px,
            "segments": [segment.as_dict() for segment in self.segments],
        }


@dataclass(frozen=True)
class SegmentCoverage:
    """Evidence measurements for one template directional segment."""

    segment_id: int
    path_id: int
    length_px: float
    coverage_fraction: float
    longest_run_fraction: float
    directional_agreement: float
    missing_penalty: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "segment_id": self.segment_id,
            "path_id": self.path_id,
            "length_px": self.length_px,
            "coverage_fraction": self.coverage_fraction,
            "longest_run_fraction": self.longest_run_fraction,
            "directional_agreement": self.directional_agreement,
            "missing_penalty": self.missing_penalty,
        }


@dataclass(frozen=True)
class TopologyScore:
    """Auditable score for two aligned skeletons; every value is in [0, 1]."""

    score: float
    coverage_score: float
    directional_score: float
    endpoint_coverage: float
    endpoint_mismatch_penalty: float
    missing_stroke_penalty: float
    unexplained_target_evidence_penalty: float
    segment_coverages: tuple[SegmentCoverage, ...]
    template_topology: SkeletonTopology
    target_topology: SkeletonTopology

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "coverage_score": self.coverage_score,
            "directional_score": self.directional_score,
            "endpoint_coverage": self.endpoint_coverage,
            "endpoint_mismatch_penalty": self.endpoint_mismatch_penalty,
            "missing_stroke_penalty": self.missing_stroke_penalty,
            "unexplained_target_evidence_penalty": self.unexplained_target_evidence_penalty,
            "segment_coverages": [item.as_dict() for item in self.segment_coverages],
            "template_topology": self.template_topology.as_dict(),
            "target_topology": self.target_topology.as_dict(),
        }


def _as_binary_2d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got {array.shape}.")
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return array.astype(bool, copy=False)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def skeletonize_input(mask_or_skeleton: np.ndarray) -> np.ndarray:
    """Return a read-only, one-pixel-wide skeleton from any 2-D binary input."""

    return _readonly(skeletonize(_as_binary_2d(mask_or_skeleton, "mask_or_skeleton")))


def _skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    degree = ndi.convolve(skeleton.astype(np.uint8), _EIGHT_CONNECTED, mode="constant", cval=0)
    degree[~skeleton] = 0
    return degree


def _neighbours(point: tuple[int, int], skeleton: np.ndarray) -> list[tuple[int, int]]:
    y, x = point
    y0, y1 = max(0, y - 1), min(skeleton.shape[0], y + 2)
    x0, x1 = max(0, x - 1), min(skeleton.shape[1], x + 2)
    neighbours = [
        (int(y0 + yy), int(x0 + xx))
        for yy, xx in np.argwhere(skeleton[y0:y1, x0:x1])
        if (int(y0 + yy), int(x0 + xx)) != point
    ]
    return sorted(neighbours)


def _representative(points_yx: np.ndarray) -> tuple[int, int]:
    centroid = points_yx.mean(axis=0)
    offset = points_yx.astype(np.float64) - centroid
    return tuple(map(int, points_yx[int(np.argmin(np.sum(offset * offset, axis=1)))]))


def _node_metadata(
    skeleton: np.ndarray,
    degree: np.ndarray,
) -> tuple[np.ndarray, dict[int, tuple[int, int]], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Cluster digital junction pixels so one junction remains one graph node."""

    node_mask = skeleton & (degree != 2)
    labels, count = ndi.label(node_mask, structure=_FULL_CONNECTIVITY)
    representatives: dict[int, tuple[int, int]] = {}
    endpoints: list[tuple[int, int]] = []
    junctions: list[tuple[int, int]] = []
    for node_id in range(1, int(count) + 1):
        points = np.argwhere(labels == node_id)
        representative = _representative(points)
        representatives[node_id] = representative
        degrees = degree[labels == node_id]
        if np.any(degrees >= 3):
            junctions.append(representative)
        elif np.any(degrees <= 1):
            endpoints.append(representative)
    return labels, representatives, tuple(sorted(endpoints)), tuple(sorted(junctions))


def _trace_from_node(
    start_node: tuple[int, int],
    first_core: tuple[int, int],
    start_node_id: int,
    skeleton: np.ndarray,
    node_labels: np.ndarray,
    visited_core: np.ndarray,
) -> tuple[list[tuple[int, int]], int | None]:
    """Trace one degree-two corridor until another clustered node is reached."""

    path = [start_node, first_core]
    visited_core[first_core] = True
    previous = start_node
    current = first_core
    limit = max(1, int(skeleton.sum()) + 1)
    for _ in range(limit):
        node_neighbours = [
            neighbour
            for neighbour in _neighbours(current, skeleton)
            if neighbour != previous and int(node_labels[neighbour]) > 0
        ]
        usable_nodes = [
            neighbour
            for neighbour in node_neighbours
            if int(node_labels[neighbour]) != start_node_id or len(path) > 3
        ]
        if usable_nodes:
            end_node = usable_nodes[0]
            path.append(end_node)
            return path, int(node_labels[end_node])

        core_neighbours = [
            neighbour
            for neighbour in _neighbours(current, skeleton)
            if neighbour != previous and int(node_labels[neighbour]) == 0
        ]
        if not core_neighbours:
            return path, None
        unvisited = [neighbour for neighbour in core_neighbours if not visited_core[neighbour]]
        following = unvisited[0] if unvisited else core_neighbours[0]
        if following in path:
            return path, None
        path.append(following)
        visited_core[following] = True
        previous, current = current, following
    return path, None


def _trace_cycle(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Trace the rare all-degree-two cycle deterministically."""

    start = tuple(map(int, np.argwhere(skeleton)[0]))
    first_choices = _neighbours(start, skeleton)
    if not first_choices:
        return [start]
    path = [start, first_choices[0]]
    previous, current = start, first_choices[0]
    limit = max(1, int(skeleton.sum()) + 1)
    for _ in range(limit):
        choices = [point for point in _neighbours(current, skeleton) if point != previous]
        if not choices:
            break
        following = choices[0]
        if following == start:
            break
        if following in path:
            break
        path.append(following)
        previous, current = current, following
    return path


def _graph_paths(
    skeleton: np.ndarray,
    node_labels: np.ndarray,
    representatives: dict[int, tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Return ordered graph corridors, with branchpoint clusters collapsed."""

    if not skeleton.any():
        return []
    if not np.any(node_labels):
        return [_trace_cycle(skeleton)]

    visited_core = np.zeros_like(skeleton, dtype=bool)
    paths: list[list[tuple[int, int]]] = []
    for node_id in sorted(representatives):
        node_pixels = np.argwhere(node_labels == node_id)
        for node_yx in node_pixels:
            start_node = (int(node_yx[0]), int(node_yx[1]))
            core_neighbours = [
                neighbour
                for neighbour in _neighbours(start_node, skeleton)
                if int(node_labels[neighbour]) == 0 and not visited_core[neighbour]
            ]
            for first_core in core_neighbours:
                path, end_node_id = _trace_from_node(
                    start_node,
                    first_core,
                    node_id,
                    skeleton,
                    node_labels,
                    visited_core,
                )
                if len(path) >= 2 and end_node_id is not None:
                    paths.append(path)

    # Preserve very short direct edges that have no degree-two pixels between nodes.
    emitted_direct_edges: set[tuple[int, int]] = set()
    for node_yx in np.argwhere(node_labels > 0):
        point = (int(node_yx[0]), int(node_yx[1]))
        left_id = int(node_labels[point])
        for neighbour in _neighbours(point, skeleton):
            right_id = int(node_labels[neighbour])
            if right_id <= 0 or right_id == left_id:
                continue
            edge = tuple(sorted((left_id, right_id)))
            if edge not in emitted_direct_edges:
                paths.append([point, neighbour])
                emitted_direct_edges.add(edge)
    return paths


def _polyline_length(points_yx: np.ndarray) -> float:
    if len(points_yx) < 2:
        return 0.0
    deltas = np.diff(points_yx.astype(np.float64), axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _rdp_indices(points_yx: np.ndarray, epsilon_px: float) -> list[int]:
    """Return Ramer-Douglas-Peucker breakpoints for an ordered pixel path."""

    if len(points_yx) <= 2:
        return list(range(len(points_yx)))
    epsilon_px = max(float(epsilon_px), 0.0)
    indices: set[int] = {0, len(points_yx) - 1}

    def split(start: int, end: int) -> None:
        if end - start <= 1:
            return
        first = points_yx[start].astype(np.float64)
        last = points_yx[end].astype(np.float64)
        interior = points_yx[start + 1 : end].astype(np.float64)
        vector = last - first
        length = float(np.linalg.norm(vector))
        if length <= _EPSILON:
            distances = np.linalg.norm(interior - first, axis=1)
        else:
            projection = ((interior - first) @ vector) / (length * length)
            closest = first + projection[:, None] * vector
            distances = np.linalg.norm(interior - closest, axis=1)
        local_index = int(np.argmax(distances))
        if float(distances[local_index]) > epsilon_px:
            middle = start + 1 + local_index
            indices.add(middle)
            split(start, middle)
            split(middle, end)

    split(0, len(points_yx) - 1)
    return sorted(indices)


def _orientation(points_yx: np.ndarray) -> float:
    start = points_yx[0]
    end = points_yx[-1]
    dy = float(end[0] - start[0])
    dx = float(end[1] - start[1])
    return float(np.mod(np.arctan2(dy, dx), pi))


def _directional_segments(paths: Iterable[list[tuple[int, int]]], epsilon_px: float) -> tuple[DirectionalSegment, ...]:
    segments: list[DirectionalSegment] = []
    segment_id = 0
    for path_id, path in enumerate(paths):
        points = np.asarray(path, dtype=np.intp)
        if len(points) < 2:
            continue
        breaks = _rdp_indices(points, epsilon_px)
        for start_index, end_index in zip(breaks[:-1], breaks[1:]):
            portion = points[start_index : end_index + 1]
            length = _polyline_length(portion)
            if length <= _EPSILON:
                continue
            frozen_points = _readonly(portion)
            segments.append(
                DirectionalSegment(
                    segment_id=segment_id,
                    path_id=path_id,
                    points_yx=frozen_points,
                    length_px=length,
                    orientation_rad=_orientation(portion),
                    start_yx=tuple(map(int, portion[0])),
                    end_yx=tuple(map(int, portion[-1])),
                )
            )
            segment_id += 1
    return tuple(segments)


def extract_skeleton_topology(
    mask_or_skeleton: np.ndarray,
    *,
    direction_simplification_px: float = 1.25,
) -> SkeletonTopology:
    """Derive endpoints, junctions, and long directional portions of a skeleton.

    ``direction_simplification_px`` controls only geometric splitting of an
    existing path.  It does not add or remove evidence pixels.
    """

    if direction_simplification_px < 0:
        raise ValueError("direction_simplification_px must be non-negative.")
    skeleton = skeletonize_input(mask_or_skeleton)
    degree = _skeleton_degree(skeleton)
    labels, representatives, endpoints, junctions = _node_metadata(skeleton, degree)
    paths = _graph_paths(skeleton, labels, representatives)
    segments = _directional_segments(paths, direction_simplification_px)
    return SkeletonTopology(
        skeleton=skeleton,
        endpoint_yx=endpoints,
        junction_yx=junctions,
        segments=segments,
        graph_path_count=len(paths),
    )


def _orientation_field(skeleton: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate axial local tangent orientation using a Gaussian-weighted PCA."""

    orientation = np.full(skeleton.shape, np.nan, dtype=np.float32)
    valid = np.zeros_like(skeleton, dtype=bool)
    if not skeleton.any():
        return orientation, valid
    sigma_px = max(float(sigma_px), 0.75)
    y, x = np.indices(skeleton.shape, dtype=np.float64)
    weights = skeleton.astype(np.float64)
    total = ndi.gaussian_filter(weights, sigma=sigma_px, mode="constant")
    total = np.maximum(total, _EPSILON)
    mean_x = ndi.gaussian_filter(weights * x, sigma=sigma_px, mode="constant") / total
    mean_y = ndi.gaussian_filter(weights * y, sigma=sigma_px, mode="constant") / total
    var_x = ndi.gaussian_filter(weights * x * x, sigma=sigma_px, mode="constant") / total - mean_x * mean_x
    var_y = ndi.gaussian_filter(weights * y * y, sigma=sigma_px, mode="constant") / total - mean_y * mean_y
    cov_xy = ndi.gaussian_filter(weights * x * y, sigma=sigma_px, mode="constant") / total - mean_x * mean_y
    discriminant = np.sqrt(np.maximum((var_x - var_y) ** 2 + 4.0 * cov_xy * cov_xy, 0.0))
    trace = np.maximum(var_x + var_y, _EPSILON)
    confidence = discriminant / trace
    local_count = ndi.convolve(weights, _FULL_CONNECTIVITY, mode="constant", cval=0.0)
    theta = np.mod(0.5 * np.arctan2(2.0 * cov_xy, var_x - var_y), pi)
    valid = skeleton & (local_count >= 2.0) & (confidence >= 0.05) & np.isfinite(theta)
    orientation[valid] = theta[valid].astype(np.float32)
    return orientation, valid


def _longest_run_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    longest = 0
    current = 0
    for value in values.astype(bool, copy=False):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest / len(values))


def _weighted_mean(values: list[float], weights: list[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    total = float(np.sum(weights))
    if total <= _EPSILON:
        return float(np.mean(values))
    return float(np.dot(values, weights) / total)


def _endpoint_coverage(
    endpoints: tuple[tuple[int, int], ...],
    distance: np.ndarray,
    tolerance_px: float,
) -> float:
    if not endpoints:
        return 1.0
    values = [distance[point] <= tolerance_px for point in endpoints]
    return float(np.mean(values))


def _empty_score(template: SkeletonTopology, target: SkeletonTopology) -> TopologyScore:
    template_nonempty = bool(template.skeleton.any())
    target_nonempty = bool(target.skeleton.any())
    return TopologyScore(
        score=0.0,
        coverage_score=0.0,
        directional_score=0.0,
        endpoint_coverage=0.0,
        endpoint_mismatch_penalty=1.0 if template_nonempty or target_nonempty else 0.0,
        missing_stroke_penalty=1.0 if template_nonempty else 0.0,
        unexplained_target_evidence_penalty=1.0 if target_nonempty else 0.0,
        segment_coverages=(),
        template_topology=template,
        target_topology=target,
    )


def score_aligned_skeletons(
    template_mask_or_skeleton: np.ndarray,
    target_mask_or_skeleton: np.ndarray,
    *,
    tolerance_px: float = 2.5,
    min_long_segment_length_px: float = 8.0,
    minimum_segment_coverage: float = 0.65,
    direction_simplification_px: float = 1.25,
) -> TopologyScore:
    """Score topology evidence after an external geometric registration.

    The inputs must share a pixel coordinate system and shape.  The function
    only measures existing pixels: it never transforms, paints, or repairs an
    input.  A caller should combine this evidence with its registration score.
    """

    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be positive.")
    if min_long_segment_length_px < 0:
        raise ValueError("min_long_segment_length_px must be non-negative.")
    if not 0.0 < minimum_segment_coverage <= 1.0:
        raise ValueError("minimum_segment_coverage must be in (0, 1].")
    template_input = _as_binary_2d(template_mask_or_skeleton, "template_mask_or_skeleton")
    target_input = _as_binary_2d(target_mask_or_skeleton, "target_mask_or_skeleton")
    if template_input.shape != target_input.shape:
        raise ValueError(
            "template_mask_or_skeleton and target_mask_or_skeleton must have the same shape, "
            f"got {template_input.shape} and {target_input.shape}."
        )

    template = extract_skeleton_topology(
        template_input,
        direction_simplification_px=direction_simplification_px,
    )
    target = extract_skeleton_topology(
        target_input,
        direction_simplification_px=direction_simplification_px,
    )
    if not template.skeleton.any() or not target.skeleton.any():
        return _empty_score(template, target)

    long_segments = [
        segment
        for segment in template.segments
        if segment.length_px >= float(min_long_segment_length_px)
    ]
    if not long_segments:
        long_segments = list(template.segments)
    if not long_segments:
        return _empty_score(template, target)

    target_distance, target_nearest = ndi.distance_transform_edt(
        ~target.skeleton,
        return_indices=True,
    )
    template_distance = ndi.distance_transform_edt(~template.skeleton)
    target_orientation, target_orientation_valid = _orientation_field(
        target.skeleton,
        sigma_px=max(1.25, tolerance_px),
    )
    segment_coverages: list[SegmentCoverage] = []
    coverage_values: list[float] = []
    directional_values: list[float] = []
    missing_values: list[float] = []
    weights: list[float] = []
    for segment in long_segments:
        points = segment.points_yx
        y, x = points[:, 0], points[:, 1]
        covered = target_distance[y, x] <= tolerance_px
        coverage = float(np.mean(covered))
        longest_run = _longest_run_fraction(covered)
        nearest_y = target_nearest[0, y, x]
        nearest_x = target_nearest[1, y, x]
        local_valid = covered & target_orientation_valid[nearest_y, nearest_x]
        if local_valid.any():
            local_orientation = target_orientation[nearest_y[local_valid], nearest_x[local_valid]]
            directional = float(
                np.mean(0.5 + 0.5 * np.cos(2.0 * (segment.orientation_rad - local_orientation)))
            )
        else:
            # Missing coverage is penalized separately; avoid inventing an angle.
            directional = 1.0
        coverage_deficit = max(0.0, minimum_segment_coverage - coverage) / minimum_segment_coverage
        continuity_deficit = max(0.0, minimum_segment_coverage - longest_run) / minimum_segment_coverage
        missing = float(np.clip(0.75 * coverage_deficit + 0.25 * continuity_deficit, 0.0, 1.0))
        segment_coverages.append(
            SegmentCoverage(
                segment_id=segment.segment_id,
                path_id=segment.path_id,
                length_px=segment.length_px,
                coverage_fraction=coverage,
                longest_run_fraction=longest_run,
                directional_agreement=directional,
                missing_penalty=missing,
            )
        )
        coverage_values.append(coverage)
        directional_values.append(directional)
        missing_values.append(missing)
        weights.append(max(segment.length_px, 1.0))

    coverage_score = float(np.clip(_weighted_mean(coverage_values, weights), 0.0, 1.0))
    directional_score = float(np.clip(_weighted_mean(directional_values, weights, default=1.0), 0.0, 1.0))
    missing_stroke_penalty = float(np.clip(_weighted_mean(missing_values, weights), 0.0, 1.0))
    unexplained_target = float(
        np.mean(template_distance[target.skeleton] > tolerance_px)
    )
    template_endpoint_coverage = _endpoint_coverage(
        template.endpoint_yx,
        target_distance,
        tolerance_px,
    )
    target_endpoint_coverage = _endpoint_coverage(
        target.endpoint_yx,
        template_distance,
        tolerance_px,
    )
    endpoint_coverage = 0.5 * (template_endpoint_coverage + target_endpoint_coverage)
    endpoint_mismatch = float(np.clip(1.0 - endpoint_coverage, 0.0, 1.0))

    # The score keeps the mechanisms separate for audit; the weights merely
    # provide a bounded combination for a caller's candidate ranking.
    score = float(
        np.clip(
            0.30 * coverage_score
            + 0.20 * directional_score
            + 0.25 * (1.0 - missing_stroke_penalty)
            + 0.15 * (1.0 - unexplained_target)
            + 0.10 * (1.0 - endpoint_mismatch),
            0.0,
            1.0,
        )
    )
    return TopologyScore(
        score=score,
        coverage_score=coverage_score,
        directional_score=directional_score,
        endpoint_coverage=float(np.clip(endpoint_coverage, 0.0, 1.0)),
        endpoint_mismatch_penalty=endpoint_mismatch,
        missing_stroke_penalty=missing_stroke_penalty,
        unexplained_target_evidence_penalty=float(np.clip(unexplained_target, 0.0, 1.0)),
        segment_coverages=tuple(segment_coverages),
        template_topology=template,
        target_topology=target,
    )
