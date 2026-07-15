"""One objective and one transform for ranking, localization, and rendering.

This module deliberately contains no file-name or label logic.  Each call
registers one target structure against one auxiliary structure and returns the
same transform used for every downstream decision and overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import cv2
import numpy as np
from scipy.optimize import minimize
from skimage.morphology import skeletonize

from .image_processing import Structure, corridor_from_points, orientation_fields
from .topology_metrics import TopologyScore, score_aligned_skeletons


@dataclass(frozen=True)
class UnifiedSearchConfig:
    """Declared, label-free search and scoring settings for one candidate."""

    generic_scale_min: float = 0.70
    generic_scale_max: float = 1.60
    generic_scale_count: int = 7
    physical_scale_half_width: float = 0.18
    include_generic_scale_fallback: bool = True
    coarse_angles_deg: tuple[float, ...] = (-5.0, 0.0, 5.0)
    coarse_translation_offsets: tuple[float, ...] = (-32.0, 0.0, 32.0)
    maximum_translation_px: float = 144.0
    top_k_peaks: int = 4
    minimum_in_bounds_fraction: float = 0.94
    fine_scale_half_width: float = 0.14
    fine_angle_half_width_deg: float = 5.0
    fine_translation_radius_px: float = 36.0
    physical_prior_sigma_log: float = 0.22
    physical_prior_weight: float = 0.08
    topology_weight: float = 0.28
    topology_tolerance_px: float = 3.0
    corridor_radius_px: int = 12
    minimum_stability_drop: float = 0.0005


@dataclass(frozen=True)
class UnifiedMatch:
    """Candidate registration result in analysis coordinates."""

    score: float
    geometry_score: float
    topology_score: float
    support: float
    forward_similarity: float
    reverse_similarity: float
    orientation: float
    physical_scale_score: float | None
    physical_scale_prior: float | None
    physical_scale_available: bool
    scale: float
    angle_deg: float
    dx: float
    dy: float
    coarse_scale: float
    coarse_angle_deg: float
    coarse_dx: float
    coarse_dy: float
    dx_plus_1_score_drop: float
    dy_plus_1_score_drop: float
    scale_plus_1pct_score_drop: float
    coarse_boundary_hit: bool
    fine_boundary_hit: bool
    topology: TopologyScore
    status_flags: tuple[str, ...]

    @property
    def stable(self) -> bool:
        return min(
            abs(self.dx_plus_1_score_drop),
            abs(self.dy_plus_1_score_drop),
            abs(self.scale_plus_1pct_score_drop),
        ) >= 0.0005


@dataclass(frozen=True)
class _Geometry:
    score: float
    support: float
    forward_similarity: float
    reverse_similarity: float
    orientation: float
    physical_scale_score: float | None


def analysis_scale_prior(
    *,
    source_pixels_per_um: float | None,
    target_pixels_per_um: float | None,
    source_native_shape: tuple[int, int],
    target_native_shape: tuple[int, int],
    source_analysis_shape: tuple[int, int],
    target_analysis_shape: tuple[int, int],
) -> float | None:
    """Return the expected auxiliary-to-target scale in analysis coordinates.

    The physical native scale is ``target_ppu / source_ppu``.  The result then
    accounts for the independent resampling performed before analysis.  A
    missing or invalid calibration returns ``None`` rather than an invented
    scale prior.
    """

    values = (source_pixels_per_um, target_pixels_per_um)
    if any(value is None or not np.isfinite(value) or value <= 0.0 for value in values):
        return None
    source_height, source_width = source_native_shape
    target_height, target_width = target_native_shape
    source_analysis_height, source_analysis_width = source_analysis_shape
    target_analysis_height, target_analysis_width = target_analysis_shape
    if min(source_height, source_width, target_height, target_width) <= 0:
        return None
    if min(source_analysis_height, source_analysis_width, target_analysis_height, target_analysis_width) <= 0:
        return None
    native = float(target_pixels_per_um) / float(source_pixels_per_um)
    scale_x = native * (target_analysis_width / target_width) / (source_analysis_width / source_width)
    scale_y = native * (target_analysis_height / target_height) / (source_analysis_height / source_height)
    if not np.isfinite(scale_x) or not np.isfinite(scale_y) or min(scale_x, scale_y) <= 0.0:
        return None
    # A physical similarity transform is valid only when the two resampling
    # axes agree.  The caller can record the missing prior if aspect ratios do
    # not permit a defensible isotropic conversion.
    if abs(log(scale_x / scale_y)) > 0.08:
        return None
    return float(np.sqrt(scale_x * scale_y))


def select_central_auxiliary_support(
    structure: Structure,
    *,
    grouping_radius_px: int | None = None,
) -> Structure:
    """Keep the central, spatially coherent observed auxiliary support.

    Dotted microscopy strokes often form several components.  The grouping
    dilation joins neighbouring stroke fragments only to choose a group; the
    returned mask contains strictly original observed pixels and therefore
    cannot fabricate template evidence.  This is applied to auxiliary images
    only, before their physical extent is used in a scale-constrained search.
    """

    mask = np.asarray(structure.mask, dtype=bool)
    height, width = mask.shape
    radius = grouping_radius_px
    if radius is None:
        radius = max(8, int(round(0.024 * min(height, width))))
    if radius < 1 or not mask.any():
        return structure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grouped = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    count, labels, _, _ = cv2.connectedComponentsWithStats(grouped.astype(np.uint8), 8)
    best_mask: np.ndarray | None = None
    best_value = -1.0
    for label in range(1, count):
        observed = mask & (labels == label)
        area = int(np.count_nonzero(observed))
        if area < 20:
            continue
        ys, xs = np.where(observed)
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
        normalized_distance = ((center_x - 0.5 * width) / (0.5 * width)) ** 2 + (
            (center_y - 0.5 * height) / (0.5 * height)
        ) ** 2
        centrality = float(np.exp(-normalized_distance))
        value = area * (0.45 + 0.55 * centrality)
        if value > best_value:
            best_value = value
            best_mask = observed
    if best_mask is None or int(np.count_nonzero(best_mask)) < 20:
        return structure
    selected_skeleton = skeletonize(best_mask).astype(bool)
    points = np.argwhere(selected_skeleton)[:, ::-1].astype(np.float32)
    if len(points) < 20:
        return structure
    if len(points) > 1000:
        points = points[np.linspace(0, len(points) - 1, 1000).astype(int)]
    ys, xs = np.where(best_mask)
    return Structure(
        image=structure.image,
        response=structure.response,
        mask=best_mask,
        skeleton=selected_skeleton,
        points_xy=points,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    )


def similarity_matrix(
    bbox: tuple[int, int, int, int],
    scale: float,
    angle_deg: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return a 2x3 auxiliary-analysis to target-analysis similarity matrix."""

    theta = np.deg2rad(float(angle_deg))
    linear = float(scale) * np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    x0, y0, x1, y1 = bbox
    center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    offset = center + np.asarray([dx, dy], dtype=np.float64) - linear @ center
    return np.column_stack([linear, offset]).astype(np.float32)


def transform_points(
    points_xy: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    angle_deg: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    matrix = similarity_matrix(bbox, scale, angle_deg, dx, dy)
    return points_xy @ matrix[:, :2].T + matrix[:, 2]


def inverse_transform_points(
    points_xy: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    angle_deg: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    matrix = similarity_matrix(bbox, scale, angle_deg, dx, dy).astype(np.float64)
    inverse = np.linalg.inv(matrix[:, :2])
    return (points_xy.astype(np.float64) - matrix[:, 2]) @ inverse.T


def warp_auxiliary_skeleton(
    auxiliary: Structure,
    target_shape: tuple[int, int],
    *,
    scale: float,
    angle_deg: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Warp observed auxiliary skeleton pixels; no target pixels are created."""

    matrix = similarity_matrix(auxiliary.bbox, scale, angle_deg, dx, dy)
    return cv2.warpAffine(
        auxiliary.skeleton.astype(np.uint8),
        matrix,
        (int(target_shape[1]), int(target_shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def native_affine(
    auxiliary: Structure,
    match: UnifiedMatch,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    analysis_shape: tuple[int, int],
) -> np.ndarray:
    """Convert the selected analysis transform to source-native -> target-native."""

    analysis_height, analysis_width = analysis_shape
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    analysis_matrix = similarity_matrix(
        auxiliary.bbox, match.scale, match.angle_deg, match.dx, match.dy
    ).astype(np.float64)
    source_to_analysis = np.diag([analysis_width / source_width, analysis_height / source_height, 1.0])
    analysis_to_target = np.diag([target_width / analysis_width, target_height / analysis_height, 1.0])
    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2, :] = analysis_matrix
    native = analysis_to_target @ homogeneous @ source_to_analysis
    return native[:2, :]


def native_bbox(
    auxiliary: Structure,
    match: UnifiedMatch,
    target_shape: tuple[int, int],
    analysis_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    points = transform_points(
        _skeleton_points(auxiliary, maximum=None),
        auxiliary.bbox,
        match.scale,
        match.angle_deg,
        match.dx,
        match.dy,
    )
    target_height, target_width = target_shape
    analysis_height, analysis_width = analysis_shape
    xs = np.clip(points[:, 0] * target_width / analysis_width, 0, target_width - 1)
    ys = np.clip(points[:, 1] * target_height / analysis_height, 0, target_height - 1)
    return (
        int(np.floor(xs.min())),
        int(np.floor(ys.min())),
        int(np.ceil(xs.max())),
        int(np.ceil(ys.max())),
    )


def refine_candidate(
    target: Structure,
    auxiliary: Structure,
    *,
    physical_scale_prior: float | None = None,
    physical_prior_confidence: float = 0.0,
    physical_scale_available: bool | None = None,
    config: UnifiedSearchConfig = UnifiedSearchConfig(),
) -> UnifiedMatch:
    """Register one candidate with a single score used for rank and display."""

    if physical_scale_available is None:
        physical_scale_available = physical_scale_prior is not None

    target_soft = cv2.GaussianBlur(target.response.astype(np.float32), (0, 0), 2.2)
    target_distance = cv2.distanceTransform((~target.mask).astype(np.uint8), cv2.DIST_L2, 3)
    auxiliary_distance = cv2.distanceTransform((~auxiliary.mask).astype(np.uint8), cv2.DIST_L2, 3)
    target_angle, target_coherence = orientation_fields(target_soft)
    auxiliary_angle, _ = orientation_fields(auxiliary.skeleton.astype(np.float32))
    source_points = _skeleton_points(auxiliary, maximum=640)
    target_points = _skeleton_points(target, maximum=720)
    source_angles = _bilinear_sample(auxiliary_angle, source_points)

    scales = _coarse_scales(physical_scale_prior, physical_prior_confidence, config)
    seeds: list[tuple[float, float, float, float, float, _Geometry]] = []
    for scale in scales:
        for angle in config.coarse_angles_deg:
            translations = _translation_seeds(target_soft, auxiliary, scale, angle, config)
            for dx, dy in translations:
                values = np.asarray([log(scale), angle, dx, dy], dtype=np.float64)
                geometry = _geometry_score(
                    values,
                    target,
                    auxiliary,
                    target_soft,
                    target_distance,
                    auxiliary_distance,
                    target_angle,
                    target_coherence,
                    source_points,
                    target_points,
                    source_angles,
                    physical_scale_prior,
                    physical_prior_confidence,
                    config,
                )
                if geometry.score > 0.0:
                    seeds.append((geometry.score, scale, angle, dx, dy, geometry))
    if not seeds:
        raise RuntimeError("No valid coarse transform candidate.")

    ordered = sorted(seeds, key=lambda item: item[0], reverse=True)
    peaks = _non_maximum_peaks(ordered, config.top_k_peaks)
    best: tuple[
        np.ndarray,
        _Geometry,
        tuple[float, float, float, float],
        np.ndarray,
        np.ndarray,
    ] | None = None
    for _, seed_scale, seed_angle, seed_dx, seed_dy, _ in peaks:
        initial = np.asarray([log(seed_scale), seed_angle, seed_dx, seed_dy], dtype=np.float64)
        lower = np.asarray(
            [
                log(max(1e-4, seed_scale * (1.0 - config.fine_scale_half_width))),
                seed_angle - config.fine_angle_half_width_deg,
                max(-config.maximum_translation_px, seed_dx - config.fine_translation_radius_px),
                max(-config.maximum_translation_px, seed_dy - config.fine_translation_radius_px),
            ],
            dtype=np.float64,
        )
        upper = np.asarray(
            [
                log(seed_scale * (1.0 + config.fine_scale_half_width)),
                seed_angle + config.fine_angle_half_width_deg,
                min(config.maximum_translation_px, seed_dx + config.fine_translation_radius_px),
                min(config.maximum_translation_px, seed_dy + config.fine_translation_radius_px),
            ],
            dtype=np.float64,
        )

        def objective(values: np.ndarray) -> float:
            if np.any(values < lower) or np.any(values > upper):
                return 1.0
            return -_geometry_score(
                values,
                target,
                auxiliary,
                target_soft,
                target_distance,
                auxiliary_distance,
                target_angle,
                target_coherence,
                source_points,
                target_points,
                source_angles,
                physical_scale_prior,
                physical_prior_confidence,
                config,
            ).score

        optimized = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            options={"maxiter": 280, "xatol": 0.002, "fatol": 1e-6},
        )
        values = np.clip(np.asarray(optimized.x, dtype=np.float64), lower, upper)
        geometry = _geometry_score(
            values,
            target,
            auxiliary,
            target_soft,
            target_distance,
            auxiliary_distance,
            target_angle,
            target_coherence,
            source_points,
            target_points,
            source_angles,
            physical_scale_prior,
            physical_prior_confidence,
            config,
        )
        if best is None or geometry.score > best[1].score:
            best = (
                values,
                geometry,
                (seed_scale, seed_angle, seed_dx, seed_dy),
                lower,
                upper,
            )
    if best is None:
        raise RuntimeError("Fine registration did not produce a candidate.")

    values, geometry, coarse, best_lower, best_upper = best
    scale = float(np.exp(values[0]))
    angle_deg = float(values[1])
    dx = float(values[2])
    dy = float(values[3])
    aligned = warp_auxiliary_skeleton(
        auxiliary,
        target.mask.shape,
        scale=scale,
        angle_deg=angle_deg,
        dx=dx,
        dy=dy,
    )
    local_corridor = corridor_from_points(
        target.mask.shape,
        transform_points(source_points, auxiliary.bbox, scale, angle_deg, dx, dy),
        radius=config.corridor_radius_px,
    )
    topology = score_aligned_skeletons(
        aligned,
        target.mask & local_corridor,
        tolerance_px=config.topology_tolerance_px,
    )
    topology_weight = float(np.clip(config.topology_weight, 0.0, 0.8))
    score = float((1.0 - topology_weight) * geometry.score + topology_weight * topology.score)
    dx_plus = values.copy()
    dy_plus = values.copy()
    scale_plus = values.copy()
    dx_plus[2] += 1.0
    dy_plus[3] += 1.0
    scale_plus[0] += log(1.01)
    dx_drop = geometry.score - _geometry_score(
        dx_plus,
        target,
        auxiliary,
        target_soft,
        target_distance,
        auxiliary_distance,
        target_angle,
        target_coherence,
        source_points,
        target_points,
        source_angles,
        physical_scale_prior,
        physical_prior_confidence,
        config,
    ).score
    dy_drop = geometry.score - _geometry_score(
        dy_plus,
        target,
        auxiliary,
        target_soft,
        target_distance,
        auxiliary_distance,
        target_angle,
        target_coherence,
        source_points,
        target_points,
        source_angles,
        physical_scale_prior,
        physical_prior_confidence,
        config,
    ).score
    scale_drop = geometry.score - _geometry_score(
        scale_plus,
        target,
        auxiliary,
        target_soft,
        target_distance,
        auxiliary_distance,
        target_angle,
        target_coherence,
        source_points,
        target_points,
        source_angles,
        physical_scale_prior,
        physical_prior_confidence,
        config,
    ).score
    coarse_boundary = bool(
        abs(coarse[2]) >= config.maximum_translation_px - 1.0
        or abs(coarse[3]) >= config.maximum_translation_px - 1.0
        or np.isclose(coarse[0], scales[0])
        or np.isclose(coarse[0], scales[-1])
    )
    fine_boundary = bool(
        np.any(np.isclose(values, best_lower, atol=0.003))
        or np.any(np.isclose(values, best_upper, atol=0.003))
    )
    flags: list[str] = []
    if not physical_scale_available:
        flags.append("physical_scale_unavailable")
    elif physical_scale_prior is None:
        flags.append("physical_scale_report_only")
    if coarse_boundary:
        flags.append("coarse_boundary_hit")
    if fine_boundary:
        flags.append("fine_boundary_hit")
    if min(abs(dx_drop), abs(dy_drop), abs(scale_drop)) < config.minimum_stability_drop:
        flags.append("flat_registration")
    if topology.missing_stroke_penalty > 0.35:
        flags.append("missing_critical_stroke")
    if topology.endpoint_coverage < 0.65:
        flags.append("low_endpoint_coverage")
    return UnifiedMatch(
        score=score,
        geometry_score=geometry.score,
        topology_score=topology.score,
        support=geometry.support,
        forward_similarity=geometry.forward_similarity,
        reverse_similarity=geometry.reverse_similarity,
        orientation=geometry.orientation,
        physical_scale_score=geometry.physical_scale_score,
        physical_scale_prior=physical_scale_prior,
        physical_scale_available=bool(physical_scale_available),
        scale=scale,
        angle_deg=angle_deg,
        dx=dx,
        dy=dy,
        coarse_scale=coarse[0],
        coarse_angle_deg=coarse[1],
        coarse_dx=coarse[2],
        coarse_dy=coarse[3],
        dx_plus_1_score_drop=float(dx_drop),
        dy_plus_1_score_drop=float(dy_drop),
        scale_plus_1pct_score_drop=float(scale_drop),
        coarse_boundary_hit=coarse_boundary,
        fine_boundary_hit=fine_boundary,
        topology=topology,
        status_flags=tuple(flags),
    )


def _skeleton_points(structure: Structure, maximum: int | None) -> np.ndarray:
    points = np.argwhere(structure.skeleton)[:, ::-1].astype(np.float32)
    if maximum is None or len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum).astype(int)
    return points[indices]


def _bilinear_sample(array: np.ndarray, points_xy: np.ndarray, border: float = 0.0) -> np.ndarray:
    return cv2.remap(
        np.asarray(array, dtype=np.float32),
        points_xy[:, 0].astype(np.float32).reshape(1, -1),
        points_xy[:, 1].astype(np.float32).reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border),
    ).ravel()


def _coarse_scales(
    physical_scale_prior: float | None,
    confidence: float,
    config: UnifiedSearchConfig,
) -> tuple[float, ...]:
    generic = np.linspace(config.generic_scale_min, config.generic_scale_max, config.generic_scale_count)
    if physical_scale_prior is None or not np.isfinite(physical_scale_prior) or physical_scale_prior <= 0.0:
        return tuple(float(value) for value in generic)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    half_width = config.physical_scale_half_width * (1.15 - 0.45 * confidence)
    physical = physical_scale_prior * np.linspace(1.0 - half_width, 1.0 + half_width, 5)
    # A strict calibrated run must not fall back to a remote residual at an
    # implausible scale.  The default keeps generic values for diagnosis; the
    # runner enables the strict branch only after both scale bars pass audit.
    values = physical if not config.include_generic_scale_fallback else np.unique(np.concatenate([generic, physical]))
    return tuple(float(value) for value in values if value > 0.05)


def _translation_seeds(
    target_soft: np.ndarray,
    auxiliary: Structure,
    scale: float,
    angle_deg: float,
    config: UnifiedSearchConfig,
) -> tuple[tuple[float, float], ...]:
    target_height, target_width = target_soft.shape
    x0, y0, x1, y1 = auxiliary.bbox
    source_center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
    target_center = np.asarray([target_width * 0.5, target_height * 0.5], dtype=np.float32)
    bases: list[tuple[float, float]] = [(0.0, 0.0), tuple((target_center - source_center).tolist())]
    matrix = similarity_matrix(auxiliary.bbox, scale, angle_deg, 0.0, 0.0)
    template = cv2.warpAffine(
        auxiliary.skeleton.astype(np.float32),
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    template = cv2.GaussianBlur(template, (0, 0), 2.2)
    try:
        phase, response = cv2.phaseCorrelate(target_soft, template)
    except cv2.error:
        phase, response = (0.0, 0.0), 0.0
    if np.isfinite(phase[0]) and np.isfinite(phase[1]) and np.isfinite(response):
        bases.extend([(float(phase[0]), float(phase[1])), (-float(phase[0]), -float(phase[1]))])
    candidates: set[tuple[float, float]] = set()
    for base_x, base_y in bases:
        for offset_y in config.coarse_translation_offsets:
            for offset_x in config.coarse_translation_offsets:
                dx = float(np.clip(base_x + offset_x, -config.maximum_translation_px, config.maximum_translation_px))
                dy = float(np.clip(base_y + offset_y, -config.maximum_translation_px, config.maximum_translation_px))
                candidates.add((round(dx, 4), round(dy, 4)))
    return tuple(sorted(candidates))


def _geometry_score(
    values: np.ndarray,
    target: Structure,
    auxiliary: Structure,
    target_soft: np.ndarray,
    target_distance: np.ndarray,
    auxiliary_distance: np.ndarray,
    target_angle: np.ndarray,
    target_coherence: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_angles: np.ndarray,
    physical_scale_prior: float | None,
    physical_prior_confidence: float,
    config: UnifiedSearchConfig,
) -> _Geometry:
    scale = float(np.exp(values[0]))
    angle_deg = float(values[1])
    dx = float(values[2])
    dy = float(values[3])
    if not np.isfinite(scale) or scale <= 0.0 or abs(dx) > config.maximum_translation_px or abs(dy) > config.maximum_translation_px:
        return _Geometry(0.0, 0.0, 0.0, 0.0, 0.0, None)
    transformed = transform_points(source_points, auxiliary.bbox, scale, angle_deg, dx, dy)
    inside = (
        (transformed[:, 0] >= 1.0)
        & (transformed[:, 0] < target_soft.shape[1] - 2.0)
        & (transformed[:, 1] >= 1.0)
        & (transformed[:, 1] < target_soft.shape[0] - 2.0)
    )
    if float(np.mean(inside)) < config.minimum_in_bounds_fraction:
        return _Geometry(0.0, 0.0, 0.0, 0.0, 0.0, None)
    active = transformed[inside]
    support = float(np.mean(_bilinear_sample(target_soft, active)))
    forward_distance = _bilinear_sample(target_distance, active, border=50.0)
    forward_similarity = float(np.mean(np.exp(-forward_distance / 4.5)))
    inverse = inverse_transform_points(target_points, auxiliary.bbox, scale, angle_deg, dx, dy)
    reverse_inside = (
        (inverse[:, 0] >= 1.0)
        & (inverse[:, 0] < auxiliary.mask.shape[1] - 2.0)
        & (inverse[:, 1] >= 1.0)
        & (inverse[:, 1] < auxiliary.mask.shape[0] - 2.0)
    )
    if float(np.mean(reverse_inside)) < 0.50:
        return _Geometry(0.0, 0.0, 0.0, 0.0, 0.0, None)
    reverse_distance = _bilinear_sample(auxiliary_distance, inverse[reverse_inside], border=50.0)
    keep = max(1, int(round(0.88 * len(reverse_distance))))
    reverse_similarity = float(np.mean(np.exp(-np.partition(reverse_distance, keep - 1)[:keep] / 4.5)))
    local_angle = _bilinear_sample(target_angle, active)
    coherence = _bilinear_sample(target_coherence, active)
    agreement = 0.5 + 0.5 * np.cos(2.0 * (source_angles[inside] + np.deg2rad(angle_deg) - local_angle))
    orientation = float(np.average(agreement, weights=coherence + 0.05))
    geometry = 0.31 * support + 0.29 * forward_similarity + 0.28 * reverse_similarity + 0.12 * orientation
    physical_score: float | None = None
    if physical_scale_prior is not None and np.isfinite(physical_scale_prior) and physical_scale_prior > 0.0:
        sigma = config.physical_prior_sigma_log / max(0.35, float(np.clip(physical_prior_confidence, 0.0, 1.0)))
        physical_score = float(np.exp(-0.5 * (log(scale / physical_scale_prior) / sigma) ** 2))
        weight = config.physical_prior_weight * float(np.clip(physical_prior_confidence, 0.0, 1.0))
        geometry = (1.0 - weight) * geometry + weight * physical_score
    return _Geometry(
        score=float(np.clip(geometry, 0.0, 1.0)),
        support=float(np.clip(support, 0.0, 1.0)),
        forward_similarity=float(np.clip(forward_similarity, 0.0, 1.0)),
        reverse_similarity=float(np.clip(reverse_similarity, 0.0, 1.0)),
        orientation=float(np.clip(orientation, 0.0, 1.0)),
        physical_scale_score=physical_score,
    )


def _non_maximum_peaks(
    ordered: list[tuple[float, float, float, float, float, _Geometry]],
    count: int,
) -> list[tuple[float, float, float, float, float, _Geometry]]:
    selected: list[tuple[float, float, float, float, float, _Geometry]] = []
    for item in ordered:
        _, scale, angle, dx, dy, _ = item
        distinct = all(
            abs(log(scale / chosen[1])) > 0.035
            or abs(angle - chosen[2]) > 2.0
            or np.hypot(dx - chosen[3], dy - chosen[4]) > 20.0
            for chosen in selected
        )
        if distinct:
            selected.append(item)
        if len(selected) >= count:
            break
    return selected or ordered[:1]
