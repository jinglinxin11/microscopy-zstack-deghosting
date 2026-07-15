"""Minimal in-memory SZTU recognition pipeline and final-result writer.

This module owns orchestration only. Image evidence extraction, registration,
topology scoring, and binary gating remain in their dedicated modules. The
default writer emits only the final selected results, not diagnostic reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sztu_image_core import Structure, build_structure, corridor_from_points, read, resize_for_analysis, write
from sztu_matched_binary import matched_only_mask, native_binary_image
from sztu_physical_calibration import PhysicalScaleEstimate, estimate_pixels_per_um
from sztu_unified_registration import (
    UnifiedMatch,
    native_affine,
    native_bbox,
    refine_candidate,
    select_central_auxiliary_support,
    transform_points,
)


DEFAULT_TARGET_SCALE_BAR_UM = 200.0
DEFAULT_AUXILIARY_SCALE_BAR_UM = 500.0
LOW_MARGIN = 0.025


@dataclass(frozen=True)
class SelectedMatch:
    """One independently selected target/candidate registration."""

    target_index: int
    candidate_index: int
    target_path: Path
    candidate_path: Path
    target_original: np.ndarray
    target: Structure
    auxiliary: Structure
    match: UnifiedMatch
    summary_row: dict[str, object]
    rendered: np.ndarray


@dataclass(frozen=True)
class PipelineRun:
    """Complete in-memory recognition state for one input directory."""

    input_root: Path
    pair_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    selections: tuple[SelectedMatch, ...]


def _calibrations(images: list[np.ndarray], scale_bar_length_um: float) -> list[PhysicalScaleEstimate]:
    return [
        estimate_pixels_per_um(image, scale_bar_length_um=scale_bar_length_um)
        for image in images
    ]


def _render_target_evidence(
    original: np.ndarray,
    target: Structure,
    auxiliary: Structure,
    match: UnifiedMatch,
) -> np.ndarray:
    """Render only target-image evidence inside the selected auxiliary corridor."""

    points = transform_points(
        np.argwhere(auxiliary.skeleton)[:, ::-1].astype(np.float32),
        auxiliary.bbox,
        match.scale,
        match.angle_deg,
        match.dx,
        match.dy,
    )
    corridor_small = corridor_from_points(target.response.shape, points, radius=10).astype(np.float32)
    response_full = cv2.resize(target.response, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    corridor_full = cv2.resize(corridor_small, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    evidence = np.clip(response_full * corridor_full, 0.0, 1.0)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = original.shape[:2]
    background_small = cv2.resize(lab, (max(1, width // 8), max(1, height // 8)), interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.GaussianBlur(background_small, (0, 0), 5.0), (width, height), interpolation=cv2.INTER_CUBIC)
    active = evidence > np.percentile(evidence[evidence > 0], 50.0) if np.any(evidence > 0) else np.zeros_like(evidence, dtype=bool)
    quiet = (corridor_full < 0.02) & (response_full < np.percentile(response_full, 70.0))
    if np.any(active) and np.any(quiet):
        ink_delta = np.median(lab[active] - background[active], axis=0)
    else:
        ink_delta = np.asarray([-24.0, 3.0, 2.0], dtype=np.float32)
    ink_delta = np.clip(ink_delta, [-45.0, -25.0, -25.0], [-8.0, 25.0, 25.0])
    ink_delta[0] = min(float(ink_delta[0]), -22.0)
    rendered = cv2.cvtColor(
        np.clip(background + 1.55 * np.power(evidence, 0.68)[..., None] * ink_delta, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )
    # Retain the measured scale annotation rather than synthesizing it.
    rendered[int(0.86 * height) :, int(0.72 * width) :] = original[int(0.86 * height) :, int(0.72 * width) :]
    return rendered


def _decision_status(match: UnifiedMatch, margin: float) -> str:
    if margin < LOW_MARGIN:
        return "review_required_low_margin"
    if match.coarse_boundary_hit or match.fine_boundary_hit:
        return "review_required_boundary_hit"
    if not match.stable:
        return "review_required_flat_registration"
    if match.topology.missing_stroke_penalty > 0.35 or match.topology.endpoint_coverage < 0.65:
        return "review_required_topology"
    if not match.physical_scale_available:
        return "review_required_physical_scale_unavailable"
    return "automatic_candidate_unvalidated"


def _pair_row(
    target_index: int,
    candidate_index: int,
    target_path: Path,
    candidate_path: Path,
    target_image: np.ndarray,
    auxiliary_image: np.ndarray,
    target_structure: Structure,
    auxiliary_structure: Structure,
    target_calibration: PhysicalScaleEstimate,
    auxiliary_calibration: PhysicalScaleEstimate,
    match: UnifiedMatch,
) -> dict[str, object]:
    native = native_affine(
        auxiliary_structure,
        match,
        auxiliary_image.shape[:2],
        target_image.shape[:2],
        target_structure.image.shape[:2],
    )
    box = native_bbox(auxiliary_structure, match, target_image.shape[:2], target_structure.image.shape[:2])
    native_scale = float(np.sqrt(abs(np.linalg.det(native[:, :2]))))
    expected_native = None
    physical_residual = None
    if target_calibration.success and auxiliary_calibration.success:
        expected_native = float(target_calibration.pixels_per_um / auxiliary_calibration.pixels_per_um)
        physical_residual = native_scale / expected_native
    return {
        "target_id": f"target_{target_index + 1:02d}",
        "target_file_audit_only": target_path.name,
        "candidate_id": f"candidate_{candidate_index + 1:02d}",
        "candidate_file_audit_only": candidate_path.name,
        "candidate_label": candidate_path.stem,
        "final_score": round(match.score, 8),
        "geometry_score": round(match.geometry_score, 8),
        "topology_score": round(match.topology_score, 8),
        "support": round(match.support, 8),
        "forward_similarity": round(match.forward_similarity, 8),
        "reverse_similarity": round(match.reverse_similarity, 8),
        "orientation": round(match.orientation, 8),
        "missing_stroke_penalty": round(match.topology.missing_stroke_penalty, 8),
        "unexplained_target_evidence_penalty": round(match.topology.unexplained_target_evidence_penalty, 8),
        "endpoint_coverage": round(match.topology.endpoint_coverage, 8),
        "analysis_scale": round(match.scale, 8),
        "analysis_angle_deg": round(match.angle_deg, 8),
        "analysis_dx": round(match.dx, 8),
        "analysis_dy": round(match.dy, 8),
        "coarse_scale": round(match.coarse_scale, 8),
        "coarse_angle_deg": round(match.coarse_angle_deg, 8),
        "coarse_dx": round(match.coarse_dx, 8),
        "coarse_dy": round(match.coarse_dy, 8),
        "physical_analysis_scale_prior": None,
        "physical_scale_score": None,
        "physical_scale_available": match.physical_scale_available,
        "target_pixels_per_um": target_calibration.pixels_per_um,
        "auxiliary_pixels_per_um": auxiliary_calibration.pixels_per_um,
        "native_scale": native_scale,
        "expected_native_scale": expected_native,
        "physical_native_scale_residual": physical_residual,
        "native_affine_2x3": " ".join(f"{value:.8f}" for value in native.ravel()),
        "native_bbox_xyxy": " ".join(str(value) for value in box),
        "dx_plus_1_score_drop": round(match.dx_plus_1_score_drop, 8),
        "dy_plus_1_score_drop": round(match.dy_plus_1_score_drop, 8),
        "scale_plus_1pct_score_drop": round(match.scale_plus_1pct_score_drop, 8),
        "coarse_boundary_hit": match.coarse_boundary_hit,
        "fine_boundary_hit": match.fine_boundary_hit,
        "status_flags": "|".join(match.status_flags),
    }


def run_pipeline(
    input_root: Path,
    *,
    target_scale_bar_um: float = DEFAULT_TARGET_SCALE_BAR_UM,
    auxiliary_scale_bar_um: float = DEFAULT_AUXILIARY_SCALE_BAR_UM,
) -> PipelineRun:
    """Run label-free, independent matching without writing intermediate files."""

    root = input_root.resolve()
    target_paths = sorted((root / "target_second_row").glob("*.png"))
    auxiliary_paths = sorted((root / "auxiliary_saved").glob("*.png"))
    if len(target_paths) != 4 or len(auxiliary_paths) != 4:
        raise RuntimeError("Expected exactly four target and four auxiliary PNG files.")

    target_images = [read(path) for path in target_paths]
    auxiliary_images = [read(path) for path in auxiliary_paths]
    targets = [build_structure(resize_for_analysis(image)) for image in target_images]
    auxiliaries = [
        select_central_auxiliary_support(build_structure(resize_for_analysis(image)))
        for image in auxiliary_images
    ]
    target_calibrations = _calibrations(target_images, target_scale_bar_um)
    auxiliary_calibrations = _calibrations(auxiliary_images, auxiliary_scale_bar_um)

    matches: list[list[UnifiedMatch]] = []
    pair_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        target_matches: list[UnifiedMatch] = []
        for candidate_index, auxiliary in enumerate(auxiliaries):
            physical_scale_available = bool(
                target_calibrations[target_index].success
                and auxiliary_calibrations[candidate_index].success
            )
            # The established default is report-only physical calibration.
            # Keep it out of ranking while retaining its availability flag.
            match = refine_candidate(
                target,
                auxiliary,
                physical_scale_prior=None,
                physical_prior_confidence=0.0,
                physical_scale_available=physical_scale_available,
            )
            target_matches.append(match)
            pair_rows.append(
                _pair_row(
                    target_index,
                    candidate_index,
                    target_paths[target_index],
                    auxiliary_paths[candidate_index],
                    target_images[target_index],
                    auxiliary_images[candidate_index],
                    target,
                    auxiliary,
                    target_calibrations[target_index],
                    auxiliary_calibrations[candidate_index],
                    match,
                )
            )
        matches.append(target_matches)

    summary_rows: list[dict[str, object]] = []
    selections: list[SelectedMatch] = []
    for target_index, candidate_matches in enumerate(matches):
        scores = np.asarray([match.score for match in candidate_matches], dtype=np.float64)
        ranked = np.argsort(scores)[::-1]
        best_index, runner_up = int(ranked[0]), int(ranked[1])
        best = candidate_matches[best_index]
        margin = float(scores[best_index] - scores[runner_up])
        selected_box = native_bbox(
            auxiliaries[best_index],
            best,
            target_images[target_index].shape[:2],
            targets[target_index].image.shape[:2],
        )
        target_pixels_per_um = target_calibrations[target_index].pixels_per_um
        selected_width_um = None
        selected_height_um = None
        if target_pixels_per_um is not None and target_pixels_per_um > 0.0:
            selected_width_um = (selected_box[2] - selected_box[0]) / target_pixels_per_um
            selected_height_um = (selected_box[3] - selected_box[1]) / target_pixels_per_um
        summary_row = {
            "mode": "automatic_independent",
            "target_id": f"target_{target_index + 1:02d}",
            "target_file_audit_only": target_paths[target_index].name,
            "selected_candidate_id": f"candidate_{best_index + 1:02d}",
            "selected_label": auxiliary_paths[best_index].stem,
            "runner_up_candidate_id": f"candidate_{runner_up + 1:02d}",
            "runner_up_label": auxiliary_paths[runner_up].stem,
            "selected_score": round(float(scores[best_index]), 8),
            "runner_up_score": round(float(scores[runner_up]), 8),
            "margin": round(margin, 8),
            "decision_status": _decision_status(best, margin),
            "analysis_scale": round(best.scale, 8),
            "analysis_angle_deg": round(best.angle_deg, 8),
            "analysis_dx": round(best.dx, 8),
            "analysis_dy": round(best.dy, 8),
            "physical_scale_prior": None,
            "physical_scale_score": None,
            "physical_scale_available": best.physical_scale_available,
            "physical_scale_mode": "report-only",
            "selected_native_bbox_xyxy": " ".join(str(value) for value in selected_box),
            "selected_width_um": None if selected_width_um is None else round(selected_width_um, 4),
            "selected_height_um": None if selected_height_um is None else round(selected_height_um, 4),
            "topology_score": round(best.topology_score, 8),
            "status_flags": "|".join(best.status_flags),
            "rendered_from_target_evidence_only": True,
        }
        summary_rows.append(summary_row)
        selections.append(
            SelectedMatch(
                target_index=target_index,
                candidate_index=best_index,
                target_path=target_paths[target_index],
                candidate_path=auxiliary_paths[best_index],
                target_original=target_images[target_index],
                target=targets[target_index],
                auxiliary=auxiliaries[best_index],
                match=best,
                summary_row=summary_row,
                rendered=_render_target_evidence(
                    target_images[target_index],
                    targets[target_index],
                    auxiliaries[best_index],
                    best,
                ),
            )
        )

    return PipelineRun(
        input_root=root,
        pair_rows=tuple(pair_rows),
        summary_rows=tuple(summary_rows),
        selections=tuple(selections),
    )


def minimal_results_payload(run: PipelineRun) -> dict[str, object]:
    """Return only selection and transform data required to interpret final images."""

    results: list[dict[str, object]] = []
    for selection in run.selections:
        row = selection.summary_row
        stem = f"{row['target_id']}_{row['selected_label']}"
        results.append(
            {
                "target_id": row["target_id"],
                "selected_label": row["selected_label"],
                "selected_score": row["selected_score"],
                "runner_up_label": row["runner_up_label"],
                "margin": row["margin"],
                "decision_status": row["decision_status"],
                "analysis_transform": {
                    "scale": row["analysis_scale"],
                    "angle_deg": row["analysis_angle_deg"],
                    "dx": row["analysis_dx"],
                    "dy": row["analysis_dy"],
                },
                "native_bbox_xyxy": row["selected_native_bbox_xyxy"],
                "status_flags": row["status_flags"],
                "presentation_file": f"presentation/{stem}.png",
                "binary_file": f"binary/{stem}.png",
            }
        )
    return {
        "mode": "automatic_independent_no_batch_assignment",
        "binary_rule": "target_foreground_and_registered_auxiliary_corridor",
        "results": results,
    }


def _selected_native_binary(selection: SelectedMatch) -> np.ndarray:
    points = transform_points(
        np.argwhere(selection.auxiliary.skeleton)[:, ::-1].astype(np.float32),
        selection.auxiliary.bbox,
        selection.match.scale,
        selection.match.angle_deg,
        selection.match.dx,
        selection.match.dy,
    )
    result = matched_only_mask(selection.target.mask, points, corridor_radius_px=12)
    assert np.all(result.mask <= selection.target.mask)
    assert np.all(result.mask <= result.corridor)
    native = native_binary_image(result.mask, selection.target_original.shape[:2])
    assert set(np.unique(native)).issubset({0, 255})
    return native


def write_minimal_output(run: PipelineRun, outdir: Path) -> dict[str, object]:
    """Write exactly one result JSON and eight final PNG images."""

    destination = outdir.resolve()
    presentation_dir = destination / "presentation"
    binary_dir = destination / "binary"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    payload = minimal_results_payload(run)
    for selection in run.selections:
        row = selection.summary_row
        stem = f"{row['target_id']}_{row['selected_label']}"
        write(presentation_dir / f"{stem}.png", selection.rendered)
        write(binary_dir / f"{stem}.png", _selected_native_binary(selection))
    (destination / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
