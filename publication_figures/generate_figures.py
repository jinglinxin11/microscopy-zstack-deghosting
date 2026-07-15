"""Generate eight publication-ready figures for microscopy pattern matching.

The script intentionally uses the same Python implementation and the same
in-memory diagnostic values as the production matcher.  It does not alter the
matching algorithm, fabricate image evidence, or infer validation labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import ndimage as ndi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microscopy_matching.image_processing import (  # noqa: E402
    Structure,
    build_structure,
    corridor_from_points,
    read,
    resize_for_analysis,
    robust_unit,
    roi_mask,
)
from microscopy_matching.evidence_mask import matched_only_mask, native_binary_image  # noqa: E402
from microscopy_matching.scale_calibration import PhysicalScaleEstimate, estimate_pixels_per_um  # noqa: E402
from microscopy_matching.pipeline import PipelineRun, SelectedMatch, run_pipeline  # noqa: E402
from microscopy_matching.topology_metrics import SkeletonTopology, extract_skeleton_topology  # noqa: E402
from microscopy_matching.registration import (  # noqa: E402
    UnifiedSearchConfig,
    _bilinear_sample,
    _geometry_score,
    _skeleton_points,
    orientation_fields,
    select_central_auxiliary_support,
    transform_points,
    warp_auxiliary_skeleton,
)


# Restrained microscopy palette: target evidence, auxiliary support, retained
# intersection, and review state always keep the same semantic colors.
INK = "#18313F"
TEXT = "#263640"
MUTED = "#6E7C82"
PAPER = "#FFFFFF"
PANEL = "#F3F6F6"
GRID = "#D8E0E2"
DARK = "#11181D"
TARGET = "#B77932"
TARGET_LIGHT = "#E8C997"
AUX = "#2F8F9D"
AUX_LIGHT = "#A9D5D9"
KEPT = "#D9654F"
KEPT_LIGHT = "#F2B8A9"
REVIEW = "#D79A2B"
REJECTED = "#9AA6AA"
GOOD = "#3E8E67"
RED = "#B84A45"
BLUE = "#315B7D"

RESPONSE_CMAP = LinearSegmentedColormap.from_list(
    "matching_response", ["#F7F4EA", "#B9D8D6", "#3D8E97", "#173A4E"]
)
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "matching_score", ["#EFF4F4", "#B8D7D8", "#4C9DA3", "#174D63"]
)


def configure_style() -> None:
    """Apply a compact double-column publication style."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 6.2,
            "lines.linewidth": 1.4,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


@dataclass
class FigureContext:
    target_dir: Path
    reference_dir: Path
    target_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    target_images: dict[str, np.ndarray]
    candidate_images: dict[str, np.ndarray]
    target_calibrations: dict[str, PhysicalScaleEstimate]
    candidate_calibrations: dict[str, PhysicalScaleEstimate]
    raw_candidate_structures: dict[str, Structure]
    candidate_structures: dict[str, Structure]
    target_structures: dict[str, Structure]
    selection_by_label: dict[str, SelectedMatch]
    pair_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    score_matrix: np.ndarray
    run: PipelineRun

    @property
    def representative(self) -> SelectedMatch:
        return self.selection_by_label[self.target_labels[0]]

    def pair(self, target_label: str, candidate_label: str) -> dict[str, object]:
        target_index = self.target_labels.index(target_label) + 1
        for row in self.pair_rows:
            if row["target_id"] == f"target_{target_index:02d}" and row["candidate_label"] == candidate_label:
                return row
        raise KeyError((target_label, candidate_label))


def _image_paths(directory: Path) -> list[Path]:
    supported_suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def load_context(target_dir: Path, reference_dir: Path) -> FigureContext:
    target_paths = _image_paths(target_dir)
    candidate_paths = _image_paths(reference_dir)
    if len(target_paths) != 4 or len(candidate_paths) != 4:
        raise RuntimeError("Expected four target and four reference image files.")

    target_labels = tuple(path.stem for path in target_paths)
    candidate_labels = tuple(path.stem for path in candidate_paths)
    target_images = {path.stem: read(path) for path in target_paths}
    candidate_images = {path.stem: read(path) for path in candidate_paths}
    target_calibrations = {
        label: estimate_pixels_per_um(image, scale_bar_length_um=200.0)
        for label, image in target_images.items()
    }
    candidate_calibrations = {
        label: estimate_pixels_per_um(image, scale_bar_length_um=500.0)
        for label, image in candidate_images.items()
    }

    run = run_pipeline(target_dir, reference_dir)
    selection_by_label = {selection.target_path.stem: selection for selection in run.selections}
    target_structures = {label: selection_by_label[label].target for label in target_labels}
    raw_candidate_structures = {
        label: build_structure(resize_for_analysis(image))
        for label, image in candidate_images.items()
    }
    candidate_structures = {
        label: select_central_auxiliary_support(raw_candidate_structures[label])
        for label in candidate_labels
    }
    score_matrix = np.zeros((4, 4), dtype=float)
    for row in run.pair_rows:
        ti = int(str(row["target_id"]).split("_")[-1]) - 1
        ci = candidate_labels.index(str(row["candidate_label"]))
        score_matrix[ti, ci] = float(row["final_score"])

    return FigureContext(
        target_dir=target_dir,
        reference_dir=reference_dir,
        target_labels=target_labels,
        candidate_labels=candidate_labels,
        target_images=target_images,
        candidate_images=candidate_images,
        target_calibrations=target_calibrations,
        candidate_calibrations=candidate_calibrations,
        raw_candidate_structures=raw_candidate_structures,
        candidate_structures=candidate_structures,
        target_structures=target_structures,
        selection_by_label=selection_by_label,
        pair_rows=run.pair_rows,
        summary_rows=run.summary_rows,
        score_matrix=score_matrix,
        run=run,
    )


def add_panel_label(ax: plt.Axes, label: str, *, color: str = TEXT, inside: bool = False) -> None:
    x, y = (0.015, 0.985) if inside else (-0.08, 1.145)
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top" if inside else "bottom",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        zorder=50,
    )


def clean_image_axis(ax: plt.Axes, *, dark: bool = False) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(DARK if dark else PAPER)


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0,
        1.105,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=TEXT,
        fontweight="bold",
        clip_on=False,
    )
    if subtitle:
        ax.text(
            0,
            1.045,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.4,
            color=MUTED,
            clip_on=False,
        )


def crop_bounds(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    pad_fraction: float = 0.12,
) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = bbox
    pad = int(round(pad_fraction * max(x1 - x0 + 1, y1 - y0 + 1)))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad + 1),
        min(height, y1 + pad + 1),
    )


def crop_array(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return array[y0:y1, x0:x1]


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def display_micrograph(image: np.ndarray, *, max_width: int = 1400) -> np.ndarray:
    """Apply only a global, display-only contrast stretch to an RGB copy."""

    rgb = to_rgb(image)
    if rgb.shape[1] > max_width:
        scale = max_width / rgb.shape[1]
        rgb = cv2.resize(rgb, (max_width, int(round(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    light = lab[..., 0]
    low, high = np.percentile(light, (0.2, 99.8))
    lab[..., 0] = np.clip((light - low) / max(high - low, 1e-6) * 220 + 25, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def response_rgb(response: np.ndarray) -> np.ndarray:
    return (RESPONSE_CMAP(np.clip(response, 0, 1))[..., :3] * 255).astype(np.uint8)


def mask_rgb(mask: np.ndarray, *, foreground: str = TARGET, background: str = DARK) -> np.ndarray:
    fg = np.asarray(mpl.colors.to_rgb(foreground))
    bg = np.asarray(mpl.colors.to_rgb(background))
    return (np.where(mask[..., None], fg, bg) * 255).astype(np.uint8)


def overlay_target_auxiliary(
    target_mask: np.ndarray,
    auxiliary_points: np.ndarray,
    *,
    corridor_radius: int | None = None,
) -> np.ndarray:
    target = np.asarray(target_mask, dtype=bool)
    image = np.zeros((*target.shape, 3), dtype=float)
    image[:] = mpl.colors.to_rgb(DARK)
    image[target] = mpl.colors.to_rgb(TARGET)
    points = np.rint(auxiliary_points).astype(int)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < target.shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < target.shape[0])
    )
    if corridor_radius is not None:
        corridor = corridor_from_points(target.shape, auxiliary_points, radius=corridor_radius)
        edge = corridor & ~ndi.binary_erosion(corridor, iterations=2)
        image[edge] = mpl.colors.to_rgb(AUX_LIGHT)
    image[points[inside, 1], points[inside, 0]] = mpl.colors.to_rgb(AUX)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def target_candidate_name(ctx: FigureContext, target_label: str, candidate_label: str | None = None) -> str:
    ti = ctx.target_labels.index(target_label) + 1
    text = f"Target {ti}"
    if candidate_label is not None:
        ci = ctx.candidate_labels.index(candidate_label)
        text += f"  |  Candidate {chr(65 + ci)}"
    return text


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.04}
        if fmt == "png":
            kwargs["dpi"] = 300
        elif fmt in {"tif", "tiff"}:
            kwargs["dpi"] = 600
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        else:
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def dark_response_components(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return dark, red, np.clip(response, 0, 1)


def transformed_points(selection: SelectedMatch) -> np.ndarray:
    source = np.argwhere(selection.auxiliary.skeleton)[:, ::-1].astype(np.float32)
    return transform_points(
        source,
        selection.auxiliary.bbox,
        selection.match.scale,
        selection.match.angle_deg,
        selection.match.dx,
        selection.match.dy,
    )


def transformed_points_from_row(structure: Structure, row: dict[str, object]) -> np.ndarray:
    source = np.argwhere(structure.skeleton)[:, ::-1].astype(np.float32)
    return transform_points(
        source,
        structure.bbox,
        float(row["analysis_scale"]),
        float(row["analysis_angle_deg"]),
        float(row["analysis_dx"]),
        float(row["analysis_dy"]),
    )


def figure_01_workflow(ctx: FigureContext) -> plt.Figure:
    """Schematic-led overview with real representative data in every stage."""

    selection = ctx.representative
    target = selection.target
    aux_points = transformed_points(selection)
    bounds = crop_bounds(target.mask.shape, target.bbox, pad_fraction=0.10)
    row_scores = ctx.score_matrix[0]

    fig = plt.figure(figsize=(7.2, 4.15), constrained_layout=False)
    fig.text(0.04, 0.965, "Label-free microscopy pattern matching", fontsize=11, fontweight="bold", color=INK, va="top")
    fig.text(
        0.04,
        0.925,
        "One transform links localization, ranking and evidence-only rendering",
        fontsize=6.5,
        color=MUTED,
        va="top",
    )

    lefts = np.linspace(0.045, 0.835, 6)
    width, height, bottom = 0.125, 0.42, 0.43
    cards: list[plt.Axes] = []
    for i, left in enumerate(lefts):
        outer = FancyBboxPatch(
            (left - 0.006, bottom - 0.06),
            width + 0.012,
            height + 0.11,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor=PANEL if i not in {3, 5} else "#EEF5F5",
            edgecolor=GRID,
            linewidth=0.8,
            zorder=-5,
        )
        fig.patches.append(outer)
        ax = fig.add_axes([left, bottom, width, height])
        clean_image_axis(ax, dark=i in {2, 3, 5})
        cards.append(ax)

    # 1. Independent target and candidate inputs.
    target_crop = crop_array(selection.target.image, bounds)
    candidate_bounds = crop_bounds(selection.auxiliary.image.shape[:2], selection.auxiliary.bbox, pad_fraction=0.10)
    candidate_crop = crop_array(selection.auxiliary.image, candidate_bounds)
    h = min(target_crop.shape[0], candidate_crop.shape[0])
    target_crop = cv2.resize(target_crop, (240, h), interpolation=cv2.INTER_AREA)
    candidate_crop = cv2.resize(candidate_crop, (240, h), interpolation=cv2.INTER_AREA)
    cards[0].imshow(np.hstack([to_rgb(target_crop), to_rgb(candidate_crop)]))
    cards[0].axvline(239.5, color="white", lw=1.2)
    cards[0].text(0.24, 0.04, "target", transform=cards[0].transAxes, color="white", ha="center", fontsize=5.5)
    cards[0].text(0.76, 0.04, "candidate", transform=cards[0].transAxes, color="white", ha="center", fontsize=5.5)

    cards[1].imshow(response_rgb(crop_array(target.response, bounds)))
    skeleton_display = ndi.binary_dilation(target.skeleton, iterations=1)
    cards[2].imshow(mask_rgb(crop_array(skeleton_display, bounds), foreground=TARGET_LIGHT))
    cards[3].imshow(crop_array(overlay_target_auxiliary(target.mask, aux_points), bounds))

    bars = cards[4].bar(np.arange(4), row_scores, color=[AUX_LIGHT, AUX_LIGHT, AUX_LIGHT, AUX_LIGHT], width=0.68)
    winner = int(np.argmax(row_scores))
    bars[winner].set_color(KEPT)
    cards[4].set_ylim(0.25, 0.60)
    cards[4].set_xticks(range(4), ["A", "B", "C", "D"])
    cards[4].set_yticks([0.3, 0.4, 0.5])
    cards[4].tick_params(length=2, pad=1)
    cards[4].spines["left"].set_color(GRID)
    cards[4].spines["bottom"].set_color(GRID)
    cards[4].text(winner, row_scores[winner] + 0.012, "max", ha="center", color=KEPT, fontsize=5.5, fontweight="bold")

    matched = matched_only_mask(target.mask, aux_points, corridor_radius_px=12)
    cards[5].imshow(mask_rgb(crop_array(matched.mask, bounds), foreground="white"))

    titles = [
        "1  Independent inputs",
        "2  Local evidence",
        "3  Skeleton topology",
        "4  Unified registration",
        "5  Candidate score",
        "6  Evidence-only output",
    ]
    subtitles = [
        "no label input",
        "darkness + red excess",
        "observed pixels only",
        "scale, angle, translation",
        "argmax + margin",
        "target intersection",
    ]
    for ax, title, subtitle in zip(cards, titles, subtitles):
        ax.text(0, 1.115, title, transform=ax.transAxes, ha="left", fontsize=6.5, fontweight="bold", color=TEXT)
        ax.text(0, 1.065, subtitle, transform=ax.transAxes, ha="left", fontsize=5.2, color=MUTED)

    for i in range(5):
        arrow = FancyArrowPatch(
            (lefts[i] + width + 0.008, bottom + height * 0.5),
            (lefts[i + 1] - 0.010, bottom + height * 0.5),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.0,
            color=INK,
        )
        fig.patches.append(arrow)

    # Mathematical invariants strip.
    strip = fig.add_axes([0.04, 0.075, 0.92, 0.22])
    strip.axis("off")
    add_panel_label(strip, "b", inside=True)
    strip.text(0.035, 0.88, "Auditable invariants", fontsize=7.2, fontweight="bold", color=TEXT, va="top")
    blocks = [
        ("Evidence", "R = max(0.74 d, 0.58 d + 0.42 r)"),
        ("Transform", "x' = s R(theta)(x - c) + c + [dx, dy]"),
        ("Decision", "F = 0.72 G + 0.28 T;  j* = argmax F"),
        ("Output", "M_out = M_target AND corridor(T*skel_aux)"),
    ]
    for i, (title, equation) in enumerate(blocks):
        x = 0.035 + i * 0.238
        patch = FancyBboxPatch(
            (x, 0.12),
            0.215,
            0.54,
            transform=strip.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F7F9F9",
            edgecolor=GRID,
            linewidth=0.8,
        )
        strip.add_patch(patch)
        strip.text(x + 0.012, 0.56, title.upper(), transform=strip.transAxes, fontsize=5.3, color=MUTED, fontweight="bold")
        strip.text(x + 0.012, 0.31, equation, transform=strip.transAxes, fontsize=5.5, color=INK, va="center")
    strip.text(
        0.5,
        0.00,
        "Guardrail: auxiliary pixels guide the corridor but are never copied into the binary output.",
        transform=strip.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=KEPT,
        fontweight="bold",
    )

    fig.text(0.022, 0.865, "a", fontsize=8.5, fontweight="bold", color=TEXT)
    return fig


def figure_02_signal_scale(ctx: FigureContext) -> plt.Figure:
    """Weak-signal extraction and report-only physical calibration."""

    label = ctx.target_labels[0]
    selection = ctx.selection_by_label[label]
    image = selection.target.image
    dark, red, response = dark_response_components(image)
    valid = roi_mask(response.shape)
    smooth = cv2.GaussianBlur(response, (0, 0), 1.6)
    threshold = max(0.14, float(np.percentile(smooth[valid], 94.3)))
    bounds = crop_bounds(image.shape[:2], selection.target.bbox, pad_fraction=0.10)

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.97, bottom=0.10, top=0.92, wspace=0.18, hspace=0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(to_rgb(image))
    x0 = int(round(image.shape[1] * 0.11))
    x1 = int(round(image.shape[1] * 0.89))
    y0 = int(round(image.shape[0] * 0.04))
    y1 = int(round(image.shape[0] * 0.88))
    ax_a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=KEPT, lw=1.2))
    clean_image_axis(ax_a)
    add_panel_label(ax_a, "a")
    add_title(ax_a, "Raw target", "analysis ROI shown in coral")

    panels = [
        (dark, "Local darkness", "d = robust(bg_L - local_L)"),
        (red, "Local red excess", "r = robust(local_a - bg_a)"),
        (response, "Combined response", "R = max(0.74d, 0.58d + 0.42r)"),
    ]
    for j, (array, title, subtitle) in enumerate(panels, start=1):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(crop_array(array, bounds), cmap=RESPONSE_CMAP, vmin=0, vmax=1)
        clean_image_axis(ax)
        add_panel_label(ax, chr(ord("a") + j))
        add_title(ax, title, subtitle)
        if j == 3:
            cax = ax.inset_axes([0.12, -0.09, 0.76, 0.035])
            fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
            cax.tick_params(labelsize=5, length=2, pad=1)
            cax.set_xlabel("normalized response", fontsize=5.3, labelpad=-1)

    ax_e = fig.add_subplot(gs[1, 0])
    mask_crop = crop_array(selection.target.mask, bounds)
    ax_e.imshow(mask_rgb(mask_crop, foreground=TARGET_LIGHT))
    clean_image_axis(ax_e, dark=True)
    add_panel_label(ax_e, "e", color=TEXT)
    add_title(ax_e, "Adaptive foreground", f"threshold = max(0.14, P94.3) = {threshold:.3f}")

    ax_f = fig.add_subplot(gs[1, 1])
    values = smooth[valid].ravel()
    ax_f.hist(values, bins=55, density=True, color=AUX_LIGHT, edgecolor="none")
    ax_f.axvline(threshold, color=KEPT, lw=1.6)
    ax_f.fill_betweenx([0, ax_f.get_ylim()[1]], threshold, values.max(), color=KEPT_LIGHT, alpha=0.35)
    ax_f.text(threshold, ax_f.get_ylim()[1] * 0.88, "P94.3", color=KEPT, ha="right", va="top", fontsize=5.8, fontweight="bold")
    ax_f.set_xlabel("smoothed response")
    ax_f.set_ylabel("density")
    ax_f.set_yticks([])
    ax_f.spines["left"].set_visible(False)
    add_panel_label(ax_f, "f")
    add_title(ax_f, "Image-wise threshold", "percentile adapts to weak contrast")

    # Scale-bar detection example.
    ax_g = fig.add_subplot(gs[1, 2])
    calibration = ctx.target_calibrations[label]
    native = ctx.target_images[label]
    bx0, by0, bx1, by1 = calibration.bar_bbox_xyxy or (0, 0, native.shape[1], native.shape[0])
    pad_x, pad_y = 90, 60
    sx0, sy0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
    sx1, sy1 = min(native.shape[1], bx1 + pad_x), min(native.shape[0], by1 + pad_y)
    ax_g.imshow(to_rgb(native[sy0:sy1, sx0:sx1]))
    ax_g.add_patch(Rectangle((bx0 - sx0, by0 - sy0), bx1 - bx0, by1 - by0, fill=False, edgecolor=KEPT, lw=1.3))
    clean_image_axis(ax_g)
    add_panel_label(ax_g, "g")
    add_title(ax_g, "Scale-bar audit", f"{calibration.scale_bar_pixels:.0f} px / 200 um = {calibration.pixels_per_um:.3f} px/um")

    ax_h = fig.add_subplot(gs[1, 3])
    labels = ["Target"] + [f"Cand. {chr(65 + i)}" for i in range(4)]
    ppu = [ctx.target_calibrations[label].pixels_per_um] + [
        ctx.candidate_calibrations[item].pixels_per_um for item in ctx.candidate_labels
    ]
    y = np.arange(len(labels))[::-1]
    ax_h.hlines(y, 0, ppu, color=GRID, lw=2)
    ax_h.scatter(ppu, y, c=[TARGET] + [AUX] * 4, s=28, zorder=3, edgecolor="white", linewidth=0.5)
    for yi, value in zip(y, ppu):
        ax_h.text(float(value) + 0.035, yi, f"{float(value):.3f}", va="center", fontsize=5.8, color=TEXT)
    ax_h.set_yticks(y, labels)
    ax_h.set_xlim(0, 1.95)
    ax_h.set_xlabel("pixels per um")
    ax_h.spines["left"].set_visible(False)
    ax_h.tick_params(axis="y", length=0)
    add_panel_label(ax_h, "h")
    add_title(ax_h, "Physical scale", "reported, excluded from ranking")
    ax_h.text(
        0.98,
        0.02,
        "report-only",
        transform=ax_h.transAxes,
        ha="right",
        va="bottom",
        color=REVIEW,
        fontweight="bold",
        fontsize=6.2,
    )
    return fig


def figure_03_support_topology(ctx: FigureContext) -> plt.Figure:
    """Observed central support selection and topology graph extraction."""

    label = ctx.candidate_labels[0]
    raw = ctx.raw_candidate_structures[label]
    selected = ctx.candidate_structures[label]
    radius = max(8, int(round(0.024 * min(raw.mask.shape))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grouped = cv2.dilate(raw.mask.astype(np.uint8), kernel) > 0
    bounds = crop_bounds(raw.mask.shape, raw.bbox, pad_fraction=0.04)
    topology = extract_skeleton_topology(selected.skeleton)

    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 5, left=0.055, right=0.98, bottom=0.09, top=0.92, wspace=0.13, hspace=0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(to_rgb(crop_array(raw.image, bounds)))
    clean_image_axis(ax_a)
    add_panel_label(ax_a, "a")
    add_title(ax_a, "Auxiliary input", "weak, fragmented observed strokes")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(mask_rgb(crop_array(raw.mask, bounds), foreground=TARGET_LIGHT))
    clean_image_axis(ax_b, dark=True)
    add_panel_label(ax_b, "b")
    add_title(ax_b, "All observed support", f"{int(raw.mask.sum()):,} foreground px")

    ax_c = fig.add_subplot(gs[0, 2])
    base = np.zeros((*raw.mask.shape, 3), dtype=float)
    base[:] = mpl.colors.to_rgb(DARK)
    base[grouped] = mpl.colors.to_rgb(AUX_LIGHT)
    base[raw.mask] = mpl.colors.to_rgb(TARGET)
    ax_c.imshow(crop_array((base * 255).astype(np.uint8), bounds))
    clean_image_axis(ax_c, dark=True)
    add_panel_label(ax_c, "c")
    add_title(ax_c, "Grouping support", f"dilation radius = {radius} px; selection only")

    ax_d = fig.add_subplot(gs[0, 3])
    selected_image = np.zeros((*raw.mask.shape, 3), dtype=float)
    selected_image[:] = mpl.colors.to_rgb(DARK)
    selected_image[raw.mask & ~selected.mask] = mpl.colors.to_rgb(REJECTED)
    selected_image[selected.mask] = mpl.colors.to_rgb(TARGET)
    ax_d.imshow(crop_array((selected_image * 255).astype(np.uint8), bounds))
    clean_image_axis(ax_d, dark=True)
    add_panel_label(ax_d, "d")
    add_title(ax_d, "Central observed support", "selected pixels are a strict subset")

    ax_e = fig.add_subplot(gs[0, 4])
    ax_e.set_facecolor(DARK)
    for segment in topology.segments:
        angle = float((segment.orientation_rad % math.pi) / math.pi)
        colors = [BLUE, AUX, TARGET, KEPT]
        color = colors[min(3, int(angle * 4))]
        pts = segment.points_yx
        ax_e.plot(pts[:, 1], pts[:, 0], color=color, lw=1.1, solid_capstyle="round")
    if topology.endpoint_yx:
        endpoints = np.asarray(topology.endpoint_yx)
        ax_e.scatter(endpoints[:, 1], endpoints[:, 0], s=7, facecolor=REVIEW, edgecolor="white", linewidth=0.25, zorder=6)
    if topology.junction_yx:
        junctions = np.asarray(topology.junction_yx)
        ax_e.scatter(junctions[:, 1], junctions[:, 0], s=11, marker="s", facecolor="white", edgecolor=INK, linewidth=0.35, zorder=6)
    x0, y0, x1, y1 = bounds
    ax_e.set_xlim(x0, x1)
    ax_e.set_ylim(y1, y0)
    clean_image_axis(ax_e, dark=True)
    add_panel_label(ax_e, "e")
    add_title(ax_e, "Topology graph", f"{len(topology.endpoint_yx)} endpoints; {len(topology.segments)} directional segments")

    ax_f = fig.add_subplot(gs[1, :3])
    ax_f.axis("off")
    add_panel_label(ax_f, "f", inside=True)
    ax_f.text(0.10, 0.92, "Observed-pixel guarantee", transform=ax_f.transAxes, fontsize=8, fontweight="bold", color=TEXT)
    steps = [
        ("raw mask", int(raw.mask.sum()), TARGET),
        ("grouping dilation", int(grouped.sum()), AUX_LIGHT),
        ("selected raw pixels", int(selected.mask.sum()), KEPT),
        ("skeleton", int(selected.skeleton.sum()), INK),
    ]
    x_positions = [0.09, 0.34, 0.60, 0.84]
    maximum = max(value for _, value, _ in steps)
    for i, ((name, value, color), x) in enumerate(zip(steps, x_positions)):
        radius_plot = 0.055 + 0.035 * math.sqrt(value / maximum)
        circle = plt.Circle((x, 0.47), radius_plot, transform=ax_f.transAxes, facecolor=color, edgecolor="white", lw=0.7)
        ax_f.add_patch(circle)
        ax_f.text(x, 0.47, f"{value:,}", transform=ax_f.transAxes, ha="center", va="center", fontsize=5.7, color="white" if color in {TARGET, KEPT, INK} else TEXT, fontweight="bold")
        ax_f.text(x, 0.23, name, transform=ax_f.transAxes, ha="center", fontsize=5.8, color=TEXT)
        if i < 3:
            ax_f.annotate("", xy=(x_positions[i + 1] - 0.07, 0.47), xytext=(x + 0.07, 0.47), xycoords=ax_f.transAxes, arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.9})
    ax_f.text(
        0.50,
        0.05,
        "Dilation selects a spatial group; it never becomes template evidence.",
        transform=ax_f.transAxes,
        ha="center",
        color=KEPT,
        fontsize=6.2,
        fontweight="bold",
    )

    ax_g = fig.add_subplot(gs[1, 3:])
    metrics = []
    for item in ctx.candidate_labels:
        topo = extract_skeleton_topology(ctx.candidate_structures[item].skeleton)
        metrics.append((item, topo.skeleton_length_px, len(topo.endpoint_yx), len(topo.segments)))
    x = np.arange(4)
    lengths = np.asarray([m[1] for m in metrics])
    endpoints = np.asarray([m[2] for m in metrics])
    ax_g.bar(x - 0.18, lengths / lengths.max(), width=0.36, color=AUX, label="skeleton length (norm.)")
    ax_g.bar(x + 0.18, endpoints / max(endpoints), width=0.36, color=REVIEW, label="endpoint count (norm.)")
    ax_g.set_xticks(x, ["A", "B", "C", "D"])
    ax_g.set_ylim(0, 1.14)
    ax_g.set_yticks([0, 0.5, 1.0])
    ax_g.set_ylabel("normalized topology measure")
    ax_g.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    add_panel_label(ax_g, "g")
    add_title(ax_g, "Candidate topology audit", "counts are descriptive, not biological nodes")
    return fig


def geometry_landscape(
    selection: SelectedMatch,
    *,
    span_px: float = 26.0,
    count: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = selection.target
    auxiliary = selection.auxiliary
    config = UnifiedSearchConfig()
    target_soft = cv2.GaussianBlur(target.response.astype(np.float32), (0, 0), 2.2)
    target_distance = cv2.distanceTransform((~target.mask).astype(np.uint8), cv2.DIST_L2, 3)
    auxiliary_distance = cv2.distanceTransform((~auxiliary.mask).astype(np.uint8), cv2.DIST_L2, 3)
    target_angle, target_coherence = orientation_fields(target_soft)
    auxiliary_angle, _ = orientation_fields(auxiliary.skeleton.astype(np.float32))
    source_points = _skeleton_points(auxiliary, maximum=640)
    target_points = _skeleton_points(target, maximum=720)
    source_angles = _bilinear_sample(auxiliary_angle, source_points)
    dxs = np.linspace(selection.match.dx - span_px, selection.match.dx + span_px, count)
    dys = np.linspace(selection.match.dy - span_px, selection.match.dy + span_px, count)
    scores = np.empty((count, count), dtype=float)
    for yi, dy in enumerate(dys):
        for xi, dx in enumerate(dxs):
            values = np.asarray([math.log(selection.match.scale), selection.match.angle_deg, dx, dy], dtype=float)
            scores[yi, xi] = _geometry_score(
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
                None,
                0.0,
                config,
            ).score
    return dxs, dys, scores


def figure_04_registration(ctx: FigureContext) -> plt.Figure:
    """One-transform coarse-to-fine registration with a real local landscape."""

    selection = ctx.representative
    target = selection.target
    source_points = np.argwhere(selection.auxiliary.skeleton)[:, ::-1].astype(np.float32)
    initial_points = transform_points(
        source_points,
        selection.auxiliary.bbox,
        selection.match.coarse_scale,
        selection.match.coarse_angle_deg,
        selection.match.coarse_dx,
        selection.match.coarse_dy,
    )
    final_points = transformed_points(selection)
    bounds = crop_bounds(target.mask.shape, target.bbox, pad_fraction=0.10)
    dxs, dys, scores = geometry_landscape(selection)

    fig = plt.figure(figsize=(7.2, 4.75), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, bottom=0.10, top=0.92, wspace=0.27, hspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    add_panel_label(ax_a, "a", inside=True)
    ax_a.text(0.10, 0.92, "Similarity transform", transform=ax_a.transAxes, fontweight="bold", fontsize=8, color=TEXT)
    ax_a.text(0.02, 0.68, "x' = s R(theta)(x - c)", transform=ax_a.transAxes, fontsize=8.2, color=INK)
    ax_a.text(0.02, 0.54, "+ c + [dx, dy]", transform=ax_a.transAxes, fontsize=8.2, color=INK)
    blocks = [("scale", "s"), ("rotation", "theta"), ("translation", "dx, dy")]
    for i, (name, symbol) in enumerate(blocks):
        y = 0.33 - i * 0.12
        ax_a.add_patch(FancyBboxPatch((0.02, y), 0.24, 0.075, boxstyle="round,pad=0.01", transform=ax_a.transAxes, facecolor=PANEL, edgecolor=GRID))
        ax_a.text(0.14, y + 0.038, symbol, transform=ax_a.transAxes, ha="center", va="center", fontweight="bold", color=AUX)
        ax_a.text(0.31, y + 0.038, name, transform=ax_a.transAxes, va="center", color=MUTED)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(crop_array(overlay_target_auxiliary(target.mask, source_points), bounds))
    clean_image_axis(ax_b, dark=True)
    add_panel_label(ax_b, "b")
    add_title(ax_b, "Before registration", "candidate skeleton in native analysis coordinates")

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(crop_array(overlay_target_auxiliary(target.mask, initial_points), bounds))
    clean_image_axis(ax_c, dark=True)
    add_panel_label(ax_c, "c")
    add_title(ax_c, "Best coarse seed", f"s={selection.match.coarse_scale:.3f}, theta={selection.match.coarse_angle_deg:.1f} deg")

    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.imshow(crop_array(overlay_target_auxiliary(target.mask, final_points, corridor_radius=12), bounds))
    clean_image_axis(ax_d, dark=True)
    add_panel_label(ax_d, "d")
    add_title(ax_d, "Refined registration", f"s={selection.match.scale:.3f}, theta={selection.match.angle_deg:.2f} deg")

    ax_e = fig.add_subplot(gs[1, :2])
    levels = np.linspace(scores.min(), scores.max(), 16)
    contour = ax_e.contourf(dxs, dys, scores, levels=levels, cmap=SCORE_CMAP)
    ax_e.contour(dxs, dys, scores, levels=levels[::3], colors="white", linewidths=0.35, alpha=0.75)
    ax_e.plot(selection.match.coarse_dx, selection.match.coarse_dy, marker="o", ms=5, color=REVIEW, label="coarse seed")
    ax_e.plot([selection.match.coarse_dx, selection.match.dx], [selection.match.coarse_dy, selection.match.dy], color="white", lw=1.2, ls="--")
    ax_e.plot(selection.match.dx, selection.match.dy, marker="*", ms=8, color=KEPT, mec="white", mew=0.5, label="refined optimum")
    ax_e.set_xlabel("translation dx (analysis px)")
    ax_e.set_ylabel("translation dy (analysis px)")
    ax_e.legend(loc="lower right")
    cbar = fig.colorbar(contour, ax=ax_e, fraction=0.035, pad=0.02)
    cbar.set_label("geometry score G", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    add_panel_label(ax_e, "e")
    add_title(ax_e, "Local objective landscape", "fixed at refined scale and angle; measured from the production score")

    ax_f = fig.add_subplot(gs[1, 2:])
    ax_f.axis("off")
    add_panel_label(ax_f, "f", inside=True)
    ax_f.text(0.10, 0.94, "Winning transforms", transform=ax_f.transAxes, fontsize=8, fontweight="bold", color=TEXT)
    headers = ["target", "s", "theta", "dx", "dy", "audit"]
    x_positions = [0.03, 0.22, 0.35, 0.50, 0.64, 0.80]
    for x, header in zip(x_positions, headers):
        ax_f.text(x, 0.79, header, transform=ax_f.transAxes, color=MUTED, fontsize=5.6, fontweight="bold")
    for i, row in enumerate(ctx.summary_rows):
        y = 0.64 - i * 0.145
        status = str(row["decision_status"]).replace("review_required_", "")
        values = [
            str(i + 1),
            f"{float(row['analysis_scale']):.3f}",
            f"{float(row['analysis_angle_deg']):.2f}",
            f"{float(row['analysis_dx']):.1f}",
            f"{float(row['analysis_dy']):.1f}",
            status,
        ]
        ax_f.add_patch(Rectangle((0.01, y - 0.045), 0.97, 0.11, transform=ax_f.transAxes, facecolor=PANEL if i % 2 == 0 else PAPER, edgecolor="none"))
        for x, value in zip(x_positions, values):
            ax_f.text(x, y, value, transform=ax_f.transAxes, fontsize=5.7, color=REVIEW if x == x_positions[-1] else TEXT, va="center")
    ax_f.text(
        0.02,
        0.02,
        "Boundary hits or flat optima trigger review; optimized does not mean globally validated.",
        transform=ax_f.transAxes,
        fontsize=5.8,
        color=KEPT,
        fontweight="bold",
    )
    return fig


def figure_05_score_math(ctx: FigureContext) -> plt.Figure:
    """Mathematical candidate-selection figure requested for the paper."""

    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=False)
    gs = fig.add_gridspec(2, 6, left=0.06, right=0.98, bottom=0.10, top=0.93, wspace=0.72, hspace=0.42)

    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.axis("off")
    add_panel_label(ax_a, "a", inside=True)
    ax_a.text(0.10, 0.95, "Hierarchical score", transform=ax_a.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    equations = [
        ("GEOMETRY", "G = 0.31 support\n   + 0.29 forward\n   + 0.28 reverse\n   + 0.12 orientation", AUX, "#EAF4F4"),
        ("TOPOLOGY", "T = 0.30 coverage + 0.20 direction\n   + 0.25(1 - missing)\n   + 0.15(1 - unexplained)\n   + 0.10 endpoint coverage", TARGET, "#F8F0E5"),
        ("FINAL", "F = 0.72 G + 0.28 T", KEPT, "#FAECE8"),
    ]
    y_positions = [0.68, 0.32, 0.06]
    heights = [0.23, 0.29, 0.16]
    for (name, equation, color, face), y, height in zip(equations, y_positions, heights):
        box = FancyBboxPatch(
            (0.02, y),
            0.96,
            height,
            transform=ax_a.transAxes,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=face,
            edgecolor=color,
            linewidth=0.9,
        )
        ax_a.add_patch(box)
        ax_a.text(0.07, y + height - 0.035, name, transform=ax_a.transAxes, fontsize=5.3, color=color, fontweight="bold", va="top")
        ax_a.text(0.07, y + height - 0.085, equation, transform=ax_a.transAxes, fontsize=4.9, color=INK, va="top", linespacing=1.18)
    ax_a.annotate("", xy=(0.50, 0.62), xytext=(0.50, 0.66), xycoords=ax_a.transAxes, arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8})
    ax_a.annotate("", xy=(0.50, 0.22), xytext=(0.50, 0.31), xycoords=ax_a.transAxes, arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8})

    ax_b = fig.add_subplot(gs[0, 2:])
    im = ax_b.imshow(ctx.score_matrix, cmap=SCORE_CMAP, vmin=0.28, vmax=0.56, aspect="auto")
    ax_b.set_xticks(range(4), [f"{chr(65 + i)} ({label})" for i, label in enumerate(ctx.candidate_labels)])
    ax_b.set_yticks(range(4), [f"{i + 1} ({label})" for i, label in enumerate(ctx.target_labels)])
    ax_b.set_xlabel("candidate (post hoc code in parentheses)")
    ax_b.set_ylabel("")
    for i in range(4):
        winner = int(np.argmax(ctx.score_matrix[i]))
        for j in range(4):
            value = ctx.score_matrix[i, j]
            text_color = "white" if value > 0.43 else TEXT
            ax_b.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=7.2, fontweight="bold" if j == winner else "normal")
        ax_b.add_patch(Rectangle((winner - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=KEPT, lw=2.0))
        ax_b.plot(winner + 0.37, i - 0.37, marker="o", ms=3.3, color=KEPT, mec="white", mew=0.4)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.025, pad=0.025)
    cbar.set_label("final score F", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    add_panel_label(ax_b, "b")
    add_title(ax_b, "All 16 measured candidate scores", "one argmax per row; no one-to-one batch assignment")

    ax_c = fig.add_subplot(gs[1, :3])
    y = np.arange(4)[::-1]
    for display_y, row in zip(y, ctx.summary_rows):
        best = float(row["selected_score"])
        second = float(row["runner_up_score"])
        margin = float(row["margin"])
        color = REVIEW if margin < 0.025 else GOOD
        ax_c.plot([second, best], [display_y, display_y], color=GRID, lw=3, solid_capstyle="round")
        ax_c.scatter(second, display_y, s=25, facecolor=PAPER, edgecolor=MUTED, lw=1.0, zorder=3)
        ax_c.scatter(best, display_y, s=34, facecolor=color, edgecolor="white", lw=0.6, zorder=4)
        ax_c.text(second - 0.006, display_y, f"{second:.3f}", ha="right", va="center", fontsize=5.7, color=MUTED)
        ax_c.text(best + 0.006, display_y, f"{best:.3f}", ha="left", va="center", fontsize=5.7, color=color, fontweight="bold")
        midpoint = 0.5 * (best + second)
        ax_c.text(midpoint, display_y + 0.17, f"margin {margin:.4f}", ha="center", va="bottom", fontsize=5.6, color=color, fontweight="bold")
    ax_c.set_yticks(y, [f"Target {i + 1}" for i in range(4)])
    ax_c.set_xlim(0.27, 0.65)
    ax_c.set_ylim(-0.35, 3.42)
    ax_c.set_xlabel("final score F")
    ax_c.tick_params(axis="y", length=0)
    ax_c.spines["left"].set_visible(False)
    ax_c.scatter([], [], s=34, color=GOOD, label="margin >= 0.025")
    ax_c.scatter([], [], s=34, color=REVIEW, label="review: margin < 0.025")
    ax_c.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, -0.30))
    add_panel_label(ax_c, "c")
    add_title(ax_c, "Winner versus runner-up", "margin is a review trigger, not a probability")

    ax_d = fig.add_subplot(gs[1, 3:])
    geometry = []
    topology = []
    finals = []
    for i, row in enumerate(ctx.summary_rows):
        target_label = ctx.target_labels[i]
        pair = ctx.pair(target_label, str(row["selected_label"]))
        geometry.append(float(pair["geometry_score"]))
        topology.append(float(pair["topology_score"]))
        finals.append(float(pair["final_score"]))
    geometry_contribution = 0.72 * np.asarray(geometry)
    topology_contribution = 0.28 * np.asarray(topology)
    x = np.arange(4)
    ax_d.bar(x, geometry_contribution, color=AUX, width=0.62, label="0.72 x geometry")
    ax_d.bar(x, topology_contribution, bottom=geometry_contribution, color=TARGET_LIGHT, width=0.62, label="0.28 x topology")
    for xi, value in zip(x, finals):
        ax_d.text(xi, value + 0.009, f"{value:.3f}", ha="center", fontsize=5.8, color=TEXT, fontweight="bold")
    ax_d.set_xticks(x, [f"Target {i + 1}" for i in range(4)])
    ax_d.set_ylim(0, 0.62)
    ax_d.set_ylabel("additive contribution to F")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    add_panel_label(ax_d, "d")
    add_title(ax_d, "Winning-score decomposition", "physical scale is report-only and absent from F")
    return fig


def candidate_overlay_from_row(
    ctx: FigureContext,
    target_label: str,
    candidate_label: str,
) -> tuple[np.ndarray, dict[str, object]]:
    target = ctx.target_structures[target_label]
    candidate = ctx.candidate_structures[candidate_label]
    row = ctx.pair(target_label, candidate_label)
    points = transformed_points_from_row(candidate, row)
    return overlay_target_auxiliary(target.mask, points, corridor_radius=12), row


def figure_06_topology_tie(ctx: FigureContext) -> plt.Figure:
    """Explain the T/Z near tie without implying high-confidence validation."""

    target_label = ctx.target_labels[1]
    winner_label = ctx.candidate_labels[1]
    rival_label = ctx.candidate_labels[3]
    target = ctx.target_structures[target_label]
    winner_overlay, winner = candidate_overlay_from_row(ctx, target_label, winner_label)
    rival_overlay, rival = candidate_overlay_from_row(ctx, target_label, rival_label)
    bounds = crop_bounds(target.mask.shape, target.bbox, pad_fraction=0.08)

    fig = plt.figure(figsize=(7.2, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, bottom=0.11, top=0.92, wspace=0.36, hspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(crop_array(winner_overlay, bounds))
    clean_image_axis(ax_a, dark=True)
    add_panel_label(ax_a, "a")
    add_title(ax_a, "Candidate B (T)", f"G={float(winner['geometry_score']):.4f}; T={float(winner['topology_score']):.4f}")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(crop_array(rival_overlay, bounds))
    clean_image_axis(ax_b, dark=True)
    add_panel_label(ax_b, "b")
    add_title(ax_b, "Candidate D (Z)", f"G={float(rival['geometry_score']):.4f}; T={float(rival['topology_score']):.4f}")

    ax_c = fig.add_subplot(gs[0, 2:])
    metrics = [
        ("topology", "topology_score", False),
        ("stroke kept", "missing_stroke_penalty", True),
        ("explained", "unexplained_target_evidence_penalty", True),
        ("endpoints", "endpoint_coverage", False),
    ]
    y = np.arange(len(metrics))[::-1]
    w_values = []
    r_values = []
    for _, key, invert in metrics:
        w = float(winner[key])
        r = float(rival[key])
        w_values.append(1.0 - w if invert else w)
        r_values.append(1.0 - r if invert else r)
    for yi, w, r in zip(y, w_values, r_values):
        ax_c.plot([w, r], [yi, yi], color=GRID, lw=2.2)
        ax_c.scatter(w, yi, s=33, color=AUX, edgecolor="white", lw=0.5, zorder=3)
        ax_c.scatter(r, yi, s=33, color=TARGET, edgecolor="white", lw=0.5, zorder=3)
    ax_c.set_yticks(y, [item[0] for item in metrics])
    ax_c.set_xlim(0, 0.55)
    ax_c.set_xlabel("higher is better")
    ax_c.tick_params(axis="y", length=0, pad=2, labelsize=5.8)
    ax_c.spines["left"].set_visible(False)
    ax_c.scatter([], [], s=30, color=AUX, label="Candidate B (T)")
    ax_c.scatter([], [], s=30, color=TARGET, label="Candidate D (Z)")
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    add_panel_label(ax_c, "c")
    add_title(ax_c, "Topology evidence", "penalties shown as their positive complements")

    ax_d = fig.add_subplot(gs[1, :2])
    stages = np.array([0, 1])
    winner_scores = [float(winner["geometry_score"]), float(winner["final_score"])]
    rival_scores = [float(rival["geometry_score"]), float(rival["final_score"])]
    ax_d.plot(stages, winner_scores, color=AUX, marker="o", ms=5, label="Candidate B (T)")
    ax_d.plot(stages, rival_scores, color=TARGET, marker="o", ms=5, label="Candidate D (Z)")
    for x, values, color in [(0, [winner_scores[0], rival_scores[0]], MUTED), (1, [winner_scores[1], rival_scores[1]], KEPT)]:
        for value in values:
            ax_d.text(x + 0.04, value, f"{value:.4f}", va="center", fontsize=5.7, color=color)
    ax_d.set_xticks(stages, ["geometry G", "final F\n(+ topology)"])
    ax_d.set_xlim(-0.10, 1.35)
    ax_d.set_ylim(0.415, 0.452)
    ax_d.set_ylabel("score")
    ax_d.legend(loc="lower left")
    add_panel_label(ax_d, "d")
    add_title(ax_d, "Topology reverses the geometry-only order", "the final difference remains deliberately flagged as low margin")

    ax_e = fig.add_subplot(gs[1, 2:])
    ax_e.axis("off")
    add_panel_label(ax_e, "e", inside=True)
    margin = float(winner["final_score"]) - float(rival["final_score"])
    cards = [
        ("geometry", float(winner["geometry_score"]) - float(rival["geometry_score"]), "D favors Z"),
        ("topology", float(winner["topology_score"]) - float(rival["topology_score"]), "B favors T"),
        ("final", margin, "B wins; review"),
    ]
    ax_e.text(0.10, 0.93, "Near-tie audit", transform=ax_e.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    for i, (name, delta, verdict) in enumerate(cards):
        y0 = 0.66 - i * 0.24
        face = "#FAECE8" if name == "final" else PANEL
        ax_e.add_patch(FancyBboxPatch((0.03, y0), 0.94, 0.17, boxstyle="round,pad=0.012", transform=ax_e.transAxes, facecolor=face, edgecolor=GRID))
        ax_e.text(0.07, y0 + 0.115, name.upper(), transform=ax_e.transAxes, fontsize=5.3, color=MUTED, fontweight="bold")
        ax_e.text(0.34, y0 + 0.085, f"delta = {delta:+.5f}", transform=ax_e.transAxes, fontsize=7, color=KEPT if name == "final" else INK, fontweight="bold")
        ax_e.text(0.72, y0 + 0.085, verdict, transform=ax_e.transAxes, fontsize=5.8, color=REVIEW if name == "final" else MUTED)
    ax_e.text(0.5, 0.02, "F is a ranking score, not a calibrated probability.", transform=ax_e.transAxes, ha="center", color=KEPT, fontsize=6.2, fontweight="bold")
    return fig


def selected_binary(selection: SelectedMatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = transformed_points(selection)
    result = matched_only_mask(selection.target.mask, points, corridor_radius_px=12)
    return result.mask, result.corridor, points


def figure_07_corridor(ctx: FigureContext) -> plt.Figure:
    """Demonstrate the strict target-mask/intersection output rule."""

    selection = ctx.representative
    output, corridor, points = selected_binary(selection)
    target = selection.target
    bounds = crop_bounds(target.mask.shape, target.bbox, pad_fraction=0.08)

    skeleton_canvas = np.zeros(target.mask.shape, dtype=bool)
    rounded = np.rint(points).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target.mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target.mask.shape[0])
    )
    skeleton_canvas[rounded[inside, 1], rounded[inside, 0]] = True
    audit_rgb = np.zeros((*target.mask.shape, 3), dtype=float)
    audit_rgb[:] = mpl.colors.to_rgb(DARK)
    audit_rgb[target.mask & ~corridor] = mpl.colors.to_rgb(REJECTED)
    audit_rgb[corridor & ~target.mask] = np.asarray(mpl.colors.to_rgb(AUX)) * 0.35
    audit_rgb[output] = mpl.colors.to_rgb(KEPT)
    audit_rgb[skeleton_canvas] = mpl.colors.to_rgb(AUX_LIGHT)

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=False)
    gs = fig.add_gridspec(2, 5, left=0.055, right=0.98, bottom=0.10, top=0.92, wspace=0.14, hspace=0.40)
    skeleton_display = ndi.binary_dilation(skeleton_canvas, iterations=1)
    image_panels = [
        (target.mask, "Target mask", TARGET_LIGHT, "all measured target evidence"),
        (skeleton_display, "Registered skeleton", AUX, "1-px display dilation only"),
        (corridor, "Dilated corridor", AUX_LIGHT, "radius = 12 analysis px"),
        (output, "Strict intersection", KEPT, "M_out = M_target AND corridor"),
    ]
    for i, (array, title, color, subtitle) in enumerate(image_panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mask_rgb(crop_array(array, bounds), foreground=color))
        clean_image_axis(ax, dark=True)
        add_panel_label(ax, chr(ord("a") + i))
        add_title(ax, title, subtitle)

    ax_e = fig.add_subplot(gs[0, 4])
    ax_e.imshow(crop_array((audit_rgb * 255).astype(np.uint8), bounds))
    clean_image_axis(ax_e, dark=True)
    add_panel_label(ax_e, "e")
    add_title(ax_e, "Pixel-source audit", "coral kept; gray rejected; cyan guide")

    ax_f = fig.add_subplot(gs[1, :3])
    ax_f.axis("off")
    add_panel_label(ax_f, "f", inside=True)
    ax_f.text(0.10, 0.90, "Set-theoretic guardrail", transform=ax_f.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    ax_f.text(0.05, 0.63, "M_out = M_target AND Dilate(T*(Skel_aux), r=12)", transform=ax_f.transAxes, fontsize=8.3, color=INK)
    invariants = [
        "M_out is a subset of M_target",
        "M_out is a subset of the registered corridor",
        "unique output values are exactly {0, 255}",
    ]
    for i, text in enumerate(invariants):
        y = 0.40 - i * 0.13
        ax_f.scatter(0.08, y, s=34, marker="o", color=GOOD, transform=ax_f.transAxes)
        ax_f.text(0.08, y, "OK", transform=ax_f.transAxes, ha="center", va="center", fontsize=4.2, color="white", fontweight="bold")
        ax_f.text(0.13, y, text, transform=ax_f.transAxes, va="center", fontsize=6.2, color=TEXT)
    ax_f.text(0.05, 0.01, "No template pixel is copied and no missing stroke is completed.", transform=ax_f.transAxes, fontsize=6.4, color=KEPT, fontweight="bold")

    ax_g = fig.add_subplot(gs[1, 3:])
    retention = []
    for label in ctx.target_labels:
        selected = ctx.selection_by_label[label]
        mask, _, _ = selected_binary(selected)
        retention.append(float(mask.sum()) / max(float(selected.target.mask.sum()), 1.0))
    x = np.arange(4)
    bars = ax_g.bar(x, np.asarray(retention) * 100, color=[AUX, REVIEW, AUX, REVIEW], width=0.64)
    for bar, value in zip(bars, retention):
        ax_g.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1.6, f"{value * 100:.1f}%", ha="center", fontsize=6, color=TEXT, fontweight="bold")
    ax_g.set_xticks(x, [f"Target {i + 1}" for i in range(4)])
    ax_g.set_ylim(0, 88)
    ax_g.set_ylabel("target foreground retained (%)")
    ax_g.axhline(50, color=GRID, lw=0.8, ls="--")
    add_panel_label(ax_g, "g")
    add_title(ax_g, "Corridor retention", "descriptive coverage, not accuracy")
    return fig


def analysis_to_native_bounds(
    bounds: tuple[int, int, int, int],
    analysis_shape: tuple[int, int],
    native_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bounds
    ah, aw = analysis_shape
    nh, nw = native_shape
    return (
        max(0, int(math.floor(x0 * nw / aw))),
        max(0, int(math.floor(y0 * nh / ah))),
        min(nw, int(math.ceil(x1 * nw / aw))),
        min(nh, int(math.ceil(y1 * nh / ah))),
    )


def figure_08_results_audit(ctx: FigureContext) -> plt.Figure:
    """Four end-to-end results with explicit decision-review status."""

    fig = plt.figure(figsize=(7.2, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        6,
        left=0.065,
        right=0.985,
        bottom=0.08,
        top=0.92,
        wspace=0.08,
        hspace=0.16,
        width_ratios=[1.02, 1.02, 1.02, 1.02, 1.02, 1.42],
    )
    headers = ["target", "selected candidate", "registered overlay", "target evidence", "final binary", "decision audit"]
    for j, header in enumerate(headers):
        fig.text(0.065 + sum([1.02, 1.02, 1.02, 1.02, 1.02, 1.42][:j]) / 6.52 * 0.92, 0.945, header.upper(), fontsize=5.5, color=MUTED, fontweight="bold", ha="left")

    for i, label in enumerate(ctx.target_labels):
        selection = ctx.selection_by_label[label]
        summary = ctx.summary_rows[i]
        target = selection.target
        candidate = selection.auxiliary
        points = transformed_points(selection)
        output, _, _ = selected_binary(selection)
        target_bounds = crop_bounds(target.mask.shape, target.bbox, pad_fraction=0.08)
        candidate_bounds = crop_bounds(candidate.mask.shape, candidate.bbox, pad_fraction=0.08)

        ax_target = fig.add_subplot(gs[i, 0])
        ax_target.imshow(to_rgb(crop_array(target.image, target_bounds)))
        clean_image_axis(ax_target)

        ax_candidate = fig.add_subplot(gs[i, 1])
        ax_candidate.imshow(to_rgb(crop_array(candidate.image, candidate_bounds)))
        clean_image_axis(ax_candidate)

        ax_overlay = fig.add_subplot(gs[i, 2])
        ax_overlay.imshow(crop_array(overlay_target_auxiliary(target.mask, points, corridor_radius=12), target_bounds))
        clean_image_axis(ax_overlay, dark=True)

        ax_present = fig.add_subplot(gs[i, 3])
        native_bounds = analysis_to_native_bounds(target_bounds, target.mask.shape, selection.rendered.shape[:2])
        ax_present.imshow(to_rgb(crop_array(selection.rendered, native_bounds)))
        clean_image_axis(ax_present)

        ax_binary = fig.add_subplot(gs[i, 4])
        native_binary = native_binary_image(output, selection.target_original.shape[:2])
        native_binary_bounds = analysis_to_native_bounds(target_bounds, target.mask.shape, native_binary.shape[:2])
        ax_binary.imshow(crop_array(native_binary, native_binary_bounds), cmap="gray", vmin=0, vmax=255)
        clean_image_axis(ax_binary, dark=True)

        ax_card = fig.add_subplot(gs[i, 5])
        ax_card.axis("off")
        status = str(summary["decision_status"]).replace("review_required_", "")
        selected_label = str(summary["selected_label"])
        candidate_index = ctx.candidate_labels.index(selected_label)
        margin = float(summary["margin"])
        card_color = REVIEW
        ax_card.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88, boxstyle="round,pad=0.018", transform=ax_card.transAxes, facecolor="#FCF6E8", edgecolor="#E7C77D", linewidth=0.8))
        ax_card.text(0.08, 0.81, f"Candidate {chr(65 + candidate_index)}  ({selected_label})", transform=ax_card.transAxes, fontsize=7.2, fontweight="bold", color=TEXT)
        ax_card.text(0.08, 0.63, f"F = {float(summary['selected_score']):.4f}", transform=ax_card.transAxes, fontsize=6.2, color=INK)
        ax_card.text(0.08, 0.50, f"margin = {margin:.4f}", transform=ax_card.transAxes, fontsize=6.2, color=card_color if margin < 0.025 else GOOD, fontweight="bold")
        ax_card.text(0.08, 0.35, f"s={float(summary['analysis_scale']):.3f}; theta={float(summary['analysis_angle_deg']):.2f} deg", transform=ax_card.transAxes, fontsize=5.4, color=MUTED)
        ax_card.text(0.08, 0.20, status.replace("_", " "), transform=ax_card.transAxes, fontsize=5.6, color=card_color, fontweight="bold")
        ax_card.text(0.08, 0.09, "human review required", transform=ax_card.transAxes, fontsize=5.2, color=card_color)

        fig.text(0.014, 0.827 - i * 0.209, f"Target {i + 1}\n({label})", fontsize=6.2, color=TEXT, fontweight="bold", ha="left", va="center")
        if i == 0:
            add_panel_label(ax_target, "a")
            add_panel_label(ax_candidate, "b")
            add_panel_label(ax_overlay, "c")
            add_panel_label(ax_present, "d")
            add_panel_label(ax_binary, "e")
            add_panel_label(ax_card, "f", inside=True)

    fig.text(
        0.52,
        0.025,
        "Parenthetical S/T/U/Z codes are shown only for post hoc audit and are not used by the matcher.  All four outputs remain automatic candidates requiring review.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=KEPT,
        fontweight="bold",
    )
    return fig


FIGURE_CONTRACTS = [
    {
        "number": 1,
        "stem": "fig01_matching_workflow",
        "archetype": "schematic-led composite",
        "conclusion": "One label-free transform links localization, candidate ranking and target-evidence-only rendering.",
    },
    {
        "number": 2,
        "stem": "fig02_signal_and_scale",
        "archetype": "image plate + quant",
        "conclusion": "Local darkness and red excess recover weak structure while physical scale remains an independent audit variable.",
    },
    {
        "number": 3,
        "stem": "fig03_support_and_topology",
        "archetype": "asymmetric mixed-modality figure",
        "conclusion": "Central support selection and graph extraction use only observed pixels and expose endpoints and directional segments for audit.",
    },
    {
        "number": 4,
        "stem": "fig04_registration_landscape",
        "archetype": "schematic-led composite",
        "conclusion": "A coarse-to-fine similarity transform jointly determines scoring, localization and rendering.",
    },
    {
        "number": 5,
        "stem": "fig05_template_score_math",
        "archetype": "quantitative grid",
        "conclusion": "Geometry and topology jointly determine the independent candidate argmax, while the winner-runner-up margin triggers review.",
    },
    {
        "number": 6,
        "stem": "fig06_topology_near_tie",
        "archetype": "asymmetric mixed-modality figure",
        "conclusion": "Topology reverses the geometry-only ordering in the T/Z near tie, but the residual margin remains too small for automatic acceptance.",
    },
    {
        "number": 7,
        "stem": "fig07_nonfabricating_corridor",
        "archetype": "image plate + quant",
        "conclusion": "The binary result is strictly the intersection of target evidence and the registered auxiliary corridor.",
    },
    {
        "number": 8,
        "stem": "fig08_results_and_audit",
        "archetype": "image plate + quant",
        "conclusion": "All four examples receive interpretable candidate outputs, and every result is correctly retained for human review.",
    },
]


FIGURE_FUNCTIONS = {
    1: figure_01_workflow,
    2: figure_02_signal_scale,
    3: figure_03_support_topology,
    4: figure_04_registration,
    5: figure_05_score_math,
    6: figure_06_topology_tie,
    7: figure_07_corridor,
    8: figure_08_results_audit,
}


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_source_data(ctx: FigureContext, outdir: Path) -> None:
    source = outdir / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    pair_fields = [
        "target_id",
        "candidate_id",
        "candidate_label",
        "final_score",
        "geometry_score",
        "topology_score",
        "support",
        "forward_similarity",
        "reverse_similarity",
        "orientation",
        "missing_stroke_penalty",
        "unexplained_target_evidence_penalty",
        "endpoint_coverage",
        "analysis_scale",
        "analysis_angle_deg",
        "analysis_dx",
        "analysis_dy",
        "coarse_scale",
        "coarse_angle_deg",
        "coarse_dx",
        "coarse_dy",
        "coarse_boundary_hit",
        "fine_boundary_hit",
        "status_flags",
    ]
    write_csv(source / "pair_scores.csv", ctx.pair_rows, pair_fields)
    summary_fields = [
        "target_id",
        "selected_candidate_id",
        "selected_label",
        "runner_up_candidate_id",
        "runner_up_label",
        "selected_score",
        "runner_up_score",
        "margin",
        "decision_status",
        "analysis_scale",
        "analysis_angle_deg",
        "analysis_dx",
        "analysis_dy",
        "physical_scale_available",
        "physical_scale_mode",
        "topology_score",
        "status_flags",
        "rendered_from_target_evidence_only",
    ]
    write_csv(source / "selection_summary.csv", ctx.summary_rows, summary_fields)

    calibration_rows: list[dict[str, object]] = []
    for group, calibrations in [("target", ctx.target_calibrations), ("candidate", ctx.candidate_calibrations)]:
        for label, item in calibrations.items():
            calibration_rows.append(
                {
                    "group": group,
                    "post_hoc_label": label,
                    "pixels_per_um": item.pixels_per_um,
                    "scale_bar_length_um": item.scale_bar_length_um,
                    "scale_bar_pixels": item.scale_bar_pixels,
                    "confidence": item.confidence,
                    "source": item.source,
                    "failure_reason": item.failure_reason,
                }
            )
    write_csv(
        source / "physical_calibration.csv",
        calibration_rows,
        ["group", "post_hoc_label", "pixels_per_um", "scale_bar_length_um", "scale_bar_pixels", "confidence", "source", "failure_reason"],
    )

    retention_rows = []
    for i, label in enumerate(ctx.target_labels):
        selection = ctx.selection_by_label[label]
        output, corridor, _ = selected_binary(selection)
        retention_rows.append(
            {
                "target_id": f"target_{i + 1:02d}",
                "post_hoc_label": label,
                "target_foreground_px": int(selection.target.mask.sum()),
                "corridor_px": int(corridor.sum()),
                "output_foreground_px": int(output.sum()),
                "retained_fraction": float(output.sum()) / max(float(selection.target.mask.sum()), 1.0),
                "output_subset_target": bool(np.all(output <= selection.target.mask)),
                "output_subset_corridor": bool(np.all(output <= corridor)),
            }
        )
    write_csv(
        source / "corridor_retention.csv",
        retention_rows,
        [
            "target_id",
            "post_hoc_label",
            "target_foreground_px",
            "corridor_px",
            "output_foreground_px",
            "retained_fraction",
            "output_subset_target",
            "output_subset_corridor",
        ],
    )

    metadata = {
        "backend": "Python/matplotlib",
        "primary_format": "SVG with editable text",
        "final_width_mm": 183,
        "score_equations": {
            "geometry": "G = 0.31*support + 0.29*forward + 0.28*reverse + 0.12*orientation",
            "topology": "T = 0.30*coverage + 0.20*direction + 0.25*(1-missing) + 0.15*(1-unexplained) + 0.10*endpoint_coverage",
            "final": "F = 0.72*G + 0.28*T",
            "selection": "j* = argmax_j F_ij; margin = F_i,j* - max_{j != j*} F_ij",
            "binary": "M_out = M_target AND Dilate(T*(Skel_aux), radius=12)",
        },
        "figure_contracts": FIGURE_CONTRACTS,
        "review_boundary": "These four cases illustrate mechanism and internal ranking; they do not estimate accuracy or generalization.",
    }
    (source / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


CAPTIONS = """# Microscopy matching figure captions

## Figure 1 | End-to-end matching workflow
The label-free workflow independently compares each target with every auxiliary candidate. Local dark/brown evidence is converted to a foreground skeleton, a single similarity transform is used for localization and scoring, and the highest-scoring candidate is retained together with its winner-runner-up margin. The final binary output contains target pixels only and is constrained to the registered auxiliary corridor.

## Figure 2 | Weak-signal extraction and physical-scale audit
A representative weak-contrast target is converted into local darkness and local red-excess components. Their bounded combination yields the structural response used for adaptive foreground extraction. A white scale-bar detector reports pixels per micrometre for audit; physical scale is explicitly report-only and is excluded from candidate ranking in the current pipeline. Display contrast in panel a is unaltered; response panels are normalized as defined by the algorithm.

## Figure 3 | Observed support and topology graph
Auxiliary foreground fragments are grouped spatially to identify the central coherent support. The dilation exists only for group selection: the retained support is a strict subset of observed pixels. Skeletonization exposes endpoints, junctions and directional segments used by the topology score. Endpoint counts are image-graph diagnostics and must not be interpreted as biological nodes.

## Figure 4 | Coarse-to-fine unified registration
Each target-candidate pair is aligned by a similarity transform parameterized by scale, rotation and translation. Label-free coarse seeds are refined with Nelder-Mead optimization. The local landscape shows the production geometry score at the refined scale and angle. Boundary hits and flat optima are retained as review flags rather than being silently accepted as validated global solutions.

## Figure 5 | Mathematical template-selection score
Geometry combines support, forward and reverse distance similarity, and orientation agreement. Topology combines directional coverage with explicit penalties for missing candidate strokes, unexplained target evidence and endpoint mismatch. The final score is 0.72 geometry plus 0.28 topology. The 4 x 4 matrix contains all measured target-candidate scores; margins below 0.025 trigger review. Scores are ranking values, not calibrated probabilities, and physical scale is absent from the formula.

## Figure 6 | Topology resolves a near tie
For Target 2, geometry alone slightly favours Candidate D (Z) over Candidate B (T), whereas topology favours Candidate B. Their weighted combination reverses the geometry-only order, but the final difference is only 0.00153. The result is therefore retained as a low-margin automatic candidate requiring human review, not as an automatically validated classification.

## Figure 7 | Non-fabricating corridor gating
The selected auxiliary skeleton is transformed into target coordinates and dilated by 12 analysis pixels to form a spatial corridor. The final mask is the strict intersection of this corridor with the measured target foreground. Consequently the output is a subset of both the target mask and the corridor, contains only binary values after native nearest-neighbour resizing, and never copies auxiliary pixels or completes missing strokes. Retention is descriptive coverage rather than accuracy.

## Figure 8 | End-to-end outputs and decision audit
Four independent examples select the post hoc S, T, U and Z candidates and generate target-evidence-only presentation and binary outputs. All four decisions carry review flags: topology review for Target 1, low-margin review for Targets 2 and 4, and boundary-related review for Target 3. Parenthetical S/T/U/Z codes are displayed only for audit and are not used by the matcher. These examples demonstrate mechanism and internal ranking, not accuracy or generalization.

## Statistical and integrity note
The current dataset contains one example per post hoc code and no independent manual ground truth. No accuracy, uncertainty interval or generalization claim is supported. All image panels derive from the supplied microscopy images; global display transformations are stated, and the binary-output subset invariants are verified programmatically.
"""


def write_qa_report(saved: Sequence[Path], outdir: Path) -> None:
    lines = [
        "# Figure export QA",
        "",
        "- Backend: Python/matplotlib only.",
        "- Width: 7.2 inches (approximately 183 mm, double-column).",
        "- SVG text policy: `svg.fonttype = none`.",
        "- PDF text policy: TrueType (`pdf.fonttype = 42`).",
        "- TIFF: 600 dpi with LZW compression.",
        "- PNG: 300 dpi preview.",
        "- Image-integrity guardrail: binary output is verified as a subset of target mask and registered corridor.",
        "",
        "| File | Size (MiB) | Check |",
        "|---|---:|---|",
    ]
    for path in sorted(saved):
        size = path.stat().st_size / (1024 * 1024)
        check = "exists"
        if path.suffix.lower() == ".svg":
            content = path.read_text(encoding="utf-8")
            check = f"editable text nodes: {content.count('<text')}"
        lines.append(f"| `{path.name}` | {size:.2f} | {check} |")
    (outdir / "qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_indices(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, 9))
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item not in FIGURE_FUNCTIONS for item in result):
        raise argparse.ArgumentTypeError("--figures must be 'all' or a comma-separated subset of 1..8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data" / "input" / "target_images",
        help="Directory containing four target JPG or PNG images.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=PROJECT_ROOT / "data" / "input" / "reference_images",
        help="Directory containing four reference images.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "publication_figures",
        help="Destination for figures, source data, captions and QA notes.",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf,png,tiff",
        help="Comma-separated output formats; default: svg,pdf,png,tiff.",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help="'all' or a comma-separated subset of figure numbers 1..8.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    allowed = {"svg", "pdf", "png", "tif", "tiff"}
    if not formats or any(item not in allowed for item in formats):
        raise ValueError(f"Unsupported format list: {formats}")
    indices = parse_indices(args.figures)
    target_dir = args.targets.resolve()
    reference_dir = args.references.resolve()
    outdir = args.outdir.resolve()
    figures_dir = outdir / "figures"

    ctx = load_context(target_dir, reference_dir)
    export_source_data(ctx, outdir)
    (outdir / "captions.md").write_text(CAPTIONS, encoding="utf-8")

    saved: list[Path] = []
    for index in indices:
        contract = FIGURE_CONTRACTS[index - 1]
        figure = FIGURE_FUNCTIONS[index](ctx)
        saved.extend(save_figure(figure, figures_dir, str(contract["stem"]), formats))
    write_qa_report(saved, outdir)

    print(
        json.dumps(
            {
                "figures": indices,
                "formats": formats,
                "saved": [str(path) for path in saved],
                "source_data": str(outdir / "source_data"),
                "captions": str(outdir / "captions.md"),
                "qa": str(outdir / "qa_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
