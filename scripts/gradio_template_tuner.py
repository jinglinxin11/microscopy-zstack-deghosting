"""Interactive Gradio tuner for template-guided digit cleanup parameters."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from scripts.make_template_guided_repair import (
    DIGITS,
    ROOT,
    AlignmentParameters,
    FilterParameters,
    _load_digit_data,
    aligned_template_full,
    change_overlay,
    compute_template_match_quality,
    default_alignment_parameters,
    default_filter_parameters,
    downsample_max,
    downsample_mean,
    find_best_template_match,
    load_templates,
    loose_repair_pool,
    read_raw_rgb,
    save_mask,
    signal_on_yellow,
    template_guided_filter,
    template_overlay_on_raw,
    refine_template_alignment,
)


OUTDIR = ROOT / "figures" / "gradio_template_tuner"


@dataclass(frozen=True)
class CaseData:
    data_id: int
    raw_rgb: np.ndarray
    before: np.ndarray
    pool: np.ndarray
    seed: np.ndarray
    signal: np.ndarray
    template: object
    match: object
    target_frame: object
    predicted_digit: int
    score: float
    template_path: str


def mask_preview(mask: np.ndarray) -> np.ndarray:
    arr = downsample_max(np.asarray(mask, dtype=bool))
    return arr.astype(np.uint8) * 255


def rgb_preview(image: np.ndarray) -> np.ndarray:
    return np.asarray(downsample_mean(np.asarray(image, dtype=np.uint8)), dtype=np.uint8)


@lru_cache(maxsize=1)
def cached_templates() -> tuple:
    templates = load_templates(ROOT)
    return templates, {template.path: template for template in templates}


@lru_cache(maxsize=len(DIGITS))
def load_case(data_id: int) -> CaseData:
    data = _load_digit_data(ROOT, data_id)
    raw_rgb = read_raw_rgb(ROOT, data_id)
    before = np.asarray(data["mask"], dtype=bool)
    tophat = np.asarray(data["tophat_norm"], dtype=np.float32)
    pool, seed, signal = loose_repair_pool(before, tophat)
    templates, template_by_path = cached_templates()
    match, target_frame = find_best_template_match(pool, seed, signal, list(templates))
    template = template_by_path[match.template_path]
    return CaseData(
        data_id=data_id,
        raw_rgb=raw_rgb,
        before=before,
        pool=pool,
        seed=seed,
        signal=signal,
        template=template,
        match=match,
        target_frame=target_frame,
        predicted_digit=match.predicted_digit,
        score=match.score,
        template_path=str(match.template_path),
    )


def defaults_for_data_id(data_id: int) -> tuple:
    params = default_filter_parameters(int(data_id))
    alignment = default_alignment_parameters(int(data_id))
    return (
        alignment.template_scale_multiplier,
        alignment.template_scale_x_multiplier,
        alignment.template_scale_y_multiplier,
        alignment.template_y_offset_px,
        alignment.template_x_offset_px,
        alignment.auto_refine_alignment,
        params.soft_margin_distance,
        params.strong_margin_distance,
        params.template_soft_threshold,
        params.repair_distance,
        params.strong_distance,
        params.template_min,
        params.close_length,
        params.bridge_length,
        params.bridge_distance,
        params.connected_zone_iterations,
        params.soft_edge_inside_fraction,
        params.soft_edge_min_signal,
        params.strong_contiguous_seed_overlap,
        params.strong_contiguous_min_signal,
    )


def make_params(
    soft_margin_distance: int,
    strong_margin_distance: int,
    template_soft_threshold: float,
    repair_distance: int,
    strong_distance: int,
    template_min: float,
    close_length: int,
    bridge_length: int,
    bridge_distance: int,
    connected_zone_iterations: int,
    soft_edge_inside_fraction: float,
    soft_edge_min_signal: float,
    strong_contiguous_seed_overlap: int,
    strong_contiguous_min_signal: float,
) -> FilterParameters:
    return FilterParameters(
        soft_margin_distance=int(soft_margin_distance),
        strong_margin_distance=int(strong_margin_distance),
        template_soft_threshold=float(template_soft_threshold),
        soft_edge_inside_fraction=float(soft_edge_inside_fraction),
        soft_edge_min_signal=float(soft_edge_min_signal),
        strong_contiguous_seed_overlap=int(strong_contiguous_seed_overlap),
        strong_contiguous_min_signal=float(strong_contiguous_min_signal),
        repair_distance=int(repair_distance),
        strong_distance=int(strong_distance),
        template_min=float(template_min),
        close_length=int(close_length),
        bridge_length=int(bridge_length),
        bridge_distance=int(bridge_distance),
        connected_zone_iterations=int(connected_zone_iterations),
    )


def current_soft_gate(template_core: np.ndarray, template_gate: np.ndarray, template_soft: np.ndarray, params: FilterParameters) -> np.ndarray:
    distance_to_core = ndi.distance_transform_edt(~template_core)
    return template_gate | (
        (distance_to_core <= params.soft_margin_distance)
        & (template_soft >= params.template_soft_threshold)
    )


def run_preview(
    data_id: int,
    template_scale_multiplier: float,
    template_scale_x_multiplier: float,
    template_scale_y_multiplier: float,
    template_y_offset_px: int,
    template_x_offset_px: int,
    auto_refine_alignment: bool,
    soft_margin_distance: int,
    strong_margin_distance: int,
    template_soft_threshold: float,
    repair_distance: int,
    strong_distance: int,
    template_min: float,
    close_length: int,
    bridge_length: int,
    bridge_distance: int,
    connected_zone_iterations: int,
    soft_edge_inside_fraction: float,
    soft_edge_min_signal: float,
    strong_contiguous_seed_overlap: int,
    strong_contiguous_min_signal: float,
) -> tuple:
    data_id = int(data_id)
    case = load_case(data_id)
    params = make_params(
        soft_margin_distance,
        strong_margin_distance,
        template_soft_threshold,
        repair_distance,
        strong_distance,
        template_min,
        close_length,
        bridge_length,
        bridge_distance,
        connected_zone_iterations,
        soft_edge_inside_fraction,
        soft_edge_min_signal,
        strong_contiguous_seed_overlap,
        strong_contiguous_min_signal,
    )
    mode_defaults = default_filter_parameters(data_id)
    params = replace(
        params,
        use_repair_pool=mode_defaults.use_repair_pool,
        allow_repair_add=mode_defaults.allow_repair_add,
    )
    initial_alignment = AlignmentParameters(
        template_scale_multiplier=float(template_scale_multiplier),
        template_scale_x_multiplier=float(template_scale_x_multiplier),
        template_scale_y_multiplier=float(template_scale_y_multiplier),
        template_y_offset_px=int(template_y_offset_px),
        template_x_offset_px=int(template_x_offset_px),
        auto_refine_alignment=bool(auto_refine_alignment),
    )
    refined_alignment, adjusted_match = refine_template_alignment(
        case.pool,
        case.signal,
        case.template,
        case.match,
        case.before.shape,
        initial_alignment,
    )
    template_core, template_gate, template_soft = aligned_template_full(
        case.before.shape,
        case.template,
        adjusted_match,
        alignment=refined_alignment,
    )
    quality = compute_template_match_quality(case.pool, case.signal, template_core, template_gate)
    final = template_guided_filter(
        data_id,
        case.before,
        case.pool,
        case.seed,
        case.signal,
        template_core,
        template_gate,
        template_soft,
        case.predicted_digit,
        params=params,
    )
    soft_gate = current_soft_gate(template_core, template_gate, template_soft, params)
    after_signal = np.where(final, np.clip(0.25 + 0.75 * case.signal, 0.0, 1.0), 0.0)
    after_yellow = signal_on_yellow(case.raw_rgb, after_signal)
    overlay = template_overlay_on_raw(
        case.raw_rgb,
        template_core,
        template_gate,
        template_soft,
        case.target_frame,
    )
    change = change_overlay(case.before, final)
    removed = case.pool & ~final

    pool_count = int(np.count_nonzero(case.pool))
    soft_gate_pool_fraction = float(np.count_nonzero(case.pool & soft_gate) / max(1, pool_count))
    final_count = int(np.count_nonzero(final))
    removed_count = int(np.count_nonzero(case.before & ~final))
    added_count = int(np.count_nonzero(final & ~case.before))

    save_dir = OUTDIR / f"data{data_id}_latest"
    save_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_preview(overlay), mode="RGB").save(save_dir / "template_on_raw.png")
    Image.fromarray(mask_preview(soft_gate), mode="L").save(save_dir / "soft_gate.png")
    Image.fromarray(mask_preview(final), mode="L").save(save_dir / "final_mask.png")
    Image.fromarray(rgb_preview(change), mode="RGB").save(save_dir / "change_map.png")
    Image.fromarray(rgb_preview(after_yellow), mode="RGB").save(save_dir / "final_yellow.png")
    save_mask(save_dir / "final_mask_full_resolution.png", final)
    with (save_dir / "params_and_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_id": data_id,
                "predicted_digit": case.predicted_digit,
                "template_path": case.template_path,
                "match_score": case.score,
                "template_match_fraction": quality.template_match_fraction,
                "template_core_fraction": quality.template_core_fraction,
                "template_dice": quality.template_dice,
                "template_signal_fraction": quality.template_signal_fraction,
                "soft_gate_pool_fraction": soft_gate_pool_fraction,
                "before_px": int(np.count_nonzero(case.before)),
                "pool_px": pool_count,
                "final_px": final_count,
                "removed_px": removed_count,
                "added_px": added_count,
                "alignment": {
                    "input_template_scale_multiplier": float(template_scale_multiplier),
                    "input_template_scale_x_multiplier": float(template_scale_x_multiplier),
                    "input_template_scale_y_multiplier": float(template_scale_y_multiplier),
                    "input_template_y_offset_px": int(template_y_offset_px),
                    "input_template_x_offset_px": int(template_x_offset_px),
                    "auto_refine_alignment": bool(auto_refine_alignment),
                    "refined_alignment_score": refined_alignment.alignment_score,
                    "refined_template_scale_multiplier": refined_alignment.template_scale_multiplier,
                    "refined_template_scale_x_multiplier": refined_alignment.template_scale_x_multiplier,
                    "refined_template_scale_y_multiplier": refined_alignment.template_scale_y_multiplier,
                    "refined_template_y_offset_px": refined_alignment.template_y_offset_px,
                    "refined_template_x_offset_px": refined_alignment.template_x_offset_px,
                    "adjusted_scale_height_frac": adjusted_match.scale_height_frac,
                    "adjusted_center_y_frac": adjusted_match.center_y_frac,
                    "adjusted_center_x_frac": adjusted_match.center_x_frac,
                },
                "parameters": asdict(params),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    metrics = (
        f"predicted_digit: {case.predicted_digit}\n"
        f"match_score: {case.score:.6f}\n"
        f"template_match_fraction: {quality.template_match_fraction:.6f}\n"
        f"template_core_fraction: {quality.template_core_fraction:.6f}\n"
        f"template_dice: {quality.template_dice:.6f}\n"
        f"template_signal_fraction: {quality.template_signal_fraction:.6f}\n"
        f"soft_gate_pool_fraction: {soft_gate_pool_fraction:.6f}\n"
        f"alignment_score: {refined_alignment.alignment_score:.6f}\n"
        f"input_template_scale_multiplier: {float(template_scale_multiplier):.3f}\n"
        f"input_template_scale_x_multiplier: {float(template_scale_x_multiplier):.3f}\n"
        f"input_template_scale_y_multiplier: {float(template_scale_y_multiplier):.3f}\n"
        f"input_template_y_offset_px: {int(template_y_offset_px)}\n"
        f"input_template_x_offset_px: {int(template_x_offset_px)}\n"
        f"auto_refine_alignment: {bool(auto_refine_alignment)}\n"
        f"refined_template_scale_multiplier: {refined_alignment.template_scale_multiplier:.3f}\n"
        f"refined_template_scale_x_multiplier: {refined_alignment.template_scale_x_multiplier:.3f}\n"
        f"refined_template_scale_y_multiplier: {refined_alignment.template_scale_y_multiplier:.3f}\n"
        f"refined_template_y_offset_px: {refined_alignment.template_y_offset_px}\n"
        f"refined_template_x_offset_px: {refined_alignment.template_x_offset_px}\n"
        f"use_repair_pool: {params.use_repair_pool}\n"
        f"allow_repair_add: {params.allow_repair_add}\n"
        f"before_px: {int(np.count_nonzero(case.before))}\n"
        f"pool_px: {pool_count}\n"
        f"final_px: {final_count}\n"
        f"removed_px: {removed_count}\n"
        f"added_px: {added_count}\n"
        f"template_path: {case.template_path}\n"
        f"saved_dir: {save_dir}"
    )
    return (
        rgb_preview(case.raw_rgb),
        mask_preview(case.before),
        mask_preview(case.pool),
        rgb_preview(overlay),
        mask_preview(soft_gate),
        mask_preview(final),
        mask_preview(removed),
        rgb_preview(change),
        rgb_preview(after_yellow),
        metrics,
    )


def build_app() -> gr.Blocks:
    default_data_id = 4
    defaults = defaults_for_data_id(default_data_id)
    with gr.Blocks(title="Template-guided digit cleanup tuner") as demo:
        gr.Markdown(
            """
            # Template-guided digit cleanup tuner

            Interactive parameter tuning for template alignment, soft gate, and repair filtering.
            The raw image and preprocessing are unchanged.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                data_id = gr.Dropdown(choices=[str(d) for d in DIGITS], value=str(default_data_id), label="data_id")
                gr.Markdown("### Template alignment")
                template_scale_multiplier = gr.Slider(0.75, 1.12, value=defaults[0], step=0.005, label="template_scale_multiplier")
                template_scale_x_multiplier = gr.Slider(0.75, 1.25, value=defaults[1], step=0.005, label="template_scale_x_multiplier")
                template_scale_y_multiplier = gr.Slider(0.75, 1.25, value=defaults[2], step=0.005, label="template_scale_y_multiplier")
                template_y_offset_px = gr.Slider(-300, 300, value=defaults[3], step=5, label="template_y_offset_px")
                template_x_offset_px = gr.Slider(-300, 300, value=defaults[4], step=5, label="template_x_offset_px")
                auto_refine_alignment = gr.Checkbox(value=defaults[5], label="auto_refine_alignment")
                gr.Markdown("### Soft gate")
                soft_margin_distance = gr.Slider(0, 140, value=defaults[6], step=1, label="soft_margin_distance")
                strong_margin_distance = gr.Slider(0, 180, value=defaults[7], step=1, label="strong_margin_distance")
                template_soft_threshold = gr.Slider(0.0, 0.05, value=defaults[8], step=0.001, label="template_soft_threshold")
                gr.Markdown("### Repair")
                repair_distance = gr.Slider(0, 120, value=defaults[9], step=1, label="repair_distance")
                strong_distance = gr.Slider(0, 90, value=defaults[10], step=1, label="strong_distance")
                template_min = gr.Slider(0.0, 0.05, value=defaults[11], step=0.001, label="template_min")
                close_length = gr.Slider(0, 55, value=defaults[12], step=1, label="close_length")
                bridge_length = gr.Slider(0, 70, value=defaults[13], step=1, label="bridge_length")
                bridge_distance = gr.Slider(0, 70, value=defaults[14], step=1, label="bridge_distance")
                connected_zone_iterations = gr.Slider(0, 25, value=defaults[15], step=1, label="connected_zone_iterations")
                gr.Markdown("### Component criteria")
                soft_edge_inside_fraction = gr.Slider(0.0, 1.0, value=defaults[16], step=0.01, label="soft_edge_inside_fraction")
                soft_edge_min_signal = gr.Slider(0.0, 1.0, value=defaults[17], step=0.01, label="soft_edge_min_signal")
                strong_contiguous_seed_overlap = gr.Slider(0, 120, value=defaults[18], step=1, label="strong_contiguous_seed_overlap")
                strong_contiguous_min_signal = gr.Slider(0.0, 1.0, value=defaults[19], step=0.01, label="strong_contiguous_min_signal")
                run_button = gr.Button("Run preview", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    raw = gr.Image(label="Raw image")
                    before = gr.Image(label="Candidate mask")
                    pool = gr.Image(label="Repair pool")
                with gr.Row():
                    overlay = gr.Image(label="Template on raw")
                    soft_gate = gr.Image(label="Soft gate")
                    final = gr.Image(label="Final mask")
                with gr.Row():
                    removed = gr.Image(label="Removed pool")
                    change = gr.Image(label="Change map")
                    yellow = gr.Image(label="Final yellow")
                metrics = gr.Textbox(label="Metrics / saved output", lines=15)

        parameter_inputs = [
            data_id,
            template_scale_multiplier,
            template_scale_x_multiplier,
            template_scale_y_multiplier,
            template_y_offset_px,
            template_x_offset_px,
            auto_refine_alignment,
            soft_margin_distance,
            strong_margin_distance,
            template_soft_threshold,
            repair_distance,
            strong_distance,
            template_min,
            close_length,
            bridge_length,
            bridge_distance,
            connected_zone_iterations,
            soft_edge_inside_fraction,
            soft_edge_min_signal,
            strong_contiguous_seed_overlap,
            strong_contiguous_min_signal,
        ]
        outputs = [raw, before, pool, overlay, soft_gate, final, removed, change, yellow, metrics]
        run_button.click(run_preview, inputs=parameter_inputs, outputs=outputs)
        data_id.change(
            defaults_for_data_id,
            inputs=[data_id],
            outputs=[
                template_scale_multiplier,
                template_scale_x_multiplier,
                template_scale_y_multiplier,
                template_y_offset_px,
                template_x_offset_px,
                auto_refine_alignment,
                soft_margin_distance,
                strong_margin_distance,
                template_soft_threshold,
                repair_distance,
                strong_distance,
                template_min,
                close_length,
                bridge_length,
                bridge_distance,
                connected_zone_iterations,
                soft_edge_inside_fraction,
                soft_edge_min_signal,
                strong_contiguous_seed_overlap,
                strong_contiguous_min_signal,
            ],
        ).then(run_preview, inputs=parameter_inputs, outputs=outputs)
        demo.load(run_preview, inputs=parameter_inputs, outputs=outputs)
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Gradio tuner for template-guided digit cleanup.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run one preview without launching the web UI.")
    parser.add_argument("--data-id", type=int, default=4, choices=list(DIGITS), help="Data id used with --dry-run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        defaults = defaults_for_data_id(args.data_id)
        result = run_preview(args.data_id, *defaults)
        print(result[-1])
        return
    demo = build_app()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
