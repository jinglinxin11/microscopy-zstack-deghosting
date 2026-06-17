from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
from PIL import Image

from process_diamond_target_images import (
    ascii_label,
    binarize_denoised,
    dark_feature_signal,
    load_gray,
    make_contact_sheet,
    natural_key,
    phase_shift,
    presentation_enhance,
    remove_interlayer_crosstalk,
    save_gray_tiff,
    suppress_frame_edges,
    to_uint8,
)


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_TEMPLATE_SCRIPT = (
    Path(__file__).resolve().parent
    / "post_template_package"
    / "post_template_repair"
    / "post_template_repair.py"
)
DEFAULT_TEMPLATE_DIR = DEFAULT_TEMPLATE_SCRIPT.parent / "templates"


def parse_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def infer_data_id(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        return sorted(files, key=natural_key)
    raise FileNotFoundError(input_path)


def load_post_template_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    spec = importlib.util.spec_from_file_location("post_template_repair", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_preprocessed_outputs(
    files: list[Path],
    data_ids: tuple[int, ...],
    preprocessed_dir: Path,
) -> None:
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    grays = [load_gray(path) for path in files]
    corrected = []
    tophat = []
    for gray in grays:
        _, corr, signal = dark_feature_signal(gray)
        corrected.append(corr)
        tophat.append(signal)

    cleaned = remove_interlayer_crosstalk(tophat)

    report_lines = [
        "Full diamond image pipeline preprocessing report",
        f"Input count: {len(files)}",
        "Method: low-frequency illumination correction, dark-feature top-hat, binary denoising.",
    ]
    if len(tophat) > 1:
        report_lines.extend(["", "Estimated phase-correlation shifts against first image (pixels):"])
        for data_id, signal in zip(data_ids, tophat):
            sx, sy, response = phase_shift(tophat[0], signal)
            report_lines.append(f"- data {data_id}: x={sx:.2f}, y={sy:.2f}, response={response:.4f}")

    overview_items = []
    binary_items = []
    for path, data_id, gray, corr, signal, clean in zip(files, data_ids, grays, corrected, tophat, cleaned):
        stem = f"data{data_id}"
        label = f"data {data_id}"
        raw8 = to_uint8(suppress_frame_edges(gray), 0.5, 99.5)
        corr8 = to_uint8(corr, 0.2, 99.8)
        top8 = to_uint8(signal, 0.5, 99.7)
        clean8 = to_uint8(clean, 0.5, 99.7)
        enhanced8 = presentation_enhance(signal)
        binary8, binary_threshold = binarize_denoised(signal)
        binary_white_bg = 255 - binary8

        Image.fromarray(raw8).save(preprocessed_dir / f"{stem}_00_raw_autocontrast.png")
        Image.fromarray(corr8).save(preprocessed_dir / f"{stem}_01_illumination_corrected.png")
        Image.fromarray(top8).save(preprocessed_dir / f"{stem}_02_tophat_denoised.png")
        Image.fromarray(clean8).save(preprocessed_dir / f"{stem}_03_interlayer_cleaned.png")
        Image.fromarray(enhanced8).save(preprocessed_dir / f"{stem}_04_final_enhanced.png")
        Image.fromarray(binary8).save(preprocessed_dir / f"{stem}_05_binary_denoised.png")
        Image.fromarray(binary_white_bg).save(preprocessed_dir / f"{stem}_05_binary_denoised_white_bg.png")
        save_gray_tiff(preprocessed_dir / f"{stem}_03_interlayer_cleaned.tiff", clean)

        report_lines.append(
            f"- data {data_id} source={path.name} label={ascii_label(path)} binary_threshold={binary_threshold:.1f}"
        )
        overview_items.extend(
            [
                (f"{label} raw", Image.fromarray(raw8)),
                (f"{label} corrected", Image.fromarray(corr8)),
                (f"{label} top-hat", Image.fromarray(top8)),
                (f"{label} enhanced", Image.fromarray(enhanced8)),
                (f"{label} cleaned", Image.fromarray(clean8)),
            ]
        )
        binary_items.extend(
            [
                (f"{label} top-hat", Image.fromarray(top8)),
                (f"{label} binary", Image.fromarray(binary8)),
                (f"{label} white-bg", Image.fromarray(binary_white_bg)),
            ]
        )

    make_contact_sheet(overview_items, preprocessed_dir / "preprocessing_overview.png", cols=5)
    make_contact_sheet(binary_items, preprocessed_dir / "binary_denoising_overview.png", cols=3)
    (preprocessed_dir / "preprocessing_report.txt").write_text("\n".join(report_lines), encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run raw diamond image preprocessing plus automatic template-matching post-processing."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw image file or a directory of raw images.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output root directory.")
    parser.add_argument(
        "--data-ids",
        type=parse_ids,
        default=None,
        help="Optional comma-separated IDs. If omitted, IDs are inferred from file names.",
    )
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--template-script", type=Path, default=DEFAULT_TEMPLATE_SCRIPT)
    parser.add_argument("--skip-template", action="store_true", help="Only run preprocessing and binary denoising.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = collect_inputs(args.input)
    if not files:
        raise FileNotFoundError(f"No supported images found under {args.input}")

    if args.data_ids is None:
        data_ids = tuple(infer_data_id(path, index + 1) for index, path in enumerate(files))
    else:
        data_ids = args.data_ids
        if len(data_ids) != len(files):
            raise ValueError(f"--data-ids has {len(data_ids)} IDs but input has {len(files)} image(s).")

    args.outdir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = args.outdir / "preprocessed"
    final_dir = args.outdir / "template_final"

    write_preprocessed_outputs(files, data_ids, preprocessed_dir)
    print(f"wrote preprocessing outputs: {preprocessed_dir}")

    if not args.skip_template:
        post_template = load_post_template_module(args.template_script)
        post_template.process_all(preprocessed_dir, args.template_dir, final_dir, data_ids)
        print(f"wrote template-matching outputs: {final_dir}")


if __name__ == "__main__":
    main()
