from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.clean_high_bin import LABELS, clean_high_bin, display_signal, load_signal, make_contact_sheet, save_binary, save_gray
from src.raw_pipeline import load_raw_rgb, run_raw_pipeline


def run_raw_estimate(source: Path, out_root: Path) -> Path:
    raw_images = [load_raw_rgb(source / "layers" / label / "raw.png") for label in LABELS]
    result = run_raw_pipeline(raw_images)

    outdir = out_root / f"raw_to_clean_high_bin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=False)

    columns: dict[str, list[np.ndarray]] = {
        "raw": raw_images,
        "dark": [display_signal(x) for x in result["dark"]],
        "enhanced": [display_signal(x) for x in result["enhanced"]],
        "source_hybrid_soft": [display_signal(x) for x in result["source_hybrid_soft"]],
        "paper_clean": [display_signal(x) for x in result["paper_clean"]],
        "clean_high": [display_signal(x) for x in result["clean_high"]],
        "clean_high_bin": [x for x in result["clean_high_bin"]],
    }

    for idx, label in enumerate(LABELS):
        layer_out = outdir / "layers" / label
        layer_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw_images[idx]).save(layer_out / "raw.png")

        for key in (
            "dark",
            "enhanced",
            "weights",
            "source_hybrid_soft",
            "paper_clean",
            "residual",
            "clean_high",
            "support_mask",
            "added_audit",
        ):
            arr = result[key][idx]
            np.save(layer_out / f"{key}.npy", arr.astype(np.float32))
            save_gray(display_signal(arr) if key in {"dark", "enhanced", "source_hybrid_soft", "paper_clean", "residual", "clean_high"} else arr, layer_out / f"{key}.png")

        save_binary(result["clean_high_bin"][idx], layer_out / "clean_high_bin.png")
        np.save(layer_out / "clean_high_bin.npy", result["clean_high_bin"][idx].astype(np.float32))

    save_gray(result["confidence"], outdir / "confidence.png")
    np.save(outdir / "confidence.npy", result["confidence"].astype(np.float32))
    make_contact_sheet(columns=columns, labels=LABELS, outpath=outdir / "contact_sheets" / "raw_to_clean_high_bin_overview.png")
    (outdir / "method.md").write_text(
        """# Raw to Clean High Binary Pipeline

Pipeline:

raw RGB image
-> dark signal extraction
-> multi-scale black-hat / DoG enhancement
-> focus-weighted soft layer separation
-> source_hybrid_soft evidence image
-> paper_clean deghosted image
-> clean_high evidence-constrained repair
-> clean_high_bin final binary output
""",
        encoding="utf-8",
    )
    return outdir


def run_legacy_reference(source: Path, reference: Path, out_root: Path) -> Path:
    raw_images = [load_raw_rgb(source / "layers" / label / "raw.png") for label in LABELS]
    outdir = out_root / f"legacy_reference_clean_high_bin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=False)

    columns: dict[str, list[np.ndarray]] = {
        "raw": raw_images,
        "paper_clean": [],
        "source_hybrid_soft": [],
        "clean_high": [],
        "clean_high_bin": [],
    }

    for idx, label in enumerate(LABELS):
        ref_layer = reference / "layers" / label
        layer_out = outdir / "layers" / label
        layer_out.mkdir(parents=True, exist_ok=True)

        paper_clean = load_signal(ref_layer / "paper_clean.png")
        source_hybrid_soft = load_signal(ref_layer / "source_hybrid_soft.png")
        result = clean_high_bin(paper_clean, source_hybrid_soft)

        Image.fromarray(raw_images[idx]).save(layer_out / "raw.png")
        save_gray(display_signal(paper_clean), layer_out / "paper_clean.png")
        save_gray(display_signal(source_hybrid_soft), layer_out / "source_hybrid_soft.png")
        save_gray(display_signal(result["clean_high"]), layer_out / "clean_high.png")
        save_binary(result["clean_high_bin"], layer_out / "clean_high_bin.png")
        save_gray(result["support_mask"], layer_out / "support_mask.png")
        save_gray(result["added_audit"], layer_out / "added_audit.png")

        np.save(layer_out / "paper_clean.npy", paper_clean.astype(np.float32))
        np.save(layer_out / "source_hybrid_soft.npy", source_hybrid_soft.astype(np.float32))
        np.save(layer_out / "clean_high.npy", result["clean_high"].astype(np.float32))
        np.save(layer_out / "clean_high_bin.npy", result["clean_high_bin"].astype(np.float32))

        columns["paper_clean"].append(display_signal(paper_clean))
        columns["source_hybrid_soft"].append(display_signal(source_hybrid_soft))
        columns["clean_high"].append(display_signal(result["clean_high"]))
        columns["clean_high_bin"].append(result["clean_high_bin"])

    make_contact_sheet(columns=columns, labels=LABELS, outpath=outdir / "contact_sheets" / "legacy_reference_clean_high_bin_overview.png")
    (outdir / "method.md").write_text(
        """# Legacy Reference Clean High Binary Pipeline

This mode reproduces the previously selected result. It reads raw images for
display, but uses the saved legacy intermediate images in sample_data:

- paper_clean.png
- source_hybrid_soft.png

Use --mode raw-estimate for a pure raw-only approximation.
""",
        encoding="utf-8",
    )
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the microscopy layer deghosting pipeline.")
    parser.add_argument("--source", default="raw_data", help="Input folder containing layers/<label>/raw.png files.")
    parser.add_argument("--reference", default="sample_data", help="Legacy reference intermediate folder.")
    parser.add_argument("--out-root", default="outputs", help="Output root directory.")
    parser.add_argument(
        "--mode",
        choices=["legacy-reference", "raw-estimate"],
        default="legacy-reference",
        help="legacy-reference reproduces the selected previous result; raw-estimate computes all intermediates from raw only.",
    )
    args = parser.parse_args()
    if args.mode == "legacy-reference":
        outdir = run_legacy_reference(Path(args.source), Path(args.reference), Path(args.out_root))
        overview = outdir / "contact_sheets" / "legacy_reference_clean_high_bin_overview.png"
    else:
        outdir = run_raw_estimate(Path(args.source), Path(args.out_root))
        overview = outdir / "contact_sheets" / "raw_to_clean_high_bin_overview.png"
    print(outdir.resolve())
    print(overview.resolve())


if __name__ == "__main__":
    main()
