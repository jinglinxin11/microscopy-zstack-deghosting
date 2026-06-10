from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.clean_high_bin import LABELS, display_signal, make_contact_sheet, save_binary, save_gray
from src.raw_pipeline import load_raw_rgb, run_raw_pipeline


def run(source: Path, out_root: Path) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full raw-to-clean_high_bin pipeline.")
    parser.add_argument("--source", default="raw_data", help="Input folder containing layers/<label>/raw.png files.")
    parser.add_argument("--out-root", default="outputs", help="Output root directory.")
    args = parser.parse_args()
    outdir = run(Path(args.source), Path(args.out_root))
    print(outdir.resolve())
    print((outdir / "contact_sheets" / "raw_to_clean_high_bin_overview.png").resolve())


if __name__ == "__main__":
    main()
