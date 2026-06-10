from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.clean_high_bin import (
    LABELS,
    clean_high_bin,
    display_signal,
    load_signal,
    make_contact_sheet,
    save_binary,
    save_gray,
)


def run(source: Path, out_root: Path) -> Path:
    outdir = out_root / f"clean_high_bin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=False)

    columns: dict[str, list[np.ndarray]] = {
        "raw": [],
        "source_clean": [],
        "clean_high": [],
        "clean_high_bin": [],
    }

    for label in LABELS:
        layer_src = source / "layers" / label
        layer_out = outdir / "layers" / label
        layer_out.mkdir(parents=True, exist_ok=True)

        raw = np.asarray(Image.open(layer_src / "raw.png").convert("RGB"))
        clean_signal = load_signal(layer_src / "paper_clean.png")
        evidence = load_signal(layer_src / "source_hybrid_soft.png")
        result = clean_high_bin(clean_signal, evidence)

        Image.fromarray(raw).save(layer_out / "raw.png")
        save_gray(display_signal(clean_signal), layer_out / "source_clean.png")
        save_gray(display_signal(result["clean_high"]), layer_out / "clean_high.png")
        save_binary(result["clean_high_bin"], layer_out / "clean_high_bin.png")
        save_gray(result["support_mask"], layer_out / "support_mask.png")
        save_gray(result["added_audit"], layer_out / "added_audit.png")

        np.save(layer_out / "clean_high.npy", result["clean_high"])
        np.save(layer_out / "clean_high_bin.npy", result["clean_high_bin"])
        np.save(layer_out / "support_mask.npy", result["support_mask"])
        np.save(layer_out / "added_audit.npy", result["added_audit"])

        columns["raw"].append(raw)
        columns["source_clean"].append(display_signal(clean_signal))
        columns["clean_high"].append(display_signal(result["clean_high"]))
        columns["clean_high_bin"].append(result["clean_high_bin"])

    make_contact_sheet(columns=columns, labels=LABELS, outpath=outdir / "contact_sheets" / "clean_high_bin_overview.png")
    (outdir / "method.md").write_text(
        """# Clean High Binary Output

This run uses the selected clean_high_bin stage only.

Inputs per layer:
- raw.png
- paper_clean.png
- source_hybrid_soft.png

Outputs per layer:
- clean_high.png
- clean_high_bin.png
- support_mask.png
- added_audit.png
""",
        encoding="utf-8",
    )
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected clean_high_bin image-processing stage.")
    parser.add_argument("--source", default="sample_data", help="Input folder containing layers/<label>/ files.")
    parser.add_argument("--out-root", default="outputs", help="Output root directory.")
    args = parser.parse_args()
    outdir = run(Path(args.source), Path(args.out_root))
    print(outdir.resolve())
    print((outdir / "contact_sheets" / "clean_high_bin_overview.png").resolve())


if __name__ == "__main__":
    main()
