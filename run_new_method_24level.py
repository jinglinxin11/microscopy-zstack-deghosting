from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.new_method import read_gray_signal, render_contact_sheet, save_level_outputs, twenty_four_level_specs


def _parse_layers(value: str) -> list[str]:
    layers = [item.strip() for item in value.split(",") if item.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def _read_signals(directory: Path, pattern: str, layers: list[str]) -> dict[str, object]:
    signals = {}
    for layer in layers:
        path = directory / pattern.format(layer=layer)
        if not path.exists():
            raise FileNotFoundError(f"Missing input for layer {layer}: {path}")
        signals[layer] = read_gray_signal(path)
    return signals


def run(args: argparse.Namespace) -> Path:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = args.layers
    clean_signals = _read_signals(Path(args.clean_dir), args.clean_pattern, layers)
    ultra_signals = _read_signals(Path(args.ultra_dir), args.ultra_pattern, layers)

    level_dir = outdir / "binary_24level"
    level_paths = save_level_outputs(clean_signals, ultra_signals, level_dir, layers)
    render_contact_sheet(level_paths, outdir / "binary_24level_comparison.png", layers=layers, thumb=args.thumb)

    selected_dir = outdir / "selected_single_images"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_spec = next(spec for spec in twenty_four_level_specs() if spec.variant == args.selected_variant)
    for layer in layers:
        source = level_dir / f"layer{layer}_{selected_spec.variant}.png"
        shutil.copy2(source, selected_dir / source.name)

    return outdir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 24-level new-method binary outputs")
    parser.add_argument("--clean-dir", required=True, help="Directory containing clean source images")
    parser.add_argument("--ultra-dir", required=True, help="Directory containing ultra source images")
    parser.add_argument("--clean-pattern", default="layer{layer}_clean.png")
    parser.add_argument("--ultra-pattern", default="layer{layer}_ultra.png")
    parser.add_argument("--layers", type=_parse_layers, default=["1", "3", "7"], help="Comma-separated layer IDs")
    parser.add_argument("--selected-variant", default="ultra_03_p62_bin")
    parser.add_argument("--thumb", type=int, default=190)
    parser.add_argument("--outdir", required=True)
    return parser


def main() -> None:
    outdir = run(build_parser().parse_args())
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
