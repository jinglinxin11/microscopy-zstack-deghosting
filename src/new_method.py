from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class LevelSpec:
    """One binary level in the clean/ultra 24-level comparison grid."""

    source: str
    index: int
    percentile: float
    min_area: int
    sigma: float

    @property
    def variant(self) -> str:
        return f"{self.source}_{self.index:02d}_p{int(self.percentile)}_bin"


def normalize01(signal: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    finite = np.isfinite(signal)
    if not np.any(finite):
        return np.zeros_like(signal, dtype=np.float32)
    lo, hi = np.percentile(signal[finite], [low, high])
    if hi <= lo + 1e-8:
        return np.zeros_like(signal, dtype=np.float32)
    return np.clip((signal - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def read_gray_signal(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return normalize01(image, 1.0, 99.5)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for component_id in range(1, num):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == component_id] = 1
    return out.astype(np.float32)


def binary_from_signal(signal: np.ndarray, spec: LevelSpec) -> np.ndarray:
    signal = np.clip(np.asarray(signal, dtype=np.float32), 0.0, None)
    if spec.sigma > 0:
        ksize = max(3, int(round(spec.sigma * 6)) | 1)
        signal = cv2.GaussianBlur(signal, (ksize, ksize), spec.sigma, borderType=cv2.BORDER_REFLECT)
    values = signal[signal > 0]
    if values.size == 0:
        return np.zeros(signal.shape, dtype=np.float32)
    threshold = float(np.percentile(values, spec.percentile))
    mask = (signal >= threshold).astype(np.float32)
    mask = remove_small_components(mask, spec.min_area)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.float32)
    return mask


def _level_specs(source: str, percentiles: list[int]) -> list[LevelSpec]:
    specs: list[LevelSpec] = []
    count = len(percentiles)
    for index, percentile in enumerate(percentiles, start=1):
        ratio = 0.0 if count == 1 else (index - 1) / (count - 1)
        specs.append(
            LevelSpec(
                source=source,
                index=index,
                percentile=float(percentile),
                min_area=int(round(55 + 125 * ratio)),
                sigma=float(0.58 + 0.32 * ratio),
            )
        )
    return specs


def twenty_four_level_specs() -> list[LevelSpec]:
    clean = _level_specs("clean", list(range(54, 78, 2)))
    ultra = _level_specs("ultra", list(range(58, 82, 2)))
    return clean + ultra


def save_binary(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.where(mask > 0, 255, 0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def save_level_outputs(
    clean_signals: dict[str, np.ndarray],
    ultra_signals: dict[str, np.ndarray],
    outdir: Path,
    layers: list[str],
) -> list[Path]:
    specs = twenty_four_level_specs()
    outputs: list[Path] = []
    signal_sets = {"clean": clean_signals, "ultra": ultra_signals}
    for layer in layers:
        for spec in specs:
            mask = binary_from_signal(signal_sets[spec.source][layer], spec)
            path = outdir / f"layer{layer}_{spec.variant}.png"
            save_binary(mask, path)
            outputs.append(path)
    return outputs


def render_contact_sheet(paths: list[Path], outpath: Path, layers: list[str], thumb: int = 190) -> None:
    specs = twenty_four_level_specs()
    variants = [spec.variant for spec in specs]
    left = 52
    header = 64
    sheet = Image.new("RGB", (left + len(variants) * thumb, header + len(layers) * thumb), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for col, variant in enumerate(variants):
        draw.text((left + col * thumb + 5, 20), variant, fill=(0, 0, 0))
    lookup = {(path.name.split("_", 1)[0].replace("layer", ""), path.name.split("_", 1)[1].removesuffix(".png")): path for path in paths}
    for row, layer in enumerate(layers):
        y = header + row * thumb
        draw.text((18, y + 20), layer, fill=(0, 0, 0))
        for col, variant in enumerate(variants):
            path = lookup.get((layer, variant))
            if path is None:
                continue
            with Image.open(path) as img:
                tile = img.convert("RGB")
            tile.thumbnail((thumb, thumb), Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (thumb, thumb), "black")
            canvas.paste(tile, ((thumb - tile.width) // 2, (thumb - tile.height) // 2))
            sheet.paste(canvas, (left + col * thumb, y))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(outpath)
