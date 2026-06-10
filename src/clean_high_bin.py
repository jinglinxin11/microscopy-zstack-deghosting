from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


LABELS = ["1", "3", "4", "7"]

HIGH_PARAMS = {
    "gain": 0.52,
    "sig_gain": 0.38,
    "gate_relax": 0.06,
    "length_scale": 1.20,
    "support_sigma": 2.4,
    "crisp": 0.56,
    "bin_q": 85.0,
}


def normalize_percentile(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[finite], [p_low, p_high])
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def load_signal(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    # Input paper-clean images are black-on-white display images.
    return normalize_percentile(1.0 - arr, 1, 99.2)


def display_signal(signal: np.ndarray, gamma: float = 0.54, darkness: float = 0.98) -> np.ndarray:
    signal = normalize_percentile(signal, 1.0, 99.15)
    return np.clip(1.0 - darkness * np.power(np.clip(signal, 0, 1), gamma), 0, 1).astype(np.float32)


def save_gray(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).save(path)


def save_binary(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def local_orientation_closing(mask: np.ndarray, max_len: int) -> np.ndarray:
    out = mask.astype(np.uint8).copy()
    for length in [max(5, max_len // 2), max(7, max_len)]:
        length = int(length) | 1
        for angle in (0, 30, 60, 90, 120, 150):
            kernel = np.zeros((length, length), dtype=np.uint8)
            center = length // 2
            theta = np.deg2rad(angle)
            dx = int(round(np.cos(theta) * center))
            dy = int(round(np.sin(theta) * center))
            cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
            closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            out = np.maximum(out, closed)
    return out.astype(np.float32)


def remove_dust(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for cid in range(1, num):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == cid] = 1
    return out.astype(np.float32)


def binarize_clean_high(signal: np.ndarray) -> np.ndarray:
    sig = normalize_percentile(signal, 1, 99.4)
    thr_percentile = np.percentile(sig, HIGH_PARAMS["bin_q"])
    otsu_input = np.clip(sig * 255, 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(thr_percentile), otsu_thr / 255.0 * 0.92)
    bw = (sig > thr).astype(np.uint8)
    bw = remove_dust(bw, min_area=max(4, sig.size // 160000)).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return bw.astype(np.float32)


def clean_high_bin(clean_signal: np.ndarray, source_evidence: np.ndarray) -> dict[str, np.ndarray]:
    sig = normalize_percentile(clean_signal, 1, 99.2)
    evidence = normalize_percentile(source_evidence, 1, 99.2)
    h, w = sig.shape
    min_area = max(3, sig.size // 180000)

    high = sig > max(0.20, float(np.percentile(sig, 84)))
    support = cv2.GaussianBlur(high.astype(np.float32), (0, 0), HIGH_PARAMS["support_sigma"])
    p_low = np.percentile(evidence, 61)
    p_high = np.percentile(evidence, 94)
    evidence_gate = np.clip((evidence - p_low) / (p_high - p_low + 1e-6), 0, 1)

    gate_thr = max(0.08, 0.16 - HIGH_PARAMS["gate_relax"])
    bridge_candidate = (support > 0.045) & (evidence_gate > gate_thr)

    base_len = max(7, min(19, int(round(max(h, w) * 0.010))))
    max_len = max(5, int(round(base_len * HIGH_PARAMS["length_scale"]))) | 1
    bridged = local_orientation_closing(bridge_candidate.astype(np.uint8), max_len=max_len)
    bridged = bridged * (cv2.GaussianBlur(high.astype(np.float32), (0, 0), 5.0) > 0.018)

    support_mask = np.maximum(high.astype(np.float32), bridged)
    support_mask = remove_dust(support_mask, min_area=min_area)
    support_mask = cv2.morphologyEx(
        support_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(np.float32)

    repaired = np.maximum(
        sig,
        HIGH_PARAMS["gain"] * support_mask * evidence_gate + HIGH_PARAMS["sig_gain"] * sig * support_mask,
    )
    repaired = cv2.GaussianBlur(repaired.astype(np.float32), (0, 0), 0.45)
    repaired = normalize_percentile(repaired, 2.0, 98.8)

    blur = cv2.GaussianBlur(repaired, (0, 0), 1.2)
    crisp = normalize_percentile(np.maximum(repaired + HIGH_PARAMS["crisp"] * (repaired - blur), 0), 2.0, 98.7)
    crisp = crisp * (0.28 + 0.72 * np.maximum(support_mask, high.astype(np.float32)))
    crisp = normalize_percentile(crisp, 2.4, 98.6)

    binary = binarize_clean_high(crisp)
    added_audit = normalize_percentile(np.maximum(crisp - sig, 0), 1, 99)
    return {
        "clean_high": crisp.astype(np.float32),
        "clean_high_bin": binary.astype(np.float32),
        "support_mask": support_mask.astype(np.float32),
        "added_audit": added_audit.astype(np.float32),
    }


def make_contact_sheet(labels: list[str], columns: dict[str, list[np.ndarray]], outpath: Path) -> None:
    names = list(columns)
    tile = (270, 270)
    header = 58
    label_w = 48
    sheet = Image.new("RGB", (label_w + len(names) * tile[0], header + len(labels) * tile[1]), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(names):
        x = label_w + c * tile[0]
        draw.rectangle([x, 0, x + tile[0], header], fill=(242, 242, 242))
        draw.text((x + 8, 10), name, fill=(0, 0, 0))
    for r, label in enumerate(labels):
        y = header + r * tile[1]
        draw.rectangle([0, y, label_w, y + tile[1]], fill=(248, 248, 248))
        draw.text((17, y + 14), label, fill=(0, 0, 0))
        for c, name in enumerate(names):
            image = columns[name][r]
            if image.ndim == 3:
                thumb = Image.fromarray(image.astype(np.uint8)).convert("RGB")
                fill = "white"
            else:
                thumb = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).convert("RGB")
                fill = "black" if name.endswith("_bin") else "white"
            thumb.thumbnail(tile, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", tile, fill)
            canvas.paste(thumb, ((tile[0] - thumb.width) // 2, (tile[1] - thumb.height) // 2))
            sheet.paste(canvas, (label_w + c * tile[0], y))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(outpath)
