from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.clean_high_bin import LABELS, clean_high_bin, display_signal, normalize_percentile


def load_raw_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def rgb_dark_signal(rgb: np.ndarray, bg_sigma: float = 80.0) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0].astype(np.float32) / 255.0
    g_chan = rgb[:, :, 1].astype(np.float32) / 255.0
    gray = normalize_percentile(0.6 * l_chan + 0.4 * g_chan, 1, 99.5)
    bg = cv2.GaussianBlur(gray, (0, 0), bg_sigma)
    flat = normalize_percentile(gray / (bg + 1e-4), 1, 99.5)
    dark = np.percentile(flat, 99.0) - flat
    dark = normalize_percentile(np.clip(dark, 0, None), 1, 99.5)
    dark = cv2.medianBlur((dark * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    return dark.astype(np.float32)


def blackhat_multiscale(dark: np.ndarray, radii: tuple[int, ...] = (5, 9, 15, 25)) -> np.ndarray:
    image = np.clip(dark * 255, 0, 255).astype(np.uint8)
    responses = []
    for radius in radii:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        responses.append(np.maximum(closed.astype(np.float32) - image.astype(np.float32), 0) / 255.0)
    return normalize_percentile(np.max(np.stack(responses, axis=0), axis=0), 1, 99.5)


def dog_multiscale(dark: np.ndarray, pairs: tuple[tuple[float, float], ...] = ((1.5, 6.0), (2.5, 10.0), (4.0, 16.0))) -> np.ndarray:
    responses = []
    for small, large in pairs:
        fine = cv2.GaussianBlur(dark, (0, 0), small)
        coarse = cv2.GaussianBlur(dark, (0, 0), large)
        responses.append(np.maximum(fine - coarse, 0))
    return normalize_percentile(np.max(np.stack(responses, axis=0), axis=0), 1, 99.5)


def enhance_dark(dark: np.ndarray) -> np.ndarray:
    blackhat = blackhat_multiscale(dark)
    dog = dog_multiscale(dark)
    return normalize_percentile(0.55 * blackhat + 0.45 * dog, 1, 99.5)


def focus_weights(enhanced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    for layer in enhanced:
        pre = cv2.GaussianBlur(layer, (0, 0), 1.0)
        gx = cv2.Scharr(pre, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(pre, cv2.CV_32F, 0, 1)
        score = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 5.0)
        scores.append(score)
    focus = np.stack(scores).astype(np.float32)
    logits = 6.0 * normalize_percentile(focus, 1, 99.5)
    logits -= logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    weights = exp_logits / (exp_logits.sum(axis=0, keepdims=True) + 1e-6)

    sorted_focus = np.sort(focus, axis=0)
    top1 = sorted_focus[-1]
    top2 = sorted_focus[-2]
    peakness = (top1 - top2) / (top1 + 1e-6)
    confidence = np.clip((peakness - 0.10) / 0.25, 0, 1).astype(np.float32)
    weights = confidence[None, :, :] * weights + (1 - confidence[None, :, :]) * (1.0 / focus.shape[0])

    for idx in range(weights.shape[0]):
        weights[idx] = cv2.GaussianBlur(weights[idx].astype(np.float32), (0, 0), 2.0)
    weights /= weights.sum(axis=0, keepdims=True) + 1e-6
    return weights.astype(np.float32), confidence.astype(np.float32)


def separate_layers(enhanced: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources = []
    paper_clean = []
    residuals = []
    for idx in range(enhanced.shape[0]):
        target = enhanced[idx]
        others = np.median(np.delete(enhanced, idx, axis=0), axis=0)
        source = normalize_percentile(target * (0.55 + 0.45 * weights[idx]), 1, 99.3)
        clean = target * weights[idx] - 0.22 * others * (1.0 - weights[idx])
        clean = normalize_percentile(np.clip(clean, 0, None), 2.0, 99.0)
        clean = cv2.morphologyEx(
            clean.astype(np.float32),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        sources.append(source)
        paper_clean.append(normalize_percentile(clean, 1, 99.0))
        residuals.append(normalize_percentile(others * (1.0 - weights[idx]), 1, 99.2))
    return np.stack(sources), np.stack(paper_clean), np.stack(residuals)


def run_raw_pipeline(raw_images: list[np.ndarray]) -> dict[str, np.ndarray]:
    dark = np.stack([rgb_dark_signal(img) for img in raw_images]).astype(np.float32)
    enhanced = np.stack([enhance_dark(layer) for layer in dark]).astype(np.float32)
    weights, confidence = focus_weights(enhanced)
    source_hybrid_soft, paper_clean, residual = separate_layers(enhanced, weights)

    clean_high, clean_high_bin, support_mask, added_audit = [], [], [], []
    for idx in range(len(raw_images)):
        result = clean_high_bin_stage(paper_clean[idx], source_hybrid_soft[idx])
        clean_high.append(result["clean_high"])
        clean_high_bin.append(result["clean_high_bin"])
        support_mask.append(result["support_mask"])
        added_audit.append(result["added_audit"])

    return {
        "dark": dark,
        "enhanced": enhanced,
        "weights": weights,
        "confidence": confidence,
        "source_hybrid_soft": source_hybrid_soft,
        "paper_clean": paper_clean,
        "residual": residual,
        "clean_high": np.stack(clean_high).astype(np.float32),
        "clean_high_bin": np.stack(clean_high_bin).astype(np.float32),
        "support_mask": np.stack(support_mask).astype(np.float32),
        "added_audit": np.stack(added_audit).astype(np.float32),
    }


def clean_high_bin_stage(paper_clean: np.ndarray, source_hybrid_soft: np.ndarray) -> dict[str, np.ndarray]:
    return clean_high_bin(paper_clean, source_hybrid_soft)


def display_stack(stack: np.ndarray) -> list[np.ndarray]:
    return [display_signal(layer) for layer in stack]
