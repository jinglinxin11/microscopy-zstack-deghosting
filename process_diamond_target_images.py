from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def robust_norm(x: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_uint8(x: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    return (robust_norm(x, low, high) * 255.0 + 0.5).astype(np.uint8)


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def ascii_label(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return f"image {match.group(1)}" if match else path.stem.encode("ascii", "ignore").decode()


def find_target_dir(cwd: Path) -> Path:
    candidates = []
    for path in cwd.rglob("*"):
        if path.is_dir():
            tiffs = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
            if tiffs:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No directory containing TIFF images was found.")
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def load_gray(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def suppress_frame_edges(gray: np.ndarray, border: int = 28) -> np.ndarray:
    out = gray.copy()
    if out.shape[0] <= 2 * border or out.shape[1] <= 2 * border:
        return out
    fill = float(np.median(out[border:-border, border:-border]))
    out[:border, :] = fill
    out[-border:, :] = fill
    out[:, :border] = fill
    out[:, -border:] = fill
    return out


def dark_feature_signal(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper-inspired correction for dark marks on a bright uneven background."""
    gray = suppress_frame_edges(gray)

    # Widefield illumination correction: estimate the low-frequency profile.
    scale = 0.25
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bg_small = cv2.GaussianBlur(small, (0, 0), sigmaX=170 * scale, sigmaY=170 * scale)
    bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    illum_corrected_dark = np.maximum(bg - gray, 0.0)

    # Equivalent to applying a white top-hat to the inverted image.
    # Rectangular kernels are separable/optimized in OpenCV and avoid the very
    # high cost of a large elliptical kernel at 3000 x 3000 pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (181, 181))
    blackhat = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_BLACKHAT, kernel)

    # Combine smooth background subtraction with morphology-based top-hat.
    signal = 0.55 * robust_norm(illum_corrected_dark, 0.2, 99.7) + 0.45 * robust_norm(
        blackhat, 0.2, 99.7
    )
    signal = cv2.medianBlur((robust_norm(signal, 0.5, 99.8) * 255).astype(np.uint8), 3)
    signal = signal.astype(np.float32) / 255.0
    return bg, illum_corrected_dark, signal


def phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    ref = cv2.resize(reference, (750, 750), interpolation=cv2.INTER_AREA).astype(np.float32)
    mov = cv2.resize(moving, (750, 750), interpolation=cv2.INTER_AREA).astype(np.float32)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    mov = (mov - mov.mean()) / (mov.std() + 1e-6)
    (shift_x, shift_y), response = cv2.phaseCorrelate(ref, mov)
    scale_x = reference.shape[1] / 750.0
    scale_y = reference.shape[0] / 750.0
    return shift_x * scale_x, shift_y * scale_y, response


def remove_interlayer_crosstalk(signals: list[np.ndarray]) -> list[np.ndarray]:
    if len(signals) == 1:
        return [robust_norm(signals[0], 0.5, 99.7)]

    stack = np.stack([robust_norm(s, 0.5, 99.5) for s in signals], axis=0)

    # The paper removes signals from other layers after inter-layer matching.
    # With aligned same-size inputs, subtract the second strongest layer response
    # pixelwise; this keeps features strongest in the current layer.
    sorted_stack = np.sort(stack, axis=0)
    second_strongest = sorted_stack[-2]
    median_signal = np.median(stack, axis=0)
    cleaned = []
    for i in range(stack.shape[0]):
        other = np.maximum(0.80 * second_strongest, 0.55 * median_signal)
        layer = np.clip(stack[i] - other, 0.0, None)
        layer = cv2.GaussianBlur(layer.astype(np.float32), (0, 0), sigmaX=1.2)
        layer = robust_norm(layer, 0.5, 99.7)
        cleaned.append(layer)
    return cleaned


def save_gray_png(path: Path, data: np.ndarray) -> None:
    Image.fromarray(to_uint8(data, 0.5, 99.7)).save(path)


def save_gray_tiff(path: Path, data: np.ndarray) -> None:
    # Avoid compressed TIFF here: this Windows Pillow/libtiff build exits
    # abruptly with tiff_deflate. Uncompressed TIFF is larger but stable.
    Image.fromarray(to_uint8(data, 0.5, 99.7)).save(path)


def presentation_enhance(data: np.ndarray) -> np.ndarray:
    x = robust_norm(data, 0.2, 99.7)
    x = np.power(x, 0.58)
    x = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(10, 10)).apply((x * 255).astype(np.uint8))
    return x


def binarize_denoised(data: np.ndarray) -> tuple[np.ndarray, float]:
    x = to_uint8(data, 0.5, 99.7)
    x = cv2.GaussianBlur(x, (0, 0), sigmaX=1.8, sigmaY=1.8)

    border = 70
    valid = x[border:-border, border:-border] if min(x.shape) > 2 * border else x
    threshold = float(np.percentile(valid, 86))
    mask = np.where(x >= threshold, 255, 0).astype(np.uint8)

    # Remove frame artifacts from acquisition and preserve the central target.
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for idx in range(1, components):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= 450:
            cleaned[labels == idx] = 255

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned, threshold


def make_contact_sheet(items: list[tuple[str, Image.Image]], out_path: Path, cols: int = 4) -> None:
    thumbs = []
    for label, image in items:
        im = image.convert("L")
        im = ImageOps.autocontrast(im, cutoff=0.5).convert("RGB")
        im.thumbnail((620, 620), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (620, 665), "white")
        canvas.paste(im, ((620 - im.width) // 2, 45))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, 620, 36], fill=(0, 0, 0))
        draw.text((10, 10), label, fill=(255, 255, 255))
        thumbs.append(canvas)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 620, rows * 665), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 620, (i // cols) * 665))
    sheet.save(out_path)


def main() -> None:
    cwd = Path.cwd()
    target_dir = find_target_dir(cwd)
    out_dir = cwd / "output" / "processed"
    diag_dir = cwd / "output" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(target_dir.glob("*.tif")) + list(target_dir.glob("*.tiff")), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No TIFF images found in {target_dir}")

    grays = [load_gray(p) for p in files]
    backgrounds = []
    corrected = []
    tophat = []
    for gray in grays:
        bg, corr, signal = dark_feature_signal(gray)
        backgrounds.append(bg)
        corrected.append(corr)
        tophat.append(signal)

    cleaned = remove_interlayer_crosstalk(tophat)

    report_lines = [
        "Diamond target image processing report",
        f"Input directory: {target_dir}",
        "Method: low-frequency illumination correction, dark-feature top-hat, inter-layer crosstalk subtraction.",
        "",
        "Estimated phase-correlation shifts against first image (pixels):",
    ]
    for idx, (path, sig) in enumerate(zip(files, tophat), start=1):
        sx, sy, response = phase_shift(tophat[0], sig)
        report_lines.append(
            f"- image {idx} ({ascii_label(path)}): x={sx:.2f}, y={sy:.2f}, response={response:.4f}"
        )

    overview_items = []
    binary_items = []
    for path, gray, corr, signal, clean in zip(files, grays, corrected, tophat, cleaned):
        stem = path.stem
        label = ascii_label(path)
        raw8 = to_uint8(suppress_frame_edges(gray), 0.5, 99.5)
        corr8 = to_uint8(corr, 0.2, 99.8)
        top8 = to_uint8(signal, 0.5, 99.7)
        clean8 = to_uint8(clean, 0.5, 99.7)
        final8 = presentation_enhance(signal)
        binary8, binary_threshold = binarize_denoised(signal)
        binary_white_bg = 255 - binary8
        report_lines.append(f"- {label} binary threshold: {binary_threshold:.1f}")

        Image.fromarray(raw8).save(out_dir / f"{stem}_00_raw_autocontrast.png")
        Image.fromarray(corr8).save(out_dir / f"{stem}_01_illumination_corrected.png")
        Image.fromarray(top8).save(out_dir / f"{stem}_02_tophat_denoised.png")
        Image.fromarray(clean8).save(out_dir / f"{stem}_03_interlayer_cleaned.png")
        Image.fromarray(final8).save(out_dir / f"{stem}_04_final_enhanced.png")
        Image.fromarray(binary8).save(out_dir / f"{stem}_05_binary_denoised.png")
        Image.fromarray(binary_white_bg).save(out_dir / f"{stem}_05_binary_denoised_white_bg.png")
        save_gray_tiff(out_dir / f"{stem}_03_interlayer_cleaned.tiff", clean)

        overview_items.extend(
            [
                (f"{label} raw", Image.fromarray(raw8)),
                (f"{label} corrected", Image.fromarray(corr8)),
                (f"{label} top-hat", Image.fromarray(top8)),
                (f"{label} final", Image.fromarray(final8)),
                (f"{label} layer-cleaned", Image.fromarray(clean8)),
            ]
        )
        binary_items.extend(
            [
                (f"{label} top-hat", Image.fromarray(top8)),
                (f"{label} binary", Image.fromarray(binary8)),
                (f"{label} white-bg", Image.fromarray(binary_white_bg)),
            ]
        )

    make_contact_sheet(overview_items, out_dir / "diamond_processing_overview.png", cols=5)
    make_contact_sheet(binary_items, out_dir / "diamond_binary_denoising_overview.png", cols=3)
    (out_dir / "processing_report.txt").write_text("\n".join(report_lines), encoding="utf-8-sig")
    print(f"Processed {len(files)} TIFF images.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
