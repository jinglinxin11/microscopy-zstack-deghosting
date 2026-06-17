from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import morphology
from skimage.transform import resize


# Fixed parameters from the selected post-template workflow.
TEMPLATE_DIGITS = (1, 3, 4)
MAXDIM = 1000
SCORE_W = {"dice": 0.42, "inside": 0.22, "core_hit": 0.28, "area": 0.08}
COMBINED_W = {"corr": 0.33, "score": 0.52, "center": 0.15}
BORDER_PENALTY = 0.88
GATE_DILATION = 16
GATE_RELAXED_DILATION = 8
REPAIR_PERCENTILE = 72
NEAR_DILATION = 10
CLOSING_RADIUS = 2
FINAL_MIN_SIZE = 160
INPUT_MIN_SIZE = 180
INPUT_HOLE_AREA = 160


def read_gray01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").save(path)


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8), "L").save(path)


def norm01(image: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(image[finite], [low, high])
    if hi <= lo + 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def clean_binary(mask: np.ndarray) -> np.ndarray:
    mask = morphology.remove_small_objects(np.asarray(mask, dtype=bool), min_size=INPUT_MIN_SIZE)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, area_threshold=INPUT_HOLE_AREA)
    return mask.astype(bool)


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def resize_float(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(
        image.astype(np.float32),
        shape,
        order=1,
        mode="constant",
        cval=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def downsample_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = mask.shape
    scale = MAXDIM / max(height, width)
    shape = (int(round(height * scale)), int(round(width * scale)))
    return resize_bool(mask, shape), scale


def crop_tight(mask: np.ndarray, pad: int = 4) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad)
    return mask[y0:y1, x0:x1], (y0, x0, y1, x1)


def place_template(shape: tuple[int, int], template: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    height, width = template.shape
    y0 = max(0, top)
    x0 = max(0, left)
    y1 = min(shape[0], top + height)
    x1 = min(shape[1], left + width)
    sy0 = max(0, -top)
    sx0 = max(0, -left)
    sy1 = sy0 + (y1 - y0)
    sx1 = sx0 + (x1 - x0)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = template[sy0:sy1, sx0:sx1]
    return out


def score_core(target: np.ndarray, core: np.ndarray) -> tuple[float, float, float, float, float]:
    gate = ndi.binary_dilation(core, iterations=max(4, int(round(target.shape[0] / 190))))
    inter = int(np.count_nonzero(target & gate))
    target_count = max(1, int(np.count_nonzero(target)))
    gate_count = max(1, int(np.count_nonzero(gate)))
    core_inter = int(np.count_nonzero(target & core))
    core_count = max(1, int(np.count_nonzero(core)))
    dice = 2 * inter / max(1, target_count + gate_count)
    inside = inter / target_count
    core_hit = core_inter / core_count
    area = min(target_count, gate_count) / max(target_count, gate_count)
    score = (
        SCORE_W["dice"] * dice
        + SCORE_W["inside"] * inside
        + SCORE_W["core_hit"] * core_hit
        + SCORE_W["area"] * area
    )
    return score, dice, inside, core_hit, area


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def match_template_low(target: np.ndarray, template_core: np.ndarray, template_digit: int) -> dict[str, object]:
    target_float = target.astype(np.float32)
    target_blur = cv2.GaussianBlur(target_float, (0, 0), 1.5)
    center_y, center_x = mask_center(target)
    best: dict[str, object] | None = None
    scales = np.linspace(0.78, 1.25, 20) if template_digit != 4 else np.linspace(0.72, 1.34, 24)

    for scale in scales:
        new_h = max(10, int(round(template_core.shape[0] * scale)))
        new_w = max(10, int(round(template_core.shape[1] * scale)))
        if new_h >= target.shape[0] or new_w >= target.shape[1]:
            continue
        resized = cv2.resize(template_core.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized = (resized > 0.25).astype(np.float32)
        if resized.sum() < 10:
            continue
        response = cv2.matchTemplate(target_blur, cv2.GaussianBlur(resized, (0, 0), 0.8), cv2.TM_CCOEFF_NORMED)
        flat = response.ravel()
        top_k = min(30, flat.size)
        candidate_indices = np.argpartition(flat, -top_k)[-top_k:]
        for flat_index in candidate_indices:
            top, left = np.unravel_index(flat_index, response.shape)
            placed = place_template(target.shape, resized > 0.5, int(top), int(left))
            score, dice, inside, core_hit, area = score_core(target, placed)
            placed_y = top + new_h / 2
            placed_x = left + new_w / 2
            distance = ((placed_y - center_y) / (0.33 * target.shape[0])) ** 2 + (
                (placed_x - center_x) / (0.33 * target.shape[1])
            ) ** 2
            center_prior = float(np.exp(-0.5 * distance))
            border_penalty = (
                BORDER_PENALTY
                if top <= 1
                or left <= 1
                or top + new_h >= target.shape[0] - 1
                or left + new_w >= target.shape[1] - 1
                else 1.0
            )
            combined = (
                COMBINED_W["corr"] * float(flat[flat_index])
                + COMBINED_W["score"] * score
                + COMBINED_W["center"] * center_prior
            ) * border_penalty
            if best is None or combined > float(best["combined"]):
                best = {
                    "combined": combined,
                    "corr": float(flat[flat_index]),
                    "score": score,
                    "dice": dice,
                    "inside": inside,
                    "core_hit": core_hit,
                    "area": area,
                    "center_prior": center_prior,
                    "scale": float(scale),
                    "top": int(top),
                    "left": int(left),
                    "core": resized > 0.5,
                    "template_digit": template_digit,
                }
    if best is None:
        raise RuntimeError(f"Template matching failed for template {template_digit}.")
    return best


def final_signal(binary: np.ndarray, signal: np.ndarray, gate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal_norm = norm01(signal, 0.8, 99.7)
    gate_relaxed = ndi.binary_dilation(gate, iterations=GATE_RELAXED_DILATION)
    retained = binary & gate_relaxed
    gate_values = signal_norm[gate_relaxed]
    threshold = float(np.percentile(gate_values, REPAIR_PERCENTILE)) if gate_values.size else 1.0
    near_existing = ndi.binary_dilation(retained, iterations=NEAR_DILATION)
    repair = (signal_norm >= threshold) & gate_relaxed & near_existing
    final = retained | repair
    final = morphology.binary_closing(final, morphology.disk(CLOSING_RADIUS))
    final = morphology.remove_small_objects(final, min_size=FINAL_MIN_SIZE)
    final_gray = np.where(final, np.clip(0.18 + 0.82 * signal_norm, 0.0, 1.0), 0.0)
    return final_gray, final.astype(bool), retained.astype(bool), repair.astype(bool)


def overlay_selection(target: np.ndarray, core: np.ndarray) -> np.ndarray:
    gate = ndi.binary_dilation(core, iterations=GATE_DILATION)
    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rgb[target] = (90, 90, 90)
    gate_edge = gate & ~ndi.binary_erosion(gate, iterations=2)
    core_edge = core & ~ndi.binary_erosion(core, iterations=2)
    rgb[gate_edge] = (0, 220, 255)
    rgb[core_edge] = (255, 255, 255)
    return rgb


def thumbnail(image: np.ndarray, title: str, maskmode: bool = False) -> Image.Image:
    if image.ndim == 2:
        array = image.astype(np.uint8) * 255 if image.dtype == bool else np.clip(image * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, "L").convert("RGB")
    else:
        pil = Image.fromarray(image.astype(np.uint8), "RGB")
    pil.thumbnail((350, 350), Image.Resampling.NEAREST if maskmode else Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (375, 405), "white")
    canvas.paste(pil, ((375 - pil.width) // 2, 38))
    ImageDraw.Draw(canvas).text((8, 10), title, fill=(0, 0, 0))
    return canvas


def find_processed_file(processed_dir: Path, data_id: int, suffix: str) -> Path:
    key = f"{data_id}_{suffix}"
    for path in processed_dir.iterdir():
        if path.is_file() and path.name.endswith(key):
            return path
    raise FileNotFoundError(f"Cannot find file ending with {key} in {processed_dir}")


def load_template_cores(template_dir: Path) -> dict[int, np.ndarray]:
    templates = {}
    for digit in TEMPLATE_DIGITS:
        path = template_dir / f"template{digit}_core.png"
        if not path.exists():
            raise FileNotFoundError(path)
        templates[digit] = read_mask(path)
    return templates


def process_all(processed_dir: Path, template_dir: Path, outdir: Path, data_ids: tuple[int, ...]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    template_cores_full = load_template_cores(template_dir)
    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    overview_rows = []

    for data_id in data_ids:
        binary = clean_binary(read_mask(find_processed_file(processed_dir, data_id, "05_binary_denoised.png")))
        signal = read_gray01(find_processed_file(processed_dir, data_id, "03_interlayer_cleaned.png"))
        if signal.shape != binary.shape:
            signal = resize_float(signal, binary.shape)

        target_low, scale = downsample_mask(binary)
        inv_scale = 1.0 / scale
        candidates = []
        for template_digit in TEMPLATE_DIGITS:
            core_low = resize_bool(template_cores_full[template_digit], target_low.shape)
            crop, bbox = crop_tight(core_low, pad=4)
            match = match_template_low(target_low, crop, template_digit)
            match["template_crop_bbox_low"] = bbox
            candidates.append(match)
            candidate_rows.append(
                {
                    "data_id": data_id,
                    "template_digit": template_digit,
                    **{
                        key: (f"{value:.6f}" if isinstance(value, float) else value)
                        for key, value in match.items()
                        if key != "core"
                    },
                }
            )

        selected = max(candidates, key=lambda item: float(item["combined"]))
        second = sorted(candidates, key=lambda item: float(item["combined"]), reverse=True)[1]
        selected_digit = int(selected["template_digit"])
        top_full = int(round(int(selected["top"]) * inv_scale))
        left_full = int(round(int(selected["left"]) * inv_scale))
        core_low = np.asarray(selected["core"], dtype=bool)
        full_h = max(10, int(round(core_low.shape[0] * inv_scale)))
        full_w = max(10, int(round(core_low.shape[1] * inv_scale)))
        core_full = cv2.resize(core_low.astype(np.uint8), (full_w, full_h), interpolation=cv2.INTER_NEAREST) > 0
        placed_core = place_template(binary.shape, core_full, top_full, left_full)
        placed_gate = ndi.binary_dilation(placed_core, iterations=GATE_DILATION)
        final_gray, final_binary, retained, repair = final_signal(binary, signal, placed_gate)
        overlay = overlay_selection(binary, placed_core)

        save_mask(outdir / f"data{data_id}_auto_selected_template{selected_digit}_core.png", placed_core)
        save_mask(outdir / f"data{data_id}_auto_selected_template{selected_digit}_gate.png", placed_gate)
        save_gray(outdir / f"data{data_id}_auto_template_final_signal_gray.png", final_gray)
        save_mask(outdir / f"data{data_id}_auto_template_final_binary.png", final_binary)
        Image.fromarray(overlay, "RGB").save(outdir / f"data{data_id}_auto_template_overlay.png")

        margin = float(selected["combined"]) - float(second["combined"])
        summary_rows.append(
            {
                "data_id": data_id,
                "selected_template_digit": selected_digit,
                "best_combined": f"{float(selected['combined']):.6f}",
                "second_template_digit": int(second["template_digit"]),
                "second_combined": f"{float(second['combined']):.6f}",
                "score_margin": f"{margin:.6f}",
                "match_scale": f"{float(selected['scale']):.4f}",
                "top_full": top_full,
                "left_full": left_full,
                "retained_px": int(retained.sum()),
                "repaired_px": int((repair & ~retained).sum()),
            }
        )
        overview_rows.append((data_id, selected_digit, binary, placed_gate, final_gray, final_binary, overlay, margin))

    write_csv(outdir / "auto_template_selection_summary.csv", summary_rows)
    write_csv(outdir / "auto_template_all_candidate_scores.csv", candidate_rows)
    write_overview(outdir / "auto_template_selection_final_overview.png", overview_rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overview(path: Path, rows: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]) -> None:
    sheet = Image.new("RGB", (375 * 5, 405 * len(rows)), "white")
    for row_idx, (data_id, selected_digit, binary, gate, final_gray, final_binary, overlay, margin) in enumerate(rows):
        tiles = [
            thumbnail(binary, f"data {data_id} binary input", True),
            thumbnail(gate, f"auto template {selected_digit}\nmargin {margin:.3f}", True),
            thumbnail(final_gray, "final signal"),
            thumbnail(final_binary, "final binary", True),
            thumbnail(overlay, "overlay"),
        ]
        for col_idx, tile in enumerate(tiles):
            sheet.paste(tile, (375 * col_idx, 405 * row_idx))
    sheet.save(path)


def parse_data_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic template selection and post-template signal repair.")
    parser.add_argument("--processed-dir", type=Path, required=True, help="Directory containing *_03_interlayer_cleaned.png and *_05_binary_denoised.png.")
    parser.add_argument("--template-dir", type=Path, default=Path(__file__).resolve().parent / "templates")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--data-ids", type=parse_data_ids, default=(1, 3, 4), help="Comma-separated IDs, e.g. 1,3,4.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_all(args.processed_dir, args.template_dir, args.outdir, args.data_ids)
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
