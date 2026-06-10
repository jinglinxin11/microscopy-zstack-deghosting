from __future__ import annotations

import numpy as np

from src.clean_high_bin import clean_high_bin, normalize_percentile
from src.raw_pipeline import run_raw_pipeline


def test_normalize_percentile_constant_image() -> None:
    image = np.ones((16, 16), dtype=np.float32)
    result = normalize_percentile(image)
    assert result.shape == image.shape
    assert np.all(result == 0)


def test_clean_high_bin_outputs_expected_keys_and_shapes() -> None:
    clean = np.zeros((64, 64), dtype=np.float32)
    evidence = np.zeros((64, 64), dtype=np.float32)
    clean[20:45, 30:34] = 1.0
    evidence[18:47, 28:36] = 1.0

    result = clean_high_bin(clean, evidence)

    assert set(result) == {"clean_high", "clean_high_bin", "support_mask", "added_audit"}
    for value in result.values():
        assert value.shape == clean.shape
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    assert result["clean_high_bin"].max() <= 1.0
    assert result["clean_high_bin"].min() >= 0.0


def test_raw_pipeline_outputs_full_stack() -> None:
    images = []
    for offset in range(4):
        image = np.full((64, 64, 3), 180, dtype=np.uint8)
        image[12 + offset * 2 : 50 + offset * 2, 28:34, :] = 45
        images.append(image)

    result = run_raw_pipeline(images)

    expected = {
        "dark",
        "enhanced",
        "weights",
        "confidence",
        "source_hybrid_soft",
        "paper_clean",
        "residual",
        "clean_high",
        "clean_high_bin",
        "support_mask",
        "added_audit",
    }
    assert set(result) == expected
    for key, value in result.items():
        if key == "confidence":
            assert value.shape == (64, 64)
        else:
            assert value.shape == (4, 64, 64)
        assert np.isfinite(value).all()
