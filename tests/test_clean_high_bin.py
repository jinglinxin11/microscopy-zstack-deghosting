from __future__ import annotations

import numpy as np

from src.clean_high_bin import clean_high_bin, normalize_percentile


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
