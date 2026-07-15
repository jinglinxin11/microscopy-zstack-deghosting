from __future__ import annotations

import numpy as np
import pytest

from microscopy_matching.evidence_mask import binary_image, matched_only_mask, native_binary_image


def test_matched_only_mask_is_a_subset_of_target_and_corridor() -> None:
    target = np.zeros((30, 30), dtype=bool)
    target[8:22, 14] = True
    target[3, 3] = True
    points = np.asarray([[14.0, float(y)] for y in range(8, 22)], dtype=np.float32)

    result = matched_only_mask(target, points, corridor_radius_px=2)

    assert result.mask[8:22, 14].all()
    assert not result.mask[3, 3]
    assert np.all(result.mask <= target)
    assert np.all(result.mask <= result.corridor)


def test_binary_image_and_native_resize_keep_binary_values() -> None:
    mask = np.asarray([[False, True], [True, False]])

    image = binary_image(mask)
    native = native_binary_image(mask, (8, 10))

    assert set(np.unique(image)) == {0, 255}
    assert native.shape == (8, 10)
    assert set(np.unique(native)) == {0, 255}


def test_matched_only_mask_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        matched_only_mask(np.zeros((2, 2, 2), dtype=bool), np.asarray([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="shape"):
        matched_only_mask(np.zeros((2, 2), dtype=bool), np.asarray([[1.0]]))
    with pytest.raises(ValueError, match="at least"):
        matched_only_mask(np.zeros((2, 2), dtype=bool), np.asarray([[1.0, 1.0]]), corridor_radius_px=0)
