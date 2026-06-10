from __future__ import annotations

import numpy as np

from src.new_method import binary_from_signal, render_contact_sheet, save_level_outputs, twenty_four_level_specs


def test_twenty_four_level_specs_are_ordered() -> None:
    specs = twenty_four_level_specs()
    assert len(specs) == 24
    assert [spec.source for spec in specs[:12]] == ["clean"] * 12
    assert [spec.source for spec in specs[12:]] == ["ultra"] * 12
    assert specs[0].variant == "clean_01_p54_bin"
    assert specs[-1].variant == "ultra_12_p80_bin"


def test_binary_from_signal_removes_small_components() -> None:
    signal = np.zeros((64, 64), dtype=np.float32)
    signal[20:45, 25:40] = 1.0
    signal[2:4, 2:4] = 1.0
    spec = twenty_four_level_specs()[0]

    mask = binary_from_signal(signal, spec)

    assert mask.shape == signal.shape
    assert mask.dtype == np.float32
    assert mask[25:40, 28:37].mean() > 0.8
    assert mask[2:4, 2:4].sum() == 0


def test_save_level_outputs_and_contact_sheet(tmp_path) -> None:
    clean = {"1": np.zeros((32, 32), dtype=np.float32)}
    ultra = {"1": np.zeros((32, 32), dtype=np.float32)}
    clean["1"][8:24, 10:16] = 1.0
    ultra["1"][12:24, 18:24] = 1.0

    outputs = save_level_outputs(clean, ultra, tmp_path / "levels", ["1"])
    render_contact_sheet(outputs, tmp_path / "overview.png", layers=["1"], thumb=32)

    assert len(outputs) == 24
    assert (tmp_path / "levels" / "layer1_ultra_03_p62_bin.png").exists()
    assert (tmp_path / "overview.png").exists()
