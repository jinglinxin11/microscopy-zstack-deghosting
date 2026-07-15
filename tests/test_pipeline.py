from pathlib import Path

import numpy as np

from microscopy_matching.pipeline import PipelineRun, SelectedMatch, minimal_results_payload


def test_minimal_results_payload_contains_only_final_result_references() -> None:
    row = {
        "target_id": "target_01",
        "selected_label": "S",
        "selected_score": 0.5,
        "runner_up_label": "T",
        "margin": 0.1,
        "decision_status": "review_required_topology",
        "analysis_scale": 1.0,
        "analysis_angle_deg": 0.0,
        "analysis_dx": 1.0,
        "analysis_dy": 2.0,
        "selected_native_bbox_xyxy": "1 2 3 4",
        "status_flags": "flag",
    }
    selection = SelectedMatch(
        target_index=0,
        candidate_index=0,
        target_path=Path("S.png"),
        candidate_path=Path("S.png"),
        target_original=np.zeros((1, 1, 3), dtype=np.uint8),
        target=None,  # type: ignore[arg-type]
        auxiliary=None,  # type: ignore[arg-type]
        match=None,  # type: ignore[arg-type]
        summary_row=row,
        rendered=np.zeros((1, 1, 3), dtype=np.uint8),
    )
    payload = minimal_results_payload(
        PipelineRun(Path("."), Path("."), (), (row,), (selection,))
    )

    assert payload["mode"] == "automatic_independent_no_batch_assignment"
    assert payload["binary_rule"] == "target_foreground_and_registered_auxiliary_corridor"
    assert payload["results"] == [
        {
            "target_id": "target_01",
            "selected_label": "S",
            "selected_score": 0.5,
            "runner_up_label": "T",
            "margin": 0.1,
            "decision_status": "review_required_topology",
            "analysis_transform": {"scale": 1.0, "angle_deg": 0.0, "dx": 1.0, "dy": 2.0},
            "native_bbox_xyxy": "1 2 3 4",
            "status_flags": "flag",
            "presentation_file": "presentation/target_01_S.png",
            "binary_file": "binary/target_01_S.png",
        }
    ]
