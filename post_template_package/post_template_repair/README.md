# Post-Template Repair Package

This package contains only the post-template stage:

1. read `*_05_binary_denoised.png` and `*_03_interlayer_cleaned.png`;
2. automatically match template digits `1`, `3`, and `4`;
3. select the highest-scoring template;
4. generate final target-supported signal and binary outputs.

It does not include Gradio.

## Run

```bash
python post_template_repair.py ^
  --processed-dir C:\Users\jingl\Desktop\final\output\processed ^
  --outdir C:\Users\jingl\Desktop\final\output\auto_template_select_fixed_params_v1 ^
  --data-ids 1,3,4
```

## Required Inputs

For each data ID, `--processed-dir` must contain:

- `*{data_id}_03_interlayer_cleaned.png`
- `*{data_id}_05_binary_denoised.png`

The package includes template core masks in `templates/`:

- `template1_core.png`
- `template3_core.png`
- `template4_core.png`

## Outputs

- `data*_auto_selected_template*_core.png`
- `data*_auto_selected_template*_gate.png`
- `data*_auto_template_final_signal_gray.png`
- `data*_auto_template_final_binary.png`
- `data*_auto_template_overlay.png`
- `auto_template_selection_summary.csv`
- `auto_template_all_candidate_scores.csv`
- `auto_template_selection_final_overview.png`

## Fixed Parameters

Template class is selected automatically. Matching weights, search ranges, gate dilation, and repair parameters are fixed in `post_template_repair.py`.
