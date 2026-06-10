# Microscopy Z-Stack Deghosting

This repository implements a compact image-processing pipeline for layered
microscopy digit images. It includes a legacy-reference mode that reproduces
the selected previous `clean_high_bin` result, and a raw-estimate mode that
computes approximate intermediate images from raw RGB inputs only.

The method does not use digit templates, OCR, label priors, or machine
learning. It uses dark-signal extraction, multi-scale enhancement, focus-based
soft separation, evidence-constrained local repair, and binary conversion.

## Project Layout

```text
.
├── run_from_raw.py              # full raw -> final pipeline
├── run_clean_high_bin.py        # final-stage-only compatibility runner
├── raw_data/                    # raw-only example input
├── sample_data/                 # final-stage example input
├── src/
│   ├── raw_pipeline.py
│   └── clean_high_bin.py
└── tests/
```

## Recommended Reproduction Pipeline

Input structure:

```text
raw_data/
└── layers/
    ├── 1/raw.png
    ├── 3/raw.png
    ├── 4/raw.png
    └── 7/raw.png
```

Run:

```bash
python run_from_raw.py
```

Default mode is `legacy-reference`. It reads raw images for display and uses the
saved legacy intermediate images in `sample_data`:

```text
raw.png
+ saved paper_clean.png
+ saved source_hybrid_soft.png
-> clean_high
-> clean_high_bin
```

This is the mode that reproduces the selected previous result most closely.

## Raw-Only Approximation

If you need a pure raw-only run, use:

```bash
python run_from_raw.py --mode raw-estimate
```

The raw-estimate pipeline is:

```text
raw RGB image
-> dark signal extraction
-> multi-scale black-hat / DoG enhancement
-> focus-weighted soft layer separation
-> source_hybrid_soft evidence image
-> paper_clean deghosted image
-> clean_high evidence-constrained repair
-> clean_high_bin final binary output
```

This mode is fully automatic from raw inputs, but it is a simplified
approximation and will not exactly match the legacy tuned result.

Outputs include:

```text
dark.png
enhanced.png
source_hybrid_soft.png
paper_clean.png
clean_high.png
clean_high_bin.png
support_mask.png
added_audit.png
```

## Final-Stage-Only Runner

If `paper_clean.png` and `source_hybrid_soft.png` already exist, run:

```bash
python run_clean_high_bin.py
```

Expected input:

```text
sample_data/layers/<label>/
├── raw.png
├── paper_clean.png
└── source_hybrid_soft.png
```

## Install

```bash
pip install -r requirements.txt
```

## Test

```bash
pytest
```
