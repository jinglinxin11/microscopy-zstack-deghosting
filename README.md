# Microscopy Z-Stack Deghosting

This repository implements a compact image-processing pipeline for layered
microscopy digit images. It can run from raw RGB layer images to the final
`clean_high_bin` binary output.

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

## Full Pipeline

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

The full pipeline is:

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
