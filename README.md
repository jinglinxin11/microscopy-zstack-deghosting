# Clean High Binary Layer Processor

This project contains a compact, reproducible implementation of the selected
`clean_high_bin` image-processing stage for layered microscopy digit images.

The code does not use digit templates, OCR, labels as priors, or machine
learning. It applies evidence-constrained local repair, mild sharpening, and
binary conversion to the `paper_clean` layer result.

## Project Layout

```text
.
├── run_clean_high_bin.py
├── requirements.txt
├── sample_data/
│   └── layers/
│       ├── 1/
│       ├── 3/
│       ├── 4/
│       └── 7/
├── src/
│   └── clean_high_bin.py
└── tests/
    └── test_clean_high_bin.py
```

Each layer folder must contain:

```text
raw.png
paper_clean.png
source_hybrid_soft.png
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_clean_high_bin.py
```

Or specify your own input and output folders:

```bash
python run_clean_high_bin.py --source sample_data --out-root outputs
```

The output folder contains per-layer results and a contact sheet:

```text
outputs/clean_high_bin_YYYYMMDD_HHMMSS/
├── layers/
│   └── <label>/
│       ├── clean_high.png
│       ├── clean_high_bin.png
│       ├── support_mask.png
│       └── added_audit.png
└── contact_sheets/
    └── clean_high_bin_overview.png
```

## Test

```bash
pytest
```
