# Publication Figures

This directory contains the reproducible Python generator for eight
publication-ready figures describing the microscopy pattern-matching workflow.

## Generate the complete export bundle

From the project root:

```powershell
python -B publication_figures\generate_figures.py
```

The default run recomputes all 16 target-candidate registrations and writes:

- editable SVG figures;
- PDF figures with TrueType text;
- 600 dpi LZW-compressed TIFF figures;
- 300 dpi PNG previews;
- CSV source data for scores, calibration and corridor retention;
- manuscript-ready captions and an export QA report.

The full diagnostic rerun normally takes about two minutes on the current
workstation.

## Fast preview

```powershell
python -B publication_figures\generate_figures.py --formats png
```

Render a subset with `--figures`, for example:

```powershell
python -B publication_figures\generate_figures.py --formats svg,png --figures 5,6
```

## Interpretation boundary

The supplied dataset contains one example per post hoc S/T/U/Z code and no
independent manual ground truth. The figures demonstrate algorithm mechanics,
internal candidate ranking and output-integrity constraints. They do not
estimate accuracy, uncertainty or generalization. All four current decisions
remain automatic candidates requiring human review.
