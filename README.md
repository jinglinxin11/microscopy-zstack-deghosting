# Microscopy Pattern Matching

This project performs label-free, independent matching between target
microscopy images and structural reference images. It registers each reference
to a target, retains only target evidence inside the selected corridor, and
exports both a natural-background presentation image and a non-fabricating
binary mask.

## Input Layout

```text
data/input/
  target_images/       # Four target JPG or PNG images, ordered by filename
  reference_images/    # Four reference PNG images; filenames supply labels
```

The committed targets are byte-identical copies of the four main-directory
JPEG images. The reference PNGs are byte-identical copies of the verified
auxiliary images.

## Run

```powershell
python -m pip install -r requirements.txt
python -B run_matching.py --targets data\input\target_images --references data\input\reference_images --outdir artifacts\matching_results
```

The generated directory contains one `results.json`, four presentation PNGs,
and four matched-only binary PNGs. Generated artifacts are intentionally not
tracked by Git.

## Test

```powershell
python -B -m pytest -q -p no:cacheprovider tests\test_scale_calibration.py tests\test_topology_metrics.py tests\test_registration.py tests\test_evidence_mask.py tests\test_pipeline.py
```
