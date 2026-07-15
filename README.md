# Minimal SZTU Microscopy Matching

This branch contains the current minimal SZTU workflow only. It matches each
target image independently against the supplied S/T/U/Z auxiliary images,
renders target evidence inside the selected registered corridor, and exports a
non-fabricating binary result.

## Run

```powershell
python -m pip install -r requirements.txt
python -B run_sztu.py --input output\saved_second_row_target_and_auxiliary_SZTU --outdir output\minimal_refactor_SZTU
```

The output contains exactly one `results.json`, four presentation PNGs, and
four native matched-only binary PNGs. The binary foreground is always target
evidence inside the selected auxiliary corridor; it never copies template
pixels or completes missing strokes.

## Test

```powershell
python -B -m pytest -q -p no:cacheprovider tests\test_sztu_physical_calibration.py tests\test_sztu_topology_match.py tests\test_sztu_unified_registration.py tests\test_sztu_matched_binary.py tests\test_sztu_pipeline.py
```

The committed example result selects `S`, `T`, `U`, and `Z` for targets 01 to
04 respectively.
