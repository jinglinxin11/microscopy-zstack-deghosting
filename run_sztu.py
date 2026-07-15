"""Run the minimal SZTU matching and binary-export workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sztu_pipeline import run_pipeline, write_minimal_output


DEFAULT_INPUT = Path("output/saved_second_row_target_and_auxiliary_SZTU")
DEFAULT_OUTDIR = Path("output/sztu_minimal_result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    run = run_pipeline(args.input)
    payload = write_minimal_output(run, args.outdir)
    print(json.dumps({"outdir": str(args.outdir.resolve()), "results": payload["results"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
