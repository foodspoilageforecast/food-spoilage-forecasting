#!/usr/bin/env python3
"""Compare freshly computed forecasts with the recorded manuscript outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spoilage_data import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT / "results" / "reproduced")
    parser.add_argument("--reference-dir", type=Path, default=PROJECT / "results" / "reference")
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    candidate = pd.read_csv(args.candidate_dir / "predictions.csv")
    reference = pd.read_csv(args.reference_dir / "predictions.csv")
    keys = ["method", "meat", "temperature", "fold", "time", "x", "y"]
    combined = candidate.merge(reference, on=keys, how="left", validate="one_to_one",
                               suffixes=("_new", "_reference"), indicator=True)
    if not combined["_merge"].eq("both").all():
        raise ValueError("Some candidate forecast identifiers are missing from the reference.")
    np.testing.assert_allclose(combined.true_depth_new, combined.true_depth_reference,
                               rtol=0, atol=1e-6)
    report = {}
    for method, group in combined.groupby("method"):
        error = group.predicted_depth_new - group.predicted_depth_reference
        report[method] = dict(
            predictions=len(group), held_out_blocks=sorted(int(x) for x in group.fold.unique()),
            mean_absolute_prediction_difference_cm=float(np.abs(error).mean()),
            max_absolute_prediction_difference_cm=float(np.abs(error).max()),
            rmse_prediction_difference_cm=float(np.sqrt(np.square(error).mean())))
    payload = json.dumps(report, indent=2) + "\n"
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
