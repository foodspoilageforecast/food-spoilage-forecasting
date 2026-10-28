#!/usr/bin/env python3
"""Train and evaluate leave-one-replicate-out spoilage forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

import baselines
import nn_replicate_attention as rca
from spoilage_data import DATA_DIR, PROJECT, load_contexts


def validate_contexts(contexts):
    selected = [c for c in contexts if c.meat in rca.MEATS and c.temp_label in rca.TEMPS]
    expected = {(meat, temp) for meat in rca.MEATS for temp in rca.TEMPS}
    if {(c.meat, c.temp_label) for c in selected} != expected:
        raise ValueError("The experiment requires all six foods at 4 C and 25 C.")
    for context in selected:
        if context.hours != list(range(37)) or context.tensor.shape != (37, 5, 5, 5):
            raise ValueError(f"Unexpected time/grid/replicate dimensions: {context.meat} {context.temp_label}")
        if context.x_values != list(range(1, 6)) or context.y_values != list(range(1, 6)):
            raise ValueError("Grid coordinates must run from 1 through 5.")
        if not np.isfinite(context.tensor).all() or np.any((context.tensor < 0) | (context.tensor > rca.D_MAX)):
            raise ValueError("Depths must be finite and within 0-3 cm.")
    return selected


def run_baselines(contexts, folds, include_gbt=True):
    rows = []
    for context in contexts:
        tensor = np.asarray(context.tensor, dtype=float)
        hours = np.asarray(context.hours, dtype=float)
        future_hours = hours[25:37]
        for held_out in folds:
            reference_ids = [b for b in range(5) if b != held_out]
            references = tensor[:, :, :, reference_ids]
            observed = tensor[:, :, :, held_out].copy()
            observed[25:] = np.nan
            priors = baselines._train_block_priors(references, hours)
            predictions = {
                "persistence": baselines.m_persistence(observed, hours, future_hours),
                "linear": baselines.m_linear(observed, hours, future_hours),
                "per_cell_gompertz": baselines.m_per_cell_gompertz(observed, hours, future_hours),
                "replicate_informed_gompertz": baselines.m_replicate_informed_gompertz(
                    observed, hours, future_hours, priors),
            }
            if include_gbt:
                predictions["grad_boost"] = baselines.m_gbt(references, observed, hours, future_hours)
            for method, values in predictions.items():
                for horizon in range(12):
                    for x in range(5):
                        for y in range(5):
                            rows.append(dict(
                                meat=context.meat, temperature=context.temp_label,
                                fold=held_out + 1, test_block_id=held_out + 1,
                                train_block_ids="+".join(str(b + 1) for b in reference_ids),
                                method=method, time=float(future_hours[horizon]),
                                horizon=horizon + 1, x=x + 1, y=y + 1,
                                true_depth=float(tensor[25 + horizon, x, y, held_out]),
                                predicted_depth=float(values[horizon, x, y])))
    return pd.DataFrame(rows)


def save_summaries(predictions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    names = ["fold_metrics", "summary_by_method", "summary_by_method_temp",
             "summary_by_meat_temp", "summary_by_horizon", "trustworthy_horizon"]
    for name, frame in zip(names, baselines.aggregate(predictions)):
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "reproduced")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--neural-only", action="store_true")
    parser.add_argument("--no-gbt", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or len(set(args.folds)) != len(args.folds):
        parser.error("Epochs/threads must be positive and folds must be unique.")
    if args.baselines_only and args.neural_only:
        parser.error("Choose at most one of --baselines-only and --neural-only.")
    torch.set_num_threads(args.threads)
    threadpool_limits(limits=args.threads)
    started = time.monotonic()
    contexts, observations = load_contexts(args.data_dir, set())
    contexts = validate_contexts(contexts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations[observations.temp_label.isin(rca.TEMPS)].to_csv(
        args.output_dir / "parsed_observations.csv", index=False)
    folds = [fold - 1 for fold in args.folds]
    parts = []
    if not args.neural_only:
        print("Fitting reference methods", flush=True)
        parts.append(run_baselines(contexts, folds, include_gbt=not args.no_gbt))
    if not args.baselines_only:
        print(f"Training RCA-Gompertz ({args.epochs} epochs per fold, {args.device})", flush=True)
        predictions, model_info = rca.run_all(
            contexts, device_str=args.device, epochs=args.epochs, folds=folds, output_dir=args.output_dir)
        parts.append(predictions.drop(columns="z", errors="ignore"))
        (args.output_dir / "neural_model_info.json").write_text(json.dumps(model_info, indent=2) + "\n")
    predictions = pd.concat(parts, ignore_index=True)
    save_summaries(predictions, args.output_dir)
    versions = {module.__name__: module.__version__ for module in
                [np, pd, scipy, sklearn, torch, matplotlib, openpyxl]}
    metadata = dict(
        python=platform.python_version(), platform=platform.platform(), versions=versions,
        seed=42, epochs=args.epochs, device=args.device, threads=args.threads,
        held_out_blocks=args.folds, query_hours=[0, 24], reference_value_hours=[0, 36],
        forecast_hours=[25, 36], depth_ceiling_cm=3.0,
        reference_trajectories_include_hours_25_to_36=True,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(args.data_dir.glob("*.xlsx"))},
        elapsed_seconds=round(time.monotonic() - started, 3))
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.read_csv(args.output_dir / "summary_by_method_temp.csv").to_string(index=False))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
