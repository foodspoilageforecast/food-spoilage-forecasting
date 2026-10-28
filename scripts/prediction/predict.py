#!/usr/bin/env python3
"""Evaluate a saved fold checkpoint on the corresponding held-out replicate."""
import argparse
from pathlib import Path

import pandas as pd
import torch

import nn_replicate_attention as rca
from run_experiment import save_summaries, validate_contexts
from spoilage_data import DATA_DIR, PROJECT, load_contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "checkpoint_predictions")
    args = parser.parse_args()
    torch.set_num_threads(1)
    contexts = validate_contexts(load_contexts(args.data_dir, set())[0])
    context_map = {(c.meat, c.temp_label): c for c in contexts}
    checkpoints = sorted(args.checkpoint_dir.glob("fold_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints in {args.checkpoint_dir}")
    rows = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["meats"] != rca.MEATS or checkpoint["temperatures"] != rca.TEMPS:
            raise ValueError("Checkpoint category order does not match this implementation.")
        model = rca.RCAGompertz(d_model=checkpoint["d_model"], nhead=checkpoint["nhead"],
                               k_scale=checkpoint["k_scale"], residual_scale=checkpoint["residual_scale"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        rows.extend(rca.predict_heldout(model, context_map, checkpoint["held_out_block"] - 1,
                                       torch.device("cpu")))
    save_summaries(pd.DataFrame(rows).drop(columns="z", errors="ignore"), args.output_dir)
    print(f"Saved {len(rows)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
