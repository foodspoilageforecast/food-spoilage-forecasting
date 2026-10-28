#!/usr/bin/env python3
"""Render the numerical AI panels in Fig. 2 and Supplementary Fig. 19."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from render_spoilage_3d import render_block
from spoilage_data import PROJECT

METHODS = {
    "persistence": ("Baseline", "#858585"),
    "linear": ("Linear", "#aaa269"),
    "per_cell_gompertz": ("Gompertz", "#cfabab"),
    "replicate_informed_gompertz": ("RI-Gompertz", "#28a9d5"),
    "replicate_attention": ("RCA-Gompertz", "#009b73"),
    "grad_boost": ("Gradient boosting", "#8b5aa6"),
}
FOODS = ["Beef", "Pork", "Chicken", "Duck", "Fish", "Shrimp"]
MM = 1 / 25.4


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 8, "legend.fontsize": 6, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(figure, output, name):
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def horizon_axis(axis, data, temperature, metric, methods):
    subset = data[data.temperature == temperature]
    for method in methods:
        series = subset[subset.method == method].sort_values("horizon")
        if series.empty:
            continue
        label, colour = METHODS[method]
        axis.plot(series.horizon, series[metric], "-o", label=label, color=colour,
                  markersize=3, linewidth=1.2)
    axis.set_xticks([1, 3, 6, 9, 12])
    axis.set_xlim(0.7, 12.3)
    axis.set_xlabel("Forecast horizon (h)")
    axis.set_ylabel("MAE (cm)" if metric == "mae" else "$R^2$")
    axis.set_title(f"{temperature[:-1]} °C")


def render_horizons(data, output, all_methods):
    methods = list(METHODS) if all_methods else list(METHODS)[:-1]
    tag = "all_methods" if all_methods else "paper"
    figure, axis = plt.subplots(figsize=(92 * MM, 75 * MM))
    horizon_axis(axis, data, "25C", "mae", methods)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, 1.02), columnspacing=1)
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    save(figure, output, f"fig2e_mae_25C_{tag}")
    figure, axes = plt.subplots(1, 3, figsize=(183 * MM, 61 * MM))
    for index, (axis, temperature, metric) in enumerate(
            zip(axes, ["25C", "4C", "4C"], ["r2", "mae", "r2"])):
        horizon_axis(axis, data, temperature, metric, methods)
        axis.text(-0.19, 1.08, "abc"[index], transform=axis.transAxes,
                  fontsize=10, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
                  ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    save(figure, output, f"supplementary_fig19_{tag}")
    data[data.method.isin(methods)].to_csv(output / f"horizon_plot_data_{tag}.csv", index=False)


def render_fields(predictions, output, fold):
    for temperature in ["25C", "4C"]:
        selected = predictions[(predictions.method == "replicate_attention") &
                               (predictions.temperature == temperature) &
                               (predictions.fold == fold) & (predictions.time == 36)]
        if len(selected) != 150:
            raise ValueError(f"Expected 150 RCA-Gompertz rows at hour 36 for fold {fold}, {temperature}.")
        figure = plt.figure(figsize=(183 * MM, 77 * MM))
        for column, food in enumerate(FOODS):
            frame = selected[selected.meat == food]
            for row, value_column in enumerate(["true_depth", "predicted_depth"]):
                axis = figure.add_subplot(2, 6, row * 6 + column + 1, projection="3d")
                field = frame.pivot(index="x", columns="y", values=value_column).sort_index().sort_index(axis=1)
                if field.shape != (5, 5):
                    raise ValueError(f"Incomplete field for {food}.")
                render_block(axis, field.to_numpy(), meat=food, factor=9, show_axes=False)
                if row == 0:
                    axis.set_title("Minced shrimp" if food == "Shrimp" else food, fontsize=7, pad=-5)
        figure.text(0.012, 0.70, "Real", rotation=90, ha="center", va="center", fontsize=8)
        figure.text(0.012, 0.27, "Forecast", rotation=90, ha="center", va="center", fontsize=8)
        figure.suptitle(f"t = 36 h, Temp = {temperature[:-1]} °C, held-out replicate {fold}", fontsize=9, y=0.99)
        figure.subplots_adjust(left=0.04, right=0.995, bottom=0.015, top=0.86, wspace=0.0, hspace=0.0)
        save(figure, output, f"fig2f_depth_blocks_{temperature}_fold{fold}")
        selected.to_csv(output / f"depth_blocks_{temperature}_fold{fold}_data.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT / "results" / "reference")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "figures" / "paper")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--all-methods", action="store_true")
    args = parser.parse_args()
    configure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_horizons(pd.read_csv(args.results_dir / "summary_by_horizon.csv"),
                    args.output_dir, args.all_methods)
    render_fields(pd.read_csv(args.results_dir / "predictions.csv"), args.output_dir, args.fold)
    print(f"Saved PDF, PNG, and plotting data to {args.output_dir}")


if __name__ == "__main__":
    main()
