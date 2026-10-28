#!/usr/bin/env python3
"""Draw the RCA-Gompertz computation as a vector schematic."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from spoilage_data import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "figures" / "paper")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 8})
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.set(xlim=(0, 23), ylim=(0, 8))
    axis.axis("off")

    def box(x, y, width, height, label, fill="#eef3f6", edge="#607a8b"):
        axis.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=0.8))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", linespacing=1.5)

    def arrow(start, end, label=None):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                      color="#3b454e", linewidth=0.8))
        if label:
            axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.23,
                      label, ha="center", va="bottom", fontsize=7)

    for y, text, token in [
        (5.4, "Target replicate\n0-24 h, 25 points", "Q"),
        (3.4, "Reference replicates\n0-24 h, 25 points", "K"),
        (1.4, "Reference replicates\n0-36 h, 37 points", "V"),
    ]:
        box(0.2, y, 3.6, 1.25, text)
        box(4.5, y, 2.5, 1.25, "Shared GRU\n1 input, 48 units", "#edf4ef", "#668a70")
        arrow((3.8, y + 0.625), (4.5, y + 0.625))
        arrow((7, y + 0.625), (8.4, 4.0), token)
    axis.text(5.75, 0.75, "One set of encoder weights", ha="center", fontsize=7)
    box(8.4, 2.9, 2.7, 2.2, "Cross-attention\n4 heads × 12 features\nReference mask", "#f1edf6", "#847096")
    box(8.0, 6.4, 3.5, 1.05, "Food + temperature + position\n48-dimensional embeddings")
    axis.text(9.75, 5.75, "Added to Q, K and V", ha="center", fontsize=7)
    arrow((9.75, 6.4), (9.75, 5.1))
    box(11.9, 3.1, 3.1, 1.8, "Query residual\nLayer normalization\nConcatenate with Q")
    arrow((11.1, 4), (11.9, 4))
    box(15.8, 4.6, 3.2, 1.5, "Gompertz heads\nAsymptote A, rate k", "#f7efe7", "#aa7b50")
    box(15.8, 1.9, 3.2, 1.5, "Correction head\nBounded residual", "#f7efe7", "#aa7b50")
    arrow((15, 4), (15.8, 5.35))
    arrow((15, 4), (15.8, 2.65))
    axis.text(14.5, 6.8, "Hour-24 anchor depth", ha="center", fontsize=7)
    arrow((14.5, 6.5), (15.8, 5.8))
    box(20.0, 3.1, 2.8, 1.8, "Clip to 0-3 cm\nCumulative maximum\nForecast 25-36 h", "#edf4ef", "#668a70")
    arrow((19, 5.35), (20, 4.2))
    arrow((19, 2.65), (20, 3.8))
    axis.text(11.5, 7.9, "RCA-Gompertz: replicate-assisted forecasting", ha="center", va="top",
              fontsize=12, weight="bold")
    axis.text(11.5, 0.15,
              "32,739 parameters  |  Five fold-specific models  |  Target future withheld  |  Complete reference trajectories available",
              ha="center", fontsize=8)
    figure.tight_layout(pad=0.5)
    for extension in ["pdf", "png", "svg"]:
        figure.savefig(args.output_dir / f"rca_gompertz_architecture.{extension}",
                       dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Saved architecture schematic to {args.output_dir}")


if __name__ == "__main__":
    main()
