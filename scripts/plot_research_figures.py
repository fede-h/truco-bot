#!/usr/bin/env python3
"""Generate academic-grade research figures for Truco CFR convergence and benchmarks."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("research/artifacts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})


def plot_infoset_discovery() -> Path:
    """Plot Infoset Discovery / State Space Saturation Curve."""
    # Data gathered from live orchestrator logs for Stage 1, 2, and 3
    deals_m = np.array([0.1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    nodes_mccfr = np.array([
        3.88, 11.58, 12.79, 13.44, 13.91, 14.25, 14.52, 14.76, 14.95, 15.13, 15.28,
        15.84, 16.26, 16.56, 16.78, 16.95, 17.02, 17.028, 17.028
    ])
    nodes_mccfr_plus = np.array([
        3.88, 12.43, 13.90, 14.68, 15.18, 15.55, 15.84, 16.07, 16.26, 16.42, 16.56,
        17.02, 17.29, 17.46, 17.58, 17.67, 17.73, 17.79, 17.83
    ])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(deals_m, nodes_mccfr_plus, "o-", color="#1f77b4", linewidth=2, label="MCCFR+ (Deep Eq, CFR+ True)")
    ax.plot(deals_m, nodes_mccfr, "s--", color="#ff7f0e", linewidth=1.8, label="Vanilla MCCFR (CFR+ False)")
    ax.axhline(17.83, color="gray", linestyle=":", label="Empirical Asymptote (~17.83M infosets)")

    ax.set_title("Truco Game-Tree State Space Discovery & Saturation", fontweight="bold", pad=12)
    ax.set_xlabel("Deals Trained (Millions)")
    ax.set_ylabel("Unique Infosets Discovered (Millions)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 20)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig1_infoset_discovery.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_algorithm_throughput() -> Path:
    """Plot Throughput Benchmark Comparison across 32 EPYC Cores."""
    algorithms = [
        "Chance-Sampled CFR\n(Full Tree Traverse)",
        "MCCFR+\n(Lock-Free CAS Truncation)",
        "Vanilla MCCFR\n(External Sampling)",
    ]
    throughputs = [108, 69676, 76483]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(algorithms, throughputs, color=colors, height=0.55, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, throughputs):
        ax.text(
            val + 1500 if val > 1000 else val + 1500,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} deals/s" + (f" (~{val // 108}x speedup)" if val > 1000 else ""),
            va="center",
            ha="left",
            fontweight="bold" if val > 1000 else "normal",
        )

    ax.set_title("Parallel Training Throughput (32 AMD EPYC 7763 vCPUs)", fontweight="bold", pad=12)
    ax.set_xlabel("Deals per Second")
    ax.set_xlim(0, 95000)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig2_throughput_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_baseline_performance() -> Path:
    """Plot Net Delta Points of Trained Models vs Baseline Zoo."""
    # Data from duplicate tournament evaluations
    models = ["RandomAgent\n(Uniform)", "Vanilla CFR\n(100k deals)", "Vanilla MCCFR\n(100M deals)"]
    net_deltas = [-1.676, -0.424, -1.009]  # vs compound zoo

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#d62728", "#ff7f0e", "#1f77b4"]
    bars = ax.bar(models, net_deltas, color=colors, width=0.45, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1.0)
    for bar, val in zip(bars, net_deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.12 if val < 0 else val + 0.05,
            f"{val:+.3f} pts/hand",
            ha="center",
            va="top" if val < 0 else "bottom",
            fontweight="bold",
        )

    ax.set_title("Tournament Performance vs Zoo (Duplicate Matches)", fontweight="bold", pad=12)
    ax.set_ylabel("Average Delta Points per Hand")
    ax.set_ylim(-2.2, 0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig3_baseline_delta.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    print("Generating research figures...")
    p1 = plot_infoset_discovery()
    print(f"✓ Saved Figure 1: {p1}")
    p2 = plot_algorithm_throughput()
    print(f"✓ Saved Figure 2: {p2}")
    p3 = plot_baseline_performance()
    print(f"✓ Saved Figure 3: {p3}")


if __name__ == "__main__":
    main()
