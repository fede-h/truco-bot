#!/usr/bin/env python3
"""Generate academic-grade research figures using Seaborn for Truco CFR convergence and benchmarks."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path("research/artifacts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ponytail: minimal high-level seaborn theme configuration
sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.05)


def plot_infoset_discovery() -> Path:
    """Plot Infoset Discovery / State Space Saturation Curve."""
    deals = [0.1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nodes_mccfr = [
        3.88, 11.58, 12.79, 13.44, 13.91, 14.25, 14.52, 14.76, 14.95, 15.13, 15.28,
        15.84, 16.26, 16.56, 16.78, 16.95, 17.02, 17.028, 17.028,
    ]
    nodes_plus = [
        3.88, 12.43, 13.90, 14.68, 15.18, 15.55, 15.84, 16.07, 16.26, 16.42, 16.56,
        17.02, 17.29, 17.46, 17.58, 17.67, 17.73, 17.79, 17.83,
    ]

    # ponytail: one tidy dataframe for seaborn lineplot
    df = pd.DataFrame({
        "Deals (M)": deals * 2,
        "Unique Infosets (M)": nodes_mccfr + nodes_plus,
        "Algorithm": ["Vanilla MCCFR"] * len(deals) + ["MCCFR+ (Deep Eq)"] * len(deals),
    })

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=df,
        x="Deals (M)",
        y="Unique Infosets (M)",
        hue="Algorithm",
        style="Algorithm",
        markers=True,
        dashes=False,
        linewidth=2.2,
        ax=ax,
        palette="deep",
    )
    ax.axhline(17.83, color="gray", linestyle=":", label="100M Empirical Frontier (~17.83M)")
    ax.scatter([1000], [18.50], color="#2b5c8f", s=80, zorder=5, label="1B Ultra-Run (18.50M)")
    ax.annotate(
        "1 Billion deals: 18.50M nodes",
        xy=(100, 17.83),
        xytext=(55, 14.2),
        arrowprops={"arrowstyle": "->", "color": "#2b5c8f", "lw": 1.2},
        fontsize=9.5,
        fontweight="semibold",
    )

    ax.set_title("Truco Game-Tree State Space Discovery & Saturation", fontweight="bold", pad=12)
    ax.set_ylim(0, 20)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig1_infoset_discovery_seaborn.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_algorithm_throughput() -> Path:
    """Plot Throughput Benchmark Comparison across 32 EPYC Cores."""
    df = pd.DataFrame({
        "Algorithm": [
            "Chance-Sampled CFR\n(Full Tree Traverse)",
            "MCCFR+\n(Lock-Free CAS Truncation)",
            "Vanilla MCCFR\n(External Sampling)",
        ],
        "Deals/sec": [108, 69676, 76483],
    })

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # ponytail: seaborn barplot with automatic palette
    sns.barplot(data=df, y="Algorithm", x="Deals/sec", ax=ax, hue="Algorithm", palette="mako", legend=False)

    for i, (_, row) in enumerate(df.iterrows()):
        val = int(row["Deals/sec"])
        label = f"{val:,} deals/s" + (f" (~{val // 108}x speedup)" if val > 1000 else "")
        ax.text(val + 1500, i, label, va="center", fontweight="bold" if val > 1000 else "normal")

    ax.set_title("Parallel Training Throughput (32 AMD EPYC 7763 vCPUs)", fontweight="bold", pad=12)
    ax.set_xlim(0, 95000)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig2_throughput_comparison_seaborn.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_baseline_performance() -> Path:
    """Plot Net Delta Points of Trained Models vs Baseline Zoo."""
    df = pd.DataFrame({
        "Model": [
            "RandomAgent\n(Uniform)",
            "Vanilla CFR\n(100k)",
            "Vanilla MCCFR\n(100M)",
            "MCCFR+\n(100M)",
            "MCCFR+\n(1000M / 1B)",
        ],
        "Delta Points": [-1.676, -0.424, -1.009, -1.005, -1.079],
    })

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=df, x="Model", y="Delta Points", ax=ax, hue="Model", palette="vlag", legend=False)

    ax.axhline(0, color="black", linewidth=1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        val = row["Delta Points"]
        ax.text(i, val - 0.12, f"{val:+.3f} pts", ha="center", va="top", fontweight="bold")

    ax.set_title("Duplicate Tournament Performance vs Zoo (6,000 matches/model)", fontweight="bold", pad=12)
    ax.set_ylabel("Average Delta Points per Hand")
    ax.set_ylim(-2.2, 0.4)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig3_baseline_delta_seaborn.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    print("Generating Seaborn research figures...")
    p1 = plot_infoset_discovery()
    print(f"✓ Saved Figure 1: {p1}")
    p2 = plot_algorithm_throughput()
    print(f"✓ Saved Figure 2: {p2}")
    p3 = plot_baseline_performance()
    print(f"✓ Saved Figure 3: {p3}")


if __name__ == "__main__":
    main()
