"""Run duplicate tournament arena across all trained CFR models and baselines."""

import json
import time
from pathlib import Path

from truco_bot.eval.benchmark import load_agent, run_tournament


def main() -> None:
    print("=" * 70)
    print("DUPLICATE ARENA TOURNAMENT BENCHMARK (1,000 Matches/Pair)")
    print("=" * 70)

    agent_names = [
        "random",
        "heuristic",
        "equity",
        "cfr_canonical_20",
        "cfr_canonical_1000",
        "mccfr_canonical_10k",
    ]

    agents = {}
    for name in agent_names:
        t0 = time.perf_counter()
        agents[name] = load_agent(name, seed=42)
        dt = time.perf_counter() - t0
        print(f"Loaded {name:<22} in {dt:.4f}s")

    num_matches = 1000
    total_matchups = len(agent_names) * (len(agent_names) - 1) // 2
    total_dups = total_matchups * num_matches
    print(f"\nRunning {total_matchups} pairings x {num_matches} duplicate matches = {total_dups:,} games...")

    t_tourn = time.perf_counter()
    stats = run_tournament(agents, num_matches=num_matches, seed=42)
    dt_tourn = time.perf_counter() - t_tourn
    print(f"Tournament finished in {dt_tourn:.2f}s ({total_dups / dt_tourn:.1f} duplicate matches/sec)\n")

    out_path = Path("research/artifacts/phase3_tournament_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("=" * 70)
    print(f"{'Rank':<5} {'Agent':<25} {'Win %':<10} {'Wins':<8} {'Losses':<8} {'Net Pts':<10} {'Avg Delta':<12}")
    print("-" * 70)
    sorted_agents = sorted(
        stats.items(),
        key=lambda item: item[1]["avg_delta_points"],
        reverse=True,
    )
    for rank, (name, d) in enumerate(sorted_agents, 1):
        total = d["wins"] + d["losses"] + d["draws"]
        win_pct = (d["wins"] / total) * 100 if total > 0 else 0.0
        print(
            f"{rank:<5} {name:<25} {win_pct:>6.1f}%   {d['wins']:>6}   {d['losses']:>6}   {d['net_points']:>8}   {d['avg_delta_points']:>10.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
