#!/usr/bin/env python3
"""Memory-efficient pairwise showdown tournament between trained CFR models and baselines.

Loads at most 2 agents into RAM per matchup to bound memory consumption under 5 GB,
enabling full round-robin tournaments on standard workstations.
"""

import argparse
import gc
import itertools
import json
import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.arena import play_duplicate_match


def instantiate_agent(spec: str, seed: int = 42) -> Agent:
    """Instantiate agent from either a baseline keyword or a model checkpoint path."""
    if spec == "Heuristic":
        return HeuristicAgent()
    elif spec == "Equity":
        return EquityAgent()
    elif spec == "Random":
        return RandomAgent(seed=seed)
    else:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return NativeCFRAgent.from_checkpoint(path, seed=seed)


def run_pairwise_showdown(
    agent_specs: dict[str, str],
    num_matches: int = 1000,
    seed: int = 42,
) -> dict:
    names = list(agent_specs.keys())
    pairs = list(itertools.combinations(names, 2))

    print("=" * 80)
    print("STARTING PAIRWISE SHOWDOWN TOURNAMENT")
    print(f"Agents ({len(names)}):    {', '.join(names)}")
    print(f"Matchups ({len(pairs)}):   {len(pairs)} pairs x {num_matches} duplicate matches = {len(pairs) * num_matches:,} matches")
    print("Memory Strategy: Lazy loading (max 2 agents in RAM per matchup)")
    print("=" * 80 + "\n")

    stats = {
        name: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "net_points": 0,
            "matches_played": 0,
            "head_to_head": {},
        }
        for name in names
    }

    t0_all = time.perf_counter()
    rng = random.Random(seed)
    env = TrucoHandEnv()

    for idx, (name_a, name_b) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Loading matchup: {name_a} vs {name_b}...")
        t_load = time.perf_counter()
        agent_a = instantiate_agent(agent_specs[name_a], seed=seed + idx)
        agent_b = instantiate_agent(agent_specs[name_b], seed=seed + idx + 100)
        dt_load = time.perf_counter() - t_load

        t_match = time.perf_counter()
        h2h_a = {"wins": 0, "losses": 0, "draws": 0, "net_pts": 0}
        h2h_b = {"wins": 0, "losses": 0, "draws": 0, "net_pts": 0}

        for _ in range(num_matches):
            m_seed = rng.randint(0, 1_000_000_000)
            delta = play_duplicate_match(agent_a, agent_b, env, seed=m_seed)

            if delta > 0:
                stats[name_a]["wins"] += 1
                stats[name_b]["losses"] += 1
                h2h_a["wins"] += 1
                h2h_b["losses"] += 1
            elif delta < 0:
                stats[name_a]["losses"] += 1
                stats[name_b]["wins"] += 1
                h2h_a["losses"] += 1
                h2h_b["wins"] += 1
            else:
                stats[name_a]["draws"] += 1
                stats[name_b]["draws"] += 1
                h2h_a["draws"] += 1
                h2h_b["draws"] += 1

            stats[name_a]["net_points"] += delta
            stats[name_b]["net_points"] -= delta
            h2h_a["net_pts"] += delta
            h2h_b["net_pts"] -= delta

        stats[name_a]["matches_played"] += num_matches
        stats[name_b]["matches_played"] += num_matches
        stats[name_a]["head_to_head"][name_b] = h2h_a
        stats[name_b]["head_to_head"][name_a] = h2h_b

        dt_match = time.perf_counter() - t_match
        speed = num_matches / dt_match if dt_match > 0 else 0
        delta_sign = "+" if h2h_a["net_pts"] > 0 else ""
        print(f"     ✓ Result: {name_a} {delta_sign}{h2h_a['net_pts']} pts vs {name_b} ({speed:.0f} dups/s)")

        # Unload and free RAM
        del agent_a
        del agent_b
        gc.collect()

    total_time = time.perf_counter() - t0_all
    total_dups = len(pairs) * num_matches

    for name in names:
        total = stats[name]["matches_played"]
        stats[name]["win_rate_pct"] = round((stats[name]["wins"] / total * 100) if total > 0 else 0.0, 2)
        stats[name]["avg_delta_pts"] = round(stats[name]["net_points"] / total if total > 0 else 0.0, 3)

    print("\n" + "=" * 80)
    print(f"FINAL SHOWDOWN STANDINGS ({total_dups:,} matches in {total_time:.2f}s | {total_dups/total_time:.0f} dups/s)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Agent':<25} {'Win %':<10} {'Wins':<8} {'Losses':<8} {'Net Pts':<10} {'Avg Delta':<12}")
    print("-" * 80)

    ranked = sorted(stats.items(), key=lambda x: x[1]["avg_delta_pts"], reverse=True)
    for rank, (name, d) in enumerate(ranked, 1):
        print(f"{rank:<5} {name:<25} {d['win_rate_pct']:>6.1f}%   {d['wins']:>6}   {d['losses']:>6}   {d['net_points']:>8}   {d['avg_delta_pts']:>+10.3f}")
    print("=" * 80 + "\n")

    return {
        "num_matches_per_pair": num_matches,
        "total_matches": total_dups,
        "wall_time_sec": total_time,
        "standings": ranked,
        "full_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory-efficient pairwise showdown tournament")
    parser.add_argument("--matches", type=int, default=500, help="Duplicate matches per pair")
    parser.add_argument("--output", type=str, default="research/artifacts/showdown_results.json", help="Output JSON path")
    args = parser.parse_args()

    # Discover available completed models
    available_models: dict[str, str] = {
        "Heuristic": "Heuristic",
        "Equity": "Equity",
        "Random": "Random",
    }

    candidates = [
        ("CFR_100k", "models/vanilla_cfr_100k.bin"),
        ("MCCFR_100M", "models/vanilla_mccfr_100M.bin"),
        ("MCCFR+_100M", "models/mccfr_plus_100M.bin"),
    ]

    for label, path_str in candidates:
        p = Path(path_str)
        if p.is_file():
            available_models[label] = str(p)

    results = run_pairwise_showdown(available_models, num_matches=args.matches)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved full showdown report to {out_file}")


if __name__ == "__main__":
    main()
