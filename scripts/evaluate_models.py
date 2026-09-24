#!/usr/bin/env python3
"""Automated evaluation harness for trained CFR binary checkpoints."""

import argparse
import json
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
from truco_bot.eval.benchmark import run_tournament


def evaluate_checkpoint(
    model_path: str | Path,
    num_matches: int = 1000,
    seed: int = 42,
) -> dict:
    """Run duplicate tournament of a checkpoint against baseline zoo."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    model_name = path.stem
    print("=" * 75)
    print(f"EVALUATING MODEL: {model_name} ({path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Tournament: {num_matches} duplicate matches per baseline pair")
    print("=" * 75)

    t0 = time.perf_counter()
    agent = NativeCFRAgent.from_checkpoint(path, seed=seed)
    print(f"✓ Model loaded into RAM in {time.perf_counter() - t0:.2f}s | Occupied infosets: {agent.table.count_occupied():,}")

    opponents: dict[str, Agent] = {
        model_name: agent,
        "heuristic": HeuristicAgent(),
        "equity": EquityAgent(),
        "random": RandomAgent(seed=seed),
    }

    t_tourn = time.perf_counter()
    stats = run_tournament(opponents, num_matches=num_matches, seed=seed)
    dt_tourn = time.perf_counter() - t_tourn
    total_dups = len(opponents) * (len(opponents) - 1) // 2 * num_matches
    print(f"✓ Completed {total_dups:,} duplicate matches in {dt_tourn:.2f}s ({total_dups / dt_tourn:.1f} dups/s)\n")

    print(f"{'Rank':<5} {'Agent':<25} {'Win %':<10} {'Wins':<8} {'Losses':<8} {'Net Pts':<10} {'Avg Delta':<12}")
    print("-" * 75)
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
    print("=" * 75 + "\n")

    return {
        "model_name": model_name,
        "model_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "occupied_infosets": agent.table.count_occupied(),
        "evaluation_time_sec": dt_tourn,
        "num_matches_per_pair": num_matches,
        "results": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CFR model checkpoints against baselines")
    parser.add_argument("--model", type=str, default=None, help="Specific model checkpoint to evaluate")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to scan for checkpoints")
    parser.add_argument("--matches", type=int, default=1000, help="Duplicate matches per pair")
    parser.add_argument("--output", type=str, default="research/artifacts/model_evaluations.json", help="Output results file")
    args = parser.parse_args()

    models_to_eval = []
    if args.model:
        models_to_eval.append(Path(args.model))
    else:
        for ext in (".bin",):
            models_to_eval.extend(sorted(Path(args.models_dir).glob(f"*{ext}")))

    if not models_to_eval:
        print(f"No models found to evaluate in {args.models_dir}")
        return

    all_evals = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in models_to_eval:
        try:
            report = evaluate_checkpoint(p, num_matches=args.matches)
            all_evals.append(report)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"Error evaluating {p}: {e}")

    with open(out_path, "w") as f:
        json.dump(all_evals, f, indent=2)
    print(f"Saved evaluation reports for {len(all_evals)} model(s) to {out_path}")


if __name__ == "__main__":
    main()
