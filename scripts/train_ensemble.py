"""Train Ensemble MCCFR agent (80% Self-Play / 20% Opponent Pool) and run 21,000 Duplicate Tournament."""

import json
import time
from pathlib import Path

from truco_bot.agents.cfr.ensemble_solver import EnsembleMCCFRSolver
from truco_bot.eval.benchmark import load_agent, run_tournament


def main() -> None:
    print("=" * 70)
    print("PHASE 3.3: ENSEMBLE MCCFR TRAINING & SAFE OPPONENT EXPLOITATION")
    print("=" * 70)

    models_dir = Path("truco_bot/agents/cfr/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("research/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Ensemble MCCFR Solver
    target_iterations = 10_000
    print(f"\n[Stage 1/2] Training EnsembleMCCFRSolver ({target_iterations:,} deals, 80% Self-Play / 20% Pool)...")
    solver = EnsembleMCCFRSolver(seed=42, self_play_ratio=0.8, is_canonical=True, cfr_plus=True)
    
    t0_train = time.perf_counter()
    solver.train(iterations=target_iterations, log_interval=2_500)
    dt_train = time.perf_counter() - t0_train
    speed = target_iterations / dt_train
    num_nodes = len(solver.nodes)
    print(f"-> Trained {target_iterations:,} deals in {dt_train:.1f}s ({speed:.1f} deals/s). Discovered {num_nodes:,} nodes.")

    # Export to SQLite
    db_path = models_dir / "mccfr_ensemble_10k.db"
    print(f"-> Exporting policy to SQLite: {db_path} ...")
    t_exp = time.perf_counter()
    solver.export_to_sqlite(db_path)
    dt_exp = time.perf_counter() - t_exp
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"-> Export complete in {dt_exp:.2f}s ({db_size_mb:.1f} MB)")

    # 2. Duplicate Arena Tournament
    print("\n" + "=" * 70)
    print("STAGE 2: DUPLICATE ARENA TOURNAMENT (1,000 Matches/Pair)")
    print("=" * 70)

    agent_names = [
        "random",
        "heuristic",
        "equity",
        "cfr_canonical_20",
        "cfr_canonical_1000",
        "mccfr_canonical_10k",
        "mccfr_ensemble_10k",
    ]

    agents = {}
    for name in agent_names:
        t_l = time.perf_counter()
        agents[name] = load_agent(name, seed=42)
        dt_l = time.perf_counter() - t_l
        print(f"Loaded {name:<24} in {dt_l:.4f}s")

    num_matches = 1000
    total_pairings = len(agent_names) * (len(agent_names) - 1) // 2
    total_games = total_pairings * num_matches
    print(f"\nRunning {total_pairings} pairings x {num_matches} duplicate matches = {total_games:,} games...")

    t_tourn = time.perf_counter()
    tournament_stats = run_tournament(agents, num_matches=num_matches, seed=42)
    dt_tourn = time.perf_counter() - t_tourn
    print(f"Tournament finished in {dt_tourn:.2f}s ({total_games / dt_tourn:.1f} duplicate matches/sec)\n")

    # Save results
    tournament_path = artifacts_dir / "phase3_3_tournament_results.json"
    with open(tournament_path, "w") as f:
        json.dump(tournament_stats, f, indent=2)
    print(f"Results saved to {tournament_path}")

    # Print Leaderboard
    print("=" * 75)
    print(f"{'Rank':<5} {'Agent':<25} {'Win %':<10} {'Wins':<8} {'Losses':<8} {'Net Pts':<10} {'Avg Delta':<12}")
    print("-" * 75)
    sorted_agents = sorted(
        tournament_stats.items(),
        key=lambda item: item[1]["avg_delta_points"],
        reverse=True,
    )
    for rank, (name, d) in enumerate(sorted_agents, 1):
        total = d["wins"] + d["losses"] + d["draws"]
        win_pct = (d["wins"] / total) * 100 if total > 0 else 0.0
        print(
            f"{rank:<5} {name:<25} {win_pct:>6.1f}%   {d['wins']:>6}   {d['losses']:>6}   {d['net_points']:>8}   {d['avg_delta_points']:>10.3f}"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
