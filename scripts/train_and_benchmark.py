"""Pipeline script for Phase 3.2: MCCFR Training, Exploitability Tracking, and Tournament Arena."""

import json
import time
from pathlib import Path

from truco_bot.agents.cfr.agent import SQLitePolicy
from truco_bot.agents.cfr.exploitability import compute_exploitability
from truco_bot.agents.cfr.mccfr_solver import ExternalSamplingMCCFRSolver
from truco_bot.eval.benchmark import load_agent, run_tournament


def main() -> None:
    print("=" * 70)
    print("PHASE 3.2: MCCFR & CFR+ SCALED SELF-PLAY AND TOURNAMENT ARENA")
    print("=" * 70)

    models_dir = Path("truco_bot/agents/cfr/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("research/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}

    # Initial baseline exploitability (uninformed / uniform policy)
    print("\n[Stage 1/4] Calculating baseline exploitability (uninformed uniform policy)...")
    t0_exp0 = time.perf_counter()
    exp_0 = compute_exploitability({}, num_deals=20, is_canonical=True, seed=42)
    dt_exp0 = time.perf_counter() - t0_exp0
    print(f"-> Baseline Exploitability (0 deals): {exp_0:.4f} pts/deal (computed in {dt_exp0:.2f}s)")
    metrics["0"] = {
        "iterations": 0,
        "nodes": 0,
        "elapsed_seconds": 0.0,
        "deals_per_sec": 0.0,
        "exploitability": exp_0,
    }

    # Initialize MCCFR Solver
    solver = ExternalSamplingMCCFRSolver(seed=42, is_canonical=True, cfr_plus=True)
    total_train_time = 0.0

    checkpoints = [10_000, 50_000, 100_000]
    current_iter = 0

    for target in checkpoints:
        step_iters = target - current_iter
        print(f"\n[Training] Scaling from {current_iter:,} to {target:,} iterations (+{step_iters:,} deals)...")
        t_start = time.perf_counter()
        
        # Train in chunks with progress logging
        log_int = 2_500 if target <= 10_000 else 5_000
        solver.train(iterations=step_iters, log_interval=log_int)
        
        dt_step = time.perf_counter() - t_start
        total_train_time += dt_step
        current_iter = target
        num_nodes = len(solver.nodes)
        speed = step_iters / dt_step

        print(f"-> Reached {target:,} iterations in {dt_step:.1f}s ({speed:.1f} deals/s). Discovered {num_nodes:,} nodes.")

        # Export to SQLite
        suffix_name = f"{target // 1000}k" if target >= 1000 else str(target)
        db_path = models_dir / f"mccfr_canonical_{suffix_name}.db"
        print(f"-> Exporting policy to SQLite: {db_path} ...")
        t_exp = time.perf_counter()
        solver.export_to_sqlite(db_path)
        dt_export = time.perf_counter() - t_exp
        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"-> Export complete in {dt_export:.2f}s ({db_size_mb:.1f} MB)")

        # Evaluate Exploitability via zero-RAM SQLitePolicy
        print(f"-> Evaluating Exploitability ({target:,} deals) on 20 sample deals...")
        t_eval = time.perf_counter()
        sql_policy = SQLitePolicy(db_path)
        exp_val = compute_exploitability(sql_policy, num_deals=20, is_canonical=True, seed=42)
        sql_policy.close()
        dt_eval = time.perf_counter() - t_eval
        print(f"-> Exploitability @ {target:,} deals: {exp_val:.4f} pts/deal (evaluated in {dt_eval:.2f}s)")

        metrics[suffix_name] = {
            "iterations": target,
            "nodes": num_nodes,
            "elapsed_seconds": total_train_time,
            "deals_per_sec": speed,
            "db_size_mb": db_size_mb,
            "exploitability": exp_val,
        }

    # Save training metrics JSON
    metrics_path = artifacts_dir / "phase3_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining metrics saved to {metrics_path}")

    # Stage 3: Arena Tournament
    print("\n" + "=" * 70)
    print("STAGE 3: DUPLICATE ARENA TOURNAMENT (1,000 Matches/Pair)")
    print("=" * 70)

    agent_names = [
        "random",
        "heuristic",
        "equity",
        "cfr_canonical_20",
        "cfr_canonical_1000",
        "mccfr_canonical_10k",
        "mccfr_canonical_50k",
        "mccfr_canonical_100k",
    ]

    agents = {}
    for name in agent_names:
        print(f"Loading agent: {name} ...")
        t_l = time.perf_counter()
        agents[name] = load_agent(name, seed=42)
        dt_l = time.perf_counter() - t_l
        print(f"  Loaded {name} in {dt_l:.4f}s")

    print("\nRunning tournament across all 8 agents (28 pairings x 1,000 duplicate matches = 28,000 matches)...")
    t_tourn = time.perf_counter()
    tournament_stats = run_tournament(agents, num_matches=1000, seed=42)
    dt_tourn = time.perf_counter() - t_tourn
    print(f"Tournament completed in {dt_tourn:.2f}s ({28_000 / dt_tourn:.1f} matches/s)")

    tournament_path = artifacts_dir / "phase3_tournament_results.json"
    with open(tournament_path, "w") as f:
        json.dump(tournament_stats, f, indent=2)
    print(f"Tournament results saved to {tournament_path}")

    print("\n" + "=" * 70)
    print("TOURNAMENT LEADERBOARD")
    print("=" * 70)
    sorted_agents = sorted(
        tournament_stats.items(),
        key=lambda item: item[1]["avg_delta_points"],
        reverse=True,
    )
    print(f"{'Rank':<5} {'Agent':<25} {'Win %':<10} {'Net Points':<12} {'Avg Delta/Match':<15}")
    print("-" * 70)
    for rank, (name, data) in enumerate(sorted_agents, 1):
        total_games = data["wins"] + data["losses"] + data["draws"]
        win_rate = (data["wins"] / total_games) * 100 if total_games > 0 else 0.0
        print(
            f"{rank:<5} {name:<25} {win_rate:>6.1f}%   {data['net_points']:>10}   {data['avg_delta_points']:>14.3f}"
        )

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()
