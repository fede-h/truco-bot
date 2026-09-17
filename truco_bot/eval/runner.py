"""CLI runner for declarative showdown experiments"""

import argparse
from pathlib import Path

from truco_bot.eval.config import ShowdownConfig
from truco_bot.eval.showdown import ShowdownRunner
from truco_bot.eval.tracking import MLflowTracker


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Run declarative Truco agent showdowns.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--tracking-uri", type=str, default="sqlite:///mlruns.db", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment", type=str, default="showdown", help="MLflow experiment name"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without sending events to MLflow server"
    )
    args = parser.parse_args()

    config = ShowdownConfig.from_yaml(args.config)
    runner = ShowdownRunner()
    results = runner.run_showdown(config)

    run_name = f"{config.candidate}_vs_{len(config.opponents)}_opponents"
    with MLflowTracker(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        dry_run=args.dry_run,
    ) as tracker:
        tracker.log_showdown(results, run_name=run_name, config=config)

    print(f"\n--- Showdown Results for {config.candidate} ---")
    for opp, metrics in results.items():
        print(
            f"{opp:<15} | Win Rate: {metrics['win_rate']:>5.1f}% | Net Pts: {metrics['net_points']:>4} | "
            f"Avg Δ: {metrics['avg_delta']:>6.2f} | Bluff: {metrics['bluff_frequency']:>4.2f} | Latency: {metrics['avg_time_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
