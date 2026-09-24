"""CLI runner for declarative showdown experiments"""

import argparse

from truco_bot.eval.config import ShowdownConfig
from truco_bot.eval.showdown import ShowdownRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run declarative Truco agent showdowns.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = ShowdownConfig.from_yaml(args.config)
    runner = ShowdownRunner()
    results = runner.run_showdown(config)

    print(f"\n--- Showdown Results for {config.candidate} ---")
    for opp, metrics in results.items():
        print(
            f"{opp:<15} | Win Rate: {metrics['win_rate']:>5.1f}% | Net Pts: {metrics['net_points']:>4} | "
            f"Avg Δ: {metrics['avg_delta']:>6.2f} | Bluff: {metrics['bluff_frequency']:>4.2f} | Latency: {metrics['avg_time_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
