#!/usr/bin/env python3
"""Continuous Auto-Evaluator & Leaderboard Daemon.

Monitors the `models/` directory for newly completed CFR checkpoints (.done or .bin),
runs full duplicate tournaments against the baseline zoo, exports compact production
artifacts, and maintains an up-to-date LEADERBOARD.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_models import evaluate_checkpoint
from scripts.export_model import export_checkpoint

LEADERBOARD_JSON = Path("research/artifacts/leaderboard.json")
LEADERBOARD_MD = Path("research/artifacts/LEADERBOARD.md")


def load_leaderboard() -> dict:
    if LEADERBOARD_JSON.is_file():
        try:
            with open(LEADERBOARD_JSON, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"models": {}, "last_updated": ""}


def save_leaderboard(data: dict) -> None:
    LEADERBOARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    with open(LEADERBOARD_JSON, "w") as f:
        json.dump(data, f, indent=2)

    # Render Markdown table
    models = list(data["models"].values())
    models.sort(key=lambda m: m.get("avg_delta_points", 0.0), reverse=True)

    lines = [
        "# Truco Bot Model Leaderboard",
        "",
        f"Last updated: **{data['last_updated']}**",
        "",
        "| Rank | Model Name | Format | Nodes | Win Rate vs Zoo | Net Pts | Avg Delta Pts | Raw Size | Export Q8 Size |",
        "| :--- | :--------- | :----- | :---: | :-------------: | :-----: | :-----------: | :------: | :------------: |",
    ]

    for rank, m in enumerate(models, 1):
        name = m["model_name"]
        nodes = f"{m.get('occupied_infosets', 0):,}"
        win_rate = f"{m.get('win_rate_pct', 0.0):.1f}%"
        net_pts = f"{m.get('net_points', 0):+d}"
        avg_delta = f"{m.get('avg_delta_points', 0.0):+.3f}"
        raw_mb = f"{m.get('raw_size_mb', 0.0):.1f} MB"
        q8_mb = f"{m.get('export_q8_mb', 0.0):.1f} MB" if m.get("export_q8_mb") else "N/A"
        lines.append(
            f"| {rank} | **{name}** | Native CFR | {nodes} | {win_rate} | {net_pts} | {avg_delta} | {raw_mb} | {q8_mb} |"
        )

    lines.extend([
        "",
        "### Baseline Zoo Reference",
        "- **HeuristicAgent**: Handcrafted domain rulebook (envido/truco heuristics).",
        "- **EquityAgent**: Monte Carlo rollout equity evaluator.",
        "- **RandomAgent**: Uniform random legal action baseline.",
        "",
    ])

    with open(LEADERBOARD_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✓ Updated leaderboard written to {LEADERBOARD_MD}")


def process_model(model_path: Path, num_matches: int = 1000) -> None:
    board = load_leaderboard()
    model_name = model_path.stem

    print(f"\n[Auto-Evaluator] Processing model: {model_name}...")
    eval_stats = evaluate_checkpoint(model_path, num_matches=num_matches)

    # Extract model stats from results
    agent_res = eval_stats["results"].get(model_name, {})
    total_matches = agent_res.get("wins", 0) + agent_res.get("losses", 0) + agent_res.get("draws", 0)
    win_rate = (agent_res.get("wins", 0) / total_matches * 100) if total_matches > 0 else 0.0

    # Auto-export quantized inference bundle
    export_info = export_checkpoint(model_path, quantize=True, compress=True)

    board["models"][model_name] = {
        "model_name": model_name,
        "model_path": str(model_path),
        "occupied_infosets": eval_stats["occupied_infosets"],
        "win_rate_pct": round(win_rate, 2),
        "net_points": agent_res.get("net_points", 0),
        "avg_delta_points": round(agent_res.get("avg_delta_points", 0.0), 3),
        "raw_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "export_q8_mb": round(export_info["exported_size_bytes"] / (1024 * 1024), 2),
        "export_reduction_pct": export_info["compression_reduction_pct"],
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    save_leaderboard(board)


def watch_models_dir(models_dir: str | Path = "models", interval: int = 20, num_matches: int = 1000) -> None:
    path = Path(models_dir)
    print(f"Starting Auto-Evaluator daemon watching '{path}' (interval: {interval}s)...")
    evaluated: set[str] = set()

    board = load_leaderboard()
    evaluated.update(board["models"].keys())

    while True:
        done_files = list(path.glob("*.done"))
        for done_file in done_files:
            if done_file.name.endswith(".bin.done"):
                target_bin = done_file.with_name(done_file.name[:-5])
            else:
                target_bin = path / f"{done_file.stem}.bin"
            if target_bin.is_file() and target_bin.stem not in evaluated:
                print(f"[Auto-Evaluator] Detected completed run marker: {done_file.name}")
                process_model(target_bin, num_matches=num_matches)
                evaluated.add(target_bin.stem)

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous evaluator and leaderboard manager for CFR checkpoints")
    parser.add_argument("--watch", action="store_true", help="Run as background daemon polling models directory")
    parser.add_argument("--interval", type=int, default=20, help="Polling interval in seconds")
    parser.add_argument("--model", type=str, default=None, help="Process a single model immediately")
    parser.add_argument("--matches", type=int, default=1000, help="Duplicate matches per baseline pair")
    args = parser.parse_args()

    if args.model:
        process_model(Path(args.model), num_matches=args.matches)
    elif args.watch:
        watch_models_dir(interval=args.interval, num_matches=args.matches)
    else:
        # One-shot scan of any unevaluated .bin
        board = load_leaderboard()
        for p in sorted(Path("models").glob("*.bin")):
            if p.stem not in board["models"]:
                process_model(p, num_matches=args.matches)


if __name__ == "__main__":
    main()
