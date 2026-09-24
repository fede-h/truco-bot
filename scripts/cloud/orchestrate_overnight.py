#!/usr/bin/env python3
"""Overnight sequential training orchestrator for CFR and MCCFR models."""

import argparse
import json
import sys
import time
from pathlib import Path

from truco_engine import SharedPolicyTable, train_cfr_parallel, train_parallel

TABLE_CAPACITY = 67_108_864  # 64M slots (~4.29 GB)


def get_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, OSError):
        return 0.0


def train_single_model(
    model_name: str,
    algorithm: str,
    target_deals: int,
    cfr_plus: bool,
    output_path: str,
    threads: int = 32,
    chunk_size: int = 1_000_000,
    checkpoint_interval: int = 25_000_000,
) -> dict:
    """Train a single model sequentially with progressive logging and checkpoints."""
    print("=" * 80)
    print(f"STARTING MODEL: {model_name}")
    print(f"  Algorithm:           {algorithm.upper()} (CFR+ = {cfr_plus})")
    print(f"  Target Deals:        {target_deals:,}")
    print(f"  Worker Threads:      {threads}")
    print(f"  Table Capacity:      {TABLE_CAPACITY:,} slots")
    print(f"  Target Output:       {output_path}")
    print("=" * 80)
    sys.stdout.flush()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    t_alloc = time.perf_counter()
    table = SharedPolicyTable(TABLE_CAPACITY)
    dt_alloc = time.perf_counter() - t_alloc
    print(f"Table allocated in {dt_alloc:.2f}s | Base RSS: {get_rss_mb():.1f} MB\n")

    train_fn = train_cfr_parallel if algorithm.lower() == "cfr" else train_parallel
    total_deals = 0
    t_start = time.perf_counter()
    last_log_time = t_start
    deals_since_log = 0

    print(f"{'Deals Done':>12} | {'Elapsed':>10} | {'Deals/Sec':>11} | {'Nodes Found':>12} | {'Load %':>7} | {'RSS (MB)':>9}")
    print("-" * 75)
    sys.stdout.flush()

    while total_deals < target_deals:
        current_chunk = min(chunk_size, target_deals - total_deals)
        train_fn(table, current_chunk, threads, cfr_plus)
        total_deals += current_chunk
        deals_since_log += current_chunk

        now = time.perf_counter()
        dt_chunk = now - last_log_time
        deals_sec = deals_since_log / dt_chunk if dt_chunk > 0 else 0
        occupied = table.count_occupied()
        load_factor = (occupied / TABLE_CAPACITY) * 100.0
        elapsed = now - t_start

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"{total_deals:>12,} | {elapsed_str:>10} | {deals_sec:>11,.0f} | {occupied:>12,} | {load_factor:>6.2f}% | {get_rss_mb():>8.1f}")
        sys.stdout.flush()

        last_log_time = now
        deals_since_log = 0

        # Intermediate checkpoint
        if checkpoint_interval > 0 and (total_deals % checkpoint_interval == 0 and total_deals < target_deals):
            step_m = total_deals // 1_000_000
            inter_path = output_path.replace(".bin", f"_step_{step_m}M.bin")
            t_s = time.perf_counter()
            saved = table.save_to_file(inter_path)
            print(f"  >>> Periodic checkpoint saved: {inter_path} ({saved:,} nodes in {time.perf_counter() - t_s:.2f}s)")
            sys.stdout.flush()

    # Final save
    t_save_start = time.perf_counter()
    final_nodes = table.save_to_file(output_path)
    save_duration = time.perf_counter() - t_save_start
    total_time = time.perf_counter() - t_start
    avg_speed = total_deals / total_time if total_time > 0 else 0

    print("=" * 80)
    print(f"COMPLETED MODEL: {model_name}")
    print(f"  Total Deals:       {total_deals:,}")
    print(f"  Total Time:        {total_time:.2f}s ({total_time/3600:.2f} hrs)")
    print(f"  Average Speed:     {avg_speed:,.0f} deals/sec")
    print(f"  Unique Infosets:   {final_nodes:,} ({final_nodes/TABLE_CAPACITY*100:.2f}% capacity)")
    print(f"  Output Saved:      {output_path} ({save_duration:.2f}s)")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    meta = {
        "model_name": model_name,
        "algorithm": algorithm,
        "target_deals": target_deals,
        "cfr_plus": cfr_plus,
        "output_path": output_path,
        "threads": threads,
        "wall_time_sec": total_time,
        "avg_deals_sec": avg_speed,
        "unique_infosets": final_nodes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{output_path}.done", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate Overnight Training Sequence")
    parser.add_argument("--threads", type=int, default=32, help="Number of worker threads")
    parser.add_argument("--stage", type=int, default=1, help="Stage to start from (1..4)")
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        {
            "stage": 1,
            "name": "vanilla_cfr_100k",
            "algorithm": "cfr",
            "deals": 100_000,
            "cfr_plus": False,
            "output": "models/vanilla_cfr_100k.bin",
            "chunk_size": 2_000,
            "checkpoint_interval": 25_000,
        },
        {
            "stage": 2,
            "name": "vanilla_mccfr_100M",
            "algorithm": "mccfr",
            "deals": 100_000_000,
            "cfr_plus": False,
            "output": "models/vanilla_mccfr_100M.bin",
            "chunk_size": 2_000_000,
            "checkpoint_interval": 25_000_000,
        },
        {
            "stage": 3,
            "name": "mccfr_plus_100M",
            "algorithm": "mccfr",
            "deals": 100_000_000,
            "cfr_plus": True,
            "output": "models/mccfr_plus_100M.bin",
            "chunk_size": 2_000_000,
            "checkpoint_interval": 25_000_000,
        },
        {
            "stage": 4,
            "name": "mccfr_plus_1000M",
            "algorithm": "mccfr",
            "deals": 1_000_000_000,
            "cfr_plus": True,
            "output": "models/mccfr_plus_1000M.bin",
            "chunk_size": 5_000_000,
            "checkpoint_interval": 100_000_000,
        },
    ]

    t_all_start = time.perf_counter()
    completed_reports = []

    for item in plan:
        if item["stage"] < args.stage:
            print(f"Skipping stage {item['stage']}: {item['name']}")
            continue

        res = train_single_model(
            model_name=item["name"],
            algorithm=item["algorithm"],
            target_deals=item["deals"],
            cfr_plus=item["cfr_plus"],
            output_path=item["output"],
            threads=args.threads,
            chunk_size=item["chunk_size"],
            checkpoint_interval=item["checkpoint_interval"],
        )
        completed_reports.append(res)

    t_total = time.perf_counter() - t_all_start
    print("*" * 80)
    print("ALL SCHEDULED MODELS COMPLETED SUCCESSFULLY")
    print(f"Total Sequence Wall Time: {t_total:.2f}s ({t_total/3600:.2f} hrs)")
    print("*" * 80)

    summary_file = Path("models/training_summary.json")
    with open(summary_file, "w") as f:
        json.dump(completed_reports, f, indent=2)
    print(f"Summary report written to {summary_file}")


if __name__ == "__main__":
    main()
