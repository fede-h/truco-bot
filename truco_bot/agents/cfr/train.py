"""Training pipeline and serialization for CFR agents."""

import argparse
import os
import time
from pathlib import Path

import truco_engine

# fixed 64M slots (about 4.86 GB RAM)
TABLE_CAPACITY = 67_108_864


def train_and_save(
    iterations: int = 25_000_000,
    output_path: str | Path | None = None,
    threads: int | None = None,
    **kwargs,
) -> str:
    """Train native CFR solver and save policy table to disk."""
    target_path = Path(output_path or f"models/mccfr_checkpoint_{iterations // 1_000_000}M.bin")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    worker_threads = threads or max(1, os.cpu_count() or 4)
    table = truco_engine.SharedPolicyTable(TABLE_CAPACITY)

    t0 = time.perf_counter()
    truco_engine.train_parallel(table, iterations, threads=worker_threads, cfr_plus=True)
    dt = time.perf_counter() - t0

    table.save_to_file(str(target_path))
    speed = iterations / dt if dt > 0 else 0.0
    print(
        f"Trained {iterations:,} deals in {dt:.2f}s ({speed:,.0f} deals/s) | "
        f"Occupied: {table.count_occupied():,} infosets"
    )
    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train native CFR policy models")
    parser.add_argument(
        "--iterations",
        type=int,
        default=25_000_000,
        help="Total training iterations",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Worker threads (defaults to CPU count)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    args = parser.parse_args()

    saved_path = train_and_save(
        iterations=args.iterations,
        threads=args.threads,
        output_path=args.output,
    )
    print(f"Policy saved to {saved_path}")
