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
    algorithm: str = "mccfr",
    cfr_plus: bool = True,
    capacity: int = TABLE_CAPACITY,
    **kwargs,
) -> str:
    """Train native CFR solver and save policy table to disk."""
    algo_key = algorithm.lower()
    if algo_key not in ("mccfr", "cfr"):
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'mccfr' or 'cfr'.")

    if output_path is not None:
        target_path = Path(output_path)
    else:
        suffix = f"{iterations // 1_000_000}M" if iterations >= 1_000_000 else f"{iterations}"
        target_path = Path(f"models/{algo_key}_checkpoint_{suffix}.bin")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    worker_threads = threads or max(1, os.cpu_count() or 4)
    table = truco_engine.SharedPolicyTable(capacity)

    t0 = time.perf_counter()
    train_fn = truco_engine.train_cfr_parallel if algo_key == "cfr" else truco_engine.train_parallel
    train_fn(table, iterations, threads=worker_threads, cfr_plus=cfr_plus)
    dt = time.perf_counter() - t0

    table.save_to_file(str(target_path))
    speed = iterations / dt if dt > 0 else 0.0
    print(
        f"Trained [{algo_key.upper()}] {iterations:,} deals in {dt:.2f}s ({speed:,.0f} deals/s) | "
        f"Occupied: {table.count_occupied():,} infosets"
    )
    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train native CFR policy models")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["mccfr", "cfr"],
        default="mccfr",
        help="CFR solver algorithm ('mccfr' for External-Sampling MCCFR, 'cfr' for Chance-Sampled CFR)",
    )
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
    parser.add_argument(
        "--cfr-plus",
        dest="cfr_plus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CFR+ non-negative regret floor",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=TABLE_CAPACITY,
        help="Table slot capacity",
    )
    args = parser.parse_args()

    saved_path = train_and_save(
        iterations=args.iterations,
        threads=args.threads,
        output_path=args.output,
        algorithm=args.algorithm,
        cfr_plus=args.cfr_plus,
        capacity=args.capacity,
    )
    print(f"Policy saved to {saved_path}")
