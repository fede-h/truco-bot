#!/usr/bin/env python3
"""Run concurrency stress test and large-scale MCCFR training benchmark on compiled truco_engine."""

import argparse
import os
import sys
import time
from pathlib import Path

import truco_engine
from truco_engine import BitboardState, SharedPolicyTable, train_parallel


def get_rss_mb() -> float:
    """Return resident set size in megabytes."""
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux, maxrss is in kilobytes
        return rusage.ru_maxrss / 1024.0
    except Exception:
        return 0.0


def run_concurrency_stress_test(threads: int = 8) -> bool:
    """Run stress test with concurrent CAS insertions and atomic regret updates."""
    print("=" * 70)
    print(f"1. RUNNING CONCURRENCY STRESS TEST ({threads} threads)")
    print("=" * 70)

    # Use a 4M slot table for stress testing
    test_table = SharedPolicyTable(4_194_304)
    t0 = time.perf_counter()
    train_parallel(test_table, 10_000, threads)
    dt = time.perf_counter() - t0
    occ = test_table.count_occupied()

    print(f"✓ Completed 10,000 deals in {dt:.3f}s ({10_000/dt:,.0f} deals/sec)")
    print(f"✓ Discovered {occ:,} unique canonical nodes without deadlocks or crashes")
    print(f"✓ Current RSS: {get_rss_mb():.1f} MB")
    print()
    return True


def run_scaling_benchmark(
    target_deals: int = 100_000_000,
    table_capacity: int = 67_108_864,
    threads: int = 8,
    chunk_size: int = 1_000_000,
    checkpoint_dir: str = "models",
) -> None:
    """Run progressive MCCFR training up to target_deals with periodic logging and checkpoints."""
    print("=" * 70)
    print(f"2. LARGE-SCALE MCCFR TRAINING BENCHMARK (Target: {target_deals:,} deals)")
    print(f"   Table Capacity: {table_capacity:,} slots ({table_capacity * 64 / 1024**3:.2f} GB)")
    print(f"   Worker Threads: {threads}")
    print(f"   Chunk Size:     {chunk_size:,} deals")
    print("=" * 70)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"Allocating {table_capacity:,} slot SharedPolicyTable in RAM...")
    t_alloc_start = time.perf_counter()
    table = SharedPolicyTable(table_capacity)
    t_alloc = time.perf_counter() - t_alloc_start
    print(f"✓ Table allocated in {t_alloc:.2f}s. Initial RSS: {get_rss_mb():.1f} MB\n")

    total_deals = 0
    t_start = time.perf_counter()
    last_log_time = t_start
    deals_since_log = 0

    print(f"{'Deals Done':>12} | {'Elapsed':>10} | {'Deals/Sec':>11} | {'Nodes Found':>12} | {'Load %':>7} | {'RSS (MB)':>9}")
    print("-" * 75)

    while total_deals < target_deals:
        current_chunk = min(chunk_size, target_deals - total_deals)
        train_parallel(table, current_chunk, threads)
        total_deals += current_chunk
        deals_since_log += current_chunk

        now = time.perf_counter()
        # Log every chunk or at least every 5 seconds
        dt_chunk = now - last_log_time
        deals_sec = deals_since_log / dt_chunk if dt_chunk > 0 else 0
        occupied = table.count_occupied()
        load_factor = (occupied / table_capacity) * 100.0
        elapsed = now - t_start

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"{total_deals:>12,} | {elapsed_str:>10} | {deals_sec:>11,.0f} | {occupied:>12,} | {load_factor:>6.2f}% | {get_rss_mb():>8.1f}")
        sys.stdout.flush()

        last_log_time = now
        deals_since_log = 0

        # Save checkpoint periodically (every 10M deals or at completion)
        if total_deals % 10_000_000 == 0 or total_deals == target_deals:
            ckpt_path = os.path.join(checkpoint_dir, f"mccfr_checkpoint_{total_deals//1_000_000}M.bin")
            t_save_start = time.perf_counter()
            saved_count = table.save_to_file(ckpt_path)
            t_save = time.perf_counter() - t_save_start
            print(f"  >>> Checkpoint saved: {ckpt_path} ({saved_count:,} nodes, {t_save:.2f}s)")
            sys.stdout.flush()

    total_time = time.perf_counter() - t_start
    overall_speed = total_deals / total_time if total_time > 0 else 0
    final_nodes = table.count_occupied()

    print("=" * 75)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Total Deals Processed: {total_deals:,}")
    print(f"Total Wall-Clock Time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
    print(f"Average Throughput:    {overall_speed:,.0f} deals/sec (~{overall_speed*60:,.0f} traversals/sec)")
    print(f"Total Unique Infosets: {final_nodes:,} ({final_nodes/table_capacity*100:.2f}% capacity)")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Run 10^8 MCCFR and Concurrency Stress Test")
    parser.add_argument("--deals", type=int, default=100_000_000, help="Total deals to train (default: 100,000,000)")
    parser.add_argument("--threads", type=int, default=8, help="Number of Rayon worker threads (default: 8)")
    parser.add_argument("--capacity", type=int, default=67_108_864, help="Table slot capacity (default: 67,108,864 = 4.29 GB)")
    parser.add_argument("--chunk", type=int, default=1_000_000, help="Deals per logging chunk (default: 1,000,000)")
    args = parser.parse_args()

    run_concurrency_stress_test(threads=args.threads)
    run_scaling_benchmark(
        target_deals=args.deals,
        table_capacity=args.capacity,
        threads=args.threads,
        chunk_size=args.chunk,
    )


if __name__ == "__main__":
    main()
