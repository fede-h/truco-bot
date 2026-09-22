"""Training pipeline and serialization for CFR agents."""

import argparse
from pathlib import Path

import truco_engine


def train_and_save(
    iterations: int = 1000,
    output_path: str | Path | None = None,
    threads: int = 4,
    capacity: int = 67_108_864,
    variant: str = "native",
    **kwargs,
) -> str:
    """Train native CFR solver and save the resulting policy table to disk."""
    table = truco_engine.SharedPolicyTable(capacity)
    target_path = Path(output_path or f"models/mccfr_checkpoint_{iterations}.bin")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    truco_engine.train_parallel(table, iterations, threads=threads, cfr_plus=True)
    table.save_to_file(str(target_path))
    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save CFR policy models")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for the checkpoint",
    )
    args = parser.parse_args()

    saved_path = train_and_save(
        iterations=args.iterations,
        output_path=args.output,
    )
    print(f"Policy saved to {saved_path}")

