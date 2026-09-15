"""Training pipeline and serialization for CFR agents."""

import argparse
import pickle
from pathlib import Path

from truco_bot.agents.cfr.canonical_solver import CanonicalCFRSolver
from truco_bot.agents.cfr.chance_sampled_solver import ChanceSampledCFRSolver


def train_and_save(
    variant: str = "chance",
    iterations: int = 1000,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> str:
    """Train a CFR solver and save the resulting policy table to disk."""
    if variant == "chance":
        solver = ChanceSampledCFRSolver(seed=seed)
    elif variant == "canonical":
        solver = CanonicalCFRSolver()
    else:
        raise ValueError(f"Unknown CFR variant: {variant}")

    solver.train(iterations=iterations)
    policy = solver.export_policy()

    target_path = Path(output_path or f"truco_bot/agents/cfr/models/cfr_{variant}_{iterations}.pkl")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "wb") as f:
        pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)

    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save CFR policy models")
    parser.add_argument(
        "--variant",
        choices=["chance", "canonical"],
        default="chance",
        help="CFR solver variant",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for ChanceSampled solver",
    )
    args = parser.parse_args()

    saved_path = train_and_save(
        variant=args.variant,
        iterations=args.iterations,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Policy saved to {saved_path}")
