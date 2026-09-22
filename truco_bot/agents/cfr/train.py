"""Training pipeline and serialization for CFR agents."""

import argparse
import pickle
from pathlib import Path

from truco_bot.agents.cfr.canonical_solver import CanonicalCFRSolver
from truco_bot.agents.cfr.chance_sampled_solver import ChanceSampledCFRSolver
from truco_bot.agents.cfr.mccfr_solver import ExternalSamplingMCCFRSolver


def train_and_save(
    variant: str = "chance",
    iterations: int = 1000,
    output_path: str | Path | None = None,
    seed: int = 42,
    log_interval: int | None = None,
) -> str:
    """Train a CFR solver and save the resulting policy table to disk."""
    if variant in ("native", "mccfr_native", "rust"):
        import truco_engine

        table = truco_engine.SharedPolicyTable(67_108_864)
        target_path = Path(output_path or f"models/mccfr_checkpoint_{iterations}.bin")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        truco_engine.train_parallel(table, iterations, threads=4, cfr_plus=True)
        table.save_to_file(str(target_path))
        return str(target_path)

    if variant == "chance":
        solver = ChanceSampledCFRSolver(seed=seed)
        default_ext = "pkl"
    elif variant == "canonical":
        solver = CanonicalCFRSolver()
        default_ext = "pkl"
    elif variant in ("mccfr", "mccfr_canonical"):
        solver = ExternalSamplingMCCFRSolver(seed=seed, is_canonical=True, cfr_plus=True)
        default_ext = "db"
    elif variant == "mccfr_chance":
        solver = ExternalSamplingMCCFRSolver(seed=seed, is_canonical=False, cfr_plus=True)
        default_ext = "db"
    else:
        raise ValueError(f"Unknown CFR variant: {variant}")

    try:
        solver.train(iterations=iterations, log_interval=log_interval)
    except TypeError:
        solver.train(iterations=iterations)

    target_path = Path(
        output_path or f"truco_bot/agents/cfr/models/{variant}_{iterations}.{default_ext}"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.suffix in (".db", ".sqlite") and hasattr(solver, "export_to_sqlite"):
        solver.export_to_sqlite(target_path)
    else:
        policy = solver.export_policy()
        with open(target_path, "wb") as f:
            pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)

    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save CFR policy models")
    parser.add_argument(
        "--variant",
        choices=["chance", "canonical", "mccfr", "mccfr_canonical", "mccfr_chance", "native", "mccfr_native"],
        default="mccfr_canonical",
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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Interval for progress logging",
    )
    args = parser.parse_args()

    saved_path = train_and_save(
        variant=args.variant,
        iterations=args.iterations,
        output_path=args.output,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    print(f"Policy saved to {saved_path}")
