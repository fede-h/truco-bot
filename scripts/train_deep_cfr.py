"""Minimal CLI to distill tabular CFR into Deep CFR PolicyNetwork."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from truco_bot.agents.deep_cfr.distill import distill_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill CFR table into Deep CFR PolicyNetwork")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mccfr_plus_1000M.bin",
        help="Path to binary CFR table checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/deep_cfr_policy.pt",
        help="Path to save distilled PyTorch policy model",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Number of state-action samples to extract",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cpu or cuda)",
    )

    args = parser.parse_args()

    # ponytail: minimal supervised distillation execution
    print(f"Distilling CFR policy from {args.checkpoint} to {args.output}...")
    distill_table(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
