#!/usr/bin/env python3
"""Run match showdowns (15 or 30 point matches) between two agents."""

import argparse
import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.env.match_env import TrucoMatchEnv


def instantiate_agent(spec: str, seed: int = 42) -> Agent:
    if spec.lower() == "heuristic":
        return HeuristicAgent()
    elif spec.lower() == "equity":
        return EquityAgent()
    elif spec.lower() == "random":
        return RandomAgent(seed=seed)
    else:
        path = Path(spec)
        if not path.is_file():
            # If not a file, try native agent directly
            try:
                return NativeCFRAgent(capacity=1024, seed=seed)
            except (RuntimeError, ValueError, OSError):
                raise FileNotFoundError(f"Agent model or spec not found: {spec}") from None
        if path.suffix == ".pt":
            from truco_bot.agents.deep_cfr.agent import DeepCFRAgent

            return DeepCFRAgent.from_checkpoint(path, seed=seed)
        return NativeCFRAgent.from_checkpoint(path, seed=seed)


def play_one_match(
    agent_0: Agent, agent_1: Agent, env: TrucoMatchEnv, seed: int
) -> tuple[int, int, int]:
    """Plays a single full match to completion. Returns (winner, score_0, score_1)."""
    env.reset(seed=seed)
    agent_0.reset()
    agent_1.reset()

    while not env.is_match_done:
        if env.is_hand_done:
            agent_0.reset()
            agent_1.reset()
            env.next_hand()
            continue

        active = env.active_player
        agent = agent_0 if active == 0 else agent_1

        if hasattr(agent, "act_from_state"):
            action = agent.act_from_state(env.hand_state)
        else:
            action = agent.act(env.get_obs(active), env.get_action_mask())

        env.step(action)

    return env.match_winner or 0, env.match_score_p0, env.match_score_p1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 15 or 30 point Truco match showdown.")
    parser.add_argument(
        "--agent0",
        type=str,
        default="Heuristic",
        help="Spec for Player 0 (Random, Heuristic, Equity, path)",
    )
    parser.add_argument(
        "--agent1",
        type=str,
        default="Random",
        help="Spec for Player 1 (Random, Heuristic, Equity, path)",
    )
    parser.add_argument("--matches", type=int, default=50, help="Number of matches to play")
    parser.add_argument(
        "--target-score",
        type=int,
        default=30,
        choices=[15, 30],
        help="Target match score (15 or 30)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = TrucoMatchEnv(target_score=args.target_score)

    agent_0 = instantiate_agent(args.agent0, seed=args.seed)
    agent_1 = instantiate_agent(args.agent1, seed=args.seed + 1)

    wins_p0 = 0
    wins_p1 = 0
    total_score_p0 = 0
    total_score_p1 = 0

    print("=" * 70)
    print(f"TRUCO MATCH SHOWDOWN: {args.agent0} vs {args.agent1}")
    print(f"Target Score: {args.target_score} pts | Matches: {args.matches} | Seed: {args.seed}")
    print("=" * 70)

    t0 = time.perf_counter()
    for m in range(args.matches):
        match_seed = rng.randint(0, 1_000_000_000)
        winner, s0, s1 = play_one_match(agent_0, agent_1, env, seed=match_seed)
        if winner == 0:
            wins_p0 += 1
        else:
            wins_p1 += 1
        total_score_p0 += s0
        total_score_p1 += s1

        if (m + 1) % max(1, args.matches // 10) == 0 or (m + 1) == args.matches:
            pct0 = wins_p0 / (m + 1) * 100
            print(
                f"Match {m + 1:4d}/{args.matches}: {args.agent0} {wins_p0} ({pct0:.1f}%) - {args.agent1} {wins_p1} ({100 - pct0:.1f}%)"
            )

    elapsed = time.perf_counter() - t0
    print("-" * 70)
    print(
        f"Final Result ({args.matches} matches in {elapsed:.2f}s, {args.matches / max(elapsed, 1e-6):.1f} matches/s):"
    )
    print(
        f"  {args.agent0:15s}: {wins_p0:4d} wins ({wins_p0 / args.matches * 100:.1f}%) | Avg Score: {total_score_p0 / args.matches:.2f}"
    )
    print(
        f"  {args.agent1:15s}: {wins_p1:4d} wins ({wins_p1 / args.matches * 100:.1f}%) | Avg Score: {total_score_p1 / args.matches:.2f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
