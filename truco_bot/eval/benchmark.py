"""Benchmark and tournament module that simulates matches between Agents."""

import argparse
import itertools
import random
from pathlib import Path
from typing import Any

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.agent import VanillaCFRAgent
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.arena import play_duplicate_match


def load_agent(
    agent_name: str,
    checkpoint_path: str | Path | None = None,
    seed: int | None = None,
) -> Agent:
    """Load an agent by name, optionally restoring from a checkpoint."""
    if agent_name == "random":
        return RandomAgent(seed=seed)
    elif agent_name == "heuristic":
        return HeuristicAgent()
    elif agent_name == "equity":
        return EquityAgent()
    elif agent_name.startswith(("cfr_chance", "cfr_canonical")):
        is_canonical = "canonical" in agent_name
        path = checkpoint_path
        if path is None:
            candidate = Path(f"truco_bot/agents/cfr/models/{agent_name}.pkl")
            if candidate.is_file():
                path = candidate
        if path is not None and Path(path).is_file():
            return VanillaCFRAgent.from_checkpoint(path, is_canonical=is_canonical, seed=seed)
        if agent_name in ("cfr_chance", "cfr_canonical"):
            return VanillaCFRAgent(seed=seed, is_canonical=is_canonical)
        raise ValueError(f"Unknown agent or missing checkpoint: {agent_name}")
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_matches(agent_a: Agent, agent_b: Agent, n: int, seed: int = 42) -> float:
    """Run n duplicate matches between two agents and return average score."""
    random.seed(seed)
    env = TrucoHandEnv()
    env.reset(seed=seed)
    results = [
        play_duplicate_match(agent_a, agent_b, env, random.randint(1, n * 100))
        for _ in range(n)
    ]
    return sum(results) / len(results)


def run_tournament(
    agents: dict[str, Agent],
    num_matches: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run a round-robin duplicate match tournament between agents."""
    if len(agents) < 2:
        raise ValueError(f"At least 2 agents required for a tournament, got {len(agents)}")
    if num_matches <= 0:
        raise ValueError(f"num_matches must be greater than 0, got {num_matches}")

    stats: dict[str, dict[str, Any]] = {
        name: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "net_points": 0,
            "avg_delta_points": 0.0,
        }
        for name in agents
    }

    rng = random.Random(seed)
    env = TrucoHandEnv()

    for name_a, name_b in itertools.combinations(agents.keys(), 2):
        agent_a = agents[name_a]
        agent_b = agents[name_b]
        for _ in range(num_matches):
            match_seed = rng.randint(0, 1_000_000_000)
            delta = play_duplicate_match(agent_a, agent_b, env, seed=match_seed)
            if delta > 0:
                stats[name_a]["wins"] += 1
                stats[name_b]["losses"] += 1
            elif delta < 0:
                stats[name_a]["losses"] += 1
                stats[name_b]["wins"] += 1
            else:
                stats[name_a]["draws"] += 1
                stats[name_b]["draws"] += 1

            stats[name_a]["net_points"] += delta
            stats[name_b]["net_points"] -= delta

    total_matches_per_agent = num_matches * (len(agents) - 1)
    for name in agents:
        stats[name]["avg_delta_points"] = float(stats[name]["net_points"] / total_matches_per_agent)
        stats[name]["win_rate"] = float((stats[name]["wins"] / total_matches_per_agent) * 100)

    return stats


def print_leaderboard(results: dict[str, dict[str, Any]]) -> None:
    """Print formatted tournament leaderboard."""
    header = f"{'Agent':<20} | {'Wins':>6} | {'Losses':>6} | {'Draws':>6} | {'Win %':>7} | {'Net Pts':>8} | {'Avg Δ Pts':>10}"
    separator = "-" * len(header)
    print(f"\n{header}")
    print(separator)
    sorted_agents = sorted(
        results.items(),
        key=lambda item: (item[1]["avg_delta_points"], item[1]["wins"]),
        reverse=True,
    )
    for name, s in sorted_agents:
        print(
            f"{name:<20} | {int(s['wins']):>6} | {int(s['losses']):>6} | {int(s['draws']):>6} | "
            f"{float(s.get('win_rate', 0.0)):>6.1f}% | {int(s['net_points']):>8} | {float(s['avg_delta_points']):>10.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark & Tournament evaluation for Truco bots")
    parser.add_argument("--tournament", action="store_true", help="Run round-robin tournament")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="List of agent names for tournament",
    )
    parser.add_argument("--p0", default="random", help="First Agent")
    parser.add_argument("--p1", default="heuristic", help="Second Agent")
    parser.add_argument("-n", "--n", type=int, default=100, help="Number of matches")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    if args.tournament:
        agent_names = args.agents if args.agents else [args.p0, args.p1]
        tournament_agents = {name: load_agent(name, seed=args.seed) for name in agent_names}
        tournament_results = run_tournament(tournament_agents, num_matches=args.n, seed=args.seed)
        print_leaderboard(tournament_results)
    else:
        p0_agent = load_agent(args.p0, seed=args.seed)
        p1_agent = load_agent(args.p1, seed=args.seed)
        print(run_matches(p0_agent, p1_agent, n=args.n, seed=args.seed))