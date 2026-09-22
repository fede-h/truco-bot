"""Tests for Arena round-robin tournament execution and agent loader."""

import math

import pytest

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.eval.benchmark import load_agent, run_tournament


def test_load_agent_baselines() -> None:
    """Verify load_agent returns correct baseline Agent instances."""
    agent_random = load_agent("random")
    assert isinstance(agent_random, Agent)
    assert isinstance(agent_random, RandomAgent)

    agent_heuristic = load_agent("heuristic")
    assert isinstance(agent_heuristic, Agent)
    assert isinstance(agent_heuristic, HeuristicAgent)

    agent_equity = load_agent("equity")
    assert isinstance(agent_equity, Agent)
    assert isinstance(agent_equity, EquityAgent)


def test_load_agent_native() -> None:
    """Verify load_agent instantiates NativeCFRAgent for native variants."""
    agent_native = load_agent("native")
    assert isinstance(agent_native, Agent)
    assert isinstance(agent_native, NativeCFRAgent)


def test_load_agent_unknown() -> None:
    """Verify load_agent raises ValueError on unrecognized agent name."""
    with pytest.raises(ValueError, match="Unknown agent"):
        load_agent("unsupported_agent_name")


def test_run_tournament_two_agents() -> None:
    """Verify run_tournament executes duplicate matches between a pair of agents."""
    agents = {
        "p0": RandomAgent(seed=11),
        "p1": RandomAgent(seed=22),
    }
    num_matches = 2
    results = run_tournament(agents, num_matches=num_matches, seed=42)

    assert set(results.keys()) == {"p0", "p1"}

    for name in ("p0", "p1"):
        stats = results[name]
        assert "wins" in stats
        assert "losses" in stats
        assert "draws" in stats
        assert "net_points" in stats
        assert "avg_delta_points" in stats

        assert isinstance(stats["wins"], int)
        assert isinstance(stats["losses"], int)
        assert isinstance(stats["draws"], int)
        assert isinstance(stats["net_points"], int)
        assert isinstance(stats["avg_delta_points"], float)

        # Total matches played by each agent
        assert stats["wins"] + stats["losses"] + stats["draws"] == num_matches

    # Pairwise symmetry
    p0_stats = results["p0"]
    p1_stats = results["p1"]
    assert p0_stats["wins"] == p1_stats["losses"]
    assert p0_stats["losses"] == p1_stats["wins"]
    assert p0_stats["draws"] == p1_stats["draws"]
    assert p0_stats["net_points"] + p1_stats["net_points"] == 0

    # Zero-sum game conservation
    assert sum(s["net_points"] for s in results.values()) == 0


def test_run_tournament_multi_agent_round_robin() -> None:
    """Verify round-robin duplicate matches between 3 agents with zero-sum conservation."""
    agents = {
        "random": RandomAgent(seed=101),
        "heuristic": HeuristicAgent(),
        "equity": EquityAgent(),
    }
    num_matches = 2
    results = run_tournament(agents, num_matches=num_matches, seed=42)

    assert set(results.keys()) == set(agents.keys())

    # In a 3-agent round-robin with num_matches=2 duplicate matches per pair:
    # Each agent plays (3 - 1) opponents * 2 matches = 4 total matches
    total_expected_matches_per_agent = num_matches * (len(agents) - 1)

    for name, stats in results.items():
        total_games = stats["wins"] + stats["losses"] + stats["draws"]
        assert total_games == total_expected_matches_per_agent, (
            f"Agent {name} played {total_games}, expected {total_expected_matches_per_agent}"
        )

        expected_avg_delta = stats["net_points"] / total_expected_matches_per_agent
        assert math.isclose(stats["avg_delta_points"], expected_avg_delta, rel_tol=1e-5)

    # Zero-sum game conservation: sum of all net points across the tournament must be 0
    total_net_points = sum(stats["net_points"] for stats in results.values())
    assert total_net_points == 0, f"Expected 0 net points, got {total_net_points}"

    # Win-loss conservation: total wins across tournament must equal total losses
    total_wins = sum(stats["wins"] for stats in results.values())
    total_losses = sum(stats["losses"] for stats in results.values())
    assert total_wins == total_losses, f"Wins ({total_wins}) != Losses ({total_losses})"


def test_run_tournament_with_native_cfr_agent() -> None:
    """Verify run_tournament successfully executes matches with NativeCFRAgent."""
    cfr_agent = NativeCFRAgent(capacity=1024, seed=42)

    agents = {
        "cfr": cfr_agent,
        "random": RandomAgent(seed=42),
    }
    results = run_tournament(agents, num_matches=1, seed=42)

    assert "cfr" in results
    assert "random" in results
    assert sum(s["net_points"] for s in results.values()) == 0


def test_run_tournament_deterministic_seed() -> None:
    """Verify tournament results are reproducible given the same seed."""
    agents_run1 = {"p0": RandomAgent(seed=1), "p1": RandomAgent(seed=2)}
    agents_run2 = {"p0": RandomAgent(seed=1), "p1": RandomAgent(seed=2)}

    results1 = run_tournament(agents_run1, num_matches=2, seed=999)
    results2 = run_tournament(agents_run2, num_matches=2, seed=999)

    assert results1 == results2


def test_run_tournament_invalid_inputs() -> None:
    """Verify run_tournament raises ValueError on invalid agent configurations or match counts."""
    with pytest.raises(ValueError, match="At least 2 agents"):
        run_tournament({"single": RandomAgent()}, num_matches=2)

    with pytest.raises(ValueError, match="At least 2 agents"):
        run_tournament({}, num_matches=2)

    with pytest.raises(ValueError, match="num_matches must be greater than 0"):
        run_tournament({"a": RandomAgent(), "b": RandomAgent()}, num_matches=0)

    with pytest.raises(ValueError, match="num_matches must be greater than 0"):
        run_tournament({"a": RandomAgent(), "b": RandomAgent()}, num_matches=-5)
