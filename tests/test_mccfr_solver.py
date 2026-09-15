"""Tests for ExternalSamplingMCCFRSolver and SQLite policy persistence."""

import math
import sqlite3
from pathlib import Path

import pytest
from truco_bot.agents.cfr.mccfr_solver import ExternalSamplingMCCFRSolver

from truco_bot.agents.cfr.agent import VanillaCFRAgent
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state


def test_mccfr_solver_initialization() -> None:
    """Verify ExternalSamplingMCCFRSolver initializes with empty nodes and configured seed."""
    solver = ExternalSamplingMCCFRSolver(seed=42, is_canonical=True, cfr_plus=True)
    assert isinstance(solver.nodes, dict)
    assert len(solver.nodes) == 0


def test_mccfr_solver_train_populates_nodes() -> None:
    """Verify train(iterations=10) executes without error and discovers game nodes."""
    solver = ExternalSamplingMCCFRSolver(seed=42, is_canonical=True, cfr_plus=True)
    solver.train(iterations=10)

    assert len(solver.nodes) > 0
    # Every node must have registered actions and strategy_sum
    for node in solver.nodes.values():
        assert len(node.actions) > 0
        assert len(node.regret_sum) > 0
        assert len(node.strategy_sum) > 0


def test_mccfr_solver_export_policy_validity() -> None:
    """Verify export_policy returns a valid mapping with normalized action probability distributions."""
    solver = ExternalSamplingMCCFRSolver(seed=42, is_canonical=True, cfr_plus=True)
    solver.train(iterations=10)

    policy = solver.export_policy()
    assert isinstance(policy, dict)
    assert len(policy) > 0

    for infoset_key, distribution in policy.items():
        assert isinstance(infoset_key, tuple)
        assert isinstance(distribution, dict)
        assert len(distribution) > 0

        total_prob = sum(distribution.values())
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5), f"Distribution doesn't sum to 1: {total_prob}"
        for action, prob in distribution.items():
            assert isinstance(action, Action)
            assert prob >= 0.0, f"Negative probability found for action {action}: {prob}"


def test_mccfr_solver_export_to_sqlite_and_agent_reload(tmp_path: Path) -> None:
    """Verify export_to_sqlite writes a SQLite db reloadable by VanillaCFRAgent.from_checkpoint."""
    solver = ExternalSamplingMCCFRSolver(seed=42, is_canonical=True, cfr_plus=True)
    solver.train(iterations=10)
    exported_policy = solver.export_policy()

    db_path = tmp_path / "mccfr_policy.sqlite"
    solver.export_to_sqlite(db_path)

    assert db_path.exists()
    assert db_path.stat().st_size > 0

    # Verify SQLite schema and records via standard library sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert len(tables) > 0
    conn.close()

    # Load back using VanillaCFRAgent.from_checkpoint with SQLite path
    loaded_agent = VanillaCFRAgent.from_checkpoint(db_path, is_canonical=True, seed=42)
    assert isinstance(loaded_agent, VanillaCFRAgent)
    assert loaded_agent.is_canonical is True
    assert len(loaded_agent.policy) > 0
    assert loaded_agent.policy.keys() == exported_policy.keys()

    # Verify distributions match exported policy
    for key, expected_dist in exported_policy.items():
        loaded_dist = loaded_agent.policy[key]
        for action, prob in expected_dist.items():
            assert math.isclose(loaded_dist[action], prob, rel_tol=1e-5)

    # Verify loaded agent acts in environment without errors
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)
    action = loaded_agent.act_from_state(state)
    assert isinstance(action, Action)


def test_mccfr_solver_export_to_sqlite_str_path(tmp_path: Path) -> None:
    """Verify export_to_sqlite accepts string paths as well as Path objects."""
    solver = ExternalSamplingMCCFRSolver(seed=42, is_canonical=False)
    solver.train(iterations=5)

    db_path_str = str(tmp_path / "str_policy.db")
    solver.export_to_sqlite(db_path_str)

    assert Path(db_path_str).exists()
    loaded_agent = VanillaCFRAgent.from_checkpoint(db_path_str, is_canonical=False)
    assert isinstance(loaded_agent, VanillaCFRAgent)
    assert len(loaded_agent.policy) > 0


def test_mccfr_solver_invalid_iterations() -> None:
    """Verify train raises ValueError when iterations <= 0."""
    solver = ExternalSamplingMCCFRSolver(seed=42)
    with pytest.raises(ValueError, match="iterations must be greater than 0"):
        solver.train(iterations=0)

    with pytest.raises(ValueError, match="iterations must be greater than 0"):
        solver.train(iterations=-1)
