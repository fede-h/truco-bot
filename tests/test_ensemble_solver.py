"""Tests for EnsembleMCCFRSolver and SQLite persistence."""

import math
import sqlite3
from pathlib import Path

import pytest

from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.cfr.ensemble_agent import EnsembleCFRAgent
from truco_bot.agents.cfr.ensemble_solver import EnsembleMCCFRSolver
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.arena import _run_single_match, play_duplicate_match


class TestEnsembleMCCFRSolver:
    """Unit and integration tests for EnsembleMCCFRSolver."""

    def test_initialization_defaults(self) -> None:
        """Verify solver initializes with default 0.8 self-play ratio, pool, and empty nodes."""
        solver = EnsembleMCCFRSolver(seed=42)

        assert solver.seed == 42
        assert solver.self_play_ratio == pytest.approx(0.8, rel=1e-5)
        assert hasattr(solver, "opponent_pool")
        assert len(solver.opponent_pool) > 0
        assert isinstance(solver.nodes, dict)
        assert len(solver.nodes) == 0

    def test_initialization_custom_pool_and_ratio(self) -> None:
        """Verify solver accepts custom opponent pool and custom self-play ratio."""
        custom_pool = [HeuristicAgent(), EquityAgent()]
        solver = EnsembleMCCFRSolver(
            seed=100,
            opponent_pool=custom_pool,
            self_play_ratio=0.75,
            is_canonical=True,
            cfr_plus=True,
        )

        assert solver.seed == 100
        assert solver.self_play_ratio == pytest.approx(0.75, rel=1e-5)
        assert solver.opponent_pool == custom_pool
        assert solver.is_canonical is True
        assert solver.cfr_plus is True

    def test_train_invalid_iterations(self) -> None:
        """Verify train raises ValueError when iterations <= 0."""
        solver = EnsembleMCCFRSolver(seed=42)
        with pytest.raises(ValueError, match="iterations must be greater than 0"):
            solver.train(iterations=0)

        with pytest.raises(ValueError, match="iterations must be greater than 0"):
            solver.train(iterations=-1)

    def test_train_discovers_nodes(self) -> None:
        """Verify train(iterations=10) executes without error and populates nodes."""
        solver = EnsembleMCCFRSolver(seed=42, self_play_ratio=0.8)
        solver.train(iterations=10)

        assert len(solver.nodes) > 0
        for node in solver.nodes.values():
            assert len(node.actions) > 0
            assert len(node.regret_sum) > 0
            assert len(node.strategy_sum) > 0

    def test_valid_strategy_distributions(self) -> None:
        """Verify exported policy distributions sum to 1.0 and contain no negative probabilities."""
        solver = EnsembleMCCFRSolver(seed=42, self_play_ratio=0.8)
        solver.train(iterations=10)

        policy = solver.export_policy()
        assert isinstance(policy, dict)
        assert len(policy) > 0

        for key, distribution in policy.items():
            assert isinstance(key, tuple)
            assert isinstance(distribution, dict)
            assert len(distribution) > 0

            total_prob = sum(distribution.values())
            assert math.isclose(total_prob, 1.0, rel_tol=1e-5), (
                f"Distribution at {key} does not sum to 1.0: {total_prob}"
            )

            for action, prob in distribution.items():
                assert isinstance(action, Action)
                assert prob >= 0.0, f"Negative probability found for action {action}: {prob}"

    def test_export_to_sqlite(self, tmp_path: Path) -> None:
        """Verify export_to_sqlite exports policy into SQLite database with correct schema."""
        solver = EnsembleMCCFRSolver(seed=42, self_play_ratio=0.8)
        solver.train(iterations=10)

        db_path = tmp_path / "ensemble_policy.db"
        solver.export_to_sqlite(db_path)

        assert db_path.exists()
        assert db_path.stat().st_size > 0

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='policy'")
        tables = cursor.fetchall()
        assert len(tables) == 1

        cursor.execute("SELECT COUNT(*) FROM policy")
        row_count = cursor.fetchone()[0]
        assert 0 < row_count <= len(solver.nodes)
        conn.close()

    def test_export_to_sqlite_str_path(self, tmp_path: Path) -> None:
        """Verify export_to_sqlite accepts string paths."""
        solver = EnsembleMCCFRSolver(seed=42)
        solver.train(iterations=5)

        db_path_str = str(tmp_path / "ensemble_str_policy.sqlite")
        solver.export_to_sqlite(db_path_str)

        assert Path(db_path_str).exists()

    def test_database_loadable_by_ensemble_agent(self, tmp_path: Path) -> None:
        """Verify exported database is loadable by EnsembleCFRAgent.from_checkpoint and executable."""
        solver = EnsembleMCCFRSolver(seed=42, self_play_ratio=0.8)
        solver.train(iterations=10)

        db_path = tmp_path / "ensemble_agent_checkpoint.db"
        solver.export_to_sqlite(db_path)

        agent = EnsembleCFRAgent.from_checkpoint(db_path, seed=42)
        assert isinstance(agent, EnsembleCFRAgent)
        assert 0 < len(agent.policy) <= len(solver.nodes)

        # Verify agent can act from GameState
        h0 = list(ALL_CARDS[:3])
        h1 = list(ALL_CARDS[3:6])
        state = create_initial_hand_state([h0, h1], mano=0)
        action = agent.act_from_state(state)
        assert isinstance(action, Action)

        # Verify agent plays in Arena without runtime errors
        env = TrucoHandEnv()
        single_res = _run_single_match(agent, HeuristicAgent(), env, seed=42)
        assert isinstance(single_res, int)

        dup_res = play_duplicate_match(agent, HeuristicAgent(), env, seed=42)
        assert isinstance(dup_res, int)
