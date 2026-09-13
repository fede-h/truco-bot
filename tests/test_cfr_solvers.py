"""Unit and convergence tests for CFR solvers"""

import math

from truco_bot.agents.cfr.canonical_solver import CanonicalCFRSolver
from truco_bot.agents.cfr.chance_sampled_solver import ChanceSampledCFRSolver


def test_chance_sampled_solver_basic_training():
    """Verify ChanceSampledCFRSolver trains and produces valid distributions."""
    solver = ChanceSampledCFRSolver(seed=42)
    solver.train(iterations=2)

    policy = solver.export_policy()
    assert len(policy) > 0

    for strategy in policy.values():
        assert len(strategy) > 0
        total_prob = sum(strategy.values())
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5), f"Strategy doesn't sum to 1: {total_prob}"
        for action, prob in strategy.items():
            assert prob >= 0.0, f"Negative probability found: {action} -> {prob}"


def test_canonical_solver_basic_training():
    """Verify CanonicalCFRSolver trains over canonical partitions."""
    solver = CanonicalCFRSolver()
    solver.train(iterations=2)

    policy = solver.export_policy()
    assert len(policy) > 0

    for strategy in policy.values():
        assert len(strategy) > 0
        total_prob = sum(strategy.values())
        assert math.isclose(total_prob, 1.0, rel_tol=1e-5), f"Strategy doesn't sum to 1: {total_prob}"
        for action, prob in strategy.items():
            assert prob >= 0.0, f"Negative probability found: {action} -> {prob}"


def test_solvers_regret_accumulation():
    """Verify solvers accumulate counterfactual regrets."""
    solver = ChanceSampledCFRSolver(seed=123)
    solver.train(iterations=1)

    has_nonzero_regret = any(
        any(r != 0.0 for r in node.regret_sum.values())
        for node in solver.nodes.values()
    )
    assert has_nonzero_regret
