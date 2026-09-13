"""Unit and property tests for CFRNode (regret matching, clipping, strategy accumulation)."""

import math

from truco_bot.agents.cfr.node import CFRNode

from truco_bot.core.actions import Action


def test_cfr_node_initialization():
    """Verify CFRNode initializes with legal actions and clean regret/strategy tables."""
    actions = [Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO]
    node = CFRNode(actions=actions)

    assert node.actions == actions or tuple(node.actions) == tuple(actions)
    # Check that regret_sum and strategy_sum are mappings
    assert isinstance(node.regret_sum, dict)
    assert isinstance(node.strategy_sum, dict)


def test_cfr_node_uniform_fallback_on_initial_call():
    """When newly initialized (all regrets zero), strategy must be uniform."""
    actions = [Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2]
    node = CFRNode(actions=actions)

    strategy = node.get_strategy(realization_weight=1.0)
    expected_prob = 1.0 / len(actions)

    for action in actions:
        assert math.isclose(strategy[action], expected_prob, rel_tol=1e-6)
    assert math.isclose(sum(strategy.values()), 1.0, rel_tol=1e-6)


def test_cfr_node_uniform_fallback_when_all_regrets_negative_or_zero():
    """When all cumulative regrets are <= 0, positive regret sum is 0 -> uniform fallback."""
    actions = [Action.ENVIDO, Action.REAL_ENVIDO]
    node = CFRNode(actions=actions)

    node.regret_sum[Action.ENVIDO] = -12.5
    node.regret_sum[Action.REAL_ENVIDO] = -3.0

    strategy = node.get_strategy(realization_weight=1.0)
    assert math.isclose(strategy[Action.ENVIDO], 0.5, rel_tol=1e-6)
    assert math.isclose(strategy[Action.REAL_ENVIDO], 0.5, rel_tol=1e-6)
    assert math.isclose(sum(strategy.values()), 1.0, rel_tol=1e-6)


def test_cfr_node_positive_regret_clipping():
    """Negative regrets must be clipped to zero (floored at 0) and not drag down positive regrets."""
    actions = [Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO]
    node = CFRNode(actions=actions)

    node.regret_sum[Action.QUIERO_TRUCO] = 8.0
    node.regret_sum[Action.NO_QUIERO_TRUCO] = -10.0

    strategy = node.get_strategy(realization_weight=1.0)
    # Positive regret for QUIERO_TRUCO is 8.0, for NO_QUIERO_TRUCO is 0.0
    assert math.isclose(strategy[Action.QUIERO_TRUCO], 1.0, rel_tol=1e-6)
    assert math.isclose(strategy[Action.NO_QUIERO_TRUCO], 0.0, abs_tol=1e-6)
    assert math.isclose(sum(strategy.values()), 1.0, rel_tol=1e-6)


def test_cfr_node_regret_matching_proportional():
    """Strategy probabilities must be strictly proportional to positive regrets."""
    actions = [Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2]
    node = CFRNode(actions=actions)

    node.regret_sum[Action.PLAY_CARD_0] = 30.0
    node.regret_sum[Action.PLAY_CARD_1] = 10.0
    node.regret_sum[Action.PLAY_CARD_2] = 0.0

    strategy = node.get_strategy(realization_weight=1.0)
    # Total positive = 40.0
    assert math.isclose(strategy[Action.PLAY_CARD_0], 30.0 / 40.0, rel_tol=1e-6)
    assert math.isclose(strategy[Action.PLAY_CARD_1], 10.0 / 40.0, rel_tol=1e-6)
    assert math.isclose(strategy[Action.PLAY_CARD_2], 0.0, abs_tol=1e-6)
    assert math.isclose(sum(strategy.values()), 1.0, rel_tol=1e-6)


def test_cfr_node_strategy_accumulation_weighted_by_realization_weight():
    """Strategy sum must accumulate the current strategy weighted by the player's reach probability."""
    actions = [Action.TRUCO, Action.IR_AL_MAZO]
    node = CFRNode(actions=actions)

    # Step 1: Initial call with realization_weight = 2.0 (uniform 0.5, 0.5)
    s1 = node.get_strategy(realization_weight=2.0)
    assert math.isclose(s1[Action.TRUCO], 0.5, rel_tol=1e-6)
    assert math.isclose(node.strategy_sum[Action.TRUCO], 1.0, rel_tol=1e-6)
    assert math.isclose(node.strategy_sum[Action.IR_AL_MAZO], 1.0, rel_tol=1e-6)

    # Step 2: Set regrets -> 75% TRUCO, 25% IR_AL_MAZO with realization_weight = 4.0
    node.regret_sum[Action.TRUCO] = 3.0
    node.regret_sum[Action.IR_AL_MAZO] = 1.0
    s2 = node.get_strategy(realization_weight=4.0)
    assert math.isclose(s2[Action.TRUCO], 0.75, rel_tol=1e-6)
    assert math.isclose(s2[Action.IR_AL_MAZO], 0.25, rel_tol=1e-6)

    # Accumulated:
    # TRUCO: 1.0 + 4.0 * 0.75 = 1.0 + 3.0 = 4.0
    # IR_AL_MAZO: 1.0 + 4.0 * 0.25 = 1.0 + 1.0 = 2.0
    assert math.isclose(node.strategy_sum[Action.TRUCO], 4.0, rel_tol=1e-6)
    assert math.isclose(node.strategy_sum[Action.IR_AL_MAZO], 2.0, rel_tol=1e-6)


def test_cfr_node_average_strategy_calculation():
    """get_average_strategy() must normalize strategy_sum to a valid probability distribution."""
    actions = [Action.QUIERO_ENVIDO, Action.NO_QUIERO_ENVIDO]
    node = CFRNode(actions=actions)

    node.strategy_sum[Action.QUIERO_ENVIDO] = 4.0
    node.strategy_sum[Action.NO_QUIERO_ENVIDO] = 2.0

    avg_strat = node.get_average_strategy()
    assert math.isclose(avg_strat[Action.QUIERO_ENVIDO], 4.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(avg_strat[Action.NO_QUIERO_ENVIDO], 2.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(sum(avg_strat.values()), 1.0, rel_tol=1e-6)


def test_cfr_node_average_strategy_zero_sum_fallback():
    """If strategy_sum is all zeros, get_average_strategy() must fall back to uniform distribution without dividing by zero."""
    actions = [Action.TRUCO, Action.RETRUCO, Action.VALE_CUATRO]
    node = CFRNode(actions=actions)

    avg_strat = node.get_average_strategy()
    for a in actions:
        assert math.isclose(avg_strat[a], 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(sum(avg_strat.values()), 1.0, rel_tol=1e-6)


def test_cfr_node_has_slots():
    """Verify CFRNode utilizes __slots__ for memory compactness in large game trees."""
    actions = [Action.QUIERO_TRUCO]
    node = CFRNode(actions=actions)
    assert hasattr(node, "__slots__") or hasattr(type(node), "__slots__")
