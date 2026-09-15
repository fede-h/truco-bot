"""Unit tests for CFRNode under CFR+ and linear MCCFR strategy accumulation."""

import math

from truco_bot.agents.cfr.node import CFRNode
from truco_bot.core.actions import Action


def test_cfr_node_apply_regret_cfr_plus_floors_at_zero() -> None:
    actions = [Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO]
    node = CFRNode(actions=actions)

    node.apply_regret(Action.QUIERO_TRUCO, -10.0, cfr_plus=True)
    assert node.regret_sum[Action.QUIERO_TRUCO] == 0.0

    node.apply_regret(Action.QUIERO_TRUCO, 5.0, cfr_plus=True)
    assert node.regret_sum[Action.QUIERO_TRUCO] == 5.0

    node.apply_regret(Action.QUIERO_TRUCO, -3.0, cfr_plus=True)
    assert node.regret_sum[Action.QUIERO_TRUCO] == 2.0

    node.apply_regret(Action.QUIERO_TRUCO, -10.0, cfr_plus=True)
    assert node.regret_sum[Action.QUIERO_TRUCO] == 0.0


def test_cfr_node_apply_regret_vanilla_allows_negative() -> None:
    actions = [Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO]
    node = CFRNode(actions=actions)

    node.apply_regret(Action.QUIERO_TRUCO, -5.0, cfr_plus=False)
    assert node.regret_sum[Action.QUIERO_TRUCO] == -5.0


def test_cfr_node_linear_strategy_accumulation_weighting() -> None:
    actions = [Action.PLAY_CARD_1, Action.PLAY_CARD_2]
    node = CFRNode(actions=actions)

    node.accumulate_strategy({Action.PLAY_CARD_1: 0.5, Action.PLAY_CARD_2: 0.5}, weight=1.0)
    assert math.isclose(node.strategy_sum[Action.PLAY_CARD_1], 0.5, rel_tol=1e-5)
    assert math.isclose(node.strategy_sum[Action.PLAY_CARD_2], 0.5, rel_tol=1e-5)

    node.regret_sum[Action.PLAY_CARD_1] = 10.0
    node.regret_sum[Action.PLAY_CARD_2] = 0.0

    node.accumulate_strategy({Action.PLAY_CARD_1: 1.0, Action.PLAY_CARD_2: 0.0}, weight=2.0)
    assert math.isclose(node.strategy_sum[Action.PLAY_CARD_1], 2.5, rel_tol=1e-5)
    assert math.isclose(node.strategy_sum[Action.PLAY_CARD_2], 0.5, rel_tol=1e-5)

    avg_strat = node.get_average_strategy()
    assert math.isclose(avg_strat[Action.PLAY_CARD_1], 2.5 / 3.0, rel_tol=1e-5)
    assert math.isclose(avg_strat[Action.PLAY_CARD_2], 0.5 / 3.0, rel_tol=1e-5)


def test_cfr_node_get_average_strategy_distribution_normalized() -> None:
    actions = [Action.ENVIDO, Action.REAL_ENVIDO, Action.FALTA_ENVIDO, Action.NO_QUIERO_ENVIDO]
    node = CFRNode(actions=actions)

    node.strategy_sum[Action.ENVIDO] = 12.0
    node.strategy_sum[Action.REAL_ENVIDO] = 8.0
    node.strategy_sum[Action.FALTA_ENVIDO] = 0.0
    node.strategy_sum[Action.NO_QUIERO_ENVIDO] = 4.0

    avg_strat = node.get_average_strategy()
    assert math.isclose(sum(avg_strat.values()), 1.0, rel_tol=1e-6)
    assert all(prob >= 0.0 for prob in avg_strat.values())
    assert math.isclose(avg_strat[Action.ENVIDO], 12.0 / 24.0, rel_tol=1e-6)
    assert math.isclose(avg_strat[Action.REAL_ENVIDO], 8.0 / 24.0, rel_tol=1e-6)
    assert math.isclose(avg_strat[Action.FALTA_ENVIDO], 0.0, abs_tol=1e-6)
    assert math.isclose(avg_strat[Action.NO_QUIERO_ENVIDO], 4.0 / 24.0, rel_tol=1e-6)
