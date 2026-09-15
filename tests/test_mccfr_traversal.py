"""Tests for External Sampling MCCFR traversal engine and CFR+ regret updates."""

import random

from truco_bot.agents.cfr.mccfr_traversal import external_sampling_traverse

from truco_bot.agents.cfr.node import CFRNode
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state, step


def test_mccfr_traversal_terminal_payoff_evaluation() -> None:
    """Verify terminal hand payoff matches Truco zero-sum game scoring rules."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    # Fold immediately: player 0 folds (IR_AL_MAZO) -> player 1 wins 1 point (or 2 points)
    terminal_state = step(state, Action.IR_AL_MAZO)
    assert terminal_state.is_hand_done

    nodes: dict[tuple, CFRNode] = {}
    rng = random.Random(42)

    # Utility for player 0 (the folding player) must be negative
    u0 = external_sampling_traverse(
        terminal_state, update_player=0, nodes=nodes, is_canonical=True, rng=rng
    )
    assert isinstance(u0, float)
    assert u0 < 0.0

    # Utility for player 1 must be positive and zero-sum conservation: u1 == -u0
    u1 = external_sampling_traverse(
        terminal_state, update_player=1, nodes=nodes, is_canonical=True, rng=rng
    )
    assert isinstance(u1, float)
    assert u1 == -u0


def test_mccfr_traversal_active_player_explores_all_legal_actions() -> None:
    """When active_player == update_player, traversal evaluates all legal actions and updates regrets."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)
    assert state.active_player == 0

    nodes: dict[tuple, CFRNode] = {}
    rng = random.Random(42)

    val = external_sampling_traverse(
        state, update_player=0, nodes=nodes, is_canonical=True, rng=rng, cfr_plus=True
    )
    assert isinstance(val, float)
    assert len(nodes) > 0

    # Root infoset must be in nodes
    from truco_bot.agents.cfr.isomorphism import canonical_infoset_key

    root_key = canonical_infoset_key(state, player=0)
    assert root_key in nodes

    root_node = nodes[root_key]
    assert len(root_node.actions) > 1
    # Every legal action must have an entry in regret_sum
    for action in root_node.actions:
        assert action in root_node.regret_sum


def test_mccfr_traversal_opponent_samples_single_action() -> None:
    """When active_player != update_player, traversal samples a single action and accumulates strategy."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)
    # Step player 0 so active player is player 1
    state_p1 = step(state, Action.PLAY_CARD_1)
    assert state_p1.active_player == 1

    nodes: dict[tuple, CFRNode] = {}
    rng = random.Random(42)

    # We update player 0; active player is player 1 (opponent)
    val = external_sampling_traverse(
        state_p1, update_player=0, nodes=nodes, is_canonical=True, rng=rng, cfr_plus=True
    )
    assert isinstance(val, float)

    from truco_bot.agents.cfr.isomorphism import canonical_infoset_key

    opp_key = canonical_infoset_key(state_p1, player=1)
    assert opp_key in nodes
    opp_node = nodes[opp_key]

    # Opponent node must have accumulated strategy sum
    total_strat_sum = sum(opp_node.strategy_sum.values())
    assert total_strat_sum > 0.0


def test_mccfr_traversal_cfr_plus_flooring() -> None:
    """Verify cfr_plus=True prevents negative regrets in traversed nodes."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    nodes: dict[tuple, CFRNode] = {}
    rng = random.Random(123)

    for _ in range(3):
        external_sampling_traverse(
            state, update_player=0, nodes=nodes, is_canonical=True, rng=rng, cfr_plus=True
        )

    # All regret sums must be non-negative under CFR+
    for node in nodes.values():
        for regret in node.regret_sum.values():
            assert regret >= 0.0, f"Found negative regret {regret} under CFR+"


def test_mccfr_traversal_raw_infoset_support() -> None:
    """Verify traversal works when is_canonical=False."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    nodes: dict[tuple, CFRNode] = {}
    rng = random.Random(42)

    val = external_sampling_traverse(
        state, update_player=0, nodes=nodes, is_canonical=False, rng=rng
    )
    assert isinstance(val, float)
    assert len(nodes) > 0


def test_mccfr_traversal_terminates_finite_time() -> None:
    """Verify traversal completes and returns bounded utility on multiple deals."""
    from truco_bot.core.deck import deal

    nodes: dict[tuple, CFRNode] = {}
    rng = random.Random(999)

    for i in range(5):
        hands = deal(num_players=2, cards_per_player=3, rng=rng)
        state = create_initial_hand_state(hands, mano=i % 2)

        u0 = external_sampling_traverse(
            state, update_player=0, nodes=nodes, is_canonical=True, rng=rng, cfr_plus=True
        )
        u1 = external_sampling_traverse(
            state, update_player=1, nodes=nodes, is_canonical=True, rng=rng, cfr_plus=True
        )
        assert isinstance(u0, float)
        assert isinstance(u1, float)
        # Maximum possible score differential in a single hand is bounded between -7 and +7
        assert -10.0 <= u0 <= 10.0
        assert -10.0 <= u1 <= 10.0
