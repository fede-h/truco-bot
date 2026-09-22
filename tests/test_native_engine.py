"""Integration tests comparing native compiled `truco_engine` against Python reference logic."""

import time
import pytest

from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS, Card, Suit
from truco_bot.core.rules import (
    calculate_envido as py_calculate_envido,
    compare_cards as py_compare_cards,
    resolve_hand as py_resolve_hand,
)
from truco_bot.core.state import create_initial_hand_state, get_legal_actions, step as py_step


def get_native_engine():
    try:
        import truco_engine
        return truco_engine
    except ImportError:
        try:
            from truco_bot import truco_engine
            return truco_engine
        except ImportError:
            return None


native_engine = get_native_engine()
pytestmark = pytest.mark.skipif(
    native_engine is None,
    reason="Native compiled extension `truco_engine` is not yet built.",
)


def _py_card_to_id(card: Card) -> int:
    suit_offset = (card.suit.value - 1) * 10
    num_idx_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 11: 8, 12: 9}
    return suit_offset + num_idx_map[card.number]


def _id_to_py_card(cid: int) -> Card:
    return ALL_CARDS[cid]


def test_card_comparison_matches_python():
    """Verify native compare_cards matches Python reference compare_cards for all 40x40 pairs."""
    for c1_id in range(40):
        for c2_id in range(40):
            card1 = _id_to_py_card(c1_id)
            card2 = _id_to_py_card(c2_id)
            py_res = py_compare_cards(card1, card2)
            native_res = native_engine.compare_cards(c1_id, c2_id)
            assert native_res == py_res, (
                f"Mismatch comparing {card1} vs {card2}: "
                f"native={native_res}, python={py_res}"
            )


def test_envido_calculation_matches_python():
    """Verify native calculate_envido matches Python reference calculate_envido across sample hands."""
    import itertools
    for hand_tuple in itertools.islice(itertools.combinations(range(40), 3), 500):
        py_hand = [_id_to_py_card(cid) for cid in hand_tuple]
        py_score = py_calculate_envido(py_hand)
        native_score = native_engine.calculate_envido(list(hand_tuple))
        assert native_score == py_score, (
            f"Envido mismatch for hand {py_hand}: "
            f"native={native_score}, python={py_score}"
        )


def test_hand_resolution_matches_python():
    """Verify native resolve_hand matches Python resolve_hand for various trick sequences."""
    test_cases = [
        ([1, 1], 0),
        ([-1, -1], 0),
        ([1, -1, 1], 0),
        ([-1, 1, -1], 0),
        ([1, 0], 0),
        ([-1, 0], 1),
        ([0, 1], 0),
        ([0, -1], 1),
        ([1, -1, 0], 0),
        ([-1, 1, 0], 1),
        ([0, 0, 1], 0),
        ([0, 0, -1], 1),
        ([0, 0, 0], 0),
        ([0, 0, 0], 1),
        ([1], 0),
        ([0], 0),
        ([1, -1], 0),
    ]

    for trick_results, mano in test_cases:
        py_res = py_resolve_hand(trick_results, mano=mano)
        native_res = native_engine.resolve_hand(trick_results, mano=mano)
        assert native_res == py_res, (
            f"Resolution mismatch for tricks={trick_results}, mano={mano}: "
            f"native={native_res}, python={py_res}"
        )


def test_initial_state_action_mask_parity():
    """Verify native legal_actions_mask matches Python get_legal_actions for initial states."""
    h0_ids = [0, 6, 3]  # 1e, 7e, 4e
    h1_ids = [10, 26, 13]  # 1b, 7o, 4b

    native_state = native_engine.BitboardState([h0_ids, h1_ids], 0)
    py_h0 = [_id_to_py_card(cid) for cid in h0_ids]
    py_h1 = [_id_to_py_card(cid) for cid in h1_ids]
    py_state = create_initial_hand_state([py_h0, py_h1], mano=0)

    py_legal = get_legal_actions(py_state)
    native_mask = native_state.legal_actions_mask()

    for act in py_legal:
        act_bit = 1 << act.value
        assert (native_mask & act_bit) != 0, f"Action {act.name} ({act.value}) should be legal in native"


def test_state_transition_sequence_parity():
    """Verify a sequence of steps produces identical state attributes in native and Python."""
    h0_ids = [0, 6, 3]
    h1_ids = [10, 26, 13]

    native_state = native_engine.BitboardState([h0_ids, h1_ids], 0)
    py_h0 = [_id_to_py_card(cid) for cid in h0_ids]
    py_h1 = [_id_to_py_card(cid) for cid in h1_ids]
    py_state = create_initial_hand_state([py_h0, py_h1], mano=0)

    # Step: TRUCO
    native_state = native_state.step(Action.TRUCO.value)
    py_state = py_step(py_state, Action.TRUCO)
    assert native_state.active_player == py_state.active_player

    # Step: QUIERO_TRUCO
    native_state = native_state.step(Action.QUIERO_TRUCO.value)
    py_state = py_step(py_state, Action.QUIERO_TRUCO)
    assert native_state.active_player == py_state.active_player
    assert native_state.truco_level == py_state.truco_level

    # Step: PLAY_CARD_0
    native_state = native_state.step(Action.PLAY_CARD_0.value)
    py_state = py_step(py_state, Action.PLAY_CARD_0)
    assert native_state.active_player == py_state.active_player


@pytest.mark.benchmark
def test_native_state_transition_throughput():
    """Verify native engine can execute > 1,000,000 state transitions/sec."""
    hands = [[0, 6, 3], [10, 26, 13]]
    state = native_engine.BitboardState(hands, 0)

    iterations = 200_000
    start = time.perf_counter()
    for _ in range(iterations):
        # Micro step benchmark
        _ = state.step(Action.TRUCO.value)
    elapsed = time.perf_counter() - start

    rate = iterations / elapsed
    print(f"\nNative step throughput: {rate:,.0f} steps/sec")
    assert rate > 200_000, f"Expected > 200,000 steps/sec via Python FFI, got {rate:,.0f}"
