# ponytail: invariant property based testing for truco core
import random
from itertools import combinations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from truco_bot.core.card import ALL_CARDS
from truco_bot.core.deck import deal
from truco_bot.core.rules import calculate_envido
from truco_bot.core.state import create_initial_hand_state, get_legal_actions, step


@pytest.mark.parametrize("hand", list(combinations(ALL_CARDS, 3)))
def test_envido_strictly_between_0_and_33(hand):
    # This exhaustively tests all 3-card combinations
    envido = calculate_envido(list(hand))
    assert 0 <= envido <= 33

@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_dealing_never_duplicates_cards(seed):
    hands, deck = deal(2, 3, seed=seed)
    
    all_dealt_cards = hands[0] + hands[1] + deck
    assert len(all_dealt_cards) == 40
    assert len(set(all_dealt_cards)) == 40
    
    # Also verify counts
    assert len(hands[0]) == 3
    assert len(hands[1]) == 3
    assert len(deck) == 34

@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_random_playout_terminates(seed):
    # ponytail: simple random playout loop to ensure we never get stuck
    rng = random.Random(seed)
    
    hands, _ = deal(2, 3, seed=seed)
    state = create_initial_hand_state(hands, mano=0)
    
    step_count = 0
    max_steps = 20
    
    while not state.is_hand_done:
        actions = get_legal_actions(state)
        # It should always have at least one legal action unless hand is done
        assert len(actions) > 0, "No legal actions found for active player"
        
        # Pick random action
        action = rng.choice(actions)
        
        # Step
        state = step(state, action)
        
        step_count += 1
        assert step_count <= max_steps, f"Playout took more than {max_steps} steps"
    
    assert state.is_hand_done
