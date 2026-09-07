import random

import pytest

from truco_bot.core.actions import Action
from truco_bot.core.card import Suit, create_card
from truco_bot.core.deck import deal
from truco_bot.core.state import _clone_state, create_initial_hand_state, get_legal_actions
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.env.masking import get_action_mask, get_action_mask_bits
from truco_bot.env.obs import CARD_TO_ID, get_observation


def test_clone_state_isolation():
    hands, _ = deal(2, 3, seed=42)
    state = create_initial_hand_state(hands, mano=0)
    state.trick_cards.append((0, hands[0][0]))
    state.trick_results.append(1)
    state.envido_chain.append(Action.ENVIDO)

    cloned = _clone_state(state)

    # Mutate cloned mutable collections
    cloned.hands[0].pop()
    cloned.trick_cards.append((1, hands[1][0]))
    cloned.trick_results.append(-1)
    cloned.envido_chain.append(Action.QUIERO_ENVIDO)

    # Check original is unchanged
    assert len(state.hands[0]) == 3
    assert len(state.trick_cards) == 1
    assert len(state.trick_results) == 1
    assert len(state.envido_chain) == 1


def test_action_mask_and_bits():
    hands, _ = deal(2, 3, seed=42)
    state = create_initial_hand_state(hands, mano=0)

    mask = get_action_mask(state)
    assert isinstance(mask, list)
    assert len(mask) == 26
    assert all(isinstance(x, bool) for x in mask)

    legal_actions = get_legal_actions(state)
    for a in legal_actions:
        assert mask[a.value] is True

    # Check non-legal actions are False
    legal_values = {a.value for a in legal_actions}
    for idx, val in enumerate(mask):
        if idx not in legal_values:
            assert val is False

    bits = get_action_mask_bits(state)
    assert isinstance(bits, int)
    for a in legal_actions:
        assert (bits & (1 << a.value)) != 0

    for idx in range(26):
        if idx not in legal_values:
            assert (bits & (1 << idx)) == 0


def test_observation_structure():
    c1 = create_card(1, Suit.ESPADAS)
    c2 = create_card(7, Suit.ESPADAS)
    c3 = create_card(3, Suit.ESPADAS)
    c4 = create_card(4, Suit.COPAS)
    c5 = create_card(5, Suit.COPAS)
    c6 = create_card(6, Suit.COPAS)
    hands = [[c1, c2, c3], [c4, c5, c6]]

    state = create_initial_hand_state(hands, mano=0)
    obs_p0 = get_observation(state, player=0)
    obs_p1 = get_observation(state, player=1)

    assert isinstance(obs_p0, tuple)
    assert len(obs_p0) == 12
    assert len(obs_p1) == 12
    assert all(isinstance(x, int) for x in obs_p0)

    # P0 sees their own cards
    assert obs_p0[0] == CARD_TO_ID[c1]
    assert obs_p0[1] == CARD_TO_ID[c2]
    assert obs_p0[2] == CARD_TO_ID[c3]

    # P1 sees their own cards
    assert obs_p1[0] == CARD_TO_ID[c4]
    assert obs_p1[1] == CARD_TO_ID[c5]
    assert obs_p1[2] == CARD_TO_ID[c6]

    # No table cards played yet
    assert obs_p0[3] == 0
    assert obs_p0[4] == 0

    # Scores, trick count, truco level, envido resolved, active player, mano
    assert obs_p0[5] == 0  # score_p0
    assert obs_p0[6] == 0  # score_p1
    assert obs_p0[7] == 0  # current_trick
    assert obs_p0[8] == 1  # truco_level
    assert obs_p0[9] == 0  # envido_resolved
    assert obs_p0[10] == 0  # active_player
    assert obs_p0[11] == 0  # mano


def test_hand_env_reset_and_step():
    env = TrucoHandEnv()

    # Calling step or obs before reset raises RuntimeError
    with pytest.raises(RuntimeError):
        env.step(Action.TRUCO)
    with pytest.raises(RuntimeError):
        env.get_obs(0)
    with pytest.raises(RuntimeError):
        env.get_mask()

    state, mask = env.reset(seed=123)
    assert not state.is_hand_done
    assert len(mask) == 26
    assert mask == env.get_mask()

    obs = env.get_obs(0)
    assert len(obs) == 12

    # Step TRUCO
    next_state, pts, done, next_mask = env.step(Action.TRUCO)
    assert next_state.pending_truco_response
    assert pts == 0
    assert not done
    assert Action.QUIERO_TRUCO.value in [i for i, v in enumerate(next_mask) if v]

    # Step NO_QUIERO_TRUCO
    final_state, pts, done, final_mask = env.step(Action.NO_QUIERO_TRUCO)
    assert done
    assert pts == 1
    assert final_state.winner == 0
    assert all(not v for v in final_mask)


def test_hand_env_custom_hands():
    c1 = create_card(1, Suit.ESPADAS)
    c2 = create_card(1, Suit.BASTOS)
    c3 = create_card(7, Suit.ESPADAS)
    c4 = create_card(4, Suit.COPAS)
    c5 = create_card(4, Suit.BASTOS)
    c6 = create_card(4, Suit.OROS)
    hands = [[c1, c2, c3], [c4, c5, c6]]

    env = TrucoHandEnv()
    state, mask = env.reset(hands=hands)
    assert len(mask) == 26
    assert state.hands[0] == [c1, c2, c3]
    assert state.hands[1] == [c4, c5, c6]


def test_hand_env_random_rollout():
    rng = random.Random(42)
    env = TrucoHandEnv()

    for seed in range(50):
        _, mask = env.reset(seed=seed)
        steps = 0
        done = False
        while not done and steps < 30:
            legal_indices = [idx for idx, is_legal in enumerate(mask) if is_legal]
            assert len(legal_indices) > 0
            action = Action(rng.choice(legal_indices))
            _, _, done, mask = env.step(action)
            steps += 1
        assert done
