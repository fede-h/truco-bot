from truco_bot.core.actions import Action
from truco_bot.core.card import Suit, create_card
from truco_bot.core.state import create_initial_hand_state, infoset_key


def _base_state():
    hands = [
        [create_card(7, Suit.ESPADAS), create_card(6, Suit.COPAS), create_card(3, Suit.OROS)],
        [create_card(1, Suit.BASTOS), create_card(2, Suit.ESPADAS), create_card(12, Suit.COPAS)],
    ]
    return create_initial_hand_state(hands=hands, mano=0)


def test_key_is_hashable():
    key = infoset_key(_base_state(), 0)
    table = {}
    table[key] = "regret row"
    assert table[infoset_key(_base_state(), 0)] == "regret row"


def test_hidden_hand_is_invisible():
    a = _base_state()
    b = _base_state()
    b.hands[1] = [create_card(5, Suit.OROS), create_card(11, Suit.BASTOS), create_card(4, Suit.COPAS)]
    assert infoset_key(a, 0) == infoset_key(b, 0)


def test_own_hand_order_is_invariant():
    a = _base_state()
    b = _base_state()
    b.hands[0] = [b.hands[0][2], b.hands[0][0], b.hands[0][1]]
    assert infoset_key(a, 0) == infoset_key(b, 0)


def test_visible_change_differs():
    a = _base_state()
    b = _base_state()
    b.envido_chain = [Action.ENVIDO]
    assert infoset_key(a, 0) != infoset_key(b, 0)


def test_envido_outcome_in_infoset():
    base = _base_state()
    assert infoset_key(base, 0)[6:8] == (None, 0)

    p1_won = _base_state()
    p1_won.envido_resolved = True
    p1_won.score_p1 = 2
    assert infoset_key(p1_won, 0)[6:8] == (1, 2)
