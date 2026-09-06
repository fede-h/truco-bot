from truco_bot.core.actions import Action
from truco_bot.core.card import Suit, create_card
from truco_bot.core.deck import deal, get_standard_deck
from truco_bot.core.rules import calculate_envido, resolve_hand
from truco_bot.core.state import create_initial_hand_state, step


def test_deck_cards():
    deck = get_standard_deck()
    assert len(deck) == 40
    for card in deck:
        assert card.number not in (8, 9)
    ranks = {card.truco_rank for card in deck}
    assert len(ranks) == 14


def test_deal():
    hands, deck = deal(2, 3)
    assert len(hands) == 2
    assert len(hands[0]) == 3
    assert len(hands[1]) == 3
    assert len(deck) == 34

    # Check no duplicates
    all_dealt = hands[0] + hands[1] + deck
    assert len(set(all_dealt)) == 40


def test_envido_calculation():
    # 7 and 6 of same suit -> 33
    cards1 = [create_card(7, Suit.ESPADAS), create_card(6, Suit.ESPADAS), create_card(1, Suit.OROS)]
    assert calculate_envido(cards1) == 33

    # 10 and 11 of same suit -> 20
    cards2 = [create_card(10, Suit.COPAS), create_card(11, Suit.COPAS), create_card(2, Suit.BASTOS)]
    assert calculate_envido(cards2) == 20

    # 1, 2, 3 in different suits -> 3
    cards3 = [create_card(1, Suit.ESPADAS), create_card(2, Suit.BASTOS), create_card(3, Suit.OROS)]
    assert calculate_envido(cards3) == 3

    # 12, 11, 10 in different suits -> 0
    cards4 = [
        create_card(12, Suit.ESPADAS),
        create_card(11, Suit.BASTOS),
        create_card(10, Suit.OROS),
    ]
    assert calculate_envido(cards4) == 0


def test_parda_resolution():
    # T1 parda, T2 P0 wins -> P0 wins hand
    assert resolve_hand([0, 1]) == 0
    # T1 P0 wins, T2 parda -> P0 wins hand
    assert resolve_hand([1, 0]) == 0
    # T1 P0 wins, T2 P1 wins, T3 parda -> P0 wins hand
    assert resolve_hand([1, -1, 0]) == 0
    # Triple parda -> Mano wins hand
    assert resolve_hand([0, 0, 0], mano=1) == 1
    assert resolve_hand([0, 0, 0], mano=0) == 0


def test_state_play_3_tricks():
    # Set up rigged hands
    hands = [
        [
            create_card(1, Suit.ESPADAS),
            create_card(7, Suit.ESPADAS),
            create_card(3, Suit.ESPADAS),
        ],  # P0: ranks 14, 12, 10
        [
            create_card(4, Suit.COPAS),
            create_card(5, Suit.COPAS),
            create_card(6, Suit.COPAS),
        ],  # P1: ranks 1, 2, 3
    ]
    state = create_initial_hand_state(hands, mano=0)

    # P0 plays highest, P1 plays lowest
    state = step(state, Action.PLAY_CARD_0)  # P0 plays 1 of Espadas
    state = step(state, Action.PLAY_CARD_0)  # P1 plays 4 of Copas
    # P0 wins trick 1, plays next
    state = step(state, Action.PLAY_CARD_0)  # P0 plays 7 of Espadas
    state = step(state, Action.PLAY_CARD_0)  # P1 plays 5 of Copas

    assert state.is_hand_done
    assert state.winner == 0
    assert state.points_won == 1


def test_state_truco_no_quiero():
    hands, _ = deal(2, 3)
    state = create_initial_hand_state(hands, mano=0)

    state = step(state, Action.TRUCO)
    assert state.pending_truco_response
    assert state.active_player == 1

    state = step(state, Action.NO_QUIERO_TRUCO)
    assert state.is_hand_done
    assert state.winner == 0
    assert state.points_won == 1


def test_state_envido_quiero():
    # P0 has 33 envido, P1 has 20
    hands = [
        [create_card(7, Suit.ESPADAS), create_card(6, Suit.ESPADAS), create_card(1, Suit.OROS)],
        [create_card(10, Suit.COPAS), create_card(11, Suit.COPAS), create_card(2, Suit.BASTOS)],
    ]
    state = create_initial_hand_state(hands, mano=0)

    state = step(state, Action.ENVIDO)
    state = step(state, Action.QUIERO_ENVIDO)

    assert state.envido_resolved
    assert state.score_p0 == 2  # P0 won Envido
    assert state.score_p1 == 0
