import random

from truco_bot.core.card import ALL_CARDS, Suit, create_card
from truco_bot.core.equity import calculate_envido_equity


def test_envido_equity_max_score():
    hand = [
        create_card(7, Suit.ESPADAS),
        create_card(6, Suit.ESPADAS),
        create_card(3, Suit.OROS)
    ]

    pie_equity, mano_equity = calculate_envido_equity(hand)
    assert mano_equity == 1.0
    assert pie_equity < 1.0

def test_envido_equity_min_score():
    hand = [
        create_card(12, Suit.ESPADAS),
        create_card(11, Suit.OROS),
        create_card(10, Suit.BASTOS)
    ]

    pie_equity, mano_equity = calculate_envido_equity(hand)
    assert mano_equity > 0.0
    assert pie_equity == 0.0

def test_envido_equity_positional_invariance():
    for _ in range(100):
        hand = random.sample(ALL_CARDS, 3)
        pie_equity, mano_equity = calculate_envido_equity(hand)
        
        # Being Mano can never hurt your equity
        assert mano_equity >= pie_equity
