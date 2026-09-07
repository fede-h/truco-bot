"""Combinatorial precomputations lookup table."""

from typing_extensions import Tuple
import itertools
from .card import Card, ALL_CARDS
from truco_bot.core.rules import calculate_envido

def calculate_envido_equity(
    hand: tuple[Card, Card, Card] | list[Card]) -> tuple[float, float]:
    wins = 0
    ties = 0

    remaining_cards = [card for card in ALL_CARDS if card not in hand]
    score = calculate_envido(list(hand))
    opp_hands = itertools.combinations(remaining_cards, 3)

    for opp_hand in opp_hands:
        opp_score = calculate_envido(opp_hand)
        if score > opp_score: 
            wins += 1
        elif score == opp_score: 
            ties += 1

    return (wins / 7770, (wins + ties) / 7770)

if __name__ == "__main__":
    import pickle

    lookup: dict[frozenset, Tuple] = {}
    for hand in itertools.combinations(ALL_CARDS, 3):
        lookup[frozenset(hand)] = (calculate_envido_equity(hand))
    
    with open('truco_bot/core/lookup/equity.pkl', 'wb') as file:
        pickle.dump(lookup, file)
