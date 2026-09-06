"""Deck handling, shuffling, and dealing."""

import random

from .card import ALL_CARDS, Card

_DEFAULT_RNG = random.Random()


def get_standard_deck() -> list[Card]:
    return list(ALL_CARDS)


def shuffle(deck: list[Card], seed: int | None = None) -> list[Card]:
    rng = random.Random(seed) if seed is not None else _DEFAULT_RNG
    shuffled = deck.copy()
    rng.shuffle(shuffled)
    return shuffled


def deal(
    num_players: int = 2, cards_per_player: int = 3, seed: int | None = None
) -> tuple[list[list[Card]], list[Card]]:
    deck = get_standard_deck()
    deck = shuffle(deck, seed=seed)

    hands = []
    for _ in range(num_players):
        hand = [deck.pop() for _ in range(cards_per_player)]
        hands.append(hand)

    return hands, deck
