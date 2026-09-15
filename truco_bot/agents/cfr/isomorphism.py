"""Canonical hand representation and suit isomorphism equivalence classes."""

import itertools
from collections import Counter
from collections.abc import Iterator

from truco_bot.core.card import ALL_CARDS, Card, Suit
from truco_bot.core.state import GameState, infoset_key

_CANONICAL_HAND_CACHE: dict[tuple[Card, ...], tuple[Card, ...]] = {}


def canonicalize_hand(hand: list[Card] | tuple[Card, ...]) -> tuple[Card, ...]:
    """Normalize suits to canonical labels preserving rank, envido value, and suit grouping."""
    if not hand:
        return ()

    cache_key = tuple(
        sorted(hand, key=lambda c: (-c.truco_rank, -c.envido_value, c.suit.value))
    )
    cached = _CANONICAL_HAND_CACHE.get(cache_key)
    if cached is not None:
        return cached

    counts = Counter(c.suit for c in hand)

    def suit_sort_key(s: Suit) -> tuple[int, int, int, int]:
        s_cards = [c for c in hand if c.suit == s]
        max_rank = max(c.truco_rank for c in s_cards)
        max_envido = max(c.envido_value for c in s_cards)
        return (-counts[s], -max_rank, -max_envido, s.value)

    sorted_suits = sorted(counts.keys(), key=suit_sort_key)
    suit_map = {s: Suit(i + 1) for i, s in enumerate(sorted_suits)}

    canon_cards = [
        Card(
            number=c.number,
            suit=suit_map[c.suit],
            truco_rank=c.truco_rank,
            envido_value=c.envido_value,
        )
        for c in hand
    ]
    canon_cards.sort(
        key=lambda c: (-c.truco_rank, -c.envido_value, c.number, c.suit.value)
    )
    res = tuple(canon_cards)
    _CANONICAL_HAND_CACHE[cache_key] = res
    return res


def canonical_infoset_key(state: GameState, player: int) -> tuple:
    """Wrap standard infoset_key with canonicalized active player hand."""
    raw_key = infoset_key(state, player)
    canon_hand = canonicalize_hand(state.hands[player])
    return (canon_hand,) + raw_key[1:]


def generate_canonical_deals(
    limit: int | None = None,
) -> Iterator[tuple[list[Card], list[Card]]]:
    """Generate canonical deal partitions across player 0 and player 1 hands."""
    seen_h0: set[tuple[Card, ...]] = set()
    yielded = 0

    for h0 in itertools.combinations(ALL_CARDS, 3):
        ch0 = canonicalize_hand(h0)
        if ch0 in seen_h0:
            continue
        seen_h0.add(ch0)

        rem_deck = [c for c in ALL_CARDS if c not in h0]
        seen_h1: set[tuple[Card, ...]] = set()

        for h1 in itertools.combinations(rem_deck, 3):
            ch1 = canonicalize_hand(h1)
            if ch1 in seen_h1:
                continue
            seen_h1.add(ch1)

            yield list(h0), list(h1)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


# Alias for alternative naming
canonical_deal_partitions = generate_canonical_deals
