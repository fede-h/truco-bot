"""Unit and property tests for CFR isomorphism and canonical equivalence classes."""

import itertools

from truco_bot.agents.cfr.isomorphism import (
    canonical_infoset_key,
    canonicalize_hand,
    generate_canonical_deals,
)
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.rules import calculate_envido
from truco_bot.core.state import create_initial_hand_state


def test_canonicalize_hand_all_9880_combinations_preserve_envido():
    """Verify calculate_envido(canonical(h)) == calculate_envido(h) for all 9880 hands."""
    for hand in itertools.combinations(ALL_CARDS, 3):
        canon_hand = canonicalize_hand(list(hand))
        assert len(canon_hand) == 3
        orig_envido = calculate_envido(list(hand))
        canon_envido = calculate_envido(list(canon_hand))
        assert orig_envido == canon_envido, f"Envido mismatch for hand {hand}: {orig_envido} vs {canon_envido}"


def test_canonicalize_hand_preserves_trick_comparison():
    """Verify card pairwise trick comparison order is preserved under canonicalization."""
    # Test that each card's rank is identical in its canonicalized counterpart
    for hand in itertools.combinations(ALL_CARDS, 3):
        canon_hand = canonicalize_hand(list(hand))
        orig_ranks = sorted(c.truco_rank for c in hand)
        canon_ranks = sorted(c.truco_rank for c in canon_hand)
        assert orig_ranks == canon_ranks


def test_canonicalize_hand_compresses_space():
    """Verify that 9880 hands compress into significantly fewer canonical classes."""
    canon_classes = set()
    for hand in itertools.combinations(ALL_CARDS, 3):
        canon_classes.add(canonicalize_hand(list(hand)))

    assert len(canon_classes) < 9880
    assert len(canon_classes) == 1812


def test_canonicalize_hand_edge_cases():
    assert canonicalize_hand([]) == ()
    single_card = [ALL_CARDS[0]]
    canon_single = canonicalize_hand(single_card)
    assert len(canon_single) == 1
    assert canon_single[0].truco_rank == single_card[0].truco_rank


def test_canonical_infoset_key_structure():
    """Verify canonical_infoset_key correctly wraps state hand with canonical hand."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    raw_canon = canonicalize_hand(h0)
    key = canonical_infoset_key(state, player=0)

    assert key[0] == raw_canon
    assert len(key) == 16


def test_generate_canonical_deals():
    deals = list(generate_canonical_deals(limit=10))
    assert len(deals) == 10
    for h0, h1 in deals:
        assert len(h0) == 3
        assert len(h1) == 3
        # No cards duplicated between h0 and h1
        assert len(set(h0).intersection(set(h1))) == 0
