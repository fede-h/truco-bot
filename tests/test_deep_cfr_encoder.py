"""Tests for Deep CFR feature encoder."""

import pytest
import torch
from truco_bot.agents.deep_cfr.encoder import (
    STATE_FEATURE_DIM,
    encode_state,
)
from truco_engine import BitboardState

from truco_bot.agents.cfr.native_agent import gamestate_to_bitboard
from truco_bot.core.actions import Action
from truco_bot.core.card import Suit, create_card
from truco_bot.core.state import create_initial_hand_state


def test_encoder_dimension_and_type():
    """Verify encode_state on BitboardState produces exact 156-dim float32 torch.Tensor."""
    assert STATE_FEATURE_DIM == 156
    bstate = BitboardState([[0, 1, 2], [3, 4, 5]], 0)
    tensor = encode_state(bstate)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (156,)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_encoder_hand_cards_flagged():
    """Verify active player's hand cards are correctly flagged in the first 40 dimensions."""
    # P0: cards 0 (1 Espada), 5 (6 Espada), 9 (12 Espada)
    # P1: cards 10 (1 Basto), 15 (6 Basto), 20 (1 Oro)
    bstate = BitboardState([[0, 5, 9], [10, 15, 20]], 0)

    # Perspective P0
    t0 = encode_state(bstate, player=0)
    assert t0[:40].sum().item() == pytest.approx(3.0)
    assert t0[0].item() == 1.0
    assert t0[5].item() == 1.0
    assert t0[9].item() == 1.0
    assert (t0[:40] == 0.0).sum().item() == 37

    # Perspective P1
    t1 = encode_state(bstate, player=1)
    assert t1[:40].sum().item() == pytest.approx(3.0)
    assert t1[10].item() == 1.0
    assert t1[15].item() == 1.0
    assert t1[20].item() == 1.0
    assert t1[0].item() == 0.0
    assert t1[5].item() == 0.0
    assert t1[9].item() == 0.0


def test_encoder_trick_cards_encoding():
    """Verify trick cards are properly encoded in dims 40..119 (self vs opp)."""
    # P0 leads card 7, P1 has not played yet
    bstate = BitboardState.from_components(
        hands=[[0, 1, 2], [3, 4, 5]],
        trick_cards=[7, 255],
        trick_results=[127, 127, 127],
        trick_leader=0,
        current_trick=0,
        active_player=1,
        mano=0,
        score_p0=0,
        score_p1=0,
        max_score=30,
        envido_chain=[],
        pending_envido_from=255,
        truco_level=1,
        truco_caller=255,
        pending_truco_from=255,
        flags=0,
        winner=255,
        points_won=0,
    )

    t0 = encode_state(bstate, player=0)
    # From P0 perspective: self played 7 (dim 40 + 7 = 47), opp played none
    assert t0[40 + 7].item() == 1.0
    assert t0[40:80].sum().item() == pytest.approx(1.0)
    assert t0[80:120].sum().item() == pytest.approx(0.0)

    t1 = encode_state(bstate, player=1)
    # From P1 perspective: opp played 7 (dim 80 + 7 = 87), self played none
    assert t1[40:80].sum().item() == pytest.approx(0.0)
    assert t1[80 + 7].item() == 1.0
    assert t1[80:120].sum().item() == pytest.approx(1.0)

    # Both players played: P0 played 7, P1 responded with 18
    bstate_both = BitboardState.from_components(
        hands=[[0, 1, 2], [3, 4, 5]],
        trick_cards=[7, 18],
        trick_results=[127, 127, 127],
        trick_leader=0,
        current_trick=0,
        active_player=0,
        mano=0,
        score_p0=0,
        score_p1=0,
        max_score=30,
        envido_chain=[],
        pending_envido_from=255,
        truco_level=1,
        truco_caller=255,
        pending_truco_from=255,
        flags=0,
        winner=255,
        points_won=0,
    )

    t0_both = encode_state(bstate_both, player=0)
    assert t0_both[40 + 7].item() == 1.0
    assert t0_both[80 + 18].item() == 1.0

    t1_both = encode_state(bstate_both, player=1)
    assert t1_both[40 + 18].item() == 1.0
    assert t1_both[80 + 7].item() == 1.0


def test_encoder_trick_results_encoding():
    """Verify trick results (win, tie, loss) are encoded relative to perspective in dims 120..128."""
    # Trick 0 won by P0 (result = 1)
    bstate_win = BitboardState.from_components(
        hands=[[0, 1, 2], [3, 4, 5]],
        trick_cards=[255, 255],
        trick_results=[1, 127, 127],
        trick_leader=0,
        current_trick=1,
        active_player=0,
        mano=0,
        score_p0=0,
        score_p1=0,
        max_score=30,
        envido_chain=[],
        pending_envido_from=255,
        truco_level=1,
        truco_caller=255,
        pending_truco_from=255,
        flags=0,
        winner=255,
        points_won=0,
    )

    t0_win = encode_state(bstate_win, player=0)
    assert t0_win[120].item() == 1.0  # self win at trick 0
    assert t0_win[121].item() == 0.0
    assert t0_win[122].item() == 0.0

    t1_win = encode_state(bstate_win, player=1)
    assert t1_win[120].item() == 0.0
    assert t1_win[121].item() == 0.0
    assert t1_win[122].item() == 1.0  # opp win (loss for P1) at trick 0

    # Trick 0 tied (parda, result = 0)
    bstate_tie = BitboardState.from_components(
        hands=[[0, 1, 2], [3, 4, 5]],
        trick_cards=[255, 255],
        trick_results=[0, 127, 127],
        trick_leader=0,
        current_trick=1,
        active_player=0,
        mano=0,
        score_p0=0,
        score_p1=0,
        max_score=30,
        envido_chain=[],
        pending_envido_from=255,
        truco_level=1,
        truco_caller=255,
        pending_truco_from=255,
        flags=0,
        winner=255,
        points_won=0,
    )

    t0_tie = encode_state(bstate_tie, player=0)
    assert t0_tie[121].item() == 1.0  # tie slot at trick 0
    t1_tie = encode_state(bstate_tie, player=1)
    assert t1_tie[121].item() == 1.0  # tie slot at trick 0

    # Trick 1 won by P1 (result = -1)
    bstate_t1 = BitboardState.from_components(
        hands=[[0, 1, 2], [3, 4, 5]],
        trick_cards=[255, 255],
        trick_results=[1, -1, 127],
        trick_leader=1,
        current_trick=2,
        active_player=1,
        mano=0,
        score_p0=0,
        score_p1=0,
        max_score=30,
        envido_chain=[],
        pending_envido_from=255,
        truco_level=1,
        truco_caller=255,
        pending_truco_from=255,
        flags=0,
        winner=255,
        points_won=0,
    )

    t0_t1 = encode_state(bstate_t1, player=0)
    assert t0_t1[123 + 2].item() == 1.0  # trick 1 loss for P0
    t1_t1 = encode_state(bstate_t1, player=1)
    assert t1_t1[123 + 0].item() == 1.0  # trick 1 win for P1


def test_encoder_structural_features():
    """Verify trick structure, envido, truco, and score features are encoded."""
    bstate = BitboardState.from_components(
        hands=[[0, 1, 2], [3, 4, 5]],
        trick_cards=[255, 255],
        trick_results=[127, 127, 127],
        trick_leader=0,
        current_trick=1,
        active_player=0,
        mano=0,
        score_p0=15,
        score_p1=10,
        max_score=30,
        envido_chain=[Action.ENVIDO.value, Action.QUIERO_ENVIDO.value],
        pending_envido_from=255,
        truco_level=2,
        truco_caller=1,
        pending_truco_from=1,
        flags=1 << 0,  # FLAG_ENVIDO_RESOLVED
        winner=255,
        points_won=0,
    )

    tensor = encode_state(bstate, player=0)

    # Current trick 1 (dim 129 + 1 = 130 is 1.0)
    assert tensor[130].item() == 1.0
    # Envido resolved (dim 135 is 1.0)
    assert tensor[135].item() == 1.0
    # Truco level 2 (dim 145 + 1 = 146 is 1.0)
    assert tensor[146].item() == 1.0
    # Truco caller is opp (player 1) -> dim 150 is 1.0
    assert tensor[150].item() == 1.0
    # Pending truco from opp -> dim 152 is 1.0
    assert tensor[152].item() == 1.0
    # Normalized score: P0 = 15/30 = 0.5, P1 = 10/30 = 0.3333
    assert tensor[154].item() == pytest.approx(0.5)
    assert tensor[155].item() == pytest.approx(10.0 / 30.0)


def test_encoder_gamestate_parity():
    """Verify GameState produces identical tensor to BitboardState."""
    h0 = [
        create_card(1, Suit.ESPADA),
        create_card(7, Suit.ESPADA),
        create_card(3, Suit.ORO),
    ]
    h1 = [
        create_card(1, Suit.BASTO),
        create_card(4, Suit.COPA),
        create_card(12, Suit.ESPADA),
    ]

    gstate = create_initial_hand_state([h0, h1], mano=0, score_p0=10, score_p1=5)
    bstate = gamestate_to_bitboard(gstate)

    enc_g = encode_state(gstate)
    enc_b = encode_state(bstate)

    assert torch.equal(enc_g, enc_b)

    # Parity from perspective of player 1
    enc_g_p1 = encode_state(gstate, player=1)
    enc_b_p1 = encode_state(bstate, player=1)
    assert torch.equal(enc_g_p1, enc_b_p1)


def test_encoder_edge_cases():
    """Verify edge cases: empty hand, default active player, completed hand."""
    # Empty hand (all cards played)
    bstate_empty = BitboardState.from_components(
        hands=[[255, 255, 255], [255, 255, 255]],
        trick_cards=[255, 255],
        trick_results=[1, -1, 1],
        trick_leader=0,
        current_trick=2,
        active_player=0,
        mano=0,
        score_p0=30,
        score_p1=28,
        max_score=30,
        envido_chain=[],
        pending_envido_from=255,
        truco_level=4,
        truco_caller=0,
        pending_truco_from=255,
        flags=(1 << 0) | (1 << 3),  # resolved and done
        winner=0,
        points_won=4,
    )

    t = encode_state(bstate_empty)
    assert t[:40].sum().item() == pytest.approx(0.0)
    assert t.shape == (156,)
    assert not torch.isnan(t).any()
