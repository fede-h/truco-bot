"""State feature encoder for Deep CFR (156 dimensions)."""

from __future__ import annotations

import torch
import truco_engine
from truco_engine import BitboardState

from truco_bot.agents.cfr.native_agent import gamestate_to_bitboard
from truco_bot.core.state import GameState

STATE_FEATURE_DIM = 156


def encode_state(state: BitboardState | GameState, player: int | None = None) -> torch.Tensor:
    """Encode BitboardState or GameState into a 156-dim float32 torch.Tensor."""
    # ponytail: convert GameState to native BitboardState to guarantee bit-exact parity
    if isinstance(state, GameState):
        bstate = gamestate_to_bitboard(state)
    else:
        bstate = state

    if player is None:
        player = bstate.active_player
    opp = 1 - player

    tensor = torch.zeros(STATE_FEATURE_DIM, dtype=torch.float32)

    # 0..39: multi-hot player's hand cards
    for c in bstate.hands[player]:
        if 0 <= c < 40:
            tensor[c] = 1.0

    # 40..119: trick cards (40..79 self, 80..119 opp)
    tc = bstate.trick_cards
    if bstate.trick_leader == player:
        self_c, opp_c = tc[0], tc[1]
    else:
        opp_c, self_c = tc[0], tc[1]

    if 0 <= self_c < 40:
        tensor[40 + self_c] = 1.0
    if 0 <= opp_c < 40:
        tensor[80 + opp_c] = 1.0

    # 120..128: trick results (3 tricks x [self_win, tie, opp_win])
    tr_results = bstate.trick_results
    for tr in range(3):
        res = tr_results[tr]
        offset = 120 + tr * 3
        if res == 0:  # tie
            tensor[offset + 1] = 1.0
        elif (res == 1 and player == 0) or (res == -1 and player == 1):  # self win
            tensor[offset + 0] = 1.0
        elif (res == -1 and player == 0) or (res == 1 and player == 1):  # opp win
            tensor[offset + 2] = 1.0

    # 129..134: trick structure
    cur_trick = min(bstate.current_trick, 2)
    tensor[129 + cur_trick] = 1.0
    if bstate.trick_leader == player:
        tensor[132] = 1.0
    if bstate.mano == player:
        tensor[133] = 1.0
    if bstate.active_player == player:
        tensor[134] = 1.0

    # 135..144: envido context
    if bstate.envido_resolved:
        tensor[135] = 1.0
    if bstate.pending_envido_from == player:
        tensor[136] = 1.0
    elif bstate.pending_envido_from == opp:
        tensor[137] = 1.0

    chain = bstate.envido_chain
    for i, act in enumerate(chain[:6]):
        act_val = act.value if hasattr(act, "value") else int(act)
        tensor[138 + i] = act_val / 25.0

    valid_cards = [c for c in bstate.hands[player] if 0 <= c < 40]
    if valid_cards:
        pts = truco_engine.calculate_envido(valid_cards)
        tensor[144] = pts / 33.0

    # 145..153: truco context
    t_level = max(1, min(bstate.truco_level, 4))
    tensor[145 + (t_level - 1)] = 1.0
    if bstate.truco_caller == player:
        tensor[149] = 1.0
    elif bstate.truco_caller == opp:
        tensor[150] = 1.0
    if bstate.pending_truco_from == player:
        tensor[151] = 1.0
    elif bstate.pending_truco_from == opp:
        tensor[152] = 1.0
    tensor[153] = t_level / 4.0

    # 154..155: scores
    max_s = float(bstate.max_score) if bstate.max_score > 0 else 30.0
    self_score = bstate.score_p0 if player == 0 else bstate.score_p1
    opp_score = bstate.score_p1 if player == 0 else bstate.score_p0
    tensor[154] = self_score / max_s
    tensor[155] = opp_score / max_s

    return tensor
