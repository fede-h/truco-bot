"""Legal action mask generator."""

from truco_bot.core.state import GameState, get_legal_actions

_EMPTY_MASK: list[bool] = [False] * 26


def get_action_mask(state: GameState) -> list[bool]:
    if state.is_hand_done:
        return _EMPTY_MASK.copy()
    mask = [False] * 26
    for a in get_legal_actions(state):
        mask[a] = True
    return mask


def get_action_mask_bits(state: GameState) -> int:
    if state.is_hand_done:
        return 0
    bits = 0
    for a in get_legal_actions(state):
        bits |= 1 << a
    return bits
