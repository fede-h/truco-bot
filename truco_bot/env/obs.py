"""Public and private observation tensors."""

from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import GameState

CARD_TO_ID = {card: i + 1 for i, card in enumerate(ALL_CARDS)}
ID_TO_CARD = {i + 1: card for i, card in enumerate(ALL_CARDS)}


def get_observation(state: GameState, player: int) -> tuple[int, ...]:
    hand = state.hands[player] if 0 <= player < len(state.hands) else []
    c0 = CARD_TO_ID.get(hand[0], 0) if len(hand) > 0 else 0
    c1 = CARD_TO_ID.get(hand[1], 0) if len(hand) > 1 else 0
    c2 = CARD_TO_ID.get(hand[2], 0) if len(hand) > 2 else 0

    tc0 = CARD_TO_ID.get(state.trick_cards[0][1], 0) if len(state.trick_cards) > 0 else 0
    tc1 = CARD_TO_ID.get(state.trick_cards[1][1], 0) if len(state.trick_cards) > 1 else 0

    return (
        c0,
        c1,
        c2,
        tc0,
        tc1,
        state.score_p0,
        state.score_p1,
        state.current_trick,
        state.truco_level,
        1 if state.envido_resolved else 0,
        state.active_player,
        state.mano,
    )
