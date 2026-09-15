"""Equity fallback heuristics for off-tree decisions."""

from collections.abc import Sequence
from dataclasses import dataclass

from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.rules import calculate_envido
from truco_bot.core.state import GameState, get_legal_actions
from truco_bot.env.obs import ID_TO_CARD


@dataclass(slots=True)
class EquityFallback:
    """Heuristic fallback for off-tree card play and betting decisions."""

    def _decide(
        self,
        hand_cards: list[Card],
        legal_actions: Sequence[Action],
        opp_card: Card | None = None,
    ) -> Action:
        legal_set = set(legal_actions)

        # Envido answering
        if Action.QUIERO_ENVIDO in legal_set:
            points = calculate_envido(hand_cards) if hand_cards else 0
            if points >= 28:
                if Action.FALTA_ENVIDO in legal_set and points >= 33:
                    return Action.FALTA_ENVIDO
                if Action.REAL_ENVIDO in legal_set and points >= 31:
                    return Action.REAL_ENVIDO
                return Action.QUIERO_ENVIDO
            if Action.NO_QUIERO_ENVIDO in legal_set:
                return Action.NO_QUIERO_ENVIDO
            return legal_actions[0]

        # Envido initiation
        if any(a in legal_set for a in (Action.ENVIDO, Action.REAL_ENVIDO, Action.FALTA_ENVIDO)):
            points = calculate_envido(hand_cards) if hand_cards else 0
            if Action.FALTA_ENVIDO in legal_set and points >= 33:
                return Action.FALTA_ENVIDO
            if Action.REAL_ENVIDO in legal_set and points >= 31:
                return Action.REAL_ENVIDO
            if Action.ENVIDO in legal_set and points >= 28:
                return Action.ENVIDO

        # Truco answering
        if Action.QUIERO_TRUCO in legal_set:
            if any(c.truco_rank >= 8 for c in hand_cards):
                if Action.RETRUCO in legal_set and any(c.truco_rank >= 13 for c in hand_cards):
                    return Action.RETRUCO
                return Action.QUIERO_TRUCO
            if Action.NO_QUIERO_TRUCO in legal_set:
                return Action.NO_QUIERO_TRUCO
            return legal_actions[0]

        # Card playing
        card_actions = [
            (act, hand_cards[idx])
            for idx, act in enumerate((Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2))
            if act in legal_set and idx < len(hand_cards)
        ]
        if card_actions:
            if opp_card is not None:
                winning = [(act, c) for act, c in card_actions if c.truco_rank > opp_card.truco_rank]
                return min(winning, key=lambda x: x[1].truco_rank)[0] if winning else min(card_actions, key=lambda x: x[1].truco_rank)[0]
            return max(card_actions, key=lambda x: x[1].truco_rank)[0]

        return legal_actions[0]

    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        legal_actions = [Action(i) for i, is_legal in enumerate(mask) if is_legal]
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        hand_cards = [ID_TO_CARD[cid] for cid in obs[:3] if cid != 0]
        opp_card = ID_TO_CARD[obs[3]] if obs[3] != 0 else None
        return self._decide(hand_cards, legal_actions, opp_card)

    decide = act

    def decide_state(
        self, state: GameState, legal_actions: list[Action] | None = None
    ) -> Action:
        if legal_actions is None:
            legal_actions = get_legal_actions(state)
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        hand = state.hands[state.active_player]
        opp_card = state.trick_cards[0][1] if state.trick_cards else None
        return self._decide(hand, legal_actions, opp_card)
