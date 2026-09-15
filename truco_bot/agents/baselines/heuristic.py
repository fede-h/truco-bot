import random

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action
from truco_bot.core.rules import calculate_envido
from truco_bot.env.obs import ID_TO_CARD


class HeuristicAgent(Agent):
    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        hand_ids = [obs[i] for i in range(3) if obs[i] != 0]
        hand = [ID_TO_CARD[card_id] for card_id in hand_ids]

        if not hand:
            return self._random_fallback(mask)

        envido_score = calculate_envido(hand)

        if envido_score >= 33:
            if mask[Action.FALTA_ENVIDO.value]:
                return Action.FALTA_ENVIDO
            if mask[Action.QUIERO_ENVIDO.value]:
                return Action.QUIERO_ENVIDO
        
        if envido_score >= 30:
            if mask[Action.REAL_ENVIDO.value]:
                return Action.REAL_ENVIDO
            if mask[Action.QUIERO_ENVIDO.value]:
                return Action.QUIERO_ENVIDO

        if envido_score >= 27:
            if mask[Action.ENVIDO.value]:
                return Action.ENVIDO
            if mask[Action.QUIERO_ENVIDO.value]:
                return Action.QUIERO_ENVIDO

        # decline if forced to answer and thresholds are not met
        if mask[Action.NO_QUIERO_ENVIDO.value]:
            return Action.NO_QUIERO_ENVIDO

        # threshold for Truco: holding an Ancho (1) or a 7
        if any(c.number in (1, 7) for c in hand):
            if mask[Action.VALE_CUATRO.value]:
                return Action.VALE_CUATRO
            if mask[Action.RETRUCO.value]:
                return Action.RETRUCO
            if mask[Action.TRUCO.value]:
                return Action.TRUCO
            if mask[Action.QUIERO_TRUCO.value]:
                return Action.QUIERO_TRUCO

        if mask[Action.NO_QUIERO_TRUCO.value]:
            return Action.NO_QUIERO_TRUCO

        return self._random_fallback(mask)

    def _random_fallback(self, mask: list[bool]) -> Action:
        legal_actions = [Action(i) for i, m in enumerate(mask) if m]
        return random.choice(legal_actions) if legal_actions else Action.IR_AL_MAZO
