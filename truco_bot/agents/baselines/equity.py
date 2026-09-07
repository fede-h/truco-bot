import pickle
from random import choice

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action
from truco_bot.env.obs import ID_TO_CARD
from truco_bot.core.equity import calculate_envido_equity


with open('truco_bot/core/lookup/equity.pkl', 'rb') as file:
        equity_lookup = pickle.load(file)


class EquityAgent(Agent):
    
    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        valids = [i for i, is_legal in enumerate(mask) if is_legal]
        
        if mask[Action.ENVIDO.value]:
            hand = frozenset([ID_TO_CARD[i] for i in obs[:3]])
            is_mano = (obs[10] == obs[11])

            if equity_lookup[hand][is_mano] > 0.65:
                return Action.ENVIDO
            else: return Action(choice(valids))
        elif mask[Action.QUIERO_ENVIDO.value]:
            hand = frozenset([ID_TO_CARD[i] for i in obs[:3]])
            is_mano = (obs[10] == obs[11])

            if equity_lookup[hand][is_mano] > 0.65:
                return Action.QUIERO_ENVIDO
            else: return Action.NO_QUIERO_ENVIDO
        else:
            return Action(choice(valids))