import pickle
from random import choice

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action
from truco_bot.env.obs import ID_TO_CARD

with open('truco_bot/core/lookup/equity.pkl', 'rb') as file:
        equity_lookup = pickle.load(file)


class EquityAgent(Agent):
    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        valids = [i for i, is_legal in enumerate(mask) if is_legal]
        card_ids = [i for i in obs[:3] if i in ID_TO_CARD]
        # If player already led a card on trick 0, recover it from table (obs[3])
        if len(card_ids) < 3 and obs[3] in ID_TO_CARD:
            card_ids.append(obs[3])

        if len(card_ids) == 3:
            hand = frozenset([ID_TO_CARD[i] for i in card_ids])
            is_mano = obs[10] == obs[11]

            if mask[Action.ENVIDO.value]:
                if equity_lookup[hand][is_mano] > 0.65:
                    return Action.ENVIDO
                else:
                    return Action(choice(valids))
            elif mask[Action.QUIERO_ENVIDO.value]:
                if equity_lookup[hand][is_mano] > 0.65:
                    return Action.QUIERO_ENVIDO
                else:
                    return Action.NO_QUIERO_ENVIDO

        return Action(choice(valids))