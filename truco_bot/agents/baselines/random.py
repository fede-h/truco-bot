from random import choice

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action

class RandomPlayer(Agent):
    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        valids = [i for i, is_legal in enumerate(mask) if is_legal]
        return Action(choice(valids))