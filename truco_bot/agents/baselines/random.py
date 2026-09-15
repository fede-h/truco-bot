import random

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        valids = [i for i, is_legal in enumerate(mask) if is_legal]
        return Action(self.rng.choice(valids))