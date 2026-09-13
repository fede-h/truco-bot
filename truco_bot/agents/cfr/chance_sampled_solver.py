"""Chance-Sampled Vanilla CFR Solver"""

import random

from truco_bot.agents.cfr.node import CFRNode
from truco_bot.agents.cfr.traversal import cfr_traverse
from truco_bot.core.actions import Action
from truco_bot.core.deck import deal
from truco_bot.core.state import create_initial_hand_state


class ChanceSampledCFRSolver:

    def __init__(self, seed: int | None = None) -> None:
        self.nodes: dict[tuple, CFRNode] = {}
        self.rng = random.Random(seed)

    def train(self, iterations: int = 1000, mano: int = 0) -> None:
        """Train CFR policy by sampling real deals and traversing game trees."""
        for _ in range(iterations):
            hands, _ = deal(2, 3, seed=self.rng.randint(0, 1_000_000_000))
            initial_state = create_initial_hand_state(hands, mano=mano)
            cfr_traverse(
                state=initial_state,
                p0_reach=1.0,
                p1_reach=1.0,
                nodes=self.nodes,
                is_canonical=False,
            )

    def export_policy(self) -> dict[tuple, dict[Action, float]]:
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}
