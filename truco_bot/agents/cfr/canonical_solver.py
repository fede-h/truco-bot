"""Canonical Isomorphic Full-Tree CFR Solver"""

from truco_bot.agents.cfr.isomorphism import generate_canonical_deals
from truco_bot.agents.cfr.node import CFRNode
from truco_bot.agents.cfr.traversal import cfr_traverse
from truco_bot.core.actions import Action
from truco_bot.core.state import create_initial_hand_state


class CanonicalCFRSolver:
    """Solves subgames via Canonical Isomorphic"""

    def __init__(self) -> None:
        self.nodes: dict[tuple, CFRNode] = {}

    def train(
        self,
        iterations: int | None = 1000,
        mano: int = 0,
        max_deals: int | None = None,
    ) -> None:
        """Train CFR policy across canonical deal partitions."""
        limit = max_deals if max_deals is not None else iterations
        for h0, h1 in generate_canonical_deals(limit=limit):
            initial_state = create_initial_hand_state([h0, h1], mano=mano)
            cfr_traverse(
                state=initial_state,
                p0_reach=1.0,
                p1_reach=1.0,
                nodes=self.nodes,
                is_canonical=True,
            )

    def export_policy(self) -> dict[tuple, dict[Action, float]]:
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}
