"""External Sampling Monte Carlo Counterfactual Regret Minimization (MCCFR) solver."""

import pickle
import random
import sqlite3
from pathlib import Path

from truco_bot.agents.cfr.mccfr_traversal import external_sampling_traverse
from truco_bot.agents.cfr.node import CFRNode
from truco_bot.core.actions import Action
from truco_bot.core.deck import deal
from truco_bot.core.state import create_initial_hand_state


class ExternalSamplingMCCFRSolver:
    """External Sampling MCCFR solver with linear strategy weighting and CFR+ support."""

    def __init__(
        self,
        seed: int | None = 42,
        is_canonical: bool = True,
        cfr_plus: bool = True,
    ) -> None:
        self.nodes: dict[tuple, CFRNode] = {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.is_canonical = is_canonical
        self.cfr_plus = cfr_plus
        self.total_iterations = 0

    def train(self, iterations: int = 1000, log_interval: int | None = None) -> None:
        """Train CFR policy via external sampling over iterations."""
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")

        for _ in range(iterations):
            self.total_iterations += 1
            t = self.total_iterations
            hands, _ = deal(num_players=2, cards_per_player=3, rng=self.rng)
            initial_state = create_initial_hand_state(hands, mano=t % 2)

            external_sampling_traverse(
                state=initial_state,
                update_player=0,
                nodes=self.nodes,
                is_canonical=self.is_canonical,
                rng=self.rng,
                cfr_plus=self.cfr_plus,
                weight=float(t),
            )
            external_sampling_traverse(
                state=initial_state,
                update_player=1,
                nodes=self.nodes,
                is_canonical=self.is_canonical,
                rng=self.rng,
                cfr_plus=self.cfr_plus,
                weight=float(t),
            )
            if log_interval and t % log_interval == 0:
                print(
                    f"[{t:,} deals] Discovered {len(self.nodes):,} canonical infosets",
                    flush=True,
                )

    def export_policy(self) -> dict[tuple, dict[Action, float]]:
        """Export normalized average strategy table across all discovered nodes with accumulated strategy."""
        return {
            k: n.get_average_strategy()
            for k, n in self.nodes.items()
            if sum(n.strategy_sum.values()) > 0
        }

    def export_to_sqlite(self, db_path: str | Path) -> None:
        """Export policy table into SQLite database for persistent storage."""
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("CREATE TABLE IF NOT EXISTS policy (k BLOB PRIMARY KEY, a BLOB)")

        def record_generator():
            for k, n in self.nodes.items():
                if sum(n.strategy_sum.values()) > 0:
                    yield (
                        pickle.dumps(k, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(n.get_average_strategy(), protocol=pickle.HIGHEST_PROTOCOL),
                    )

        conn.executemany("INSERT OR REPLACE INTO policy (k, a) VALUES (?, ?)", record_generator())
        conn.commit()
        conn.close()
