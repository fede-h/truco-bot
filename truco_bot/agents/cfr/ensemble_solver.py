import random
from collections.abc import Sequence

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.cfr.isomorphism import canonical_infoset_key
from truco_bot.agents.cfr.mccfr_solver import ExternalSamplingMCCFRSolver
from truco_bot.agents.cfr.mccfr_traversal import external_sampling_traverse
from truco_bot.agents.cfr.node import CFRNode
from truco_bot.core.actions import Action
from truco_bot.core.deck import deal
from truco_bot.core.state import (
    GameState,
    create_initial_hand_state,
    get_legal_actions,
    infoset_key,
    step,
)
from truco_bot.env.obs import get_observation


def _traverse_against_opponent(
    state: GameState,
    update_player: int,
    opponent: Agent,
    nodes: dict[tuple, CFRNode],
    is_canonical: bool = True,
    rng: random.Random | None = None,
    cfr_plus: bool = True,
    weight: float = 1.0,
) -> float:
    """Traverse game tree with Player 0 as learner and Player 1 actions sampled from an external agent."""
    if rng is None:
        rng = random

    legal_actions = get_legal_actions(state)
    if state.is_hand_done or not legal_actions:
        p0_pts = state.score_p0 + (state.points_won if state.winner == 0 else 0)
        p1_pts = state.score_p1 + (state.points_won if state.winner == 1 else 0)
        return float(p0_pts - p1_pts) if update_player == 0 else float(p1_pts - p0_pts)

    player = state.active_player
    if player == update_player:
        key = canonical_infoset_key(state, player) if is_canonical else infoset_key(state, player)
        node = nodes.get(key)
        if node is None:
            node = CFRNode(actions=legal_actions)
            nodes[key] = node

        strategy = node.get_strategy()
        action_utils: dict[Action, float] = {}
        for a in legal_actions:
            next_state = step(state, a)
            action_utils[a] = _traverse_against_opponent(
                next_state,
                update_player=update_player,
                opponent=opponent,
                nodes=nodes,
                is_canonical=is_canonical,
                rng=rng,
                cfr_plus=cfr_plus,
                weight=weight,
            )

        node_util = sum(strategy[a] * action_utils[a] for a in legal_actions)
        for a in legal_actions:
            node.apply_regret(a, action_utils[a] - node_util, cfr_plus=cfr_plus)
        node.accumulate_strategy(strategy, weight=weight)
        return node_util
    else:
        mask = [False] * 26
        for a in legal_actions:
            mask[a.value] = True
        obs = get_observation(state, player)

        if hasattr(opponent, "act_from_state"):
            a_star = opponent.act_from_state(state)
        else:
            a_star = opponent.act(obs, mask)

        if a_star not in legal_actions:
            a_star = rng.choice(legal_actions)

        next_state = step(state, a_star)
        return _traverse_against_opponent(
            next_state,
            update_player=update_player,
            opponent=opponent,
            nodes=nodes,
            is_canonical=is_canonical,
            rng=rng,
            cfr_plus=cfr_plus,
            weight=weight,
        )


class EnsembleMCCFRSolver(ExternalSamplingMCCFRSolver):
    """Ensemble MCCFR solver with self-play and opponent pool sampling."""

    __slots__ = ("opponent_pool", "self_play_ratio")

    def __init__(
        self,
        seed: int | None = 42,
        opponent_pool: Sequence[Agent] | None = None,
        self_play_ratio: float = 0.8,
        is_canonical: bool = True,
        cfr_plus: bool = True,
    ) -> None:
        super().__init__(seed=seed, is_canonical=is_canonical, cfr_plus=cfr_plus)
        self.self_play_ratio = self_play_ratio
        if opponent_pool is None:
            self.opponent_pool: list[Agent] = [HeuristicAgent(), EquityAgent()]
        else:
            self.opponent_pool = list(opponent_pool)

    def train(self, iterations: int = 1000, log_interval: int | None = None) -> None:
        """Train CFR policy via mixed self-play and pool opponent sampling."""
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")

        for _ in range(iterations):
            self.total_iterations += 1
            t = self.total_iterations
            hands, _ = deal(num_players=2, cards_per_player=3, rng=self.rng)
            initial_state = create_initial_hand_state(hands, mano=t % 2)

            if self.rng.random() < self.self_play_ratio:
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
            else:
                opp = self.rng.choice(self.opponent_pool)
                learner = t % 2
                _traverse_against_opponent(
                    state=initial_state,
                    update_player=learner,
                    opponent=opp,
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


__all__ = ["EnsembleMCCFRSolver"]
