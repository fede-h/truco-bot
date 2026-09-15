"""External Sampling Monte Carlo CFR (MCCFR) traversal engine."""

import random

from truco_bot.agents.cfr.isomorphism import canonical_infoset_key
from truco_bot.agents.cfr.node import CFRNode
from truco_bot.core.actions import Action
from truco_bot.core.state import GameState, get_legal_actions, infoset_key, step


def external_sampling_traverse(
    state: GameState,
    update_player: int,
    nodes: dict[tuple, CFRNode],
    is_canonical: bool = True,
    rng: random.Random | None = None,
    cfr_plus: bool = True,
    weight: float = 1.0,
) -> float:
    """Traverse game tree using External Sampling MCCFR."""
    if rng is None:
        rng = random

    legal_actions = get_legal_actions(state)
    if state.is_hand_done or not legal_actions:
        p0_pts = state.score_p0 + (state.points_won if state.winner == 0 else 0)
        p1_pts = state.score_p1 + (state.points_won if state.winner == 1 else 0)
        return float(p0_pts - p1_pts) if update_player == 0 else float(p1_pts - p0_pts)

    player = state.active_player
    key = canonical_infoset_key(state, player) if is_canonical else infoset_key(state, player)

    node = nodes.get(key)
    if node is None:
        node = CFRNode(actions=legal_actions)
        nodes[key] = node

    if player == update_player:
        strategy = node.get_strategy()
        action_utils: dict[Action, float] = {}
        for a in legal_actions:
            next_state = step(state, a)
            action_utils[a] = external_sampling_traverse(
                next_state,
                update_player=update_player,
                nodes=nodes,
                is_canonical=is_canonical,
                rng=rng,
                cfr_plus=cfr_plus,
                weight=weight,
            )

        node_util = sum(strategy[a] * action_utils[a] for a in legal_actions)
        for a in legal_actions:
            node.apply_regret(a, action_utils[a] - node_util, cfr_plus=cfr_plus)

        return node_util
    else:
        strategy = node.get_strategy()
        weights = [strategy.get(a, 0.0) for a in legal_actions]
        total_w = sum(weights)
        a_star = (
            rng.choices(legal_actions, weights=weights, k=1)[0]
            if total_w > 0
            else rng.choice(legal_actions)
        )

        node.accumulate_strategy(strategy, weight=weight)
        next_state = step(state, a_star)
        return external_sampling_traverse(
            next_state,
            update_player=update_player,
            nodes=nodes,
            is_canonical=is_canonical,
            rng=rng,
            cfr_plus=cfr_plus,
            weight=weight,
        )
