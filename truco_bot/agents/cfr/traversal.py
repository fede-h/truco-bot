from truco_bot.agents.cfr.isomorphism import canonical_infoset_key
from truco_bot.agents.cfr.node import CFRNode
from truco_bot.core.actions import Action
from truco_bot.core.state import GameState, get_legal_actions, infoset_key, step


def cfr_traverse(
    state: GameState,
    p0_reach: float,
    p1_reach: float,
    nodes: dict[tuple, CFRNode],
    is_canonical: bool = False,
) -> tuple[float, float]:
    """Traverse the game tree using Vanilla CFR
    Returns:
        (u0, u1): Expected utility tuple for player 0 and player 1.
    """
    legal_actions = get_legal_actions(state)
    if state.is_hand_done or not legal_actions:
        p0_pts = state.score_p0 + (state.points_won if state.winner == 0 else 0)
        p1_pts = state.score_p1 + (state.points_won if state.winner == 1 else 0)
        u0 = float(p0_pts - p1_pts)
        return u0, -u0

    player = state.active_player
    key = canonical_infoset_key(state, player) if is_canonical else infoset_key(state, player)

    node = nodes.get(key)
    if node is None:
        node = CFRNode(actions=legal_actions)
        nodes[key] = node

    own_reach = p0_reach if player == 0 else p1_reach
    opp_reach = p1_reach if player == 0 else p0_reach

    strategy = node.get_strategy(realization_weight=own_reach)

    util_p0 = 0.0
    util_p1 = 0.0
    action_utils: dict[Action, float] = {}

    for action in legal_actions:
        prob = strategy[action]
        if player == 0:
            next_p0 = p0_reach * prob
            next_p1 = p1_reach
        else:
            next_p0 = p0_reach
            next_p1 = p1_reach * prob

        next_state = step(state, action)
        sub_u0, sub_u1 = cfr_traverse(
            next_state, next_p0, next_p1, nodes, is_canonical=is_canonical
        )

        action_util = sub_u0 if player == 0 else sub_u1
        action_utils[action] = action_util

        util_p0 += prob * sub_u0
        util_p1 += prob * sub_u1

    node_util = util_p0 if player == 0 else util_p1
    for action in legal_actions:
        regret = action_utils[action] - node_util
        node.regret_sum[action] = (
            node.regret_sum.get(action, 0.0) + opp_reach * regret
        )

    return util_p0, util_p1
