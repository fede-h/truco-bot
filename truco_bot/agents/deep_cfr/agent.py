"""Deep CFR Agent evaluating neural policy with action masking."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from truco_engine import BitboardState

from truco_bot.agents.base import Agent
from truco_bot.agents.cfr.native_agent import gamestate_to_bitboard
from truco_bot.agents.deep_cfr.encoder import STATE_FEATURE_DIM, encode_state
from truco_bot.agents.deep_cfr.network import PolicyNetwork
from truco_bot.core.actions import Action
from truco_bot.core.state import GameState, get_legal_actions


def _encode_obs(obs: tuple[int, ...] | list[int]) -> torch.Tensor:
    """Fast observation vector to 156-dim feature tensor encoder."""
    tensor = torch.zeros(STATE_FEATURE_DIM, dtype=torch.float32)
    c0, c1, c2, tc0, tc1, score_p0, score_p1, cur_trick, truco_lvl, env_res, active_p, mano = obs[
        :12
    ]

    # 0..39: hand cards (obs uses 1-based card IDs 1..40)
    for c in (c0, c1, c2):
        if 1 <= c <= 40:
            tensor[c - 1] = 1.0

    # 40..119: trick cards
    leader_is_self = (tc0 == 0) or (tc1 > 0 and active_p == mano)
    if leader_is_self:
        self_c, opp_c = tc0, tc1
    else:
        opp_c, self_c = tc0, tc1

    if 1 <= self_c <= 40:
        tensor[40 + self_c - 1] = 1.0
    if 1 <= opp_c <= 40:
        tensor[80 + opp_c - 1] = 1.0

    # 129..134: trick structure
    tensor[129 + min(cur_trick, 2)] = 1.0
    if leader_is_self:
        tensor[132] = 1.0
    if mano == active_p:
        tensor[133] = 1.0
    tensor[134] = 1.0

    # 135: envido resolved
    if env_res:
        tensor[135] = 1.0

    # 145..153: truco level
    t_l = max(1, min(truco_lvl, 4))
    tensor[145 + (t_l - 1)] = 1.0
    tensor[153] = t_l / 4.0

    # 154..155: scores
    max_s = 30.0
    self_score = score_p0 if active_p == 0 else score_p1
    opp_score = score_p1 if active_p == 0 else score_p0
    tensor[154] = self_score / max_s
    tensor[155] = opp_score / max_s

    return tensor


class DeepCFRAgent(Agent):
    """Deep CFR Agent using ResMLP policy network with action masking."""

    def __init__(
        self,
        policy_net: PolicyNetwork | None = None,
        model_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        if policy_net is not None:
            self.policy_net = policy_net
        elif model_path is not None:
            self.policy_net = PolicyNetwork()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.policy_net.load_state_dict(state_dict)
        else:
            self.policy_net = PolicyNetwork()

        self.policy_net.eval()

    def reset(self) -> None:
        """Reset internal hand state tracker."""
        # ponytail: stateless feedforward agent requires no history cleanup

    def save(self, path: str | Path) -> None:
        """Save policy network weights to checkpoint file."""
        torch.save(self.policy_net.state_dict(), str(path))

    @classmethod
    def from_checkpoint(cls, path: str | Path, seed: int | None = None) -> DeepCFRAgent:
        """Load DeepCFRAgent from a saved .pt checkpoint."""
        return cls(model_path=path, seed=seed)

    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        """Choose an action given observation tuple and 26-dim boolean action mask."""
        legal_actions = [Action(i) for i, is_legal in enumerate(mask) if is_legal]
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        feat = _encode_obs(obs)
        return self._evaluate_and_sample(feat, legal_actions)

    def act_from_state(self, state: BitboardState | GameState) -> Action:
        """Choose an action directly from BitboardState or GameState."""
        if isinstance(state, BitboardState):
            mask_u32 = state.legal_actions_mask()
            if mask_u32 == 0:
                return Action.IR_AL_MAZO
            legal_actions = [Action(a) for a in range(26) if (mask_u32 & (1 << a))]
            bstate = state
        else:
            legal_actions = get_legal_actions(state)
            if not legal_actions:
                return Action.IR_AL_MAZO
            bstate = gamestate_to_bitboard(state)

        if len(legal_actions) == 1:
            return legal_actions[0]

        feat = encode_state(bstate)
        return self._evaluate_and_sample(feat, legal_actions)

    def _evaluate_and_sample(self, feat: torch.Tensor, legal_actions: list[Action]) -> Action:
        """Evaluate policy network on features and sample among legal actions."""
        k = min(len(legal_actions), self.policy_net.num_actions)
        net_mask = torch.zeros(self.policy_net.num_actions, dtype=torch.bool)
        net_mask[:k] = True

        with torch.no_grad():
            probs = self.policy_net(feat, mask=net_mask)
            probs_list = probs[:k].tolist()

        total = sum(probs_list)
        if total <= 1e-8:
            return self.rng.choice(legal_actions)

        # ponytail: sample legal action proportional to network softmax distribution
        return self.rng.choices(legal_actions[:k], weights=probs_list, k=1)[0]
