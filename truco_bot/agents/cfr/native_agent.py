"""Native Compiled Shared-Memory CFR Agent using truco_engine."""

import random
from collections.abc import Sequence
from pathlib import Path

import truco_engine
from truco_engine import BitboardState, SharedPolicyTable, get_policy_distribution
from truco_bot.agents.base import Agent
from truco_bot.agents.cfr.fallback import EquityFallback
from truco_bot.agents.cfr.history import HandHistoryTracker
from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.state import GameState, get_legal_actions
from truco_bot.env.obs import ID_TO_CARD

_NUM_IDX_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 11: 8, 12: 9}


def card_to_id(card: Card) -> int:
    """Convert Card object to native engine u8 index (0..39)."""
    suit_offset = (card.suit.value - 1) * 10
    return suit_offset + _NUM_IDX_MAP[card.number]


def gamestate_to_bitboard(state: GameState) -> BitboardState:
    """Convert Python GameState into 32-byte native BitboardState."""
    hands: list[list[int]] = []
    for p in range(2):
        p_hand = [card_to_id(c) for c in state.hands[p]]
        while len(p_hand) < 3:
            p_hand.append(255)
        hands.append(p_hand)

    trick_cards = [card_to_id(c) for _, c in state.trick_cards[:2]]
    while len(trick_cards) < 2:
        trick_cards.append(255)

    trick_results = list(state.trick_results[:3])
    while len(trick_results) < 3:
        trick_results.append(127)

    envido_chain = [a.value if hasattr(a, "value") else int(a) for a in state.envido_chain[:6]]

    flags = 0
    if state.envido_resolved:
        flags |= 1 << 0
    if state.pending_envido_response:
        flags |= 1 << 1
    if state.pending_truco_response:
        flags |= 1 << 2
    if state.is_hand_done:
        flags |= 1 << 3

    pending_envido_from = state.pending_envido_from if state.pending_envido_from is not None else 255
    truco_caller = state.truco_caller if state.truco_caller is not None else 255
    pending_truco_from = state.pending_truco_from if state.pending_truco_from is not None else 255
    winner = state.winner if state.winner is not None else 255

    return BitboardState.from_components(
        hands=hands,
        trick_cards=trick_cards,
        trick_results=trick_results,
        trick_leader=state.trick_leader,
        current_trick=min(state.current_trick, 2),
        active_player=state.active_player,
        mano=state.mano,
        score_p0=min(state.score_p0, state.max_score),
        score_p1=min(state.score_p1, state.max_score),
        max_score=state.max_score,
        envido_chain=envido_chain,
        pending_envido_from=pending_envido_from,
        truco_level=state.truco_level,
        truco_caller=truco_caller,
        pending_truco_from=pending_truco_from,
        flags=flags,
        winner=winner,
        points_won=state.points_won,
    )


class NativeCFRAgent(Agent):
    """High-performance CFR agent evaluating policies directly from native SharedPolicyTable."""

    def __init__(
        self,
        table: SharedPolicyTable | None = None,
        seed: int | None = None,
        model_path: str | Path | None = None,
        capacity: int = 67_108_864,
    ) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.tracker = HandHistoryTracker()
        self.fallback = EquityFallback()
        if table is not None:
            self.table = table
        elif model_path is not None:
            self.table = SharedPolicyTable(capacity)
            self.table.load_from_file(str(model_path))
        else:
            self.table = SharedPolicyTable(capacity)

    def reset(self) -> None:
        """Reset internal history tracker for a new hand."""
        self.tracker.reset_hand()

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str | Path,
        seed: int | None = None,
        capacity: int = 67_108_864,
    ) -> "NativeCFRAgent":
        """Load native agent directly from binary .bin checkpoint."""
        return cls(model_path=filepath, seed=seed, capacity=capacity)

    def act_from_state(self, state: GameState) -> Action:
        """Choose action given exact GameState using compiled native table lookup."""
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        bstate = gamestate_to_bitboard(state)
        policy_dist = get_policy_distribution(bstate, self.table)

        if policy_dist:
            probs = [p for _, p in policy_dist]
            if max(probs) - min(probs) >= 1e-5:
                # Sample action from native probability distribution
                actions = [Action(a) for a, _ in policy_dist]
                return self.rng.choices(actions, weights=probs, k=1)[0]

        # Delegate off-tree or unvisited states to domain-informed EquityFallback
        return self.fallback.decide_state(state, legal_actions)

    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        """Choose action from observation and mask, consulting fallback tracker."""
        self.tracker.update(obs, mask)
        legal_actions = [Action(i) for i, is_legal in enumerate(mask) if is_legal]
        if not legal_actions:
            return Action.IR_AL_MAZO
        if len(legal_actions) == 1:
            return legal_actions[0]

        return self.fallback.act(obs, mask)


