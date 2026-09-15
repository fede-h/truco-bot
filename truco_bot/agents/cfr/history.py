"""Stateful history tracker for reconstructing trick play and envido state."""

from dataclasses import dataclass, field

from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.rules import compare_cards
from truco_bot.env.obs import ID_TO_CARD


@dataclass(slots=True)
class HandHistoryTracker:
    """Tracks played cards, trick results, and envido chain across observations."""

    played_cards: list[tuple[int, Card]] = field(default_factory=list)
    trick_results: list[int] = field(default_factory=list)
    envido_chain: list[Action] = field(default_factory=list)
    current_trick: int = 0
    hands: dict[int, list[Card]] = field(default_factory=lambda: {0: [], 1: []})
    last_tc0: Card | None = None
    last_trick_leader: int = 0
    _last_scores: tuple[int, int] = (0, 0)

    def reset_hand(self) -> None:
        """Reset state for a new deal."""
        self.played_cards.clear()
        self.trick_results.clear()
        self.envido_chain.clear()
        self.current_trick = 0
        self.hands = {0: [], 1: []}
        self.last_tc0 = None
        self.last_trick_leader = 0
        self._last_scores = (0, 0)

    reset = reset_hand

    def update(self, obs: tuple[int, ...], mask: list[bool]) -> None:
        """Update tracked state given observation and action mask."""
        if self.played_cards and obs[7] == 0 and obs[3] == 0:
            self.reset_hand()

        active = obs[10]
        hand_cards = [ID_TO_CARD[cid] for cid in obs[:3] if cid != 0]
        self.hands[active] = hand_cards

        pending_envido = (
            bool(mask[Action.QUIERO_ENVIDO.value])
            if Action.QUIERO_ENVIDO.value < len(mask)
            else False
        )
        if pending_envido and (
            not self.envido_chain
            or self.envido_chain[-1] in (Action.QUIERO_ENVIDO, Action.NO_QUIERO_ENVIDO)
        ):
            if Action.ENVIDO.value < len(mask) and mask[Action.ENVIDO.value]:
                self.envido_chain.append(Action.ENVIDO)
            elif Action.REAL_ENVIDO.value < len(mask) and mask[Action.REAL_ENVIDO.value]:
                self.envido_chain.append(Action.REAL_ENVIDO)
            elif Action.FALTA_ENVIDO.value < len(mask) and mask[Action.FALTA_ENVIDO.value]:
                self.envido_chain.append(Action.FALTA_ENVIDO)

        envido_resolved = bool(obs[9])
        if (
            envido_resolved
            and self.envido_chain
            and self.envido_chain[-1] not in (Action.QUIERO_ENVIDO, Action.NO_QUIERO_ENVIDO)
        ):
            score_diff = (obs[5] + obs[6]) - (self._last_scores[0] + self._last_scores[1])
            if score_diff == 1 and self.envido_chain == [Action.ENVIDO]:
                self.envido_chain.append(Action.NO_QUIERO_ENVIDO)
            else:
                self.envido_chain.append(Action.QUIERO_ENVIDO)

        self._last_scores = (obs[5], obs[6])

        tc0_id = obs[3]
        tc1_id = obs[4]
        current_trick = obs[7]
        self.current_trick = current_trick

        if tc0_id != 0:
            tc0_card = ID_TO_CARD[tc0_id]
            leader = 1 - active
            follower = active

            if len(self.played_cards) < (current_trick + 1) * 2:
                if (
                    not self.played_cards
                    or self.played_cards[-1] != (leader, tc0_card)
                ) and len(self.played_cards) % 2 == 0:
                    self.played_cards.append((leader, tc0_card))
                    self.last_tc0 = tc0_card
                    self.last_trick_leader = leader

                follower_card = None
                if tc1_id != 0:
                    follower_card = ID_TO_CARD[tc1_id]
                elif hand_cards and (
                    Action.PLAY_CARD_0.value < len(mask)
                    and mask[Action.PLAY_CARD_0.value]
                ):
                    follower_card = hand_cards[0]

                if follower_card is not None and len(self.played_cards) % 2 == 1:
                    self.played_cards.append((follower, follower_card))
                    p0_c = tc0_card if leader == 0 else follower_card
                    p1_c = follower_card if leader == 0 else tc0_card
                    res = compare_cards(p0_c, p1_c)
                    self.trick_results.append(res)
