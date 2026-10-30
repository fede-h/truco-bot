from dataclasses import dataclass
from typing import Any

from truco_bot.core.actions import Action
from truco_bot.core.deck import deal
from truco_bot.core.state import GameState, create_initial_hand_state, get_legal_actions
from truco_bot.core.state import step as core_step


@dataclass(slots=True)
class MatchState:
    target_score: int
    match_score_p0: int = 0
    match_score_p1: int = 0
    mano: int = 0
    hand_index: int = 0
    is_match_done: bool = False
    match_winner: int | None = None


class TrucoMatch:
    def __init__(self, target_score: int = 30, seed: int | None = None) -> None:
        self.state = MatchState(target_score=target_score)
        self.seed = seed
        self.current_hand: GameState | None = None
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> MatchState:
        if seed is not None:
            self.seed = seed
        self.state = MatchState(
            target_score=self.state.target_score,
            match_score_p0=0,
            match_score_p1=0,
            mano=0,
            hand_index=0,
            is_match_done=False,
            match_winner=None,
        )
        self._deal_hand()
        return self.state

    def _deal_hand(self) -> None:
        hand_seed = None if self.seed is None else (self.seed + self.state.hand_index * 10007)
        hands, _ = deal(2, 3, seed=hand_seed)
        # ponytail: hand state scores strictly start at (0, 0)
        self.current_hand = create_initial_hand_state(
            hands,
            mano=self.state.mano,
            score_p0=0,
            score_p1=0,
            max_score=self.state.target_score,
        )

    def step(self, action: Action) -> tuple[MatchState, bool]:
        if self.state.is_match_done:
            return self.state, True

        if self.current_hand is None or self.current_hand.is_hand_done:
            raise RuntimeError("Hand is done; call next_hand() before calling step().")

        p0_before = self.current_hand.score_p0
        p1_before = self.current_hand.score_p1

        self.current_hand = core_step(self.current_hand, action)

        delta_p0 = self.current_hand.score_p0 - p0_before
        delta_p1 = self.current_hand.score_p1 - p1_before

        if delta_p0 > 0:
            self.state.match_score_p0 += delta_p0
        if delta_p1 > 0:
            self.state.match_score_p1 += delta_p1

        # "El envido va primero" early match termination
        if self.state.match_score_p0 >= self.state.target_score:
            self.state.is_match_done = True
            self.state.match_winner = 0
            self.current_hand.is_hand_done = True
            return self.state, True

        if self.state.match_score_p1 >= self.state.target_score:
            self.state.is_match_done = True
            self.state.match_winner = 1
            self.current_hand.is_hand_done = True
            return self.state, True

        if self.current_hand.is_hand_done:
            if self.current_hand.winner == 0:
                self.state.match_score_p0 += self.current_hand.points_won
            elif self.current_hand.winner == 1:
                self.state.match_score_p1 += self.current_hand.points_won

            if self.state.match_score_p0 >= self.state.target_score:
                self.state.is_match_done = True
                self.state.match_winner = 0
            elif self.state.match_score_p1 >= self.state.target_score:
                self.state.is_match_done = True
                self.state.match_winner = 1

            return self.state, True

        return self.state, False

    def next_hand(self) -> GameState:
        if self.state.is_match_done or self.current_hand is None:
            return self.current_hand

        self.state.hand_index += 1
        self.state.mano = 1 - self.state.mano
        self._deal_hand()
        return self.current_hand

    def get_view(self, player: int) -> dict[str, Any]:
        if self.current_hand is None:
            raise RuntimeError("Match not initialized.")

        my_cards = [
            {"number": c.number, "suit": c.suit.name} for c in self.current_hand.hands[player]
        ]
        opp_count = len(self.current_hand.hands[1 - player])
        table_cards = [
            {"player": p, "number": c.number, "suit": c.suit.name}
            for p, c in self.current_hand.trick_cards
        ]
        # ponytail: pair consecutive played_cards so completed tricks survive trick_cards resets
        tricks = [
            [
                {"player": p, "number": c.number, "suit": c.suit.name}
                for p, c in self.current_hand.played_cards[i : i + 2]
            ]
            for i in range(0, len(self.current_hand.played_cards), 2)
        ]
        legal_actions = (
            [a.name for a in get_legal_actions(self.current_hand)]
            if self.current_hand.active_player == player and not self.current_hand.is_hand_done
            else []
        )

        return {
            "match_score": (self.state.match_score_p0, self.state.match_score_p1),
            "target_score": self.state.target_score,
            "hand_index": self.state.hand_index,
            "mano": self.state.mano,
            "active_player": self.current_hand.active_player,
            "is_hand_done": self.current_hand.is_hand_done,
            "is_match_done": self.state.is_match_done,
            "match_winner": self.state.match_winner,
            "my_hand": my_cards,
            "opponent_card_count": opp_count,
            "table_cards": table_cards,
            "tricks": tricks,
            "trick_results": list(self.current_hand.trick_results),
            "truco_level": self.current_hand.truco_level,
            "legal_actions": legal_actions,
        }

    @property
    def is_match_done(self) -> bool:
        return self.state.is_match_done

    @property
    def is_hand_done(self) -> bool:
        return self.current_hand.is_hand_done if self.current_hand else True

    @property
    def match_winner(self) -> int | None:
        return self.state.match_winner

    @property
    def active_player(self) -> int:
        if self.current_hand is None:
            return 0
        return self.current_hand.active_player

    def get_legal_actions(self) -> list[Action]:
        if self.current_hand is None or self.current_hand.is_hand_done:
            return []
        return get_legal_actions(self.current_hand)
