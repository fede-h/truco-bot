from truco_bot.core.actions import Action
from truco_bot.core.match import MatchState, TrucoMatch
from truco_bot.core.state import GameState
from truco_bot.env.masking import get_action_mask
from truco_bot.env.obs import get_observation


class TrucoMatchEnv:
    def __init__(self, target_score: int = 30, seed: int | None = None) -> None:
        self.match = TrucoMatch(target_score=target_score, seed=seed)

    def reset(self, seed: int | None = None) -> MatchState:
        return self.match.reset(seed=seed)

    def step(self, action: Action) -> tuple[MatchState, int, bool, list[bool]]:
        match_state, hand_done = self.match.step(action)
        points_won = self.match.current_hand.points_won if self.match.current_hand else 0
        mask = (
            get_action_mask(self.match.current_hand)
            if self.match.current_hand and not self.match.current_hand.is_hand_done
            else [False] * 26
        )
        return match_state, points_won, hand_done, mask

    def next_hand(self) -> GameState:
        return self.match.next_hand()

    def get_obs(self, player: int) -> tuple[int, ...]:
        if self.match.current_hand is None:
            raise RuntimeError("Environment must be reset before calling get_obs().")
        return get_observation(self.match.current_hand, player)

    def get_action_mask(self) -> list[bool]:
        if self.match.current_hand is None:
            raise RuntimeError("Environment must be reset before calling get_action_mask().")
        return get_action_mask(self.match.current_hand)

    @property
    def is_match_done(self) -> bool:
        return self.match.is_match_done

    @property
    def is_hand_done(self) -> bool:
        return self.match.is_hand_done

    @property
    def match_winner(self) -> int | None:
        return self.match.match_winner

    @property
    def active_player(self) -> int:
        return self.match.active_player

    @property
    def hand_state(self) -> GameState:
        if self.match.current_hand is None:
            raise RuntimeError("Environment has no active hand.")
        return self.match.current_hand

    @property
    def match_score_p0(self) -> int:
        return self.match.state.match_score_p0

    @property
    def match_score_p1(self) -> int:
        return self.match.state.match_score_p1
