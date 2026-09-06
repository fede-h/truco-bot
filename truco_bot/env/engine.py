"""Fast step() / reset() runner for the Truco environment."""

# ponytail: minimal hand simulation environment
from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.deck import deal
from truco_bot.core.state import GameState, create_initial_hand_state
from truco_bot.core.state import step as core_step
from truco_bot.env.masking import get_action_mask
from truco_bot.env.obs import get_observation


class TrucoHandEnv:
    def __init__(self) -> None:
        self.state: GameState | None = None

    def reset(
        self, seed: int | None = None, hands: list[list[Card]] | None = None
    ) -> tuple[GameState, list[bool]]:
        if hands is None:
            hands, _ = deal(2, 3, seed=seed)
        self.state = create_initial_hand_state(hands, mano=0)
        return self.state, get_action_mask(self.state)

    def step(self, action: Action) -> tuple[GameState, int, bool, list[bool]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        self.state = core_step(self.state, action)
        return (
            self.state,
            self.state.points_won,
            self.state.is_hand_done,
            get_action_mask(self.state),
        )

    def get_obs(self, player: int) -> tuple[int, ...]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling get_obs().")
        return get_observation(self.state, player)

    def get_mask(self) -> list[bool]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling get_mask().")
        return get_action_mask(self.state)
