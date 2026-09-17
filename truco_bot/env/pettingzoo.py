"""PettingZoo AEC environment for Truco."""

from typing import Any, ClassVar
from pettingzoo import AECEnv

from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.deck import deal
from truco_bot.core.state import GameState, create_initial_hand_state
from truco_bot.core.state import step as core_step
from truco_bot.env.masking import get_action_mask
from truco_bot.env.obs import get_observation


class TrucoAECEnv(AECEnv):
    """PettingZoo-compatible AEC Environment for 2-player Argentine Truco"""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"]}
    possible_agents: ClassVar[list[str]] = ["player_0", "player_1"]

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        self.agent_selection: str = "player_0"
        self.rewards: dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.terminations: dict[str, bool] = {a: False for a in self.possible_agents}
        self.truncations: dict[str, bool] = {a: False for a in self.possible_agents}
        self.infos: dict[str, dict[str, Any]] = {a: {} for a in self.possible_agents}
        self._cumulative_rewards: dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.render_mode = render_mode
        self.state: GameState | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        hands: list[list[Card]] | None = None,
    ) -> None:
        mano = options.get("mano", 0) if isinstance(options, dict) else 0
        if hands is None:
            hands_dealt, _ = deal(2, 3, seed=seed)
        else:
            hands_dealt = hands

        self.state = create_initial_hand_state(hands_dealt, mano=mano)
        self.agents = self.possible_agents[:]
        self.agent_selection = f"player_{self.state.mano}"
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

    def observe(self, agent: str) -> dict[str, Any]:
        if agent not in self.possible_agents:
            raise KeyError(f"Agent '{agent}' not in possible_agents: {self.possible_agents}")
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling observe().")

        player_idx = 0 if agent == "player_0" else 1
        obs = get_observation(self.state, player_idx)
        # action mask is only available to the currently active agent
        mask = get_action_mask(self.state) if agent == self.agent_selection else [False] * 26
        return {"observation": obs, "action_mask": mask}

    def step(self, action: int | Action) -> None:
        if self.state is None or self.state.is_hand_done or any(self.terminations.values()):
            raise RuntimeError("Cannot step in a terminated or uninitialized environment.")

        mask = get_action_mask(self.state)
        try:
            action_idx = int(action)
            if action_idx < 0 or action_idx >= len(mask) or not mask[action_idx]:
                raise ValueError(f"Illegal action {action} for agent {self.agent_selection}")
        except (TypeError, IndexError):
            raise ValueError(f"Invalid action {action}")


        self.state = core_step(self.state, Action(action_idx))
        self.agent_selection = f"player_{self.state.active_player}"

        if self.state.is_hand_done:
            self.terminations = {a: True for a in self.possible_agents}
            delta = float(self.state.score_p0 - self.state.score_p1)
            self.rewards = {"player_0": delta, "player_1": -delta}
        else:
            self.rewards = {a: 0.0 for a in self.possible_agents}

    def render(self) -> str:

        if self.state is None:
            return "Uninitialized"
        info = (
            f"Truco: P0={self.state.score_p0} P1={self.state.score_p1} | "
            f"Active={self.agent_selection} | Done={self.state.is_hand_done}"
        )
        if self.render_mode == "human":
            print(info)
        return info

    def last(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # ponytail: standard AEC last() helper
        agent = self.agent_selection
        return (
            self.observe(agent),
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )
