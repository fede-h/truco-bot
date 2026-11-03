import time
from pathlib import Path
from typing import Any

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.core.actions import Action
from truco_bot.core.match import MatchState, TrucoMatch
from truco_bot.core.state import GameState
from truco_bot.env.masking import get_action_mask
from truco_bot.env.obs import get_observation


def discover_agents(models_dir: Path = Path("models")) -> list[str]:
    baselines = ["random", "heuristic", "equity"]
    if not models_dir.is_dir():
        return baselines
    models = []
    for p in models_dir.iterdir():
        if "_step_" in p.name:
            continue
        if (p.suffix == ".bin" and (models_dir / f"{p.name}.done").is_file()) or (
            p.suffix == ".pt" and p.is_file()
        ):
            models.append(p.stem)
    return baselines + sorted(models)


# ponytail: cache loaded checkpoint tables by path so starting a match doesn't re-read 1.4GB .bin
_AGENT_CACHE: dict[Path, Any] = {}


def _load_cached_cfr(path: Path, seed: int) -> NativeCFRAgent:
    agent = _AGENT_CACHE.get(path)
    if agent is None:
        agent = NativeCFRAgent.from_checkpoint(path, seed=seed)
        _AGENT_CACHE[path] = agent
    return NativeCFRAgent(table=agent.table, seed=seed, capacity=agent.capacity)


def instantiate_agent(name: str, seed: int | None = None) -> Agent:
    low = name.lower()
    agent_seed = seed if seed is not None else 42
    if low == "random":
        return RandomAgent(seed=agent_seed)
    elif low == "heuristic":
        return HeuristicAgent()
    elif low == "equity":
        return EquityAgent()
    elif low == "mccfr_1000m":
        p = Path("models/mccfr_plus_1000M.bin")
        if p.is_file():
            return _load_cached_cfr(p, agent_seed)
        raise ValueError(f"Unknown or unavailable agent: {name}")

    if ".." in name or "/" in name or "\\" in name or "_step_" in name:
        raise ValueError(f"Unknown or unavailable agent: {name}")

    if name in discover_agents():
        bin_p = Path(f"models/{name}.bin")
        if bin_p.is_file():
            return _load_cached_cfr(bin_p, agent_seed)
        pt_p = Path(f"models/{name}.pt")
        if pt_p.is_file():
            from truco_bot.agents.deep_cfr.agent import DeepCFRAgent

            return DeepCFRAgent.from_checkpoint(pt_p, seed=agent_seed)

    raise ValueError(f"Unknown or unavailable agent: {name}")


class MatchSession:
    def __init__(
        self,
        mode: str = "human_vs_bot",
        bot_name: str = "heuristic",
        bot_1: str | None = None,
        target_score: int = 30,
        simulate_delay: bool = True,
        seed: int | None = None,
    ) -> None:
        self.mode = mode
        self.bot_name = bot_name
        self.bot_1_name = bot_1
        self.simulate_delay = simulate_delay
        self.delay_seconds = 0.5 if simulate_delay else 0.0
        self.seed = seed

        self.match = TrucoMatch(target_score=target_score, seed=seed)
        self._event_cursor = 0

        if mode == "human_vs_bot":
            self.bot_agent = instantiate_agent(bot_name, seed=seed)
            self.bot_1_agent = None
            # If bot starts hand (mano == 1)
            self._advance_bot_if_needed()
        elif mode == "bot_vs_bot":
            b0_seed = seed if seed is not None else 42
            b1_seed = (seed + 1) if seed is not None else 43
            self.bot_agent = instantiate_agent(bot_name, seed=b0_seed)
            self.bot_1_agent = instantiate_agent(bot_1 or bot_name, seed=b1_seed)
        else:
            self.bot_agent = None
            self.bot_1_agent = None

    def _get_agent_action(self, agent: Agent, player: int) -> Action:
        hand = self.match.current_hand
        if hand is None:
            raise RuntimeError("Hand is not active.")
        if hasattr(agent, "act_from_state"):
            return agent.act_from_state(hand)
        obs = get_observation(hand, player)
        mask = get_action_mask(hand)
        return agent.act(obs, mask)

    def _advance_bot_if_needed(self) -> None:
        if self.mode != "human_vs_bot" or self.bot_agent is None:
            return
        while not self.is_match_done:
            if self.is_hand_done:
                break
            if self.active_player == 1:
                if self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)
                action = self._get_agent_action(self.bot_agent, player=1)
                self.match.step(action)
            else:
                break

    def step_action(self, action: Action | str) -> MatchState:
        if isinstance(action, str):
            action = Action[action]
        self.match.step(action)
        self._advance_bot_if_needed()
        return self.match.state

    def step_bot(self) -> MatchState:
        if self.mode != "bot_vs_bot":
            return self.match.state

        if self.is_match_done:
            return self.match.state

        if self.is_hand_done:
            self.next_hand()
            if self.is_match_done:
                return self.match.state

        active = self.active_player
        agent = self.bot_agent if active == 0 else self.bot_1_agent
        if agent is None:
            return self.match.state

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        action = self._get_agent_action(agent, player=active)
        self.match.step(action)
        return self.match.state

    def next_hand(self) -> GameState | None:
        hand = self.match.next_hand()
        if self.mode == "human_vs_bot":
            self._advance_bot_if_needed()
        return hand

    def get_view(self, player: int = 0) -> dict[str, Any]:
        return self.match.get_view(player)

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.match.events

    def pop_events(self) -> list[dict[str, Any]]:
        """Return only the events recorded since the previous WebSocket frame."""
        events = self.match.events
        if self._event_cursor > len(events):  # match was reset under us
            self._event_cursor = 0
        new_events = events[self._event_cursor :]
        self._event_cursor = len(events)
        return new_events

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
