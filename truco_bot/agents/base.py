from abc import ABC, abstractmethod

from truco_bot.core.actions import Action


class Agent(ABC):
    @abstractmethod
    def act(self, obs: tuple[int, ...], mask: list[bool]) -> Action:
        """Choose an action given the current observation and legal action mask."""

    def reset(self) -> None:
        """Reset internal agent state between hands/matches."""
        pass
