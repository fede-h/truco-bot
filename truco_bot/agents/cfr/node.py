"""CFRNode data structure with regret matching and strategy accumulation"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from truco_bot.core.actions import Action


@dataclass(slots=True)
class CFRNode:
    actions: tuple[Action, ...] = field(default_factory=tuple)
    regret_sum: dict[Action, float] = field(default_factory=dict)
    strategy_sum: dict[Action, float] = field(default_factory=dict)

    def __init__(
        self,
        actions: Sequence[Action] | None = None,
        regret_sum: dict[Action, float] | None = None,
        strategy_sum: dict[Action, float] | None = None,
    ) -> None:
        if actions is not None:
            self.actions = tuple(actions)
        elif regret_sum is not None:
            self.actions = tuple(regret_sum.keys())
        elif strategy_sum is not None:
            self.actions = tuple(strategy_sum.keys())
        else:
            self.actions = ()

        self.regret_sum = (
            regret_sum
            if regret_sum is not None
            else {a: 0.0 for a in self.actions}
        )
        self.strategy_sum = (
            strategy_sum
            if strategy_sum is not None
            else {a: 0.0 for a in self.actions}
        )

    def apply_regret(self, action: Action, regret: float, cfr_plus: bool = True) -> None:
        """Accumulate instantaneous regret, optionally floored at zero for CFR+."""
        current = self.regret_sum.get(action, 0.0) + regret
        self.regret_sum[action] = max(current, 0.0) if cfr_plus else current

    def accumulate_strategy(self, strategy: dict[Action, float], weight: float = 1.0) -> None:
        """Accumulate strategy distribution weighted by iteration or realization weight."""
        for action, prob in strategy.items():
            self.strategy_sum[action] = self.strategy_sum.get(action, 0.0) + weight * prob

    def get_strategy(self, realization_weight: float = 0.0) -> dict[Action, float]:
        """Compute current strategy via regret matching and optionally accumulate strategy sum."""
        normalizing_sum = 0.0
        strategy: dict[Action, float] = {}

        for action in self.actions:
            pos_regret = max(self.regret_sum.get(action, 0.0), 0.0)
            strategy[action] = pos_regret
            normalizing_sum += pos_regret

        num_actions = len(self.actions)
        if normalizing_sum > 0.0:
            for action in self.actions:
                strategy[action] /= normalizing_sum
        elif num_actions > 0:
            uniform = 1.0 / num_actions
            for action in self.actions:
                strategy[action] = uniform

        if realization_weight > 0.0:
            self.accumulate_strategy(strategy, weight=realization_weight)

        return strategy

    def get_average_strategy(self) -> dict[Action, float]:
        """Compute the average strategy accumulated across iterations."""
        normalizing_sum = sum(self.strategy_sum.get(a, 0.0) for a in self.actions)
        num_actions = len(self.actions)
        avg_strat: dict[Action, float] = {}

        if normalizing_sum > 0.0:
            for action in self.actions:
                avg_strat[action] = self.strategy_sum.get(action, 0.0) / normalizing_sum
        elif num_actions > 0:
            uniform = 1.0 / num_actions
            for action in self.actions:
                avg_strat[action] = uniform

        return avg_strat
