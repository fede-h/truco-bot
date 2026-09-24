"""Showdown runner for head-to-head bot evaluation."""

import random
import time
from collections.abc import Callable
from typing import Any

from truco_bot.agents.base import Agent
from truco_bot.core.actions import Action
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.config import ShowdownConfig

_TRUCO_CALLS = {Action.TRUCO, Action.RETRUCO, Action.VALE_CUATRO}


def _seed_agent(agent: Any, s: int) -> None:

    if hasattr(agent, "rng"):
        agent.rng = random.Random(s)
    elif hasattr(agent, "seed") and callable(agent.seed):
        agent.seed(s)


class ShowdownRunner:
    """Runs head-to-head duplicate matches between a candidate and benchmark opponents"""

    def __init__(self, agent_loader: Callable[..., Agent] | None = None) -> None:
        if agent_loader is None:
            from truco_bot.eval.benchmark import load_agent

            self.agent_loader = load_agent
        else:
            self.agent_loader = agent_loader

    def _load(self, name: str) -> Agent:
        try:
            return self.agent_loader(name)
        except TypeError:
            return self.agent_loader(name, None)

    def _run_single_match(
        self,
        p0_agent: Agent,
        p1_agent: Agent,
        env: TrucoHandEnv,
        seed: int,
        candidate_role: int,
        track_bluff: bool,
        stats: dict[str, float],
    ) -> int:
        env.reset(seed=seed)
        p0_agent.reset()
        p1_agent.reset()

        while not env.state.is_hand_done:
            active = env.state.active_player
            agent = p0_agent if active == 0 else p1_agent

            t0 = time.perf_counter() if active == candidate_role else 0.0
            action = (
                agent.act_from_state(env.state)
                if hasattr(agent, "act_from_state")
                else agent.act(env.get_obs(active), env.get_mask())
            )
            if active == candidate_role:
                stats["decisions"] += 1
                stats["total_time_ms"] += (time.perf_counter() - t0) * 1000.0
                if track_bluff and action in _TRUCO_CALLS:
                    stats["truco_calls"] += 1
                    highest_rank = max((c.truco_rank for c in env.state.hands[active]), default=0)
                    if highest_rank < 10:
                        stats["bluff_calls"] += 1

            env.step(action)

        return env.state.score_p0 - env.state.score_p1

    def run_showdown(self, config: ShowdownConfig) -> dict[str, dict[str, Any]]:
        """Run duplicate matches against all configured opponents"""
        results: dict[str, dict[str, Any]] = {}
        env = TrucoHandEnv()

        for opp_name in config.opponents:
            cand_agent = self._load(config.candidate)
            opp_agent = self._load(opp_name)

            wins = 0
            losses = 0
            draws = 0
            net_points = 0
            stats: dict[str, float] = {
                "decisions": 0.0,
                "total_time_ms": 0.0,
                "truco_calls": 0.0,
                "bluff_calls": 0.0,
            }

            rng = random.Random(config.seed)
            for _ in range(config.n_matches):
                match_seed = rng.randint(0, 2**31 - 1)
                s0 = (match_seed * 2) & 0x7FFFFFFF
                s1 = (match_seed * 2 + 1) & 0x7FFFFFFF

                # Match A: Candidate as P0, Opponent as P1
                random.seed(match_seed)
                _seed_agent(cand_agent, s0)
                _seed_agent(opp_agent, s1)
                d_A = self._run_single_match(
                    p0_agent=cand_agent,
                    p1_agent=opp_agent,
                    env=env,
                    seed=match_seed,
                    candidate_role=0,
                    track_bluff=config.track_bluff,
                    stats=stats,
                )

                # Match B: Opponent as P0, Candidate as P1
                random.seed(match_seed)
                _seed_agent(opp_agent, s0)
                _seed_agent(cand_agent, s1)
                d_B = self._run_single_match(
                    p0_agent=opp_agent,
                    p1_agent=cand_agent,
                    env=env,
                    seed=match_seed,
                    candidate_role=1,
                    track_bluff=config.track_bluff,
                    stats=stats,
                )

                delta = d_A - d_B
                net_points += delta
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    draws += 1

            win_rate = float((wins / config.n_matches) * 100.0)
            avg_delta = float(net_points / config.n_matches)
            bluff_frequency = (
                float(stats["bluff_calls"] / stats["truco_calls"])
                if stats["truco_calls"] > 0
                else 0.0
            )
            avg_time_ms = (
                float(stats["total_time_ms"] / stats["decisions"])
                if stats["decisions"] > 0
                else 0.0
            )

            results[opp_name] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "net_points": net_points,
                "avg_delta": avg_delta,
                "bluff_frequency": bluff_frequency,
                "avg_time_ms": avg_time_ms,
            }

        return results
