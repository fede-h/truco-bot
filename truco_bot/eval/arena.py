"""Head-to-head match runner and Elo tracker."""

from truco_bot.agents.base import Agent
from truco_bot.env.engine import TrucoHandEnv


def _run_single_match(p0_agent: Agent, p1_agent: Agent, env: TrucoHandEnv, seed: int = 42) -> int:
    env.reset(seed=seed)
    p0_agent.reset()
    p1_agent.reset()

    while not env.state.is_hand_done:
        active = env.state.active_player
        agent = p0_agent if active == 0 else p1_agent

        if hasattr(agent, "act_from_state"):
            action = agent.act_from_state(env.state)
        else:
            obs = env.get_obs(active)
            mask = env.get_mask()
            action = agent.act(obs, mask)

        env.step(action)

    return env.state.score_p0 - env.state.score_p1



def play_duplicate_match(agent_a, agent_b, env:TrucoHandEnv, seed:int=42) -> int:
    match_a = _run_single_match(agent_a, agent_b, env, seed)
    match_b = _run_single_match(agent_b, agent_a, env, seed)

    return match_a - match_b
