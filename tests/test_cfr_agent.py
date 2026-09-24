"""Tests for NativeCFRAgent integration with Truco environment and Arena."""

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.arena import _run_single_match, play_duplicate_match


def test_native_cfr_agent_implements_agent_abc():
    assert issubclass(NativeCFRAgent, Agent)
    agent = NativeCFRAgent(capacity=1024, seed=42)
    assert isinstance(agent, Agent)


def test_native_cfr_agent_fallback_on_empty_policy():
    """Verify agent falls back to domain-informed EquityFallback when table is empty."""
    agent = NativeCFRAgent(capacity=1024, seed=42)
    mask = [False] * 26
    mask[Action.QUIERO_TRUCO.value] = True
    mask[Action.NO_QUIERO_TRUCO.value] = True

    obs = (1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    action = agent.act(obs, mask)
    assert action in (Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO)


def test_native_cfr_agent_act_from_state():
    """Verify agent correctly selects actions given direct GameState."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    agent = NativeCFRAgent(capacity=1024, seed=42)
    action = agent.act_from_state(state)
    assert isinstance(action, Action)


def test_native_cfr_agent_arena_head_to_head():
    """Verify NativeCFRAgent executes head-to-head duplicate matches."""
    agent_a = NativeCFRAgent(capacity=1024, seed=42)
    agent_b = NativeCFRAgent(capacity=1024, seed=99)

    env = TrucoHandEnv()
    # Direct single match verification
    single_res = _run_single_match(agent_a, agent_b, env, seed=42)
    assert isinstance(single_res, int)

    # Duplicate match between Variant A and Variant B
    res = play_duplicate_match(agent_a, agent_b, env, seed=42)
    assert isinstance(res, int)

    random_agent = RandomAgent(seed=42)
    res_rand = play_duplicate_match(agent_a, random_agent, env, seed=42)
    assert isinstance(res_rand, int)

    heuristic_agent = HeuristicAgent()
    res_heur = play_duplicate_match(agent_a, heuristic_agent, env, seed=42)
    assert isinstance(res_heur, int)
