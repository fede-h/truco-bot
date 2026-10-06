"""Tests for DeepCFRAgent integration with Truco environment and Arena."""

from pathlib import Path

from truco_bot.agents.deep_cfr.agent import DeepCFRAgent
from truco_bot.agents.deep_cfr.network import PolicyNetwork
from truco_engine import BitboardState

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state, get_legal_actions
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.arena import play_duplicate_match


def test_deep_cfr_agent_implements_agent_abc():
    """Verify DeepCFRAgent inherits from Agent ABC."""
    assert issubclass(DeepCFRAgent, Agent)
    agent = DeepCFRAgent(seed=42)
    assert isinstance(agent, Agent)


def test_deep_cfr_agent_instantiation_random_and_weights(tmp_path: Path):
    """Verify agent can be initialized randomly or loaded from checkpoint weights."""
    # Random initialization
    agent = DeepCFRAgent(seed=42)
    assert agent is not None

    # Instantiation with explicit network
    net = PolicyNetwork(in_dim=156, num_actions=8)
    agent_custom = DeepCFRAgent(policy_net=net, seed=42)
    assert agent_custom is not None

    # Checkpoint save & load
    ckpt_path = tmp_path / "model_weights.pt"
    agent.save(ckpt_path)
    assert ckpt_path.exists()

    loaded_agent = DeepCFRAgent.from_checkpoint(ckpt_path, seed=42)
    assert isinstance(loaded_agent, DeepCFRAgent)


def test_deep_cfr_agent_act_returns_legal_action():
    """Verify agent.act(obs, mask) always selects a legal action."""
    agent = DeepCFRAgent(seed=42)
    obs = (1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    # Partial action mask
    mask = [False] * 26
    mask[Action.PLAY_CARD_0.value] = True
    mask[Action.PLAY_CARD_1.value] = True
    mask[Action.QUIERO_TRUCO.value] = True

    action = agent.act(obs, mask)
    assert isinstance(action, Action)
    assert mask[action.value] is True, f"Selected action {action} was illegal under mask"


def test_deep_cfr_agent_act_single_legal_action():
    """Verify agent returns the forced action when only one legal action exists."""
    agent = DeepCFRAgent(seed=42)
    obs = (1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    mask = [False] * 26
    mask[Action.QUIERO_ENVIDO.value] = True

    action = agent.act(obs, mask)
    assert action == Action.QUIERO_ENVIDO


def test_deep_cfr_agent_act_from_state_gamestate():
    """Verify agent.act_from_state(state) selects legal action given GameState."""
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    agent = DeepCFRAgent(seed=42)
    action = agent.act_from_state(state)

    assert isinstance(action, Action)
    legal_actions = get_legal_actions(state)
    assert action in legal_actions, f"Selected action {action} not in legal actions {legal_actions}"


def test_deep_cfr_agent_act_from_state_bitboard():
    """Verify agent.act_from_state(state) selects legal action given BitboardState."""
    bstate = BitboardState([[0, 1, 2], [3, 4, 5]], 0)

    agent = DeepCFRAgent(seed=42)
    action = agent.act_from_state(bstate)

    assert isinstance(action, Action)
    legal_mask = bstate.legal_actions_mask()
    assert (legal_mask & (1 << action.value)) != 0


def test_deep_cfr_agent_duplicate_match_against_random():
    """Verify running a 1-duplicate-hand match against RandomAgent runs without error."""
    agent = DeepCFRAgent(seed=42)
    opponent = RandomAgent(seed=42)
    env = TrucoHandEnv()

    score_diff = play_duplicate_match(agent, opponent, env, seed=42)
    assert isinstance(score_diff, int)


def test_deep_cfr_agent_duplicate_match_self_play():
    """Verify running duplicate match in self-play completes and returns an integer score."""
    agent_a = DeepCFRAgent(seed=42)
    agent_b = DeepCFRAgent(seed=99)
    env = TrucoHandEnv()

    score_diff = play_duplicate_match(agent_a, agent_b, env, seed=42)
    assert isinstance(score_diff, int)


def test_deep_cfr_agent_reset():
    """Verify agent.reset() resets per-hand state without raising errors."""
    agent = DeepCFRAgent(seed=42)
    agent.reset()  # Should execute cleanly
