"""Tests for CFR model persistence, serialization, and training pipeline."""

import math
import pickle
from pathlib import Path

import pytest

from truco_bot.agents.cfr.agent import VanillaCFRAgent
from truco_bot.agents.cfr.train import train_and_save
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state


def _sample_policy() -> dict[tuple, dict[Action, float]]:
    return {
        ("test_infoset_0",): {Action.QUIERO_TRUCO: 0.75, Action.NO_QUIERO_TRUCO: 0.25},
        ("test_infoset_1",): {Action.ENVIDO: 0.6, Action.NO_QUIERO_ENVIDO: 0.4},
    }


def test_vanilla_cfr_agent_save_and_from_checkpoint_chance(tmp_path: Path) -> None:
    """Verify VanillaCFRAgent (Chance-Sampled) policy saves and reloads via from_checkpoint."""
    original_policy = _sample_policy()
    agent = VanillaCFRAgent(policy=original_policy, is_canonical=False, seed=42)
    checkpoint_path = tmp_path / "cfr_chance_agent.pkl"

    agent.save(checkpoint_path)
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0

    loaded_agent = VanillaCFRAgent.from_checkpoint(checkpoint_path, is_canonical=False, seed=42)
    assert isinstance(loaded_agent, VanillaCFRAgent)
    assert loaded_agent.is_canonical is False

    # 100% key and distribution matching
    assert loaded_agent.policy.keys() == agent.policy.keys()
    for key, original_distribution in agent.policy.items():
        loaded_distribution = loaded_agent.policy[key]
        assert loaded_distribution.keys() == original_distribution.keys()
        for action, prob in original_distribution.items():
            assert math.isclose(
                loaded_distribution[action],
                prob,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )


def test_vanilla_cfr_agent_save_and_from_checkpoint_canonical(tmp_path: Path) -> None:
    """Verify VanillaCFRAgent (Canonical Isomorphic) policy saves and reloads."""
    original_policy = _sample_policy()
    agent = VanillaCFRAgent(policy=original_policy, is_canonical=True, seed=123)
    checkpoint_path = tmp_path / "cfr_canonical_agent.pkl"

    agent.save(str(checkpoint_path))  # also verify str path support
    assert checkpoint_path.exists()

    loaded_agent = VanillaCFRAgent.from_checkpoint(
        str(checkpoint_path), is_canonical=True, seed=123
    )
    assert isinstance(loaded_agent, VanillaCFRAgent)
    assert loaded_agent.is_canonical is True

    assert loaded_agent.policy.keys() == agent.policy.keys()
    for key, original_distribution in agent.policy.items():
        loaded_distribution = loaded_agent.policy[key]
        assert loaded_distribution.keys() == original_distribution.keys()
        for action, prob in original_distribution.items():
            assert math.isclose(
                loaded_distribution[action],
                prob,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )


def test_from_checkpoint_compatible_with_raw_pickle_dump(tmp_path: Path) -> None:
    """Verify from_checkpoint can load a policy dumped directly via standard pickle.dump."""
    dummy_policy = {
        ("dummy_key_1",): {Action.QUIERO_TRUCO: 0.8, Action.NO_QUIERO_TRUCO: 0.2},
        ("dummy_key_2",): {Action.ENVIDO: 0.5, Action.NO_QUIERO_ENVIDO: 0.5},
    }
    raw_pickle_path = tmp_path / "raw_policy.pkl"
    with open(raw_pickle_path, "wb") as f:
        pickle.dump(dummy_policy, f)

    loaded_agent = VanillaCFRAgent.from_checkpoint(raw_pickle_path, is_canonical=False, seed=42)
    assert loaded_agent.policy.keys() == dummy_policy.keys()
    assert loaded_agent.policy[("dummy_key_1",)][Action.QUIERO_TRUCO] == pytest.approx(0.8)


def test_cfr_agent_act_determinism_before_and_after_serialization(tmp_path: Path) -> None:
    """Verify act(obs, mask) and act_from_state(state) produce identical decisions."""
    policy = _sample_policy()
    agent = VanillaCFRAgent(policy=policy, is_canonical=False, seed=999)
    checkpoint_path = tmp_path / "deterministic_agent.pkl"
    agent.save(checkpoint_path)

    loaded_agent = VanillaCFRAgent.from_checkpoint(checkpoint_path, is_canonical=False, seed=999)

    # 1. act(obs, mask) determinism over multiple scenarios
    test_masks = [
        [i in (Action.QUIERO_TRUCO.value, Action.NO_QUIERO_TRUCO.value) for i in range(26)],
        [i in (Action.ENVIDO.value, Action.REAL_ENVIDO.value, Action.NO_QUIERO_ENVIDO.value) for i in range(26)],
        [i in (Action.PLAY_CARD_1.value, Action.PLAY_CARD_2.value) for i in range(26)],
    ]
    test_obs = [
        (1, 2, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (4, 5, 6, 1, 0, 2, 0, 1, 2, 1, 1, 0),
        (7, 8, 9, 0, 0, 0, 3, 2, 0, 0, 0, 1),
    ]

    for obs, mask in zip(test_obs, test_masks, strict=True):
        # Reset RNG seeds to identical state to ensure deterministic sampling
        agent.rng.seed(12345)
        loaded_agent.rng.seed(12345)

        action_orig = agent.act(obs, mask)
        action_loaded = loaded_agent.act(obs, mask)
        assert action_orig == action_loaded, f"Mismatch on obs={obs}: {action_orig} != {action_loaded}"

    # 2. act_from_state(state) determinism
    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)

    agent.rng.seed(54321)
    loaded_agent.rng.seed(54321)

    state_action_orig = agent.act_from_state(state)
    state_action_loaded = loaded_agent.act_from_state(state)
    assert state_action_orig == state_action_loaded


def test_train_and_save_native(tmp_path: Path) -> None:
    """Verify train_and_save produces a valid .bin native checkpoint."""
    output_path = tmp_path / "trained_native.bin"
    result_path = train_and_save(iterations=10, output_path=output_path)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0


def test_train_and_save_cfr(tmp_path: Path) -> None:
    """Verify train_and_save produces a valid .bin native checkpoint using CFR."""
    output_path = tmp_path / "trained_cfr.bin"
    result_path = train_and_save(iterations=10, output_path=output_path, algorithm="cfr")
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0


def test_from_checkpoint_nonexistent_file() -> None:
    """Verify from_checkpoint raises FileNotFoundError if file does not exist."""
    with pytest.raises(FileNotFoundError):
        VanillaCFRAgent.from_checkpoint("nonexistent_model_checkpoint_path_xyz.pkl")


def test_from_checkpoint_loads_sqlite_if_present(tmp_path: Path) -> None:
    """Verify from_checkpoint attaches sqlite db when adjacent .db exists."""
    import sqlite3

    pkl_path = tmp_path / "model.pkl"
    db_path = tmp_path / "model.db"

    with open(pkl_path, "wb") as f:
        pickle.dump({}, f)

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE policy (k BLOB PRIMARY KEY, a BLOB)")
        k = pickle.dumps(("test_key",), protocol=pickle.HIGHEST_PROTOCOL)
        a = pickle.dumps({Action.QUIERO_TRUCO: 1.0}, protocol=pickle.HIGHEST_PROTOCOL)
        conn.execute("INSERT INTO policy VALUES (?, ?)", (k, a))

    agent = VanillaCFRAgent.from_checkpoint(pkl_path)
    assert agent._conn is not None
    assert agent.db_path == str(db_path)

