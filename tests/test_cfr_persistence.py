"""Tests for CFR model persistence, serialization, and training pipeline."""

from pathlib import Path

import pytest

from truco_bot.agents.cfr.native_agent import NativeCFRAgent
from truco_bot.agents.cfr.train import train_and_save
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS
from truco_bot.core.state import create_initial_hand_state


def test_train_and_save_native(tmp_path: Path) -> None:
    """Verify train_and_save produces a valid .bin native checkpoint with MCCFR."""
    output_path = tmp_path / "trained_native.bin"
    result_path = train_and_save(iterations=10, output_path=output_path, algorithm="mccfr", capacity=65536)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0


def test_train_and_save_cfr(tmp_path: Path) -> None:
    """Verify train_and_save produces a valid .bin native checkpoint using CFR."""
    output_path = tmp_path / "trained_cfr.bin"
    result_path = train_and_save(iterations=2, output_path=output_path, algorithm="cfr", capacity=262_144)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0


def test_native_cfr_agent_save_and_from_checkpoint(tmp_path: Path) -> None:
    """Verify NativeCFRAgent loads correctly from checkpoint file and decides actions."""
    output_path = tmp_path / "model.bin"
    train_and_save(iterations=20, output_path=output_path, algorithm="mccfr", capacity=65536)

    agent = NativeCFRAgent.from_checkpoint(output_path, capacity=65536, seed=42)
    assert isinstance(agent, NativeCFRAgent)
    assert agent.table.capacity() == 65536

    h0 = list(ALL_CARDS[:3])
    h1 = list(ALL_CARDS[3:6])
    state = create_initial_hand_state([h0, h1], mano=0)
    action = agent.act_from_state(state)
    assert isinstance(action, Action)


def test_from_checkpoint_nonexistent_file() -> None:
    """Verify from_checkpoint raises an error if file does not exist."""
    with pytest.raises(OSError):
        NativeCFRAgent.from_checkpoint("nonexistent_model_checkpoint_path_xyz.bin", capacity=1024)
