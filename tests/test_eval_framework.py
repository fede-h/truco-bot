from pathlib import Path
import pytest
from truco_bot.core.actions import Action
from truco_bot.env.pettingzoo import TrucoAECEnv
from truco_bot.eval.config import ShowdownConfig
from truco_bot.eval.showdown import ShowdownRunner
from truco_bot.eval.tracking import MLflowTracker, generate_heatmaps, save_figures


@pytest.fixture
def env():
    e = TrucoAECEnv()
    e.reset(seed=42)
    return e


def test_env_lifecycle(env):
    assert env.possible_agents == ["player_0", "player_1"]
    assert env.agent_selection == "player_0"
    obs = env.observe("player_0")
    assert len(obs["observation"]) == 12
    assert len(obs["action_mask"]) == 26
    assert env.observe("player_1")["action_mask"] == [False] * 26
    with pytest.raises(KeyError):
        env.observe("invalid")


def test_env_steps(env):
    mask = env.observe(env.agent_selection)["action_mask"]
    legal_act = next(i for i, v in enumerate(mask) if v)
    illegal_act = next(i for i, v in enumerate(mask) if not v)
    with pytest.raises(ValueError):
        env.step(illegal_act)
    env.step(legal_act)
    assert env.agent_selection in env.possible_agents


def test_env_play_through(env):
    while not all(env.terminations.values()):
        cur = env.agent_selection
        act = next(i for i, v in enumerate(env.observe(cur)["action_mask"]) if v)
        env.step(act)
    assert env.rewards["player_0"] + env.rewards["player_1"] == 0.0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_config_valid(tmp_path):
    c = ShowdownConfig(candidate="random", opponents=["heuristic"], n_matches=10, seed=1)
    assert c.to_dict()["n_matches"] == 10
    d = {"candidate": "c", "opponents": ["o"], "n_matches": 5, "seed": 2, "track_bluff": False}
    assert ShowdownConfig.from_dict(d).seed == 2
    f = tmp_path / "c.yaml"
    f.write_text("candidate: c\nopponents:\n  - o\nn_matches: 10\nseed: 1\ntrack_bluff: true")
    assert ShowdownConfig.from_yaml(f).candidate == "c"


@pytest.mark.parametrize(
    "bad",
    [
        {"candidate": "", "opponents": ["o"]},
        {"candidate": "c", "opponents": []},
        {"candidate": "c", "opponents": ["o"], "n_matches": 0},
        {"candidate": "c", "opponents": ["o"], "extra": 1},
    ],
)
def test_config_invalid(bad):
    with pytest.raises(ValueError):
        ShowdownConfig.from_dict(bad)


def test_showdown_symmetry():
    cfg = ShowdownConfig(candidate="heuristic", opponents=["heuristic"], n_matches=10, seed=42)
    res = ShowdownRunner().run_showdown(cfg)["heuristic"]
    assert res["net_points"] == 0
    assert res["wins"] == res["losses"]
    assert res["avg_delta"] == 0.0
    assert res["win_rate"] == 0.0


def test_showdown_metrics():
    cfg = ShowdownConfig(candidate="random", opponents=["heuristic"], n_matches=10, seed=42)
    res = ShowdownRunner().run_showdown(cfg)["heuristic"]
    assert res["wins"] + res["losses"] + res["draws"] == 10
    assert 0.0 <= res["bluff_frequency"] <= 1.0
    assert res["avg_time_ms"] >= 0.0


def test_tracking_and_heatmaps(tmp_path):
    t = MLflowTracker(dry_run=True)
    t.log_params({"p": 1})
    t.log_metrics({"m": 0.5})
    assert t.logged_params["p"] == 1
    assert t.logged_metrics["m"] == 0.5
    res = {"opp": {"win_rate": 50.0, "net_points": 2, "bluff_frequency": 0.1, "avg_time_ms": 1.0}}
    t.log_showdown(res, "run")
    figs = generate_heatmaps(res)
    assert len(figs) == 2
    paths = save_figures(figs, tmp_path)
    assert len(paths) == 2 and all(p.exists() for p in paths)
