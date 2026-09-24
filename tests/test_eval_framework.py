import pytest

from truco_bot.eval.config import ShowdownConfig
from truco_bot.eval.showdown import ShowdownRunner


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
