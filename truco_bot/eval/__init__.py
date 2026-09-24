"""Tournaments and evaluation metrics."""

from truco_bot.eval.arena import play_duplicate_match
from truco_bot.eval.benchmark import load_agent, run_matches, run_tournament
from truco_bot.eval.config import ShowdownConfig
from truco_bot.eval.runner import main as run_cli
from truco_bot.eval.showdown import ShowdownRunner

__all__ = [
    "ShowdownConfig",
    "ShowdownRunner",
    "load_agent",
    "play_duplicate_match",
    "run_cli",
    "run_matches",
    "run_tournament",
]
