"""Core Truco rules, deck, and match state engine."""

from truco_bot.core.actions import Action
from truco_bot.core.card import Card, Suit
from truco_bot.core.deck import deal
from truco_bot.core.match import MatchState, TrucoMatch
from truco_bot.core.state import GameState, create_initial_hand_state, get_legal_actions, step

__all__ = [
    "Action",
    "Card",
    "GameState",
    "MatchState",
    "Suit",
    "TrucoMatch",
    "create_initial_hand_state",
    "deal",
    "get_legal_actions",
    "step",
]
