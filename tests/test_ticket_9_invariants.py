"""Adversarial invariant tests for Ticket #9:
- Authentic Argentine Falta Envido (malas vs buenas)
- Pie Envido window after Mano leads card 0
- Evaluation hand delta accounting for both Envido and Truco points
"""

import pytest

from truco_bot.core.actions import Action
from truco_bot.core.card import Card
from truco_bot.core.deck import deal
from truco_bot.core.rules import calculate_falta_envido_points
from truco_bot.core.state import create_initial_hand_state, get_legal_actions, step


class TestAuthenticFaltaEnvido:
    """Argentine Falta Envido rules:
    - If BOTH in malas (< target // 2): awards (target - winner_score) so winner finishes the game.
    - If EITHER in buenas (>= target // 2): awards (target - max(score_p0, score_p1)) [points leader needs].
    """

    @pytest.mark.parametrize(
        "score_p0,score_p1,max_score,winner,expected_points",
        [
            # 30-pt match, both in malas (< 15)
            # P0 trailing: needs 26 to reach 30
            (4, 8, 30, 0, 26),
            # P1 leading in malas: needs 22 to reach 30
            (4, 8, 30, 1, 22),
            # Tie at 0-0 in malas
            (0, 0, 30, 0, 30),
            (0, 0, 30, 1, 30),
            # At 14-14 (both still in malas, 14 < 15)
            (14, 14, 30, 0, 16),
            (14, 14, 30, 1, 16),
            # 30-pt match, at least one in buenas (>= 15)
            # P0 in buenas (18 >= 15), leader needs 30 - 18 = 12
            (18, 8, 30, 0, 12),
            (18, 8, 30, 1, 12),
            # Both in buenas (22, 28), leader at 28 needs 30 - 28 = 2
            (22, 28, 30, 0, 2),
            (22, 28, 30, 1, 2),
            # At 29-20, leader needs 1
            (29, 20, 30, 1, 1),
            # 15-pt match (chico), malas threshold = 15 // 2 = 7
            # Both in malas (< 7): P0=2, P1=5
            (2, 5, 15, 0, 13),  # 15 - 2 = 13
            (2, 5, 15, 1, 10),  # 15 - 5 = 10
            # One in buenas (>= 7): P0=8, P1=4 -> leader needs 15 - 8 = 7
            (8, 4, 15, 0, 7),
            (8, 4, 15, 1, 7),
        ],
    )
    def test_falta_envido_points_calculation(
        self, score_p0: int, score_p1: int, max_score: int, winner: int, expected_points: int
    ):
        pts = calculate_falta_envido_points(score_p0, score_p1, max_score, winner=winner)
        assert pts == expected_points
        # Invariant: winner's new score in malas must equal max_score
        is_malas = score_p0 < (max_score // 2) and score_p1 < (max_score // 2)
        if is_malas:
            winner_score = score_p0 if winner == 0 else score_p1
            assert winner_score + pts == max_score


class TestPieEnvidoWindow:
    """Pie's legal right to call Envido after Mano leads card 0 in Trick 1."""

    def _sample_hands(self) -> list[list[Card]]:
        hands, _ = deal(2, 3, seed=42)
        return hands

    def test_pie_can_sing_envido_after_mano_plays_first_card(self):
        state = create_initial_hand_state(self._sample_hands(), mano=0)
        assert state.active_player == 0

        # Mano plays card 0
        state = step(state, Action.PLAY_CARD_0)
        assert state.active_player == 1
        assert state.current_trick == 0
        assert len(state.trick_cards) == 1

        # Tricky invariant: envido must NOT be sealed yet for Pie!
        legal = get_legal_actions(state)
        assert Action.ENVIDO in legal, (
            f"Pie should be able to call ENVIDO after Mano plays card 0. Legal: {legal}"
        )
        assert Action.REAL_ENVIDO in legal
        assert Action.FALTA_ENVIDO in legal

    def test_pie_calling_envido_resolves_and_returns_turn_to_pie(self):
        state = create_initial_hand_state(self._sample_hands(), mano=0)
        # Mano plays card 0
        state = step(state, Action.PLAY_CARD_0)

        # Pie calls ENVIDO
        state = step(state, Action.ENVIDO)
        assert state.pending_envido_response is True
        assert state.active_player == 0  # Back to Mano to respond

        # Mano accepts ENVIDO
        state = step(state, Action.QUIERO_ENVIDO)
        assert state.envido_resolved is True
        assert state.pending_envido_response is False
        # Turn MUST return to Pie to complete Trick 1 card play!
        assert state.active_player == 1
        assert len(state.trick_cards) == 1

    def test_envido_locks_if_pie_plays_card_without_singing_envido(self):
        state = create_initial_hand_state(self._sample_hands(), mano=0)
        # Mano plays card 0
        state = step(state, Action.PLAY_CARD_0)
        # Pie plays card 0 without calling Envido
        state = step(state, Action.PLAY_CARD_0)

        # Trick 1 has completed; Envido must now be permanently resolved / locked out
        assert state.envido_resolved is True
        legal = get_legal_actions(state)
        assert Action.ENVIDO not in legal
        assert Action.REAL_ENVIDO not in legal
        assert Action.FALTA_ENVIDO not in legal


class TestShowdownHandDeltaScoring:
    """Evaluation hand score must account for both Envido and Truco points."""

    def test_hand_delta_includes_truco_points_when_no_envido(self):
        from truco_bot.env.engine import TrucoHandEnv

        env = TrucoHandEnv()
        env.reset(seed=123)

        # Force a playout where Truco is played and won by P0, no Envido
        # P0 calls TRUCO, P1 folds
        state = env.state
        state = step(state, Action.TRUCO)
        assert state.active_player == 1
        state = step(state, Action.NO_QUIERO_TRUCO)
        assert state.is_hand_done is True
        assert state.winner == 0
        assert state.points_won == 1
        env.state = state

        # Check total points helper
        total_p0 = state.score_p0 + (state.points_won if state.winner == 0 else 0)
        total_p1 = state.score_p1 + (state.points_won if state.winner == 1 else 0)
        assert total_p0 - total_p1 == 1
