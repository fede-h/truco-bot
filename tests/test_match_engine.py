"""Adversarial test suite for Ticket #10:
- Core 15/30-point match engine (TrucoMatch)
- In-process environment (TrucoMatchEnv)
- Score decoupling invariant
- Early Envido match termination ("el envido va primero")
- Information hiding in player views
- Full match playouts with existing agents (Random, Heuristic, Native CFR, Deep CFR)
"""

from truco_bot.core.actions import Action
from truco_bot.core.match import TrucoMatch
from truco_bot.env.match_env import TrucoMatchEnv


class TestTrucoMatchState:
    """Core state machine invariants for TrucoMatch."""

    def test_match_initialization(self):
        match = TrucoMatch(target_score=30, seed=42)
        state = match.state
        assert state.target_score == 30
        assert state.match_score_p0 == 0
        assert state.match_score_p1 == 0
        assert state.hand_index == 0
        assert state.mano == 0
        assert state.is_match_done is False
        assert state.match_winner is None
        assert match.current_hand is not None
        # Hand score must strictly start at (0, 0)
        assert match.current_hand.score_p0 == 0
        assert match.current_hand.score_p1 == 0

    def test_mano_alternation_across_hands(self):
        match = TrucoMatch(target_score=30, seed=100)
        assert match.state.mano == 0
        assert match.current_hand.mano == 0

        # Conclude hand 0 by fold
        match.step(Action.IR_AL_MAZO)
        assert match.is_hand_done is True

        # Advance to hand 1
        match.next_hand()
        assert match.state.hand_index == 1
        assert match.state.mano == 1
        assert match.current_hand.mano == 1

        # Conclude hand 1 by fold
        match.step(Action.IR_AL_MAZO)
        match.next_hand()
        assert match.state.hand_index == 2
        assert match.state.mano == 0
        assert match.current_hand.mano == 0

    def test_score_decoupling_hand_resets_to_zero(self):
        match = TrucoMatch(target_score=30, seed=123)
        # P0 folds -> P1 gets 1 truco point
        match.step(Action.IR_AL_MAZO)
        assert match.state.match_score_p0 == 0
        assert match.state.match_score_p1 == 1

        # Advance to hand 1
        match.next_hand()
        # Invariant: hand state score must be strictly (0, 0)
        assert match.current_hand.score_p0 == 0
        assert match.current_hand.score_p1 == 0
        # Running match score remains tracked
        assert match.state.match_score_p0 == 0
        assert match.state.match_score_p1 == 1

    def test_early_match_termination_on_envido(self):
        """'El envido va primero': If a player reaches target score during Envido,
        the match terminates immediately without playing out trick cards."""
        match = TrucoMatch(target_score=15, seed=777)
        # Pre-set scores near match point in malas/buenas
        match.state.match_score_p0 = 13
        match.state.match_score_p1 = 12

        # P0 (mano) calls REAL_ENVIDO (3 pts), P1 accepts
        match.step(Action.REAL_ENVIDO)
        match.step(Action.QUIERO_ENVIDO)

        # Envido is resolved; whoever won earned 3 pts, pushing their score >= 15!
        assert match.state.is_match_done is True
        assert match.state.match_winner is not None
        winner = match.state.match_winner
        winner_score = match.state.match_score_p0 if winner == 0 else match.state.match_score_p1
        assert winner_score >= 15
        # The hand does NOT require tricks to be played
        assert match.is_match_done is True

    def test_information_hiding_masks_opponent_cards(self):
        match = TrucoMatch(target_score=30, seed=999)
        view_p0 = match.get_view(player=0)
        view_p1 = match.get_view(player=1)

        # P0 view contains P0's exact cards
        assert len(view_p0["my_hand"]) == 3
        assert "suit" in view_p0["my_hand"][0]
        assert "number" in view_p0["my_hand"][0]
        # P0 view MUST NOT reveal P1's card details
        assert view_p0["opponent_card_count"] == 3
        assert "opponent_hand" not in view_p0 or view_p0["opponent_hand"] is None

        # Symmetric for P1 view
        assert len(view_p1["my_hand"]) == 3
        assert view_p1["opponent_card_count"] == 3


class TestTrucoMatchEnv:
    """In-process gym-style environment wrapper for multi-hand match play."""

    def test_env_full_match_random_vs_random(self):
        from truco_bot.agents.baselines.random import RandomAgent

        env = TrucoMatchEnv(target_score=15)
        p0 = RandomAgent(seed=1)
        p1 = RandomAgent(seed=2)

        match_state = env.reset(seed=42)
        assert match_state.target_score == 15

        step_count = 0
        while not env.is_match_done:
            step_count += 1
            assert step_count < 2000, "Match exceeded safety step limit (possible deadlock)"

            if env.is_hand_done:
                env.next_hand()
                continue

            active = env.active_player
            agent = p0 if active == 0 else p1
            mask = env.get_action_mask()
            action = agent.act(env.get_obs(active), mask)
            env.step(action)

        assert env.is_match_done is True
        assert env.match_winner in (0, 1)
        final_scores = (env.match_score_p0, env.match_score_p1)
        assert max(final_scores) >= 15

    def test_env_full_match_heuristic_vs_heuristic(self):
        from truco_bot.agents.baselines.heuristic import HeuristicAgent

        env = TrucoMatchEnv(target_score=15)
        p0 = HeuristicAgent()
        p1 = HeuristicAgent()

        env.reset(seed=12345)
        steps = 0
        while not env.is_match_done:
            steps += 1
            assert steps < 2000

            if env.is_hand_done:
                env.next_hand()
                continue

            active = env.active_player
            agent = p0 if active == 0 else p1
            mask = env.get_action_mask()
            action = agent.act(env.get_obs(active), mask)
            env.step(action)

        assert env.is_match_done is True
        assert max(env.match_score_p0, env.match_score_p1) >= 15


class TestAgentMatchPlayouts:
    """Stress test executing full matches with competitive CFR agents."""

    def test_native_cfr_agent_can_play_full_30_point_match(self):
        from truco_bot.agents.baselines.heuristic import HeuristicAgent
        from truco_bot.agents.cfr.native_agent import NativeCFRAgent

        env = TrucoMatchEnv(target_score=30)
        cfr_bot = NativeCFRAgent(capacity=1024, seed=42)
        heuristic_bot = HeuristicAgent()

        env.reset(seed=999)
        cfr_bot.reset()
        heuristic_bot.reset()

        steps = 0
        while not env.is_match_done:
            steps += 1
            assert steps < 3000

            if env.is_hand_done:
                cfr_bot.reset()
                heuristic_bot.reset()
                env.next_hand()
                continue

            active = env.active_player
            if active == 0:
                action = cfr_bot.act_from_state(env.hand_state)
            else:
                action = heuristic_bot.act(env.get_obs(active), env.get_action_mask())

            env.step(action)

        assert env.is_match_done is True
        assert max(env.match_score_p0, env.match_score_p1) >= 30
