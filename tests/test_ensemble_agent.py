"""Tests for EnsembleCFRAgent, EquityFallback, and HandHistoryTracker."""

import sqlite3
from pathlib import Path

from truco_bot.agents.base import Agent
from truco_bot.agents.baselines.equity import EquityAgent
from truco_bot.agents.baselines.heuristic import HeuristicAgent
from truco_bot.agents.baselines.random import RandomAgent
from truco_bot.agents.cfr.ensemble_agent import EnsembleCFRAgent
from truco_bot.agents.cfr.fallback import EquityFallback
from truco_bot.agents.cfr.history import HandHistoryTracker
from truco_bot.core.actions import Action
from truco_bot.core.card import ALL_CARDS, Card, Suit, create_card
from truco_bot.core.state import create_initial_hand_state
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.env.obs import CARD_TO_ID
from truco_bot.eval.arena import _run_single_match, play_duplicate_match


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_obs(
    hand_cards: list[Card],
    tc0: Card | None = None,
    tc1: Card | None = None,
    score_p0: int = 0,
    score_p1: int = 0,
    current_trick: int = 0,
    truco_level: int = 1,
    envido_resolved: bool = False,
    active_player: int = 0,
    mano: int = 0,
) -> tuple[int, ...]:
    """Construct a standard 12-integer observation tuple from domain objects."""
    c0 = CARD_TO_ID.get(hand_cards[0], 0) if len(hand_cards) > 0 else 0
    c1 = CARD_TO_ID.get(hand_cards[1], 0) if len(hand_cards) > 1 else 0
    c2 = CARD_TO_ID.get(hand_cards[2], 0) if len(hand_cards) > 2 else 0
    t0 = CARD_TO_ID.get(tc0, 0) if tc0 is not None else 0
    t1 = CARD_TO_ID.get(tc1, 0) if tc1 is not None else 0
    return (
        c0,
        c1,
        c2,
        t0,
        t1,
        score_p0,
        score_p1,
        current_trick,
        truco_level,
        1 if envido_resolved else 0,
        active_player,
        mano,
    )


def _make_mask(legal_actions: list[Action]) -> list[bool]:
    """Construct a 26-element action mask with given legal actions enabled."""
    mask = [False] * 26
    for a in legal_actions:
        mask[a.value] = True
    return mask


def _call_fallback(
    fallback: EquityFallback,
    obs: tuple[int, ...],
    mask: list[bool],
) -> Action:
    """Invoke fallback action using act or decide interface."""
    if hasattr(fallback, "act"):
        return fallback.act(obs, mask)
    if hasattr(fallback, "decide"):
        return fallback.decide(obs, mask)
    if hasattr(fallback, "choose_action"):
        return fallback.choose_action(obs, mask)
    if callable(fallback):
        return fallback(obs, mask)
    raise AttributeError("EquityFallback must provide act(obs, mask) or decide(obs, mask)")


def _feed_tracker(
    tracker: HandHistoryTracker,
    obs: tuple[int, ...],
    mask: list[bool],
) -> None:
    """Feed observation to tracker via update, observe, or step."""
    if hasattr(tracker, "update"):
        tracker.update(obs, mask)
    elif hasattr(tracker, "observe"):
        tracker.observe(obs, mask)
    elif hasattr(tracker, "step"):
        tracker.step(obs, mask)
    elif callable(tracker):
        tracker(obs, mask)
    else:
        raise AttributeError("HandHistoryTracker must provide update(obs, mask) or observe(obs, mask)")


def _reset_tracker(tracker: HandHistoryTracker) -> None:
    """Reset tracker state for a new hand."""
    if hasattr(tracker, "reset_hand"):
        tracker.reset_hand()
    elif hasattr(tracker, "reset"):
        tracker.reset()
    else:
        raise AttributeError("HandHistoryTracker must provide reset_hand() or reset()")


# ---------------------------------------------------------------------------
# Pillar 2: EquityFallback Unit Tests
# ---------------------------------------------------------------------------


class TestEquityFallback:
    """Unit tests for off-tree EquityFallback card play and bet answering rules."""

    def test_selects_lowest_winning_card_when_opponent_played(self) -> None:
        """Verify fallback picks the lowest winning card against an opponent card."""
        fallback = EquityFallback()

        # Opponent played 10 de Copas (rank 5)
        opp_card = create_card(10, Suit.COPAS)

        # Hand:
        # Card 0: 4 de Copas (rank 1) -> loses to rank 5
        # Card 1: 11 de Espadas (rank 6) -> wins against rank 5 (rank difference = +1)
        # Card 2: 3 de Espadas (rank 10) -> wins against rank 5 (rank difference = +5, overkill)
        hand = [
            create_card(4, Suit.COPAS),
            create_card(11, Suit.ESPADAS),
            create_card(3, Suit.ESPADAS),
        ]
        obs = _make_obs(hand, tc0=opp_card, current_trick=0, active_player=1, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2])

        action = _call_fallback(fallback, obs, mask)
        assert action == Action.PLAY_CARD_1, f"Expected PLAY_CARD_1 (rank 6), got {action}"

    def test_selects_lowest_winning_card_two_cards_remaining(self) -> None:
        """Verify lowest winning card is chosen when only 2 cards remain in hand."""
        fallback = EquityFallback()

        # Opponent played 11 de Oros (rank 6)
        opp_card = create_card(11, Suit.OROS)

        # Hand:
        # Card 0: 1 de Espadas (rank 14) -> wins (highest card in game)
        # Card 1: 12 de Copas (rank 7) -> wins (lowest winning rank)
        hand = [
            create_card(1, Suit.ESPADAS),
            create_card(12, Suit.COPAS),
        ]
        obs = _make_obs(hand, tc0=opp_card, current_trick=1, active_player=1, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1])

        action = _call_fallback(fallback, obs, mask)
        assert action == Action.PLAY_CARD_1, f"Expected PLAY_CARD_1 (rank 7), got {action}"

    def test_discards_lowest_card_when_unable_to_win(self) -> None:
        """Verify fallback discards the lowest rank card when cannot beat opponent."""
        fallback = EquityFallback()

        # Opponent played 7 de Espadas (rank 12)
        opp_card = create_card(7, Suit.ESPADAS)

        # Hand:
        # Card 0: 2 de Copas (rank 9) -> loses (< 12)
        # Card 1: 4 de Oros (rank 1) -> loses (< 12, lowest card in hand)
        # Card 2: 12 de Bastos (rank 7) -> loses (< 12)
        hand = [
            create_card(2, Suit.COPAS),
            create_card(4, Suit.OROS),
            create_card(12, Suit.BASTOS),
        ]
        obs = _make_obs(hand, tc0=opp_card, current_trick=0, active_player=1, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2])

        action = _call_fallback(fallback, obs, mask)
        assert action == Action.PLAY_CARD_1, f"Expected PLAY_CARD_1 (rank 1 discard), got {action}"

    def test_discards_lowest_card_two_cards_left(self) -> None:
        """Verify lowest card discard with two losing cards in hand."""
        fallback = EquityFallback()

        # Opponent played 3 de Bastos (rank 10)
        opp_card = create_card(3, Suit.BASTOS)

        # Hand:
        # Card 0: 6 de Oros (rank 3)
        # Card 1: 5 de Bastos (rank 2) -> lowest card
        hand = [
            create_card(6, Suit.OROS),
            create_card(5, Suit.BASTOS),
        ]
        obs = _make_obs(hand, tc0=opp_card, current_trick=1, active_player=1, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1])

        action = _call_fallback(fallback, obs, mask)
        assert action == Action.PLAY_CARD_1, f"Expected PLAY_CARD_1 (rank 2 discard), got {action}"

    def test_leads_with_highest_card_rank_when_leading_trick(self) -> None:
        """Verify fallback leads trick with the highest card rank in hand."""
        fallback = EquityFallback()

        # Leading trick: no opponent card on table (tc0 is None)
        # Hand:
        # Card 0: 5 de Bastos (rank 2)
        # Card 1: 1 de Bastos (rank 13) -> highest rank
        # Card 2: 11 de Copas (rank 6)
        hand = [
            create_card(5, Suit.BASTOS),
            create_card(1, Suit.BASTOS),
            create_card(11, Suit.COPAS),
        ]
        obs = _make_obs(hand, tc0=None, current_trick=0, active_player=0, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2])

        action = _call_fallback(fallback, obs, mask)
        assert action == Action.PLAY_CARD_1, f"Expected PLAY_CARD_1 (rank 13 lead), got {action}"

    def test_leads_with_highest_card_rank_later_trick(self) -> None:
        """Verify fallback leads with highest remaining card in trick 2."""
        fallback = EquityFallback()

        # Hand:
        # Card 0: 7 de Copas (rank 4)
        # Card 1: 3 de Espadas (rank 10) -> highest rank
        hand = [
            create_card(7, Suit.COPAS),
            create_card(3, Suit.ESPADAS),
        ]
        obs = _make_obs(hand, tc0=None, current_trick=1, active_player=0, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1])

        action = _call_fallback(fallback, obs, mask)
        assert action == Action.PLAY_CARD_1, f"Expected PLAY_CARD_1 (rank 10 lead), got {action}"

    def test_envido_answering_threshold_accepts_when_gte_28(self) -> None:
        """Verify fallback answers Quiero/Real/Falta when Envido points >= 28."""
        fallback = EquityFallback()

        # Hand with 33 envido points: 7 de Copas + 6 de Copas = 33
        hand_33 = [
            create_card(7, Suit.COPAS),
            create_card(6, Suit.COPAS),
            create_card(1, Suit.ESPADAS),
        ]
        obs_33 = _make_obs(hand_33, current_trick=0)
        mask = _make_mask([
            Action.QUIERO_ENVIDO,
            Action.NO_QUIERO_ENVIDO,
            Action.REAL_ENVIDO,
            Action.FALTA_ENVIDO,
        ])
        action_33 = _call_fallback(fallback, obs_33, mask)
        assert action_33 in (
            Action.QUIERO_ENVIDO,
            Action.REAL_ENVIDO,
            Action.FALTA_ENVIDO,
        ), f"Expected positive Envido response for 33 pts, got {action_33}"

        # Hand with exactly 28 envido points: 5 de Oros + 3 de Oros = 28
        hand_28 = [
            create_card(5, Suit.OROS),
            create_card(3, Suit.OROS),
            create_card(10, Suit.COPAS),
        ]
        obs_28 = _make_obs(hand_28, current_trick=0)
        action_28 = _call_fallback(fallback, obs_28, mask)
        assert action_28 in (
            Action.QUIERO_ENVIDO,
            Action.REAL_ENVIDO,
            Action.FALTA_ENVIDO,
        ), f"Expected positive Envido response for 28 pts, got {action_28}"

    def test_envido_answering_threshold_folds_when_lt_28(self) -> None:
        """Verify fallback folds (NO_QUIERO_ENVIDO) when Envido points < 28."""
        fallback = EquityFallback()

        # Hand with 27 envido points: 7 de Oros + 10 de Oros (0) = 27
        hand_27 = [
            create_card(7, Suit.OROS),
            create_card(10, Suit.OROS),
            create_card(4, Suit.ESPADAS),
        ]
        obs_27 = _make_obs(hand_27, current_trick=0)
        mask = _make_mask([
            Action.QUIERO_ENVIDO,
            Action.NO_QUIERO_ENVIDO,
            Action.REAL_ENVIDO,
            Action.FALTA_ENVIDO,
        ])
        action_27 = _call_fallback(fallback, obs_27, mask)
        assert action_27 == Action.NO_QUIERO_ENVIDO, f"Expected NO_QUIERO_ENVIDO for 27 pts, got {action_27}"

        # Hand with 6 envido points (all different suits: 4, 5, 6)
        hand_6 = [
            create_card(4, Suit.OROS),
            create_card(5, Suit.COPAS),
            create_card(6, Suit.ESPADAS),
        ]
        obs_6 = _make_obs(hand_6, current_trick=0)
        action_6 = _call_fallback(fallback, obs_6, mask)
        assert action_6 == Action.NO_QUIERO_ENVIDO, f"Expected NO_QUIERO_ENVIDO for 6 pts, got {action_6}"

    def test_truco_answering_threshold_accepts_when_holding_rank_gte_8(self) -> None:
        """Verify fallback accepts Truco (QUIERO_TRUCO) when holding any card with rank >= 8."""
        fallback = EquityFallback()

        mask = _make_mask([Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO])

        # Hand with rank 9 (2 de Copas) and low cards
        hand_rank_9 = [
            create_card(2, Suit.COPAS),  # rank 9 >= 8
            create_card(4, Suit.OROS),   # rank 1
            create_card(5, Suit.ESPADAS),  # rank 2
        ]
        obs_9 = _make_obs(hand_rank_9, current_trick=0)
        action_9 = _call_fallback(fallback, obs_9, mask)
        assert action_9 in (Action.QUIERO_TRUCO, Action.RETRUCO), f"Expected QUIERO_TRUCO with rank 9, got {action_9}"

        # Hand with rank 8 (1 de Oros) and low cards
        hand_rank_8 = [
            create_card(1, Suit.OROS),    # rank 8 >= 8
            create_card(4, Suit.ESPADAS), # rank 1
            create_card(6, Suit.BASTOS),  # rank 3
        ]
        obs_8 = _make_obs(hand_rank_8, current_trick=0)
        action_8 = _call_fallback(fallback, obs_8, mask)
        assert action_8 in (Action.QUIERO_TRUCO, Action.RETRUCO), f"Expected QUIERO_TRUCO with rank 8, got {action_8}"

    def test_truco_answering_threshold_folds_when_holding_only_low_cards(self) -> None:
        """Verify fallback folds Truco (NO_QUIERO_TRUCO) when all cards in hand have rank < 8."""
        fallback = EquityFallback()

        mask = _make_mask([Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO])

        # Hand with only low cards: ranks 7, 6, 1
        hand_low = [
            create_card(12, Suit.COPAS),  # rank 7 < 8
            create_card(11, Suit.OROS),   # rank 6 < 8
            create_card(4, Suit.COPAS),   # rank 1 < 8
        ]
        obs_low = _make_obs(hand_low, current_trick=0)
        action_low = _call_fallback(fallback, obs_low, mask)
        assert action_low == Action.NO_QUIERO_TRUCO, f"Expected NO_QUIERO_TRUCO with low cards, got {action_low}"


# ---------------------------------------------------------------------------
# Pillar 1: HandHistoryTracker Unit Tests
# ---------------------------------------------------------------------------


class TestHandHistoryTracker:
    """Unit tests for stateful HandHistoryTracker reconstructing played cards, tricks, and envido."""

    def test_reconstruction_across_tricks_in_env(self) -> None:
        """Verify feeding consecutive (obs, mask) across tricks reconstructs exact state."""
        env = TrucoHandEnv()
        env.reset(seed=42)
        tracker = HandHistoryTracker()

        while not env.state.is_hand_done:
            active = env.state.active_player
            obs = env.get_obs(active)
            mask = env.get_mask()

            _feed_tracker(tracker, obs, mask)

            legal_actions = [Action(i) for i, m in enumerate(mask) if m]
            env.step(legal_actions[0])

        # Verify played_cards match exactly
        assert list(tracker.played_cards) == list(env.state.played_cards), (
            f"Played cards mismatch: {tracker.played_cards} vs {env.state.played_cards}"
        )

        # Verify trick_results match exactly
        assert list(tracker.trick_results) == list(env.state.trick_results), (
            f"Trick results mismatch: {tracker.trick_results} vs {env.state.trick_results}"
        )

        # Verify envido_chain matches exactly
        assert list(tracker.envido_chain) == list(env.state.envido_chain), (
            f"Envido chain mismatch: {tracker.envido_chain} vs {env.state.envido_chain}"
        )

    def test_reconstruction_synthetic_three_tricks(self) -> None:
        """Verify tracker reconstructs played cards and trick outcomes step-by-step."""
        tracker = HandHistoryTracker()

        # Step 1: P0 hand cards, leading trick 0
        c_p0_t0 = create_card(1, Suit.ESPADAS)   # rank 14
        c_p0_t1 = create_card(3, Suit.ESPADAS)   # rank 10
        c_p0_t2 = create_card(4, Suit.COPAS)     # rank 1
        obs_0 = _make_obs([c_p0_t0, c_p0_t1, c_p0_t2], tc0=None, current_trick=0, active_player=0, mano=0)
        mask_0 = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2])
        _feed_tracker(tracker, obs_0, mask_0)

        # Step 2: P1 responds to P0's card in trick 0
        c_p1_t0 = create_card(1, Suit.BASTOS)    # rank 13
        c_p1_t1 = create_card(2, Suit.OROS)      # rank 9
        c_p1_t2 = create_card(5, Suit.OROS)      # rank 2
        obs_1 = _make_obs([c_p1_t0, c_p1_t1, c_p1_t2], tc0=c_p0_t0, current_trick=0, active_player=1, mano=0)
        mask_1 = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2])
        _feed_tracker(tracker, obs_1, mask_1)

        # Trick 0 concluded: P0 (rank 14) beat P1 (rank 13) -> trick 0 winner = 1 (P0)
        # Step 3: P0 leads trick 1
        obs_2 = _make_obs([c_p0_t1, c_p0_t2], tc0=None, current_trick=1, active_player=0, mano=0)
        mask_2 = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1])
        _feed_tracker(tracker, obs_2, mask_2)

        # Verify played cards from trick 0 are recorded
        assert len(tracker.played_cards) >= 2
        assert tracker.played_cards[0] == (0, c_p0_t0)
        assert tracker.played_cards[1] == (1, c_p1_t0)
        assert len(tracker.trick_results) >= 1
        assert tracker.trick_results[0] == 1  # P0 won trick 0

    def test_reset_on_new_hand(self) -> None:
        """Verify tracker state is cleared on reset_hand."""
        env = TrucoHandEnv()
        env.reset(seed=10)
        tracker = HandHistoryTracker()

        while not env.state.is_hand_done:
            active = env.state.active_player
            obs = env.get_obs(active)
            mask = env.get_mask()
            _feed_tracker(tracker, obs, mask)
            legal_actions = [Action(i) for i, m in enumerate(mask) if m]
            env.step(legal_actions[0])

        assert len(tracker.played_cards) > 0

        _reset_tracker(tracker)

        assert len(tracker.played_cards) == 0
        assert len(tracker.trick_results) == 0
        assert len(tracker.envido_chain) == 0


# ---------------------------------------------------------------------------
# Pillar 1 & 2: EnsembleCFRAgent Integration Tests
# ---------------------------------------------------------------------------


class TestEnsembleCFRAgent:
    """Tests for EnsembleCFRAgent policy execution, fallback integration, and arena play."""

    def test_ensemble_cfr_agent_implements_agent_abc(self) -> None:
        """Verify EnsembleCFRAgent implements the Agent ABC."""
        assert issubclass(EnsembleCFRAgent, Agent)
        agent = EnsembleCFRAgent(policy={}, seed=42)
        assert isinstance(agent, Agent)

    def test_init_with_dict_and_sqlite(self, tmp_path: Path) -> None:
        """Verify agent initializes from an in-memory dict or SQLite checkpoint."""
        dict_agent = EnsembleCFRAgent(policy={}, seed=42)
        assert isinstance(dict_agent.policy, dict)

        # Create temporary SQLite policy database
        db_path = tmp_path / "test_ensemble_init.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE policy (k BLOB PRIMARY KEY, a BLOB)")
        import pickle

        sample_key = ("test_key",)
        sample_probs = {Action.PLAY_CARD_0: 1.0}
        conn.execute(
            "INSERT INTO policy (k, a) VALUES (?, ?)",
            (pickle.dumps(sample_key), pickle.dumps(sample_probs)),
        )
        conn.commit()
        conn.close()

        loaded_agent = EnsembleCFRAgent.from_checkpoint(db_path, seed=42)
        assert isinstance(loaded_agent, EnsembleCFRAgent)
        assert len(loaded_agent.policy) > 0

    def test_in_policy_action_execution(self) -> None:
        """Verify agent executes CFR policy action when key is present in policy."""
        h0 = list(ALL_CARDS[:3])
        h1 = list(ALL_CARDS[3:6])
        state = create_initial_hand_state([h0, h1], mano=0)

        # Construct a known deterministic policy entry
        from truco_bot.agents.cfr.isomorphism import canonical_infoset_key
        key = canonical_infoset_key(state, 0)
        policy = {key: {Action.PLAY_CARD_2: 1.0}}

        agent = EnsembleCFRAgent(policy=policy, is_canonical=True, seed=42)

        # act_from_state should pick PLAY_CARD_2 with 100% certainty
        action = agent.act_from_state(state)
        assert action == Action.PLAY_CARD_2

    def test_off_tree_invokes_equity_fallback_instead_of_uniform_random(self) -> None:
        """Verify agent delegates to EquityFallback on off-tree states, avoiding blunders."""
        # Opponent played 10 de Copas (rank 5)
        opp_card = create_card(10, Suit.COPAS)
        # Hand: 4 de Copas (rank 1), 11 de Espadas (rank 6), 3 de Espadas (rank 10)
        hand = [
            create_card(4, Suit.COPAS),
            create_card(11, Suit.ESPADAS),
            create_card(3, Suit.ESPADAS),
        ]
        obs = _make_obs(hand, tc0=opp_card, current_trick=0, active_player=1, mano=0)
        mask = _make_mask([Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2])

        # EquityFallback dictates PLAY_CARD_1 (lowest winning card, rank 6)
        # A uniform random fallback would pick PLAY_CARD_0 (loses) or PLAY_CARD_2 (wasteful) ~66% of the time
        for test_seed in (1, 42, 99, 123, 777):
            seeded_agent = EnsembleCFRAgent(policy={}, seed=test_seed)
            action = seeded_agent.act(obs, mask)
            assert action == Action.PLAY_CARD_1, (
                f"Agent with seed {test_seed} failed to invoke EquityFallback, picked {action}"
            )

    def test_off_tree_truco_folding_with_low_cards(self) -> None:
        """Verify off-tree agent folds Truco with only low cards instead of random call."""
        hand_low = [
            create_card(12, Suit.COPAS),  # rank 7
            create_card(11, Suit.OROS),   # rank 6
            create_card(4, Suit.COPAS),   # rank 1
        ]
        obs = _make_obs(hand_low, current_trick=0)
        mask = _make_mask([Action.QUIERO_TRUCO, Action.NO_QUIERO_TRUCO])

        agent = EnsembleCFRAgent(policy={}, seed=42)
        action = agent.act(obs, mask)
        assert action == Action.NO_QUIERO_TRUCO, (
            f"Off-tree agent must fold Truco with low cards, got {action}"
        )

    def test_deterministic_behavior_given_same_seed(self) -> None:
        """Verify two agents with identical seeds produce identical action sequences."""
        agent_a = EnsembleCFRAgent(policy={}, seed=1337)
        agent_b = EnsembleCFRAgent(policy={}, seed=1337)

        env = TrucoHandEnv()
        env.reset(seed=100)

        for _ in range(20):
            obs = env.get_obs(env.state.active_player)
            mask = env.get_mask()
            action_a = agent_a.act(obs, mask)
            action_b = agent_b.act(obs, mask)
            assert action_a == action_b

    def test_compatibility_with_arena_match_runners(self) -> None:
        """Verify agent seamlessly executes in _run_single_match and play_duplicate_match."""
        agent = EnsembleCFRAgent(policy={}, seed=42)
        heuristic = HeuristicAgent()
        equity = EquityAgent()
        random_agent = RandomAgent()
        env = TrucoHandEnv()

        # Single match vs heuristic
        single_res = _run_single_match(agent, heuristic, env, seed=42)
        assert isinstance(single_res, int)

        # Duplicate match vs equity
        dup_equity = play_duplicate_match(agent, equity, env, seed=42)
        assert isinstance(dup_equity, int)

        # Duplicate match vs random
        dup_random = play_duplicate_match(agent, random_agent, env, seed=42)
        assert isinstance(dup_random, int)

        # Duplicate match self-play
        agent_twin = EnsembleCFRAgent(policy={}, seed=99)
        dup_self = play_duplicate_match(agent, agent_twin, env, seed=42)
        assert isinstance(dup_self, int)
