"""Hypothesis property-based invariant tests for Truco simulation engine."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from truco_bot.core.actions import Action
from truco_bot.core.deck import deal
from truco_bot.core.state import create_initial_hand_state, get_legal_actions
from truco_bot.env.engine import TrucoHandEnv
from truco_bot.env.masking import get_action_mask, get_action_mask_bits
from truco_bot.env.obs import get_observation


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), mano=st.sampled_from([0, 1]))
@settings(max_examples=150)
def test_random_actions_respecting_mask_never_cause_exception_or_illegal_transition(
    seed: int, mano: int
) -> None:
    """Legal action selections from get_action_mask() must never raise or cause illegal transitions."""
    rng = random.Random(seed)
    env = TrucoHandEnv()
    hands, _ = deal(2, 3, seed=seed)
    env.state = create_initial_hand_state(hands, mano=mano)
    mask = env.get_mask()

    done = False
    while not done:
        legal_actions = [Action(i) for i, is_legal in enumerate(mask) if is_legal]
        core_legal = get_legal_actions(env.state)

        # Invariant: Mask legal actions must match core legal actions
        assert set(legal_actions) == set(core_legal)
        assert len(legal_actions) > 0, "Non-terminated state must have at least one legal action"

        chosen_action = rng.choice(legal_actions)
        assert chosen_action in core_legal, f"Illegal action selected: {chosen_action}"

        state, _pts, done, mask = env.step(chosen_action)
        assert state is not None
        assert state.active_player in (0, 1)

    # Invariant: Terminal state properties
    assert env.state.is_hand_done
    assert env.state.winner in (0, 1)
    assert all(not is_legal for is_legal in mask)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), mano=st.sampled_from([0, 1]))
@settings(max_examples=150)
def test_random_hand_rollout_strictly_terminates_within_20_steps(seed: int, mano: int) -> None:
    """Every random hand rollout must strictly terminate within 20 steps."""
    rng = random.Random(seed)
    env = TrucoHandEnv()
    hands, _ = deal(2, 3, seed=seed)
    env.state = create_initial_hand_state(hands, mano=mano)
    mask = env.get_mask()

    steps = 0
    done = False
    while not done:
        legal_indices = [i for i, is_legal in enumerate(mask) if is_legal]
        assert len(legal_indices) > 0
        action = Action(rng.choice(legal_indices))
        _, _, done, mask = env.step(action)
        steps += 1
        assert steps <= 20, f"Hand rollout failed to terminate within 20 steps (took {steps})"

    assert done
    assert env.state.is_hand_done


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), mano=st.sampled_from([0, 1]))
@settings(max_examples=150)
def test_points_won_never_exceed_theoretical_maximum(seed: int, mano: int) -> None:
    """Points won in a hand must never exceed theoretical bounds (1 to 4 for truco hand points)."""
    rng = random.Random(seed)
    env = TrucoHandEnv()
    hands, _ = deal(2, 3, seed=seed)
    env.state = create_initial_hand_state(hands, mano=mano)
    mask = env.get_mask()

    done = False
    while not done:
        legal_indices = [i for i, is_legal in enumerate(mask) if is_legal]
        action = Action(rng.choice(legal_indices))
        _, pts, done, mask = env.step(action)

        # Points during intermediate steps are 0
        if not done:
            assert pts == 0

    # Invariant: Hand outcome points
    # Hand points for truco resolution: 1 (no truco/no quiero), 2 (truco), 3 (retruco), 4 (vale cuatro)
    assert 1 <= env.state.points_won <= 4, f"Invalid points_won: {env.state.points_won}"
    assert env.state.points_won == pts

    # Invariant: Scores accumulated (e.g. from envido) never exceed max_score or go negative
    assert 0 <= env.state.score_p0 <= env.state.max_score
    assert 0 <= env.state.score_p1 <= env.state.max_score


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), mano=st.sampled_from([0, 1]))
@settings(max_examples=150)
def test_observation_tuples_never_contain_invalid_indices(seed: int, mano: int) -> None:
    """Observation tuples for both players must always have valid dimensions and bounded indices."""
    rng = random.Random(seed)
    env = TrucoHandEnv()
    hands, _ = deal(2, 3, seed=seed)
    env.state = create_initial_hand_state(hands, mano=mano)
    mask = env.get_mask()

    done = False
    while not done:
        for player in (0, 1):
            obs = env.get_obs(player)
            core_obs = get_observation(env.state, player)
            assert obs == core_obs
            assert isinstance(obs, tuple)
            assert len(obs) == 12

            c0, c1, c2, tc0, tc1, s0, s1, trick, truco_lvl, env_res, active_p, m = obs

            # Hand card IDs: 0 is empty slot, 1..40 are valid card IDs
            assert 0 <= c0 <= 40, f"Invalid card index c0: {c0}"
            assert 0 <= c1 <= 40, f"Invalid card index c1: {c1}"
            assert 0 <= c2 <= 40, f"Invalid card index c2: {c2}"

            # Table trick cards: 0 is empty slot, 1..40 are valid card IDs
            assert 0 <= tc0 <= 40, f"Invalid trick card tc0: {tc0}"
            assert 0 <= tc1 <= 40, f"Invalid trick card tc1: {tc1}"

            # Scores
            assert 0 <= s0 <= env.state.max_score
            assert 0 <= s1 <= env.state.max_score

            # Current trick (0, 1, 2)
            assert 0 <= trick <= 2

            # Truco level (1..4)
            assert 1 <= truco_lvl <= 4

            # Envido resolved flag
            assert env_res in (0, 1)

            # Active player
            assert active_p in (0, 1)

            # Mano player
            assert m in (0, 1)

        legal_indices = [i for i, is_legal in enumerate(mask) if is_legal]
        action = Action(rng.choice(legal_indices))
        _, _, done, mask = env.step(action)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), mano=st.sampled_from([0, 1]))
@settings(max_examples=150)
def test_action_mask_matches_action_mask_bits_for_all_states(seed: int, mano: int) -> None:
    """get_action_mask() list must strictly match get_action_mask_bits() integer bitmask across all states."""
    rng = random.Random(seed)
    env = TrucoHandEnv()
    hands, _ = deal(2, 3, seed=seed)
    env.state = create_initial_hand_state(hands, mano=mano)

    done = False
    while not done:
        mask = env.get_mask()
        bits = get_action_mask_bits(env.state)
        direct_mask = get_action_mask(env.state)

        assert mask == direct_mask
        assert len(mask) == 26

        # Invariant: Each bit corresponds 1:1 with boolean mask
        for i in range(26):
            expected_bit = bool(bits & (1 << i))
            assert mask[i] is expected_bit, (
                f"Bit mismatch at index {i}: mask={mask[i]}, bit={expected_bit}"
            )

        # Invariant: Sum of bits matches integer value
        reconstructed_bits = sum(1 << i for i, is_legal in enumerate(mask) if is_legal)
        assert reconstructed_bits == bits

        legal_indices = [i for i, is_legal in enumerate(mask) if is_legal]
        action = Action(rng.choice(legal_indices))
        _, _, done, _ = env.step(action)

    # Invariant: Also verify for terminal state
    terminal_mask = env.get_mask()
    terminal_bits = get_action_mask_bits(env.state)
    assert terminal_bits == 0
    assert all(not is_legal for is_legal in terminal_mask)
