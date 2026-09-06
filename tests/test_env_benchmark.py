"""High-performance throughput benchmark suite for Truco simulation engine."""

import random
import time

import pytest

from truco_bot.core.actions import Action
from truco_bot.core.deck import deal
from truco_bot.core.state import create_initial_hand_state, get_legal_actions
from truco_bot.core.state import step as core_step
from truco_bot.env.engine import TrucoHandEnv

TARGET_THROUGHPUT_STEPS_PER_SEC = 200_000


def benchmark_state_step_throughput(
    target_steps: int = 100_000, seed: int = 42
) -> dict[str, float | int]:
    """Benchmark raw throughput of core state.step() across target_steps random actions."""
    rng = random.Random(seed)
    steps = 0
    hands_completed = 0
    step_duration_acc = 0.0

    wall_start = time.perf_counter()
    while steps < target_steps:
        hands, _ = deal(2, 3)
        state = create_initial_hand_state(hands, mano=0)

        while not state.is_hand_done:
            actions = get_legal_actions(state)
            action = rng.choice(actions)

            t0 = time.perf_counter()
            state = core_step(state, action)
            t1 = time.perf_counter()

            step_duration_acc += t1 - t0
            steps += 1
            if steps >= target_steps:
                break

        if state.is_hand_done:
            hands_completed += 1

    wall_time = time.perf_counter() - wall_start
    raw_steps_per_sec = steps / step_duration_acc if step_duration_acc > 0 else 0.0
    wall_steps_per_sec = steps / wall_time if wall_time > 0 else 0.0

    return {
        "steps": steps,
        "step_time": step_duration_acc,
        "wall_time": wall_time,
        "raw_steps_per_sec": raw_steps_per_sec,
        "wall_steps_per_sec": wall_steps_per_sec,
        "hands_completed": hands_completed,
    }


def benchmark_env_step_throughput(
    target_steps: int = 100_000, seed: int = 42
) -> dict[str, float | int]:
    """Benchmark raw throughput of TrucoHandEnv.step() across target_steps random actions."""
    env = TrucoHandEnv()
    rng = random.Random(seed)
    steps = 0
    hands_completed = 0
    step_duration_acc = 0.0

    wall_start = time.perf_counter()
    while steps < target_steps:
        _, mask = env.reset()
        done = False

        while not done:
            legal_indices = [i for i, is_legal in enumerate(mask) if is_legal]
            action = Action(rng.choice(legal_indices))

            t0 = time.perf_counter()
            _, _, done, mask = env.step(action)
            t1 = time.perf_counter()

            step_duration_acc += t1 - t0
            steps += 1
            if steps >= target_steps:
                break

        if done:
            hands_completed += 1

    wall_time = time.perf_counter() - wall_start
    raw_steps_per_sec = steps / step_duration_acc if step_duration_acc > 0 else 0.0
    wall_steps_per_sec = steps / wall_time if wall_time > 0 else 0.0

    return {
        "steps": steps,
        "step_time": step_duration_acc,
        "wall_time": wall_time,
        "raw_steps_per_sec": raw_steps_per_sec,
        "wall_steps_per_sec": wall_steps_per_sec,
        "hands_completed": hands_completed,
    }


@pytest.mark.benchmark
def test_raw_state_step_throughput() -> None:
    """Validate that raw state.step() performance meets >= 200,000 steps/second."""
    target_steps = 100_000
    metrics = benchmark_state_step_throughput(target_steps=target_steps, seed=42)

    steps = int(metrics["steps"])
    elapsed = float(metrics["step_time"])
    steps_per_sec = float(metrics["raw_steps_per_sec"])
    hands = int(metrics["hands_completed"])
    wall = float(metrics["wall_time"])

    print("\n" + "=" * 60)
    print("BENCHMARK: core.state.step (Raw Throughput)")
    print("=" * 60)
    print(f"  Total Steps Completed : {steps:,}")
    print(f"  Total Hands Completed : {hands:,}")
    print(f"  Step Execution Time   : {elapsed:.4f} seconds")
    print(f"  Total Wall-Clock Time : {wall:.4f} seconds")
    print(f"  Raw Throughput Rate   : {steps_per_sec:,.2f} steps/second")
    print(f"  Required Threshold    : {TARGET_THROUGHPUT_STEPS_PER_SEC:,} steps/second")
    print(
        f"  Performance Ratio     : {steps_per_sec / TARGET_THROUGHPUT_STEPS_PER_SEC:.2f}x threshold"
    )
    print("=" * 60)

    assert steps_per_sec >= TARGET_THROUGHPUT_STEPS_PER_SEC, (
        f"state.step throughput {steps_per_sec:,.2f} below target {TARGET_THROUGHPUT_STEPS_PER_SEC:,}"
    )


@pytest.mark.benchmark
def test_raw_env_step_throughput() -> None:
    """Validate that raw TrucoHandEnv.step() performance meets >= 200,000 steps/second."""
    target_steps = 100_000
    metrics = benchmark_env_step_throughput(target_steps=target_steps, seed=42)

    steps = int(metrics["steps"])
    elapsed = float(metrics["step_time"])
    steps_per_sec = float(metrics["raw_steps_per_sec"])
    hands = int(metrics["hands_completed"])
    wall = float(metrics["wall_time"])

    print("\n" + "=" * 60)
    print("BENCHMARK: env.engine.TrucoHandEnv.step (Raw Throughput)")
    print("=" * 60)
    print(f"  Total Steps Completed : {steps:,}")
    print(f"  Total Hands Completed : {hands:,}")
    print(f"  Step Execution Time   : {elapsed:.4f} seconds")
    print(f"  Total Wall-Clock Time : {wall:.4f} seconds")
    print(f"  Raw Throughput Rate   : {steps_per_sec:,.2f} steps/second")
    print(f"  Required Threshold    : {TARGET_THROUGHPUT_STEPS_PER_SEC:,} steps/second")
    print(
        f"  Performance Ratio     : {steps_per_sec / TARGET_THROUGHPUT_STEPS_PER_SEC:.2f}x threshold"
    )
    print("=" * 60)

    assert steps_per_sec >= TARGET_THROUGHPUT_STEPS_PER_SEC, (
        f"TrucoHandEnv throughput {steps_per_sec:,.2f} below target {TARGET_THROUGHPUT_STEPS_PER_SEC:,}"
    )


if __name__ == "__main__":
    print("Running Truco simulation throughput benchmarks...")
    test_raw_state_step_throughput()
    test_raw_env_step_throughput()
