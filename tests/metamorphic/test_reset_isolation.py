"""Reset-isolation metamorphism for repeated crowd episodes."""

from __future__ import annotations

from tests.metamorphic.support import (
    BASE_MAP,
    EPISODE_SEED,
    assert_trace_equal,
    capture_episode,
    make_env,
    run_episode,
)


def test_reset_isolation_discards_previous_episode_state() -> None:
    """Reset order must not change either episode's seeded trace."""
    env = make_env(BASE_MAP)
    try:
        row_keys = tuple(pedestrian.id for pedestrian in BASE_MAP.single_pedestrians)
        first_a = capture_episode(env, row_keys=row_keys, seed=EPISODE_SEED)
        first_b = capture_episode(env, row_keys=row_keys, seed=EPISODE_SEED + 1, steps=1)
        repeated_a = capture_episode(env, row_keys=row_keys, seed=EPISODE_SEED)
    finally:
        env.close()

    reverse_env = make_env(BASE_MAP)
    try:
        reverse_b = capture_episode(
            reverse_env,
            row_keys=row_keys,
            seed=EPISODE_SEED + 1,
            steps=1,
        )
        reverse_a = capture_episode(reverse_env, row_keys=row_keys, seed=EPISODE_SEED)
    finally:
        reverse_env.close()

    fresh_a = run_episode(BASE_MAP, seed=EPISODE_SEED)
    fresh_b = run_episode(BASE_MAP, seed=EPISODE_SEED + 1, steps=1)
    assert_trace_equal(fresh_a, first_a)
    assert_trace_equal(fresh_a, repeated_a)
    assert_trace_equal(fresh_a, reverse_a)
    assert_trace_equal(fresh_b, first_b)
    assert_trace_equal(fresh_b, reverse_b)
