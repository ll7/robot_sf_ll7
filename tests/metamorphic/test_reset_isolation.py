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
    """A reset after another episode must reproduce a fresh seeded episode exactly."""
    env = make_env(BASE_MAP)
    try:
        row_keys = tuple(pedestrian.id for pedestrian in BASE_MAP.single_pedestrians)
        capture_episode(env, row_keys=row_keys, seed=EPISODE_SEED)
        capture_episode(env, row_keys=row_keys, seed=EPISODE_SEED + 1, steps=1)
        repeated = capture_episode(env, row_keys=row_keys, seed=EPISODE_SEED)
    finally:
        env.close()

    fresh = run_episode(BASE_MAP, seed=EPISODE_SEED)
    assert_trace_equal(fresh, repeated)
