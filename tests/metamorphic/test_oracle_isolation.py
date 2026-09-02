"""Oracle-isolation metamorphism for actor-visible crowd observations."""

from __future__ import annotations

from tests.metamorphic.support import (
    BASE_MAP,
    assert_trace_equal,
    capture_episode,
    make_env,
    run_episode,
)


def test_oracle_trace_is_invisible_to_actor_observations() -> None:
    """Opt-in privileged traces must not alter or leak into the actor observation contract."""
    baseline = run_episode(BASE_MAP, oracle_enabled=False)
    env = make_env(BASE_MAP, oracle_enabled=True)
    try:
        privileged = capture_episode(
            env,
            row_keys=tuple(pedestrian.id for pedestrian in BASE_MAP.single_pedestrians),
        )
        assert env.sim.oracle_force_trace_payload is not None
    finally:
        env.close()

    assert all("oracle_transition_trace" not in info for info in privileged.infos)
    assert_trace_equal(baseline, privileged)
