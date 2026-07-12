"""Regression tests for per-arm learned-policy caching (issue #5347)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from robot_sf.benchmark import map_runner


def test_cached_policy_is_reset_between_episodes_and_closed_after_arm(
    monkeypatch, tmp_path
) -> None:
    """Two cached episodes produce fresh state while constructing PPO only once."""
    builder_calls = 0
    outputs: list[int] = []
    close_calls = 0

    def stateful_policy(_obs):
        stateful_policy.calls += 1
        return stateful_policy.calls

    stateful_policy.calls = 0

    def reset(*, seed: int | None = None) -> None:
        del seed
        stateful_policy.calls = 0

    def close() -> None:
        nonlocal close_calls
        close_calls += 1

    stateful_policy._planner_reset = reset
    stateful_policy._planner_close = close

    def fake_build_policy(*_args, **_kwargs):
        nonlocal builder_calls
        builder_calls += 1
        return stateful_policy, {"algorithm": "ppo"}

    def fake_run_map_episode(*_args, close_policy, policy_builder, **_kwargs):
        assert close_policy is False
        policy, _metadata = policy_builder("ppo", {"model_id": "fixture"})
        policy._planner_reset(seed=7)
        outputs.append(policy({}))
        return {"scenario_id": "fixture", "seed": 7}

    def fake_execute_map_jobs(*, jobs, run_map_job, **_kwargs):
        for scenario, seed in jobs:
            run_map_job((scenario, seed, {"scenario_path": str(tmp_path / "matrix.yaml")}))
        return SimpleNamespace()

    monkeypatch.setattr(map_runner, "_build_policy", fake_build_policy)
    monkeypatch.setattr(map_runner, "_run_map_episode", fake_run_map_episode)
    monkeypatch.setattr(map_runner, "_execute_map_jobs", fake_execute_map_jobs)

    map_runner._run_map_jobs_with_policy_cache(
        jobs=[({"name": "first"}, 7), ({"name": "second"}, 7)],
        fixed_params={"scenario_path": str(Path(tmp_path / "matrix.yaml"))},
        out_path=tmp_path / "episodes.jsonl",
        schema={},
        workers=1,
    )

    assert builder_calls == 1
    assert outputs == [1, 1]
    assert close_calls == 1
