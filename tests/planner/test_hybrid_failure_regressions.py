"""Development regressions for reflection and progress defects in enabled switches."""

import numpy as np
import pytest

from scripts.validation import run_hybrid_feasibility_diagnostics as diagnostic
from tests.metamorphic import planner_arms as harness


@pytest.mark.parametrize("reflection", (harness.mirror_y, harness.mirror_x))
def test_physical_static_rollout_reflects_without_raster_score_bias(monkeypatch, reflection):
    """Reflecting the entire physical scene must reflect the executed robot trajectory."""
    algo, raw = harness.resolve_release_algo_config(
        "hybrid_rule_local_planner", harness.HYBRID_V4_DIAGNOSTIC_CONFIG, "metamorphic"
    )
    raw.update(physical_static_exclusion_enabled=True, goal_next_validity_enabled=False)
    monkeypatch.setattr(harness, "release_arm", lambda _, **__: (algo, dict(raw)))
    original_env = harness.robot_env_config

    def config(*args, **kwargs):
        result = original_env(*args, **kwargs)
        result.include_goal_next_valid = False
        return result

    monkeypatch.setattr(harness, "robot_env_config", config)
    direct = harness.run_arm_episode(
        "reflection-control", harness.interaction_scene(), seed=1001, max_steps=60
    )
    reflected = harness.run_arm_episode(
        "reflection-control", harness.interaction_scene(reflection), seed=1001, max_steps=60
    )
    expected = np.asarray([reflection(p[:2]) for p in direct.poses])
    np.testing.assert_allclose(np.asarray(reflected.poses)[:, :2], expected, atol=0.0001, rtol=0)


@pytest.mark.parametrize(
    ("scenario", "seed", "arm", "horizon"),
    (
        ("francis2023_narrow_hallway", 1001, "static_only", 400),
        ("francis2023_robot_crowding", 1013, "static_only", 600),
        ("francis2023_exiting_room", 1026, "goal_validity_with_sensor", 600),
        ("francis2023_exiting_elevator", 1026, "goal_validity_with_sensor", 400),
    ),
)
def test_enabled_switch_completes_the_reproduced_failure_cell(
    tmp_path, scenario, seed, arm, horizon
):
    """The real scenario, drive, sensor and scorer must complete the failing development cell."""
    result = diagnostic.run_cell((scenario, seed, arm, False, str(tmp_path), horizon))
    assert result["outcome"] == "success", result["outcome"]
    assert result["runtime"]["fallback_count"] == 0
