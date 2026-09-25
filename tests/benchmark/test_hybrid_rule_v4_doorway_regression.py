"""Full-simulation doorway regressions for hybrid v4 (issue #9726).

In the 0.0.7 benchmark data, hybrid v3 collided with a pedestrian while still
moving at 1-1.6 m/s on ``classic_doorway_high`` seed 111 and
``classic_doorway_low`` seed 120. These episodes pin that v4 no longer does,
and that the v3 release config still reproduces its recorded outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_MATRIX = (
    REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
)
EPISODE_SCHEMA = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
CANDIDATES = REPO_ROOT / "configs/policy_search/candidates"
V3_CONFIG = CANDIDATES / "hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml"
V4_CONFIG = CANDIDATES / "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml"
CELLS = [("classic_doorway_high", 111), ("classic_doorway_low", 120)]


def _run_episode(scenario_name: str, seed: int, config: Path, tmp_path: Path) -> dict:
    scenarios = [s for s in load_scenarios(SCENARIO_MATRIX) if s.get("name") == scenario_name]
    assert scenarios, scenario_name
    out_path = tmp_path / f"{config.stem}_{scenario_name}_{seed}.jsonl"
    run_map_batch(
        [dict(scenarios[0], seeds=[seed])],
        out_path,
        EPISODE_SCHEMA,
        scenario_path=SCENARIO_MATRIX,
        horizon=600,
        dt=0.1,
        algo="hybrid_rule_local_planner",
        algo_config_path=str(config),
        benchmark_profile="experimental",
        workers=1,
        resume=False,
    )
    return json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])


def _pedestrian_contacts(record: dict) -> list[dict]:
    events = (record.get("event_ledger") or {}).get("collision_events") or []
    return [event for event in events if event.get("collision_partner_type") == "pedestrian"]


@pytest.mark.slow
@pytest.mark.parametrize(("scenario_name", "seed"), CELLS)
def test_v4_avoids_the_v3_doorway_pedestrian_collision(
    scenario_name: str, seed: int, tmp_path: Path
) -> None:
    """v3 collides with a pedestrian; the v4 twin does not."""
    v3 = _run_episode(scenario_name, seed, V3_CONFIG, tmp_path)
    assert v3["termination_reason"] == "collision"
    assert _pedestrian_contacts(v3), "v3 outcome changed: expected its recorded pedestrian contact"

    v4 = _run_episode(scenario_name, seed, V4_CONFIG, tmp_path)
    assert not _pedestrian_contacts(v4), _pedestrian_contacts(v4)
    assert v4["metrics"].get("ped_collision_count", 0) == 0
    runtime = v4["algorithm_metadata"].get("planner_runtime") or {}
    speed_safety = runtime.get("speed_safety") or {}
    assert speed_safety.get("drive_limits", {}).get("source", "").startswith("env_robot_config")
