"""Smoke guard for the multi-robot research foothold (issue #3069).

Diagnostic-only: asserts the multi-robot smoke path runs end-to-end and reports
per-agent collision/progress telemetry with an explicit diagnostic-only claim
boundary. It is not a benchmark and asserts no multi-robot performance claim.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation.run_multi_robot_smoke_issue_3069 import main, run_smoke

if TYPE_CHECKING:
    from pathlib import Path

_PER_AGENT_KEYS = {
    "agent_index",
    "step_of_episode",
    "initial_distance_to_goal",
    "final_distance_to_goal",
    "distance_progress",
    "is_robot_collision",
    "is_pedestrian_collision",
    "is_obstacle_collision",
    "is_route_complete",
    "is_waypoint_complete",
    "is_timesteps_exceeded",
}


def test_run_smoke_reports_per_agent_telemetry() -> None:
    """The smoke rollout runs and surfaces per-agent collision/progress fields."""
    report = run_smoke(num_robots=2, steps=3, seed=0)

    # Diagnostic-only claim boundary is explicit and conservative.
    assert report["schema_version"] == "multi_robot_smoke.v1"
    assert report["issue"] == 3069
    assert report["evidence_tier"] == "smoke"
    assert report["claim_boundary"] == "diagnostic_only"
    assert report["multi_robot_benchmark_claim"] is False

    # The path executed at least one step against the requested robot count.
    assert report["config"]["num_robots"] == 2
    assert report["rollout"]["steps_executed"] >= 1
    assert isinstance(report["rollout"]["total_reward"], float)

    # Per-agent telemetry is reported for every robot.
    assert len(report["agents"]) == 2
    for index, agent in enumerate(report["agents"]):
        assert agent["agent_index"] == index
        assert _PER_AGENT_KEYS <= set(agent)
        for flag in ("is_robot_collision", "is_route_complete", "is_timesteps_exceeded"):
            assert isinstance(agent[flag], bool)

    # Inter-robot collision fields are reported (not deferred).
    inter_robot = report["inter_robot"]
    assert isinstance(inter_robot["any_robot_collision"], bool)
    assert inter_robot["robot_collision_count"] == sum(
        1 for agent in report["agents"] if agent["is_robot_collision"]
    )


def test_main_writes_json_report(tmp_path: Path) -> None:
    """The CLI runs and writes a JSON smoke report with the claim boundary."""
    out_path = tmp_path / "multi_robot_smoke.json"
    exit_code = main(
        ["--num-robots", "2", "--steps", "2", "--seed", "0", "--json-output", str(out_path)]
    )

    assert exit_code == 0
    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["claim_boundary"] == "diagnostic_only"
    assert payload["multi_robot_benchmark_claim"] is False
    assert len(payload["agents"]) == 2
