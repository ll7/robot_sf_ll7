"""Regression contract for the bounded issue #9645 search space."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random

import yaml

from robot_sf.adversarial.bundle import build_candidate_payload
from robot_sf.adversarial.config import SearchSpaceConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR_NAME = "issue_9645_bounded_falsification_2026-09-24"
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence" / EVIDENCE_DIR_NAME
AS_RUN_SPACE_PATH = EVIDENCE_DIR / "payload/inputs/issue_9645_pilot_space.v1.yaml"
LIVE_SPACE_PATH = REPO_ROOT / "configs/adversarial/issue_9645_pilot_space.v1.yaml"


def test_issue_9645_space_varies_only_effective_dimensions_and_freezes_environment() -> None:
    """Keep the pilot's searched controls effective and its simulator seed fixed."""
    search_space_path = AS_RUN_SPACE_PATH
    template_path = REPO_ROOT / "configs/scenarios/templates/crossing_ttc.yaml"
    search_space = SearchSpaceConfig.from_file(search_space_path)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    template_scenario = template["scenarios"][0]

    assert search_space.pedestrian_id is None
    assert (search_space.spawn_time_s.min, search_space.spawn_time_s.max) == (0.0, 0.0)
    assert (search_space.pedestrian_delay_s.min, search_space.pedestrian_delay_s.max) == (0.0, 0.0)
    assert (search_space.scenario_seed.min, search_space.scenario_seed.max) == (123, 123)
    assert (search_space.start_x.min, search_space.start_x.max) == (2.5, 3.0)
    assert (search_space.goal_x.min, search_space.goal_x.max) == (7.0, 9.0)
    assert (search_space.goal_y.min, search_space.goal_y.max) == (2.5, 4.0)
    assert (search_space.pedestrian_speed_mps.min, search_space.pedestrian_speed_mps.max) == (
        0.8,
        1.4,
    )

    candidate = search_space.sample_candidate(Random(1101))
    specialized, route_payload = build_candidate_payload(
        candidate,
        index=0,
        template_scenario=template_scenario,
        pedestrian_id=search_space.pedestrian_id,
        pedestrian_route_mode=search_space.pedestrian_route_mode,
    )

    assert candidate.scenario_seed == 123
    assert candidate.spawn_time_s == candidate.pedestrian_delay_s == 0.0
    assert specialized["simulation_config"]["route_spawn_seed"] == 123
    assert specialized["simulation_config"]["peds_speed_mult"] == candidate.pedestrian_speed_mps
    assert route_payload["robot_routes"][0]["waypoints"] == [
        candidate.start.as_waypoint(),
        candidate.goal.as_waypoint(),
    ]
    assert route_payload["ped_routes"] == []
    assert "single_pedestrians" not in specialized


def test_live_config_is_reseeded_and_otherwise_matches_as_run_input() -> None:
    """The in-tree config uses a dev seed; only scenario_seed differs from the as-run input."""
    live = yaml.safe_load(LIVE_SPACE_PATH.read_text(encoding="utf-8"))
    as_run = yaml.safe_load(AS_RUN_SPACE_PATH.read_text(encoding="utf-8"))

    assert live["variables"]["scenario_seed"] == {"min": 1001, "max": 1001}
    assert as_run["variables"]["scenario_seed"] == {"min": 123, "max": 123}
    live["variables"].pop("scenario_seed")
    as_run["variables"].pop("scenario_seed")
    live.pop("description")
    as_run.pop("description")
    assert live == as_run
    assert SearchSpaceConfig.from_file(LIVE_SPACE_PATH).scenario_seed.min == 1001


def test_manifest_records_holdout_quarantine_and_reseed() -> None:
    """The evidence manifest states the holdout use, the forbidden uses and the re-seed."""
    manifest = json.loads((EVIDENCE_DIR / "evidence_bundle_manifest.json").read_text("utf-8"))
    quarantine = manifest["quarantine"]

    assert quarantine["reason"] == "holdout_seed_used"
    assert quarantine["holdout_seed"] == 123
    assert quarantine["used_on"] == "2026-09-24"
    assert set(quarantine["must_not_inform"]) == {
        "hybrid_v4_tuning",
        "v4_freeze",
        "snqi_calibration",
        "anchors",
        "0.0.8_campaign",
    }
    reseed = quarantine["config_path_reseed"]
    assert reseed["path"] == "configs/adversarial/issue_9645_pilot_space.v1.yaml"
    assert reseed["as_run_input"] == "inputs/issue_9645_pilot_space.v1.yaml"
    assert reseed["as_run_sha256"] == hashlib.sha256(AS_RUN_SPACE_PATH.read_bytes()).hexdigest()
    assert reseed["current_scenario_seed"] == 1001
