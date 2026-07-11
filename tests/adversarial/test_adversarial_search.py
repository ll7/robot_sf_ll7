"""Tests for adversarial scenario search, materialization, and certification."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from robot_sf.adversarial import certification, objectives, search
from robot_sf.adversarial.attribution import attribution_from_episode_record, attribution_from_error
from robot_sf.adversarial.bundle import write_trajectory_csv
from robot_sf.adversarial.certification import failed_status, not_available_status, passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    MultiPedAdversarialConfig,
    MultiPedCandidateSpec,
    Pose2D,
    SearchConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.materialize import (
    materialize_manifest_route_overrides,
    materialize_manifest_scenario_payload,
    materialize_manifest_single_pedestrian_override,
    materialize_multi_ped_scenario_payload,
    materialize_multi_ped_single_pedestrian_overrides,
)
from robot_sf.adversarial.runtime import (
    build_multi_ped_adversarial_robot_config,
    multi_ped_config_to_single_pedestrian_definitions,
)
from robot_sf.adversarial.samplers import (
    CoordinateRefinementSampler,
    OptunaCandidateSampler,
    RandomCandidateSampler,
)
from robot_sf.adversarial.scenario_manifest import build_manifest
from robot_sf.adversarial.seed_sensitivity import (
    SeedSensitivityPerturbation,
    run_seed_sensitivity,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.ped_npc.ped_population import populate_single_pedestrians
from scripts.tools.compare_adversarial_samplers import (
    _comparison_row_from_manifest,
    run_sampler_comparison,
)

_MULTI_PED_FAMILY_FIXTURES = (
    (
        Path("configs/adversarial/group_squeeze_multi_ped_example.yaml"),
        "group_squeeze",
        ("left_blocker", "right_blocker"),
    ),
    (
        Path("configs/adversarial/doorway_blocker_multi_ped_example.yaml"),
        "doorway_blocker",
        ("door_left", "door_right"),
    ),
)


def _write_template(path: Path) -> None:
    """Write a minimal scenario template fixture."""
    path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "template",
                        "map_id": "classic_cross_trap",
                        "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
                        "robot_config": {},
                        "metadata": {"archetype": "test"},
                        "seeds": [1],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_space(path: Path, *, min_distance: float = 0.5) -> None:
    """Write a search-space fixture with a configurable distance constraint."""
    path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": min_distance},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    *,
    require_certification: bool = False,
    workers: int = 1,
) -> SearchConfig:
    """Build a search config backed by temporary template and space files."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    return SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=2,
        seed=123,
        workers=workers,
        require_certification=require_certification,
    )


class _SequenceSampler:
    """Sampler that returns a prepared candidate sequence."""

    def __init__(self, candidates: list[CandidateSpec]) -> None:
        self._candidates = list(candidates)

    def sample(self) -> CandidateSpec:
        """Return the next prepared candidate."""
        return self._candidates.pop(0)


def _candidate(seed: int, *, goal_x: float = 5.0) -> CandidateSpec:
    """Build a candidate fixture with configurable seed and goal x-coordinate."""
    return CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(goal_x, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=seed,
    )


def _runtime_base_map(*, obstacles: list[Obstacle] | None = None) -> MapDefinition:
    """Build a compact map fixture for multi-pedestrian runtime checks."""
    width, height = 8.0, 6.0
    robot_spawn_zones = [((0.5, 0.5), (1.0, 0.5), (1.0, 1.0))]
    robot_goal_zones = [((7.0, 5.0), (7.5, 5.0), (7.5, 5.5))]
    robot_routes = [
        GlobalRoute(
            spawn_id=0,
            goal_id=0,
            waypoints=[(0.75, 0.75), (4.0, 3.0), (7.25, 5.25)],
            spawn_zone=robot_spawn_zones[0],
            goal_zone=robot_goal_zones[0],
        )
    ]
    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles or [],
        robot_spawn_zones=robot_spawn_zones,
        ped_spawn_zones=[],
        robot_goal_zones=robot_goal_zones,
        bounds=[
            (0.0, width, 0.0, 0.0),
            (0.0, width, height, height),
            (0.0, 0.0, 0.0, height),
            (width, width, 0.0, height),
        ],
        robot_routes=robot_routes,
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[],
    )


def _runtime_multi_ped_config(
    *,
    second_start_y: float = 3.2,
    first_start: Pose2D | None = None,
) -> MultiPedAdversarialConfig:
    """Build a two-pedestrian adversarial runtime config."""
    return MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=41,
        pedestrians=[
            MultiPedCandidateSpec(
                id="left_blocker",
                start=first_start or Pose2D(1.5, 2.0),
                goal=Pose2D(6.5, 2.0),
                spawn_time_s=0.2,
                speed_mps=1.1,
                delay_s=0.1,
                metadata={"lane": "left"},
            ),
            MultiPedCandidateSpec(
                id="right_blocker",
                start=Pose2D(1.5, second_start_y),
                goal=Pose2D(6.5, second_start_y),
                spawn_time_s=0.4,
                speed_mps=1.2,
                metadata={"lane": "right"},
            ),
        ],
    )


def test_search_config_from_files_validates_candidate(tmp_path: Path) -> None:
    """SearchConfig should load files and validate a sampled candidate."""
    config = _config(tmp_path)

    candidate = config.search_space.sample_candidate(__import__("random").Random(1))

    assert config.search_space.validate_candidate(candidate) == []
    assert candidate.scenario_seed == 7


def test_search_space_validates_all_configured_candidate_ranges(tmp_path: Path) -> None:
    """Search-space validation should flag every out-of-range candidate field."""
    config = _config(tmp_path)

    errors = config.search_space.validate_candidate(
        CandidateSpec(
            start=Pose2D(1.0, 2.0),
            goal=Pose2D(5.0, 2.0),
            spawn_time_s=1.0,
            pedestrian_speed_mps=2.0,
            pedestrian_delay_s=1.0,
            scenario_seed=8,
        )
    )

    assert "spawn_time_s outside search space" in errors
    assert "pedestrian_speed_mps outside search space" in errors
    assert "pedestrian_delay_s outside search space" in errors
    assert "scenario_seed outside search space" in errors


def test_search_space_rejects_invalid_min_start_goal_distance() -> None:
    """Search-space config should reject negative min-distance constraints."""
    payload = {
        "variables": {
            "start_x": {"min": 0, "max": 1},
            "start_y": {"min": 0, "max": 1},
            "goal_x": {"min": 2, "max": 3},
            "goal_y": {"min": 2, "max": 3},
        },
        "constraints": {"min_start_goal_distance_m": -1.0},
    }

    with pytest.raises(ValueError, match="min_start_goal_distance_m"):
        SearchSpaceConfig.from_mapping(payload)


def test_search_space_rejects_non_integral_seed_bounds() -> None:
    """Scenario seed ranges must describe a discrete integer sampling space."""
    payload = {
        "variables": {
            "start_x": {"min": 0, "max": 1},
            "start_y": {"min": 0, "max": 1},
            "goal_x": {"min": 2, "max": 3},
            "goal_y": {"min": 2, "max": 3},
            "scenario_seed": {"min": 1.5, "max": 2.5},
        }
    }

    with pytest.raises(ValueError, match="scenario_seed bounds must be integers"):
        SearchSpaceConfig.from_mapping(payload)


def test_multi_ped_adversarial_config_parses_and_serializes_yaml(tmp_path: Path) -> None:
    """Multi-ped adversarial candidates should have a deterministic schema contract."""
    path = tmp_path / "multi_ped.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "adversarial-multi-ped.v1",
                "family": "group_squeeze",
                "scenario_seed": 41,
                "constraints": {"min_start_goal_distance_m": 1.0},
                "pedestrians": [
                    {
                        "id": "left_blocker",
                        "start": {"x": 1.0, "y": 2.0},
                        "goal": {"x": 5.0, "y": 2.0},
                        "spawn_time_s": 0.5,
                        "speed_mps": 1.1,
                        "delay_s": 0.25,
                    },
                    {
                        "id": "right_blocker",
                        "start": {"x": 1.0, "y": 3.0},
                        "goal": {"x": 5.0, "y": 3.0},
                        "spawn_time_s": 0.75,
                        "speed_mps": 1.2,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = MultiPedAdversarialConfig.from_file(path)

    assert config.schema_version == "adversarial-multi-ped.v1"
    assert config.family == "group_squeeze"
    assert config.scenario_seed == 41
    assert config.validate() == []
    assert [ped.id for ped in config.pedestrians] == ["left_blocker", "right_blocker"]
    assert config.to_json() == {
        "schema_version": "adversarial-multi-ped.v1",
        "family": "group_squeeze",
        "scenario_seed": 41,
        "constraints": {"min_start_goal_distance_m": 1.0},
        "pedestrians": [
            {
                "id": "left_blocker",
                "start": {"x": 1.0, "y": 2.0, "theta": 0.0},
                "goal": {"x": 5.0, "y": 2.0, "theta": 0.0},
                "spawn_time_s": 0.5,
                "speed_mps": 1.1,
                "delay_s": 0.25,
                "metadata": {},
            },
            {
                "id": "right_blocker",
                "start": {"x": 1.0, "y": 3.0, "theta": 0.0},
                "goal": {"x": 5.0, "y": 3.0, "theta": 0.0},
                "spawn_time_s": 0.75,
                "speed_mps": 1.2,
                "delay_s": 0.0,
                "metadata": {},
            },
        ],
    }


def test_multi_ped_adversarial_config_reports_validation_errors() -> None:
    """Invalid multi-ped contracts should fail before runtime integration."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=-1,
        min_start_goal_distance_m=2.0,
        pedestrians=[
            MultiPedCandidateSpec(
                id="blocker",
                start=Pose2D(1.0, 2.0),
                goal=Pose2D(1.5, 2.0),
                spawn_time_s=-0.1,
                speed_mps=0.0,
            ),
            MultiPedCandidateSpec(
                id="blocker",
                start=Pose2D(float("nan"), 3.0),
                goal=Pose2D(5.0, 3.0),
                spawn_time_s=0.0,
                speed_mps=1.0,
                delay_s=-0.5,
            ),
        ],
    )

    errors = config.validate()

    assert "scenario_seed must be non-negative" in errors
    assert any("pedestrians ids must be unique" in err for err in errors)
    assert any("duplicates: blocker" in err for err in errors)
    assert "pedestrians[0].spawn_time_s must be non-negative" in errors
    assert "pedestrians[0].speed_mps must be positive" in errors
    assert any(err.startswith("pedestrians[0] start and goal distance") for err in errors)
    assert "pedestrians[1].start.x must be finite" in errors
    assert "pedestrians[1].delay_s must be non-negative" in errors


def test_multi_ped_adversarial_example_config_is_valid() -> None:
    """The checked-in example should stay aligned with the schema parser."""
    config = MultiPedAdversarialConfig.from_file(
        Path("configs/adversarial/group_squeeze_multi_ped_example.yaml")
    )

    assert config.family == "group_squeeze"
    assert len(config.pedestrians) == 2
    assert config.validate() == []


def test_multi_ped_config_materializes_single_pedestrian_overrides() -> None:
    """Multi-ped adversarial configs should bridge to scenario-loader override entries."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=41,
        pedestrians=[
            MultiPedCandidateSpec(
                id="left_blocker",
                start=Pose2D(1.0, 2.0, 0.1),
                goal=Pose2D(5.0, 2.0),
                spawn_time_s=0.5,
                speed_mps=1.1,
                delay_s=0.25,
                metadata={"role": "left"},
            ),
            MultiPedCandidateSpec(
                id="right_blocker",
                start=Pose2D(1.0, 3.0),
                goal=Pose2D(5.0, 3.0),
                spawn_time_s=0.75,
                speed_mps=1.2,
            ),
        ],
    )

    overrides = materialize_multi_ped_single_pedestrian_overrides(config)

    assert overrides == [
        {
            "id": "left_blocker",
            "start": [1.0, 2.0],
            "goal": [5.0, 2.0],
            "speed_m_s": 1.1,
            "start_delay_s": 0.75,
            "note": "adversarial-multi-ped.v1 group_squeeze seed=41 ped=left_blocker",
            "metadata": {
                "adversarial_family": "group_squeeze",
                "adversarial_schema_version": "adversarial-multi-ped.v1",
                "adversarial_scenario_seed": 41,
                "pedestrian_metadata": {"role": "left"},
                "spawn_time_s": 0.5,
                "delay_s": 0.25,
            },
        },
        {
            "id": "right_blocker",
            "start": [1.0, 3.0],
            "goal": [5.0, 3.0],
            "speed_m_s": 1.2,
            "start_delay_s": 0.75,
            "note": "adversarial-multi-ped.v1 group_squeeze seed=41 ped=right_blocker",
            "metadata": {
                "adversarial_family": "group_squeeze",
                "adversarial_schema_version": "adversarial-multi-ped.v1",
                "adversarial_scenario_seed": 41,
                "pedestrian_metadata": {},
                "spawn_time_s": 0.75,
                "delay_s": 0.0,
            },
        },
    ]


def test_multi_ped_materialized_overrides_are_yaml_safe() -> None:
    """Materialized overrides should serialize without custom YAML representers."""
    config = MultiPedAdversarialConfig(
        family="late_stop",
        scenario_seed=7,
        pedestrians=[
            MultiPedCandidateSpec(
                id="stopper",
                start=Pose2D(0.0, 0.0),
                goal=Pose2D(1.0, 0.0),
                speed_mps=0.8,
            )
        ],
    )

    dumped = yaml.safe_dump(
        {"single_pedestrians": materialize_multi_ped_single_pedestrian_overrides(config)},
        sort_keys=False,
    )

    loaded = yaml.safe_load(dumped)
    assert loaded["single_pedestrians"][0]["id"] == "stopper"
    assert loaded["single_pedestrians"][0]["start_delay_s"] == 0.0


def test_multi_ped_config_materializes_scenario_payload_with_template_merge() -> None:
    """Multi-ped configs should produce scenario-loader-ready manifest payloads."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=41,
        pedestrians=[
            MultiPedCandidateSpec(
                id="left_blocker",
                start=Pose2D(1.0, 2.0),
                goal=Pose2D(5.0, 2.0),
                spawn_time_s=0.5,
                speed_mps=1.1,
            ),
            MultiPedCandidateSpec(
                id="right_blocker",
                start=Pose2D(1.0, 3.0),
                goal=Pose2D(5.0, 3.0),
                spawn_time_s=0.75,
                speed_mps=1.2,
            ),
        ],
    )
    template = {
        "scenarios": [
            {
                "name": "template",
                "map_id": "classic_cross_trap",
                "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
                "metadata": {"archetype": "test"},
                "single_pedestrians": [
                    {"id": "left_blocker", "role": "wait", "note": "template note"}
                ],
            }
        ]
    }

    payload = materialize_multi_ped_scenario_payload(config, template)

    scenario = payload["scenarios"][0]
    assert scenario["name"] == "template_multi_ped_adversarial_0041"
    assert scenario["seeds"] == [41]
    assert scenario["simulation_config"]["route_spawn_seed"] == 41
    assert scenario["metadata"]["archetype"] == "test"
    assert scenario["metadata"]["adversarial_multi_ped"]["family"] == "group_squeeze"
    assert scenario["single_pedestrians"][0]["id"] == "left_blocker"
    assert scenario["single_pedestrians"][0]["role"] == "wait"
    assert scenario["single_pedestrians"][0]["start"] == [1.0, 2.0]
    assert scenario["single_pedestrians"][0]["note"].startswith("adversarial-multi-ped.v1")
    assert scenario["single_pedestrians"][1]["id"] == "right_blocker"


def test_multi_ped_materialized_scenario_payload_is_yaml_safe() -> None:
    """Scenario payload helper should emit plain YAML-safe primitives."""
    config = MultiPedAdversarialConfig(
        family="late_stop",
        scenario_seed=7,
        pedestrians=[
            MultiPedCandidateSpec(
                id="stopper",
                start=Pose2D(0.0, 0.0),
                goal=Pose2D(1.0, 0.0),
                speed_mps=0.8,
            )
        ],
    )

    dumped = yaml.safe_dump(
        materialize_multi_ped_scenario_payload(
            config,
            {"scenarios": [{"name": "template", "map_id": "classic_cross_trap"}]},
        ),
        sort_keys=False,
    )

    loaded = yaml.safe_load(dumped)
    assert loaded["scenarios"][0]["single_pedestrians"][0]["id"] == "stopper"
    assert loaded["scenarios"][0]["metadata"]["adversarial_multi_ped"]["schema_version"] == (
        "adversarial-multi-ped.v1"
    )


def test_manifest_materializes_single_pedestrian_override() -> None:
    """A valid single-ped manifest should bridge to scenario-loader overrides."""
    manifest = build_manifest(_candidate(17), generator=None)

    override = materialize_manifest_single_pedestrian_override(
        manifest,
        pedestrian_id="crossing_probe",
    )

    assert override["id"] == "crossing_probe"
    assert override["start"] == [1.0, 2.0]
    assert override["goal"] == [5.0, 2.0]
    assert override["speed_m_s"] == pytest.approx(1.0)
    assert override["start_delay_s"] == pytest.approx(0.0)
    assert override["metadata"]["adversarial_schema_version"] == (
        "adversarial_scenario_manifest.v1"
    )
    assert override["metadata"]["adversarial_scenario_seed"] == 17
    assert override["metadata"]["validation_status"] == "valid"
    assert override["metadata"]["normalized_control_hash"]


def test_manifest_materializes_scenario_payload_with_template_merge() -> None:
    """A valid manifest should produce a runnable scenario payload from a template."""
    manifest = build_manifest(_candidate(23))
    template = {
        "scenarios": [
            {
                "name": "crossing_ttc_template",
                "map_id": "classic_cross_trap",
                "simulation_config": {"max_episode_steps": 30, "ped_density": 0.2},
                "metadata": {"archetype": "test"},
            }
        ]
    }

    payload = materialize_manifest_scenario_payload(manifest, template)

    scenario = payload["scenarios"][0]
    assert scenario["name"] == "crossing_ttc_template_manifest_0000"
    assert scenario["seeds"] == [23]
    assert scenario["simulation_config"]["ped_density"] == 0.2
    assert scenario["simulation_config"]["route_spawn_seed"] == 23
    assert scenario["simulation_config"]["peds_speed_mult"] == pytest.approx(1.0)
    assert scenario["metadata"]["archetype"] == "test"
    assert scenario["metadata"]["adversarial_scenario_manifest"]["schema_version"] == (
        "adversarial_scenario_manifest.v1"
    )
    assert scenario["metadata"]["adversarial_manifest_runtime"]["benchmark_frozen"] is False
    assert scenario["single_pedestrians"][0]["id"] == "manifest_candidate_0000"


def test_manifest_materializes_route_overrides_for_route_smoke() -> None:
    """A manifest route candidate should produce route overrides for benchmark smoke runs."""
    manifest = build_manifest(_candidate(23))

    route_payload = materialize_manifest_route_overrides(manifest)
    scenario_payload = materialize_manifest_scenario_payload(
        manifest,
        {
            "scenarios": [
                {
                    "name": "crossing_ttc_template",
                    "map_id": "classic_cross_trap",
                    "simulation_config": {"max_episode_steps": 30, "ped_density": 0.2},
                }
            ]
        },
        route_file_name="routes/candidate_0000_route_overrides.yaml",
    )

    assert route_payload == {
        "robot_routes": [
            {
                "spawn_id": 100000,
                "goal_id": 100000,
                "waypoints": [[1.0, 2.0], [5.0, 2.0]],
            }
        ],
        "ped_routes": [],
    }
    scenario = scenario_payload["scenarios"][0]
    assert scenario["route_overrides_file"] == "routes/candidate_0000_route_overrides.yaml"
    assert "single_pedestrians" not in scenario


def test_manifest_route_overrides_reject_missing_pose_coordinate() -> None:
    """Malformed pose mappings should fail closed with a clear coordinate error."""
    manifest = build_manifest(_candidate(23))
    assert manifest.candidate_controls is not None
    del manifest.candidate_controls["start"]["x"]

    with pytest.raises(ValueError, match="requires both 'x' and 'y' keys"):
        materialize_manifest_route_overrides(manifest)


def test_manifest_materialization_rejects_degenerate_manifest() -> None:
    """Invalid or degenerate manifests should fail closed before planner execution."""
    manifest = build_manifest(
        CandidateSpec(
            start=Pose2D(1.0, 1.0),
            goal=Pose2D(1.0, 1.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=0.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
    )

    with pytest.raises(ValueError, match="only valid manifests"):
        materialize_manifest_scenario_payload(
            manifest,
            {"scenarios": [{"name": "template", "map_id": "classic_cross_trap"}]},
        )


def test_multi_ped_config_converts_to_runtime_single_pedestrians_with_metadata() -> None:
    """Runtime definitions should preserve adversarial attribution metadata per pedestrian."""
    config = _runtime_multi_ped_config()

    ped_defs = multi_ped_config_to_single_pedestrian_definitions(config)
    _ped_states, population_metadata = populate_single_pedestrians(ped_defs)

    assert [ped.id for ped in ped_defs] == ["left_blocker", "right_blocker"]
    assert ped_defs[0].start == (1.5, 2.0)
    assert ped_defs[0].goal == (6.5, 2.0)
    assert ped_defs[0].speed_m_s == pytest.approx(1.1)
    assert ped_defs[0].start_delay_s == pytest.approx(0.3)
    assert ped_defs[0].metadata["adversarial_family"] == "group_squeeze"
    assert ped_defs[0].metadata["pedestrian_metadata"] == {"lane": "left"}
    assert population_metadata[0]["metadata"]["adversarial_scenario_seed"] == 41


def test_multi_ped_adversarial_runtime_config_resets_and_steps() -> None:
    """A validated N>1 multi-ped config should run through the robot env reset/step path."""
    config = _runtime_multi_ped_config()
    base_map = _runtime_base_map()
    robot_config = build_multi_ped_adversarial_robot_config(
        config,
        base_map,
        map_id="issue_870_runtime",
        sim_time_in_secs=1.0,
    )

    assert base_map.single_pedestrians == []
    runtime_map = robot_config.map_pool.get_map("issue_870_runtime")
    assert len(runtime_map.single_pedestrians) == 2
    assert runtime_map.single_pedestrians[1].metadata["pedestrian_metadata"] == {"lane": "right"}

    env = make_robot_env(config=robot_config, debug=True, seed=config.scenario_seed)
    try:
        _obs, _info = env.reset(seed=config.scenario_seed)
        first_reset_positions = env.simulator.ped_pos.copy()

        assert env.simulator.ped_pos.shape[0] >= 2
        for _ in range(3):
            _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        env.reset(seed=config.scenario_seed)
        assert env.simulator.ped_pos.shape == first_reset_positions.shape
        assert (env.simulator.ped_pos == first_reset_positions).all()
    finally:
        env.close()


def test_multi_ped_adversarial_runtime_config_resets_and_steps_single_pedestrian() -> None:
    """The 1-N adversarial runtime contract should include N=1 reset/step coverage."""
    config = MultiPedAdversarialConfig(
        family="late_stop",
        scenario_seed=43,
        pedestrians=[
            MultiPedCandidateSpec(
                id="stopper",
                start=Pose2D(1.5, 2.0),
                goal=Pose2D(6.5, 2.0),
                spawn_time_s=0.1,
                speed_mps=0.8,
                metadata={"role": "single_blocker"},
            )
        ],
    )
    robot_config = build_multi_ped_adversarial_robot_config(
        config,
        _runtime_base_map(),
        map_id="late_stop_runtime",
        sim_time_in_secs=0.5,
    )
    runtime_map = robot_config.map_pool.get_map("late_stop_runtime")
    assert [ped.id for ped in runtime_map.single_pedestrians] == ["stopper"]
    assert runtime_map.single_pedestrians[0].metadata["pedestrian_metadata"] == {
        "role": "single_blocker"
    }

    env = make_robot_env(config=robot_config, debug=True, seed=config.scenario_seed)
    try:
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _obs, _info = env.reset(seed=config.scenario_seed)
        first_positions = env.simulator.ped_pos.copy()

        _obs, _reward, terminated, truncated, _info = env.step(action)
        assert not (terminated and truncated)

        env.reset(seed=config.scenario_seed)
        np.testing.assert_allclose(env.simulator.ped_pos, first_positions)
    finally:
        env.close()


@pytest.mark.parametrize(("fixture_path", "family", "expected_ids"), _MULTI_PED_FAMILY_FIXTURES)
def test_multi_ped_adversarial_family_fixtures_reset_step_deterministically(
    fixture_path: Path,
    family: str,
    expected_ids: tuple[str, ...],
) -> None:
    """Certified development fixtures should reset and step deterministically."""
    config = MultiPedAdversarialConfig.from_file(fixture_path)
    assert config.family == family

    payload = materialize_multi_ped_scenario_payload(
        config,
        {"scenarios": [{"name": f"{family}_smoke", "map_id": "synthetic_runtime"}]},
    )
    scenario = payload["scenarios"][0]
    status = scenario["metadata"]["adversarial_multi_ped_runtime"]
    assert status["benchmark_frozen"] is False
    assert status["certification_status"] == "uncertified_development_smoke"
    assert tuple(status["pedestrian_ids"]) == expected_ids

    robot_config = build_multi_ped_adversarial_robot_config(
        config,
        _runtime_base_map(),
        map_id=f"{family}_runtime",
        sim_time_in_secs=0.5,
    )
    runtime_map = robot_config.map_pool.get_map(f"{family}_runtime")
    assert tuple(ped.id for ped in runtime_map.single_pedestrians) == expected_ids
    assert all(
        ped.metadata["adversarial_family"] == family for ped in runtime_map.single_pedestrians
    )

    env = make_robot_env(config=robot_config, debug=True, seed=config.scenario_seed)
    try:
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        def rollout_positions() -> list[np.ndarray]:
            """Roll out the env and capture pedestrian positions per step."""
            env.reset(seed=config.scenario_seed)
            positions = [env.simulator.ped_pos.copy()]
            for _ in range(2):
                _obs, _reward, terminated, truncated, _info = env.step(action)
                positions.append(env.simulator.ped_pos.copy())
                if terminated or truncated:
                    break
            return positions

        first = rollout_positions()
        second = rollout_positions()
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_allclose(left, right)
    finally:
        env.close()


def test_multi_ped_adversarial_runtime_rejects_impossible_initial_collision() -> None:
    """Runtime plausibility checks should fail closed on overlapping starts."""
    config = _runtime_multi_ped_config(second_start_y=2.1)

    with pytest.raises(ValueError, match="start separation"):
        build_multi_ped_adversarial_robot_config(config, _runtime_base_map())


def test_multi_ped_adversarial_runtime_rejects_obstacle_intersection() -> None:
    """Runtime plausibility checks should fail closed when a pedestrian starts in an obstacle."""
    obstacle = Obstacle([(1.0, 1.5), (2.0, 1.5), (2.0, 2.5), (1.0, 2.5)])
    config = _runtime_multi_ped_config(first_start=Pose2D(1.5, 2.0))

    with pytest.raises(ValueError, match="inside obstacle"):
        build_multi_ped_adversarial_robot_config(config, _runtime_base_map(obstacles=[obstacle]))


def test_programmatic_search_scores_candidates_without_subprocess(tmp_path: Path) -> None:
    """Programmatic search should score candidates through injected evaluator hooks."""
    config = _config(tmp_path)
    scores = [0.8, 0.2]

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Write one successful episode record and return candidate evaluation."""
        snqi = scores.pop(0)
        record: dict[str, Any] = {
            "episode_id": f"episode-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": snqi, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=_SequenceSampler([_candidate(7), _candidate(7)]),
    )

    assert result.num_candidates == 2
    assert result.num_invalid_candidates == 0
    assert result.best_objective_value == -0.2
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["best_bundle_path"].endswith("candidate_0001")
    assert (config.output_dir / "candidate_0001" / "scenario.yaml").exists()
    assert (config.output_dir / "candidate_0001" / "route_overrides.yaml").exists()


def test_coordinate_refinement_sampler_improves_synthetic_objective(tmp_path: Path) -> None:
    """Feedback-capable optimizer samplers should run through the existing search API."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    search_space.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 0.0, "max": 2.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=3,
        seed=123,
    )
    sampler = CoordinateRefinementSampler(config.search_space, seed=123)
    sampled: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Record sampled candidates and score them by start position."""
        sampled.append(candidate)
        record: dict[str, Any] = {
            "episode_id": f"candidate-{len(sampled)}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 2.0 - candidate.start.x, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=sampler,
    )

    assert [candidate.start.x for candidate in sampled[:2]] == [1.0, 2.0]
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    assert result.best_objective_value == pytest.approx(-0.0)


def test_optuna_candidate_sampler_is_deterministic_and_feedback_capable(
    tmp_path: Path,
) -> None:
    """Optuna-backed sampling should be deterministic and accept scored feedback."""

    pytest.importorskip("optuna")
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        seed=19,
    )

    left = OptunaCandidateSampler(config.search_space, seed=19)
    right = OptunaCandidateSampler(config.search_space, seed=19)
    first = left.sample()
    assert first == right.sample()

    left.observe(
        CandidateEvaluation(
            candidate=first,
            certification_status=passed_status(),
            objective_value=0.25,
            failure_attribution=None,
            episode_record_path=None,
            trajectory_csv_path=None,
            scenario_yaml_path=None,
        )
    )
    second = left.sample()

    assert config.search_space.validate_candidate(second) == []
    assert second.scenario_seed == int(second.scenario_seed)


def test_optuna_candidate_sampler_reports_missing_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimizer sampler should fail with an actionable dependency error."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setitem(sys.modules, "optuna", None)

    with pytest.raises(RuntimeError, match="OptunaCandidateSampler requires optuna"):
        OptunaCandidateSampler(config.search_space, seed=7)


def test_seed_sensitivity_classifies_stable_and_brittle_failures(tmp_path: Path) -> None:
    """Seed sensitivity should separate persistent failures from one-seed artifacts."""
    config = _config(tmp_path)

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        _scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Emit deterministic collision records for selected replay seeds."""
        collision = candidate.scenario_seed in {7, 8}
        record = {
            "episode_id": f"seed-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "collision" if collision else "success",
            "steps": 1,
            "termination_reason": "collision" if collision else "success",
            "outcome": {
                "route_complete": not collision,
                "collision": collision,
                "timeout": False,
            },
            "metrics": {"success": 0.0 if collision else 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("seed sensitivity test"),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=_scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    stable = run_seed_sensitivity(
        config,
        candidate=_candidate(7),
        seeds=[7, 8, 9],
        output_dir=tmp_path / "stable",
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("seed sensitivity test"),
        min_persistence_rate=0.5,
    )
    brittle = run_seed_sensitivity(
        config,
        candidate=_candidate(7),
        seeds=[7, 9, 10],
        output_dir=tmp_path / "brittle",
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("seed sensitivity test"),
        min_persistence_rate=0.5,
    )

    assert stable.classification == "stable_failure"
    assert stable.failure_persistence_rate == pytest.approx(2 / 3)
    assert stable.objective_score_spread == pytest.approx(11.0)
    assert [replay.outcome for replay in stable.replays] == ["collision", "collision", "success"]
    assert all(replay.started_at for replay in stable.replays)
    assert (tmp_path / "stable" / "seed_sensitivity_summary.json").exists()

    assert brittle.classification == "brittle_failure"
    assert brittle.failure_persistence_rate == pytest.approx(1 / 3)


def test_seed_sensitivity_records_fail_closed_rejected_perturbations(tmp_path: Path) -> None:
    """Rejected perturbations should be recorded without weakening the failure denominator."""
    config = _config(tmp_path, require_certification=True)
    evaluated: list[int] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        _scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Emit a successful replay for certified perturbations."""
        evaluated.append(candidate.scenario_seed)
        record = {
            "episode_id": f"seed-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 1,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("seed sensitivity test"),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=_scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    summary = run_seed_sensitivity(
        config,
        candidate=_candidate(7),
        seeds=[7, 8],
        output_dir=tmp_path / "rejected",
        evaluator=evaluator,
        certifier=lambda candidate, _path, _required: (
            not_available_status("seed 8 rejected by certification")
            if candidate.scenario_seed == 8
            else passed_status("seed sensitivity test")
        ),
    )

    assert evaluated == [7]
    assert summary.num_fail_closed_exclusions == 1
    assert summary.failure_persistence_rate == 0.0
    assert summary.replays[1].status == "not_available"
    assert summary.replays[1].outcome == "fail_closed_exclusion"
    assert summary.replays[1].reason == "seed 8 rejected by certification"
    assert summary.replays[1].started_at


def test_seed_sensitivity_records_timing_speed_perturbation_grid(tmp_path: Path) -> None:
    """Seed sensitivity should replay a deterministic grid of timing/speed perturbations."""
    config = _config(tmp_path)
    evaluated: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Record perturbed candidates and emit a compact replay record."""
        evaluated.append(candidate)
        scenario_payload = yaml.safe_load(scenario_yaml_path.read_text(encoding="utf-8"))
        scenario = scenario_payload["scenarios"][0]
        assert scenario["simulation_config"]["peds_speed_mult"] == pytest.approx(
            candidate.pedestrian_speed_mps
        )
        assert scenario["metadata"]["adversarial_candidate"]["spawn_time_s"] == pytest.approx(
            candidate.spawn_time_s
        )
        record = {
            "episode_id": f"seed-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 1,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"success": 1.0, "snqi": candidate.pedestrian_speed_mps},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("perturbation grid test"),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    summary = run_seed_sensitivity(
        config,
        candidate=_candidate(7),
        seeds=[7, 8],
        output_dir=tmp_path / "perturbations",
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("perturbation grid test"),
        perturbations=[
            SeedSensitivityPerturbation(label="base"),
            SeedSensitivityPerturbation(
                label="faster_later",
                pedestrian_speed_delta_mps=0.2,
                pedestrian_delay_delta_s=0.3,
                spawn_time_delta_s=0.1,
            ),
        ],
    )

    assert [
        (
            candidate.scenario_seed,
            candidate.spawn_time_s,
            candidate.pedestrian_speed_mps,
            candidate.pedestrian_delay_s,
        )
        for candidate in evaluated
    ] == [
        (7, 0.0, 1.0, 0.0),
        (7, 0.1, 1.2, 0.3),
        (8, 0.0, 1.0, 0.0),
        (8, 0.1, 1.2, 0.3),
    ]
    assert summary.perturbations[1].label == "faster_later"
    assert summary.replays[1].perturbation.label == "faster_later"
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert payload["perturbations"][1]["pedestrian_speed_delta_mps"] == pytest.approx(0.2)
    assert payload["replays"][1]["perturbation"]["spawn_time_delta_s"] == pytest.approx(0.1)


def test_seed_sensitivity_rejects_unbounded_perturbations(tmp_path: Path) -> None:
    """Perturbation grids should fail early when deltas exceed the bounded opt-in surface."""
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="pedestrian_speed_delta_mps"):
        run_seed_sensitivity(
            config,
            candidate=_candidate(7),
            seeds=[7],
            output_dir=tmp_path / "invalid",
            evaluator=lambda _config, _candidate, _scenario, _bundle: pytest.fail(
                "invalid perturbation should not evaluate"
            ),
            perturbations=[SeedSensitivityPerturbation(pedestrian_speed_delta_mps=2.0)],
        )


def test_seed_sensitivity_rejects_empty_perturbation_iterables(tmp_path: Path) -> None:
    """Empty perturbation iterables should fail consistently for all iterable types."""
    config = _config(tmp_path)

    def evaluator(
        _config: SearchConfig,
        _candidate: CandidateSpec,
        _scenario_yaml_path: Path,
        _candidate_dir: Path,
    ) -> CandidateEvaluation:
        raise AssertionError("empty perturbations should not evaluate")

    for perturbations in ([], iter(())):
        with pytest.raises(ValueError, match="perturbations must contain at least one entry"):
            run_seed_sensitivity(
                config,
                candidate=_candidate(7),
                seeds=[7],
                output_dir=tmp_path / "empty_perturbations",
                evaluator=evaluator,
                perturbations=perturbations,
            )


def test_seed_sensitivity_records_fail_closed_evaluator_rejections(tmp_path: Path) -> None:
    """Evaluator failures should be recorded per seed instead of aborting the summary."""
    config = _config(tmp_path)
    evaluated: list[int] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        _scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Reject one replay and emit a successful record for the other."""
        evaluated.append(candidate.scenario_seed)
        if candidate.scenario_seed == 8:
            raise RuntimeError("seed 8 benchmark rejected")
        record = {
            "episode_id": f"seed-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 1,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("seed sensitivity test"),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=_scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    summary = run_seed_sensitivity(
        config,
        candidate=_candidate(7),
        seeds=[7, 8],
        output_dir=tmp_path / "rejected_evaluator",
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("seed sensitivity test"),
    )

    assert evaluated == [7, 8]
    assert summary.num_fail_closed_exclusions == 1
    assert summary.failure_persistence_rate == 0.0
    assert summary.classification == "no_failure"
    assert summary.replays[1].status == "evaluation_failed"
    assert summary.replays[1].outcome == "fail_closed_exclusion"
    assert "seed 8 benchmark rejected" in str(summary.replays[1].reason)
    assert summary.replays[1].started_at


def test_seed_sensitivity_rejects_non_integral_replay_seeds(tmp_path: Path) -> None:
    """Replay seed coercion should reject lossy non-integer values."""
    config = _config(tmp_path)

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        _scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Return a no-failure replay for valid seeds."""
        record = {
            "episode_id": f"seed-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 1,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("seed sensitivity test"),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=_scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    summary = run_seed_sensitivity(
        config,
        candidate=_candidate(7),
        seeds=["7", "+8", 9.0],
        output_dir=tmp_path / "valid_seed_coercion",
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("seed sensitivity test"),
    )
    assert summary.seeds == (7, 8, 9)

    with pytest.raises(ValueError, match="seed values must be integers"):
        run_seed_sensitivity(
            config,
            candidate=_candidate(7),
            seeds=[7, 8.5],
            output_dir=tmp_path / "invalid_seed_coercion",
            evaluator=evaluator,
        )


def test_sampler_comparison_synthetic_smoke(tmp_path: Path) -> None:
    """Comparison helper should run random, coordinate, and optuna samplers."""

    pytest.importorskip("optuna")
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "comparison",
        budget=2,
        seed=23,
    )

    rows = run_sampler_comparison(
        config=config,
        sampler_names=("random", "coordinate", "optuna"),
        synthetic=True,
    )

    assert [row.sampler for row in rows] == ["random", "coordinate", "optuna"]
    assert {row.num_candidates for row in rows} == {2}
    assert all(Path(row.manifest_path).exists() for row in rows)
    assert all(row.num_failed_evaluations == 0 for row in rows)
    assert all(row.best_valid_objective is not None for row in rows)


def test_sampler_comparison_package_b_budget_seed_grid(tmp_path: Path) -> None:
    """Package-B helper should run fixed candidate budgets under repeated seeds."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "comparison",
        budget=1,
        seed=23,
    )

    rows = run_sampler_comparison(
        config=config,
        sampler_names=("random", "coordinate"),
        objective_names=("worst_case_snqi",),
        synthetic=True,
        budgets=(16, 32, 64),
        seeds=(101, 202),
    )

    assert {(row.budget, row.seed) for row in rows} == {
        (16, 101),
        (16, 202),
        (32, 101),
        (32, 202),
        (64, 101),
        (64, 202),
    }
    assert {row.num_candidates for row in rows} == {16, 32, 64}
    assert all(row.held_out_family_status == "not_evaluated_narrow_archive" for row in rows)
    assert all("learned failure proposal #2921" in " ".join(row.caveats) for row in rows)


def test_sampler_comparison_reports_certified_replayable_failures(tmp_path: Path) -> None:
    """Manifest-derived rows should separate certified failures from exclusions."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    scenario_path = bundle_dir / "scenario.yaml"
    episode_path = bundle_dir / "episode_records.jsonl"
    trajectory_path = bundle_dir / "trajectory.csv"
    for path in (scenario_path, episode_path, trajectory_path):
        path.write_text("fixture\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "adversarial-search-manifest.v1",
                "candidates": [
                    {
                        "certification_status": {"status": "passed"},
                        "objective_value": 4.0,
                        "failure_attribution": {
                            "primary_failure": "collision",
                            "details": {},
                        },
                        "scenario_yaml_path": scenario_path.as_posix(),
                        "episode_record_path": episode_path.as_posix(),
                        "trajectory_csv_path": trajectory_path.as_posix(),
                        "bundle_path": bundle_dir.as_posix(),
                        "error": None,
                    },
                    {
                        "certification_status": {"status": "failed"},
                        "objective_value": None,
                        "failure_attribution": {
                            "primary_failure": "invalid_candidate",
                            "details": {},
                        },
                        "error": "invalid",
                    },
                    {
                        "certification_status": {"status": "passed"},
                        "objective_value": 2.0,
                        "failure_attribution": {
                            "primary_failure": "success",
                            "details": {"readiness_status": "fallback"},
                        },
                        "error": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    row = _comparison_row_from_manifest(
        objective="worst_case_snqi",
        sampler="random",
        budget=3,
        seed=11,
        manifest_path=manifest_path,
        best_bundle_path=bundle_dir,
        best_objective_value=4.0,
        num_candidates=3,
        num_valid_candidates=2,
        num_invalid_candidates=1,
        num_failed_evaluations=0,
    )

    assert row.first_failure_iteration == 1
    assert row.best_valid_objective == 4.0
    assert row.invalid_candidate_rate == pytest.approx(1 / 3)
    assert row.certified_valid_failure_count == 1
    assert row.replayable_valid_failure_count == 1
    assert row.replay_success_rate == 1.0
    assert row.fallback_candidate_count == 1


def test_sampler_comparison_multi_objective(tmp_path: Path) -> None:
    """Comparison helper should run multiple objectives and tag rows correctly."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "comparison",
        budget=2,
        seed=23,
    )

    rows = run_sampler_comparison(
        config=config,
        sampler_names=("random",),
        objective_names=("worst_case_snqi", "temporal_robustness"),
        synthetic=True,
    )

    assert len(rows) == 2
    assert {row.objective for row in rows} == {"worst_case_snqi", "temporal_robustness"}
    assert all(row.sampler == "random" for row in rows)
    assert all(row.num_candidates == 2 for row in rows)
    assert all(Path(row.manifest_path).exists() for row in rows)
    assert all(row.num_failed_evaluations == 0 for row in rows)
    assert all(row.best_valid_objective is not None for row in rows)


def test_invalid_optimizer_proposals_are_rejected_before_evaluation(tmp_path: Path) -> None:
    """Search-space validation must fail closed before benchmark evaluation."""
    config = _config(tmp_path)

    class InvalidThenValidSampler:
        """Sampler that emits one invalid candidate before a valid one."""

        def __init__(self) -> None:
            self._candidates = [
                CandidateSpec(
                    start=Pose2D(-10.0, 2.0),
                    goal=Pose2D(5.0, 2.0),
                    spawn_time_s=0.0,
                    pedestrian_speed_mps=1.0,
                    pedestrian_delay_s=0.0,
                    scenario_seed=7,
                ),
                _candidate(7),
            ]
            self.observed: list[CandidateEvaluation] = []

        def sample(self) -> CandidateSpec:
            """Return the next invalid-or-valid candidate."""
            return self._candidates.pop(0)

        def observe(self, evaluation: CandidateEvaluation) -> None:
            """Record evaluations observed by the optimizer."""
            self.observed.append(evaluation)

    sampler = InvalidThenValidSampler()
    evaluated: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Evaluate only valid candidates for invalid-proposal tests."""
        evaluated.append(candidate)
        record: dict[str, Any] = {
            "episode_id": "valid",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.5, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=sampler,
    )

    assert evaluated == [_candidate(7)]
    assert result.num_invalid_candidates == 1
    assert len(sampler.observed) == 2
    assert sampler.observed[0].objective_value is None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["error"] == "start.x outside search space"


def test_default_search_keeps_candidate_evaluation_sequential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Candidate search order is deterministic; workers are scoped to each benchmark call."""
    config = _config(tmp_path, workers=4)
    call_order: list[tuple[Path, int]] = []

    def fake_run_batch(
        _scenario_yaml_path: Path,
        *,
        out_path: Path,
        workers: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        """Write a successful episode record and record worker usage."""
        call_order.append((out_path.parent, workers))
        record: dict[str, Any] = {
            "episode_id": out_path.parent.name,
            "seed": len(call_order),
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.5, "success": 1.0},
        }
        out_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return {"failures": []}

    monkeypatch.setattr(search, "run_batch", fake_run_batch)

    result = search.run_adversarial_search(
        config,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=_SequenceSampler([_candidate(7), _candidate(7)]),
    )

    assert result.num_candidates == 2
    assert call_order == [
        (config.output_dir / "candidate_0000", 4),
        (config.output_dir / "candidate_0001", 4),
    ]


def test_required_certification_fails_closed_when_adapter_missing(tmp_path: Path) -> None:
    """Required certification should stop search before evaluation when unavailable."""
    config = _config(tmp_path, require_certification=True)
    config = SearchConfig.from_files(
        policy=config.policy,
        scenario_template=config.scenario_template,
        search_space=config.search_space_path,
        objective=config.objective,
        output_dir=config.output_dir,
        budget=1,
        seed=config.seed,
        require_certification=True,
    )

    def evaluator(*_args: object, **_kwargs: object) -> CandidateEvaluation:
        """Fail if evaluation runs after unavailable strict certification."""
        raise AssertionError("strict certification should reject before evaluation")

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: not_available_status(
            "scenario_cert.v1 adapter is not available"
        ),
        sampler=_SequenceSampler([_candidate(7)]),
    )

    assert result.best_candidate is None
    assert result.num_invalid_candidates == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["certification_status"]["status"] == "not_available"
    assert manifest["candidates"][0]["error"] == "scenario_cert.v1 adapter is not available"


def test_required_certification_uses_real_scenario_certification_api(tmp_path: Path) -> None:
    """Strict adversarial search should exclude invalid candidates and evaluate certified ones."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    search_space.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 2.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 4.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=2,
        seed=123,
        require_certification=True,
    )
    invalid_candidate = _candidate(7)
    valid_candidate = CandidateSpec(
        start=Pose2D(2.0, 2.0),
        goal=Pose2D(4.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    evaluated: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        """Evaluate strict-certification candidates after valid certification."""
        evaluated.append(candidate)
        record: dict[str, Any] = {
            "episode_id": "strict-cert-valid",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.5, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        sampler=_SequenceSampler([invalid_candidate, valid_candidate]),
    )

    assert evaluated == [valid_candidate]
    assert result.num_candidates == 2
    assert result.num_invalid_candidates == 1
    assert result.num_valid_candidates == 1
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    invalid_status = manifest["candidates"][0]["certification_status"]
    valid_status = manifest["candidates"][1]["certification_status"]
    assert invalid_status["status"] == "failed"
    assert "start_inside_static_obstacle" in invalid_status["reason"]
    assert manifest["candidates"][0]["error"] == "start_inside_static_obstacle"
    assert valid_status["status"] == "passed"
    assert valid_status["details"]["certificates"][0]["benchmark_eligibility"] != "excluded"
    assert manifest["candidates"][1]["trajectory_csv_path"].endswith(
        "candidate_0001/trajectory.csv"
    )


def test_default_evaluator_treats_failures_as_failed_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Benchmark failure summaries must fail closed before objective scoring."""
    config = _config(tmp_path)

    def fake_run_batch(*_args: object, **_kwargs: object) -> dict[str, object]:
        """Return benchmark failures without writing episode records."""
        return {"failures": [{"scenario_id": "candidate", "error": "boom"}]}

    monkeypatch.setattr(search, "run_batch", fake_run_batch)

    with pytest.raises(RuntimeError, match="candidate evaluation failed"):
        search._default_evaluator(
            config,
            _candidate(1),
            tmp_path / "scenario.yaml",
            tmp_path / "candidate",
        )


def test_failure_attribution_covers_primary_outcomes() -> None:
    """Failure attribution must distinguish collision, timeout, incomplete, and errors."""
    collision = attribution_from_episode_record(
        {"status": "done", "outcome": {"collision": True, "route_complete": False}}
    )
    legacy_collision = attribution_from_episode_record(
        {"status": "done", "outcome": {"collision_event": True, "route_complete": False}}
    )
    timeout = attribution_from_episode_record(
        {"status": "done", "outcome": {"timeout_event": True, "route_complete": False}}
    )
    incomplete = attribution_from_episode_record(
        {"status": "done", "outcome": {"route_complete": False}}
    )
    error = attribution_from_error("boom")

    assert collision.primary_failure == "collision"
    assert legacy_collision.primary_failure == "collision"
    assert timeout.primary_failure == "timeout"
    assert incomplete.primary_failure == "incomplete"
    assert error.to_json()["status"] == "evaluation_failed"


def test_read_first_jsonl_record_rejects_malformed_lines(tmp_path: Path) -> None:
    """JSONL helper should fail closed for malformed records with source context."""
    path = tmp_path / "episode.jsonl"
    path.write_text("{bad json}\n\n" + json.dumps({"episode_id": "ok"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"episode\.jsonl: invalid JSON on line 1"):
        read_first_jsonl_record(path)


def test_write_trajectory_csv_escapes_fields(tmp_path: Path) -> None:
    """Trajectory index rows should remain valid CSV when fields contain punctuation."""
    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        {
            "episode_id": "episode,1",
            "seed": 7,
            "status": "done",
            "steps": 3,
            "termination_reason": 'quote "and" comma, here',
        },
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[1] == ["episode,1", "7", "done", "3", 'quote "and" comma, here']


def test_write_trajectory_csv_exports_dense_trajectory_data(tmp_path: Path) -> None:
    """Trajectory data in episode records should become per-entity CSV rows."""
    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        {
            "episode_id": "episode-1",
            "seed": 7,
            "trajectory_data": [
                {
                    "step": 0,
                    "time_s": 0.0,
                    "robot": {"x": 1.0, "y": 2.0, "theta": 0.1},
                    "pedestrians": {"ped-1": [3.0, 4.0, 0.2]},
                },
                {
                    "step": 1,
                    "time_s": 0.1,
                    "robot_position": [1.5, 2.5, 0.15],
                    "pedestrian_positions": [[3.5, 4.5, 0.25]],
                },
            ],
        },
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "0",
            "entity_type": "robot",
            "entity_id": "robot",
            "time_s": "0.0",
            "x": "1.0",
            "y": "2.0",
            "theta": "0.1",
        },
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "0",
            "entity_type": "pedestrian",
            "entity_id": "ped-1",
            "time_s": "0.0",
            "x": "3.0",
            "y": "4.0",
            "theta": "0.2",
        },
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "1",
            "entity_type": "robot",
            "entity_id": "robot",
            "time_s": "0.1",
            "x": "1.5",
            "y": "2.5",
            "theta": "0.15",
        },
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "1",
            "entity_type": "pedestrian",
            "entity_id": "0",
            "time_s": "0.1",
            "x": "3.5",
            "y": "4.5",
            "theta": "0.25",
        },
    ]


def test_write_trajectory_csv_supports_legacy_coordinate_lists(tmp_path: Path) -> None:
    """Existing visualization-style coordinate lists should produce dense robot rows."""
    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        {"episode_id": "episode-2", "seed": 11, "trajectory_data": [[0.0, 1.0], [0.5, 1.5]]},
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["step"] for row in rows] == ["0", "1"]
    assert [row["entity_type"] for row in rows] == ["robot", "robot"]
    assert [(row["x"], row["y"]) for row in rows] == [("0.0", "1.0"), ("0.5", "1.5")]


def test_certification_adapter_handles_missing_and_mocked_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Certification must fail closed when required and normalize adapter payloads."""
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", None)

    advisory = certification.certify_candidate(
        _candidate(1), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=False
    )
    assert advisory.passed
    assert failed_status("bad", details={"why": "test"}).to_json()["details"] == {"why": "test"}

    fake_module = types.ModuleType("robot_sf.scenario_certification")
    responses: list[object] = [
        {"status": "valid", "reason": "ok", "details": ["raw"]},
        {"status": "unavailable", "reason": "backend absent"},
        "not a mapping",
        {"status": "invalid", "reason": "outside map", "details": {"field": "start"}},
    ]

    def fake_certify_scenario(*_args: object, **_kwargs: object) -> object:
        """Return queued legacy certification payloads."""
        return responses.pop(0)

    fake_module.certify_scenario = fake_certify_scenario
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", fake_module)

    passed = certification.certify_candidate(
        _candidate(2), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )
    unavailable = certification.certify_candidate(
        _candidate(3), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )
    non_mapping = certification.certify_candidate(
        _candidate(4), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )
    failed = certification.certify_candidate(
        _candidate(5), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )

    assert passed.passed
    assert passed.details == {"raw_details": ["raw"]}
    assert unavailable.status == "not_available"
    assert non_mapping.status == "failed"
    assert failed.reason == "outside map"


def test_certification_adapter_uses_current_scenario_certification_file_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adversarial certification should call the in-repo scenario-cert file API."""
    fake_module = types.ModuleType("robot_sf.scenario_certification")

    def fake_certify_scenario_file(*_args: object, **_kwargs: object) -> list[object]:
        """Return one opaque certificate for file-API normalization."""
        return [object()]

    def fake_certificate_to_dict(_certificate: object) -> dict[str, object]:
        """Return a valid certificate payload."""
        return {
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": ["ok"],
        }

    fake_module.certify_scenario_file = fake_certify_scenario_file
    fake_module.certificate_to_dict = fake_certificate_to_dict
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", fake_module)

    status = certification.certify_candidate(
        _candidate(6), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )

    assert status.passed
    assert status.reason == "ok"
    assert status.details["certificates"][0]["classification"] == "valid"


def test_certification_adapter_preserves_worst_file_api_eligibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """File-API normalization should keep stress-only/excluded evidence visible."""
    fake_module = types.ModuleType("robot_sf.scenario_certification")

    def fake_certify_scenario_file(*_args: object, **_kwargs: object) -> list[object]:
        """Return opaque certificates for worst-eligibility normalization."""
        return [object(), object(), object()]

    payloads: list[dict[str, object]] = [
        {
            "classification": "valid",
            "benchmark_eligibility": None,
            "reasons": ["valid fallback"],
        },
        {
            "classification": "knife_edge",
            "benchmark_eligibility": "stress_only",
            "reasons": ["knife-edge clearance"],
        },
        {
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": ["eligible route"],
        },
    ]

    def fake_certificate_to_dict(_certificate: object) -> dict[str, object]:
        """Return queued certificate payloads."""
        return payloads.pop(0)

    fake_module.certify_scenario_file = fake_certify_scenario_file
    fake_module.certificate_to_dict = fake_certificate_to_dict
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", fake_module)

    status = certification.certify_candidate(
        _candidate(7), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )

    assert status.passed
    assert status.reason == "knife-edge clearance"


def test_objective_registry_and_fallback_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Objectives should score SNQI records, fallback failures, and registry errors."""
    monkeypatch.setattr(objectives, "_OBJECTIVES", dict(objectives._OBJECTIVES))
    episode_path = tmp_path / "episode.jsonl"
    episode_path.write_text(
        json.dumps({"metrics": {"snqi": "nan"}, "outcome": {"route_complete": True}}) + "\n",
        encoding="utf-8",
    )
    empty_eval = CandidateEvaluation(
        candidate=_candidate(1),
        certification_status=passed_status(),
        objective_value=None,
        failure_attribution=None,
        episode_record_path=None,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )
    scored_eval = CandidateEvaluation(
        candidate=_candidate(2),
        certification_status=passed_status(),
        objective_value=None,
        failure_attribution=None,
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )

    assert objectives.worst_case_snqi(empty_eval) is None
    assert objectives.worst_case_snqi(scored_eval) == -0.0

    episode_path.write_text(
        json.dumps(
            {
                "metrics": {"success": "bad", "near_misses": 2},
                "outcome": {"collision": True, "timeout": True, "route_complete": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert objectives.worst_case_snqi(scored_eval) == 15.0

    objectives.register_objective("unit_test_constant", lambda _evaluation: 42.0)
    try:
        assert objectives.get_objective("unit_test_constant")(scored_eval) == 42.0
        assert "unit_test_constant" in objectives.list_objectives()
    finally:
        objectives.unregister_objective("unit_test_constant")
    with pytest.raises(ValueError, match="objective name"):
        objectives.register_objective("", lambda _evaluation: 0.0)
    with pytest.raises(ValueError, match="Unknown adversarial objective"):
        objectives.get_objective("missing")


def test_random_sampler_is_deterministic(tmp_path: Path) -> None:
    """Random search must be repeatable for the same seed and search space."""
    template_path = tmp_path / "template.yaml"
    space_path = tmp_path / "space.yaml"
    _write_template(template_path)
    _write_space(space_path)
    payload = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    payload["variables"]["scenario_seed"] = {"min": 7, "max": 9}
    space_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    space = SearchConfig.from_files(
        policy="goal",
        scenario_template=template_path,
        search_space=space_path,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
    ).search_space
    left = RandomCandidateSampler(space, seed=7).sample()
    right = RandomCandidateSampler(space, seed=7).sample()

    assert left == right
    assert isinstance(left.scenario_seed, int)
    assert 7 <= left.scenario_seed <= 9
