"""Regression tests for issue #5829's design-independent campaign fixes."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark import campaign_logging
from robot_sf.benchmark.heterogeneous_population_ablation_runner import (
    ManifestSpawnRealizabilityError,
    assert_manifest_spawn_realizable,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts/benchmark/build_heterogeneity_ablation_manifest_issue_3574.py"
RUN_SCRIPT = REPO_ROOT / "scripts/benchmark/run_heterogeneous_population_ablation_issue_3574.py"
FRANCIS_SINGLE_INTERACTION_MAP = (
    REPO_ROOT / "maps/svg_maps/francis2023/francis2023_frontal_approach.svg"
)


def _load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module: ModuleType, argv: list[str]) -> int:
    previous = sys.argv
    sys.argv = argv
    try:
        return int(module.main())
    finally:
        sys.argv = previous


def _francis_manifest_config() -> dict[str, Any]:
    return {
        "schema_version": "mean_matched_heterogeneity_harness.config.v1",
        "issue": 5829,
        "planners": [{"key": "goal", "algo": "goal"}],
        "seeds": [101, 102],
        "response_law_fractions": [0.0],
        "scenarios": [
            {
                "id": "francis2023_frontal_approach",
                "map_file": str(FRANCIS_SINGLE_INTERACTION_MAP),
                "density": 0.02,
                "population_size": 12,
                "archetype_seed": 3574,
                "composition": {"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
                "archetypes": {
                    "cautious": {"desired_speed_factor": 0.7, "radius_m": 0.35},
                    "standard": {"desired_speed_factor": 1.0, "radius_m": 0.3},
                    "hurried": {"desired_speed_factor": 1.4, "radius_m": 0.25},
                },
            }
        ],
    }


def test_manifest_build_rejects_francis_forced_population_cell_once(tmp_path: Path) -> None:
    """Francis single-interaction geometry fails before output despite repeated rows."""

    config_path = tmp_path / "francis.yaml"
    config_path.write_text(yaml.safe_dump(_francis_manifest_config()), encoding="utf-8")
    output_path = tmp_path / "manifest.json"
    module = _load_script(BUILD_SCRIPT, "issue_5829_manifest_builder")

    with pytest.raises(ManifestSpawnRealizabilityError) as exc_info:
        _run_main(
            module,
            [
                BUILD_SCRIPT.name,
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ],
        )

    message = str(exc_info.value)
    assert "1 scenario/population cell" in message
    assert "scenario=francis2023_frontal_approach" in message
    assert "population_size=12" in message
    assert "remaining 11 background pedestrians" in message
    assert not output_path.exists()


def test_manifest_preflight_rejects_present_but_unsampleable_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route blocked by obstacle geometry fails during build-time spawn construction."""

    spawn_zone = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (8.5, 8.5)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    blocked_map = MapDefinition(
        width=10.0,
        height=10.0,
        obstacles=[Obstacle([(-10.0, -10.0), (20.0, -10.0), (20.0, 20.0), (-10.0, 20.0)])],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=[
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 10.0, 10.0, 10.0),
            (0.0, 0.0, 0.0, 10.0),
            (10.0, 10.0, 0.0, 10.0),
        ],
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
    )
    map_path = tmp_path / "blocked-route.svg"
    map_path.write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(
        "robot_sf.benchmark.heterogeneous_population_ablation_runner.resolve_map_definition",
        lambda *_args, **_kwargs: blocked_map,
    )
    row = {
        "scenario_id": "blocked_route",
        "map_file": str(map_path),
        "seed": 101,
        "density": 0.02,
        "arm_population": {
            "counts": {"standard": 2},
            "composition": {"standard": 1.0},
            "pedestrian_control_trace_labels": [],
        },
    }

    with pytest.raises(ManifestSpawnRealizabilityError, match="Failed to sample"):
        assert_manifest_spawn_realizable([row], scenario_path=tmp_path / "scenario.yaml")


def test_runner_preserves_completed_jsonl_records_on_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-campaign abort leaves each completed episode as parseable JSONL."""

    module = _load_script(RUN_SCRIPT, "issue_5829_campaign_runner")
    rows = [
        {
            "scenario_id": f"scenario_{index}",
            "planner": "goal",
            "seed": 100 + index,
            "population_arm": "heterogeneous",
        }
        for index in range(3)
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"manifest_rows": rows}), encoding="utf-8")
    output_path = tmp_path / "episode_records.jsonl"
    calls = 0

    def abort_after_two(row: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated abrupt campaign stop")
        return {"scenario_id": row["scenario_id"], "completed": True}

    fsync_calls: list[int] = []
    monkeypatch.setattr(module, "_run_manifest_row", abort_after_two)
    monkeypatch.setattr(module, "configure_campaign_logging", lambda **_kwargs: None)
    monkeypatch.setattr(module.os, "fsync", fsync_calls.append)

    with pytest.raises(RuntimeError, match="simulated abrupt campaign stop"):
        _run_main(
            module,
            [
                RUN_SCRIPT.name,
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
                "--fsync-every",
                "2",
            ],
        )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"scenario_id": "scenario_0", "completed": True},
        {"scenario_id": "scenario_1", "completed": True},
    ]
    assert len(fsync_calls) == 1


@pytest.mark.parametrize("script_path", [BUILD_SCRIPT, RUN_SCRIPT])
def test_campaign_entry_points_default_to_info_with_explicit_debug_opt_in(
    script_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both campaign CLIs default to INFO and honor flag/environment DEBUG opt-in."""

    module = _load_script(script_path, f"issue_5829_logging_{script_path.stem}")
    monkeypatch.delenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", [script_path.name])
    assert module.parse_args().debug is False

    monkeypatch.setenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, "DEBUG")
    assert module.parse_args().debug is True

    monkeypatch.setenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, "INFO")
    monkeypatch.setattr(sys, "argv", [script_path.name, "--debug"])
    assert module.parse_args().debug is True


@pytest.mark.parametrize(
    ("debug", "expected_level"),
    [(False, logging.INFO), (True, logging.DEBUG)],
)
def test_campaign_logging_configures_loguru_and_stdlib_levels(
    debug: bool,
    expected_level: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared campaign setup applies one level to Loguru and stdlib logging."""

    root = logging.getLogger()
    previous_root_level = root.level
    previous_handlers = list(root.handlers)
    previous_handler_levels = {handler: handler.level for handler in previous_handlers}
    loguru_verbose: list[bool] = []
    monkeypatch.setattr(
        campaign_logging,
        "configure_logging",
        lambda *, verbose: loguru_verbose.append(verbose),
    )
    try:
        campaign_logging.configure_campaign_logging(debug=debug)

        assert loguru_verbose == [debug]
        assert root.level == expected_level
        assert all(handler.level == expected_level for handler in root.handlers)
    finally:
        root.setLevel(previous_root_level)
        for handler in list(root.handlers):
            if handler not in previous_handler_levels:
                root.removeHandler(handler)
        for handler, previous_level in previous_handler_levels.items():
            handler.setLevel(previous_level)
