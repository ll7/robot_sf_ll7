"""Tests for the issue #5303 search-promotion timing-control preflight (issue #6475).

These tests prove the side-effect-free preflight binds the frozen timing dimensions
``spawn_time_s`` and ``pedestrian_delay_s`` to a concrete pedestrian, accepts a non-inert
search space, and fails closed when a dimension targets no pedestrian, is missing, or is
metadata-only. They run no search, planner execution, replay, campaign, or outcome inspection.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D, RangeConfig, SearchSpaceConfig
from robot_sf.benchmark import issue_5303_search_promotion_preflight as preflight_mod
from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    SearchPromotionPreflightError,
    evaluate_preflight,
    evaluate_preflight_from_files,
    load_template_scenario,
    render_markdown,
    to_dict,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


def _space(*, pedestrian_id: str | None = "crossing_probe") -> SearchSpaceConfig:
    """Build a search space with non-degenerate frozen timing ranges."""
    return SearchSpaceConfig(
        start_x=RangeConfig(1.0, 1.0),
        start_y=RangeConfig(2.0, 2.0),
        goal_x=RangeConfig(5.0, 5.0),
        goal_y=RangeConfig(2.0, 2.0),
        spawn_time_s=RangeConfig(0.0, 2.0),
        pedestrian_speed_mps=RangeConfig(1.0, 1.0),
        pedestrian_delay_s=RangeConfig(0.0, 1.5),
        scenario_seed=RangeConfig(7.0, 7.0),
        min_start_goal_distance_m=0.5,
        pedestrian_id=pedestrian_id,
    )


def _template(*, pedestrian_id: str = "crossing_probe") -> dict:
    """Return a minimal scenario-template mapping."""
    return {
        "name": "template",
        "map_id": "classic_cross_trap",
        "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
        "metadata": {"archetype": "test"},
        "seeds": [7],
        "single_pedestrians": [
            {
                "id": pedestrian_id,
                "start": [0.0, 0.0],
                "goal": None,
                "trajectory": [[0.0, 0.0], [1.0, 1.0]],
                "speed_m_s": 1.0,
            }
        ],
    }


def test_preflight_accepts_non_inert_search_space() -> None:
    """A declared pedestrian with bound timing dimensions reaches promotion_timing_ready."""
    result = evaluate_preflight(search_space=_space(), template_scenario=_template())

    assert result.status == "promotion_timing_ready"
    assert result.promotion_ready is True
    assert result.pedestrian_id == "crossing_probe"
    assert result.materialized_pedestrian_id == "crossing_probe"
    assert result.single_pedestrian_populated is True
    assert result.pedestrian_route_populated is True
    assert result.blockers == ()
    assert {probe.name for probe in result.dimensions} == {"spawn_time_s", "pedestrian_delay_s"}
    for probe in result.dimensions:
        assert probe.status == "effective"
        assert probe.declared is True
        assert probe.hash_changed is True
        assert probe.bound_to_pedestrian is True
        assert probe.baseline_hash != probe.perturbed_hash


def test_preflight_rejects_search_space_without_pedestrian_id() -> None:
    """No declared pedestrian.id fails closed with blocked_no_pedestrian (PR #6291 mode)."""
    result = evaluate_preflight(
        search_space=_space(pedestrian_id=None), template_scenario=_template()
    )

    assert result.status == "blocked_no_pedestrian"
    assert result.promotion_ready is False
    assert result.materialized_pedestrian_id is None
    assert result.single_pedestrian_populated is False
    assert result.pedestrian_route_populated is False
    assert result.blockers
    for probe in result.dimensions:
        assert probe.status == "no_pedestrian"
        assert probe.hash_changed is False
        assert probe.bound_to_pedestrian is False


def test_preflight_explicit_pedestrian_override_binds_concrete_pedestrian() -> None:
    """An explicit pedestrian_id override binds a pedestrian even if the space declares none."""
    result = evaluate_preflight(
        search_space=_space(pedestrian_id=None),
        template_scenario=_template(pedestrian_id="override_probe"),
        pedestrian_id="override_probe",
    )

    assert result.status == "promotion_timing_ready"
    assert result.materialized_pedestrian_id == "override_probe"


def test_preflight_normalizes_template_pedestrian_id_like_runtime_loader() -> None:
    """Whitespace around a template id must not hide the loader-bound candidate pedestrian."""
    result = evaluate_preflight(
        search_space=_space(),
        template_scenario=_template(pedestrian_id="  crossing_probe  "),
    )

    assert result.status == "promotion_timing_ready"
    assert result.materialized_pedestrian_id == "crossing_probe"
    assert all(probe.status == "effective" for probe in result.dimensions)


def test_preflight_inspects_candidate_pedestrian_among_preexisting_entries() -> None:
    """A template with other pedestrians must not cause probing the wrong pedestrian.

    The candidate pedestrian is appended after a pre-existing entry; the probe must inspect
    the candidate pedestrian (by identity), not the first list entry, so a non-inert search
    space is not falsely reported as blocked.
    """
    template = _template()
    template["single_pedestrians"] = [
        {
            "id": "aaa_preexisting",
            "start": [0.0, 0.0],
            "goal": [1.0, 1.0],
            "speed_m_s": 1.0,
            "start_delay_s": 99.0,
            "wait_at": [{"waypoint_index": 0, "wait_s": 99.0}],
        },
        {
            "id": "crossing_probe",
            "start": [0.0, 0.0],
            "goal": None,
            "trajectory": [[0.0, 0.0], [1.0, 1.0]],
            "speed_m_s": 1.0,
        },
    ]

    result = evaluate_preflight(search_space=_space(), template_scenario=template)

    assert result.status == "promotion_timing_ready"
    assert result.materialized_pedestrian_id == "crossing_probe"
    assert result.single_pedestrian_populated is True
    for probe in result.dimensions:
        assert probe.status == "effective"
        assert probe.bound_to_pedestrian is True
        assert probe.bound_value != pytest.approx(99.0)


def test_preflight_rejects_metadata_only_timing_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timing dimensions that survive only in metadata must be rejected as inert."""
    real_builder = preflight_mod.build_candidate_payload

    def _metadata_only_builder(
        candidate,
        *,
        index,
        template_scenario,
        pedestrian_id,
        route_file_name="route_overrides.yaml",
    ):
        scenario, route = real_builder(
            candidate,
            index=index,
            template_scenario=template_scenario,
            pedestrian_id=pedestrian_id,
            route_file_name=route_file_name,
        )
        # Simulate the PR #6291 inert materialization: timing survives only in provenance
        # metadata, never in the runtime-effective single_pedestrians or pedestrian route.
        for entry in scenario.get("single_pedestrians") or []:
            entry.pop("start_delay_s", None)
            entry.pop("wait_at", None)
        route["ped_routes"] = []
        return scenario, route

    monkeypatch.setattr(preflight_mod, "build_candidate_payload", _metadata_only_builder)

    result = evaluate_preflight(search_space=_space(), template_scenario=_template())

    assert result.status == "blocked_inert_dimensions"
    assert result.promotion_ready is False
    for probe in result.dimensions:
        assert probe.status == "inert_metadata_only"
        assert probe.hash_changed is False
        assert probe.bound_to_pedestrian is False
    assert any("metadata-only" in blocker for blocker in result.blockers)


def test_preflight_rejects_missing_timing_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """An undeclared frozen timing dimension fails closed with blocked_missing_dimension."""
    monkeypatch.setattr(
        preflight_mod, "PROMOTION_TIMING_DIMENSIONS", ("spawn_time_s", "undeclared_dimension")
    )

    result = evaluate_preflight(search_space=_space(), template_scenario=_template())

    assert result.status == "blocked_missing_dimension"
    assert result.promotion_ready is False
    by_name = {probe.name: probe for probe in result.dimensions}
    assert by_name["undeclared_dimension"].status == "missing"
    assert by_name["undeclared_dimension"].declared is False
    assert by_name["spawn_time_s"].status == "effective"


def test_preflight_rejects_omitted_declared_timing_dimension(tmp_path: Path) -> None:
    """A defaulted timing range must not masquerade as a YAML-declared dimension."""
    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 2.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "pedestrian": {"id": "crossing_probe"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    template_path = tmp_path / "template.yaml"
    template_path.write_text(
        yaml.safe_dump({"scenarios": [_template()]}, sort_keys=False), encoding="utf-8"
    )

    result = evaluate_preflight_from_files(
        search_space_path=space_path,
        scenario_template_path=template_path,
    )

    assert result.status == "blocked_missing_dimension"
    delay_probe = next(probe for probe in result.dimensions if probe.name == "pedestrian_delay_s")
    assert delay_probe.declared is False
    assert delay_probe.status == "missing"


def test_preflight_perturbations_stay_inside_declared_ranges() -> None:
    """Timing probes must use values the configured search space can actually sample."""
    space = _space()
    result = evaluate_preflight(search_space=space, template_scenario=_template())

    for probe in result.dimensions:
        timing_range = space.timing_dimension_range(probe.name)
        assert timing_range is not None
        assert timing_range.contains(probe.perturbed_value)


def test_preflight_rejects_degenerate_timing_range() -> None:
    """A fixed timing value cannot prove a one-at-a-time runtime perturbation."""
    space = replace(_space(), spawn_time_s=RangeConfig(0.0, 0.0))

    result = evaluate_preflight(search_space=space, template_scenario=_template())

    assert result.status == "blocked_inert_dimensions"
    spawn_probe = next(probe for probe in result.dimensions if probe.name == "spawn_time_s")
    assert spawn_probe.perturbed_value == pytest.approx(spawn_probe.baseline_value)
    assert spawn_probe.hash_changed is False
    assert spawn_probe.status == "inert_metadata_only"


def test_preflight_is_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe materializes in memory: it writes no files and is deterministic."""
    monkeypatch.chdir(tmp_path)

    first = evaluate_preflight(search_space=_space(), template_scenario=_template())
    second = evaluate_preflight(search_space=_space(), template_scenario=_template())

    assert list(tmp_path.iterdir()) == []
    assert to_dict(first) == to_dict(second)


def test_preflight_from_files_round_trip(tmp_path: Path) -> None:
    """evaluate_preflight_from_files loads on-disk inputs read-only and reaches ready."""
    space_path = tmp_path / "space.yaml"
    template_path = tmp_path / "template.yaml"
    space_path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 2.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 1.5},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
                "pedestrian": {"id": "crossing_probe"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    template_path.write_text(
        yaml.safe_dump({"scenarios": [_template()]}, sort_keys=False), encoding="utf-8"
    )

    result = evaluate_preflight_from_files(
        search_space_path=space_path, scenario_template_path=template_path
    )

    assert result.status == "promotion_timing_ready"


def test_preflight_from_files_fails_closed_on_missing_inputs(tmp_path: Path) -> None:
    """Missing search-space or template files raise the fail-closed preflight error."""
    template_path = tmp_path / "template.yaml"
    template_path.write_text(
        yaml.safe_dump({"scenarios": [_template()]}, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(SearchPromotionPreflightError):
        evaluate_preflight_from_files(
            search_space_path=tmp_path / "absent_space.yaml",
            scenario_template_path=template_path,
        )

    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(SearchPromotionPreflightError):
        evaluate_preflight_from_files(
            search_space_path=space_path,
            scenario_template_path=tmp_path / "absent_template.yaml",
        )


def test_load_template_scenario_rejects_missing_scenario(tmp_path: Path) -> None:
    """A template without a scenario mapping fails closed."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"scenarios": []}, sort_keys=False), encoding="utf-8")

    with pytest.raises(SearchPromotionPreflightError):
        load_template_scenario(bad)


def test_materialized_timing_controls_load_into_runtime_single_pedestrian(tmp_path: Path) -> None:
    """Generated wait controls must survive the canonical scenario loader."""
    source_path = Path("configs/scenarios/single/francis2023_intersection_wait.yaml").resolve()
    source_template = yaml.safe_load(source_path.read_text(encoding="utf-8"))["scenarios"][0]
    candidate = CandidateSpec(
        start=Pose2D(14.0, 17.5),
        goal=Pose2D(14.0, 4.0),
        spawn_time_s=1.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.75,
        scenario_seed=240,
    )
    scenario, route_payload = preflight_mod.build_candidate_payload(
        candidate,
        index=0,
        template_scenario=source_template,
        pedestrian_id="h1",
    )
    scenario["map_file"] = str(
        Path("maps/svg_maps/francis2023/francis2023_intersection_no_gesture.svg").resolve()
    )
    route_path = tmp_path / "routes.yaml"
    route_path.write_text(yaml.safe_dump(route_payload, sort_keys=False), encoding="utf-8")
    scenario_path = tmp_path / "scenario.yaml"
    scenario["route_overrides_file"] = route_path.name
    scenario_path.write_text(yaml.safe_dump({"scenarios": [scenario]}), encoding="utf-8")

    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    map_def = next(iter(config.map_pool.map_defs.values()))
    pedestrian = next(ped for ped in map_def.single_pedestrians if ped.id == "h1")

    assert pedestrian.start == (14.0, 17.5)
    assert pedestrian.goal is None
    assert pedestrian.trajectory == [(14.0, 17.5), (14.0, 4.0)]
    assert pedestrian.start_delay_s == pytest.approx(1.0)
    assert pedestrian.wait_at is not None
    assert pedestrian.wait_at[0].wait_s == pytest.approx(0.75)
    assert map_def.ped_routes[0].waypoints == [(14.0, 17.5), (14.0, 4.0)]


def test_preflight_to_dict_and_markdown_surface_status() -> None:
    """Serialization surfaces the fail-closed status, blockers, and campaign gates."""
    result = evaluate_preflight(
        search_space=_space(pedestrian_id=None), template_scenario=_template()
    )

    payload = to_dict(result)
    assert payload["schema_version"] == "issue-5303-search-promotion-preflight.v1"
    assert payload["status"] == "blocked_no_pedestrian"
    assert payload["promotion_ready"] is False
    assert payload["blockers"]
    assert payload["campaign_gates"]
    assert len(payload["dimensions"]) == 2

    markdown = render_markdown(result)
    assert "blocked_no_pedestrian" in markdown
    assert "campaign gates" in markdown.lower()
