"""Issue #3970 coverage for parameterized draft scenario generation."""

from __future__ import annotations

import pytest
import yaml

from robot_sf.benchmark.scenario_generator import (
    PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION,
    derive_generation_parameters_from_physical_slice,
    normalize_parameterized_scenario_parameters,
    select_map_id_for_parameterized_scenario,
)
from robot_sf.training.scenario_loader import load_scenarios
from scripts.tools.scenario_authoring import build_scenario_payload, dump_scenario_yaml


def test_parameterized_physical_slice_normalizes_and_changes_layout_contract() -> None:
    """Width, density, bottleneck, crossing, and occlusion knobs affect generated output."""

    easy = {
        "sidewalk_width": 6.0,
        "obstacle_density": 0.0,
        "pedestrian_density": 0.02,
        "bottleneck_width": 5.5,
        "crossing_angle": 10.0,
        "occlusion_probability": 0.0,
    }
    dense_bottleneck = {
        "sidewalk_width": 4.0,
        "obstacle_density": 0.1,
        "pedestrian_density": 0.18,
        "bottleneck_width": 1.0,
        "crossing_angle": 130.0,
        "occlusion_probability": 0.75,
    }

    normalized = normalize_parameterized_scenario_parameters(
        {
            "parameterized_profile": {
                "schema_version": PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION,
                "parameters": dense_bottleneck,
            }
        }
    )

    assert normalized == dense_bottleneck
    assert select_map_id_for_parameterized_scenario(easy) == "classic_head_on_corridor"
    assert select_map_id_for_parameterized_scenario(dense_bottleneck) == "classic_bottleneck_high"
    assert derive_generation_parameters_from_physical_slice(easy)["density"] == "low"
    assert derive_generation_parameters_from_physical_slice(dense_bottleneck)["density"] == "high"
    assert derive_generation_parameters_from_physical_slice(dense_bottleneck)["speed_var"] == "high"


def test_parameterized_physical_slice_defaults_and_thresholds() -> None:
    """Default, medium, maze, crossing, and bottleneck thresholds are deterministic."""

    defaults = normalize_parameterized_scenario_parameters()
    medium_crossing = {
        **defaults,
        "pedestrian_density": 0.08,
        "crossing_angle": 90.0,
        "bottleneck_width": defaults["sidewalk_width"],
    }
    medium_bottleneck = {**defaults, "bottleneck_width": defaults["sidewalk_width"] * 0.5}
    wide_bottleneck = {**defaults, "bottleneck_width": defaults["sidewalk_width"] * 0.7}
    maze = {**defaults, "obstacle_density": 0.5}
    merge = {**defaults, "crossing_angle": 60.0}

    assert defaults["sidewalk_width"] == 4.0
    assert select_map_id_for_parameterized_scenario(medium_crossing) == "classic_crossing"
    assert (
        select_map_id_for_parameterized_scenario(medium_bottleneck) == "classic_bottleneck_medium"
    )
    assert select_map_id_for_parameterized_scenario(wide_bottleneck) == "classic_bottleneck"
    assert select_map_id_for_parameterized_scenario(maze) == "classic_cross_trap"
    assert derive_generation_parameters_from_physical_slice(medium_crossing)["density"] == "med"
    assert derive_generation_parameters_from_physical_slice(medium_crossing)["flow"] == "cross"
    assert derive_generation_parameters_from_physical_slice(merge)["flow"] == "merge"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "must be provided as a mapping"),
        (
            {
                "parameterized_profile": {
                    "schema_version": "wrong",
                    "parameters": {},
                }
            },
            "schema_version",
        ),
        (
            {
                "parameterized_profile": {
                    "schema_version": PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION,
                    "parameters": [],
                }
            },
            "parameters must be a mapping",
        ),
        ({"unexpected": 1.0}, "Unknown parameterized scenario keys"),
        ({"sidewalk_width": 0.0}, "sidewalk_width"),
        ({"obstacle_density": 1.5}, "obstacle_density"),
        ({"pedestrian_density": -0.1}, "pedestrian_density"),
        ({"sidewalk_width": 2.0, "bottleneck_width": 3.0}, "bottleneck_width"),
        ({"crossing_angle": 181.0}, "crossing_angle"),
        ({"occlusion_probability": -0.1}, "occlusion_probability"),
        ({"sidewalk_width": float("nan")}, "sidewalk_width must be finite"),
        ({"pedestrian_density": float("inf")}, "pedestrian_density must be finite"),
    ],
)
def test_parameterized_physical_slice_rejects_invalid_inputs(raw, message: str) -> None:
    """Invalid physical knobs fail before draft scenarios can be consumed as evidence."""

    with pytest.raises(ValueError, match=message):
        normalize_parameterized_scenario_parameters(raw)


def test_parameterized_template_is_deterministic_and_loader_consumable(tmp_path) -> None:
    """Generated YAML is deterministic for same params/seed and loads through scenario loader."""

    params = {
        "sidewalk_width": 4.0,
        "obstacle_density": 0.1,
        "pedestrian_density": 0.18,
        "bottleneck_width": 1.0,
        "crossing_angle": 130.0,
        "occlusion_probability": 0.75,
    }
    payload = build_scenario_payload(
        template="parameterized",
        name="issue_3970_dense_bottleneck",
        seeds=(3970,),
        source_issue="#3970",
        parameterized_profile=params,
    )

    first = dump_scenario_yaml(payload)
    second = dump_scenario_yaml(
        build_scenario_payload(
            template="parameterized",
            name="issue_3970_dense_bottleneck",
            seeds=(3970,),
            source_issue="#3970",
            parameterized_profile=params,
        )
    )
    scenario_path = tmp_path / "issue_3970_dense_bottleneck.yaml"
    scenario_path.write_text(first, encoding="utf-8")

    parsed = yaml.safe_load(first)
    scenario = parsed["scenarios"][0]
    loaded = load_scenarios(scenario_path)

    assert first == second
    assert scenario["map_id"] == "classic_bottleneck_high"
    assert scenario["simulation_config"]["ped_density"] == 0.18
    assert scenario["metadata"]["authoring"]["benchmark_evidence"] is False
    assert scenario["metadata"]["authoring"]["source_issue"] == "#3970"
    assert (
        scenario["metadata"]["parameterized_profile"]["schema_version"]
        == PARAMETERIZED_SCENARIO_PARAMS_SCHEMA_VERSION
    )
    assert scenario["metadata"]["parameterized_profile"]["parameters"] == params
    assert loaded[0]["name"] == "issue_3970_dense_bottleneck"


def test_parameterized_template_defaults_to_issue_3970_provenance() -> None:
    """Parameterized authoring defaults to the issue that owns the template."""

    payload = build_scenario_payload(template="parameterized", name="issue_3970_default_source")
    scenario = payload["scenarios"][0]

    assert scenario["metadata"]["authoring"]["source_issue"] == "#3970"
