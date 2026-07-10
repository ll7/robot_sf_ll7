"""Contract tests for issue #5203 generated critical-segment replay materialization."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.scenario_generation import (
    dump_generated_scenario_yaml,
    extract_critical_segment,
    generated_replay_status_entry,
    materialize_generated_scenario,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios


def _entry() -> dict:
    """Return a safely in-bounds critical segment with a stable pedestrian roster."""

    return extract_critical_segment(
        {
            "episode_id": "generated-replay-fixture",
            "seed": 5203,
            "source_map": str(
                Path(__file__).resolve().parents[2] / "maps/svg_maps/classic_crossing.svg"
            ),
            "steps": [
                {
                    "time_s": 0.0,
                    "robot": {"position": [5.0, 5.0]},
                    "pedestrians": [{"position": [10.0, 10.0]}],
                },
                {
                    "time_s": 1.0,
                    "robot": {"position": [6.0, 5.0]},
                    "pedestrians": [{"position": [9.5, 10.0]}],
                },
                {
                    "time_s": 2.0,
                    "robot": {"position": [7.0, 5.0]},
                    "pedestrians": [{"position": [9.0, 10.0]}],
                },
            ],
        },
        pre_margin_s=1.0,
        post_margin_s=1.0,
    )


def test_materializer_writes_deterministic_generated_only_loader_scenario(tmp_path: Path) -> None:
    """A representable entry becomes stable YAML that pins source-map actor state."""

    entry = _entry()
    result = materialize_generated_scenario(entry, max_episode_steps=3)

    assert result.status == "loads_only"
    assert result.warnings == ()
    assert dump_generated_scenario_yaml(result) == dump_generated_scenario_yaml(result)
    scenario_path = tmp_path / "generated-replay.yaml"
    scenario_path.write_text(dump_generated_scenario_yaml(result), encoding="utf-8")
    scenario = load_scenarios(scenario_path)[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    map_def = next(iter(config.map_pool.map_defs.values()))

    assert scenario["metadata"]["required_manual_review"] is True
    assert scenario["metadata"]["benchmark_evidence"] is False
    assert map_def.robot_routes[0].waypoints == [(6.0, 5.0), (7.0, 5.0)]
    assert map_def.robot_spawn_zones == [((6.0, 5.0), (6.0, 5.0), (6.0, 5.0))]
    assert map_def.single_pedestrians[0].start == (9.5, 10.0)
    assert map_def.single_pedestrians[0].trajectory == [
        (9.5, 10.0),
        (9.0, 10.0),
    ]


def test_materializer_fails_closed_for_a_dynamic_pedestrian_roster() -> None:
    """Actor insertion/removal cannot be silently converted into a replay route."""

    entry = _entry()
    entry["segment"]["trace_frames"][1]["pedestrians"].append({"position": [8.0, 8.0]})

    result = materialize_generated_scenario(entry)
    status_entry = generated_replay_status_entry(entry, result)

    assert result.scenario_document is None
    assert result.status == "not_representable_yet"
    assert result.warnings == ("replay_gap: pedestrian count changes at trace frame 1 (1 -> 2)",)
    assert status_entry["replay"] == {
        "schema_version": "generated-scenario-replay.v1",
        "source_seed": 5203,
        "replay_contract": "source_episode_seed_pinned.v1",
        "status": "not_representable_yet",
        "warnings": ["replay_gap: pedestrian count changes at trace frame 1 (1 -> 2)"],
    }


def test_materialized_yaml_marks_the_generated_hypothesis_as_manual_review_only(
    tmp_path: Path,
) -> None:
    """The generated YAML keeps its non-benchmark manual-review boundary explicit."""

    result = materialize_generated_scenario(_entry())
    scenario_path = tmp_path / "generated.yaml"
    scenario_path.write_text(dump_generated_scenario_yaml(result), encoding="utf-8")
    loaded = load_scenarios(scenario_path)
    assert loaded[0]["metadata"]["benchmark_evidence"] is False
    assert loaded[0]["metadata"]["required_manual_review"] is True
