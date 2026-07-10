"""CPU-only tests for issue #4932's stage 1--3 generation MVP."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark.scenario_generation.catalog_writer import (
    deduplicate_catalog_entries,
)
from robot_sf.benchmark.scenario_generation.pipeline import run_generation_pipeline
from robot_sf.benchmark.scenario_generation.random_sampler import sample_episode_jobs
from robot_sf.benchmark.scenario_generation.replay_validation import assess_replay_status
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

if TYPE_CHECKING:
    from pathlib import Path


def _source_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": "crossing-low",
            "map_file": "maps/svg_maps/classic_crossing.svg",
            "simulation_config": {"ped_density": 0.02},
            "metadata": {"archetype": "crossing"},
        },
        {
            "name": "crossing-high",
            "map_file": "maps/svg_maps/classic_crossing.svg",
            "simulation_config": {"ped_density": 0.08},
            "metadata": {"archetype": "crossing"},
        },
    ]


def _trace_episode(
    episode_id: str,
    *,
    clearance_m: float,
    offset_x: float = 0.0,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "seed": 4932,
        "source_map": "maps/svg_maps/classic_crossing.svg",
        "steps": [
            {
                "time_s": 0.0,
                "robot": {"position": [offset_x, 0.0]},
                "pedestrians": [{"position": [offset_x + 3.0, 0.0]}],
            },
            {
                "time_s": 1.0,
                "robot": {"position": [offset_x + 1.0, 0.0]},
                "pedestrians": [{"position": [offset_x + 1.0 + clearance_m, 0.0]}],
            },
            {
                "time_s": 2.0,
                "robot": {"position": [offset_x + 2.0, 0.0]},
                "pedestrians": [{"position": [offset_x + 5.0, 0.0]}],
            },
        ],
    }


def test_fixed_seed_produces_identical_sampled_scenarios() -> None:
    """The sampler owns a local RNG and records every selected random parameter."""

    first = sample_episode_jobs(_source_scenarios(), seed=4932, episode_budget=8)
    second = sample_episode_jobs(_source_scenarios(), seed=4932, episode_budget=8)

    assert [sample.manifest_record() for sample in first] == [
        sample.manifest_record() for sample in second
    ]
    assert [sample.scenario for sample in first] == [sample.scenario for sample in second]
    assert len({sample.episode_seed for sample in first}) == 8
    assert all(
        sample.scenario["metadata"]["scenario_generation"]["benchmark_evidence"] is False
        for sample in first
    )


def test_dedup_keeps_higher_criticality_near_duplicate() -> None:
    """Nearby features retain the smaller-clearance exemplar with a reason record."""

    less_critical = extract_critical_segment(
        _trace_episode("episode-less", clearance_m=0.6, offset_x=0.2)
    )
    more_critical = extract_critical_segment(
        _trace_episode("episode-more", clearance_m=0.2, offset_x=0.0)
    )

    kept, dropped = deduplicate_catalog_entries(
        [less_critical, more_critical], distance_threshold=1.0
    )

    assert [entry["scenario_id"] for entry in kept] == [more_critical["scenario_id"]]
    assert dropped == [
        {
            "dropped_scenario_id": less_critical["scenario_id"],
            "kept_scenario_id": more_critical["scenario_id"],
            "reason": "near_duplicate_lower_criticality",
            "feature_distance": pytest.approx(0.2),
            "distance_threshold": 1.0,
            "group": {
                "source_map_family": "classic_crossing",
                "criticality_signal": "min_clearance",
                "actor_count": 1,
            },
        }
    ]


def test_replay_status_distinguishes_source_load_from_exact_replay(tmp_path: Path) -> None:
    """A source loader pass cannot promote an unrepresentable generated segment."""

    entry = extract_critical_segment(_trace_episode("episode-load", clearance_m=0.2))
    loaded = assess_replay_status(
        entry,
        source_scenario=_source_scenarios()[0],
        scenario_path=tmp_path / "source.yaml",
        config_builder=lambda *_args, **_kwargs: object(),
    )

    assert loaded["replay"]["status"] == "not_representable_yet"
    assert loaded["replay"]["warnings"][0].startswith("replay_gap:")
    assert "source scenario load passed" in loaded["replay"]["warnings"][0]
    assert "not representable" in loaded["replay"]["warnings"][0]


def test_replay_status_records_loader_failure_without_overclaim(tmp_path: Path) -> None:
    """An unrepresentable source records the concrete loader failure."""

    def fail_loader(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("bad map reference")

    entry = extract_critical_segment(_trace_episode("episode-fail", clearance_m=0.2))
    failed = assess_replay_status(
        entry,
        source_scenario=_source_scenarios()[0],
        scenario_path=tmp_path / "source.yaml",
        config_builder=fail_loader,
    )

    assert failed["replay"]["status"] == "not_representable_yet"
    assert "bad map reference" in failed["replay"]["warnings"][0]


def test_pipeline_writes_deterministic_hypothesis_catalog_and_provenance(
    tmp_path: Path,
) -> None:
    """One CLI-level function joins sampling, distillation, catalog, and replay status."""

    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "data-driven-scenario-generation.v1",
                "seed": 4932,
                "source_scenarios": "unused-source.yaml",
                "episode_budget": 2,
                "sampler": {
                    "type": "monte_carlo",
                    "robot_start_goal_policy": "sampled_map_route_pairs",
                    "pedestrian_policy": "sampled_map_routes_and_population",
                    "obstacle_policy": "disabled_for_mvp",
                },
                "runner": {"algo": "goal", "horizon": 3, "dt": 1.0},
                "extraction": {"pre_margin_s": 1.0, "post_margin_s": 1.0},
                "deduplication": {"distance_threshold": 0.0},
                "output_root": output_root.as_posix(),
                "claim_boundary": "generated scenario hypotheses only",
            }
        ),
        encoding="utf-8",
    )

    def fake_batch_runner(
        scenarios: list[dict[str, Any]], out_path: Path, *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        rows = []
        for index, scenario in enumerate(scenarios):
            seed = scenario["seeds"][0]
            episode = _trace_episode(
                f"episode-{index}", clearance_m=0.2 + index, offset_x=index * 10.0
            )
            steps = []
            for step in episode["steps"]:
                expanded = deepcopy(step)
                expanded["robot"].update({"heading": 0.0, "velocity": [1.0, 0.0]})
                steps.append(expanded)
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "scenario_id": scenario["name"],
                    "seed": seed,
                    "status": "failure",
                    "termination_reason": "timeout",
                    "metrics": {"min_distance": 0.2 + index},
                    "algorithm_metadata": {
                        "simulation_step_trace": {
                            "schema_version": "simulation-step-trace.v1",
                            "steps": steps,
                        }
                    },
                }
            )
        out_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {
            "total_jobs": len(rows),
            "written": len(rows),
            "successful_jobs": len(rows),
            "failed_jobs": 0,
            "failures": [],
        }

    manifest = run_generation_pipeline(
        config_path,
        scenario_loader=lambda _path: _source_scenarios(),
        batch_runner=fake_batch_runner,
        replay_config_builder=lambda *_args, **_kwargs: object(),
    )

    catalog = yaml.safe_load((output_root / "generated_catalog.yaml").read_text())
    provenance = json.loads((output_root / "generated_catalog.provenance.json").read_text())
    outcomes = [
        json.loads(line)
        for line in (output_root / "episode_outcomes.jsonl").read_text().splitlines()
    ]
    assert manifest["schema_version"] == "scenario-generation-run.v1"
    assert manifest["episode_count"] == 2
    assert manifest["catalog"] == {
        "candidate_count": 2,
        "kept_count": 2,
        "dropped_duplicate_count": 0,
    }
    assert catalog["metadata"] == {
        "source": "auto_generated",
        "required_manual_review": True,
        "benchmark_evidence": False,
        "claim_boundary": "generated scenario hypotheses only",
        "release_matrix_inclusion": False,
    }
    assert all(entry["replay"]["status"] == "not_representable_yet" for entry in catalog["entries"])
    assert all(entry["metadata"]["benchmark_evidence"] is False for entry in catalog["entries"])
    assert all(outcome["benchmark_evidence"] is False for outcome in outcomes)
    assert all(outcome["criticality_time_series"] for outcome in outcomes)
    assert len(provenance["entries"]) == 2


def test_pipeline_refuses_to_mix_with_existing_output(tmp_path: Path) -> None:
    """Existing output fails closed instead of mixing run provenance."""

    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "old-run.json").write_text("{}", encoding="utf-8")
    config = {
        "schema_version": "data-driven-scenario-generation.v1",
        "seed": 1,
        "source_scenarios": "unused.yaml",
        "episode_budget": 1,
        "sampler": {"type": "monte_carlo", "obstacle_policy": "disabled_for_mvp"},
        "runner": {"algo": "goal", "horizon": 1},
        "deduplication": {"distance_threshold": 1.0},
        "output_root": output_root.as_posix(),
        "claim_boundary": "generated scenario hypotheses only",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        run_generation_pipeline(config_path)
