"""CPU-only wiring test: real pipeline run -> persistence gate -> promote + reject.

This is the slice the gate schema/unit tests (test_scenario_persistence_gate.py)
do not cover: it actually runs the stage-1 generation pipeline (#4932) with a
fake batch runner, then wires the produced catalog + episodes through
``evaluate_pipeline_persistence_gate`` (issue #5600) and asserts that the
promotion gate emits the expected promote/reject verdicts.  No simulations, no
Slurm, no campaigns; the fake runner supplies deterministic step traces.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark.scenario_generation.persistence_gate import validate_persistence_record
from robot_sf.benchmark.scenario_generation.pipeline import (
    run_generation_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
_PERSISTENCE_CONFIG = "configs/analysis/issue_5600_persistence_gate.yaml"
_FROZEN_CONFIG_ID = "issue-5600-persistence-gate"


def _source_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": "crossing-low",
            "map_file": "maps/svg_maps/classic_crossing.svg",
            "simulation_config": {"ped_density": 0.02},
            "metadata": {"archetype": "crossing"},
        }
    ]


def _promote_trace(episode_id: str, *, offset_x: float = 0.0) -> dict[str, Any]:
    """A near-static robot/pedestrian pair: the critical event persists under perturbations."""

    pedestrian = [offset_x + 0.3, 0.0]
    return {
        "episode_id": episode_id,
        "seed": 4932,
        "source_map": "maps/svg_maps/classic_crossing.svg",
        "steps": [
            {
                "time_s": 0.0,
                "robot": {"position": [offset_x, 0.0]},
                "pedestrians": [{"position": pedestrian}],
            },
            {
                "time_s": 1.0,
                "robot": {"position": [offset_x, 0.0]},
                "pedestrians": [{"position": pedestrian}],
            },
            {
                "time_s": 2.0,
                "robot": {"position": [offset_x, 0.0]},
                "pedestrians": [{"position": pedestrian}],
            },
        ],
    }


def _reject_trace(episode_id: str, *, offset_x: float = 0.0) -> dict[str, Any]:
    """A moving robot/pedestrian pair: the critical event dissipates under perturbation."""

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
                "pedestrians": [{"position": [offset_x + 1.0 + 0.2, 0.0]}],
            },
            {
                "time_s": 2.0,
                "robot": {"position": [offset_x + 2.0, 0.0]},
                "pedestrians": [{"position": [offset_x + 5.0, 0.0]}],
            },
        ],
    }


def _fake_batch_runner(
    scenarios: Sequence[Mapping[str, Any]],
    out_path: Path,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        trace = _promote_trace(scenario["name"]) if index == 0 else _reject_trace(scenario["name"])
        steps = [
            {**step, "robot": {**step["robot"], "heading": 0.0, "velocity": [0.0, 0.0]}}
            for step in trace["steps"]
        ]
        rows.append(
            {
                "episode_id": trace["episode_id"],
                "scenario_id": scenario["name"],
                "seed": scenario["seeds"][0],
                "scenario_params": {"map_file": trace["source_map"]},
                "status": "failure",
                "termination_reason": "timeout",
                "metrics": {},
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


def _run_wiring(tmp_path: Path) -> list[dict[str, Any]]:
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

    from scripts.tools.evaluate_pipeline_persistence_gate import (
        evaluate_pipeline_candidates,
    )

    run_generation_pipeline(
        config_path,
        scenario_loader=lambda _path: _source_scenarios(),
        batch_runner=_fake_batch_runner,
        replay_config_builder=lambda *_args, **_kwargs: object(),
    )
    episodes = [
        json.loads(line)
        for line in (output_root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries = yaml.safe_load((output_root / "generated_catalog.yaml").read_text(encoding="utf-8"))[
        "entries"
    ]
    episodes_by_id = {episode["episode_id"]: episode for episode in episodes}
    replay_results: dict[str, Any] = {}
    for index, entry in enumerate(entries):
        episode = episodes_by_id[entry["source_episode"]["episode_id"]]
        replay_results[entry["scenario_id"]] = {
            "episode": {
                "episode_id": episode["episode_id"],
                "source_seed": episode["seed"],
                "source_map": entry["source_episode"]["source_map"],
                "steps": episode["algorithm_metadata"]["simulation_step_trace"]["steps"],
            },
            "cells": [
                {
                    "timing_offset_s": timing_offset_s,
                    "speed_delta_m_s": speed_delta_m_s,
                    "verdict": "pass" if index == 0 else "fail",
                    "reason": "deterministic replay fixture",
                }
                for timing_offset_s in (-0.25, 0.0, 0.25)
                for speed_delta_m_s in (-0.2, 0.0, 0.2)
            ],
            "missing_cell_reasons": [],
        }
    replay_results_path = tmp_path / "replay_results.json"
    replay_results_path.write_text(json.dumps(replay_results), encoding="utf-8")
    records = evaluate_pipeline_candidates(
        manifest_path=output_root / "run_manifest.json",
        candidate_path=None,
        episodes_path=None,
        config_path=REPO_ROOT / _PERSISTENCE_CONFIG,
        output_dir=tmp_path / "records",
        replay_results_path=replay_results_path,
        commit_hash="deadbee",
        config_hash="c0ffee",
    )
    return records


def test_wiring_runs_pipeline_artifact_through_gate_and_promotes_one() -> None:
    """A real pipeline run yields one promote + one reject persistence record."""

    with tempfile.TemporaryDirectory() as tmp:
        records = _run_wiring(Path(tmp))

    assert len(records) == 2
    verdicts = {record["scenario_id"]: record["promotion"]["verdict"] for record in records}
    assert set(verdicts.values()) == {"promote", "reject"}

    promoted = next(record for record in records if record["promotion"]["verdict"] == "promote")
    rejected = next(record for record in records if record["promotion"]["verdict"] == "reject")

    # The promoted candidate passes all three independent status checks.
    assert promoted["exact_replay"]["status"] == "pass"
    assert promoted["critical_event_reproduced"]["status"] == "pass"
    assert promoted["perturbation_persistence"]["persistence_rate"] == 1.0
    assert promoted["promotion"]["exclusion_reason"] == "all three independent status checks passed"

    # The rejected candidate fails closed on perturbation cells, even though
    # its exact replay and critical event both pass.
    assert rejected["exact_replay"]["status"] == "pass"
    assert rejected["critical_event_reproduced"]["status"] == "pass"
    assert rejected["perturbation_persistence"]["persistence_rate"] < 1.0
    assert "perturbation_cell:" in rejected["promotion"]["exclusion_reason"]

    for record in records:
        validate_persistence_record(record)


def test_wiring_emits_schema_valid_evidence_records(tmp_path: Path) -> None:
    """Each emitted record is a valid generated_scenario_persistence.v1 file on disk."""

    records = _run_wiring(tmp_path)
    assert len(list(tmp_path.glob("records/*.persistence.json"))) == len(records)
    for record in records:
        assert record["config"]["frozen"] is True
        assert record["config"]["config_id"] == _FROZEN_CONFIG_ID
        assert record["commit_hashes"] == {"code": "deadbee", "config": "c0ffee"}
        validate_persistence_record(record)


def test_cpu_speed_delta_uses_frame_duration_not_timing_offset() -> None:
    """Timing sign must not change the physical duration of a speed perturbation."""

    from scripts.tools.evaluate_pipeline_persistence_gate import _CpuCellVerdict

    evaluator = _CpuCellVerdict(
        original_steps=[
            {"time_s": 0.0, "robot": {"position": [0.0, 0.0]}, "pedestrians": []},
            {"time_s": 1.0, "robot": {"position": [1.0, 0.0]}, "pedestrians": []},
        ],
        original_critical_value=None,
        event_type="min_clearance",
    )

    for timing_offset_s in (-0.25, 0.25):
        shifted = evaluator._shift_steps(timing_offset_s, 0.2)
        assert shifted[-1]["robot"]["position"] == pytest.approx([1.2, 0.0])


def test_cpu_only_mode_does_not_claim_exact_replay() -> None:
    """Without independent replay input, exact replay remains unknown."""

    from scripts.tools.evaluate_pipeline_persistence_gate import _assess_exact_replay

    source = {
        "episode_id": "episode-1",
        "source_seed": 7,
        "source_map": "maps/svg_maps/classic_crossing.svg",
    }
    block = _assess_exact_replay(
        source,
        source_trace={**source, "seed": 7, "steps": []},
        replayed_episode=None,
        replay_error=None,
    )
    assert block["status"] == "unknown"


def test_precomputed_replay_validation_errors(tmp_path: Path) -> None:
    """The persistence gate must raise ValueError on duplicate, extra, missing, or malformed cells."""

    # 1. Run generation pipeline once to produce valid catalog and episodes
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

    from scripts.tools.evaluate_pipeline_persistence_gate import (
        evaluate_pipeline_candidates,
    )

    run_generation_pipeline(
        config_path,
        scenario_loader=lambda _path: _source_scenarios(),
        batch_runner=_fake_batch_runner,
        replay_config_builder=lambda *_args, **_kwargs: object(),
    )

    # Resolve paths
    catalog_path = output_root / "generated_catalog.yaml"
    entries = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2

    # Find the scenario IDs to construct replay_results
    entry = entries[0]

    episodes_path = output_root / "episodes.jsonl"
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episodes_by_id = {episode["episode_id"]: episode for episode in episodes}

    # Construct the valid replay results structure for entries[0]
    episode = episodes_by_id[entry["source_episode"]["episode_id"]]

    # timing_offsets_s: [-0.25, 0.0, 0.25]
    # speed_deltas_m_s: [-0.2, 0.0, 0.2]
    # Total 9 cells.
    valid_cells = [
        {
            "timing_offset_s": timing_offset_s,
            "speed_delta_m_s": speed_delta_m_s,
            "verdict": "pass",
            "reason": "deterministic replay fixture",
        }
        for timing_offset_s in (-0.25, 0.0, 0.25)
        for speed_delta_m_s in (-0.2, 0.0, 0.2)
    ]

    valid_replay_info = {
        "episode": {
            "episode_id": episode["episode_id"],
            "source_seed": episode["seed"],
            "source_map": entry["source_episode"]["source_map"],
            "steps": episode["algorithm_metadata"]["simulation_step_trace"]["steps"],
        },
        "cells": valid_cells,
        "missing_cell_reasons": [],
    }

    # We evaluate just for candidate_path pointing to a single candidate (entries[0])
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(yaml.safe_dump(entry), encoding="utf-8")

    # Helper to run validation and assert failure
    def assert_validation_fails(replay_results_data: dict[str, Any], match_err: str) -> None:
        replay_results_path = tmp_path / "replay_results.json"
        replay_results_path.write_text(json.dumps(replay_results_data), encoding="utf-8")
        with pytest.raises(ValueError, match=match_err):
            evaluate_pipeline_candidates(
                manifest_path=None,
                candidate_path=candidate_path,
                episodes_path=episodes_path,
                config_path=REPO_ROOT / _PERSISTENCE_CONFIG,
                output_dir=tmp_path / "records",
                replay_results_path=replay_results_path,
                commit_hash="deadbee",
                config_hash="c0ffee",
            )

    # 2. Test missing cells (partial grid)
    partial_cells = valid_cells[:-1]
    assert_validation_fails(
        {entry["scenario_id"]: {**valid_replay_info, "cells": partial_cells}},
        "missing cells for coordinates",
    )

    # 3. Test extra/unregistered cells
    extra_cell = {"timing_offset_s": 1.0, "speed_delta_m_s": 0.0, "verdict": "pass", "reason": "ok"}
    assert_validation_fails(
        {entry["scenario_id"]: {**valid_replay_info, "cells": valid_cells + [extra_cell]}},
        "extra/unregistered cells",
    )

    # 4. Test duplicate cells
    duplicate_cell = valid_cells[0].copy()
    assert_validation_fails(
        {entry["scenario_id"]: {**valid_replay_info, "cells": valid_cells + [duplicate_cell]}},
        "duplicate cells for coordinates",
    )

    # 5. Test malformed cell (missing required keys)
    malformed_cell = {"timing_offset_s": 0.0, "speed_delta_m_s": 0.0}  # missing verdict
    assert_validation_fails(
        {entry["scenario_id"]: {**valid_replay_info, "cells": valid_cells[:-1] + [malformed_cell]}},
        "missing required keys",
    )

    # 6. Test explicitly missing cells (non-empty missing_cell_reasons)
    explicit_missing_reasons = [
        {"timing_offset_s": 0.0, "speed_delta_m_s": 0.0, "reason": "timeout"}
    ]
    assert_validation_fails(
        {
            entry["scenario_id"]: {
                **valid_replay_info,
                "missing_cell_reasons": explicit_missing_reasons,
            }
        },
        "explicitly missing cell reasons",
    )
