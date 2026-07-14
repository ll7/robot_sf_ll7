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

import yaml

from robot_sf.benchmark.scenario_generation.persistence_gate import validate_persistence_record
from robot_sf.benchmark.scenario_generation.pipeline import (
    run_generation_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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
    records = evaluate_pipeline_candidates(
        manifest_path=output_root / "run_manifest.json",
        candidate_path=None,
        episodes_path=None,
        config_path=Path(_PERSISTENCE_CONFIG),
        output_dir=tmp_path / "records",
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
