"""Evidence checks for the frozen-release trace comparison."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.validation import check_issue_9671_trace_reexport as checker
from scripts.validation.check_issue_9671_trace_reexport import SOURCE_SHA, check

if TYPE_CHECKING:
    from pathlib import Path


def _row(seed: int, status: str, *, trace: bool) -> dict:
    params = {
        "algo": "ppo",
        "run_horizon": 600,
        "run_dt": 0.1,
        "simulation_config": {
            "route_spawn_seed": seed,
            "goal_completion_policy": "goal_zone_entry_v1",
        },
    }
    if trace:
        params.update(
            record_forces=True,
            record_planner_decision_trace=True,
            record_simulation_step_trace=True,
        )
    return {
        "algo": "ppo",
        "scenario_id": "classic_doorway_medium",
        "seed": seed,
        "status": status,
        "git_hash": SOURCE_SHA,
        "scenario_params": params,
        "algorithm_metadata": {
            "simulation_step_trace": {
                "steps": [
                    {
                        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "heading": 0.0},
                        "pedestrians": [],
                    }
                ]
            }
        },
    }


def _archive(path: Path, rows: list[dict]) -> None:
    data = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    member = tarfile.TarInfo("bundle/payload/runs/ppo__differential_drive/episodes.jsonl")
    member.size = len(data)
    with tarfile.open(path, "w:gz") as bundle:
        bundle.addfile(member, io.BytesIO(data))


def _bindings(tmp_path: Path, seeds: list[int]) -> tuple[dict, dict, dict, dict]:
    configs = {}
    manifests = {}
    digests = {}
    effective = {"headon_group": "test-headon-hash", "doorway": "test-doorway-hash"}
    for name in effective:
        selected = seeds if name == "headon_group" else []
        config = {
            "horizon": 600,
            "dt": 0.1,
            "kinematics_matrix": ["differential_drive"],
            "record_forces": True,
            "record_planner_decision_trace": True,
            "record_simulation_step_trace": True,
            "scenario_matrix": "test-matrix.yaml",
            "scenario_candidates": ["classic_doorway_medium"] if selected else [],
            "seed_policy": {"seeds": selected},
            "planners": [{"key": "ppo"}] if selected else [],
        }
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        configs[name] = config_path
        digests[name] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path = tmp_path / f"{name}-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "git": {"commit": SOURCE_SHA},
                    "config_hash": effective[name],
                    "scenario_matrix": "test-matrix.yaml",
                    "seed_policy": {"resolved_seeds": selected},
                }
            )
        )
        manifests[name] = manifest_path
    return configs, manifests, digests, effective


def _check(archive: Path, traces: Path, tmp_path: Path, seeds: list[int]) -> dict:
    configs, manifests, digests, effective = _bindings(tmp_path, seeds)
    return check(
        archive,
        [traces],
        configs,
        manifests,
        expected={("ppo", "classic_doorway_medium", seed) for seed in seeds},
        expected_archive_sha256=None,
        expected_config_sha256=digests,
        expected_effective_hash=effective,
    )


def test_reports_mismatch_and_absent_release_row(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_row(113, "success", trace=True), _row(22, "collision", trace=True)]
        )
        + "\n"
    )
    report = _check(archive, traces, tmp_path, [113, 22])
    assert [row["comparison"] for row in report["comparisons"]] == [
        "no_release_row",
        "mismatch",
    ]


def test_rejects_parameter_drift(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["scenario_params"]["run_dt"] = 0.2
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="changes release scenario/planner parameters"):
        _check(archive, traces, tmp_path, [113])


def test_rejects_missing_per_pedestrian_force(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(22, "collision", trace=True)
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["pedestrians"] = [
        {"position": [1.0, 1.0], "velocity": [0.0, 0.0]}
    ]
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"] = {
        "ammv": {"pedestrian_force_vectors": [None]}
    }
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="lacks pedestrian forces"):
        _check(archive, traces, tmp_path, [22])


def test_rejects_unpaired_scientific_parameter_drift(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(22, "collision", trace=True)
    trace["scenario_params"]["run_dt"] = 0.2
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="changes release scenario/planner parameters"):
        _check(archive, traces, tmp_path, [22])


def test_rejects_unbound_config_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(_row(113, "collision", trace=True)) + "\n")
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    configs["doorway"].write_text(configs["doorway"].read_text() + "# drift\n")
    with pytest.raises(ValueError, match="diagnostic config SHA-256 mismatch"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_cli_keeps_report_and_exits_nonzero_on_outcome_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"comparison_counts": {"mismatch": 1}, "comparisons": [{"comparison": "mismatch"}]}
    monkeypatch.setattr(checker, "check", lambda *_args, **_kwargs: report)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_issue_9671_trace_reexport.py",
            "--release-archive",
            "unused.tar.gz",
            "--traces",
            "unused.jsonl",
            "--headon-config",
            "unused-headon.yaml",
            "--doorway-config",
            "unused-doorway.yaml",
            "--headon-manifest",
            "unused-headon.json",
            "--doorway-manifest",
            "unused-doorway.json",
            "--output",
            str(output),
        ],
    )
    assert checker.main() == 2
    assert json.loads(output.read_text()) == report
