"""Evidence checks for the frozen-release trace comparison."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING

import pytest

from scripts.validation.check_issue_9671_trace_reexport import SOURCE_SHA, check

if TYPE_CHECKING:
    from pathlib import Path


def _row(seed: int, status: str, *, trace: bool) -> dict:
    params = {"algo": "ppo", "run_horizon": 600, "run_dt": 0.1}
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
                "steps": [{"robot": {"position": [0.0, 0.0]}, "pedestrians": []}]
            }
        },
    }


def _archive(path: Path, rows: list[dict]) -> None:
    data = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    member = tarfile.TarInfo("bundle/payload/runs/ppo__differential_drive/episodes.jsonl")
    member.size = len(data)
    with tarfile.open(path, "w:gz") as bundle:
        bundle.addfile(member, io.BytesIO(data))


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
    report = check(
        archive,
        [traces],
        expected={("ppo", "classic_doorway_medium", 113), ("ppo", "classic_doorway_medium", 22)},
        expected_archive_sha256=None,
    )
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
        check(
            archive,
            [traces],
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
        )


def test_rejects_missing_per_pedestrian_force(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [])
    trace = _row(22, "collision", trace=True)
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["pedestrians"] = [
        {"position": [1.0, 1.0]}
    ]
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="lacks pedestrian forces"):
        check(
            archive,
            [traces],
            expected={("ppo", "classic_doorway_medium", 22)},
            expected_archive_sha256=None,
        )
