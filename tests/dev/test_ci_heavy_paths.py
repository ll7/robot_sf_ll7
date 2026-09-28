"""Path selection for PR-only heavy CI jobs."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.dev.check_ci_needs import REQUIRED_JOBS, evaluate_needs
from scripts.dev.ci_heavy_paths import LANES, main, select_lanes

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("path", ["pyproject.toml", "uv.lock", ".github/workflows/ci.yml"])
def test_shared_inputs_run_every_heavy_lane(path: str) -> None:
    assert all(select_lanes([path]).values())


def test_unrelated_script_keeps_specialized_jobs_off() -> None:
    selected = select_lanes(["scripts/dev/unrelated.py"])
    assert selected == {
        "compat_macos": True,
        "examples_smoke": False,
        "notebooks_smoke": False,
        "xdist_scratch_isolation": False,
    }


@pytest.mark.parametrize("lane", LANES)
def test_each_lane_has_a_relevant_path(lane: str) -> None:
    examples = {
        "compat_macos": "fast-pysf/pyproject.toml",
        "examples_smoke": "examples/demo.py",
        "notebooks_smoke": "notebooks/demo.ipynb",
        "xdist_scratch_isolation": "tests/test_runner_policy_timeout.py",
    }
    assert select_lanes([examples[lane]])[lane]


@pytest.mark.parametrize("files", [[], ["../outside"], ["/absolute"], ["a//b"]])
def test_incomplete_or_unsafe_inputs_fail_closed(files: list[str]) -> None:
    with pytest.raises(ValueError):
        select_lanes(files)


def test_rename_runs_all_heavy_lanes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = tmp_path / "changed.txt"
    statuses = tmp_path / "statuses.tsv"
    output = tmp_path / "output.txt"
    changed.write_text("docs/new.md\n", encoding="utf-8")
    statuses.write_text("renamed\tdocs/new.md\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_heavy_paths.py",
            "--changed-files",
            str(changed),
            "--files-status",
            str(statuses),
            "--github-output",
            str(output),
        ],
    )
    assert main() == 0
    assert all(f"{lane}=true" in output.read_text(encoding="utf-8") for lane in LANES)
    assert json.loads(
        output.read_text(encoding="utf-8").split("compat_os=", 1)[1].splitlines()[0]
    ) == [
        "ubuntu-latest",
        "macos-latest",
    ]


def test_aggregate_accepts_only_proven_pr_path_skips() -> None:
    results = dict.fromkeys(REQUIRED_JOBS, "success")
    results.update({"changed-coverage-gate": "success", "examples-smoke": "skipped"})
    assert evaluate_needs(results, "pull_request", lane_outputs={"examples_smoke": "false"}) == []
    assert evaluate_needs(results, "pull_request", lane_outputs={"examples_smoke": "true"}) == [
        "examples-smoke"
    ]
    assert evaluate_needs(results, "pull_request") == ["examples-smoke"]
    results["coverage-gate"] = "success"
    assert evaluate_needs(results, "merge_group", lane_outputs={"examples_smoke": "false"}) == [
        "examples-smoke"
    ]
