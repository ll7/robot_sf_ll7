"""Contract tests for the issue #5263 exact-repeat campaign packet."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.exact_repeat_campaign import (
    HOST_REPORT_SCHEMA_VERSION,
    build_manifest,
    compare_verified_hosts,
    verify_host_report,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "benchmark"
    / "scenario_flakiness_issue_4978"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """Compile the retained #4978 data into the #5263 request."""
    episodes = [
        json.loads(line)
        for line in (FIXTURE_DIR / "real_campaign_episodes.jsonl").read_text().splitlines()
    ]
    return build_manifest(_load(FIXTURE_DIR / "real_campaign_flakiness_report.json"), episodes)


def _host_report(manifest: dict[str, Any], machine_id: str) -> dict[str, Any]:
    repeats = [{"outcome": 1, "trajectory_sha256": "a" * 64, "near_misses": 0} for _ in range(3)]
    return {
        "schema_version": HOST_REPORT_SCHEMA_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "environment": {
            "machine_id": machine_id,
            "cpu_only": True,
            "workers": 1,
            "numpy_version": "2.3.5",
            "numba_version": "0.65.1",
            "python_version": "3.13.14",
            "git_commit": manifest["targets"][0]["source_git_hash"],
        },
        "results": [
            {**target, "repeats": copy.deepcopy(repeats)} for target in manifest["targets"]
        ],
    }


def test_manifest_pins_all_seven_knife_edge_cells_and_their_420_runs(manifest):
    """The retained fixture compiles to the exact bounded campaign request."""
    assert len(manifest["cells"]) == 7
    assert len(manifest["targets"]) == 140
    assert manifest["execution_contract"] == {
        "cpu_only": True,
        "workers": 1,
        "repeats_per_target": 3,
        "trajectory_hash": "sha256",
        "required_runtime_versions": ["numpy_version", "numba_version"],
    }
    assert manifest["source"]["runnable_definitions_required"] == [
        "scenario_params",
        "planner_config",
    ]


def test_host_verifier_requires_all_targets_and_reports_cell_verdicts(manifest):
    """A complete identical host report proves all seven cell verdicts."""
    verified = verify_host_report(manifest, _host_report(manifest, "host-a"))
    assert verified["summary"] == {
        "n_targets": 140,
        "n_cells": 7,
        "all_cells_bitwise_identical": True,
    }
    assert all(cell["exact_repeat_determinism"] is True for cell in verified["cells"])


def test_host_verifier_requires_the_first_trajectory_divergence(manifest):
    """A divergent digest is accepted only with its computed first difference."""
    report = _host_report(manifest, "host-a")
    result = report["results"][0]
    result["repeats"][1]["trajectory_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="computed first divergence"):
        verify_host_report(manifest, report)
    result["first_divergence"] = {
        "repeat_index": 1,
        "field": "trajectory_sha256",
        "expected": "a" * 64,
        "observed": "b" * 64,
    }
    verified = verify_host_report(manifest, report)
    assert verified["targets"][0]["bitwise_identical"] is False
    assert verified["cells"][0]["exact_repeat_determinism"] is False


def test_host_verifier_rejects_a_mismatched_commit_or_config(manifest):
    """A matching seed alone cannot stand in for the pinned execution identity."""
    report = _host_report(manifest, "host-a")
    report["environment"]["git_commit"] = "not-the-source-revision"
    with pytest.raises(ValueError, match="git_commit"):
        verify_host_report(manifest, report)
    report["environment"]["git_commit"] = manifest["targets"][0]["source_git_hash"]
    report["results"][0]["source_config_hash"] = "different"
    with pytest.raises(ValueError, match="source_config_hash"):
        verify_host_report(manifest, report)


def test_cross_host_comparison_requires_pinned_versions_and_exact_fingerprints(manifest):
    """Version or fingerprint mismatches remain explicit divergent evidence."""
    first = verify_host_report(manifest, _host_report(manifest, "host-a"))
    second = verify_host_report(manifest, _host_report(manifest, "host-b"))
    comparison = compare_verified_hosts(manifest, first, second)
    assert comparison["summary"]["all_cells_bitwise_identical"] is True

    second["environment"]["numba_version"] = "different"
    comparison = compare_verified_hosts(manifest, first, second)
    assert comparison["pinned_runtime_versions_match"] is False
    assert comparison["summary"]["all_cells_bitwise_identical"] is False


def test_manifest_fails_closed_when_a_source_seed_is_missing():
    """No retained seed may silently disappear from the requested repeat set."""
    report = _load(FIXTURE_DIR / "real_campaign_flakiness_report.json")
    episodes = [
        json.loads(line)
        for line in (FIXTURE_DIR / "real_campaign_episodes.jsonl").read_text().splitlines()
    ]
    knife_edge = next(cell for cell in report["cells"] if cell["knife_edge"])
    filtered = [
        row
        for row in episodes
        if not (
            row["scenario_id"] == knife_edge["scenario_id"]
            and row["algo"] == knife_edge["planner"]
            and row["seed"] == 111
        )
    ]
    with pytest.raises(ValueError, match="do not cover every seed"):
        build_manifest(report, filtered)
