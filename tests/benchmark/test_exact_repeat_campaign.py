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
    canonical_sha256,
    compare_verified_hosts,
    resolve_runnable_definitions,
    verify_host_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "benchmark"
    / "scenario_flakiness_issue_4978"
)
CAMPAIGN_CONFIG = (
    REPOSITORY_ROOT
    / "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"
)
EVIDENCE_DIR = REPOSITORY_ROOT / "docs/context/evidence/issue_5263_exact_repeat"


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
            "lockfile_sha256": "b" * 64,
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
    assert {target["source_observation_mode"] for target in manifest["targets"]} == {
        "goal_state",
        "socnav_state",
    }


def test_resolver_hash_matches_all_140_runnable_source_definitions(manifest):
    """The canonical source config recovers every target without inventing definitions."""
    resolved = resolve_runnable_definitions(manifest, CAMPAIGN_CONFIG)
    assert resolved["summary"] == {
        "n_targets": 140,
        "n_cells": 7,
        "all_source_config_hashes_match": True,
        "runnable_definitions_remaining": [],
    }
    assert all(
        target["computed_config_hash"] == target["source_config_hash"]
        for target in resolved["targets"]
    )
    assert len(resolved["scenario_definitions"]) == 80
    assert set(resolved["planner_definitions"]) == {"goal", "orca", "ppo"}
    assert resolved["planner_definitions"]["ppo"]["planner_config_path"].startswith("configs/")
    assert all(
        target["scenario_definition_id"] in resolved["scenario_definitions"]
        for target in resolved["targets"]
    )


def test_registered_definition_artifacts_match_fresh_resolution(manifest):
    """Committed evidence stays reproducible from the retained fixture and canonical config."""
    assert _load(EVIDENCE_DIR / "exact_repeat_manifest.json") == manifest
    assert _load(EVIDENCE_DIR / "resolved_definitions.json") == resolve_runnable_definitions(
        manifest, CAMPAIGN_CONFIG
    )


def test_resolver_fails_closed_on_any_unmatched_source_hash(manifest):
    """One stale or invented target definition invalidates the complete recovery bundle."""
    drifted = copy.deepcopy(manifest)
    drifted["targets"][0]["source_config_hash"] = "0" * 16
    without_hash = {key: value for key, value in drifted.items() if key != "manifest_sha256"}
    drifted["manifest_sha256"] = canonical_sha256(without_hash)
    with pytest.raises(ValueError, match="config hash mismatch"):
        resolve_runnable_definitions(drifted, CAMPAIGN_CONFIG)


def test_host_verifier_requires_all_targets_and_reports_cell_verdicts(manifest):
    """A complete identical host report proves all seven cell verdicts."""
    verified = verify_host_report(manifest, _host_report(manifest, "host-a"))
    assert verified["summary"] == {
        "n_targets": 140,
        "n_runnable_targets": 140,
        "n_unrunnable_targets": 0,
        "n_cells": 7,
        "n_runnable_cells": 7,
        "n_unrunnable_cells": 0,
        "all_cells_bitwise_identical": True,
    }
    assert all(cell["exact_repeat_determinism"] is True for cell in verified["cells"])


def test_all_unrunnable_host_reports_never_claim_determinism(manifest):
    """Empty runnable evidence is fail-closed in verification and comparison."""
    first_report = _host_report(manifest, "host-a")
    for result in first_report["results"]:
        result["disposition"] = "unrunnable_on_current_main"
        result["disposition_reason"] = "fixture disposition"
        result["repeats"] = []
    first_verified = verify_host_report(manifest, first_report)
    assert first_verified["summary"]["all_cells_bitwise_identical"] is False

    second_verified = copy.deepcopy(first_verified)
    second_verified["environment"]["machine_id"] = "host-b"
    comparison = compare_verified_hosts(manifest, first_verified, second_verified)
    assert comparison["summary"]["all_cells_bitwise_identical"] is False


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


def test_host_verifier_requires_a_lockfile_hash(manifest):
    """A host result without its dependency lockfile identity is rejected."""
    report = _host_report(manifest, "host-a")
    del report["environment"]["lockfile_sha256"]
    with pytest.raises(ValueError, match="lockfile_sha256"):
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
