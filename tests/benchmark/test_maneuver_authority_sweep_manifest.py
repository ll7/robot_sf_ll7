"""Tests for the issue #3213 maneuver-authority sweep manifest checker."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_sf.benchmark.maneuver_authority_sweep_manifest import (
    MANIFEST_STATUS_MALFORMED,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_READY,
    check_maneuver_authority_sweep_manifest,
)
from scripts.tools.check_maneuver_authority_sweep_manifest import main as cli_main

CANONICAL_MANIFEST = Path("configs/benchmarks/maneuver_authority_sweep_manifest_issue_3213.yaml")


def _load_manifest(repo_root: Path) -> dict:
    return yaml.safe_load((repo_root / CANONICAL_MANIFEST).read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_manifest_is_ready_and_declares_sweep_arm_metadata() -> None:
    """The committed issue #3213 manifest should be a ready preflight contract."""
    repo_root = Path.cwd()

    report = check_maneuver_authority_sweep_manifest(CANONICAL_MANIFEST, repo_root=repo_root)

    assert report["status"] == MANIFEST_STATUS_READY
    assert report["issue"] == 3213
    assert {arm["name"] for arm in report["arms"]} >= {
        "baseline",
        "high_angular",
        "dense_lattice",
        "nearfield_turn",
        "combined_max_authority",
    }
    dense_lattice = next(arm for arm in report["arms"] if arm["name"] == "dense_lattice")
    assert dense_lattice["action_lattice"]["candidate_speed_count"] == 7
    assert dense_lattice["turn_authority"]["heading_delta_count"] == 9
    assert dense_lattice["kinematic_adapter"]["command_space"] == "unicycle_vw"
    assert dense_lattice["expected_outputs"]["campaign_root"].startswith(
        "output/benchmarks/issue_3213_maneuver_authority"
    )


def test_missing_arm_config_fails_closed_with_path_diagnostic(tmp_path: Path) -> None:
    """Missing ordinary inputs should be diagnosed instead of treated as runnable."""
    repo_root = Path.cwd()
    payload = _load_manifest(repo_root)
    payload["arms"][0]["algo_config"] = "configs/algos/hardcase_authority/absent.yaml"
    manifest = _write_manifest(tmp_path, payload)

    report = check_maneuver_authority_sweep_manifest(manifest, repo_root=repo_root)

    assert report["status"] == MANIFEST_STATUS_MISSING
    assert report["diagnostics"][0]["arm"] == "baseline"
    assert report["diagnostics"][0]["code"] == "missing_algo_config"
    assert report["diagnostics"][0]["path"] == "configs/algos/hardcase_authority/absent.yaml"


def test_missing_benchmark_grid_fails_closed_without_config_mismatch(tmp_path: Path) -> None:
    """Unavailable grid inputs should remain ordinary missing inputs, not arm drift."""
    repo_root = Path.cwd()
    payload = _load_manifest(repo_root)
    payload["benchmark_grid"] = "configs/benchmarks/absent_authority_grid.yaml"
    manifest = _write_manifest(tmp_path, payload)

    report = check_maneuver_authority_sweep_manifest(manifest, repo_root=repo_root)

    assert report["status"] == MANIFEST_STATUS_MISSING
    assert {
        "code": "missing_benchmark_grid",
        "field": "benchmark_grid",
        "path": "configs/benchmarks/absent_authority_grid.yaml",
        "arm": None,
    } in report["diagnostics"]
    assert not any(
        diagnostic["code"] == "algo_config_mismatch" for diagnostic in report["diagnostics"]
    )


def test_arm_config_must_match_referenced_grid_variant(tmp_path: Path) -> None:
    """Sweep arm metadata should fail closed when it drifts from the benchmark grid."""
    repo_root = Path.cwd()
    payload = _load_manifest(repo_root)
    payload["arms"][1]["algo_config"] = payload["arms"][0]["algo_config"]
    manifest = _write_manifest(tmp_path, payload)

    report = check_maneuver_authority_sweep_manifest(manifest, repo_root=repo_root)

    assert report["status"] == MANIFEST_STATUS_MALFORMED
    assert {
        "arm": "high_angular",
        "code": "algo_config_mismatch",
        "grid_variant": "high_angular",
        "expected": (
            "configs/algos/hardcase_authority/prediction_planner_authority_high_angular.yaml"
        ),
        "actual": ("configs/algos/hardcase_authority/prediction_planner_authority_baseline.yaml"),
    } in report["diagnostics"]


def test_malformed_arm_metadata_fails_closed(tmp_path: Path) -> None:
    """Sweep arms without action/turn/adapter metadata should fail closed."""
    repo_root = Path.cwd()
    payload = _load_manifest(repo_root)
    payload["arms"][1].pop("kinematic_adapter")
    manifest = _write_manifest(tmp_path, payload)

    report = check_maneuver_authority_sweep_manifest(manifest, repo_root=repo_root)

    assert report["status"] == MANIFEST_STATUS_MALFORMED
    assert {
        "arm": "high_angular",
        "code": "missing_arm_metadata",
        "field": "kinematic_adapter",
    } in report["diagnostics"]


def test_cli_exits_nonzero_for_fail_closed_manifest(tmp_path: Path, capsys) -> None:
    """The command-line checker should gate malformed manifests with exit code 2."""
    repo_root = Path.cwd()
    payload = _load_manifest(repo_root)
    payload["arms"][0]["expected_outputs"]["campaign_root"] = "tmp/not-durable"
    manifest = _write_manifest(tmp_path, payload)

    exit_code = cli_main(["--manifest", str(manifest), "--repo-root", str(repo_root)])

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == MANIFEST_STATUS_MALFORMED
    assert any(
        diagnostic["code"] == "invalid_expected_output_root" for diagnostic in report["diagnostics"]
    )
