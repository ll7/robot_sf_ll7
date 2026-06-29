"""Tests for the issue #3079 package-B readiness preflight."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.adversarial_package_b_preflight import preflight_package_b_manifest
from scripts.tools import preflight_adversarial_package_b

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _base_manifest(repo_root: Path) -> Path:
    _write_file(repo_root / "configs/scenarios/templates/crossing_ttc.yaml")
    _write_file(repo_root / "configs/adversarial/crossing_ttc_space.yaml")
    _write_file(repo_root / "scripts/tools/compare_adversarial_samplers.py")
    manifest = repo_root / "configs/adversarial/issue_3079_package_b_budget_matched.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "adversarial-package-b-comparison.v1",
                "issue": 3079,
                "status": "diagnostic_local_nominal",
                "claim_scope": "not_paper_facing_benchmark_evidence",
                "runner": "scripts/tools/compare_adversarial_samplers.py",
                "base_config": {
                    "scenario_template": "configs/scenarios/templates/crossing_ttc.yaml",
                    "search_space": "configs/adversarial/crossing_ttc_space.yaml",
                    "policy": "goal",
                    "objective": "worst_case_snqi",
                },
                "budget_grid": [16, 32, 64],
                "repeated_seeds": [1101, 2202, 3303],
                "samplers": ["random", "coordinate", "optuna"],
                "reporting_contract": [
                    "first_failure_iteration",
                    "best_valid_objective",
                    "invalid_candidate_rate",
                    "replay_success_rate",
                    "certified_valid_failure_count",
                    "replayable_valid_failure_count",
                    "fallback_candidate_count",
                    "degraded_candidate_count",
                    "held_out_family_yield",
                ],
                "explicit_exclusions": {
                    "learned_failure_proposal_issue_2921": "stretch_out_of_scope",
                    "held_out_family_yield": "not_evaluated_narrow_archive_caveat",
                    "paper_facing_success_claims": "forbidden",
                },
                "example_command": (
                    "uv run python scripts/tools/compare_adversarial_samplers.py "
                    "--package-b-budget-grid --seed 1101 --seed 2202 --seed 3303 "
                    "--output-dir output/adversarial/issue_3079_package_b "
                    "--out-json output/adversarial/issue_3079_package_b/report.json"
                ),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_committed_package_b_manifest_preflights_without_running_benchmark() -> None:
    """The checked-in manifest has complete readiness metadata."""
    result = preflight_package_b_manifest()

    assert result.ready is True
    assert result.blocked is False
    assert result.metadata["budget_grid"] == [16, 32, 64]
    assert result.metadata["repeated_seeds"] == [1101, 2202, 3303]
    assert result.metadata["does_not_execute_benchmark"] is True
    assert result.metadata["does_not_submit_slurm_or_gpu"] is True


def test_preflight_fails_closed_for_missing_budget_seed_and_provenance(tmp_path: Path) -> None:
    """Missing package-B readiness metadata blocks before any benchmark run."""
    manifest = _base_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    del payload["budget_grid"]
    payload["repeated_seeds"] = []
    payload["explicit_exclusions"] = {}
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = preflight_package_b_manifest(manifest, repo_root=tmp_path)

    assert result.ready is False
    assert result.blocked is True
    assert result.checks["budget_grid"] is False
    assert result.checks["repeated_seeds"] is False
    assert any("budget_grid" in blocker for blocker in result.blockers)
    assert any("repeated_seeds" in blocker for blocker in result.blockers)
    assert any("paper_facing_success_claims" in blocker for blocker in result.blockers)


def test_preflight_blocks_example_command_seed_drift(tmp_path: Path) -> None:
    """The executable seed plan must match manifest repeated_seeds."""
    manifest = _base_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["example_command"] = (
        "uv run python scripts/tools/compare_adversarial_samplers.py "
        "--package-b-budget-grid --seed 1101 --seed 2202 --seed 4404 "
        "--output-dir output/adversarial/issue_3079_package_b "
        "--out-json output/adversarial/issue_3079_package_b/report.json"
    )
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = preflight_package_b_manifest(manifest, repo_root=tmp_path)

    assert result.ready is False
    assert result.blocked is True
    assert result.checks["example_command_repeated_seeds"] is False
    assert any("example_command --seed values" in blocker for blocker in result.blockers)


def test_preflight_accepts_equals_form_example_command_seed_args(tmp_path: Path) -> None:
    """The seed parser should mirror argparse's --seed=value spelling."""
    manifest = _base_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["example_command"] = (
        "uv run python scripts/tools/compare_adversarial_samplers.py "
        "--package-b-budget-grid --seed=1101 --seed=2202 --seed=3303 "
        "--output-dir output/adversarial/issue_3079_package_b "
        "--out-json output/adversarial/issue_3079_package_b/report.json"
    )
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = preflight_package_b_manifest(manifest, repo_root=tmp_path)

    assert result.ready is True
    assert result.checks["example_command_repeated_seeds"] is True
    assert result.metadata["example_command_repeated_seeds"] == [1101, 2202, 3303]


def test_preflight_blocks_example_command_without_package_b_budget_grid(tmp_path: Path) -> None:
    """The example command must opt into the fixed package-B budget grid."""
    manifest = _base_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["example_command"] = (
        "uv run python scripts/tools/compare_adversarial_samplers.py "
        "--budget 16 --budget 32 --budget 64 --seed 1101 --seed 2202 --seed 3303 "
        "--output-dir output/adversarial/issue_3079_package_b "
        "--out-json output/adversarial/issue_3079_package_b/report.json"
    )
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = preflight_package_b_manifest(manifest, repo_root=tmp_path)

    assert result.ready is False
    assert result.blocked is True
    assert result.checks["example_command_uses_package_b_grid"] is False
    assert any("--package-b-budget-grid" in blocker for blocker in result.blockers)


def test_preflight_warns_for_malformed_example_command_seed_args(tmp_path: Path) -> None:
    """Malformed command seed flags are warnings plus fail-closed blockers."""
    manifest = _base_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["example_command"] = (
        "uv run python scripts/tools/compare_adversarial_samplers.py "
        "--package-b-budget-grid --seed not-an-int --seed "
        "--output-dir output/adversarial/issue_3079_package_b "
        "--out-json output/adversarial/issue_3079_package_b/report.json --seed"
    )
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = preflight_package_b_manifest(manifest, repo_root=tmp_path)

    assert result.ready is False
    assert result.checks["example_command_repeated_seeds"] is False
    assert any("non-integer --seed value" in warning for warning in result.warnings)
    assert any("--seed without value" in warning for warning in result.warnings)


def test_preflight_raises_filenotfound_for_missing_manifest(tmp_path: Path) -> None:
    """Missing manifest path raises before producing cascading blockers."""
    manifest = tmp_path / "configs/adversarial/missing_package_b_manifest.yaml"

    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        preflight_package_b_manifest(manifest, repo_root=tmp_path)


def test_preflight_blocks_missing_inputs_and_output_paths(tmp_path: Path) -> None:
    """Missing local inputs or wrong output roots cannot pass readiness."""
    manifest = _base_manifest(tmp_path)
    (tmp_path / "configs/adversarial/crossing_ttc_space.yaml").unlink()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["example_command"] = (
        "uv run python scripts/tools/compare_adversarial_samplers.py "
        "--package-b-budget-grid --seed 1101 --seed 2202 --seed 3303 "
        "--output-dir /tmp/package_b --out-json /tmp/package_b/report.json"
    )
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = preflight_package_b_manifest(manifest, repo_root=tmp_path)

    assert result.ready is False
    assert result.checks["search_space_exists"] is False
    assert result.checks["output_dir_under_issue_path"] is False
    assert result.checks["out_json_under_issue_path"] is False


def test_package_b_preflight_cli_writes_report_and_returns_nonzero_on_blocker(
    tmp_path: Path,
) -> None:
    """CLI exits non-zero with a structured blocked report."""
    manifest = _base_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["samplers"] = ["random", "coordinate"]
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    report_path = tmp_path / "preflight.json"

    exit_code = preflight_adversarial_package_b.main(
        [
            "--manifest",
            str(manifest),
            "--repo-root",
            str(tmp_path),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["blocked"] is True
    assert report["checks"]["samplers"] is False
