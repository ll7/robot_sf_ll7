"""Tests for the issue #3799 Package B post-readiness gap packet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.adversarial_package_b_gap_packet import build_package_b_gap_packet

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/tools/prepare_package_b_post_readiness_gap_packet.py"
SHIPPED_MANIFEST = REPO_ROOT / "configs/adversarial/issue_3079_package_b_budget_matched.yaml"


def _write_file(path: Path, text: str = "placeholder\n") -> None:
    """Write a fixture file and create parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_registry(repo_root: Path, *, include_manifest: bool = True) -> None:
    """Write the minimum research package registry fixture."""
    required = (
        ["configs/adversarial/issue_3079_package_b_budget_matched.yaml"] if include_manifest else []
    )
    _write_file(
        repo_root / "configs/research/research_package_registry_issue_3057.yaml",
        yaml.safe_dump(
            {
                "packages": [
                    {
                        "id": "package_b_adversarial",
                        "issue": 3079,
                        "required_artifacts": required,
                    }
                ]
            },
            sort_keys=False,
        ),
    )


def _write_complete_manifest(repo_root: Path) -> Path:
    """Write a self-contained complete Package B readiness fixture."""
    _write_registry(repo_root)
    _write_file(repo_root / "configs/scenarios/templates/crossing_ttc.yaml")
    _write_file(repo_root / "configs/adversarial/crossing_ttc_space.yaml")
    _write_file(
        repo_root / "scripts/tools/compare_adversarial_samplers.py",
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplerComparisonRow:
    first_failure_iteration: int | None
    best_valid_objective: float | None
    invalid_candidate_rate: float
    replay_success_rate: float | None
    certified_valid_failure_count: int
    replayable_valid_failure_count: int
    fallback_candidate_count: int
    degraded_candidate_count: int
    held_out_family_yield: float | None
""",
    )
    manifest = repo_root / "configs/adversarial/issue_3079_package_b_budget_matched.yaml"
    _write_file(
        manifest,
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
                "output_artifacts": {
                    "output_dir": "output/adversarial/issue_3079_package_b",
                    "report_json": "output/adversarial/issue_3079_package_b/report.json",
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
    )
    return manifest


def test_shipped_manifest_builds_ready_gap_packet() -> None:
    """The checked-in Package B readiness artifacts map to a local handoff packet."""
    packet = build_package_b_gap_packet(SHIPPED_MANIFEST, repo_root=REPO_ROOT).to_payload()

    assert packet["schema_version"] == "package-b-post-readiness-gap-packet.v1"
    assert packet["status"] == "ready_for_local_dry_run_handoff"
    assert packet["readiness_ready"] is True
    assert "no full benchmark campaign run" in packet["out_of_scope"]
    assert "no Slurm/GPU submission" in packet["out_of_scope"]
    assert "no paper/dissertation claim edits" in packet["out_of_scope"]
    assert any(
        output["path"] == "output/adversarial/issue_3079_package_b/report.json"
        for output in packet["expected_outputs"]
    )
    assert any(
        "prepare_package_b_post_readiness_gap_packet.py" in command
        for command in packet["validation_commands"]
    )
    assert packet["source_preflight"]["metadata"]["does_not_execute_benchmark"] is True


def test_gap_packet_fails_closed_when_readiness_metadata_is_incomplete(tmp_path: Path) -> None:
    """Incomplete readiness metadata blocks both local and Slurm campaign decisions."""
    manifest = _write_complete_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    del payload["output_artifacts"]
    payload["repeated_seeds"] = []
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    packet = build_package_b_gap_packet(manifest, repo_root=tmp_path).to_payload()

    assert packet["status"] == "blocked_on_readiness"
    assert packet["readiness_ready"] is False
    assert packet["source_preflight"]["blocked"] is True
    assert any("output_artifacts" in blocker for blocker in packet["blockers"])
    assert any("repeated_seeds" in blocker for blocker in packet["blockers"])
    assert "Repair the readiness blockers" in packet["safe_next_decision"]


def test_gap_packet_cli_emits_stable_dry_run_snapshot(tmp_path: Path) -> None:
    """The CLI writes and prints the same dry-run packet without launching a campaign."""
    manifest = _write_complete_manifest(tmp_path)
    output = tmp_path / "packet.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["safe_next_decision"].startswith("Run only the listed local validation")
    assert stdout_payload["expected_outputs"] == [
        {
            "artifact_class": "worktree_local_campaign_output",
            "durability": "not_durable_until_promoted_or_manifested",
            "path": "output/adversarial/issue_3079_package_b",
            "required_before": "local_or_slurm_campaign_execution",
        },
        {
            "artifact_class": "package_b_sampler_comparison_report",
            "durability": "not_durable_until_promoted_or_manifested",
            "path": "output/adversarial/issue_3079_package_b/report.json",
            "required_before": "evidence_review",
        },
        {
            "artifact_class": "dry_run_decision_packet",
            "durability": "disposable_unless_copied_to_reviewable_evidence",
            "path": "output/adversarial/issue_3079_package_b/post_readiness_gap_packet.json",
            "required_before": "handoff_to_campaign_operator",
        },
    ]
    assert all("sbatch" not in command for command in stdout_payload["validation_commands"])


def test_gap_packet_cli_returns_nonzero_for_incomplete_readiness(tmp_path: Path) -> None:
    """CLI exit code stays fail-closed when readiness preflight is blocked."""
    manifest = _write_complete_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["budget_grid"] = [16]
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "blocked_on_readiness"
    assert any("budget_grid" in blocker for blocker in payload["blockers"])


@pytest.mark.parametrize("forbidden", ["sbatch", "srun", "wandb"])
def test_gap_packet_validation_commands_stay_dry_run_only(forbidden: str) -> None:
    """The shipped packet validation commands do not include submission surfaces."""
    packet = build_package_b_gap_packet(SHIPPED_MANIFEST, repo_root=REPO_ROOT).to_payload()
    command_text = "\n".join(packet["validation_commands"]).lower()
    assert forbidden not in command_text
