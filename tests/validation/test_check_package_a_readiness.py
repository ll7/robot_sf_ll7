"""Tests for the Package A readiness checker (issue #3078).

Tests use synthetic on-disk fixtures and never execute a benchmark, submit
compute, or interpret planner ranks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "validation" / "check_package_a_readiness.py"
SHIPPED_MANIFEST = REPO_ROOT / "configs" / "benchmarks" / "issue_3078_package_a_readiness.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "validation"))
import check_package_a_readiness as checker  # noqa: E402


def _synthetic_manifest(root: Path, *, create_inputs: bool = True) -> Path:
    """Write self-contained manifest plus, optionally, all input files."""
    inputs = {
        "heldout_family_inputs": [
            "experiments/heldout_pilot.yaml",
            "configs/sets/train_pool.yaml",
            "configs/sets/eval_set.yaml",
        ],
        "seed_plan": [
            "scripts/tools/analyze_seed_sufficiency.py",
            "scripts/tools/seed_sufficiency_gate.py",
        ],
        "frozen_protocol": [
            "configs/suite.yaml",
            "scripts/tools/result_store.py",
        ],
        "command_contracts": [
            "scripts/tools/analyze_seed_sufficiency.py",
            "scripts/tools/run_camera_ready_benchmark.py",
        ],
    }
    if create_inputs:
        for rel_paths in inputs.values():
            for rel_path in rel_paths:
                target = root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# fixture\n", encoding="utf-8")

    manifest = {
        "schema_version": "package-a-readiness.v0.2",
        "package": {"id": "synthetic_package_a", "issue": 3078},
        "heldout_family_inputs": {
            "required_paths": inputs["heldout_family_inputs"],
            "leakage_audit_tool": "scripts/tools/analyze_seed_sufficiency.py",
        },
        "seed_plan": {
            "required_paths": inputs["seed_plan"],
            "required_metadata": ["mode", "min_seeds", "rank_metric"],
            "mode": "seed_sufficiency_then_rank_stability",
            "min_seeds": 10,
            "rank_metric": "snqi",
        },
        "frozen_protocol": {"required_paths": inputs["frozen_protocol"]},
        "command_contracts": {
            "contracts": [
                {
                    "id": "seed_surface",
                    "stage": "readiness_probe",
                    "command": "uv run python scripts/tools/analyze_seed_sufficiency.py --help",
                    "allowed_in_readiness_check": True,
                    "executes_benchmark_campaign": False,
                    "required_paths": ["scripts/tools/analyze_seed_sufficiency.py"],
                },
                {
                    "id": "future_campaign",
                    "stage": "future_campaign_execution",
                    "command": "uv run python scripts/tools/run_camera_ready_benchmark.py --config x",
                    "allowed_in_readiness_check": False,
                    "executes_benchmark_campaign": True,
                    "required_paths": ["scripts/tools/run_camera_ready_benchmark.py"],
                },
            ]
        },
        "outputs": {
            "local_root": "output/benchmarks/synthetic_package_a",
            "disposable": True,
        },
        "durable_evidence": {
            "plan": {
                "path": "docs/context/evidence/synthetic_package_a",
                "required_before_claim": True,
            }
        },
        "readiness_decision": {
            "benchmark_campaign_run": False,
            "compute_submit_authorized": False,
            "ranking_claim_promotion": False,
            "paper_claim_edits": False,
            "fallback_degraded_success_allowed": False,
            "result_classification_required": True,
        },
    }
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_ready_when_all_declared_inputs_exist(tmp_path: Path) -> None:
    """Synthetic fixture is ready when every declared prerequisite exists."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "ready"
    assert report.missing_paths == []
    assert report.issues == []
    assert {command["id"] for command in report.checked_commands} == {
        "seed_surface",
        "future_campaign",
    }


def test_fails_closed_on_missing_heldout_input(tmp_path: Path) -> None:
    """A missing held-out-family input makes verdict not_ready."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    (tmp_path / "configs" / "sets" / "eval_set.yaml").unlink()
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert "configs/sets/eval_set.yaml" in report.missing_paths


def test_fails_closed_on_missing_seed_metadata(tmp_path: Path) -> None:
    """A seed plan missing required metadata is not_ready even if files exist."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del data["seed_plan"]["rank_metric"]
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert any("rank_metric" in issue for issue in report.issues)


def test_fails_closed_when_campaign_command_allowed_in_readiness(tmp_path: Path) -> None:
    """Campaign execution commands cannot be marked runnable by this checker."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["command_contracts"]["contracts"][1]["allowed_in_readiness_check"] = True
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert any("cannot both execute benchmark campaign" in issue for issue in report.issues)


def test_rejects_unsafe_output_root(tmp_path: Path) -> None:
    """An output root outside output/ is flagged."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["outputs"]["local_root"] = "docs/benchmarks/leak"
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert any("output/" in issue for issue in report.issues)


def test_rejects_durable_evidence_under_output(tmp_path: Path) -> None:
    """Durable evidence plan cannot point into disposable output."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["durable_evidence"]["plan"]["path"] = "output/leaky_evidence"
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert any("durable_evidence.plan.path" in issue for issue in report.issues)


def test_rejects_claim_or_compute_enabled_decision(tmp_path: Path) -> None:
    """Readiness packet fails closed if compute or claim promotion is enabled."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["readiness_decision"]["compute_submit_authorized"] = True
    data["readiness_decision"]["ranking_claim_promotion"] = True
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert any("compute_submit_authorized" in issue for issue in report.issues)
    assert any("ranking_claim_promotion" in issue for issue in report.issues)


def test_manifest_error_on_missing_section(tmp_path: Path) -> None:
    """A manifest missing required sections raises ManifestError."""
    manifest_path = tmp_path / "bad.yaml"
    manifest_path.write_text(yaml.safe_dump({"package": {"id": "x"}}), encoding="utf-8")
    with pytest.raises(checker.ManifestError):
        checker.check_readiness(manifest_path, repo_root=tmp_path)


def test_cli_json_exit_code_not_ready(tmp_path: Path) -> None:
    """CLI returns exit 1 JSON status not_ready when prerequisites are missing."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=False)
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "--manifest", str(manifest_path), "--json"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_ready"
    assert payload["missing_paths"]


def test_shipped_manifest_is_ready_against_repo() -> None:
    """The shipped Package A manifest's declared inputs exist in the repo tree."""
    report = checker.check_readiness(SHIPPED_MANIFEST, repo_root=REPO_ROOT)
    assert report.status == "ready", {
        "missing_paths": report.missing_paths,
        "issues": report.issues,
    }
