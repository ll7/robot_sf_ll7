"""Tests for the Package A readiness checker (issue #3078).

These tests use synthetic on-disk fixtures so they never execute the benchmark, touch
Slurm, or interpret ranks. They prove the checker fails closed on missing prerequisites
and passes only when every declared input exists. One smoke test exercises the real
shipped manifest against the live repository tree.
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
    """Write a self-contained manifest plus (optionally) all its input files."""
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
    }
    if create_inputs:
        for rel_paths in inputs.values():
            for rel in rel_paths:
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# fixture\n", encoding="utf-8")

    manifest = {
        "schema_version": "package-a-readiness.v0.1",
        "package": {"id": "synthetic_package_a", "issue": 3078},
        "heldout_family_inputs": {"required_paths": inputs["heldout_family_inputs"]},
        "seed_plan": {
            "mode": "seed-sufficiency-gated",
            "min_seeds": 5,
            "rank_metric": "kendall_tau",
            "required_paths": inputs["seed_plan"],
            "required_metadata": ["mode", "min_seeds", "rank_metric"],
        },
        "frozen_protocol": {"required_paths": inputs["frozen_protocol"]},
        "outputs": {"local_root": "output/benchmarks/synthetic", "disposable": True},
    }
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_ready_when_all_inputs_present(tmp_path: Path) -> None:
    """A manifest whose declared inputs all exist reports ready."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "ready"
    assert report.missing_paths == []
    assert report.issues == []


def test_fails_closed_on_missing_input(tmp_path: Path) -> None:
    """A missing held-out-family input makes the verdict not_ready."""
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


def test_rejects_unsafe_output_root(tmp_path: Path) -> None:
    """An output root outside output/ or not disposable is flagged as an issue."""
    manifest_path = _synthetic_manifest(tmp_path, create_inputs=True)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["outputs"]["local_root"] = "docs/benchmarks/leak"
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = checker.check_readiness(manifest_path, repo_root=tmp_path)
    assert report.status == "not_ready"
    assert any("output/" in issue for issue in report.issues)


def test_manifest_error_on_missing_section(tmp_path: Path) -> None:
    """A manifest missing a required section raises ManifestError."""
    manifest_path = tmp_path / "bad.yaml"
    manifest_path.write_text(yaml.safe_dump({"package": {"id": "x"}}), encoding="utf-8")
    with pytest.raises(checker.ManifestError):
        checker.check_readiness(manifest_path, repo_root=tmp_path)


def test_cli_json_exit_code_not_ready(tmp_path: Path) -> None:
    """CLI returns exit 1 and JSON status not_ready when a prerequisite is missing."""
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
    """The shipped Package A manifest's declared inputs exist in the live repo tree."""
    report = checker.check_readiness(SHIPPED_MANIFEST, repo_root=REPO_ROOT)
    assert report.status == "ready", {
        "missing_paths": report.missing_paths,
        "issues": report.issues,
    }
