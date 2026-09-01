"""Tests for rejected software-candidate diagnostic preservation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev.software_promotion import PromotionError, _artifact_name, _load_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "dev" / "software_candidate_manifest.py"


def _source_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_source(tmp_path: Path) -> Path:
    source = tmp_path / "clean-source"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return source


def _run_helper(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_rejected_diagnostic_preserves_failed_dependency_report_and_bundle(tmp_path: Path) -> None:
    source_sha = _source_sha()
    source = _clean_source(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate-manifest.json").write_text(
        json.dumps({"source_sha": source_sha, "status": "built"}) + "\n", encoding="utf-8"
    )
    (bundle / "candidate-provenance.json").write_text("provenance bytes\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "robot_sf-0.0.6-py3-none-any.whl").write_bytes(b"wheel bytes")
    dependency_report = tmp_path / "dependency-license-inventory.json"
    dependency_report.write_text(
        json.dumps(
            {
                "summary": {"status": "blocked", "unresolved_count": 229},
                "repository_inputs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rights_log = tmp_path / "strict-distribution-rights.log"
    rights_log.write_text("PASS: strict rights gate\n", encoding="utf-8")
    dependency_log = tmp_path / "strict-supported-dependency-surface.log"
    dependency_log.write_text(
        "FAIL: dependency license inventory remains blocked for 229 row(s)\n", encoding="utf-8"
    )
    output = tmp_path / "rejected-diagnostic"

    result = _run_helper(
        "rejected-diagnostic",
        "--repo-root",
        str(source),
        "--output-dir",
        str(output),
        "--source-sha",
        source_sha,
        "--repository",
        "ll7/robot_sf_ll7",
        "--workflow-run-id",
        "33498526595",
        "--workflow-run-attempt",
        "1",
        "--candidate-version",
        "0.0.6",
        "--candidate-bundle",
        str(bundle),
        "--dist-dir",
        str(dist),
        "--dependency-report",
        str(dependency_report),
        "--strict-rights-log",
        str(rights_log),
        "--dependency-log",
        str(dependency_log),
        "--strict-rights-exit",
        "0",
        "--dependency-exit",
        "2",
        "--rights-admission-exit",
        "not-run",
    )
    assert result.returncode == 0, result.stderr

    metadata = json.loads((output / "rejected-diagnostic.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "robot_sf.software_candidate.rejected_diagnostic.v1"
    assert metadata["artifact_name"] == (
        f"robot-sf-software-candidate-rejected-{source_sha}-33498526595-1"
    )
    assert metadata["status"] == "rejected"
    assert metadata["evidence_status"] == "rejected"
    assert metadata["publishable"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["candidate"]["source_sha"] == source_sha
    assert metadata["candidate"]["workflow"] == {"run_id": "33498526595", "run_attempt": 1}
    assert metadata["blocker_count"] == 1
    blocker = metadata["blockers"][0]
    assert blocker["gate_id"] == "strict-supported-dependency-surface"
    assert blocker["exit_code"] == 2
    assert blocker["reported_blocker_count"] == 229
    assert "229" in blocker["reason"]
    payload_paths = {entry["path"] for entry in metadata["payload"]}
    assert "candidate-bundle/candidate-provenance.json" in payload_paths
    assert "reports/dependency-license-inventory.json" in payload_paths

    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksums
    for line in checksums:
        digest, relative = line.split("  ", maxsplit=1)
        assert digest == hashlib.sha256((output / relative).read_bytes()).hexdigest()


def test_rejected_diagnostic_requires_a_failed_strict_gate(tmp_path: Path) -> None:
    source = _clean_source(tmp_path)
    result = _run_helper(
        "rejected-diagnostic",
        "--repo-root",
        str(source),
        "--output-dir",
        str(tmp_path / "diagnostic"),
        "--source-sha",
        _source_sha(),
        "--repository",
        "ll7/robot_sf_ll7",
        "--workflow-run-id",
        "33498526595",
        "--workflow-run-attempt",
        "1",
        "--candidate-version",
        "0.0.6",
        "--strict-rights-exit",
        "0",
        "--dependency-exit",
        "not-run",
        "--rights-admission-exit",
        "0",
        check=False,
    )
    assert result.returncode == 1
    assert "at least one failed strict gate" in result.stderr


@pytest.mark.parametrize(
    ("gate", "status", "log_option", "log_name"),
    (
        ("strict-distribution-rights", "13", "--strict-rights-log", "strict-rights.log"),
        ("rights-admission", "7", "--rights-admission-log", "rights-admission.log"),
    ),
)
def test_rejected_diagnostic_preserves_non_dependency_gate_logs(
    tmp_path: Path,
    gate: str,
    status: str,
    log_option: str,
    log_name: str,
) -> None:
    source = _clean_source(tmp_path)
    source_sha = _source_sha()
    log = tmp_path / log_name
    log.write_text(f"FAIL: {gate} blocked the candidate\n", encoding="utf-8")
    output = tmp_path / "diagnostic"
    exit_values = {
        "--strict-rights-exit": "0",
        "--dependency-exit": "not-run",
        "--rights-admission-exit": "0",
    }
    exit_values[
        {
            "strict-distribution-rights": "--strict-rights-exit",
            "rights-admission": "--rights-admission-exit",
        }[gate]
    ] = status
    exit_args = [value for pair in exit_values.items() for value in pair]

    result = _run_helper(
        "rejected-diagnostic",
        "--repo-root",
        str(source),
        "--output-dir",
        str(output),
        "--source-sha",
        source_sha,
        "--repository",
        "ll7/robot_sf_ll7",
        "--workflow-run-id",
        "33498526595",
        "--workflow-run-attempt",
        "1",
        "--candidate-version",
        "0.0.6",
        log_option,
        str(log),
        *exit_args,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "rejected-diagnostic.json").read_text(encoding="utf-8"))
    assert metadata["blockers"][0]["gate_id"] == gate
    assert metadata["blockers"][0]["exit_code"] == int(status)
    assert metadata["blockers"][0]["report"] is None
    assert metadata["blockers"][0]["log"]["path"].startswith("reports/")
    assert f"{gate} blocked" in metadata["terminal"]["reason"]


def test_rejected_diagnostic_marker_is_not_a_candidate_bundle(tmp_path: Path) -> None:
    """The protected promotion verifier rejects a diagnostic marker before candidate loading."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "rejected-diagnostic.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_sf.software_candidate.rejected_diagnostic.v1",
                "evidence_status": "rejected",
                "publishable": False,
                "promotion_eligible": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PromotionError, match="rejected diagnostic"):
        _load_candidate(
            bundle,
            expected_source_sha="a" * 40,
            expected_workflow_run_id="33498526595",
            expected_workflow_run_attempt=1,
            expected_version="0.0.6",
        )


def test_rejected_diagnostic_artifact_name_is_not_a_candidate_identity() -> None:
    with pytest.raises(PromotionError, match="rejected diagnostic"):
        _artifact_name(
            "robot-sf-software-candidate-rejected-" + "a" * 40 + "-33498526595-1",
            kind="candidate",
        )
