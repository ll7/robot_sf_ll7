"""Tests for rejected software-candidate diagnostic preservation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev.software_candidate_manifest import CandidateError, _diagnostic_binding_payload
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
        json.dumps(
            {
                "source_sha": source_sha,
                "status": "built",
                "workflow": {"run_id": "33498526595", "run_attempt": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "candidate-provenance.json").write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "workflow": {"run_id": "33498526595", "run_attempt": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "robot_sf-0.0.6-py3-none-any.whl").write_bytes(b"wheel bytes")
    (dist / ".gitignore").write_text("ignored local files\n", encoding="utf-8")
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
    assert "dist/.gitignore" not in payload_paths
    assert metadata["embedded_provenance"]["candidate-manifest"]["status"] == "verified"
    assert metadata["embedded_provenance"]["candidate-provenance"]["status"] == "verified"
    assert metadata["embedded_provenance"]["dependency-report"]["status"] == "unverified"

    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksums
    checksum_paths = {line.split("  ", maxsplit=1)[1] for line in checksums}
    assert checksum_paths == payload_paths
    for line in checksums:
        digest, relative = line.split("  ", maxsplit=1)
        assert digest == hashlib.sha256((output / relative).read_bytes()).hexdigest()
    uploaded_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(output).parts)
    }
    assert uploaded_paths == payload_paths | {"SHA256SUMS", "rejected-diagnostic.json"}


@pytest.mark.parametrize(
    ("report_kind", "option", "payload_path", "binding_key", "exit_values", "raw"),
    (
        (
            "materialization",
            "--materialization-report",
            "reports/materialization.json",
            "materialization-report",
            ("13", "not-run", "not-run"),
            b'{"unterminated": ' + (b"x" * 2048),
        ),
        (
            "dependency",
            "--dependency-report",
            "reports/dependency-license-inventory.json",
            "dependency-report",
            ("not-run", "2", "not-run"),
            b'{"unterminated": ' + (b"x" * 2048),
        ),
        (
            "materialization",
            "--materialization-report",
            "reports/materialization.json",
            "materialization-report",
            ("13", "not-run", "not-run"),
            b'{"source_sha": "\xff"}',
        ),
        (
            "dependency",
            "--dependency-report",
            "reports/dependency-license-inventory.json",
            "dependency-report",
            ("not-run", "2", "not-run"),
            b'{"value": NaN}',
        ),
    ),
)
def test_rejected_diagnostic_preserves_malformed_optional_report_bytes(
    tmp_path: Path,
    report_kind: str,
    option: str,
    payload_path: str,
    binding_key: str,
    exit_values: tuple[str, str, str],
    raw: bytes,
) -> None:
    """Malformed optional reports remain forensic bytes, not assembly failures."""
    source_sha = _source_sha()
    source = _clean_source(tmp_path)
    report = tmp_path / f"malformed-{report_kind}.json"
    report.write_bytes(raw)
    output = tmp_path / "diagnostic"

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
        option,
        str(report),
        "--strict-rights-exit",
        exit_values[0],
        "--dependency-exit",
        exit_values[1],
        "--rights-admission-exit",
        exit_values[2],
    )
    assert result.returncode == 0, result.stderr

    metadata = json.loads((output / "rejected-diagnostic.json").read_text(encoding="utf-8"))
    binding = metadata["embedded_provenance"][binding_key]
    assert binding["status"] == "unverified"
    assert set(binding) == {"status", "reason"}
    parse_label = (
        "materialization report"
        if report_kind == "materialization"
        else "diagnostic dependency report"
    )
    assert binding["reason"] == f"{parse_label} failed strict UTF-8 JSON parsing"
    assert raw.decode("utf-8", errors="replace") not in binding["reason"]
    assert metadata["status"] == metadata["evidence_status"] == "rejected"
    assert metadata["promotion_eligible"] is False
    assert metadata["publishable"] is False
    preserved = output / payload_path
    assert preserved.read_bytes() == raw
    digest = hashlib.sha256(raw).hexdigest()
    payload_entry = next(entry for entry in metadata["payload"] if entry["path"] == payload_path)
    assert payload_entry == {"path": payload_path, "sha256": digest, "size": len(raw)}
    assert (output / "SHA256SUMS").read_text(encoding="ascii") == f"{digest}  {payload_path}\n"
    if report_kind == "dependency":
        assert metadata["blockers"][0]["reported_blocker_count"] == 1
    with pytest.raises(PromotionError, match="rejected diagnostic"):
        _load_candidate(
            output,
            expected_source_sha=source_sha,
            expected_workflow_run_id="33498526595",
            expected_workflow_run_attempt=1,
            expected_version="0.0.6",
        )


@pytest.mark.parametrize("invalid_attempt", (True, 1.0))
@pytest.mark.parametrize("report_kind", ("materialization", "dependency"))
def test_rejected_diagnostic_rejects_type_confused_optional_report_attempt(
    tmp_path: Path, invalid_attempt: object, report_kind: str
) -> None:
    """Optional reports with ambiguous workflow attempts cannot bind as verified."""
    source_sha = _source_sha()
    source = _clean_source(tmp_path)
    report = tmp_path / f"{report_kind}.json"
    binding = {
        "source_sha": source_sha,
        "workflow": {"run_id": "33498526595", "run_attempt": invalid_attempt},
    }
    payload = binding if report_kind == "materialization" else {"candidate_binding": binding}
    report.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "diagnostic"
    option = (
        "--materialization-report" if report_kind == "materialization" else "--dependency-report"
    )
    exit_values = (
        ("13", "not-run", "not-run")
        if report_kind == "materialization"
        else (
            "not-run",
            "2",
            "not-run",
        )
    )
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
        option,
        str(report),
        "--strict-rights-exit",
        exit_values[0],
        "--dependency-exit",
        exit_values[1],
        "--rights-admission-exit",
        exit_values[2],
        check=False,
    )
    assert result.returncode == 1
    assert "workflow run_attempt is invalid" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("invalid_attempt", ("1", 0, -1))
def test_diagnostic_binding_rejects_invalid_attempt(invalid_attempt: object) -> None:
    with pytest.raises(
        CandidateError, match="materialization report workflow run_attempt is invalid"
    ):
        _diagnostic_binding_payload(
            {
                "source_sha": "a" * 40,
                "workflow": {"run_id": "33498526595", "run_attempt": invalid_attempt},
            },
            label="materialization report",
            source_sha="a" * 40,
            workflow_run_id="33498526595",
            workflow_run_attempt=1,
            require_workflow=False,
        )


def test_diagnostic_binding_accepts_positive_attempt() -> None:
    assert _diagnostic_binding_payload(
        {
            "source_sha": "a" * 40,
            "workflow": {"run_id": "33498526595", "run_attempt": 1},
        },
        label="materialization report",
        source_sha="a" * 40,
        workflow_run_id="33498526595",
        workflow_run_attempt=1,
        require_workflow=False,
    ) == {
        "source_sha": "a" * 40,
        "workflow": {"run_id": "33498526595", "run_attempt": 1},
        "status": "verified",
    }


@pytest.mark.parametrize("report_kind", ("materialization", "dependency"))
def test_rejected_diagnostic_rejects_mismatched_optional_report_attempt(
    tmp_path: Path, report_kind: str
) -> None:
    """A correctly typed but different attempt remains a fatal identity mismatch."""
    source_sha = _source_sha()
    source = _clean_source(tmp_path)
    report = tmp_path / f"{report_kind}.json"
    binding = {
        "source_sha": source_sha,
        "workflow": {"run_id": "33498526595", "run_attempt": 2},
    }
    payload = binding if report_kind == "materialization" else {"candidate_binding": binding}
    report.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "diagnostic"
    option = (
        "--materialization-report" if report_kind == "materialization" else "--dependency-report"
    )
    exit_values = (
        ("13", "not-run", "not-run")
        if report_kind == "materialization"
        else ("not-run", "2", "not-run")
    )

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
        option,
        str(report),
        "--strict-rights-exit",
        exit_values[0],
        "--dependency-exit",
        exit_values[1],
        "--rights-admission-exit",
        exit_values[2],
        check=False,
    )
    assert result.returncode == 1
    assert "workflow identity does not match rejected diagnostic" in result.stderr
    assert not output.exists()


def test_rejected_diagnostic_classifies_matching_source_only_materialization(
    tmp_path: Path,
) -> None:
    source_sha = _source_sha()
    source = _clean_source(tmp_path)
    report = tmp_path / "materialization.json"
    report.write_text(json.dumps({"source_sha": source_sha}), encoding="utf-8")
    output = tmp_path / "diagnostic"
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
        "--materialization-report",
        str(report),
        "--strict-rights-exit",
        "13",
        "--dependency-exit",
        "not-run",
        "--rights-admission-exit",
        "not-run",
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "rejected-diagnostic.json").read_text(encoding="utf-8"))
    assert metadata["embedded_provenance"]["materialization-report"] == {
        "source_sha": source_sha,
        "status": "verified-source-only",
    }


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
    "binding",
    ("candidate-manifest-source", "candidate-manifest-workflow", "materialization-report"),
)
def test_rejected_diagnostic_rejects_mismatched_embedded_source_binding(
    tmp_path: Path, binding: str
) -> None:
    source = _clean_source(tmp_path)
    source_sha = _source_sha()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate-manifest.json").write_text(
        json.dumps(
            {
                "source_sha": ("f" * 40 if binding == "candidate-manifest-source" else source_sha),
                "workflow": {
                    "run_id": "99999999999"
                    if binding == "candidate-manifest-workflow"
                    else "33498526595",
                    "run_attempt": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "candidate-provenance.json").write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "workflow": {"run_id": "33498526595", "run_attempt": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    materialization = tmp_path / "materialization.json"
    materialization.write_text(
        json.dumps({"source_sha": "f" * 40 if binding == "materialization-report" else source_sha})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "diagnostic"
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
        "--materialization-report",
        str(materialization),
        "--strict-rights-exit",
        "0",
        "--dependency-exit",
        "2",
        "--rights-admission-exit",
        "not-run",
        check=False,
    )
    assert result.returncode == 1
    assert "does not match rejected diagnostic" in result.stderr
    assert not output.exists()


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
