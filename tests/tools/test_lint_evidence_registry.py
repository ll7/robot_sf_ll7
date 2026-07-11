"""Contract tests for the evidence-registry integrity linter (issue #5255)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINTER = ROOT / "scripts" / "tools" / "lint_evidence_registry.py"


def _load_linter():
    spec = importlib.util.spec_from_file_location("_evidence_registry_linter", LINTER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _make_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    evidence = repo / "docs" / "context" / "evidence"
    config = repo / "configs" / "campaign.yaml"
    artifact = evidence / "artifact.json"
    config.parent.mkdir(parents=True)
    evidence.mkdir(parents=True)
    config.write_text("seed: 7\n", encoding="utf-8")
    artifact.write_text('{"result": "ok"}\n', encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Evidence linter test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return (
        repo,
        evidence,
        _git(repo, "rev-parse", "HEAD"),
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )


def _write_entry(
    evidence: Path,
    *,
    campaign_id: str,
    commit: str,
    config_sha256: str,
    artifact_sha256: str,
    name: str,
) -> Path:
    entry = {
        "campaign_id": campaign_id,
        "config_path": "configs/campaign.yaml",
        "config_sha256": config_sha256,
        "commit": commit,
        "artifact_path": "docs/context/evidence/artifact.json",
        "sha256": artifact_sha256,
    }
    path = evidence / name
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def test_valid_registry_entry_has_no_findings(tmp_path: Path) -> None:
    """A campaign with committed config and matching artifact hash passes."""
    linter = _load_linter()
    repo, evidence, commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-valid",
        commit=commit,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="valid.json",
    )

    report = linter.lint_evidence_registry(repo, evidence)

    assert report["issues"] == []
    assert report["campaign_ids"] == ["campaign-valid"]


def test_dangling_commit_is_classified(tmp_path: Path) -> None:
    """A full-length but unknown commit is reported without stopping report mode."""
    linter = _load_linter()
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-dangling",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="dangling.json",
    )

    report = linter.lint_evidence_registry(repo, evidence)

    assert {issue["code"] for issue in report["issues"]} >= {"dangling_commit"}


def test_hash_mismatch_is_classified(tmp_path: Path) -> None:
    """A hash next to a committed artifact must match its current tracked bytes."""
    linter = _load_linter()
    repo, evidence, commit, config_sha256 = _make_repo(tmp_path)
    _write_entry(
        evidence,
        campaign_id="campaign-hash-mismatch",
        commit=commit,
        config_sha256=config_sha256,
        artifact_sha256="0" * 64,
        name="mismatch.json",
    )

    report = linter.lint_evidence_registry(repo, evidence)

    assert {issue["code"] for issue in report["issues"]} >= {"artifact_hash_mismatch"}


def test_config_hash_mismatch_is_classified(tmp_path: Path) -> None:
    """A declared producing-config hash must match the blob at the declared commit."""
    linter = _load_linter()
    repo, evidence, commit, _config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-config-hash-mismatch",
        commit=commit,
        config_sha256="0" * 64,
        artifact_sha256=artifact_sha256,
        name="config-mismatch.json",
    )

    report = linter.lint_evidence_registry(repo, evidence)

    assert {issue["code"] for issue in report["issues"]} >= {"config_sha256_mismatch"}


def test_duplicate_campaign_ids_are_classified(tmp_path: Path) -> None:
    """Campaign identifiers are collisions only when distinct bundles claim ownership."""
    linter = _load_linter()
    repo, evidence, commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    for bundle in ("one", "two"):
        bundle_evidence = evidence / bundle
        bundle_evidence.mkdir()
        _write_entry(
            bundle_evidence,
            campaign_id="campaign-duplicate",
            commit=commit,
            config_sha256=config_sha256,
            artifact_sha256=artifact_sha256,
            name="manifest.json",
        )

    report = linter.lint_evidence_registry(repo, evidence)

    duplicate_issues = [
        issue for issue in report["issues"] if issue["code"] == "duplicate_campaign_id"
    ]
    assert len(duplicate_issues) == 1
    assert "multiple evidence bundles" in duplicate_issues[0]["message"]


def test_campaign_metadata_is_aggregated_within_one_bundle(tmp_path: Path) -> None:
    """Child reports inherit campaign provenance from their bundle's manifest."""
    linter = _load_linter()
    repo, evidence, commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    bundle = evidence / "campaign-bundle"
    bundle.mkdir()
    _write_entry(
        bundle,
        campaign_id="campaign-bundle",
        commit=commit,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="campaign_manifest.json",
    )
    (bundle / "summary.json").write_text(
        json.dumps({"campaign_id": "campaign-bundle", "result": "diagnostic"}),
        encoding="utf-8",
    )

    report = linter.lint_evidence_registry(repo, evidence)

    codes = {issue["code"] for issue in report["issues"]}
    assert "duplicate_campaign_id" not in codes
    assert not {
        "missing_config_path",
        "missing_config_sha256",
        "missing_commit",
    }.intersection(codes)


def test_disposition_packet_classifies_without_suppressing_findings(tmp_path: Path) -> None:
    """A report-mode packet labels categories while leaving strict findings intact."""
    linter = _load_linter()
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-disposition",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="disposition.json",
    )
    disposition_path = repo / "dispositions.yaml"
    disposition_path.write_text(
        "\n".join(
            [
                "schema_version: evidence_registry_disposition.v1",
                "categories:",
                "  - code: dangling_commit",
                "    status: historical_commit_unavailable",
            ]
        ),
        encoding="utf-8",
    )

    report = linter.lint_evidence_registry(repo, evidence, disposition_path)

    assert any(issue["code"] == "dangling_commit" for issue in report["issues"])
    assert report["disposition_summary"] == {
        "by_status": {"historical_commit_unavailable": 1},
        "unclassified_by_code": {"config_missing_at_commit": 1},
    }


def test_reports_dir_filename_manifest_hash_is_verified(tmp_path: Path) -> None:
    """Nested filename hashes resolve against the declared reports directory."""
    linter = _load_linter()
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    report_artifact = evidence / "reports" / "artifact.json"
    report_artifact.parent.mkdir()
    report_artifact.write_text('{"report": true}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add report artifact")
    commit = _git(repo, "rev-parse", "HEAD")
    entry = {
        "campaign_id": "campaign-nested-artifact",
        "config_path": "configs/campaign.yaml",
        "config_sha256": config_sha256,
        "commit": commit,
        "reports_dir": "docs/context/evidence/reports",
        "files": {
            "artifact.json": {"sha256": hashlib.sha256(report_artifact.read_bytes()).hexdigest()}
        },
    }
    (evidence / "nested.json").write_text(json.dumps(entry), encoding="utf-8")

    report = linter.lint_evidence_registry(repo, evidence)

    assert report["issues"] == []


def test_report_mode_is_non_blocking_and_strict_mode_fails(tmp_path: Path) -> None:
    """Report mode preserves existing registry findings; strict mode is CI-ready."""
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-strict",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="strict.json",
    )
    command = [
        sys.executable,
        str(LINTER),
        "--repo-root",
        str(repo),
        "--registry-root",
        str(evidence),
    ]

    report_mode = subprocess.run(command, capture_output=True, text=True, check=False)
    strict_mode = subprocess.run(
        [*command, "--strict"], capture_output=True, text=True, check=False
    )

    assert report_mode.returncode == 0
    assert json.loads(report_mode.stdout)["summary"]["findings"] > 0
    assert strict_mode.returncode == 1


def test_exclude_codes_removes_findings_from_strict_gating(tmp_path: Path) -> None:
    """Excluded codes are removed from active findings and do not block strict mode."""
    linter = _load_linter()
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-exclude",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="exclude.json",
    )

    report = linter.lint_evidence_registry(
        repo, evidence, exclude_codes=frozenset({"dangling_commit"})
    )

    assert report["summary"]["findings"] > 0
    assert not any(issue["code"] == "dangling_commit" for issue in report["issues"])


def test_strict_mode_with_all_codes_excluded_passes(tmp_path: Path) -> None:
    """Strict mode passes when all finding codes are excluded."""
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-strict-exclude",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="strict-exclude.json",
    )
    command = [
        sys.executable,
        str(LINTER),
        "--repo-root",
        str(repo),
        "--registry-root",
        str(evidence),
        "--strict",
        "--exclude-codes",
        "dangling_commit,config_missing_at_commit",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["summary"]["findings"] == 0
    assert "excluded_summary" in report


def test_excluded_codes_appear_in_report(tmp_path: Path) -> None:
    """Excluded findings are recorded separately in the report for transparency."""
    linter = _load_linter()
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-excluded-report",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="excluded-report.json",
    )

    report = linter.lint_evidence_registry(
        repo, evidence, exclude_codes=frozenset({"dangling_commit"})
    )

    assert "excluded_codes" in report
    assert "excluded_issues" in report
    assert "excluded_summary" in report
    assert report["excluded_codes"] == ["dangling_commit"]
    assert report["excluded_summary"]["findings"] > 0
    assert "dangling_commit" in report["excluded_summary"]["by_code"]


def test_strict_exclusion_policy_file_is_loaded(tmp_path: Path) -> None:
    """A YAML policy file provides excluded codes for strict gating."""
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-policy",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="policy.json",
    )
    policy_path = repo / "strict_ci_policy.yaml"
    policy_path.write_text(
        "schema_version: evidence_registry_strict_ci_policy.v1\n"
        "excluded_codes:\n"
        "  - dangling_commit\n"
        "  - config_missing_at_commit\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(LINTER),
        "--repo-root",
        str(repo),
        "--registry-root",
        str(evidence),
        "--strict",
        "--strict-exclusion-policy",
        str(policy_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["summary"]["findings"] == 0


def test_strict_exclusion_policy_fails_closed_when_missing_or_malformed(tmp_path: Path) -> None:
    """A policy option never silently turns a strict check into a vacuous pass."""
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-invalid-policy",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="invalid-policy.json",
    )
    command = [
        sys.executable,
        str(LINTER),
        "--repo-root",
        str(repo),
        "--registry-root",
        str(evidence),
        "--strict",
        "--strict-exclusion-policy",
    ]
    missing = subprocess.run(
        [*command, str(repo / "missing.yaml")], capture_output=True, text=True, check=False
    )
    malformed = repo / "malformed.yaml"
    malformed.write_text("excluded_codes: dangling_commit\\n", encoding="utf-8")
    malformed_result = subprocess.run(
        [*command, str(malformed)], capture_output=True, text=True, check=False
    )

    assert missing.returncode != 0
    assert "Strict exclusion policy file not found" in missing.stderr
    assert malformed_result.returncode != 0
    assert "must contain an 'excluded_codes' list" in malformed_result.stderr


def test_exclude_codes_and_policy_merge(tmp_path: Path) -> None:
    """CLI --exclude-codes and --strict-exclusion-policy are merged."""
    repo, evidence, _commit, config_sha256 = _make_repo(tmp_path)
    artifact_sha256 = hashlib.sha256((evidence / "artifact.json").read_bytes()).hexdigest()
    _write_entry(
        evidence,
        campaign_id="campaign-merge",
        commit="f" * 40,
        config_sha256=config_sha256,
        artifact_sha256=artifact_sha256,
        name="merge.json",
    )
    policy_path = repo / "policy.yaml"
    policy_path.write_text("excluded_codes:\n  - dangling_commit\n", encoding="utf-8")
    command = [
        sys.executable,
        str(LINTER),
        "--repo-root",
        str(repo),
        "--registry-root",
        str(evidence),
        "--strict",
        "--exclude-codes",
        "config_missing_at_commit",
        "--strict-exclusion-policy",
        str(policy_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert set(report["excluded_codes"]) == {"dangling_commit", "config_missing_at_commit"}
