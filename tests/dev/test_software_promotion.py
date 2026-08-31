"""Offline contract tests for protected software-package promotion."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_HELPER = REPO_ROOT / "scripts" / "dev" / "software_candidate_manifest.py"
PROMOTION_HELPER = REPO_ROOT / "scripts" / "dev" / "software_promotion.py"
VALIDATORS = (
    "version-alignment",
    "metadata",
    "archive-license",
    "wheel-install",
)


def _run_helper(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROMOTION_HELPER), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _source_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Promotion Test"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "promotion@example.invalid"],
        check=True,
    )
    (path / "source.txt").write_text("promotion fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, sha


def _distributions(path: Path, version: str = "0.0.6") -> Path:
    path.mkdir()
    wheel = path / f"robot_sf-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"robot_sf-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: robot_sf\nVersion: {version}\n\n",
        )
        archive.writestr("robot_sf/__init__.py", "")
    sdist = path / f"robot_sf-{version}.tar.gz"
    metadata = f"Metadata-Version: 2.4\nName: robot_sf\nVersion: {version}\n\n".encode()
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo(f"robot_sf-{version}/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    return path


def _raw_sbom(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "version": 1,
                "serialNumber": "urn:uuid:promotion-test",
                "metadata": {
                    "timestamp": "volatile",
                    "tools": [{"name": "uv", "version": "0.11.21"}],
                    "component": {
                        "type": "library",
                        "bom-ref": "robot-sf-1",
                        "name": "robot-sf",
                    },
                },
                "components": [],
                "dependencies": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _candidate(tmp_path: Path) -> tuple[Path, str, Path]:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    bundle = tmp_path / "bundle"
    args = [
        "assemble",
        "--repo-root",
        str(source),
        "--dist-dir",
        str(dist),
        "--raw-sbom",
        str(raw_sbom),
        "--bundle-dir",
        str(bundle),
        "--source-sha",
        source_sha,
        "--repository",
        "ll7/robot_sf_ll7",
        "--workflow-run-id",
        "123456",
        "--workflow-run-attempt",
        "1",
    ]
    for validator in VALIDATORS:
        args.extend(("--validated", validator))
    subprocess.run([sys.executable, str(CANDIDATE_HELPER), *args], check=True)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rights_dir = bundle.parent / "rights-admission-artifact"
    rights_dir.mkdir()
    rights_receipt = rights_dir / "rights-admission.json"
    rights_receipt.write_text(
        json.dumps(
            {
                "candidate": {
                    "artifact_digest": "sha256:" + "a" * 64,
                    "artifact_id": "987654",
                    "artifact_name": "robot-sf-software-candidate-" + source_sha + "-123456-1",
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "members": manifest["members"],
                    "package": manifest["package"],
                    "provenance_sha256": manifest["members"][3]["sha256"],
                    "sbom_sha256": manifest["members"][2]["sha256"],
                    "source_sha": source_sha,
                    "workflow_run_id": "123456",
                },
                "sanitized": {
                    "policy_id": "robot_sf.software_release_rights_policy.v1",
                    "policy_path": "scripts/validation/software_release_rights_policy.v1.json",
                    "policy_sha256": "b" * 64,
                    "schema_version": "robot_sf.software_sanitized_candidate.v1",
                    "source_sha": source_sha,
                    "tree_sha256": "c" * 64,
                },
                "schema_version": "robot_sf.software_rights_admission.v1",
                "status": "accepted",
                "strict_gate": {
                    "command": (
                        "python scripts/tools/check_distribution_licenses.py $DIST_DIR "
                        "--strict-asset-rights --repo-root $BUILD_SOURCE "
                        "--source-tree-ref $SOURCE_SHA"
                    ),
                    "findings": 0,
                    "id": "strict-distribution-rights",
                    "policy_sha256": "b" * 64,
                    "source_sha": source_sha,
                    "status": "passed",
                },
                "supported_dependency_gate": {
                    "candidate_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "candidate_tree_sha256": "c" * 64,
                    "command": (
                        "python scripts/tools/check_dependency_license_inventory.py "
                        "--repo-root $BUILD_SOURCE --output $DEPENDENCY_REPORT "
                        "--candidate-bundle $CANDIDATE_BUNDLE --fail-on-unresolved"
                    ),
                    "id": "strict-supported-dependency-surface",
                    "policy_path": "scripts/validation/dependency_license_policy.v1.json",
                    "policy_sha256": "d" * 64,
                    "profile_manifest_path": (
                        "scripts/validation/dependency_license_profiles.v1.json"
                    ),
                    "profile_manifest_sha256": "e" * 64,
                    "report_filename": "dependency-license-inventory.json",
                    "report_sha256": "f" * 64,
                    "schema_version": "robot-sf.dependency-license-inventory.v1",
                    "source_sha": source_sha,
                    "status": "passed",
                    "unresolved_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source, source_sha, bundle


def _candidate_args(source_sha: str, bundle: Path) -> list[str]:
    return [
        "--candidate-dir",
        str(bundle),
        "--rights-receipt",
        str(bundle.parent / "rights-admission-artifact" / "rights-admission.json"),
        "--source-sha",
        source_sha,
        "--candidate-run-id",
        "123456",
        "--candidate-artifact-id",
        "987654",
        "--candidate-artifact-name",
        "robot-sf-software-candidate-" + source_sha + "-123456-1",
        "--candidate-artifact-digest",
        "sha256:" + "a" * 64,
        "--version",
        "0.0.6",
    ]


def _write_upload_receipt(tmp_path: Path, source_sha: str, bundle: Path) -> Path:
    receipt = tmp_path / "testpypi-upload-receipt.json"
    result = _run_helper(
        "write-receipt",
        *_candidate_args(source_sha, bundle),
        "--channel",
        "testpypi",
        "--promotion-run-id",
        "222222",
        "--promotion-run-attempt",
        "1",
        "--receipt",
        str(receipt),
    )
    assert result.returncode == 0, result.stderr
    return receipt


def test_candidate_verification_rejects_wrong_artifact_identity(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    args = ["verify-candidate", *_candidate_args(source_sha, bundle)]
    accepted = _run_helper(*args)
    assert accepted.returncode == 0, accepted.stderr

    # The offline candidate contains no artifact ID; a malformed dispatch
    # identity is rejected before any upload and cannot create a receipt.
    invalid = _candidate_args(source_sha, bundle)
    invalid[invalid.index("--candidate-artifact-id") + 1] = "0"
    receipt = tmp_path / "wrong-receipt.json"
    rejected = _run_helper(
        "write-receipt",
        *invalid,
        "--channel",
        "testpypi",
        "--promotion-run-id",
        "222222",
        "--promotion-run-attempt",
        "1",
        "--receipt",
        str(receipt),
        check=False,
    )
    assert rejected.returncode == 1
    assert not receipt.exists()


def test_candidate_verification_rejects_missing_or_unresolved_rights_admission(
    tmp_path: Path,
) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    missing = _candidate_args(source_sha, bundle)
    missing[missing.index("--rights-receipt") + 1] = str(tmp_path / "missing-rights.json")
    rejected = _run_helper("verify-candidate", *missing, check=False)
    assert rejected.returncode == 1
    assert "rights admission receipt" in rejected.stderr

    receipt = bundle.parent / "rights-admission-artifact" / "rights-admission.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.pop("supported_dependency_gate")
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rejected = _run_helper("verify-candidate", *_candidate_args(source_sha, bundle), check=False)
    assert rejected.returncode == 1
    assert "missing or unclassified" in rejected.stderr

    payload["supported_dependency_gate"] = {
        "candidate_manifest_sha256": hashlib.sha256(
            (bundle / "candidate-manifest.json").read_bytes()
        ).hexdigest(),
        "candidate_tree_sha256": "c" * 64,
        "command": (
            "python scripts/tools/check_dependency_license_inventory.py "
            "--repo-root $BUILD_SOURCE --output $DEPENDENCY_REPORT "
            "--candidate-bundle $CANDIDATE_BUNDLE --fail-on-unresolved"
        ),
        "id": "strict-supported-dependency-surface",
        "policy_path": "scripts/validation/dependency_license_policy.v1.json",
        "policy_sha256": "d" * 64,
        "profile_manifest_path": "scripts/validation/dependency_license_profiles.v1.json",
        "profile_manifest_sha256": "e" * 64,
        "report_filename": "dependency-license-inventory.json",
        "report_sha256": "f" * 64,
        "schema_version": "robot-sf.dependency-license-inventory.v1",
        "source_sha": source_sha,
        "status": "passed",
        "unresolved_count": 0,
    }
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    payload["supported_dependency_gate"]["unresolved_count"] = 1
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rejected = _run_helper("verify-candidate", *_candidate_args(source_sha, bundle), check=False)
    assert rejected.returncode == 1
    assert "unresolved rows" in rejected.stderr
    payload["supported_dependency_gate"]["unresolved_count"] = 0
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    payload["strict_gate"]["findings"] = 1
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rejected = _run_helper("verify-candidate", *_candidate_args(source_sha, bundle), check=False)
    assert rejected.returncode == 1
    assert "unresolved findings" in rejected.stderr


def test_candidate_verification_rejects_forged_rights_binding(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    receipt = bundle.parent / "rights-admission-artifact" / "rights-admission.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["candidate"]["artifact_id"] = "987655"
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rejected = _run_helper("verify-candidate", *_candidate_args(source_sha, bundle), check=False)
    assert rejected.returncode == 1
    assert "different candidate" in rejected.stderr


def test_candidate_validation_roster_allows_strict_rights_upgrade(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_check = next(
        check for check in manifest["validation"]["checks"] if check["id"] == "archive-license"
    )
    archive_check["command"] = (
        "python scripts/tools/check_distribution_licenses.py $DIST_DIR "
        "--strict-asset-rights --repo-root $BUILD_SOURCE --source-tree-ref $SOURCE_SHA"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rights = json.loads(
        (bundle.parent / "rights-admission-artifact" / "rights-admission.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rights["candidate"]["manifest_sha256"] = manifest_sha256
    rights["supported_dependency_gate"]["candidate_manifest_sha256"] = manifest_sha256
    (bundle.parent / "rights-admission-artifact" / "rights-admission.json").write_text(
        json.dumps(rights, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    accepted = _run_helper("verify-candidate", *_candidate_args(source_sha, bundle))
    assert accepted.returncode == 0, accepted.stderr


def test_receipt_round_trip_is_exact_and_channel_replay_fails(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    receipt = _write_upload_receipt(tmp_path, source_sha, bundle)
    verified = _run_helper(
        "verify-receipt",
        *_candidate_args(source_sha, bundle),
        "--channel",
        "testpypi",
        "--receipt",
        str(receipt),
    )
    assert verified.returncode == 0, verified.stderr

    replay = _run_helper(
        "verify-receipt",
        *_candidate_args(source_sha, bundle),
        "--channel",
        "pypi",
        "--receipt",
        str(receipt),
        check=False,
    )
    assert replay.returncode == 1
    assert "channel" in replay.stderr


def test_receipt_rejects_member_hash_drift_and_version_collision(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    receipt = _write_upload_receipt(tmp_path, source_sha, bundle)
    collision = _run_helper(
        "write-receipt",
        *_candidate_args(source_sha, bundle),
        "--channel",
        "testpypi",
        "--promotion-run-id",
        "222222",
        "--promotion-run-attempt",
        "1",
        "--receipt",
        str(receipt),
        check=False,
    )
    assert collision.returncode == 1
    assert "overwrite" in collision.stderr

    wrong_version = _candidate_args(source_sha, bundle)
    wrong_version[wrong_version.index("--version") + 1] = "0.0.7"
    replay = _run_helper(
        "verify-receipt",
        *wrong_version,
        "--channel",
        "testpypi",
        "--receipt",
        str(receipt),
        check=False,
    )
    assert replay.returncode == 1
    assert "package identity" in replay.stderr

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["published"]["files"][0]["sha256"] = "f" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    rejected = _run_helper(
        "verify-receipt",
        *_candidate_args(source_sha, bundle),
        "--channel",
        "testpypi",
        "--receipt",
        str(receipt),
        check=False,
    )
    assert rejected.returncode == 1
    assert "published file hashes" in rejected.stderr


def test_cold_install_receipt_binds_test_receipt_wheel_and_report(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    test_receipt = _write_upload_receipt(tmp_path, source_sha, bundle)
    wheel = next(bundle.glob("*.whl"))
    downloaded = tmp_path / wheel.name
    downloaded.write_bytes(wheel.read_bytes())
    report = tmp_path / "wheel-install-smoke.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "source_checkout_import": False,
                "console_scripts_failed": 0,
            }
        ),
        encoding="utf-8",
    )
    cold = tmp_path / "testpypi-cold-install-receipt.json"
    written = _run_helper(
        "write-cold-install-receipt",
        *_candidate_args(source_sha, bundle),
        "--test-receipt",
        str(test_receipt),
        "--downloaded-wheel",
        str(downloaded),
        "--index-url",
        "https://test.pypi.org/simple",
        "--report",
        str(report),
        "--receipt",
        str(cold),
    )
    assert written.returncode == 0, written.stderr
    verified = _run_helper(
        "verify-cold-install",
        *_candidate_args(source_sha, bundle),
        "--test-receipt",
        str(test_receipt),
        "--report",
        str(report),
        "--receipt",
        str(cold),
    )
    assert verified.returncode == 0, verified.stderr

    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected = _run_helper(
        "verify-cold-install",
        *_candidate_args(source_sha, bundle),
        "--test-receipt",
        str(test_receipt),
        "--report",
        str(report),
        "--receipt",
        str(cold),
        check=False,
    )
    assert rejected.returncode == 1
    assert "report bytes" in rejected.stderr


def test_github_artifact_metadata_requires_exact_digest_and_run(tmp_path: Path) -> None:
    metadata = tmp_path / "artifact.json"
    metadata.write_text(
        json.dumps(
            {
                "id": 987654,
                "name": "robot-sf-software-candidate-" + "a" * 40 + "-123456-1",
                "digest": "sha256:" + "a" * 64,
                "expired": False,
                "archive_download_url": (
                    "https://api.github.com/repos/ll7/robot_sf_ll7/actions/artifacts/987654/zip"
                ),
                "workflow_run": {"id": 123456, "head_sha": "a" * 40},
            }
        ),
        encoding="utf-8",
    )
    accepted = _run_helper(
        "check-artifact",
        "--metadata",
        str(metadata),
        "--artifact-id",
        "987654",
        "--artifact-name",
        "robot-sf-software-candidate-" + "a" * 40 + "-123456-1",
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--run-id",
        "123456",
        "--kind",
        "candidate",
        "--source-sha",
        "a" * 40,
    )
    assert accepted.returncode == 0, accepted.stderr
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace("" + "a" * 64, "" + "b" * 64, 1),
        encoding="utf-8",
    )
    rejected = _run_helper(
        "check-artifact",
        "--metadata",
        str(metadata),
        "--artifact-id",
        "987654",
        "--artifact-name",
        "robot-sf-software-candidate-" + "a" * 40 + "-123456-1",
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--run-id",
        "123456",
        "--kind",
        "candidate",
        "--source-sha",
        "a" * 40,
        check=False,
    )
    assert rejected.returncode == 1
    assert "digest" in rejected.stderr


def test_rights_artifact_and_sanctioned_workflow_require_exact_run_and_digest(
    tmp_path: Path,
) -> None:
    source_sha = "a" * 40
    rights_name = "robot-sf-software-rights-admission-" + source_sha + "-123456-1"
    metadata = tmp_path / "rights-artifact.json"
    metadata.write_text(
        json.dumps(
            {
                "id": 987654,
                "name": rights_name,
                "digest": "sha256:" + "a" * 64,
                "expired": False,
                "archive_download_url": (
                    "https://api.github.com/repos/ll7/robot_sf_ll7/actions/artifacts/987654/zip"
                ),
                "workflow_run": {"id": 123456, "head_sha": source_sha},
            }
        ),
        encoding="utf-8",
    )
    accepted = _run_helper(
        "check-artifact",
        "--metadata",
        str(metadata),
        "--artifact-id",
        "987654",
        "--artifact-name",
        rights_name,
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--run-id",
        "123456",
        "--kind",
        "rights",
        "--source-sha",
        source_sha,
    )
    assert accepted.returncode == 0, accepted.stderr

    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace("sha256:" + "a" * 64, "sha256:" + "b" * 64),
        encoding="utf-8",
    )
    rejected = _run_helper(
        "check-artifact",
        "--metadata",
        str(metadata),
        "--artifact-id",
        "987654",
        "--artifact-name",
        rights_name,
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--run-id",
        "123456",
        "--kind",
        "rights",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "digest" in rejected.stderr

    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace('"id": 987654', '"id": 987655'),
        encoding="utf-8",
    )
    rejected = _run_helper(
        "check-artifact",
        "--metadata",
        str(metadata),
        "--artifact-id",
        "987654",
        "--artifact-name",
        rights_name,
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--run-id",
        "123456",
        "--kind",
        "rights",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "artifact ID" in rejected.stderr

    metadata.write_text(
        metadata.read_text(encoding="utf-8")
        .replace("sha256:" + "b" * 64, "sha256:" + "a" * 64)
        .replace('"id": 987655', '"id": 987654')
        .replace('"id": 123456', '"id": 123457'),
        encoding="utf-8",
    )
    rejected = _run_helper(
        "check-artifact",
        "--metadata",
        str(metadata),
        "--artifact-id",
        "987654",
        "--artifact-name",
        rights_name,
        "--artifact-digest",
        "sha256:" + "a" * 64,
        "--run-id",
        "123456",
        "--kind",
        "rights",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "workflow run" in rejected.stderr

    run_metadata = tmp_path / "rights-run.json"
    run_metadata.write_text(
        json.dumps(
            {
                "id": 123456,
                "path": ".github/workflows/software-candidate.yml",
                "head_sha": source_sha,
                "event": "workflow_call",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "workflow_id": 8101,
                "repository": {"full_name": "ll7/robot_sf_ll7"},
            }
        ),
        encoding="utf-8",
    )
    accepted = _run_helper(
        "check-rights-run",
        "--metadata",
        str(run_metadata),
        "--run-id",
        "123456",
        "--source-sha",
        source_sha,
    )
    assert accepted.returncode == 0, accepted.stderr
    run_payload = json.loads(run_metadata.read_text(encoding="utf-8"))
    run_payload["event"] = "workflow_dispatch"
    run_metadata.write_text(json.dumps(run_payload), encoding="utf-8")
    accepted = _run_helper(
        "check-rights-run",
        "--metadata",
        str(run_metadata),
        "--run-id",
        "123456",
        "--source-sha",
        source_sha,
    )
    assert accepted.returncode == 0, accepted.stderr
    run_payload["event"] = "push"
    run_metadata.write_text(json.dumps(run_payload), encoding="utf-8")
    rejected = _run_helper(
        "check-rights-run",
        "--metadata",
        str(run_metadata),
        "--run-id",
        "123456",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "not sanctioned" in rejected.stderr
    run_payload["event"] = "workflow_call"
    run_metadata.write_text(json.dumps(run_payload), encoding="utf-8")
    run_payload.pop("repository")
    run_metadata.write_text(json.dumps(run_payload), encoding="utf-8")
    rejected = _run_helper(
        "check-rights-run",
        "--metadata",
        str(run_metadata),
        "--run-id",
        "123456",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "repository" in rejected.stderr
    run_payload["repository"] = {"full_name": "ll7/robot_sf_ll7"}
    run_metadata.write_text(json.dumps(run_payload), encoding="utf-8")
    run_metadata.write_text(
        run_metadata.read_text(encoding="utf-8").replace(
            ".github/workflows/software-candidate.yml", ".github/workflows/ci.yml"
        ),
        encoding="utf-8",
    )
    rejected = _run_helper(
        "check-rights-run",
        "--metadata",
        str(run_metadata),
        "--run-id",
        "123456",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "sanctioned workflow" in rejected.stderr

    run_metadata.write_text(
        run_metadata.read_text(encoding="utf-8").replace('"id": 123456', '"id": 123457'),
        encoding="utf-8",
    )
    rejected = _run_helper(
        "check-rights-run",
        "--metadata",
        str(run_metadata),
        "--run-id",
        "123456",
        "--source-sha",
        source_sha,
        check=False,
    )
    assert rejected.returncode == 1
    assert "workflow-run ID" in rejected.stderr


def test_receipts_never_include_credential_fields(tmp_path: Path) -> None:
    _source, source_sha, bundle = _candidate(tmp_path)
    receipt = _write_upload_receipt(tmp_path, source_sha, bundle)
    text = receipt.read_text(encoding="utf-8").lower()
    assert "password" not in text
    assert "token" not in text
    assert "secret" not in text
