"""Contract tests for rights-scoped software-candidate materialization."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "dev" / "software_candidate_manifest.py"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _fixture_source(path: Path, *, include_asset: bool = False) -> tuple[Path, str]:
    """Create a clean Git source fixture with a closed materialization policy."""
    path.mkdir()
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Candidate Test"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "candidate@example.invalid"],
        check=True,
    )
    (path / "source.txt").write_text("frozen source\n", encoding="utf-8")
    (path / "target.txt").write_text("symlink target\n", encoding="utf-8")
    (path / "inventory.yaml").write_text("schema_version: fixture\n", encoding="utf-8")
    (path / "secret").mkdir()
    (path / "secret" / "excluded.txt").write_text("excluded\n", encoding="utf-8")
    (path / "link.bin").symlink_to("target.txt")
    if include_asset:
        (path / "assets").mkdir()
        (path / "assets" / "logo.png").write_bytes(b"asset\n")

    policy = {
        "schema_version": "robot_sf.software_candidate_policy.v1",
        "package": {"name": "robot_sf", "version": "0.0.6"},
        "source_inventory_path": "inventory.yaml",
        "candidate_inventory_path": "generated/candidate-inventory.json",
        "metadata_path": "SOFTWARE_CANDIDATE.json",
        "include": ["**"],
        "exclude": ["secret/**"],
        "required": ["source.txt", "inventory.yaml"],
        "asset_rules": [],
    }
    (path / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    source_sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, source_sha


def _materialize(source: Path, source_sha: str, candidate: Path, report: Path) -> None:
    """Materialize one fixture candidate using its tracked policy."""
    _run(
        "materialize-source",
        "--repo-root",
        str(source),
        "--candidate-root",
        str(candidate),
        "--source-sha",
        source_sha,
        "--policy",
        str(source / "policy.json"),
        "--report",
        str(report),
    )


def _distributions(path: Path) -> Path:
    """Create the smallest valid Robot SF wheel and source distribution fixture."""
    path.mkdir()
    wheel = path / "robot_sf-0.0.6-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "robot_sf-0.0.6.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: robot_sf\nVersion: 0.0.6\n\n",
        )
        archive.writestr("robot_sf/__init__.py", "\n")
    sdist = path / "robot_sf-0.0.6.tar.gz"
    metadata = b"Metadata-Version: 2.4\nName: robot_sf\nVersion: 0.0.6\n\n"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("robot_sf-0.0.6/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    return path


def _raw_sbom(path: Path) -> Path:
    """Create a minimal CycloneDX export for bundle assembly."""
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "version": 1,
                "serialNumber": "urn:uuid:fixture",
                "metadata": {
                    "timestamp": "volatile",
                    "component": {"type": "library", "name": "robot-sf"},
                },
                "components": [],
                "dependencies": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_materialization_is_deterministic_and_reports_exclusions(tmp_path: Path) -> None:
    """The same reviewed tree must produce byte-identical candidate identities."""
    source, source_sha = _fixture_source(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = tmp_path / "first-report.json"
    second_report = tmp_path / "second-report.json"

    _materialize(source, source_sha, first, first_report)
    _materialize(source, source_sha, second, second_report)

    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    second_payload = json.loads(second_report.read_text(encoding="utf-8"))
    assert first_payload == second_payload
    assert first_payload["source_sha"] == source_sha
    assert first_payload["excluded_paths"] == ["secret/excluded.txt"]
    assert first_payload["excluded_non_regular_paths"] == ["link.bin"]
    assert (
        first_payload["candidate_commit_sha"]
        == subprocess.run(
            ["git", "-C", str(first), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    metadata = json.loads((first / "SOFTWARE_CANDIDATE.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (first / "generated" / "candidate-inventory.json").read_text(encoding="utf-8")
    )
    assert metadata["source"]["commit_sha"] == source_sha
    assert metadata["excluded_paths"] == ["secret/excluded.txt"]
    assert metadata["excluded_non_regular_paths"] == ["link.bin"]
    assert inventory["source_sha"] == source_sha
    assert {
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file() and ".git" not in path.parts
    } == {
        "source.txt",
        "target.txt",
        "inventory.yaml",
        "policy.json",
        "generated/candidate-inventory.json",
        "SOFTWARE_CANDIDATE.json",
    }


def test_assemble_and_verify_bind_materialization_identity(tmp_path: Path) -> None:
    """The bundle and provenance envelopes must carry the exact candidate identity."""
    source, source_sha = _fixture_source(tmp_path / "source")
    candidate = tmp_path / "candidate"
    report = tmp_path / "materialization.json"
    _materialize(source, source_sha, candidate, report)
    distributions = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    bundle = tmp_path / "bundle"
    args = [
        "assemble",
        "--repo-root",
        str(source),
        "--dist-dir",
        str(distributions),
        "--raw-sbom",
        str(raw_sbom),
        "--bundle-dir",
        str(bundle),
        "--source-sha",
        source_sha,
        "--candidate-source-root",
        str(candidate),
        "--materialization-report",
        str(report),
        "--repository",
        "ll7/robot_sf_ll7",
        "--workflow-run-id",
        "123456",
        "--workflow-run-attempt",
        "1",
    ]
    for validator in ("version-alignment", "metadata", "archive-license", "wheel-install"):
        args.extend(("--validated", validator))

    _run(*args)
    manifest = json.loads((bundle / "candidate-manifest.json").read_text(encoding="utf-8"))
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    expected = {
        key: report_payload[key]
        for key in (
            "candidate_commit_sha",
            "candidate_tree_sha",
            "policy_path",
            "policy_sha256",
            "source_inventory_path",
            "source_inventory_sha256",
            "candidate_inventory_path",
            "candidate_metadata_path",
        )
    }
    assert manifest["materialization"] == expected
    provenance = json.loads((bundle / "candidate-provenance.json").read_text(encoding="utf-8"))
    assert provenance["materialization"] == expected
    _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
    )


def test_materialization_rejects_unclassified_asset_paths(tmp_path: Path) -> None:
    """Asset-like members cannot enter a candidate without release-safe evidence."""
    source, source_sha = _fixture_source(tmp_path / "source", include_asset=True)
    result = _run(
        "materialize-source",
        "--repo-root",
        str(source),
        "--candidate-root",
        str(tmp_path / "candidate"),
        "--source-sha",
        source_sha,
        "--policy",
        str(source / "policy.json"),
        check=False,
    )

    assert result.returncode == 1
    assert "asset member is not covered" in result.stderr
    assert not (tmp_path / "candidate").exists()
