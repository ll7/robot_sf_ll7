"""Offline contract tests for the immutable software-candidate bundle."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "dev" / "software_candidate_manifest.py"
VALIDATORS = (
    "version-alignment",
    "metadata",
    "archive-license",
    "wheel-install",
)


def _run(
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def _source_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Candidate Test"], check=True)
    subprocess.run(
        ["git", "-C", path, "config", "user.email", "candidate@example.invalid"],
        check=True,
    )
    (path / "source.txt").write_text("frozen source\n", encoding="utf-8")
    subprocess.run(["git", "-C", path, "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", path, "commit", "-qm", "fixture"], check=True)
    sha = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
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


def _raw_sbom(path: Path, *, serial: str = "one", timestamp: str = "now") -> Path:
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "version": 1,
                "serialNumber": f"urn:uuid:{serial}",
                "metadata": {
                    "timestamp": timestamp,
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


def _assemble_args(
    source: Path,
    source_sha: str,
    dist: Path,
    raw_sbom: Path,
    bundle: Path,
) -> list[str]:
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
    return args


def _assembled_candidate(tmp_path: Path) -> tuple[Path, str, Path]:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    bundle = tmp_path / "bundle"
    _run(*_assemble_args(source, source_sha, dist, raw_sbom, bundle))
    return source, source_sha, bundle


def test_assemble_is_deterministic_and_offline_verify_reuses_exact_bytes(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    first = tmp_path / "first"
    second = tmp_path / "second"

    _run(*_assemble_args(source, source_sha, dist, raw_sbom, first))
    _run(*_assemble_args(source, source_sha, dist, raw_sbom, second))

    first_files = {path.name: path.read_bytes() for path in first.iterdir()}
    second_files = {path.name: path.read_bytes() for path in second.iterdir()}
    assert first_files == second_files

    manifest = json.loads((first / "candidate-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "robot_sf.software_candidate.v1"
    assert manifest["source_sha"] == source_sha
    assert manifest["package"] == {"name": "robot_sf", "version": "0.0.6"}
    assert manifest["validation"]["status"] == "passed"
    assert {member["kind"] for member in manifest["members"]} == {
        "wheel",
        "sdist",
        "sbom",
        "provenance",
    }

    sbom_member = next(member for member in manifest["members"] if member["kind"] == "sbom")
    sbom = json.loads((first / sbom_member["filename"]).read_text(encoding="utf-8"))
    assert "serialNumber" not in sbom
    assert "timestamp" not in sbom["metadata"]
    assert sbom["metadata"]["component"]["version"] == "0.0.6"

    result = _run(
        "verify",
        "--bundle-dir",
        str(first),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
    )
    assert "PASS" in result.stdout


def test_sbom_volatile_identity_is_removed_deterministically(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    first_raw = _raw_sbom(tmp_path / "first-raw.json", serial="first", timestamp="one")
    second_raw = _raw_sbom(tmp_path / "second-raw.json", serial="second", timestamp="two")
    first = tmp_path / "first"
    second = tmp_path / "second"

    _run(*_assemble_args(source, source_sha, dist, first_raw, first))
    _run(*_assemble_args(source, source_sha, dist, second_raw, second))

    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_assemble_classifies_only_the_exact_pinned_uv_out_dir_marker(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    marker = dist / ".gitignore"
    marker.write_bytes(b"*")

    _run(*_assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "accepted"))
    assert not (tmp_path / "accepted" / marker.name).exists()

    marker.write_bytes(b"*\nchanged\n")
    rejected = _run(
        *_assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "rejected"),
        check=False,
    )
    assert rejected.returncode == 1
    assert "pinned uv out-dir marker" in rejected.stderr


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing", "exactly one Robot SF wheel and one Robot SF sdist"),
        ("duplicate", "exactly one Robot SF wheel and one Robot SF sdist"),
        ("unclassified", "unclassified distribution members"),
    ),
)
def test_assemble_rejects_missing_duplicate_or_unclassified_distribution_members(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    if case == "missing":
        next(dist.glob("*.tar.gz")).unlink()
    elif case == "duplicate":
        shutil.copyfile(
            next(dist.glob("*.whl")),
            dist / "robot_sf-0.0.6-1-py3-none-any.whl",
        )
    else:
        (dist / "unexpected.txt").write_text("not a candidate member\n", encoding="utf-8")

    result = _run(
        *_assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "bundle"),
        check=False,
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert not (tmp_path / "bundle").exists()


def test_assemble_rejects_duplicate_archive_member_names(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    wheel = next(dist.glob("*.whl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(
                "robot_sf-0.0.6.dist-info/METADATA",
                "Metadata-Version: 2.4\nName: robot_sf\nVersion: 0.0.6\n\n",
            )

    result = _run(
        *_assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "bundle"),
        check=False,
    )

    assert result.returncode == 1
    assert "duplicate archive member names" in result.stderr


def test_assemble_rejects_dirty_source_and_fuzzy_or_drifted_identity(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    dirty = _run(
        *_assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "dirty-bundle"),
        check=False,
    )
    assert dirty.returncode == 1
    assert "dirty or ambiguous" in dirty.stderr

    (source / "untracked.txt").unlink()
    fuzzy_args = _assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "fuzzy-bundle")
    fuzzy_args[fuzzy_args.index("--source-sha") + 1] = source_sha[:12]
    fuzzy = _run(*fuzzy_args, check=False)
    assert fuzzy.returncode == 1
    assert "exact lowercase 40-hex" in fuzzy.stderr

    drift_args = _assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "drift-bundle")
    drift_args[drift_args.index("--source-sha") + 1] = "f" * 40
    drift = _run(*drift_args, check=False)
    assert drift.returncode == 1
    assert "source SHA drift" in drift.stderr


def test_assemble_requires_every_validator_once_in_canonical_order(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    args = _assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "bundle")
    del args[-2:]

    result = _run(*args, check=False)

    assert result.returncode == 1
    assert "supplied exactly once in canonical order" in result.stderr
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("case", ("missing", "unclassified", "hash"))
def test_verify_rejects_missing_unclassified_or_hash_drifted_bundle_members(
    tmp_path: Path,
    case: str,
) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    manifest = json.loads((bundle / "candidate-manifest.json").read_text(encoding="utf-8"))
    wheel = bundle / manifest["members"][0]["filename"]
    if case == "missing":
        wheel.unlink()
    elif case == "unclassified":
        (bundle / "unexpected.txt").write_text("unclassified\n", encoding="utf-8")
    else:
        with wheel.open("ab") as stream:
            stream.write(b"drift")

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        check=False,
    )

    assert result.returncode == 1
    if case == "hash":
        assert "candidate member drift" in result.stderr
    else:
        assert "candidate bundle membership drift" in result.stderr


def test_verify_rejects_duplicate_manifest_filenames(tmp_path: Path) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"][1]["filename"] = manifest["members"][0]["filename"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        check=False,
    )

    assert result.returncode == 1
    assert "duplicate filenames" in result.stderr


def test_verify_rejects_source_and_workflow_run_drift(tmp_path: Path) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    source_drift = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        "f" * 40,
        "--expected-workflow-run-id",
        "123456",
        check=False,
    )
    run_drift = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "999999",
        check=False,
    )

    assert source_drift.returncode == 1
    assert "candidate source drift" in source_drift.stderr
    assert run_drift.returncode == 1
    assert "candidate workflow-run drift" in run_drift.stderr


def test_schema_accepts_emitted_manifest_and_invalid_schema_fails_closed(tmp_path: Path) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    schema_path = HELPER.with_name("software_candidate_manifest.v1.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "candidate-manifest.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)

    invalid_schema = tmp_path / "invalid-schema.json"
    invalid_schema.write_text("{}\n", encoding="utf-8")
    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--schema",
        str(invalid_schema),
        check=False,
    )

    assert result.returncode == 1
    assert "draft 2020-12" in result.stderr


def test_verify_is_offline_and_has_no_build_tool_dependency(tmp_path: Path) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        env={**os.environ, "PATH": ""},
    )

    assert "reused exact bytes" in result.stdout
