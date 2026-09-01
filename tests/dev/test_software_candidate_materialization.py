"""Contract tests for rights-scoped software-candidate materialization."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

import scripts.dev.software_candidate_manifest as candidate_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "dev" / "software_candidate_manifest.py"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(HELPER), *args]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "candidate helper failed with exit code "
            f"{result.returncode}: {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


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


def _archive_extras(path: Path) -> set[str]:
    """Read the exact Provides-Extra values from one built archive."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            assert len(metadata_names) == 1
            metadata = archive.read(metadata_names[0]).decode("utf-8")
    else:
        with tarfile.open(path, "r:*") as archive:
            metadata_names = [
                member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")
            ]
            assert len(metadata_names) == 1
            stream = archive.extractfile(metadata_names[0])
            assert stream is not None
            metadata = stream.read().decode("utf-8")
    return {
        line.partition(":")[2].strip().lower()
        for line in metadata.splitlines()
        if line.startswith("Provides-Extra:")
    }


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


def test_materialization_rejects_non_root_or_altered_candidate_commit(
    tmp_path: Path,
) -> None:
    """A same-tree rebound cannot replace the fixed root commit identity."""
    source, source_sha = _fixture_source(tmp_path / "source")
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "report.json"
    _materialize(source, source_sha, candidate, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tree = report["candidate_tree_sha"]

    def commit_with(*extra: str, author: str = "Robot SF candidate") -> str:
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": author,
                "GIT_AUTHOR_EMAIL": "candidate@robot-sf.invalid",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
                "GIT_COMMITTER_NAME": "Robot SF candidate",
                "GIT_COMMITTER_EMAIL": "candidate@robot-sf.invalid",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
            }
        )
        return subprocess.run(
            ["git", "-C", str(candidate), "commit-tree", tree, *extra],
            input="Materialize Robot SF software candidate\n",
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()

    rebound = commit_with("-p", report["candidate_commit_sha"])
    report["candidate_commit_sha"] = rebound
    with pytest.raises(candidate_manifest.CandidateError, match="metadata or parent"):
        candidate_manifest._validate_candidate_commit_identity(candidate, report)

    altered = commit_with(author="Untrusted candidate")
    report["candidate_commit_sha"] = altered
    with pytest.raises(candidate_manifest.CandidateError, match="metadata or parent"):
        candidate_manifest._validate_candidate_commit_identity(candidate, report)


def test_real_materialized_candidate_build_has_only_supported_extras(tmp_path: Path) -> None:
    """The sanitized candidate archives retain ``all`` but never advertise ``rllib``."""
    source = REPO_ROOT
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate = tmp_path / "candidate"
    report = tmp_path / "materialization.json"
    _run(
        "materialize-source",
        "--repo-root",
        str(source),
        "--candidate-root",
        str(candidate),
        "--source-sha",
        source_sha,
        "--policy",
        str(source / "scripts/validation/software_candidate_policy.v1.json"),
        "--report",
        str(report),
    )
    policy = json.loads(
        (source / "scripts/validation/software_candidate_policy.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["include"].count("scripts/__init__.py") == 1
    assert policy["required"].count("scripts/__init__.py") == 1
    assert "scripts/**" not in policy["include"]
    rights_policy = json.loads(
        (source / "scripts/validation/software_release_rights_policy.v1.json").read_text(
            encoding="utf-8"
        )
    )
    rights_selection = rights_policy["source_selection"]
    assert rights_selection["allow_globs"].count("scripts/__init__.py") == 1
    assert rights_selection["required_paths"].count("scripts/__init__.py") == 1
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    source_marker = source / "scripts/__init__.py"
    candidate_marker = candidate / "scripts/__init__.py"
    marker_bytes = source_marker.read_bytes()
    assert candidate_marker.read_bytes() == marker_bytes
    marker_member = next(
        member for member in report_payload["members"] if member["path"] == "scripts/__init__.py"
    )
    assert marker_member["sha256"] == hashlib.sha256(marker_bytes).hexdigest()
    assert [
        member["path"]
        for member in report_payload["members"]
        if member["path"].startswith("scripts/")
    ] == [
        "scripts/__init__.py",
        "scripts/carla_bridge/diagnose_replay_semantics.py",
        "scripts/dev/check_version_alignment.py",
        "scripts/dev/software_candidate_manifest.py",
        "scripts/dev/software_candidate_manifest.v1.schema.json",
        "scripts/tools/__init__.py",
        "scripts/tools/check_asset_rights_inventory.py",
        "scripts/tools/check_dependency_license_inventory.py",
        "scripts/tools/check_distribution_licenses.py",
        "scripts/tools/manage_external_data.py",
        "scripts/tools/migrate_artifacts.py",
        "scripts/validation/asset_rights_inventory.v1.yaml",
        "scripts/validation/dependency_license_policy.v1.json",
        "scripts/validation/dependency_license_policy.v1.schema.json",
        "scripts/validation/dependency_license_profiles.v1.json",
        "scripts/validation/dependency_license_profiles.v1.schema.json",
        "scripts/validation/software_candidate_policy.v1.json",
        "scripts/validation/software_release_rights_policy.v1.json",
        "scripts/validation/software_release_rights_policy.v1.schema.json",
        "scripts/validation/software_rights_admission.v1.schema.json",
        "scripts/validation/software_sanitized_candidate.v1.schema.json",
        "scripts/validation/wheel_install_smoke.sh",
    ]
    source_pyproject = (source / "pyproject.toml").read_text(encoding="utf-8")
    candidate_pyproject = (candidate / "pyproject.toml").read_text(encoding="utf-8")
    assert "rllib = [" in source_pyproject
    assert "rllib = [" not in candidate_pyproject
    candidate_optional = tomllib.loads(candidate_pyproject)["project"]["optional-dependencies"]
    assert set(candidate_optional) == {
        "all",
        "viz",
        "maps",
        "benchmark",
        "training",
        "gpu",
        "recurrent",
        "progress",
        "analytics",
        "browser",
        "sacadrl",
        "socnav",
        "criticality",
    }

    build_root = tmp_path / "build-source"
    _run(
        "stage-build-source",
        "--repo-root",
        str(candidate),
        "--build-root",
        str(build_root),
        "--source-sha",
        json.loads(report.read_text(encoding="utf-8"))["candidate_commit_sha"],
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    build_environment = os.environ.copy()
    build_environment["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.0.6"
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist), "--quiet"],
        cwd=build_root,
        env=build_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = {
        "all",
        "viz",
        "maps",
        "benchmark",
        "training",
        "gpu",
        "recurrent",
        "progress",
        "analytics",
        "browser",
        "sacadrl",
        "socnav",
        "criticality",
    }
    archives = [*dist.glob("*.whl"), *dist.glob("*.tar.gz")]
    assert len(archives) == 2
    assert all(_archive_extras(archive) == expected for archive in archives)


def test_materialized_candidate_entrypoints_resist_hostile_regular_scripts_package(
    tmp_path: Path,
) -> None:
    """Both helper entrypoints stay candidate-local when a regular package shadows ``scripts``."""
    source = REPO_ROOT
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidate = tmp_path / "candidate"
    report = tmp_path / "materialization.json"
    _run(
        "materialize-source",
        "--repo-root",
        str(source),
        "--candidate-root",
        str(candidate),
        "--source-sha",
        source_sha,
        "--policy",
        str(source / "scripts/validation/software_candidate_policy.v1.json"),
        "--report",
        str(report),
    )

    hostile = tmp_path / "hostile-site-packages"
    hostile_scripts = hostile / "scripts"
    hostile_tools = hostile_scripts / "tools"
    hostile_tools.mkdir(parents=True)
    scripts_sentinel = tmp_path / "hostile-scripts-imported"
    helper_sentinel = tmp_path / "hostile-helper-imported"
    (hostile_scripts / "__init__.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['HOSTILE_SCRIPTS_SENTINEL']).write_text(__file__, encoding='utf-8')\n",
        encoding="utf-8",
    )
    (hostile_tools / "__init__.py").write_text("\n", encoding="utf-8")
    (hostile_tools / "check_distribution_licenses.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['HOSTILE_HELPER_SENTINEL']).write_text(__file__, encoding='utf-8')\n"
        "raise RuntimeError('hostile distribution-rights helper imported')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(hostile)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HOSTILE_SCRIPTS_SENTINEL"] = str(scripts_sentinel)
    environment["HOSTILE_HELPER_SENTINEL"] = str(helper_sentinel)

    direct = subprocess.run(
        [sys.executable, str(candidate / "scripts/dev/software_candidate_manifest.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    module = subprocess.run(
        [sys.executable, "-m", "scripts.dev.software_candidate_manifest", "--help"],
        cwd=candidate,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert direct.returncode == 0, direct.stderr
    assert module.returncode == 0, module.stderr
    assert not scripts_sentinel.exists()
    assert not helper_sentinel.exists()

    path_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.tools.check_distribution_licenses as helper; print(helper.__file__)",
        ],
        cwd=candidate,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert path_probe.returncode == 0, path_probe.stderr
    assert Path(path_probe.stdout.strip()).resolve().is_relative_to(candidate.resolve())
    assert not scripts_sentinel.exists()
    assert not helper_sentinel.exists()


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
        "--expected-workflow-run-attempt",
        "1",
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
