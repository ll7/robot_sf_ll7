"""Offline contract tests for the immutable software-candidate bundle."""

from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
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
    command = list(args)
    if command and command[0] == "verify" and "--expected-workflow-run-attempt" not in command:
        command.extend(("--expected-workflow-run-attempt", "1"))
    return subprocess.run(
        [sys.executable, str(HELPER), *command],
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


def _candidate_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _bundle_snapshot(bundle: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in bundle.iterdir()}


def _refresh_member(manifest: dict[str, object], *, kind: str, path: Path) -> None:
    members = manifest["members"]
    assert isinstance(members, list)
    member = next(item for item in members if item["kind"] == kind)
    member["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    member["size"] = path.stat().st_size


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
    provenance_member = next(
        member for member in manifest["members"] if member["kind"] == "provenance"
    )
    provenance = json.loads((first / provenance_member["filename"]).read_text(encoding="utf-8"))
    assert provenance["build"] == {
        "command": "cd $BUILD_SOURCE && uv build --out-dir $DIST_DIR",
        "count": 1,
        "source_role": "disposable-exact-commit",
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
        "--expected-workflow-run-attempt",
        "1",
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


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
@pytest.mark.parametrize("preexisting_bundle", (False, True))
def test_assemble_rejects_nonfinite_raw_sbom_without_admitting_output(
    tmp_path: Path,
    constant: str,
    preexisting_bundle: bool,
) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    raw_sbom.write_text(
        raw_sbom.read_text(encoding="utf-8").replace(
            '"components":',
            f'"poison_non_json_number": {constant}, "components":',
            1,
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    if preexisting_bundle:
        bundle.mkdir()
        (bundle / "sentinel.txt").write_text("do not overwrite\n", encoding="utf-8")
    before = _bundle_snapshot(bundle) if preexisting_bundle else None

    result = _run(
        *_assemble_args(source, source_sha, dist, raw_sbom, bundle),
        check=False,
    )

    assert result.returncode == 1
    assert "non-finite JSON constant" in result.stderr
    if preexisting_bundle:
        assert _bundle_snapshot(bundle) == before
    else:
        assert not bundle.exists()


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


def test_check_source_rejects_ignored_workspace_mutation(tmp_path: Path) -> None:
    source, _source_sha = _source_repo(tmp_path / "source")
    (source / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    subprocess.run(["git", "-C", source, "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "ignore fixture"], check=True)
    source_sha = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (source / "payload.ignored").write_text("ignored mutation\n", encoding="utf-8")

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "dirty or ambiguous" in result.stderr


def test_checkout_root_build_version_file_mutation_fails_source_gate(tmp_path: Path) -> None:
    source, _source_sha = _source_repo(tmp_path / "source")
    (source / "robot_sf").mkdir()
    (source / "robot_sf" / "__init__.py").write_text("", encoding="utf-8")
    (source / ".gitignore").write_text("robot_sf/_version.py\n", encoding="utf-8")
    subprocess.run(["git", "-C", source, "add", ".gitignore", "robot_sf"], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "ignore generated version"], check=True)
    source_sha = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "Path('robot_sf').mkdir(exist_ok=True); "
            "Path('robot_sf/_version.py').write_text('generated\\n')",
        ],
        check=True,
        cwd=source,
    )
    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "untracked or ignored entry: robot_sf/_version.py" in result.stderr


def test_stage_build_source_materializes_clean_exact_commit_outside_checkout(
    tmp_path: Path,
) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    build_root = tmp_path / "candidate" / "source"

    result = _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"PASS: staged disposable build source at {source_sha}" in result.stdout
    staged = _run(
        "check-source",
        "--repo-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
    )
    assert staged.returncode == 0, staged.stderr
    source_tree = subprocess.run(
        ["/usr/bin/git", "-C", source, "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_tree = subprocess.run(
        ["/usr/bin/git", "-C", build_root, "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert build_tree == source_tree


def test_stage_build_source_rejects_wrong_commit_before_materialization(tmp_path: Path) -> None:
    source, previous_sha = _source_repo(tmp_path / "source")
    (source / "source.txt").write_text("new source\n", encoding="utf-8")
    subprocess.run(["git", "-C", source, "commit", "-qam", "new head"], check=True)
    build_root = tmp_path / "candidate" / "source"

    result = _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        previous_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "source SHA drift" in result.stderr
    assert not build_root.exists()


def test_stage_build_source_rejects_preexisting_dirty_build_root(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    build_root = tmp_path / "candidate" / "source"
    build_root.mkdir(parents=True)
    sentinel = build_root / "do-not-overwrite.txt"
    sentinel.write_text("untrusted staged bytes\n", encoding="utf-8")

    result = _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "must not already exist" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "untrusted staged bytes\n"


def test_stage_build_source_rejects_build_root_inside_authoritative_checkout(
    tmp_path: Path,
) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    build_root = source / "generated-build-root"

    result = _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "must be outside the source repository" in result.stderr
    assert not build_root.exists()


def test_stage_build_source_rejects_symlink_path_escape(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    escape = tmp_path / "external-looking"
    escape.symlink_to(source, target_is_directory=True)
    build_root = escape / "generated-build-root"

    result = _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "cannot traverse a symlink" in result.stderr
    assert not (source / "generated-build-root").exists()


def test_external_build_mutation_leaves_authoritative_checkout_exact(tmp_path: Path) -> None:
    source, _source_sha = _source_repo(tmp_path / "source")
    (source / "robot_sf").mkdir()
    (source / "robot_sf" / "__init__.py").write_text("", encoding="utf-8")
    (source / ".gitignore").write_text("robot_sf/_version.py\n", encoding="utf-8")
    subprocess.run(["git", "-C", source, "add", ".gitignore", "robot_sf"], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "build fixture"], check=True)
    source_sha = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_root = tmp_path / "candidate" / "source"
    _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        source_sha,
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('robot_sf/_version.py').write_text('generated\\n')",
        ],
        check=True,
        cwd=build_root,
    )
    authoritative = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )
    disposable = _run(
        "check-source",
        "--repo-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert authoritative.returncode == 0, authoritative.stderr
    assert not (source / "robot_sf" / "_version.py").exists()
    assert disposable.returncode == 1
    assert "untracked or ignored entry: robot_sf/_version.py" in disposable.stderr


def test_stage_build_source_ignores_path_git_repository_config_and_hooks(
    tmp_path: Path,
) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    fake_git_marker = tmp_path / "fake-git-ran"
    hook_marker = tmp_path / "source-hook-ran"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!/bin/sh\nprintf invoked > {fake_git_marker}\nexit 97\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    hooks = tmp_path / "source-hooks"
    hooks.mkdir()
    post_checkout = hooks / "post-checkout"
    post_checkout.write_text(
        f"#!/bin/sh\nprintf invoked > {hook_marker}\n",
        encoding="utf-8",
    )
    post_checkout.chmod(0o755)
    subprocess.run(
        ["git", "-C", source, "config", "core.hooksPath", str(hooks)],
        check=True,
    )
    build_root = tmp_path / "candidate" / "source"

    result = _run(
        "stage-build-source",
        "--repo-root",
        str(source),
        "--build-root",
        str(build_root),
        "--source-sha",
        source_sha,
        check=False,
        env={**os.environ, "PATH": str(fake_bin)},
    )

    assert result.returncode == 0, result.stderr
    assert not fake_git_marker.exists()
    assert not hook_marker.exists()


def test_check_source_rejects_mode_drift_hidden_by_repository_config(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    subprocess.run(["git", "-C", source, "config", "core.filemode", "false"], check=True)
    (source / "source.txt").chmod(0o755)

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "dirty or ambiguous" in result.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("content", "content or symlink-target drift"), ("removal", "missing tracked entry")),
)
def test_check_source_rejects_tracked_content_and_removal(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    if mutation == "content":
        (source / "source.txt").write_text("changed source\n", encoding="utf-8")
    else:
        (source / "source.txt").unlink()

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert message in result.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("target-drift", "content or symlink-target drift"), ("escape", "unsafe symlink target")),
)
def test_check_source_rejects_symlink_target_drift_and_escape(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source, _source_sha = _source_repo(tmp_path / "source")
    (source / "target-a.txt").write_text("a\n", encoding="utf-8")
    (source / "target-b.txt").write_text("b\n", encoding="utf-8")
    link = source / "linked.txt"
    link.symlink_to("target-a.txt" if mutation == "target-drift" else "../outside.txt")
    subprocess.run(["git", "-C", source, "add", "."], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "symlink fixture"], check=True)
    source_sha = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if mutation == "target-drift":
        link.unlink()
        link.symlink_to("target-b.txt")

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 1
    assert message in result.stderr


def test_check_source_does_not_execute_git_from_path(tmp_path: Path) -> None:
    source, source_sha = _source_repo(tmp_path / "source")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-git-ran"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\nprintf 'invoked\\n' > \"${FAKE_GIT_MARKER}\"\nexit 97\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
        env={
            **os.environ,
            "FAKE_GIT_MARKER": str(marker),
            "PATH": str(fake_bin),
        },
    )

    assert result.returncode == 0, result.stderr

    assert not marker.exists()


def test_check_source_rejects_non_commit_head_identity(tmp_path: Path) -> None:
    source, _source_sha = _source_repo(tmp_path / "source")
    tree_sha = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (source / ".git" / "HEAD").write_text(f"{tree_sha}\n", encoding="ascii")

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        tree_sha,
        check=False,
    )

    assert result.returncode == 1
    assert "not an exact commit" in result.stderr


def test_check_source_accepts_materialized_lfs_bytes_bound_by_pointer(tmp_path: Path) -> None:
    source, _source_sha = _source_repo(tmp_path / "source")
    payload = b"materialized immutable payload\n"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    lfs_path = source / "payload.bin"
    lfs_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{payload_sha256}\n"
        f"size {len(payload)}\n",
        encoding="ascii",
    )
    subprocess.run(["git", "-C", source, "add", "payload.bin"], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "LFS pointer fixture"], check=True)
    (source / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    subprocess.run(["git", "-C", source, "add", ".gitattributes"], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "LFS attributes fixture"], check=True)
    source_sha = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lfs_path.write_bytes(payload)

    result = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    lfs_path.write_bytes(payload + b"drift")
    drifted = _run(
        "check-source",
        "--repo-root",
        str(source),
        "--source-sha",
        source_sha,
        check=False,
    )
    assert drifted.returncode == 1
    assert "content or symlink-target drift" in drifted.stderr


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


def test_assemble_rejects_partial_materialization_identity(tmp_path: Path) -> None:
    """Assembly must not carry a candidate root without its materialization report."""
    source, source_sha = _source_repo(tmp_path / "source")
    dist = _distributions(tmp_path / "dist")
    raw_sbom = _raw_sbom(tmp_path / "raw-sbom.json")
    args = _assemble_args(source, source_sha, dist, raw_sbom, tmp_path / "bundle")
    args.extend(("--candidate-source-root", str(tmp_path / "candidate")))

    result = _run(*args, check=False)

    assert result.returncode == 1
    assert "must be supplied together" in result.stderr
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
        "--expected-workflow-run-attempt",
        "1",
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
        "--expected-workflow-run-attempt",
        "1",
        check=False,
    )

    assert result.returncode == 1
    assert "duplicate filenames" in result.stderr


def test_verify_rejects_nonfinite_candidate_manifest_without_mutating_bundle(
    tmp_path: Path,
) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workflow"]["run_attempt"] = float("nan")
    manifest_path.write_bytes(_candidate_json_bytes(manifest))
    before = _bundle_snapshot(bundle)

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--expected-workflow-run-attempt",
        "1",
        check=False,
    )

    assert result.returncode == 1
    assert "non-finite JSON constant" in result.stderr
    assert _bundle_snapshot(bundle) == before


def test_verify_rejects_nonfinite_normalised_sbom_without_mutating_bundle(
    tmp_path: Path,
) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sbom_member = next(member for member in manifest["members"] if member["kind"] == "sbom")
    sbom_path = bundle / sbom_member["filename"]
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    sbom["metadata"]["poison_non_json_number"] = float("nan")
    sbom_path.write_bytes(_candidate_json_bytes(sbom))
    _refresh_member(manifest, kind="sbom", path=sbom_path)

    provenance_member = next(
        member for member in manifest["members"] if member["kind"] == "provenance"
    )
    provenance_path = bundle / provenance_member["filename"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["sbom"] = next(member for member in manifest["members"] if member["kind"] == "sbom")
    provenance_path.write_bytes(_candidate_json_bytes(provenance))
    _refresh_member(manifest, kind="provenance", path=provenance_path)
    manifest_path.write_bytes(_candidate_json_bytes(manifest))
    before = _bundle_snapshot(bundle)

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--expected-workflow-run-attempt",
        "1",
        check=False,
    )

    assert result.returncode == 1
    assert "non-finite JSON constant" in result.stderr
    assert _bundle_snapshot(bundle) == before


def test_verify_rejects_nonfinite_provenance_without_mutating_bundle(
    tmp_path: Path,
) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_member = next(
        member for member in manifest["members"] if member["kind"] == "provenance"
    )
    provenance_path = bundle / provenance_member["filename"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["poison_non_json_number"] = float("nan")
    provenance_path.write_bytes(_candidate_json_bytes(provenance))
    _refresh_member(manifest, kind="provenance", path=provenance_path)
    manifest_path.write_bytes(_candidate_json_bytes(manifest))
    before = _bundle_snapshot(bundle)

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--expected-workflow-run-attempt",
        "1",
        check=False,
    )

    assert result.returncode == 1
    assert "non-finite JSON constant" in result.stderr
    assert _bundle_snapshot(bundle) == before


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
        "--expected-workflow-run-attempt",
        "1",
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
        "--expected-workflow-run-attempt",
        "1",
        check=False,
    )
    attempt_drift = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--expected-workflow-run-attempt",
        "2",
        check=False,
    )

    assert source_drift.returncode == 1
    assert "candidate source drift" in source_drift.stderr
    assert run_drift.returncode == 1
    assert "candidate workflow-run drift" in run_drift.stderr
    assert attempt_drift.returncode == 1
    assert "candidate workflow-attempt drift" in attempt_drift.stderr


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
        "--expected-workflow-run-attempt",
        "1",
        "--schema",
        str(invalid_schema),
        check=False,
    )

    assert result.returncode == 1
    assert "draft 2020-12" in result.stderr


def test_schema_rejects_nonfinite_json_before_contract_validation(tmp_path: Path) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    schema_path = HELPER.with_name("software_candidate_manifest.v1.schema.json")
    poisoned_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    poisoned_schema["poison_non_json_number"] = float("nan")
    poisoned_path = tmp_path / "poisoned-schema.json"
    poisoned_path.write_bytes(_candidate_json_bytes(poisoned_schema))

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--expected-workflow-run-attempt",
        "1",
        "--schema",
        str(poisoned_path),
        check=False,
    )

    assert result.returncode == 1
    assert "non-finite JSON constant" in result.stderr


def test_verify_rejects_syntactically_valid_weakened_schema(tmp_path: Path) -> None:
    _source, source_sha, bundle = _assembled_candidate(tmp_path)
    schema_path = HELPER.with_name("software_candidate_manifest.v1.schema.json")
    poisoned_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    poisoned_schema["properties"]["members"] = {"type": "string"}
    poisoned_path = tmp_path / "poisoned-schema.json"
    poisoned_path.write_text(json.dumps(poisoned_schema) + "\n", encoding="utf-8")

    result = _run(
        "verify",
        "--bundle-dir",
        str(bundle),
        "--expected-source-sha",
        source_sha,
        "--expected-workflow-run-id",
        "123456",
        "--expected-workflow-run-attempt",
        "1",
        "--schema",
        str(poisoned_path),
        check=False,
    )

    assert result.returncode == 1
    assert "unreviewed contract drift" in result.stderr


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
        "--expected-workflow-run-attempt",
        "1",
        env={**os.environ, "PATH": ""},
    )

    assert "reused exact bytes" in result.stdout


def test_json_output_serialization_rejects_nonfinite_values() -> None:
    helper = runpy.run_path(str(HELPER), run_name="software_candidate_manifest_test")

    with pytest.raises(helper["CandidateError"], match="non-finite JSON value"):
        helper["_json_bytes"]({"poison_non_json_number": float("nan")})
