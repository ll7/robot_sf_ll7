#!/usr/bin/env python3
"""Assemble and verify a build-once, non-publishing Robot SF candidate bundle.

The helper never builds or publishes packages. ``assemble`` admits exactly one
wheel and one source distribution that were built elsewhere, normalises a uv
CycloneDX export, and binds every payload byte to deterministic provenance.
``verify`` is deliberately offline and only revalidates the admitted bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "robot_sf.software_candidate.v1"
PROVENANCE_VERSION = "robot_sf.software_candidate.provenance.v1"
SCHEMA_PATH = Path(__file__).with_name("software_candidate_manifest.v1.schema.json")
MANIFEST_NAME = "candidate-manifest.json"
PROVENANCE_NAME = "candidate-provenance.json"
SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
RUN_ID_PATTERN = re.compile(r"[1-9][0-9]*\Z")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+!_-]*\Z")
VALIDATION_COMMANDS = (
    (
        "version-alignment",
        "python scripts/dev/check_version_alignment.py",
    ),
    (
        "metadata",
        "twine check --strict $DIST_DIR/*.whl $DIST_DIR/*.tar.gz",
    ),
    (
        "archive-license",
        "python scripts/tools/check_distribution_licenses.py $DIST_DIR",
    ),
    (
        "wheel-install",
        "bash scripts/validation/wheel_install_smoke.sh $DIST_DIR/robot_sf-*.whl",
    ),
)
VALIDATOR_IDS = tuple(identifier for identifier, _command in VALIDATION_COMMANDS)
MEMBER_KINDS = ("wheel", "sdist", "sbom", "provenance")


class CandidateError(ValueError):
    """Raised when candidate admission or offline verification fails closed."""


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
    except CandidateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateError(f"cannot hash candidate member {path}: {exc}") from exc
    return digest.hexdigest()


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_member_name(name: str, *, archive: Path) -> str:
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        raise CandidateError(f"{archive.name}: unsafe archive member path: {name!r}")
    raw_parts = name.rstrip("/").split("/")
    if not raw_parts or any(
        not part
        or part in {".", ".."}
        or ":" in part
        or part.endswith((".", " "))
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in part)
        for part in raw_parts
    ):
        raise CandidateError(f"{archive.name}: unsafe archive member path: {name!r}")
    return PurePosixPath(*raw_parts).as_posix()


def _validate_unique_archive_members(archive: Path) -> None:
    try:
        if archive.suffix in {".whl", ".zip"}:
            with zipfile.ZipFile(archive) as source:
                infos = source.infolist()
                links = sorted(
                    info.filename for info in infos if stat.S_ISLNK(info.external_attr >> 16)
                )
                if links:
                    raise CandidateError(
                        f"{archive.name}: symbolic-link archive members are forbidden: "
                        f"{', '.join(links)}"
                    )
                names = [_safe_member_name(info.filename, archive=archive) for info in infos]
        else:
            with tarfile.open(archive, mode="r:*") as source:
                members = source.getmembers()
                non_regular = sorted(
                    member.name for member in members if not member.isfile() and not member.isdir()
                )
                if non_regular:
                    raise CandidateError(
                        f"{archive.name}: non-regular archive members are forbidden: "
                        f"{', '.join(non_regular)}"
                    )
                names = [_safe_member_name(member.name, archive=archive) for member in members]
    except CandidateError:
        raise
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise CandidateError(f"cannot inspect archive {archive}: {exc}") from exc

    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise CandidateError(
            f"{archive.name}: duplicate archive member names: {', '.join(duplicates)}"
        )


def _metadata_fields(raw: bytes, *, archive: Path) -> tuple[str, str]:
    message = BytesParser(policy=email_policy).parsebytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise CandidateError(f"{archive.name}: package metadata must contain Name and Version")
    if _normalise_distribution_name(str(name)) != "robot-sf":
        raise CandidateError(f"{archive.name}: package metadata is not for Robot SF: {name!r}")
    version_text = str(version)
    if not VERSION_PATTERN.fullmatch(version_text):
        raise CandidateError(
            f"{archive.name}: unsafe or ambiguous package version: {version_text!r}"
        )
    return str(name), version_text


def _wheel_metadata(archive: Path) -> tuple[str, str]:
    _validate_unique_archive_members(archive)
    try:
        with zipfile.ZipFile(archive) as source:
            matches = [
                info
                for info in source.infolist()
                if PurePosixPath(info.filename).name == "METADATA"
                and PurePosixPath(info.filename).parent.name.endswith(".dist-info")
            ]
            if len(matches) != 1:
                raise CandidateError(
                    f"{archive.name}: expected exactly one .dist-info/METADATA member; "
                    f"found {len(matches)}"
                )
            metadata = _metadata_fields(source.read(matches[0]), archive=archive)
    except CandidateError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise CandidateError(f"cannot read wheel metadata from {archive}: {exc}") from exc

    filename_parts = archive.name.removesuffix(".whl").split("-")
    if len(filename_parts) < 5 or filename_parts[0] != "robot_sf":
        raise CandidateError(f"unexpected Robot SF wheel filename: {archive.name}")
    if filename_parts[1] != metadata[1].replace("-", "_"):
        raise CandidateError(
            f"{archive.name}: filename version {filename_parts[1]!r} does not match "
            f"METADATA version {metadata[1]!r}"
        )
    return metadata


def _sdist_metadata(archive: Path) -> tuple[str, str]:
    _validate_unique_archive_members(archive)
    try:
        with tarfile.open(archive, mode="r:*") as source:
            matches = [
                member
                for member in source.getmembers()
                if member.isfile()
                and len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).name == "PKG-INFO"
            ]
            if len(matches) != 1:
                raise CandidateError(
                    f"{archive.name}: expected exactly one root PKG-INFO member; "
                    f"found {len(matches)}"
                )
            extracted = source.extractfile(matches[0])
            if extracted is None:
                raise CandidateError(f"{archive.name}: cannot read root PKG-INFO")
            metadata = _metadata_fields(extracted.read(), archive=archive)
    except CandidateError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise CandidateError(f"cannot read sdist metadata from {archive}: {exc}") from exc

    suffix = next((item for item in SDIST_SUFFIXES if archive.name.endswith(item)), None)
    if suffix is None:
        raise CandidateError(f"unsupported Robot SF sdist filename: {archive.name}")
    expected = f"robot_sf-{metadata[1]}{suffix}"
    if archive.name != expected:
        raise CandidateError(
            f"{archive.name}: filename does not match PKG-INFO version; expected {expected}"
        )
    return metadata


def _distribution_inputs(dist_dir: Path) -> tuple[Path, Path, str]:
    if not dist_dir.is_dir() or dist_dir.is_symlink():
        raise CandidateError(f"distribution input is not a real directory: {dist_dir}")
    entries = sorted(dist_dir.iterdir(), key=lambda path: path.name)
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"unclassified distribution member: {path.name}")
    wheels = [path for path in entries if path.suffix == ".whl"]
    sdists = [
        path
        for path in entries
        if any(path.name.endswith(suffix) for suffix in SDIST_SUFFIXES) and path.suffix != ".whl"
    ]
    classified = {*wheels, *sdists}
    unclassified = [path.name for path in entries if path not in classified]
    if unclassified:
        raise CandidateError(
            f"unclassified distribution members: {', '.join(sorted(unclassified))}"
        )
    if len(wheels) != 1 or len(sdists) != 1 or len(entries) != 2:
        raise CandidateError(
            "candidate requires exactly one Robot SF wheel and one Robot SF sdist "
            f"(found {len(wheels)} wheel(s), {len(sdists)} sdist(s), "
            f"{len(entries)} total member(s))"
        )

    wheel_name, wheel_version = _wheel_metadata(wheels[0])
    sdist_name, sdist_version = _sdist_metadata(sdists[0])
    if _normalise_distribution_name(wheel_name) != _normalise_distribution_name(sdist_name):
        raise CandidateError("wheel and sdist package names do not match")
    if wheel_version != sdist_version:
        raise CandidateError(
            f"wheel and sdist versions do not match: {wheel_version!r} != {sdist_version!r}"
        )
    return wheels[0], sdists[0], wheel_version


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise CandidateError(f"cannot execute git for source identity: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise CandidateError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _validate_source(repo_root: Path, expected_sha: str) -> None:
    if not SHA_PATTERN.fullmatch(expected_sha):
        raise CandidateError("source SHA must be one exact lowercase 40-hex commit identity")
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise CandidateError(f"source repository is not a real directory: {repo_root}")
    head = _run_git(repo_root, "rev-parse", "--verify", "HEAD")
    if head != expected_sha:
        raise CandidateError(f"source SHA drift: expected {expected_sha}, found {head}")
    status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise CandidateError(f"source repository is dirty or ambiguous:\n{status}")


def _require_external(path: Path, *, repo_root: Path, label: str) -> None:
    resolved = path.resolve(strict=False)
    root = repo_root.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise CandidateError(f"{label} must be outside the source repository: {path}")


def _normalised_sbom(raw_sbom: Path, version: str) -> bytes:
    payload = _load_json(raw_sbom, label="raw SBOM")
    if not isinstance(payload, dict):
        raise CandidateError("raw SBOM must be a JSON object")
    if payload.get("bomFormat") != "CycloneDX" or payload.get("specVersion") != "1.5":
        raise CandidateError("raw SBOM must be CycloneDX 1.5")
    if payload.get("version") != 1:
        raise CandidateError("raw SBOM must have document version 1")
    if not isinstance(payload.get("components"), list) or not isinstance(
        payload.get("dependencies"), list
    ):
        raise CandidateError("raw SBOM must contain components and dependencies arrays")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CandidateError("raw SBOM must contain metadata")
    component = metadata.get("component")
    if (
        not isinstance(component, dict)
        or _normalise_distribution_name(str(component.get("name", ""))) != "robot-sf"
    ):
        raise CandidateError("raw SBOM root component must be Robot SF")

    payload.pop("serialNumber", None)
    metadata.pop("timestamp", None)
    component["bom-ref"] = f"pkg:pypi/robot-sf@{version}"
    component["name"] = "robot-sf"
    component["purl"] = f"pkg:pypi/robot-sf@{version}"
    component["type"] = "library"
    component["version"] = version
    return _json_bytes(payload)


def _member(path: Path, kind: str) -> dict[str, Any]:
    return {
        "filename": path.name,
        "kind": kind,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _validation_payload() -> dict[str, Any]:
    return {
        "checks": [
            {"command": command, "id": identifier, "status": "passed"}
            for identifier, command in VALIDATION_COMMANDS
        ],
        "status": "passed",
    }


def _validate_schema_file(schema_path: Path) -> None:
    schema = _load_json(schema_path, label="candidate manifest schema")
    if not isinstance(schema, dict):
        raise CandidateError("candidate manifest schema must be a JSON object")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise CandidateError("candidate manifest schema must use JSON Schema draft 2020-12")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise CandidateError("candidate manifest schema must be a closed object schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise CandidateError("candidate manifest schema is missing properties")
    version_schema = properties.get("schema_version")
    if not isinstance(version_schema, dict) or version_schema.get("const") != SCHEMA_VERSION:
        raise CandidateError("candidate manifest schema has the wrong schema_version contract")
    required = schema.get("required")
    if not isinstance(required, list) or set(required) != {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }:
        raise CandidateError("candidate manifest schema has the wrong required fields")


def _validate_workflow_identity(workflow: Any) -> None:
    if not isinstance(workflow, dict) or set(workflow) != {"run_id", "run_attempt"}:
        raise CandidateError("candidate manifest workflow identity is invalid")
    run_id = workflow["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise CandidateError("candidate manifest workflow run_id is invalid")
    attempt = workflow["run_attempt"]
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise CandidateError("candidate manifest workflow run_attempt is invalid")


def _validate_package_identity(package: Any) -> str:
    if not isinstance(package, dict) or set(package) != {"name", "version"}:
        raise CandidateError("candidate manifest package identity is invalid")
    version = package["version"]
    if package["name"] != "robot_sf" or not isinstance(version, str):
        raise CandidateError("candidate manifest package must identify robot_sf")
    if not VERSION_PATTERN.fullmatch(version):
        raise CandidateError("candidate manifest package version is invalid")
    return version


def _validated_member(member: Any) -> dict[str, Any]:
    if not isinstance(member, dict) or set(member) != {"filename", "kind", "sha256", "size"}:
        raise CandidateError("candidate manifest member record is invalid")
    filename = member["filename"]
    if (
        not isinstance(filename, str)
        or not filename
        or filename != Path(filename).name
        or filename in {".", "..", MANIFEST_NAME}
    ):
        raise CandidateError("candidate manifest member filename is unsafe or reserved")
    digest = member["sha256"]
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise CandidateError(f"candidate member {filename} has an invalid SHA-256")
    size = member["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise CandidateError(f"candidate member {filename} has an invalid size")
    return member


def _validate_members(members: Any, *, version: str) -> None:
    if not isinstance(members, list) or len(members) != len(MEMBER_KINDS):
        raise CandidateError("candidate manifest must bind exactly four payload members")
    validated = [_validated_member(member) for member in members]
    if [member["kind"] for member in validated] != list(MEMBER_KINDS):
        raise CandidateError("candidate manifest member kinds or ordering are invalid")
    filenames = [member["filename"] for member in validated]
    if len(filenames) != len(set(filenames)):
        raise CandidateError("candidate manifest contains duplicate filenames")
    if validated[-1]["filename"] != PROVENANCE_NAME:
        raise CandidateError("candidate manifest provenance filename is invalid")
    if validated[2]["filename"] != f"robot_sf-{version}.cyclonedx.json":
        raise CandidateError("candidate manifest SBOM filename is invalid")


def _validate_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CandidateError("candidate manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }
    if set(payload) != expected_keys:
        raise CandidateError("candidate manifest has missing or unclassified top-level fields")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CandidateError("candidate manifest schema_version is invalid")
    repository = payload.get("repository")
    if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
        raise CandidateError("candidate manifest repository is invalid")
    source_sha = payload.get("source_sha")
    if not isinstance(source_sha, str) or not SHA_PATTERN.fullmatch(source_sha):
        raise CandidateError("candidate manifest source_sha is invalid")
    _validate_workflow_identity(payload.get("workflow"))
    version = _validate_package_identity(payload.get("package"))
    if payload.get("validation") != _validation_payload():
        raise CandidateError("candidate manifest validation roster is missing or invalid")
    _validate_members(payload.get("members"), version=version)
    return payload


def _provenance_payload(
    *,
    repository: str,
    source_sha: str,
    workflow: dict[str, Any],
    package: dict[str, str],
    wheel: dict[str, Any],
    sdist: dict[str, Any],
    sbom: dict[str, Any],
) -> dict[str, Any]:
    return {
        "build": {"command": "uv build --out-dir $DIST_DIR", "count": 1},
        "package": package,
        "repository": repository,
        "sbom": sbom,
        "schema_version": PROVENANCE_VERSION,
        "source_sha": source_sha,
        "subjects": [wheel, sdist],
        "validation": _validation_payload(),
        "workflow": workflow,
    }


def _fresh_bundle_dir(path: Path) -> None:
    if path.is_symlink():
        raise CandidateError(f"bundle directory cannot be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise CandidateError(f"bundle path is not a directory: {path}")
        if any(path.iterdir()):
            raise CandidateError(f"bundle directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _assemble(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    _validate_schema_file(args.schema)
    _validate_source(repo_root, args.source_sha)
    for path, label in (
        (args.dist_dir, "distribution directory"),
        (args.raw_sbom, "raw SBOM"),
        (args.bundle_dir, "bundle directory"),
    ):
        _require_external(path, repo_root=repo_root, label=label)
    if list(args.validated) != list(VALIDATOR_IDS):
        raise CandidateError(
            "validated checks must be supplied exactly once in canonical order: "
            + ", ".join(VALIDATOR_IDS)
        )
    if not REPOSITORY_PATTERN.fullmatch(args.repository):
        raise CandidateError("repository must be an exact owner/name identity")
    if not RUN_ID_PATTERN.fullmatch(args.workflow_run_id):
        raise CandidateError("workflow run ID must be a positive decimal identity")
    if args.workflow_run_attempt < 1:
        raise CandidateError("workflow run attempt must be positive")

    wheel_input, sdist_input, version = _distribution_inputs(args.dist_dir)
    sbom_bytes = _normalised_sbom(args.raw_sbom, version)
    _validate_source(repo_root, args.source_sha)
    _fresh_bundle_dir(args.bundle_dir)

    wheel = args.bundle_dir / wheel_input.name
    sdist = args.bundle_dir / sdist_input.name
    sbom = args.bundle_dir / f"robot_sf-{version}.cyclonedx.json"
    provenance = args.bundle_dir / PROVENANCE_NAME
    shutil.copyfile(wheel_input, wheel)
    shutil.copyfile(sdist_input, sdist)
    sbom.write_bytes(sbom_bytes)

    wheel_member = _member(wheel, "wheel")
    sdist_member = _member(sdist, "sdist")
    sbom_member = _member(sbom, "sbom")
    workflow = {
        "run_attempt": args.workflow_run_attempt,
        "run_id": args.workflow_run_id,
    }
    package = {"name": "robot_sf", "version": version}
    provenance.write_bytes(
        _json_bytes(
            _provenance_payload(
                repository=args.repository,
                source_sha=args.source_sha,
                workflow=workflow,
                package=package,
                wheel=wheel_member,
                sdist=sdist_member,
                sbom=sbom_member,
            )
        )
    )
    manifest = {
        "members": [wheel_member, sdist_member, sbom_member, _member(provenance, "provenance")],
        "package": package,
        "repository": args.repository,
        "schema_version": SCHEMA_VERSION,
        "source_sha": args.source_sha,
        "validation": _validation_payload(),
        "workflow": workflow,
    }
    _validate_manifest(manifest)
    (args.bundle_dir / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
    _validate_source(repo_root, args.source_sha)
    print(
        f"PASS: admitted immutable Robot SF candidate {version} at {args.source_sha} "
        f"({len(manifest['members'])} bound payload members)"
    )


def _bundle_entries(bundle_dir: Path) -> list[Path]:
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise CandidateError(f"candidate bundle is not a real directory: {bundle_dir}")
    entries = sorted(bundle_dir.iterdir(), key=lambda path: path.name)
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"unclassified candidate bundle member: {path.name}")
    return entries


def _verify_bundle_membership(
    bundle_dir: Path,
    entries: list[Path],
    manifest: dict[str, Any],
) -> None:
    expected_names = {MANIFEST_NAME, *(member["filename"] for member in manifest["members"])}
    actual_names = {path.name for path in entries}
    if actual_names != expected_names or len(entries) != len(expected_names):
        missing = sorted(expected_names - actual_names)
        unclassified = sorted(actual_names - expected_names)
        raise CandidateError(
            "candidate bundle membership drift "
            f"(missing={missing or 'none'}, unclassified={unclassified or 'none'})"
        )
    for member in manifest["members"]:
        path = bundle_dir / member["filename"]
        size = path.stat().st_size
        digest = _sha256(path)
        if size != member["size"] or digest != member["sha256"]:
            raise CandidateError(
                f"candidate member drift: {path.name} expected "
                f"size={member['size']} sha256={member['sha256']}, "
                f"found size={size} sha256={digest}"
            )


def _verify_archives_and_sbom(
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> None:
    wheel_member, sdist_member, sbom_member, _provenance_member = manifest["members"]
    _wheel_name, wheel_version = _wheel_metadata(bundle_dir / wheel_member["filename"])
    _sdist_name, sdist_version = _sdist_metadata(bundle_dir / sdist_member["filename"])
    if wheel_version != manifest["package"]["version"] or sdist_version != wheel_version:
        raise CandidateError("candidate archive metadata drifted from the manifest package version")
    sbom = _load_json(bundle_dir / sbom_member["filename"], label="normalised SBOM")
    if not isinstance(sbom, dict) or "serialNumber" in sbom:
        raise CandidateError("candidate SBOM is not deterministically normalised")
    metadata = sbom.get("metadata")
    if not isinstance(metadata, dict) or "timestamp" in metadata:
        raise CandidateError("candidate SBOM contains a nondeterministic timestamp")
    component = metadata.get("component")
    if not isinstance(component, dict) or component.get("version") != wheel_version:
        raise CandidateError("candidate SBOM package version drifted from the archives")


def _verify_provenance(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    wheel_member, sdist_member, sbom_member, provenance_member = manifest["members"]
    provenance = _load_json(
        bundle_dir / provenance_member["filename"], label="candidate provenance"
    )
    expected_provenance = _provenance_payload(
        repository=manifest["repository"],
        source_sha=manifest["source_sha"],
        workflow=manifest["workflow"],
        package=manifest["package"],
        wheel=wheel_member,
        sdist=sdist_member,
        sbom=sbom_member,
    )
    if provenance != expected_provenance:
        raise CandidateError("candidate provenance does not exactly bind the manifest subjects")


def _verify(args: argparse.Namespace) -> None:
    _validate_schema_file(args.schema)
    if not SHA_PATTERN.fullmatch(args.expected_source_sha):
        raise CandidateError("expected source SHA must be one exact lowercase 40-hex identity")
    if not RUN_ID_PATTERN.fullmatch(args.expected_workflow_run_id):
        raise CandidateError("expected workflow run ID must be a positive decimal identity")
    entries = _bundle_entries(args.bundle_dir)
    manifest = _validate_manifest(
        _load_json(args.bundle_dir / MANIFEST_NAME, label="candidate manifest")
    )
    if manifest["source_sha"] != args.expected_source_sha:
        raise CandidateError(
            f"candidate source drift: expected {args.expected_source_sha}, "
            f"found {manifest['source_sha']}"
        )
    if manifest["workflow"]["run_id"] != args.expected_workflow_run_id:
        raise CandidateError(
            f"candidate workflow-run drift: expected {args.expected_workflow_run_id}, "
            f"found {manifest['workflow']['run_id']}"
        )
    _verify_bundle_membership(args.bundle_dir, entries, manifest)
    _verify_archives_and_sbom(args.bundle_dir, manifest)
    _verify_provenance(args.bundle_dir, manifest)
    print(
        f"PASS: offline candidate verification reused exact bytes from run "
        f"{manifest['workflow']['run_id']} at {manifest['source_sha']}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("check-source", help="fail unless source identity is exact")
    source.add_argument("--repo-root", type=Path, required=True)
    source.add_argument("--source-sha", required=True)

    assemble = subparsers.add_parser("assemble", help="admit already-built candidate bytes")
    assemble.add_argument("--repo-root", type=Path, required=True)
    assemble.add_argument("--dist-dir", type=Path, required=True)
    assemble.add_argument("--raw-sbom", type=Path, required=True)
    assemble.add_argument("--bundle-dir", type=Path, required=True)
    assemble.add_argument("--source-sha", required=True)
    assemble.add_argument("--repository", required=True)
    assemble.add_argument("--workflow-run-id", required=True)
    assemble.add_argument("--workflow-run-attempt", type=int, required=True)
    assemble.add_argument("--validated", action="append", default=[])
    assemble.add_argument("--schema", type=Path, default=SCHEMA_PATH)

    verify = subparsers.add_parser("verify", help="offline verification without rebuilding")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--expected-source-sha", required=True)
    verify.add_argument("--expected-workflow-run-id", required=True)
    verify.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed candidate command."""
    args = _parse_args(argv)
    try:
        if args.command == "check-source":
            _validate_source(args.repo_root.resolve(), args.source_sha)
            print(f"PASS: source identity is clean and exact at {args.source_sha}")
        elif args.command == "assemble":
            _assemble(args)
        elif args.command == "verify":
            _verify(args)
        else:  # pragma: no cover - argparse prevents this branch.
            raise CandidateError(f"unsupported candidate command: {args.command}")
    except CandidateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
