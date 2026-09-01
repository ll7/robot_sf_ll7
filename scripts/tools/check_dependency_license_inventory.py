#!/usr/bin/env python3
"""Generate a deterministic, fail-closed dependency license inventory.

The inventory has three deliberately separate layers:

* lock/package identity and profile membership are resolver facts;
* observed distribution metadata is captured without translating prose into
  permission; and
* the checked-in policy records the release-surface disposition that still
  requires maintainer or legal review.

The normal command emits blocked evidence and exits successfully so CI can
publish the report. Release preflight must add ``--fail-on-unresolved``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import stat
import sys
import tarfile
import tomllib
import zipfile
from collections import Counter, defaultdict, deque
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


SCHEMA_VERSION = "robot-sf.dependency-license-inventory.v1"
CANONICAL_PROFILE_MANIFEST = "scripts/validation/dependency_license_profiles.v1.json"
CANONICAL_POLICY = "scripts/validation/dependency_license_policy.v1.json"
PROFILE_SCHEMA_VERSION = "robot-sf.dependency-license-profiles.v1"
UNREPRESENTED_POLICY_SCHEMA_VERSION = "robot-sf.dependency-license-unrepresented.v1"
POLICY_SCHEMA_VERSION = "robot-sf.dependency-license-policy.v1"
_UNKNOWN_VALUES = frozenset({"", "unknown", "unknown license", "none", "null"})
_REVIEW_MARKERS = (
    "licenseref-",
    "proprietary",
    "nvidia",
    "non-commercial",
    "non commercial",
    "no redistribution",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_MANIFEST_NAME = "candidate-manifest.json"
_CANDIDATE_SCHEMA_VERSION = "robot_sf.software_candidate.v1"
_CANDIDATE_PROVENANCE_VERSION = "robot_sf.software_candidate.provenance.v1"
_CANDIDATE_MEMBER_KINDS = ("wheel", "sdist", "sbom", "provenance")
_CANDIDATE_MATERIALIZATION_FIELDS = {
    "candidate_commit_sha",
    "candidate_tree_sha",
    "policy_path",
    "policy_sha256",
    "source_inventory_path",
    "source_inventory_sha256",
    "candidate_inventory_path",
    "candidate_metadata_path",
}
_CANDIDATE_MATERIALIZATION_PATH_FIELDS = {
    "policy_path",
    "source_inventory_path",
    "candidate_inventory_path",
    "candidate_metadata_path",
}
_CANDIDATE_MATERIALIZATION_SHA_FIELDS = {
    "policy_sha256",
    "source_inventory_sha256",
}
_CANDIDATE_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_CANDIDATE_SOURCE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_RUN_ID_RE = re.compile(r"^[1-9][0-9]*$")
_CANDIDATE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]*$")
_CANDIDATE_VALIDATION_COMMANDS = (
    ("version-alignment", "python scripts/dev/check_version_alignment.py"),
    ("metadata", "twine check --strict $DIST_DIR/*.whl $DIST_DIR/*.tar.gz"),
    (
        "archive-license",
        "cd $BUILD_SOURCE && python scripts/tools/check_distribution_licenses.py "
        "$DIST_DIR --strict-asset-rights --repo-root $BUILD_SOURCE "
        "--inventory $BUILD_SOURCE/scripts/validation/software_candidate_asset_rights.v1.json "
        "--source-tree-ref HEAD",
    ),
    (
        "wheel-install",
        "cd $BUILD_SOURCE && bash scripts/validation/wheel_install_smoke.sh "
        "$DIST_DIR/robot_sf-*.whl",
    ),
)
_CANDIDATE_SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz")
_REQUIREMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[A-Za-z0-9._,-]+)\])?"
)
_EXTRA_RE = re.compile(r"extra\s*(==|!=)\s*['\"]([^'\"]+)['\"]")
_PYTHON_RE = re.compile(
    r"(?:python_full_version|python_version)\s*(==|!=|<=|>=|<|>)\s*['\"]([0-9.]+)['\"]"
)
_MARKER_ATOM_RE = re.compile(
    r"^(?P<key>[A-Za-z_]+)\s*(?P<operator>==|!=|<=|>=|<|>)\s*['\"](?P<value>[^'\"]+)['\"]$"
)
_KNOWN_SPDX_IDS = frozenset(
    {
        "0BSD",
        "AGPL-3.0-only",
        "Apache-1.1",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.1",
        "CC0-1.0",
        "CDDL-1.0",
        "EPL-2.0",
        "GPL-2.0-only",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "ISC",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "LLVM-exception",
        "MIT",
        "MIT-0",
        "MPL-1.1",
        "MPL-2.0",
        "OFL-1.1",
        "OpenSSL",
        "PSF-2.0",
        "Python-2.0",
        "Unlicense",
        "Zlib",
    }
)


def _canonical_json(value: Any) -> str:
    """Render JSON deterministically for identity hashes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonicalize_name(name: str) -> str:
    """Normalize a distribution name according to PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    """Hash a canonical JSON value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _relative_path(repo_root: Path, path: Path) -> str:
    """Return a repository-relative POSIX path without exposing host paths."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve a manifest path relative to the checked-out repository."""
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _normalise_json(value: Any) -> Any:
    """Normalize nested lock data while preserving its factual values."""
    if isinstance(value, dict):
        return {str(key): _normalise_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalise_json(item) for item in value]
    return value


def _requirement_record(requirement: str) -> dict[str, Any]:
    """Extract a requirement name and optional-extra references."""
    match = _REQUIREMENT_RE.match(requirement)
    if not match:
        raise ValueError(f"could not parse requirement {requirement!r}")
    extras = tuple(
        sorted(
            {
                item.strip().lower()
                for item in (match.group("extras") or "").split(",")
                if item.strip()
            }
        )
    )
    return {
        "raw": requirement,
        "name": match.group("name"),
        "normalized_name": _canonicalize_name(match.group("name")),
        "extras": list(extras),
    }


def _project_document(path: Path) -> dict[str, Any]:
    """Read one pyproject and require its project table."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{path} is missing its [project] table")
    return document


def _lock_source_type(source: dict[str, Any]) -> str:
    """Classify a uv lock source without inferring licensing."""
    for key, source_type in (
        ("editable", "editable"),
        ("directory", "directory"),
        ("git", "git"),
        ("url", "url"),
        ("registry", "registry"),
    ):
        if key in source:
            return source_type
    return "unknown"


def _artifact_record(kind: str, artifact: dict[str, Any]) -> dict[str, Any]:
    """Capture lock-provided artifact identity and hashes."""
    url = artifact.get("url")
    filename = None
    platform_tags: list[str] = []
    if isinstance(url, str):
        filename = unquote(Path(urlsplit(url).path).name) or None
    if kind == "wheel" and filename and filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        platform_tags = parts[-3:] if len(parts) >= 5 else []
    raw_hash = artifact.get("hash")
    sha256 = None
    if isinstance(raw_hash, str) and raw_hash.startswith("sha256:"):
        sha256 = raw_hash.removeprefix("sha256:")
    return {
        "kind": kind,
        "filename": filename,
        "url": url if isinstance(url, str) else None,
        "sha256": sha256,
        "size": artifact.get("size") if isinstance(artifact.get("size"), int) else None,
        "platform_tags": platform_tags,
    }


def _dependency_record(dependency: Any) -> dict[str, Any] | None:
    """Normalize one lock dependency edge without dropping marker facts."""
    if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
        return None
    entry: dict[str, Any] = {"name": dependency["name"]}
    for key in ("marker", "version"):
        if isinstance(dependency.get(key), str):
            entry[key] = dependency[key]
    if isinstance(dependency.get("source"), dict):
        entry["source"] = _normalise_json(dependency["source"])
    return entry


def _lock_packages(path: Path, repo_relative_path: str) -> list[dict[str, Any]]:
    """Read and normalize all package identities from one uv lock."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    packages = payload.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError(f"{path} must contain non-empty [[package]] entries")

    normalized: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str):
            raise ValueError(f"{path} contains a package without a string name")
        version = package.get("version")
        if version is not None and not isinstance(version, str):
            raise ValueError(f"{path} contains a package with a non-string version")
        source = package.get("source")
        if not isinstance(source, dict):
            source = {}
        source = _normalise_json(source)
        resolution_markers = [
            value for value in package.get("resolution-markers", []) if isinstance(value, str)
        ]
        artifacts: list[dict[str, Any]] = []
        sdist = package.get("sdist")
        if isinstance(sdist, dict):
            artifacts.append(_artifact_record("sdist", sdist))
        wheels = package.get("wheels")
        if isinstance(wheels, list):
            artifacts.extend(
                _artifact_record("wheel", wheel) for wheel in wheels if isinstance(wheel, dict)
            )
        dependencies: list[dict[str, Any]] = []
        for dependency in package.get("dependencies", []):
            entry = _dependency_record(dependency)
            if entry is not None:
                dependencies.append(entry)
        identity = {
            "lockfile": repo_relative_path,
            "name": package["name"],
            "version": version,
            "source": source,
            "resolution_markers": resolution_markers,
            "artifacts": artifacts,
        }
        package_id = (
            f"{_canonicalize_name(package['name'])}@{version or 'editable'}"
            f"#{_sha256_value(identity)[:16]}"
        )
        normalized.append(
            {
                "package_id": package_id,
                "lockfile": repo_relative_path,
                "name": package["name"],
                "normalized_name": _canonicalize_name(package["name"]),
                "version": version,
                "source_type": _lock_source_type(source),
                "source": source,
                "resolution_markers": resolution_markers,
                "artifacts": artifacts,
                "dependencies": dependencies,
                "identity_sha256": _sha256_value(identity),
            }
        )
    return sorted(normalized, key=lambda item: item["package_id"])


def _metadata_value(metadata: Any, key: str) -> str | None:
    """Return a trimmed metadata value unless it is an explicit empty marker."""
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value.lower() in _UNKNOWN_VALUES else value


def _looks_like_spdx(expression: str) -> bool:
    """Recognize only the conservative SPDX subset used by this gate."""
    if any(marker in expression.lower() for marker in _REVIEW_MARKERS):
        return False
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", expression)
    if not tokens:
        return False
    for token in tokens:
        if token in {"AND", "OR", "WITH"}:
            continue
        if token not in _KNOWN_SPDX_IDS:
            return False
    return True


def _license_record(distribution: Any) -> dict[str, Any]:
    """Capture raw license fields and classify them without legal inference."""
    metadata = distribution.metadata
    expression = _metadata_value(metadata, "License-Expression")
    license_value = _metadata_value(metadata, "License")
    classifiers = sorted(
        value
        for value in (metadata.get_all("Classifier") or [])
        if isinstance(value, str) and value.startswith("License :: ")
    )
    searchable = " ".join(
        value for value in (expression, license_value, *classifiers) if value
    ).lower()
    reasons: list[str] = []
    if not expression and not license_value and not classifiers:
        status = "unknown"
        reasons.append("distribution metadata contains no license fields")
    elif (
        expression
        and license_value
        and license_value.strip()
        not in {
            expression,
            f"SPDX: {expression}",
        }
    ):
        status = "metadata_conflict"
        reasons.append("License-Expression and legacy License fields disagree")
    elif any(marker in searchable for marker in _REVIEW_MARKERS):
        status = "proprietary"
        reasons.append("metadata contains a custom, proprietary, or restricted marker")
    elif expression and _looks_like_spdx(expression):
        status = "spdx_expression"
    elif expression:
        status = "non_spdx_text"
        reasons.append("License-Expression is not a recognized SPDX expression")
    else:
        status = "non_spdx_text"
        reasons.append("license metadata is not a normalized SPDX expression")

    raw_metadata = {
        "License-Expression": expression,
        "License": license_value,
        "Classifier": classifiers,
    }
    record: dict[str, Any] = {
        "license_status": status,
        "raw_license_metadata": raw_metadata,
        "license_expression": expression,
        "license_classifiers": classifiers,
        "review_reasons": reasons,
    }
    if license_value:
        record["license_field_sha256"] = hashlib.sha256(license_value.encode("utf-8")).hexdigest()
        record["license_field_first_line"] = license_value.splitlines()[0][:160]
    return record


def _observed_distributions(
    distributions: Iterable[Any] | None = None,
) -> dict[str, list[Any]]:
    """Index installed distributions by normalized name."""
    source = distributions if distributions is not None else importlib.metadata.distributions()
    observed: dict[str, list[Any]] = defaultdict(list)
    for distribution in source:
        name = distribution.metadata.get("Name") or distribution.name
        if isinstance(name, str) and name.strip():
            observed[_canonicalize_name(name)].append(distribution)
    for values in observed.values():
        values.sort(key=lambda item: (str(item.version), str(item.name).lower()))
    return dict(observed)


def _candidate_archive_name(name: str, *, archive: Path) -> str:
    """Normalize one candidate archive member name and reject unsafe paths."""
    normalized = name.replace("\\", "/")
    parts = normalized.rstrip("/").split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or not parts
        or any(
            not part
            or part in {".", ".."}
            or ":" in part
            or part.endswith((".", " "))
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in part)
            for part in parts
        )
    ):
        raise ValueError(f"{archive.name}: unsafe candidate archive member path: {name!r}")
    return "/".join(parts)


def _candidate_metadata(raw: bytes, *, archive: Path) -> dict[str, Any]:
    """Read the package identity and declared extras from an archive metadata file."""
    message = BytesParser(policy=email_policy).parsebytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{archive.name}: archive metadata has no Name field")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{archive.name}: archive metadata has no Version field")
    requires_dist = sorted(
        str(value).strip()
        for value in (message.get_all("Requires-Dist") or [])
        if str(value).strip()
    )
    provides_extra = sorted(
        str(value).strip().lower()
        for value in (message.get_all("Provides-Extra") or [])
        if str(value).strip()
    )
    return {
        "name": name,
        "normalized_name": _canonicalize_name(name),
        "version": version,
        "requires_dist": requires_dist,
        "provides_extra": provides_extra,
    }


def _candidate_zip_infos(archive: Path) -> tuple[list[zipfile.ZipInfo], list[str]]:
    """Read and validate the member names of a candidate zip archive."""
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        names = [_candidate_archive_name(info.filename, archive=archive) for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"{archive.name}: candidate archive has duplicate members")
        if any(stat.S_ISLNK(info.external_attr >> 16) for info in infos):
            raise ValueError(f"{archive.name}: candidate archive contains a symlink")
        return infos, names


def _candidate_zip_metadata(archive: Path, *, member_suffix: str) -> dict[str, Any]:
    """Read one metadata member from a candidate wheel or zip sdist."""
    try:
        infos, names = _candidate_zip_infos(archive)
        matches = [
            (info, name)
            for info, name in zip(infos, names, strict=True)
            if name.count("/") == 1 and name.endswith(member_suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{archive.name}: expected one {member_suffix} member; found {len(matches)}"
            )
        with zipfile.ZipFile(archive) as source:
            return _candidate_metadata(source.read(matches[0][0]), archive=archive)
    except ValueError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"cannot read candidate zip metadata from {archive}: {exc}") from exc


def _candidate_tar_metadata(archive: Path) -> dict[str, Any]:
    """Read one root PKG-INFO member from a candidate tar sdist."""
    try:
        with tarfile.open(archive, mode="r:*") as source:
            members = source.getmembers()
            names = [_candidate_archive_name(member.name, archive=archive) for member in members]
            if len(names) != len(set(names)):
                raise ValueError(f"{archive.name}: candidate archive has duplicate members")
            non_regular = [
                member.name for member in members if not member.isfile() and not member.isdir()
            ]
            if non_regular:
                raise ValueError(
                    f"{archive.name}: candidate archive has non-regular members: "
                    f"{', '.join(sorted(non_regular))}"
                )
            matches = [
                (member, name)
                for member, name in zip(members, names, strict=True)
                if name.count("/") == 1 and name.endswith("/PKG-INFO")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{archive.name}: expected one root PKG-INFO member; found {len(matches)}"
                )
            extracted = source.extractfile(matches[0][0])
            if extracted is None:
                raise ValueError(f"{archive.name}: cannot read root PKG-INFO")
            return _candidate_metadata(extracted.read(), archive=archive)
    except ValueError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"cannot read candidate sdist metadata from {archive}: {exc}") from exc


def _candidate_archive_metadata(archive: Path, kind: str) -> dict[str, Any]:
    """Read exactly one root package metadata member from a candidate archive."""
    if kind == "wheel":
        return _candidate_zip_metadata(archive, member_suffix=".dist-info/METADATA")
    if kind != "sdist":
        raise ValueError(f"unsupported candidate archive kind: {kind}")
    if archive.name.endswith(".zip"):
        return _candidate_zip_metadata(archive, member_suffix="/PKG-INFO")
    return _candidate_tar_metadata(archive)


def _candidate_member_path(
    bundle: Path,
    member: Any,
    *,
    expected_kind: str,
) -> tuple[Path, dict[str, Any]]:
    """Validate one candidate manifest member and return its bound file."""
    if not isinstance(member, dict):
        raise ValueError(f"candidate {expected_kind} member is not an object")
    filename = member.get("filename")
    digest = member.get("sha256")
    size = member.get("size")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", "..", _CANDIDATE_MANIFEST_NAME}
        or Path(filename).name != filename
        or "\\" in filename
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
    ):
        raise ValueError(f"candidate {expected_kind} member has an invalid binding")
    path = bundle / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"candidate {expected_kind} member is not a regular file: {filename}")
    actual_size = path.stat().st_size
    actual_digest = _sha256_file(path)
    if actual_size != size or actual_digest != digest:
        raise ValueError(
            f"candidate {expected_kind} member drift: {filename} expected "
            f"size={size} sha256={digest}, found size={actual_size} sha256={actual_digest}"
        )
    return path, {"filename": filename, "kind": expected_kind, "sha256": digest, "size": size}


def _candidate_validation_payload() -> dict[str, Any]:
    """Return the canonical software-candidate validation roster."""
    return {
        "checks": [
            {"command": command, "id": identifier, "status": "passed"}
            for identifier, command in _CANDIDATE_VALIDATION_COMMANDS
        ],
        "status": "passed",
    }


def _candidate_materialization_contract(value: Any) -> dict[str, Any]:
    """Validate the optional rights-scoped source identity envelope."""
    if not isinstance(value, dict) or set(value) != _CANDIDATE_MATERIALIZATION_FIELDS:
        raise ValueError("candidate materialization identity is missing or unclassified")
    for field in ("candidate_commit_sha", "candidate_tree_sha"):
        identity = value.get(field)
        if not isinstance(identity, str) or _CANDIDATE_SOURCE_SHA_RE.fullmatch(identity) is None:
            raise ValueError(f"candidate materialization {field} is invalid")
    for field in _CANDIDATE_MATERIALIZATION_SHA_FIELDS:
        digest = value.get(field)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"candidate materialization {field} is invalid")
    for field in _CANDIDATE_MATERIALIZATION_PATH_FIELDS:
        path = value.get(field)
        if not isinstance(path, str) or not path:
            raise ValueError(f"candidate materialization {field} is invalid")
        if (
            path.startswith("/")
            or "\\" in path
            or "\x00" in path
            or PurePosixPath(path).as_posix() != path
        ):
            raise ValueError(f"candidate materialization {field} is invalid")
        parts = PurePosixPath(path).parts
        if (
            not parts
            or parts[0] == ".git"
            or any(part in {"", ".", ".."} for part in parts)
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in path)
        ):
            raise ValueError(f"candidate materialization {field} is invalid")
    return value


def _candidate_manifest_contract(  # noqa: C901, PLR0912
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the closed v1 candidate manifest shape before binding its bytes."""
    required_keys = {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }
    if (
        set(manifest) not in (required_keys, required_keys | {"materialization"})
        or manifest.get("schema_version") != _CANDIDATE_SCHEMA_VERSION
    ):
        raise ValueError("candidate manifest has missing or unclassified contract fields")
    repository = manifest.get("repository")
    if not isinstance(repository, str) or _CANDIDATE_REPOSITORY_RE.fullmatch(repository) is None:
        raise ValueError("candidate manifest repository identity is invalid")
    source_sha = manifest.get("source_sha")
    if not isinstance(source_sha, str) or _CANDIDATE_SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("candidate manifest source SHA is invalid")
    workflow = manifest.get("workflow")
    if (
        not isinstance(workflow, dict)
        or set(workflow) != {"run_id", "run_attempt"}
        or not isinstance(workflow.get("run_id"), str)
        or _CANDIDATE_RUN_ID_RE.fullmatch(workflow["run_id"]) is None
        or not isinstance(workflow.get("run_attempt"), int)
        or isinstance(workflow["run_attempt"], bool)
        or workflow["run_attempt"] < 1
    ):
        raise ValueError("candidate manifest workflow identity is invalid")
    package = manifest.get("package")
    if (
        not isinstance(package, dict)
        or set(package) != {"name", "version"}
        or package.get("name") != "robot_sf"
        or not isinstance(package.get("version"), str)
        or _CANDIDATE_VERSION_RE.fullmatch(package["version"]) is None
    ):
        raise ValueError("candidate manifest package identity is invalid")
    if manifest.get("validation") != _candidate_validation_payload():
        raise ValueError("candidate manifest validation roster is invalid")
    if "materialization" in manifest:
        _candidate_materialization_contract(manifest["materialization"])
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(_CANDIDATE_MEMBER_KINDS):
        raise ValueError("candidate manifest must bind exactly four payload members")
    filenames: set[str] = set()
    for member, expected_kind in zip(members, _CANDIDATE_MEMBER_KINDS, strict=True):
        if not isinstance(member, dict) or set(member) != {"filename", "kind", "sha256", "size"}:
            raise ValueError("candidate manifest member record is invalid")
        filename = member["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", "..", _CANDIDATE_MANIFEST_NAME}
            or Path(filename).name != filename
            or "\\" in filename
        ):
            raise ValueError("candidate manifest member filename is unsafe or reserved")
        if member["kind"] != expected_kind:
            raise ValueError("candidate manifest member kinds or ordering are invalid")
        if not isinstance(member["sha256"], str) or _SHA256_RE.fullmatch(member["sha256"]) is None:
            raise ValueError(f"candidate member {filename} has an invalid SHA-256")
        if (
            not isinstance(member["size"], int)
            or isinstance(member["size"], bool)
            or member["size"] < 1
        ):
            raise ValueError(f"candidate member {filename} has an invalid size")
        filenames.add(filename)
    if len(filenames) != len(members):
        raise ValueError("candidate manifest contains duplicate filenames")
    if members[2]["filename"] != f"robot_sf-{package['version']}.cyclonedx.json":
        raise ValueError("candidate manifest SBOM filename is invalid")
    if members[3]["filename"] != "candidate-provenance.json":
        raise ValueError("candidate manifest provenance filename is invalid")
    return package


def _candidate_manifest_members(
    bundle: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Validate and resolve all members named by a candidate manifest."""
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("candidate manifest has no member list")
    members_by_kind: dict[str, dict[str, Any]] = {}
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("kind"), str):
            raise ValueError("candidate manifest contains an unclassified member")
        kind = member["kind"]
        if kind in members_by_kind:
            raise ValueError(f"candidate manifest has duplicate {kind} members")
        members_by_kind[kind] = member
    for kind in ("wheel", "sdist", "sbom", "provenance"):
        if kind not in members_by_kind:
            raise ValueError(f"candidate manifest is missing its {kind} member")
    expected_names = {_CANDIDATE_MANIFEST_NAME}
    paths_by_kind: dict[str, Path] = {}
    bound_members: dict[str, dict[str, Any]] = {}
    for kind, member in members_by_kind.items():
        path, bound = _candidate_member_path(bundle, member, expected_kind=kind)
        paths_by_kind[kind] = path
        bound_members[kind] = bound
        expected_names.add(bound["filename"])
    actual_names = {path.name for path in bundle.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unclassified = sorted(actual_names - expected_names)
        raise ValueError(
            "candidate bundle membership drift "
            f"(missing={missing or 'none'}, unclassified={unclassified or 'none'})"
        )
    return paths_by_kind, bound_members


def _candidate_archive_contract(
    paths_by_kind: dict[str, Path],
    package: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate matching wheel and sdist metadata for a candidate package."""
    wheel = paths_by_kind["wheel"]
    if not wheel.name.endswith(".whl"):
        raise ValueError(f"candidate wheel has an unsupported filename: {wheel.name}")
    wheel_parts = wheel.name.removesuffix(".whl").split("-")
    if (
        len(wheel_parts) < 5
        or wheel_parts[0] != "robot_sf"
        or wheel_parts[1] != package["version"].replace("-", "_")
    ):
        raise ValueError(f"candidate wheel filename does not match package version: {wheel.name}")
    sdist = paths_by_kind["sdist"]
    if not any(
        sdist.name == f"robot_sf-{package['version']}{suffix}"
        for suffix in _CANDIDATE_SDIST_SUFFIXES
    ):
        raise ValueError(f"candidate sdist filename does not match package version: {sdist.name}")
    wheel_metadata = _candidate_archive_metadata(paths_by_kind["wheel"], "wheel")
    sdist_metadata = _candidate_archive_metadata(paths_by_kind["sdist"], "sdist")
    for label, metadata in (("wheel", wheel_metadata), ("sdist", sdist_metadata)):
        if metadata["normalized_name"] != "robot-sf":
            raise ValueError(f"candidate {label} metadata is not for robot_sf")
        if metadata["version"] != package["version"]:
            raise ValueError(
                f"candidate {label} version {metadata['version']!r} does not match "
                f"manifest version {package['version']!r}"
            )
    if set(wheel_metadata["provides_extra"]) != set(sdist_metadata["provides_extra"]):
        raise ValueError("candidate wheel and sdist advertise different optional extras")
    wheel_names = {
        _canonicalize_name(value.split("[", maxsplit=1)[0].split(";", maxsplit=1)[0])
        for value in wheel_metadata["requires_dist"]
    }
    sdist_names = {
        _canonicalize_name(value.split("[", maxsplit=1)[0].split(";", maxsplit=1)[0])
        for value in sdist_metadata["requires_dist"]
    }
    if wheel_names != sdist_names:
        raise ValueError("candidate wheel and sdist advertise different direct dependencies")
    return wheel_metadata, sdist_metadata


def _candidate_provenance_contract(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    """Verify that candidate provenance exactly binds the manifest subjects."""
    provenance = _read_json(path)
    members = manifest["members"]
    expected = {
        "build": {
            "command": "cd $BUILD_SOURCE && uv build --out-dir $DIST_DIR",
            "count": 1,
            "source_role": "disposable-exact-commit",
        },
        "package": manifest["package"],
        "repository": manifest["repository"],
        "sbom": members[2],
        "schema_version": _CANDIDATE_PROVENANCE_VERSION,
        "source_sha": manifest["source_sha"],
        "subjects": [members[0], members[1]],
        "validation": manifest["validation"],
        "workflow": manifest["workflow"],
    }
    if "materialization" in manifest:
        expected["materialization"] = manifest["materialization"]
    if provenance != expected:
        raise ValueError("candidate provenance does not exactly bind the manifest subjects")


def _candidate_sbom_components(path: Path, package: dict[str, Any]) -> set[tuple[str, str]]:
    """Read a normalized candidate SBOM and return its component identities."""
    sbom = _read_json(path)
    if (
        sbom.get("bomFormat") != "CycloneDX"
        or sbom.get("specVersion") != "1.5"
        or sbom.get("version") != 1
        or "serialNumber" in sbom
    ):
        raise ValueError("candidate SBOM must be CycloneDX 1.5 document version 1")
    sbom_metadata = sbom.get("metadata")
    if not isinstance(sbom_metadata, dict) or "timestamp" in sbom_metadata:
        raise ValueError("candidate SBOM metadata is not deterministic")
    root_component = sbom_metadata.get("component") if isinstance(sbom_metadata, dict) else None
    if (
        not isinstance(root_component, dict)
        or _canonicalize_name(str(root_component.get("name", ""))) != "robot-sf"
        or root_component.get("version") != package["version"]
    ):
        raise ValueError("candidate SBOM root identity does not match the archives")
    components = sbom.get("components")
    if not isinstance(components, list) or not isinstance(sbom.get("dependencies"), list):
        raise ValueError("candidate SBOM must contain components and dependencies arrays")
    identities: set[tuple[str, str]] = set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("candidate SBOM contains an unclassified component")
        name = component.get("name")
        version = component.get("version")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str):
            raise ValueError("candidate SBOM component is missing name or version")
        identity = (_canonicalize_name(name), version)
        if identity in identities:
            raise ValueError(f"candidate SBOM contains duplicate component: {name}@{version}")
        identities.add(identity)
    return identities


def _candidate_bundle_binding(  # noqa: C901
    bundle_path: Path,
    *,
    selected_profile_ids: set[str],
    selected_package_ids: set[str],
    all_packages: dict[str, dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind a candidate bundle's archives and SBOM to selected lock closures."""
    if bundle_path.is_symlink() or not bundle_path.is_dir():
        raise ValueError(f"candidate bundle is not a real directory: {bundle_path}")
    bundle = bundle_path.resolve()
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError(f"candidate bundle is not a real directory: {bundle_path}")
    manifest_path = bundle / _CANDIDATE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("candidate manifest is not a regular file")
    manifest = _read_json(manifest_path)
    package = _candidate_manifest_contract(manifest)
    paths_by_kind, bound_members = _candidate_manifest_members(bundle, manifest)
    wheel_metadata, sdist_metadata = _candidate_archive_contract(paths_by_kind, package)
    _candidate_provenance_contract(paths_by_kind["provenance"], manifest)
    actual_components = _candidate_sbom_components(paths_by_kind["sbom"], package)

    selected_profiles = [
        profile for profile in profiles if profile.get("id") in selected_profile_ids
    ]
    non_robot_roots = sorted(
        str(profile.get("id"))
        for profile in selected_profiles
        if _canonicalize_name(str(profile.get("root_package", ""))) != "robot-sf"
    )
    if non_robot_roots:
        raise ValueError(
            "candidate bundle can only bind profiles rooted at robot-sf; "
            f"non-robot profiles={non_robot_roots}"
        )
    expected_components: set[tuple[str, str]] = set()
    for package_id in selected_package_ids:
        package_row = all_packages.get(package_id)
        if package_row is None:
            raise ValueError(f"selected profile references an unknown lock package: {package_id}")
        if package_row["normalized_name"] == "robot-sf":
            continue
        version = package_row.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError(
                f"selected lock package has no version for SBOM binding: {package_row['name']}"
            )
        expected_components.add((package_row["normalized_name"], version))
    missing_components = sorted(expected_components - actual_components)
    unexpected_components = sorted(actual_components - expected_components)
    if missing_components or unexpected_components:
        raise ValueError(
            "candidate SBOM component closure differs from selected lock profiles "
            f"(missing={missing_components or 'none'}, "
            f"unexpected={unexpected_components or 'none'})"
        )
    expected_extras = {
        extra for profile in selected_profiles for extra in _profile_extra_names(profile)
    }
    provided_extras = set(wheel_metadata["provides_extra"])
    if not expected_extras <= provided_extras:
        raise ValueError(
            "candidate archives do not advertise selected extras: "
            f"{sorted(expected_extras - provided_extras)}"
        )
    return {
        "status": "bound",
        "manifest_sha256": _sha256_file(manifest_path),
        "repository": manifest["repository"],
        "source_sha": manifest["source_sha"],
        "workflow": _normalise_json(manifest["workflow"]),
        "materialization": _normalise_json(manifest["materialization"])
        if "materialization" in manifest
        else None,
        "package": {"name": package["name"], "version": package["version"]},
        "members": [bound_members[kind] for kind in sorted(bound_members)],
        "archives": {"wheel": wheel_metadata, "sdist": sdist_metadata},
        "sbom": {
            "filename": bound_members["sbom"]["filename"],
            "sha256": bound_members["sbom"]["sha256"],
            "component_count": len(actual_components),
            "component_set_sha256": _sha256_value(
                [f"{name}@{version}" for name, version in sorted(actual_components)]
            ),
        },
        "profile_ids": sorted(selected_profile_ids),
        "expected_component_count": len(expected_components),
    }


def _profile_extra_names(profile: dict[str, Any]) -> set[str]:
    """Return extras whose marker dependencies are allowed for one profile."""
    values = profile.get("extras")
    if not isinstance(values, list):
        values = []
    extra = profile.get("extra")
    if isinstance(extra, str) and extra != "all":
        values = [*values, extra]
    return {str(value).lower() for value in values if isinstance(value, str)}


def _version_tuple(value: str) -> tuple[int, ...]:
    """Convert a dotted Python version into a comparable tuple."""
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def _comparison_applies(operator: str, observed: str, expected: str) -> bool:
    """Evaluate one simple PEP 508 version comparison."""
    left = _version_tuple(observed)
    right = _version_tuple(expected)
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left >= right


def _marker_applies(
    marker: str | None,
    extras: set[str],
    python_version: str | None = None,
) -> bool:
    """Conservatively select a lock dependency for a profile."""
    if not marker:
        return True

    def atom_applies(atom: str) -> bool:
        atom = atom.strip().strip("()")
        extra_matches = _EXTRA_RE.findall(atom)
        if extra_matches:
            return all(
                (value.lower() in extras) if operator == "==" else (value.lower() not in extras)
                for operator, value in extra_matches
            )
        python_matches = _PYTHON_RE.findall(atom)
        if python_matches and python_version is not None:
            return all(
                _comparison_applies(operator, python_version, expected)
                for operator, expected in python_matches
            )
        return True

    or_terms = re.split(r"\s+or\s+", marker, flags=re.IGNORECASE)
    return any(
        all(atom_applies(atom) for atom in re.split(r"\s+and\s+", term, flags=re.IGNORECASE))
        for term in or_terms
    )


def _strip_marker_parentheses(value: str) -> str:
    """Remove balanced outer parentheses from one marker term."""
    value = value.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        balanced = True
        for index, character in enumerate(value):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        value = value[1:-1].strip()
    return value


def _target_marker_environment(manifest: dict[str, Any]) -> dict[str, str]:
    """Map the profile target to the marker variables used by uv lock rows."""
    target = manifest.get("target")
    target = target if isinstance(target, dict) else {}
    python = target.get("python")
    python = python if isinstance(python, dict) else {}
    python_version = str(python.get("version") or "")
    version_parts = python_version.split(".")
    if len(version_parts) == 2 and all(part.isdigit() for part in version_parts):
        python_full_version = f"{python_version}.0"
    else:
        python_full_version = python_version
    operating_system = str(target.get("os") or "")
    if operating_system in {"linux", "darwin"}:
        os_name = "posix"
    elif operating_system == "win32":
        os_name = "nt"
    else:
        os_name = operating_system
    implementation = str(python.get("implementation") or "")
    return {
        "python_full_version": python_full_version,
        "python_version": python_version,
        "sys_platform": operating_system,
        "os_name": os_name,
        "platform_machine": str(target.get("architecture") or ""),
        "platform_python_implementation": implementation,
        "implementation_name": implementation.lower(),
    }


def _marker_comparison_state(
    operator: str,
    observed: str,
    expected: str,
    *,
    version: bool,
) -> bool | None:
    """Evaluate a marker comparison, returning ``None`` when unsupported."""
    if version:
        if expected.endswith(".*"):
            prefix = expected[:-2]
            equal = observed == prefix or observed.startswith(f"{prefix}.")
            if operator == "==":
                return equal
            if operator == "!=":
                return not equal
            return None
        return _comparison_applies(operator, observed, expected)
    if operator == "==":
        return observed.casefold() == expected.casefold()
    if operator == "!=":
        return observed.casefold() != expected.casefold()
    return None


def _marker_state(  # noqa: C901
    marker: str | None,
    extras: set[str],
    environment: dict[str, str],
) -> bool | None:
    """Evaluate a marker conservatively for the manifest target.

    ``None`` means that the expression uses a variable or operator this small
    offline evaluator cannot prove.  Unknown expressions are included by
    dependency traversal and remain unresolved in the disposition report.
    """
    if not marker:
        return True

    def atom_state(atom: str) -> bool | None:
        atom = _strip_marker_parentheses(atom)
        match = _MARKER_ATOM_RE.fullmatch(atom)
        if match is None:
            return None
        key = match.group("key")
        operator = match.group("operator")
        expected = match.group("value")
        if key == "extra":
            if operator == "==":
                return expected.casefold() in {value.casefold() for value in extras}
            if operator == "!=":
                return expected.casefold() not in {value.casefold() for value in extras}
            return None
        observed = environment.get(key)
        if observed is None:
            return None
        return _marker_comparison_state(
            operator,
            observed,
            expected,
            version=key in {"python_full_version", "python_version"},
        )

    term_states: list[bool | None] = []
    for term in re.split(r"\s+or\s+", marker, flags=re.IGNORECASE):
        atom_states = [
            atom_state(atom) for atom in re.split(r"\s+and\s+", term, flags=re.IGNORECASE)
        ]
        if any(state is False for state in atom_states):
            term_states.append(False)
        elif all(state is True for state in atom_states):
            term_states.append(True)
        else:
            term_states.append(None)
    if any(state is True for state in term_states):
        return True
    if all(state is False for state in term_states):
        return False
    return None


def _resolution_marker_state(
    package: dict[str, Any],
    environment: dict[str, str],
) -> bool | None:
    """Return whether a lock package row applies to the manifest target."""
    markers = package.get("resolution_markers")
    if not isinstance(markers, list) or not markers:
        return True
    states = [
        _marker_state(value, set(), environment) for value in markers if isinstance(value, str)
    ]
    if any(state is True for state in states):
        return True
    if all(state is False for state in states):
        return False
    return None


def _validate_manifest(  # noqa: C901
    manifest: dict[str, Any],
    root_project: dict[str, Any],
) -> list[str]:
    """Validate profile coverage and the declared all-extra closure."""
    issues: list[str] = []
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        issues.append("profile manifest has an unsupported schema_version")
    target = manifest.get("target")
    if not isinstance(target, dict):
        issues.append("profile manifest is missing target metadata")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        return [*issues, "profile manifest has no profiles"]
    profile_ids: list[str] = []
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("id"), str):
            issues.append("profile manifest contains a profile without a string id")
            continue
        profile_ids.append(profile["id"])
    duplicates = sorted(name for name, count in Counter(profile_ids).items() if count > 1)
    issues.extend(f"duplicate profile id: {name}" for name in duplicates)

    project = root_project.get("project")
    optional = project.get("optional-dependencies", {}) if isinstance(project, dict) else {}
    if not isinstance(optional, dict):
        optional = {}
    declared = {str(name) for name in optional}
    required = {"core", *declared}
    issues.extend(
        f"declared optional extra is missing a profile: {name}"
        for name in sorted(required - set(profile_ids))
    )
    if "all" in optional:
        all_refs: set[str] = set()
        for requirement in optional["all"]:
            if isinstance(requirement, str):
                all_refs.update(_requirement_record(requirement)["extras"])
        all_profile = next(
            (
                profile
                for profile in profiles
                if isinstance(profile, dict) and profile.get("id") == "all"
            ),
            None,
        )
        exclusions = {
            str(value).lower()
            for value in (all_profile or {}).get("excluded_extras", [])
            if isinstance(value, str)
        }
        undeclared_exclusions = exclusions - declared
        if undeclared_exclusions:
            issues.append(
                f"all profile excludes undeclared extras: {sorted(undeclared_exclusions)}"
            )
        expected = declared - {"all"} - exclusions
        if all_refs != expected:
            issues.append(
                "all extra does not reference every declared optional extra: "
                f"expected={sorted(expected)} observed={sorted(all_refs)}"
            )
        if isinstance(all_profile, dict):
            manifest_refs = {
                str(value).lower()
                for value in all_profile.get("extras", [])
                if isinstance(value, str)
            }
            if manifest_refs != all_refs:
                issues.append(
                    "all profile closure disagrees with pyproject.toml: "
                    f"manifest={sorted(manifest_refs)} project={sorted(all_refs)}"
                )
    return issues


def _validate_unrepresented_policy(  # noqa: C901, PLR0912
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Validate the reviewed reasons used for unrepresented lock rows."""
    issues: list[str] = []
    policy = manifest.get("unrepresented_policy")
    if not isinstance(policy, dict):
        return [], {}, ["profile manifest has no unrepresented_policy"]
    if policy.get("schema_version") != UNREPRESENTED_POLICY_SCHEMA_VERSION:
        issues.append("unrepresented policy has an unsupported schema_version")
    raw_rules = policy.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        issues.append("unrepresented policy has no rules")
        raw_rules = []
    rules: list[dict[str, Any]] = []
    rule_ids: list[str] = []
    allowed_reason_codes = {
        "alternate_lock_context",
        "development_group",
        "marker_inactive",
        "unresolved_membership",
    }
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            issues.append(f"unrepresented policy rule {index} is not an object")
            continue
        rule = dict(raw_rule)
        rule_id = rule.get("id")
        reason_code = rule.get("reason_code")
        if not isinstance(rule_id, str) or not rule_id:
            issues.append(f"unrepresented policy rule {index} has no id")
        else:
            rule_ids.append(rule_id)
        if reason_code not in allowed_reason_codes:
            issues.append(f"unrepresented policy rule {rule_id!r} has an invalid reason_code")
        if not isinstance(rule.get("reviewed"), bool):
            issues.append(f"unrepresented policy rule {rule_id!r} has no boolean reviewed field")
        if not isinstance(rule.get("rationale"), str) or not rule["rationale"].strip():
            issues.append(f"unrepresented policy rule {rule_id!r} has no rationale")
        if reason_code == "development_group":
            for key in ("lockfile", "root_package", "field"):
                if not isinstance(rule.get(key), str) or not rule[key]:
                    issues.append(f"development-group rule {rule_id!r} is missing {key}")
            if rule.get("field") not in {"dev-dependencies", "optional-dependencies"}:
                issues.append(f"development-group rule {rule_id!r} has an invalid field")
            groups = rule.get("groups")
            if (
                not isinstance(groups, list)
                or not groups
                or not all(isinstance(group, str) and group for group in groups)
            ):
                issues.append(f"development-group rule {rule_id!r} has no groups")
        rules.append(rule)
    duplicates = sorted(name for name, count in Counter(rule_ids).items() if count > 1)
    issues.extend(f"duplicate unrepresented policy rule id: {name}" for name in duplicates)

    fallback = policy.get("unresolved")
    if not isinstance(fallback, dict):
        issues.append("unrepresented policy has no unresolved fallback")
        fallback = {}
    elif fallback.get("reason_code") != "unresolved_membership":
        issues.append("unrepresented policy unresolved fallback must use unresolved_membership")
    if not isinstance(fallback.get("reviewed"), bool):
        issues.append("unrepresented policy unresolved fallback has no boolean reviewed field")
    if not isinstance(fallback.get("rationale"), str) or not fallback["rationale"].strip():
        issues.append("unrepresented policy unresolved fallback has no rationale")
    return rules, fallback, issues


def _profile_requirements(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Extract direct requirements for one profile."""
    project_path = _resolve_path(repo_root, str(profile.get("pyproject", "pyproject.toml")))
    document = _project_document(project_path)
    project = document["project"]
    if profile.get("kind") in {"built-companion", "vendored-companion"}:
        return []
    base_requirements = project.get("dependencies", [])
    if not isinstance(base_requirements, list):
        raise ValueError(f"profile {profile.get('id')} has non-list base requirements")
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        optional = {}
    selected_extras: list[str] = []
    extra = profile.get("extra")
    if isinstance(extra, str) and extra != "all":
        selected_extras.append(extra)
    selected_extras.extend(
        str(value)
        for value in profile.get("extras", [])
        if isinstance(value, str) and value not in selected_extras
    )
    requirements = [*base_requirements]
    for selected_extra in selected_extras:
        if selected_extra not in optional:
            raise ValueError(
                f"profile {profile.get('id')} references undeclared extra {selected_extra!r} "
                f"in {_relative_path(repo_root, project_path)}"
            )
        extra_requirements = optional.get(selected_extra, [])
        if not isinstance(extra_requirements, list):
            raise ValueError(f"profile {profile.get('id')} extra {selected_extra} is not a list")
        requirements.extend(extra_requirements)
    return [
        _requirement_record(requirement)
        for requirement in requirements
        if isinstance(requirement, str)
    ]


def _package_variants(
    packages_by_name: dict[str, list[dict[str, Any]]],
    dependency: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select lock variants matching a dependency edge."""
    candidates = packages_by_name.get(_canonicalize_name(dependency["name"]), [])
    version = dependency.get("version")
    if isinstance(version, str):
        candidates = [candidate for candidate in candidates if candidate.get("version") == version]
    source = dependency.get("source")
    if isinstance(source, dict):
        normalized_source = _normalise_json(source)
        candidates = [
            candidate for candidate in candidates if candidate.get("source") == normalized_source
        ]
    return candidates


def _lock_group_dependencies(
    path: Path,
    root_package: str,
    field: str,
    groups: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Read named development/alternate-context groups from one lock root."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    packages = payload.get("package")
    if not isinstance(packages, list):
        return {}, [f"{path} has no package rows"]
    root = next(
        (
            package
            for package in packages
            if isinstance(package, dict)
            and package.get("name") == root_package
            and isinstance(package.get(field), dict)
        ),
        None,
    )
    if root is None:
        return {}, [f"{path} has no {field} for root package {root_package}"]
    raw_groups = root[field]
    result: dict[str, list[dict[str, Any]]] = {}
    issues: list[str] = []
    for group in groups:
        raw_edges = raw_groups.get(group)
        if not isinstance(raw_edges, list):
            issues.append(f"{path} {root_package} has no {field} group {group}")
            continue
        edges = [
            dependency
            for raw_edge in raw_edges
            if (dependency := _dependency_record(raw_edge)) is not None
        ]
        result[group] = edges
    return result, issues


def _dependency_context_closure(
    lock_packages: list[dict[str, Any]],
    direct_dependencies: list[dict[str, Any]],
    environment: dict[str, str],
) -> set[str]:
    """Resolve one reviewed non-release group conservatively from a lock."""
    packages_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for package in lock_packages:
        packages_by_name[package["normalized_name"]].append(package)
    queue: deque[dict[str, Any]] = deque()
    for dependency in direct_dependencies:
        marker_state = _marker_state(dependency.get("marker"), set(), environment)
        if marker_state is False:
            continue
        queue.extend(
            variant
            for variant in _package_variants(packages_by_name, dependency)
            if _resolution_marker_state(variant, environment) is not False
        )
    seen: set[str] = set()
    while queue:
        package = queue.popleft()
        package_id = package["package_id"]
        if package_id in seen:
            continue
        seen.add(package_id)
        for dependency in package["dependencies"]:
            marker_state = _marker_state(dependency.get("marker"), set(), environment)
            if marker_state is False:
                continue
            queue.extend(
                variant
                for variant in _package_variants(packages_by_name, dependency)
                if _resolution_marker_state(variant, environment) is not False
            )
    return seen


def _classify_unrepresented_packages(  # noqa: C901
    packages: list[dict[str, Any]],
    package_profiles: dict[str, set[str]],
    packages_by_lockfile: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    rules: list[dict[str, Any]],
    fallback: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach a reviewed or fail-closed disposition to every unrepresented row."""
    environment = _target_marker_environment(manifest)
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[str] = []
    for rule in rules:
        if rule.get("reason_code") != "development_group":
            continue
        lockfile = rule.get("lockfile")
        root_package = rule.get("root_package")
        field = rule.get("field")
        groups = rule.get("groups")
        if not all(isinstance(value, str) for value in (lockfile, root_package, field)):
            continue
        if not isinstance(groups, list) or not all(isinstance(group, str) for group in groups):
            continue
        lock_path = _resolve_path(repo_root, lockfile)
        try:
            group_edges, group_issues = _lock_group_dependencies(
                lock_path,
                root_package,
                field,
                groups,
            )
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            issues.append(f"could not read unrepresented context {lockfile}: {exc}")
            continue
        issues.extend(group_issues)
        lock_packages = packages_by_lockfile.get(lockfile, [])
        for group, direct_dependencies in group_edges.items():
            for package_id in _dependency_context_closure(
                lock_packages,
                direct_dependencies,
                environment,
            ):
                contexts[package_id].append({"rule": rule, "group": group})

    marker_rule = next(
        (rule for rule in rules if rule.get("reason_code") == "marker_inactive"),
        None,
    )
    dispositions: list[dict[str, Any]] = []
    for package in packages:
        package_id = package["package_id"]
        if package_profiles.get(package_id):
            continue
        marker_state = _resolution_marker_state(package, environment)
        if marker_state is False and marker_rule is not None:
            selected_rules = [marker_rule]
            groups: list[str] = []
            reason_codes = ["marker_inactive"]
        elif contexts.get(package_id):
            selected_rules = [item["rule"] for item in contexts[package_id]]
            groups = sorted({item["group"] for item in contexts[package_id]})
            reason_codes = sorted({str(rule.get("reason_code")) for rule in selected_rules})
        else:
            selected_rules = [fallback]
            groups = []
            reason_codes = [str(fallback.get("reason_code", "unresolved_membership"))]
        reviewed = all(bool(rule.get("reviewed")) for rule in selected_rules)
        status = "reviewed_exclusion" if reviewed else "unresolved"
        dispositions.append(
            {
                "package_id": package_id,
                "name": package["name"],
                "version": package.get("version"),
                "lockfile": package["lockfile"],
                "status": status,
                "reviewed": reviewed,
                "reason_codes": reason_codes,
                "groups": groups,
                "rule_ids": sorted(
                    str(rule.get("id"))
                    for rule in selected_rules
                    if isinstance(rule.get("id"), str)
                ),
                "rationales": sorted(
                    str(rule.get("rationale"))
                    for rule in selected_rules
                    if isinstance(rule.get("rationale"), str)
                ),
                "resolution_markers": package.get("resolution_markers", []),
            }
        )
    return sorted(dispositions, key=lambda item: item["package_id"]), sorted(set(issues))


def _direct_package_variants(
    packages_by_name: dict[str, list[dict[str, Any]]],
    dependency: dict[str, Any],
    roots: list[dict[str, Any]],
    extras: set[str],
    python_version: str | None,
) -> list[dict[str, Any]]:
    """Prefer the root lock's selected edge before reporting variant conflicts."""
    selected: dict[str, dict[str, Any]] = {}
    for root in roots:
        for edge in root["dependencies"]:
            if _canonicalize_name(edge["name"]) != dependency["normalized_name"]:
                continue
            marker = edge.get("marker")
            if isinstance(marker, str) and not _marker_applies(marker, extras, python_version):
                continue
            for variant in _package_variants(packages_by_name, edge):
                selected[variant["package_id"]] = variant
    return list(selected.values()) or _package_variants(packages_by_name, dependency)


def _resolve_profile(  # noqa: C901
    profile: dict[str, Any],
    lock_packages: list[dict[str, Any]],
    manifest: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve one profile's lock closure without performing network access."""
    profile_id = str(profile["id"])
    result: dict[str, Any] = {
        "id": profile_id,
        "kind": profile.get("kind"),
        "extra": profile.get("extra"),
        "extras": [value for value in profile.get("extras", []) if isinstance(value, str)],
        "excluded_extras": [
            value for value in profile.get("excluded_extras", []) if isinstance(value, str)
        ],
        "expected_resolution": profile.get("expected_resolution"),
        "package_ids": [],
        "missing_dependencies": [],
        "conflicting_dependencies": [],
    }
    try:
        result["direct_requirements"] = _profile_requirements(profile, manifest, repo_root)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        result["direct_requirements"] = []
        result["missing_dependencies"] = [f"profile input could not be read: {exc}"]

    if profile.get("expected_resolution") != "locked":
        result["status"] = str(profile.get("expected_resolution"))
        return result

    packages_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for package in lock_packages:
        packages_by_name[package["normalized_name"]].append(package)
    root_name = profile.get("root_package")
    if not isinstance(root_name, str):
        result["missing_dependencies"] = ["locked profile is missing root_package"]
        result["status"] = "blocked"
        return result
    roots = packages_by_name.get(_canonicalize_name(root_name), [])
    if not roots:
        result["missing_dependencies"] = [f"root package is absent from lock: {root_name}"]
        result["status"] = "blocked"
        return result

    allowed_extras = _profile_extra_names(profile)
    target = manifest.get("target")
    target_python = (
        target.get("python", {}).get("version")
        if isinstance(target, dict) and isinstance(target.get("python"), dict)
        else None
    )
    queue: deque[tuple[dict[str, Any], str]] = deque((root, "direct") for root in roots)
    seen: dict[str, str] = {}
    for direct_requirement in result.get("direct_requirements", []):
        variants = _direct_package_variants(
            packages_by_name,
            direct_requirement,
            roots,
            allowed_extras,
            target_python if isinstance(target_python, str) else None,
        )
        if not variants:
            result["missing_dependencies"].append(
                f"profile {profile_id} -> {direct_requirement['name']}"
            )
            continue
        if len(variants) > 1:
            result["conflicting_dependencies"].append(
                f"profile {profile_id} -> {direct_requirement['name']} has "
                f"{len(variants)} lock variants"
            )
        queue.extend((variant, "direct") for variant in variants)
    while queue:
        package, relationship = queue.popleft()
        package_id = package["package_id"]
        previous = seen.get(package_id)
        if previous is None or relationship == "direct":
            seen[package_id] = relationship
        if previous is not None:
            continue
        for dependency in package["dependencies"]:
            marker = dependency.get("marker")
            if not _marker_applies(
                marker if isinstance(marker, str) else None,
                allowed_extras,
                target_python if isinstance(target_python, str) else None,
            ):
                continue
            variants = _package_variants(packages_by_name, dependency)
            if not variants:
                result["missing_dependencies"].append(f"{package['name']} -> {dependency['name']}")
                continue
            if len(variants) > 1:
                result["conflicting_dependencies"].append(
                    f"{package['name']} -> {dependency['name']} has {len(variants)} lock variants"
                )
            queue.extend((variant, "transitive") for variant in variants)

    result["package_ids"] = sorted(seen)
    result["relationships"] = [
        {
            "package_id": package_id,
            "relationship": seen[package_id],
            "originating_extras": sorted(allowed_extras),
        }
        for package_id in sorted(seen)
    ]
    result["missing_dependencies"] = sorted(set(result["missing_dependencies"]))
    result["conflicting_dependencies"] = sorted(set(result["conflicting_dependencies"]))
    result["status"] = (
        "complete"
        if not result["missing_dependencies"] and not result["conflicting_dependencies"]
        else "blocked"
    )
    return result


def _policy_records(  # noqa: C901, PLR0912, PLR0915
    policy: dict[str, Any],
    repo_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[str],
]:
    """Validate policy rules and capture component/package evidence digests."""
    issues: list[str] = []
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        issues.append("dependency policy has an unsupported schema_version")
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        issues.append("dependency policy has no rules")
        rules = []
    by_mode: dict[str, dict[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            issues.append("dependency policy contains a non-object rule")
            continue
        mode = rule.get("distribution_mode")
        if not isinstance(mode, str) or mode in by_mode:
            issues.append(f"dependency policy has duplicate or invalid mode: {mode!r}")
            continue
        by_mode[mode] = rule

    components_out: list[dict[str, Any]] = []
    components = policy.get("components")
    if not isinstance(components, list):
        issues.append("dependency policy has no components list")
        components = []
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("id"), str):
            issues.append("dependency policy contains a component without an id")
            continue
        evidence_digests: list[dict[str, str]] = []
        for value in component.get("evidence_paths", []):
            if not isinstance(value, str):
                issues.append(f"component {component['id']} has a non-string evidence path")
                continue
            path = _resolve_path(repo_root, value)
            if not path.is_file():
                issues.append(f"component {component['id']} evidence is missing: {value}")
                continue
            evidence_digests.append({"path": value, "sha256": _sha256_file(path)})
        record = {
            "id": component["id"],
            "distribution_mode": component.get("distribution_mode"),
            "status": component.get("status"),
            "reviewer": component.get("reviewer"),
            "reviewed_at": component.get("reviewed_at"),
            "license_facts": _normalise_json(component.get("license_facts", {})),
            "evidence": evidence_digests,
            "disposition": component.get("disposition"),
        }
        components_out.append(record)
    package_dispositions = policy.get("package_dispositions")
    if not isinstance(package_dispositions, list):
        issues.append("dependency policy has no package_dispositions list")
        package_dispositions = []
    package_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    package_out: list[dict[str, Any]] = []
    package_ids: set[str] = set()
    valid_modes = {"user_installed", "bundled_source", "built_companion", "not_distributed"}
    valid_conditions = {
        "mirrored",
        "vendored",
        "container_bundled",
        "unknown",
        "unavailable",
        "conflicting",
    }
    for package in package_dispositions:
        if not isinstance(package, dict):
            issues.append("dependency policy contains a non-object package disposition")
            continue
        package_id = package.get("id")
        package_name = package.get("package")
        if not isinstance(package_id, str) or not package_id:
            issues.append("package disposition has no id")
            continue
        if package_id in package_ids:
            issues.append(f"duplicate package disposition id: {package_id}")
        package_ids.add(package_id)
        if not isinstance(package_name, str) or not package_name:
            issues.append(f"package disposition {package_id} has no package name")
            continue
        normalized_name = _canonicalize_name(package_name)
        required_strings = (
            "version",
            "license_expression",
            "python_requires",
            "ruling",
            "rationale",
            "disposition",
        )
        for field in required_strings:
            value = package.get(field)
            if field == "python_requires" and value is None:
                if field not in package:
                    issues.append(f"package disposition {package_id} has no {field}")
                continue
            if not isinstance(value, str) or not value.strip():
                issues.append(f"package disposition {package_id} has no {field}")
        source = package.get("source")
        if not isinstance(source, dict) or not source:
            issues.append(f"package disposition {package_id} has no source")
            source = {}
        metadata_url = source.get("metadata_url")
        if "metadata_url" in source and (
            not isinstance(metadata_url, str) or not metadata_url.strip()
        ):
            issues.append(f"package disposition {package_id} has an invalid metadata_url")
        if "metadata_url" in source and isinstance(package_name, str):
            expected_metadata_url = (
                f"https://pypi.org/pypi/{package_name}/{package.get('version')}/json"
            )
            if metadata_url != expected_metadata_url:
                issues.append(
                    f"package disposition {package_id} metadata_url does not match package identity"
                )
        target = package.get("target")
        if not isinstance(target, dict) or not target:
            issues.append(f"package disposition {package_id} has no target")
            target = {}
        upstream = package.get("upstream")
        if not isinstance(upstream, dict) or not upstream:
            issues.append(f"package disposition {package_id} has no upstream provenance")
            upstream = {}
        notice_paths = upstream.get("notice_paths")
        if (
            not isinstance(notice_paths, list)
            or len(notice_paths) < 2
            or not all(isinstance(value, str) and value for value in notice_paths)
        ):
            issues.append(f"package disposition {package_id} has incomplete notice references")
        if "metadata_url" in source:
            for field in ("archive_notice_paths", "archive_notice_absences"):
                values = upstream.get(field)
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and value for value in values
                ):
                    issues.append(f"package disposition {package_id} has invalid {field}")
            if not upstream.get("archive_notice_paths") and not upstream.get(
                "archive_notice_absences"
            ):
                issues.append(
                    f"package disposition {package_id} has no archive notice presence/absence evidence"
                )
            if package.get("status") == "reviewed":
                for field in ("reviewer", "reviewed_at"):
                    value = package.get(field)
                    if not isinstance(value, str) or not value.strip():
                        issues.append(
                            f"package disposition {package_id} reviewed status requires {field}"
                        )
        artifacts = package.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            issues.append(f"package disposition {package_id} has no artifacts")
            artifacts = []
        artifact_keys: set[tuple[str, str, str]] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                issues.append(f"package disposition {package_id} has a non-object artifact")
                continue
            kind = artifact.get("kind")
            filename = artifact.get("filename")
            sha256 = artifact.get("sha256")
            if not all(isinstance(value, str) and value for value in (kind, filename, sha256)):
                issues.append(f"package disposition {package_id} has an incomplete artifact")
                continue
            key = (kind, filename, sha256)
            if key in artifact_keys:
                issues.append(f"package disposition {package_id} has a duplicate artifact")
            artifact_keys.add(key)
        evidence_paths = package.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            issues.append(f"package disposition {package_id} has no evidence_paths")
            evidence_paths = []
        evidence_digests: list[dict[str, str]] = []
        for value in evidence_paths:
            if not isinstance(value, str):
                issues.append(f"package disposition {package_id} has a non-string evidence path")
                continue
            path = _resolve_path(repo_root, value)
            if not path.is_file():
                issues.append(f"package disposition {package_id} evidence is missing: {value}")
                continue
            evidence_digests.append({"path": value, "sha256": _sha256_file(path)})
        profiles = package.get("profiles")
        if (
            not isinstance(profiles, list)
            or not profiles
            or not all(isinstance(value, str) and value for value in profiles)
        ):
            issues.append(f"package disposition {package_id} has no valid profiles")
            profiles = []
        allowed_modes = package.get("allowed_distribution_modes")
        blocked_modes = package.get("blocked_distribution_modes")
        if not isinstance(allowed_modes, list) or not allowed_modes:
            issues.append(f"package disposition {package_id} has no allowed modes")
            allowed_modes = []
        if not isinstance(blocked_modes, list):
            issues.append(f"package disposition {package_id} has no blocked modes")
            blocked_modes = []
        invalid_modes = {
            str(value) for value in [*allowed_modes, *blocked_modes] if value not in valid_modes
        }
        issues.extend(
            f"package disposition {package_id} has invalid distribution mode: {mode}"
            for mode in sorted(invalid_modes)
        )
        overlap = set(allowed_modes) & set(blocked_modes)
        if overlap:
            issues.append(f"package disposition {package_id} allows and blocks: {sorted(overlap)}")
        blocked_conditions = package.get("blocked_surface_conditions")
        if not isinstance(blocked_conditions, list):
            issues.append(f"package disposition {package_id} has no blocked surface conditions")
            blocked_conditions = []
        invalid_conditions = {
            str(value) for value in blocked_conditions if value not in valid_conditions
        }
        issues.extend(
            f"package disposition {package_id} has invalid blocked surface condition: {condition}"
            for condition in sorted(invalid_conditions)
        )
        record = {
            "id": package_id,
            "package": package_name,
            "version": package.get("version"),
            "license_expression": package.get("license_expression"),
            "python_requires": package.get("python_requires"),
            "target": _normalise_json(target),
            "source": _normalise_json(source),
            "artifacts": _normalise_json(artifacts),
            "upstream": _normalise_json(upstream),
            "profiles": sorted(set(profiles)),
            "allowed_distribution_modes": sorted(set(allowed_modes)),
            "blocked_distribution_modes": sorted(set(blocked_modes)),
            "blocked_surface_conditions": sorted(set(blocked_conditions)),
            "status": package.get("status"),
            "reviewer": package.get("reviewer"),
            "reviewed_at": package.get("reviewed_at"),
            "disposition": package.get("disposition"),
            "ruling": package.get("ruling"),
            "rationale": package.get("rationale"),
            "evidence_paths": sorted(set(evidence_paths)),
            "evidence": evidence_digests,
        }
        package_by_name[normalized_name].append(record)
        package_out.append(record)
    return (
        by_mode,
        sorted(components_out, key=lambda item: item["id"]),
        {
            name: sorted(rows, key=lambda item: item["id"])
            for name, rows in sorted(package_by_name.items())
        },
        sorted(package_out, key=lambda item: item["id"]),
        issues,
    )


def _policy_source_matches(
    package_source: dict[str, Any],
    policy_source: dict[str, Any],
) -> bool:
    """Compare lock source identity while retaining policy metadata provenance.

    The frozen lock records the package index, whereas the policy additionally
    retains the exact PyPI version-response URL used for archive and metadata
    review.  The response URL is evidence about the registry row, not a second
    lock source field, so it must not make an otherwise exact source mismatch.
    """
    lock_source = {key: value for key, value in policy_source.items() if key != "metadata_url"}
    return _normalise_json(package_source) == _normalise_json(lock_source)


def _exact_artifact_failures(
    package: dict[str, Any],
    expected_artifacts: list[dict[str, Any]],
) -> list[str]:
    """Require every reviewed artifact identity to occur in the selected lock row."""
    actual = {
        (artifact.get("kind"), artifact.get("filename"), artifact.get("sha256")): artifact
        for artifact in package.get("artifacts", [])
        if isinstance(artifact, dict)
    }
    failures: list[str] = []
    for expected in expected_artifacts:
        if not isinstance(expected, dict):
            failures.append("exact policy contains a non-object artifact")
            continue
        key = (expected.get("kind"), expected.get("filename"), expected.get("sha256"))
        observed = actual.get(key)
        if observed is None:
            failures.append(
                f"missing exact artifact {expected.get('filename')} with SHA-256 "
                f"{expected.get('sha256')}"
            )
            continue
        for field in ("size", "platform_tags"):
            if field in expected and observed.get(field) != expected.get(field):
                failures.append(
                    f"artifact {expected.get('filename')} {field} differs from exact policy"
                )
    return failures


def _match_package_disposition(
    package: dict[str, Any],
    observation: dict[str, Any],
    mode: str,
    profiles: set[str],
    target: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Match a package row against its exact reviewed disposition, fail-closed."""
    rows = candidates.get(package["normalized_name"], [])
    if not rows:
        return None, []
    policy = rows[0]
    failures: list[str] = []
    if package.get("version") != policy.get("version"):
        failures.append(
            f"lock version {package.get('version')} does not match exact policy "
            f"{policy.get('version')}"
        )
    if not _policy_source_matches(package.get("source", {}), policy.get("source", {})):
        failures.append("lock source/index does not match exact policy")
    expected_target = policy.get("target")
    if isinstance(expected_target, dict) and _normalise_json(expected_target) != _normalise_json(
        {
            "os": target.get("os"),
            "architecture": target.get("architecture"),
            "python": {
                "implementation": target.get("python", {}).get("implementation")
                if isinstance(target.get("python"), dict)
                else None,
                "version": target.get("python", {}).get("version")
                if isinstance(target.get("python"), dict)
                else None,
            },
        }
    ):
        failures.append("profile target does not match exact policy")
    if mode not in policy.get("allowed_distribution_modes", []):
        failures.append(f"distribution mode {mode} is not allowed by exact policy")
    if policy.get("status") != "reviewed":
        failures.append(f"exact policy status is {policy.get('status')}")
    if observation.get(
        "metadata_binding"
    ) != "candidate_sbom_component_identity" and observation.get(
        "license_expression"
    ) != policy.get("license_expression"):
        failures.append("observed license expression does not match exact policy")
    expected_profiles = set(policy.get("profiles", []))
    if not profiles <= expected_profiles:
        failures.append(
            "package profile membership exceeds exact policy: "
            f"{sorted(profiles - expected_profiles)}"
        )
    failures.extend(_exact_artifact_failures(package, policy.get("artifacts", [])))
    return policy, sorted(set(failures))


def _exact_policy_coverage_failures(
    package_dispositions: list[dict[str, Any]],
    package_records: list[dict[str, Any]],
    profile_ids: set[str],
) -> list[str]:
    """Ensure each exact disposition covers exactly its declared profile set."""
    failures: list[str] = []
    for policy in package_dispositions:
        policy_id = policy["id"]
        expected_profiles = set(policy.get("profiles", []))
        unknown_profiles = expected_profiles - profile_ids
        if unknown_profiles:
            failures.append(
                f"package disposition {policy_id} references unknown profiles: "
                f"{sorted(unknown_profiles)}"
            )
        matches = [
            record
            for record in package_records
            if record.get("normalized_name") == _canonicalize_name(policy["package"])
            and record.get("version") == policy.get("version")
            and _policy_source_matches(record.get("source", {}), policy.get("source", {}))
        ]
        actual_profiles = {profile for record in matches for profile in record.get("profiles", [])}
        if not matches:
            failures.append(f"package disposition {policy_id} has no matching lock row")
        elif actual_profiles != expected_profiles:
            failures.append(
                f"package disposition {policy_id} profile coverage differs: "
                f"expected={sorted(expected_profiles)} actual={sorted(actual_profiles)}"
            )
    return sorted(set(failures))


def _distribution_mode(package: dict[str, Any]) -> str:
    """Map lock source/package identity to a release-surface category."""
    name = package["normalized_name"]
    if name == "pyrvo2":
        return "built_companion"
    if name in {"robot-sf", "pysocialforce"}:
        return "bundled_source"
    return "user_installed"


def _package_observation(
    package: dict[str, Any],
    observed: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Join one lock package to installed metadata and return failures."""
    failures: list[str] = []
    matches = observed.get(package["normalized_name"], [])
    record: dict[str, Any] = {
        "observed_version": None,
        "observation_status": "not_installed",
        "metadata_binding": "not_observed",
        "license_status": "unknown",
        "raw_license_metadata": {
            "License-Expression": None,
            "License": None,
            "Classifier": [],
        },
        "license_expression": None,
        "license_classifiers": [],
        "review_reasons": ["locked package is not installed in the captured environment"],
    }
    if not matches:
        failures.append(f"{package['name']}: package is not installed in the captured environment")
        return record, failures
    if len(matches) > 1:
        record.update(
            {
                "observation_status": "duplicate_distribution_name",
                "metadata_binding": "ambiguous_installed_metadata",
                "license_status": "metadata_conflict",
                "review_reasons": ["multiple installed distributions share this normalized name"],
            }
        )
        failures.append(f"{package['name']}: multiple installed distributions share this name")
        return record, failures
    distribution = matches[0]
    observed_version = str(distribution.version)
    record["observed_version"] = observed_version
    record["observation_status"] = "observed"
    record["metadata_binding"] = "installed_distribution_not_artifact_bound"
    if package["version"] is not None and observed_version != package["version"]:
        record["version_match"] = False
        failures.append(
            f"{package['name']}: lock version {package['version']} != observed {observed_version}"
        )
    else:
        record["version_match"] = True
    record.update(_license_record(distribution))
    if record["license_status"] != "spdx_expression":
        failures.append(f"{package['name']}: {record['license_status']} license metadata")
    return record, failures


def _candidate_package_observation(
    package: dict[str, Any],
    *,
    candidate_version: str,
) -> dict[str, Any]:
    """Represent candidate-bound identity without inventing license metadata."""
    return {
        "observed_version": package.get("version") or candidate_version,
        "observation_status": "artifact_bound",
        "metadata_binding": "candidate_sbom_component_identity",
        "license_status": "unknown",
        "raw_license_metadata": {
            "License-Expression": None,
            "License": None,
            "Classifier": [],
        },
        "license_expression": None,
        "license_classifiers": [],
        "review_reasons": [
            "candidate SBOM binds package identity; a reviewed policy must supply license facts"
        ],
    }


def _input_paths(  # noqa: C901, PLR0912
    repo_root: Path,
    manifest: dict[str, Any],
    policy: dict[str, Any],
    profiles: list[dict[str, Any]],
    manifest_path: Path,
    policy_path: Path,
    generator_path: Path | None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Hash all source, schema, lock, and provenance inputs for freshness."""
    values: set[str] = {
        "pyproject.toml",
        _relative_path(repo_root, manifest_path),
        _relative_path(repo_root, policy_path),
    }
    for profile in profiles:
        for key in ("pyproject", "lockfile"):
            value = profile.get(key)
            if isinstance(value, str):
                values.add(value)
        for key in ("provenance_paths",):
            for value in profile.get(key, []):
                if isinstance(value, str):
                    values.add(value)
    for component in policy.get("components", []):
        if isinstance(component, dict):
            values.update(
                value for value in component.get("evidence_paths", []) if isinstance(value, str)
            )
    for package in policy.get("package_dispositions", []):
        if isinstance(package, dict):
            values.update(
                value for value in package.get("evidence_paths", []) if isinstance(value, str)
            )
    for source in (manifest.get("$schema"), policy.get("$schema")):
        if isinstance(source, str) and not source.startswith(("http://", "https://")):
            values.add(source)
    if generator_path is not None and generator_path.is_file():
        try:
            generator_relative = generator_path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            generator_relative = None
        if generator_relative is not None:
            values.add(generator_relative.as_posix())

    inputs: list[dict[str, str]] = []
    issues: list[str] = []
    for relative in sorted(values):
        path = _resolve_path(repo_root, relative)
        if not path.is_file():
            issues.append(f"inventory input is missing: {relative}")
            continue
        inputs.append({"path": relative, "sha256": _sha256_file(path)})
    return inputs, issues


def build_inventory(  # noqa: C901, PLR0912, PLR0915
    repo_root: Path,
    *,
    distributions: Iterable[Any] | None = None,
    profile_manifest_path: Path | None = None,
    policy_path: Path | None = None,
    generator_path: Path | None = None,
    selected_profile_ids: Iterable[str] | None = None,
    candidate_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Build a lock/profile/environment inventory without network or writes.

    By default every declared profile is audited. ``selected_profile_ids`` narrows the
    strict release surface while retaining all declared profiles and lock rows in the
    report. ``candidate_bundle_path`` additionally binds the selected lock closure to
    the exact wheel, sdist, and SBOM admitted by a software-candidate manifest.
    """
    repo_root = repo_root.resolve()
    manifest_path = profile_manifest_path or (repo_root / CANONICAL_PROFILE_MANIFEST)
    policy_file = policy_path or repo_root / CANONICAL_POLICY
    manifest = _read_json(manifest_path)
    policy = _read_json(policy_file)
    root_document = _project_document(repo_root / "pyproject.toml")
    root_project = root_document["project"]
    profiles = [
        profile
        for profile in manifest.get("profiles", [])
        if isinstance(profile, dict) and isinstance(profile.get("id"), str)
    ]
    structural_issues = _validate_manifest(manifest, root_document)
    profile_id_set = {profile["id"] for profile in profiles}
    if selected_profile_ids is None:
        selected_ids = set(profile_id_set)
    else:
        selected_ids = {str(profile_id) for profile_id in selected_profile_ids}
        if not selected_ids:
            structural_issues.append("selected profile surface cannot be empty")
        structural_issues.extend(
            f"selected profile does not exist: {profile_id}"
            for profile_id in sorted(selected_ids - profile_id_set)
        )
        selected_ids &= profile_id_set
    selected_profiles = [profile for profile in profiles if profile["id"] in selected_ids]
    unrepresented_rules, unrepresented_fallback, unrepresented_policy_issues = (
        _validate_unrepresented_policy(manifest)
    )
    structural_issues.extend(unrepresented_policy_issues)
    (
        policy_rules,
        components,
        package_disposition_by_name,
        package_dispositions,
        policy_issues,
    ) = _policy_records(policy, repo_root)
    structural_issues.extend(policy_issues)

    all_packages: dict[str, dict[str, Any]] = {}
    packages_by_lockfile: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        lockfile = profile.get("lockfile")
        if not isinstance(lockfile, str):
            continue
        if lockfile not in packages_by_lockfile:
            path = _resolve_path(repo_root, lockfile)
            try:
                packages_by_lockfile[lockfile] = _lock_packages(path, lockfile)
            except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
                structural_issues.append(f"could not read {lockfile}: {exc}")
                packages_by_lockfile[lockfile] = []
        for package in packages_by_lockfile[lockfile]:
            all_packages.setdefault(package["package_id"], package)

    profile_results: list[dict[str, Any]] = []
    package_profiles: dict[str, set[str]] = defaultdict(set)
    selected_package_profiles: dict[str, set[str]] = defaultdict(set)
    resolution_failures: list[str] = []
    for profile in profiles:
        lockfile = profile.get("lockfile")
        lock_packages = packages_by_lockfile.get(lockfile, []) if isinstance(lockfile, str) else []
        result = _resolve_profile(profile, lock_packages, manifest, repo_root)
        profile_results.append(result)
        for package_id in result.get("package_ids", []):
            package_profiles[package_id].add(result["id"])
            if result["id"] in selected_ids:
                selected_package_profiles[package_id].add(result["id"])
        if result["id"] in selected_ids:
            resolution_failures.extend(
                f"{result['id']}: {failure}"
                for failure in [
                    *result.get("missing_dependencies", []),
                    *result.get("conflicting_dependencies", []),
                ]
            )

    selected_package_ids = set(selected_package_profiles)
    selected_lockfiles = {
        profile["lockfile"]
        for profile in selected_profiles
        if isinstance(profile.get("lockfile"), str)
    }
    candidate_binding: dict[str, Any] | None = None
    if candidate_bundle_path is not None:
        try:
            candidate_binding = _candidate_bundle_binding(
                candidate_bundle_path,
                selected_profile_ids=selected_ids,
                selected_package_ids=selected_package_ids,
                all_packages=all_packages,
                profiles=profiles,
            )
        except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
            candidate_binding = {
                "status": "blocked",
                "failures": [str(exc)],
            }
            structural_issues.append(f"candidate bundle binding failed: {exc}")

    unrepresented_dispositions, unrepresented_context_issues = _classify_unrepresented_packages(
        list(all_packages.values()),
        package_profiles,
        packages_by_lockfile,
        manifest,
        unrepresented_rules,
        unrepresented_fallback,
        repo_root,
    )
    structural_issues.extend(unrepresented_context_issues)
    unrepresented_by_id = {record["package_id"]: record for record in unrepresented_dispositions}
    full_surface_selection = selected_ids == profile_id_set
    for record in unrepresented_dispositions:
        record["surface_membership"] = (
            "selected_profile_closure"
            if record.get("package_id") in selected_package_ids
            else "unresolved_membership"
            if full_surface_selection and record.get("status") == "unresolved"
            else "outside_selected_profiles"
        )
    observed = _observed_distributions(distributions)
    package_records: list[dict[str, Any]] = []
    package_failures: list[str] = []
    policy_failures: list[str] = []
    status_counts: Counter[str] = Counter()
    exact_policy_match_count = 0
    for package_id in sorted(all_packages):
        package = all_packages[package_id]
        mode = _distribution_mode(package)
        rule = policy_rules.get(mode)
        if rule is None:
            structural_issues.append(f"no policy rule for distribution mode: {mode}")
            rule = {"id": "missing-policy-rule", "disposition": "review_required"}
        observation, failures = _package_observation(package, observed)
        package_profile_ids = package_profiles.get(package_id, set())
        selected_package_profile_ids = selected_package_profiles.get(package_id, set())
        if (
            selected_package_profile_ids
            and candidate_binding is not None
            and candidate_binding.get("status") == "bound"
        ):
            observation = _candidate_package_observation(
                package,
                candidate_version=candidate_binding["package"]["version"],
            )
            failures = []
        package_failures.extend(failure for failure in failures if selected_package_profile_ids)
        exact_policy, exact_failures = _match_package_disposition(
            package,
            observation,
            mode,
            selected_package_profile_ids,
            manifest.get("target", {}) if isinstance(manifest.get("target"), dict) else {},
            package_disposition_by_name,
        )
        record = {
            **package,
            **observation,
            "distribution_mode": mode,
            "policy_rule_id": rule.get("id"),
            "policy_disposition": rule.get("disposition"),
            "profiles": sorted(package_profile_ids),
            "selected_profiles": sorted(selected_package_profile_ids),
            "surface_membership": (
                "selected" if selected_package_profile_ids else "outside_selected_profiles"
            ),
            "originating_extras": sorted(
                {
                    extra
                    for profile_result in profile_results
                    if profile_result["id"] in package_profile_ids
                    for extra in profile_result.get("extras", [])
                }
            ),
        }
        if exact_policy is not None:
            record["exact_policy_id"] = exact_policy["id"]
            record["exact_policy_status"] = "accepted" if not exact_failures else "blocked"
            record["exact_policy_disposition"] = exact_policy.get("disposition")
            record["exact_policy_evidence"] = exact_policy.get("evidence", [])
            record["policy_disposition"] = exact_policy.get("disposition")
            if exact_failures and selected_package_profile_ids:
                policy_failures.extend(
                    f"{package['name']}: exact policy {exact_policy['id']}: {failure}"
                    for failure in exact_failures
                )
            elif not exact_failures and selected_package_profile_ids:
                exact_policy_match_count += 1
        if package_id in unrepresented_by_id:
            record["unrepresented_disposition"] = unrepresented_by_id[package_id]
        status_counts[record["license_status"]] += 1
        if (
            selected_package_profile_ids
            and exact_policy is None
            and rule.get("disposition") == "review_required"
        ):
            policy_failures.append(
                f"{package['name']}: {mode} requires an explicit reviewed disposition"
            )
        package_records.append(record)

    policy_failures.extend(
        _exact_policy_coverage_failures(
            package_dispositions,
            package_records,
            {profile["id"] for profile in profiles},
        )
    )

    component_failures = [
        f"component {component['id']}: disposition is {component['disposition']}"
        for component in components
        if component.get("id")
        in {
            component_id
            for profile in selected_profiles
            for component_id in profile.get("components", [])
            if isinstance(component_id, str)
        }
        if component.get("status") != "reviewed" or component.get("disposition") != "approved"
    ]
    unrepresented = [record["package_id"] for record in unrepresented_dispositions]
    unrepresented_failures = [
        f"{record['name']} ({record['lockfile']}): unrepresented lock row has no reviewed exclusion reason"
        for record in unrepresented_dispositions
        if record["status"] == "unresolved"
        and (full_surface_selection or record.get("package_id") in selected_package_ids)
    ]
    unrepresented_reason_counts: Counter[str] = Counter(
        reason_code
        for record in unrepresented_dispositions
        for reason_code in record["reason_codes"]
    )
    unrepresented_reviewed_count = sum(
        record["status"] == "reviewed_exclusion" for record in unrepresented_dispositions
    )
    unrepresented_unresolved_count = sum(
        record["status"] == "unresolved" for record in unrepresented_dispositions
    )
    inputs, input_issues = _input_paths(
        repo_root,
        manifest,
        policy,
        profiles,
        manifest_path,
        policy_file,
        generator_path or Path(__file__).resolve(),
    )
    structural_issues.extend(input_issues)

    failures = sorted(
        {
            *structural_issues,
            *resolution_failures,
            *package_failures,
            *policy_failures,
            *component_failures,
            *unrepresented_failures,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_inputs": inputs,
        "target": _normalise_json(manifest.get("target", {})),
        "profile_manifest": {
            "path": _relative_path(repo_root, manifest_path),
            "schema_version": manifest.get("schema_version"),
            "profile_ids": [profile["id"] for profile in profiles],
        },
        "surface": {
            "profile_ids": sorted(selected_ids),
            "selection": (
                "all_declared_profiles"
                if selected_ids == profile_id_set
                else "explicit_profile_selection"
            ),
            "selected_lockfiles": sorted(selected_lockfiles),
        },
        "policy": {
            "path": _relative_path(repo_root, policy_file),
            "schema_version": policy.get("schema_version"),
            "claim_boundary": policy.get("claim_boundary"),
            "rules": _normalise_json(policy.get("rules", [])),
            "components": components,
            "package_dispositions": package_dispositions,
        },
        "project": {
            "name": root_project.get("name"),
            "license": _normalise_json(root_project.get("license")),
            "distribution_boundary": (
                "dependencies are user-installed unless a package or companion row says otherwise;"
                " policy disposition is not inferred from metadata"
            ),
        },
        "environment": {
            "python": sys.version,
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "candidate_binding": candidate_binding,
        "profiles": profile_results,
        "packages": package_records,
        "unrepresented_lock_packages": unrepresented,
        "unrepresented_lock_package_dispositions": unrepresented_dispositions,
        "installed_not_locked": sorted(
            {
                f"{distribution.metadata.get('Name') or distribution.name}=={distribution.version}"
                for key, values in observed.items()
                if key not in {package["normalized_name"] for package in package_records}
                for distribution in values
            },
            key=str.lower,
        ),
        "structural_issues": sorted(set(structural_issues)),
        "failures": failures,
        "summary": {
            "profile_count": len(profile_results),
            "selected_profile_count": len(selected_profiles),
            "locked_package_count": len(package_records),
            "selected_package_count": len(selected_package_ids),
            "outside_selected_package_count": len(all_packages) - len(selected_package_ids),
            "profile_membership_edge_count": sum(
                len(result.get("package_ids", [])) for result in profile_results
            ),
            "unrepresented_lock_package_count": len(unrepresented),
            "unrepresented_reviewed_exclusion_count": unrepresented_reviewed_count,
            "unrepresented_unresolved_count": unrepresented_unresolved_count,
            "unrepresented_reason_counts": dict(sorted(unrepresented_reason_counts.items())),
            "installed_distribution_count": sum(len(values) for values in observed.values()),
            "installed_not_locked_count": len(
                {
                    key
                    for key in observed
                    if key not in {package["normalized_name"] for package in package_records}
                }
            ),
            "license_status_counts": dict(sorted(status_counts.items())),
            "policy_pending_package_count": len(policy_failures),
            "policy_pending_component_count": len(component_failures),
            "policy_exact_disposition_count": len(package_dispositions),
            "policy_exact_match_count": exact_policy_match_count,
            "candidate_bound": candidate_binding is not None
            and candidate_binding.get("status") == "bound",
            "structural_issue_count": len(set(structural_issues)),
            "unresolved_count": len(failures),
            "status": "blocked" if failures else "complete",
        },
    }


def check_report_freshness(  # noqa: C901
    repo_root: Path,
    report_path: Path,
    *,
    candidate_bundle_path: Path | None = None,
) -> list[str]:
    """Recompute every recorded input digest for an existing report.

    Freshness also binds the report to the canonical profile manifest and policy. A report
    generated with ``--profile-manifest``/``--policy`` pointing at a substitute file records
    that substitute instead, so without this binding a relaxed policy could produce a report
    that still validates as fresh.
    """
    report = _read_json(report_path)
    issues: list[str] = []
    inputs = report.get("repository_inputs")
    if not isinstance(inputs, list) or not inputs:
        return ["report has no repository_inputs digest list"]
    recorded_paths = {item.get("path") for item in inputs if isinstance(item, dict)}
    issues.extend(
        f"report was not generated from the canonical input: {canonical}"
        for canonical in (CANONICAL_PROFILE_MANIFEST, CANONICAL_POLICY)
        if canonical not in recorded_paths
    )
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            issues.append("report contains an invalid repository input row")
            continue
        path = _resolve_path(repo_root, item["path"])
        if not path.is_file():
            issues.append(f"freshness input is missing: {item['path']}")
            continue
        actual = _sha256_file(path)
        if actual != item.get("sha256"):
            issues.append(
                f"freshness digest mismatch for {item['path']}: "
                f"report={item.get('sha256')} actual={actual}"
            )
    recorded_candidate = report.get("candidate_binding")
    if recorded_candidate is not None:
        if not isinstance(recorded_candidate, dict):
            issues.append("report contains an invalid candidate_binding record")
        elif candidate_bundle_path is None:
            issues.append("candidate-bound report freshness requires --candidate-bundle")
        else:
            surface = report.get("surface")
            profile_ids = surface.get("profile_ids") if isinstance(surface, dict) else None
            if not isinstance(profile_ids, list) or not all(
                isinstance(profile_id, str) for profile_id in profile_ids
            ):
                issues.append("candidate-bound report has no valid selected profile surface")
            else:
                try:
                    current = build_inventory(
                        repo_root,
                        distributions=[],
                        selected_profile_ids=profile_ids,
                        candidate_bundle_path=candidate_bundle_path,
                    )
                except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
                    issues.append(f"candidate bundle freshness could not be checked: {exc}")
                else:
                    if current.get("candidate_binding") != recorded_candidate:
                        issues.append("candidate bundle binding differs from the recorded report")
    elif candidate_bundle_path is not None:
        issues.append("candidate bundle supplied for a report without candidate_binding")
    return sorted(set(issues))


def _reported_unresolved_count(report_path: Path) -> int:
    """Return an existing report's recorded unresolved-row count."""
    summary = _read_json(report_path).get("summary")
    value = summary.get("unresolved_count") if isinstance(summary, dict) else None
    return value if isinstance(value, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run inventory generation or freshness validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the profile, policy, and lock inputs.",
    )
    parser.add_argument("--output", type=Path, help="Write the generated JSON report to this path.")
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        help="Profile manifest path, relative to --repo-root by default.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="Disposition policy path, relative to --repo-root by default.",
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        help=(
            "Audit one declared release profile (repeat for a union of profiles); "
            "all profiles are retained as visible context."
        ),
    )
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        help=(
            "Bind the selected profile closure to an exact software-candidate bundle "
            "containing a wheel, sdist, provenance, and CycloneDX SBOM."
        ),
    )
    parser.add_argument(
        "--check-freshness",
        type=Path,
        help="Check an existing report's recorded input digests and do not regenerate it.",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Return exit code 2 when metadata, profile, provenance, or policy rows remain unresolved.",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    candidate_bundle_path = (
        _resolve_path(repo_root, args.candidate_bundle) if args.candidate_bundle else None
    )

    try:
        if args.check_freshness:
            report_path = args.check_freshness.resolve()
            issues = check_report_freshness(
                repo_root,
                report_path,
                candidate_bundle_path=candidate_bundle_path,
            )
            unresolved = _reported_unresolved_count(report_path)
            print(
                json.dumps(
                    {
                        "schema_version": "dependency_license_freshness.v1",
                        "issues": issues,
                        "unresolved_count": unresolved,
                    },
                    indent=2,
                )
            )
            if issues:
                return 1
            if args.fail_on_unresolved and unresolved:
                print(
                    f"FAIL: dependency license inventory remains blocked for {unresolved} row(s)",
                    file=sys.stderr,
                )
                return 2
            return 0
        inventory = build_inventory(
            repo_root,
            profile_manifest_path=(
                _resolve_path(repo_root, args.profile_manifest) if args.profile_manifest else None
            ),
            policy_path=_resolve_path(repo_root, args.policy) if args.policy else None,
            selected_profile_ids=args.profiles,
            candidate_bundle_path=candidate_bundle_path,
        )
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"FAIL: dependency license inventory could not be built: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")
    if args.fail_on_unresolved and inventory["summary"]["unresolved_count"]:
        print(
            "FAIL: dependency license inventory remains blocked for "
            f"{inventory['summary']['unresolved_count']} row(s)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
