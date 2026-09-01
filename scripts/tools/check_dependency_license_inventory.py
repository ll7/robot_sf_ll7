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

# evidence-writer-exempt: this standalone CLI writes only local/CI output outside
# docs/context/evidence and must preserve its existing byte-stable report contract;
# durable checked-in evidence uses the shared writer and review sidecar.

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
CANONICAL_GENERATOR = "scripts/tools/check_dependency_license_inventory.py"
PROFILE_SCHEMA_VERSION = "robot-sf.dependency-license-profiles.v1"
UNREPRESENTED_POLICY_SCHEMA_VERSION = "robot-sf.dependency-license-unrepresented.v1"
POLICY_SCHEMA_VERSION = "robot-sf.dependency-license-policy.v1"
_UNKNOWN_VALUES = frozenset({"", "unknown", "unknown license", "none", "null"})
_REVIEW_MARKER_JSON = "AI-GENERATED NEEDS-REVIEW"
_REVIEW_MARKERS = (
    "licenseref-",
    "proprietary",
    "nvidia",
    "non-commercial",
    "non commercial",
    "no redistribution",
)
_REPORT_CONTENT_DIGEST_FIELD = "report_content_sha256"
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
_UPSTREAM_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
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
# This is the v0.0.6 public software surface. The checked-in ``all`` profile is
# the reviewed closure of these twelve extras; keep both lists stable because
# they are emitted into the rights receipt and form the cross-workflow contract.
SUPPORTED_SOFTWARE_CANDIDATE_PROFILE_IDS = ("all",)
SUPPORTED_SOFTWARE_CANDIDATE_EXTRA_IDS = (
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
)
SUPPORTED_SOFTWARE_CANDIDATE_DISTRIBUTION_EXTRA_IDS = (
    *SUPPORTED_SOFTWARE_CANDIDATE_EXTRA_IDS,
    "all",
)
# The source checkout declares this development-only extra, while the
# rights-clean materialization removes it from the copied project metadata.
# Keep the exclusion explicit in the profile manifest without treating the
# candidate's absence of ``rllib`` as an unresolved all-profile mismatch.
SUPPORTED_SOFTWARE_CANDIDATE_EXCLUDED_EXTRA_IDS = frozenset({"rllib"})
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


def _report_content_digest(report: dict[str, Any]) -> str:
    """Hash report content while excluding only the marker and this digest."""
    content = {
        key: value
        for key, value in report.items()
        if key not in {"review_marker", _REPORT_CONTENT_DIGEST_FIELD}
    }
    return _sha256_value(content)


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


def _github_notice_reference(url: str) -> tuple[str, str, str] | None:
    """Return ``(repository, ref, path_kind)`` for a GitHub tree/blob URL.

    Release evidence must point at an immutable commit rather than a branch or
    tag which can be moved after the evidence was reviewed.  This parser is
    intentionally narrow: only HTTPS github.com URLs using the public
    ``blob/<commit>/<path>`` or ``tree/<commit>`` forms are accepted.
    """
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
    ):
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 4 or parts[2] not in {"blob", "tree"}:
        return None
    ref = parts[3]
    if _UPSTREAM_COMMIT_SHA_RE.fullmatch(ref) is None:
        return None
    if parts[2] == "blob" and len(parts) < 5:
        return None
    return "/".join(parts[:2]), ref, parts[2]


def _effective_profile_coverage(actual: set[str], expected: set[str]) -> set[str]:
    """Collapse transitive memberships covered by an aggregate ``all`` row.

    ``all`` represents one declared release profile, not every independent
    lockfile profile.  When a policy row includes ``all``, memberships that are
    not explicitly named by that policy are therefore treated as transitive
    context.  Explicit standalone memberships (for example ``fast-pysf`` on
    the existing llvmlite control row) remain visible and exact.
    """
    if "all" not in expected or "all" not in actual:
        return set(actual)
    explicit = expected - {"all"}
    return {"all"} | (actual & explicit)


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
    extras = dependency.get("extras", dependency.get("extra"))
    if isinstance(extras, str):
        extras = [extras]
    if isinstance(extras, list):
        entry["extras"] = sorted(
            value for value in extras if isinstance(value, str) and value.strip()
        )
    for key in ("marker", "version"):
        if isinstance(dependency.get(key), str):
            entry[key] = dependency[key]
    if isinstance(dependency.get("source"), dict):
        entry["source"] = _normalise_json(dependency["source"])
    return entry


def _lock_packages(path: Path, repo_relative_path: str) -> list[dict[str, Any]]:  # noqa: C901
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
        optional_dependencies: dict[str, list[dict[str, Any]]] = {}
        raw_optional_dependencies = package.get("optional-dependencies")
        if isinstance(raw_optional_dependencies, dict):
            for extra, raw_edges in raw_optional_dependencies.items():
                if not isinstance(extra, str) or not isinstance(raw_edges, list):
                    continue
                edges = [
                    entry
                    for raw_edge in raw_edges
                    if (entry := _dependency_record(raw_edge)) is not None
                ]
                optional_dependencies[extra.casefold()] = edges
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
                "optional_dependencies": optional_dependencies,
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
    target: dict[str, Any],
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
    lock_rows_by_component: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for package_row in all_packages.values():
        version = package_row.get("version")
        if isinstance(version, str) and version:
            lock_rows_by_component[(package_row["normalized_name"], version)].append(package_row)
    target_environment = _target_marker_environment({"target": target})
    inactive_components = sorted(
        identity
        for identity in actual_components - expected_components
        if lock_rows_by_component.get(identity)
        and all(
            _resolution_marker_state(package_row, target_environment) is False
            for package_row in lock_rows_by_component[identity]
        )
    )
    unexpected_components = sorted(
        actual_components - expected_components - set(inactive_components)
    )
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
    if selected_profile_ids == set(SUPPORTED_SOFTWARE_CANDIDATE_PROFILE_IDS):
        unsupported_advertised_extras = provided_extras - set(
            SUPPORTED_SOFTWARE_CANDIDATE_DISTRIBUTION_EXTRA_IDS
        )
        missing_supported_extras = (
            set(SUPPORTED_SOFTWARE_CANDIDATE_DISTRIBUTION_EXTRA_IDS) - provided_extras
        )
        if missing_supported_extras or unsupported_advertised_extras:
            raise ValueError(
                "candidate archives do not match the closed v0.0.6 supported extra roster "
                f"(missing={sorted(missing_supported_extras) or 'none'}, "
                f"unsupported={sorted(unsupported_advertised_extras) or 'none'})"
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
        "members": [bound_members[kind] for kind in _CANDIDATE_MEMBER_KINDS],
        "archives": {"wheel": wheel_metadata, "sdist": sdist_metadata},
        "sbom": {
            "filename": bound_members["sbom"]["filename"],
            "sha256": bound_members["sbom"]["sha256"],
            "component_count": len(actual_components),
            "component_set_sha256": _sha256_value(
                [f"{name}@{version}" for name, version in sorted(actual_components)]
            ),
            "target_inactive_components": [
                f"{name}@{version}" for name, version in inactive_components
            ],
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
        undeclared_exclusions = (
            exclusions - declared - set(SUPPORTED_SOFTWARE_CANDIDATE_EXCLUDED_EXTRA_IDS)
        )
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


def _dependency_context_closure(  # noqa: C901
    lock_packages: list[dict[str, Any]],
    direct_dependencies: list[dict[str, Any]],
    environment: dict[str, str],
) -> set[str]:
    """Resolve one reviewed non-release group conservatively from a lock."""
    packages_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for package in lock_packages:
        packages_by_name[package["normalized_name"]].append(package)
    queue: deque[tuple[dict[str, Any], frozenset[str]]] = deque()
    for dependency in direct_dependencies:
        marker_state = _marker_state(dependency.get("marker"), set(), environment)
        if marker_state is False:
            continue
        dependency_extras = frozenset(
            value.casefold()
            for value in dependency.get("extras", [])
            if isinstance(value, str) and value.strip()
        )
        queue.extend(
            (variant, dependency_extras)
            for variant in _package_variants(packages_by_name, dependency)
            if _resolution_marker_state(variant, environment) is not False
        )
    seen: set[str] = set()
    processed_extras: dict[str, set[str]] = defaultdict(set)
    while queue:
        package, requested_extras = queue.popleft()
        package_id = package["package_id"]
        new_extras = requested_extras - processed_extras[package_id]
        if package_id in seen and not new_extras:
            continue
        seen.add(package_id)
        processed_extras[package_id].update(requested_extras)
        dependencies = list(package["dependencies"])
        optional_dependencies = package.get("optional_dependencies", {})
        if isinstance(optional_dependencies, dict):
            for extra in requested_extras:
                edges = optional_dependencies.get(extra)
                if isinstance(edges, list):
                    dependencies.extend(edges)
        for dependency in dependencies:
            marker_state = _marker_state(
                dependency.get("marker"), set(requested_extras), environment
            )
            if marker_state is False:
                continue
            dependency_extras = frozenset(
                value.casefold()
                for value in dependency.get("extras", [])
                if isinstance(value, str) and value.strip()
            )
            queue.extend(
                (variant, dependency_extras)
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


def _resolve_profile(  # noqa: C901, PLR0912, PLR0915
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
    queue: deque[tuple[dict[str, Any], str, frozenset[str]]] = deque(
        (root, "direct", frozenset()) for root in roots
    )
    seen: dict[str, str] = {}
    processed_extras: dict[str, set[str]] = defaultdict(set)
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
        requested_extras = frozenset(
            value.casefold()
            for value in direct_requirement.get("extras", [])
            if isinstance(value, str) and value.strip()
        )
        queue.extend((variant, "direct", requested_extras) for variant in variants)
    while queue:
        package, relationship, requested_extras = queue.popleft()
        package_id = package["package_id"]
        previous = seen.get(package_id)
        if previous is None or relationship == "direct":
            seen[package_id] = relationship
        new_extras = requested_extras - processed_extras[package_id]
        if previous is not None and not new_extras:
            continue
        processed_extras[package_id].update(requested_extras)
        marker_extras = set(requested_extras) or allowed_extras
        dependencies = list(package["dependencies"])
        optional_dependencies = package.get("optional_dependencies", {})
        if isinstance(optional_dependencies, dict):
            for extra in requested_extras:
                edges = optional_dependencies.get(extra)
                if isinstance(edges, list):
                    dependencies.extend(edges)
        for dependency in dependencies:
            marker = dependency.get("marker")
            if not _marker_applies(
                marker if isinstance(marker, str) else None,
                marker_extras,
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
            dependency_extras = frozenset(
                value.casefold()
                for value in dependency.get("extras", [])
                if isinstance(value, str) and value.strip()
            )
            queue.extend((variant, "transitive", dependency_extras) for variant in variants)

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
    package_identity_keys: dict[tuple[str, str, str], str] = {}
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
        identity_key = (
            normalized_name,
            str(package.get("version")),
            _canonical_json(
                _normalise_json(
                    {key: value for key, value in source.items() if key != "metadata_url"}
                )
            ),
        )
        previous_id = package_identity_keys.get(identity_key)
        if previous_id is not None:
            issues.append(
                "duplicate package disposition identity: "
                f"{package_id} duplicates {previous_id} for {package_name}"
            )
        else:
            package_identity_keys[identity_key] = package_id
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
            notice_paths = []
        evidence_blockers = package.get("evidence_blockers", [])
        if not isinstance(evidence_blockers, list) or not all(
            isinstance(value, str) and value.strip() for value in evidence_blockers
        ):
            issues.append(f"package disposition {package_id} has invalid evidence_blockers")
            evidence_blockers = []
        commit_sha = upstream.get("commit_sha")
        if not isinstance(commit_sha, str) or _UPSTREAM_COMMIT_SHA_RE.fullmatch(commit_sha) is None:
            issues.append(
                f"package disposition {package_id} must bind upstream provenance to a 40-digit commit_sha"
            )
            commit_sha = None
        moving_notice_paths = [
            value
            for value in notice_paths
            if isinstance(value, str) and _github_notice_reference(value) is None
        ]
        if moving_notice_paths:
            if package.get("status") == "reviewed":
                issues.append(
                    f"package disposition {package_id} reviewed evidence contains moving or unversioned notice URLs"
                )
            elif not evidence_blockers or not any(
                any(
                    marker in blocker.lower()
                    for marker in (
                        "moving",
                        "unversioned",
                        "unpinned",
                        "not immutable",
                        "unresolved",
                    )
                )
                for blocker in evidence_blockers
            ):
                issues.append(
                    f"package disposition {package_id} must record a durable blocker for moving notice URLs"
                )
        if commit_sha is not None:
            repository = upstream.get("repository")
            repository_name = (
                repository.removeprefix("https://github.com/")
                if isinstance(repository, str)
                else None
            )
            for notice_path in notice_paths:
                reference = (
                    _github_notice_reference(notice_path) if isinstance(notice_path, str) else None
                )
                if reference is not None and reference[0] == repository_name:
                    if reference[1] != commit_sha:
                        issues.append(
                            f"package disposition {package_id} notice URL does not match upstream commit_sha"
                        )
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
            "evidence_blockers": sorted(set(evidence_blockers)),
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


def _policy_identity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the digest-safe normalized policy records.

    ``evidence`` contains byte digests for the files named by
    ``evidence_paths``.  The batch receipt is itself one of those evidence
    paths, so including that self-referential digest would make a receipt
    impossible to bind deterministically.  The normalized record identity
    therefore retains the declared paths and excludes only their derived byte
    digest map; the receipt binds the evidence bytes separately.
    """
    return [
        {key: value for key, value in record.items() if key != "evidence"}
        for record in sorted(records, key=lambda item: item.get("id", ""))
    ]


def _issue_8163_license_files(policy: dict[str, Any]) -> list[dict[str, str]]:
    """Return the deterministic archive-license path manifest for the batch."""
    rows = [
        row
        for row in policy.get("package_dispositions", [])
        if isinstance(row, dict)
        and "docs/context/evidence/dependency_license_batch_2026-09-01.md"
        in row.get("evidence_paths", [])
    ]
    values = [
        {
            "package": row["package"],
            "version": row["version"],
            "archive_path": archive_path,
        }
        for row in rows
        for archive_path in row.get("upstream", {}).get("archive_notice_paths", [])
        if isinstance(archive_path, str)
    ]
    return sorted(values, key=lambda item: (item["package"], item["version"], item["archive_path"]))


def _issue_8163_receipt_binding(
    policy: dict[str, Any], repo_root: Path
) -> tuple[dict[str, Any], list[str]]:
    """Compute deterministic policy/license bindings used by the batch receipt."""
    _rules, _components, _by_name, records, policy_issues = _policy_records(policy, repo_root)
    batch_records = [
        record
        for record in records
        if "docs/context/evidence/dependency_license_batch_2026-09-01.md"
        in record.get("evidence_paths", [])
    ]
    license_files = _issue_8163_license_files(policy)
    evidence_files = []
    evidence_path = repo_root / "docs/context/evidence/dependency_license_batch_2026-09-01.md"
    if evidence_path.is_file():
        evidence_files.append(
            {
                "path": "docs/context/evidence/dependency_license_batch_2026-09-01.md",
                "sha256": _sha256_file(evidence_path),
            }
        )
    binding = {
        "normalized_records_sha256": _sha256_value(_policy_identity_records(batch_records)),
        "normalized_record_count": len(batch_records),
        "license_files": license_files,
        "license_files_sha256": _sha256_value(license_files),
        "evidence_files": evidence_files,
    }
    return binding, policy_issues


def _receipt_path(repo_root: Path, value: str) -> Path:
    """Resolve a receipt path after removing its operator-local marker."""
    raw = value.removeprefix("operator-local:")
    path = Path(raw)
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"receipt path contains lexical traversal: {value}")
    return path if path.is_absolute() else repo_root / path


def _receipt_path_has_symlink(path: Path) -> bool:
    """Return whether a receipt path or one of its existing parents is a symlink."""
    return any(parent.is_symlink() for parent in (path, *path.parents) if parent.exists())


def _audit_identity(name: Any, version: Any) -> tuple[str, Any]:
    """Build a hashable audit identity even for malformed parseable JSON values."""
    safe_name = _canonicalize_name(name) if isinstance(name, str) else "<invalid-name>"
    try:
        hash(version)
    except TypeError:
        version = repr(version)
    return safe_name, version


def _policy_batch_rows(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the exact Issue #8163 policy rows represented by the receipt."""
    return [
        row
        for row in policy.get("package_dispositions", [])
        if isinstance(row, dict)
        and isinstance(row.get("evidence_paths"), list)
        and "docs/context/evidence/dependency_license_batch_2026-09-01.md"
        in row.get("evidence_paths", [])
    ]


def _archive_identity(artifact: dict[str, Any]) -> tuple[str, str, str, int]:
    """Return the immutable archive identity fields used by policy binding."""
    return (
        artifact.get("kind") if isinstance(artifact.get("kind"), str) else "<invalid-kind>",
        artifact.get("filename")
        if isinstance(artifact.get("filename"), str)
        else "<invalid-filename>",
        artifact.get("sha256") if isinstance(artifact.get("sha256"), str) else "<invalid-sha256>",
        int(artifact.get("size", -1)) if isinstance(artifact.get("size"), int) else -1,
    )


def _archive_notice_paths(package: dict[str, Any]) -> set[str]:
    """Collect package/archive notice paths from either supported audit layout."""
    paths = {path for path in package.get("archive_notice_paths", []) if isinstance(path, str)}
    for artifact in package.get("artifacts", []):
        if isinstance(artifact, dict):
            paths.update(path for path in artifact.get("notice_paths", []) if isinstance(path, str))
    return paths


def _policy_archive_notice_mapping(
    policy_row: dict[str, Any],
) -> set[tuple[str, str, str, str]]:
    """Derive exact archive-kind/upstream-URL mappings from one policy row."""
    immutable_urls = [
        (url, _upstream_notice_path(url))
        for url in policy_row.get("upstream", {}).get("notice_paths", [])
        if isinstance(url, str) and _upstream_notice_path(url) is not None
    ]
    mappings: set[tuple[str, str, str, str]] = set()
    sdist_stems = {
        str(artifact.get("filename", "")).removesuffix(".tar.gz")
        for artifact in policy_row.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("kind") == "sdist"
    }
    for archive_path in policy_row.get("upstream", {}).get("archive_notice_paths", []):
        if not isinstance(archive_path, str):
            continue
        kind = (
            "sdist" if any(archive_path.startswith(f"{stem}/") for stem in sdist_stems) else "wheel"
        )
        matching_urls = [
            (url, upstream_path)
            for url, upstream_path in immutable_urls
            if archive_path.endswith(f"/{upstream_path}") or archive_path.endswith(upstream_path)
        ]
        if matching_urls:
            # A broad ``LICENSE`` URL must not also claim a nested vendored notice.
            longest_path = max(len(upstream_path) for _url, upstream_path in matching_urls)
            mappings.update(
                (archive_path, kind, upstream_path, url)
                for url, upstream_path in matching_urls
                if len(upstream_path) == longest_path
            )
    return mappings


def _policy_archive_notice_kinds(policy_row: dict[str, Any]) -> set[tuple[str, str]]:
    """Derive exact artifact-kind bindings for every declared archive notice."""
    sdist_stems = {
        str(artifact.get("filename", "")).removesuffix(".tar.gz")
        for artifact in policy_row.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("kind") == "sdist"
    }
    return {
        (
            path,
            "sdist" if any(path.startswith(f"{stem}/") for stem in sdist_stems) else "wheel",
        )
        for path in policy_row.get("upstream", {}).get("archive_notice_paths", [])
        if isinstance(path, str)
    }


def _archive_artifact_issues(
    package: dict[str, Any],
    policy_row: dict[str, Any],
    identity: tuple[str, Any],
) -> list[str]:
    """Validate nested archive artifact schemas and authoritative policy fields."""
    issues: list[str] = []
    artifacts = package.get("artifacts")
    expected_artifacts = policy_row.get("artifacts", [])
    if not isinstance(artifacts, list):
        return [f"dependency archive audit artifacts are missing for {identity[0]}"]
    allowed_keys = {
        "kind",
        "filename",
        "url",
        "sha256",
        "size",
        "platform_tags",
        "archive_path",
        "member_count",
        "notice_paths",
        "metadata_path",
        "metadata_license",
        "metadata_project_urls",
        "metadata_fields",
    }
    authoritative = ("kind", "filename", "sha256", "size", "platform_tags")
    actual_ids: set[tuple[str, str, str, int]] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            issues.append(f"dependency archive audit artifact {index} is not an object")
            continue
        if set(artifact) - allowed_keys:
            issues.append(f"dependency archive audit artifact {index} has unclassified fields")
        for field in authoritative:
            if artifact.get(field) != next(
                (
                    expected.get(field)
                    for expected in expected_artifacts
                    if isinstance(expected, dict)
                    and expected.get("kind") == artifact.get("kind")
                    and expected.get("filename") == artifact.get("filename")
                ),
                None,
            ):
                issues.append(
                    f"dependency archive audit artifact {index} {field} differs from policy"
                )
        actual_ids.add(_archive_identity(artifact))
    expected_ids = {
        _archive_identity(item) for item in expected_artifacts if isinstance(item, dict)
    }
    if actual_ids != expected_ids or len(actual_ids) != len(artifacts):
        issues.append(f"dependency archive audit artifact identities differ for {identity[0]}")
    actual_mappings = {
        (path, str(artifact.get("kind")))
        for artifact in artifacts
        if isinstance(artifact, dict)
        for path in artifact.get("notice_paths", [])
        if isinstance(path, str)
    }
    expected_mappings = _policy_archive_notice_kinds(policy_row)
    if actual_mappings != expected_mappings:
        issues.append(f"dependency archive audit notice mappings differ for {identity[0]}")
    return issues


def _archive_audit_semantic_issues(  # noqa: C901, PLR0912, PLR0915
    archive_file: dict[str, Any],
    batch_rows: list[dict[str, Any]],
) -> list[str]:
    """Bind archive-audit rows and artifact identities to the checked-in policy."""
    issues: list[str] = []
    if set(archive_file) != {"schema_version", "packages", "failures"}:
        issues.append("dependency archive audit has missing or unclassified top-level fields")
    packages = archive_file.get("packages")
    if not isinstance(packages, list):
        return [*issues, "dependency archive audit has no package rows"]
    expected_by_identity = {
        _audit_identity(row.get("package"), row.get("version")): row for row in batch_rows
    }
    observed_identities: set[tuple[str, Any]] = set()
    if len(packages) != len(batch_rows):
        issues.append("dependency archive audit package rows differ from the canonical policy")
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            issues.append(f"dependency archive audit package row {index} is not an object")
            continue
        identity = _audit_identity(package.get("name"), package.get("version"))
        if identity in observed_identities:
            issues.append(f"dependency archive audit has duplicate package identity {identity}")
        observed_identities.add(identity)
        policy_row = expected_by_identity.get(identity)
        if policy_row is None:
            issues.append(f"dependency archive audit has unexpected package identity {identity}")
            continue
        expected_package_keys = {
            "name",
            "version",
            "expected_expression",
            "source",
            "pypi_metadata_url",
            "pypi_info",
            "artifacts",
        }
        if set(package) != expected_package_keys:
            issues.append(f"dependency archive audit package {identity[0]} has unclassified fields")
        if not isinstance(package.get("name"), str) or _canonicalize_name(
            package["name"]
        ) != _canonicalize_name(policy_row.get("package")):
            issues.append(f"dependency archive audit package name differs for {identity[0]}")
        if package.get("expected_expression") != policy_row.get("license_expression"):
            issues.append(f"dependency archive audit license expression differs for {identity[0]}")
        expected_source = {
            key: value
            for key, value in policy_row.get("source", {}).items()
            if key != "metadata_url"
        }
        if package.get("source") != expected_source:
            issues.append(f"dependency archive audit source differs for {identity[0]}")
        expected_url = (
            policy_row.get("source", {}).get("metadata_url")
            or f"https://pypi.org/pypi/{policy_row.get('package')}/{policy_row.get('version')}/json"
        )
        if package.get("pypi_metadata_url") != expected_url:
            issues.append(f"dependency archive audit metadata URL differs for {identity[0]}")
        pypi_info = package.get("pypi_info")
        expected_pypi_keys = {
            "name",
            "version",
            "requires_python",
            "license",
            "classifiers",
            "home_page",
            "project_urls",
        }
        if not isinstance(pypi_info, dict) or set(pypi_info) != expected_pypi_keys:
            issues.append(
                f"dependency archive audit PyPI metadata schema is invalid for {identity[0]}"
            )
        if (
            not isinstance(pypi_info, dict)
            or not isinstance(pypi_info.get("name"), str)
            or _canonicalize_name(pypi_info["name"])
            != _canonicalize_name(policy_row.get("package"))
        ):
            issues.append(f"dependency archive audit PyPI identity is invalid for {identity[0]}")
        if not isinstance(pypi_info, dict) or pypi_info.get("version") != policy_row.get("version"):
            issues.append(f"dependency archive audit PyPI version is invalid for {identity[0]}")
        if isinstance(pypi_info, dict):
            if not isinstance(pypi_info.get("name"), str) or not isinstance(
                pypi_info.get("version"), str
            ):
                issues.append(
                    f"dependency archive audit PyPI identity types are invalid for {identity[0]}"
                )
            requires_python = pypi_info.get("requires_python")
            if requires_python != policy_row.get("python_requires"):
                issues.append(
                    f"dependency archive audit PyPI requires_python differs for {identity[0]}"
                )
            if requires_python is not None and not isinstance(requires_python, str):
                issues.append(
                    f"dependency archive audit PyPI requires_python type is invalid for {identity[0]}"
                )
            if pypi_info.get("license") is not None and not isinstance(
                pypi_info.get("license"), str
            ):
                issues.append(
                    f"dependency archive audit PyPI license type is invalid for {identity[0]}"
                )
            if not isinstance(pypi_info.get("classifiers"), list) or not all(
                isinstance(item, str) for item in pypi_info["classifiers"]
            ):
                issues.append(
                    f"dependency archive audit PyPI classifiers are invalid for {identity[0]}"
                )
            if pypi_info.get("home_page") is not None and not isinstance(
                pypi_info.get("home_page"), str
            ):
                issues.append(
                    f"dependency archive audit PyPI home_page type is invalid for {identity[0]}"
                )
            if not isinstance(pypi_info.get("project_urls"), dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in pypi_info["project_urls"].items()
            ):
                issues.append(
                    f"dependency archive audit PyPI project_urls are invalid for {identity[0]}"
                )
        issues.extend(_archive_artifact_issues(package, policy_row, identity))
        expected_notice_paths = {
            path
            for path in policy_row.get("upstream", {}).get("archive_notice_paths", [])
            if isinstance(path, str)
        }
        if _archive_notice_paths(package) != expected_notice_paths:
            issues.append(f"dependency archive audit notice paths differ for {identity[0]}")
        expected_absences = policy_row.get("upstream", {}).get("archive_notice_absences", [])
        if expected_absences:
            issues.append(
                f"dependency archive audit notice absences are not supported for {identity[0]}"
            )
    if observed_identities != set(expected_by_identity):
        issues.append("dependency archive audit identities are not an exact policy set")
    return issues


def _upstream_notice_path(url: Any) -> str | None:
    """Extract the immutable upstream path from a GitHub blob URL."""
    if not isinstance(url, str):
        return None
    reference = _github_notice_reference(url)
    if reference is None:
        return None
    parts = [unquote(part) for part in urlsplit(url).path.split("/") if part]
    return "/".join(parts[4:]) if len(parts) > 4 else None


def _upstream_tags_semantic_issues(  # noqa: C901, PLR0912, PLR0915
    tags_file: Any,
    batch_rows: list[dict[str, Any]],
    archive_file: dict[str, Any] | None,
) -> list[str]:
    """Bind upstream tag and notice checks to policy commits and archive rows."""
    issues: list[str] = []
    if not isinstance(tags_file, list):
        return ["dependency upstream tags evidence must be an array"]
    expected_by_identity = {
        _audit_identity(row.get("package"), row.get("version")): row for row in batch_rows
    }
    observed: set[tuple[str, Any]] = set()
    archive_by_identity = {
        _audit_identity(row.get("name"), row.get("version")): row
        for row in (archive_file or {}).get("packages", [])
        if isinstance(row, dict)
    }
    if len(tags_file) != len(batch_rows):
        issues.append("dependency upstream tags rows differ from the canonical policy")
    for index, entry in enumerate(tags_file):
        if not isinstance(entry, dict):
            issues.append(f"dependency upstream tags row {index} is not an object")
            continue
        expected_tag_keys = {
            "name",
            "version",
            "repository",
            "tags",
            "tag",
            "matching_tags",
            "errors",
            "source_url_key",
            "notice_checks",
        }
        if set(entry) != expected_tag_keys:
            issues.append(f"dependency upstream tags row {index} has unclassified fields")
        identity = _audit_identity(entry.get("name"), entry.get("version"))
        if identity in observed:
            issues.append(f"dependency upstream tags has duplicate package identity {identity}")
        observed.add(identity)
        policy_row = expected_by_identity.get(identity)
        if policy_row is None:
            issues.append(f"dependency upstream tags has unexpected package identity {identity}")
            continue
        upstream = policy_row.get("upstream", {})
        expected_repo = upstream.get("repository")
        expected_tag = upstream.get("tag")
        expected_commit = upstream.get("commit_sha")
        if not all(
            isinstance(value, str) for value in (expected_repo, expected_tag, expected_commit)
        ):
            issues.append(f"dependency upstream policy identity is invalid for {identity[0]}")
            continue
        if entry.get("repository") != expected_repo:
            issues.append(f"dependency upstream repository differs for {identity[0]}")
        tags = entry.get("tags")
        matching_tags = entry.get("matching_tags")
        if (
            entry.get("tag") != expected_tag
            or not isinstance(tags, list)
            or not all(isinstance(value, str) for value in tags)
            or len(tags) != len(set(tags))
            or expected_tag not in tags
            or not isinstance(matching_tags, list)
            or not all(isinstance(value, str) for value in matching_tags)
            or not set(matching_tags).issubset(tags)
        ):
            issues.append(f"dependency upstream tag identity differs for {identity[0]}")
        if entry.get("errors") != [] or entry.get("source_url_key") != "Source":
            issues.append(f"dependency upstream tag result is not clean for {identity[0]}")
        checks = entry.get("notice_checks")
        archive_row = archive_by_identity.get(identity, {})
        expected_mappings = _policy_archive_notice_mapping(policy_row)
        expected_archive_paths = {
            archive_path for archive_path, _kind, _path, _url in expected_mappings
        }
        expected_upstream_paths = {
            path
            for path in (_upstream_notice_path(url) for url in upstream.get("notice_paths", []))
            if path is not None
        }
        actual_archive_paths: set[str] = set()
        actual_upstream_paths: set[str] = set()
        actual_mappings: set[tuple[Any, str, str, str]] = set()
        if not isinstance(checks, list):
            issues.append(f"dependency upstream notice checks are missing for {identity[0]}")
            continue
        for check in checks:
            if not isinstance(check, dict):
                issues.append(f"dependency upstream notice check is invalid for {identity[0]}")
                continue
            if set(check) != {
                "archive_kind",
                "archive_path",
                "review_url",
                "status",
                "upstream_path",
            }:
                issues.append(
                    f"dependency upstream notice check has unclassified fields for {identity[0]}"
                )
            archive_kind = check.get("archive_kind")
            archive_path = check.get("archive_path")
            if isinstance(archive_path, str):
                actual_archive_paths.add(archive_path)
            upstream_path = check.get("upstream_path")
            if isinstance(upstream_path, str):
                actual_upstream_paths.add(upstream_path)
            review_url = check.get("review_url")
            reference = _github_notice_reference(review_url)
            review_path = _upstream_notice_path(review_url)
            if archive_kind not in {
                artifact.get("kind")
                for artifact in archive_row.get("artifacts", [])
                if isinstance(artifact, dict)
            }:
                issues.append(f"dependency upstream archive kind is invalid for {identity[0]}")
            if archive_path not in expected_archive_paths:
                issues.append(
                    f"dependency upstream archive path is not policy-bound for {identity[0]}"
                )
            if review_path != upstream_path or upstream_path not in expected_upstream_paths:
                issues.append(
                    f"dependency upstream notice URL path differs from policy for {identity[0]}"
                )
            if all(
                isinstance(value, str)
                for value in (archive_path, archive_kind, upstream_path, review_url)
            ):
                actual_mappings.add((archive_path, archive_kind, upstream_path, review_url))
            if check.get("status") != "present" or reference is None:
                issues.append(
                    f"dependency upstream notice check is not immutably bound for {identity[0]}"
                )
        if actual_archive_paths != expected_archive_paths:
            issues.append(f"dependency upstream archive notice paths differ for {identity[0]}")
        if actual_upstream_paths != expected_upstream_paths:
            issues.append(f"dependency upstream notice paths differ for {identity[0]}")
        if actual_mappings != expected_mappings:
            issues.append(f"dependency upstream notice mappings differ for {identity[0]}")
        if len(actual_mappings) != len(checks):
            issues.append(f"dependency upstream notice checks contain duplicates for {identity[0]}")
    if observed != set(expected_by_identity):
        issues.append("dependency upstream tags identities are not an exact policy set")
    return issues


def _candidate_receipt_semantic_issues(  # noqa: C901, PLR0912
    candidate: dict[str, Any],
    *,
    repo_root: Path,
    expected_profiles: list[str],
) -> tuple[list[str], Path | None, dict[str, Any] | None]:
    """Validate receipt candidate files with the complete canonical bundle contract."""
    issues: list[str] = []
    expected_candidate_keys = {
        "expected_component_count",
        "manifest_path",
        "manifest_sha256",
        "materialization",
        "members",
        "package",
        "profile_ids",
        "repository",
        "sbom",
        "source_sha",
        "status",
        "workflow",
    }
    if set(candidate) not in (expected_candidate_keys, expected_candidate_keys | {"archives"}):
        issues.append("dependency receipt candidate has missing or unclassified fields")
    manifest_path = candidate.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        return ["dependency receipt candidate manifest_path is missing or unverifiable"], None, None
    resolved_manifest = _receipt_path(repo_root, manifest_path)
    if _receipt_path_has_symlink(resolved_manifest) or not resolved_manifest.is_file():
        return [f"dependency receipt candidate manifest is missing: {manifest_path}"], None, None
    bundle = resolved_manifest.parent
    try:
        manifest = _read_json(resolved_manifest)
        package = _candidate_manifest_contract(manifest)
        paths_by_kind, bound_members = _candidate_manifest_members(bundle, manifest)
        _candidate_archive_contract(paths_by_kind, package)
        _candidate_provenance_contract(paths_by_kind["provenance"], manifest)
        actual_components = _candidate_sbom_components(paths_by_kind["sbom"], package)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return [f"dependency receipt candidate bundle is invalid: {exc}"], bundle, None
    if candidate.get("manifest_sha256") != _sha256_file(resolved_manifest):
        issues.append("dependency receipt candidate manifest SHA-256 differs from bound file")
    expected_members = [bound_members[kind] for kind in _CANDIDATE_MEMBER_KINDS]
    receipt_members = candidate.get("members")
    if not isinstance(receipt_members, list):
        issues.append("dependency receipt candidate members summary is missing")
    else:
        normalized_members = [
            {key: member.get(key) for key in ("filename", "kind", "sha256", "size")}
            for member in receipt_members
            if isinstance(member, dict)
        ]
        if normalized_members != expected_members:
            issues.append("dependency receipt candidate members differ from canonical bundle")
        for member in receipt_members:
            if not isinstance(member, dict):
                issues.append("dependency receipt candidate member is incomplete")
                continue
            path_value = member.get("path")
            if not isinstance(path_value, str) or not path_value:
                issues.append("dependency receipt candidate member path is missing or unverifiable")
                continue
            resolved_member = _receipt_path(repo_root, path_value)
            if resolved_member.parent != bundle or resolved_member.name != member.get("filename"):
                issues.append(
                    "dependency receipt candidate member path is outside its manifest bundle"
                )
            elif _receipt_path_has_symlink(resolved_member) or not resolved_member.is_file():
                issues.append(f"dependency receipt candidate member is missing: {path_value}")
    try:
        inventory = build_inventory(
            repo_root,
            distributions=[],
            selected_profile_ids=expected_profiles,
            candidate_bundle_path=bundle,
        )
        canonical_binding = inventory.get("candidate_binding")
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        issues.append(f"dependency receipt candidate canonical binding failed: {exc}")
        canonical_binding = None
    if not isinstance(canonical_binding, dict):
        issues.append("dependency receipt candidate has no canonical bundle binding")
    else:
        for field in (
            "status",
            "repository",
            "source_sha",
            "workflow",
            "materialization",
            "package",
            "sbom",
            "profile_ids",
            "expected_component_count",
        ):
            if candidate.get(field) != canonical_binding.get(field):
                issues.append(f"dependency receipt candidate {field} differs from canonical bundle")
        if candidate.get("members") and [
            {key: member.get(key) for key in ("filename", "kind", "sha256", "size")}
            for member in candidate["members"]
            if isinstance(member, dict)
        ] != canonical_binding.get("members"):
            issues.append(
                "dependency receipt candidate member identities differ from canonical bundle"
            )
        if "archives" in candidate and candidate.get("archives") != canonical_binding.get(
            "archives"
        ):
            issues.append("dependency receipt candidate archives differ from canonical bundle")
        if candidate.get("expected_component_count") != len(actual_components):
            issues.append("dependency receipt candidate component count differs from SBOM")
    return issues, bundle, canonical_binding


def _strict_report_semantic_issues(
    report: dict[str, Any],
    *,
    repo_root: Path,
    candidate_bundle: Path | None,
) -> list[str]:
    """Rebuild and compare the strict report's complete canonical semantics."""
    issues: list[str] = []
    surface = report.get("surface")
    profile_ids = surface.get("profile_ids") if isinstance(surface, dict) else None
    if not isinstance(profile_ids, list) or not all(
        isinstance(profile_id, str) for profile_id in profile_ids
    ):
        return ["dependency receipt strict report has no valid selected profile surface"]
    try:
        canonical = build_inventory(
            repo_root,
            selected_profile_ids=profile_ids,
            candidate_bundle_path=candidate_bundle,
        )
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        return [f"dependency receipt strict report canonical rebuild failed: {exc}"]
    for field in (
        "environment",
        "failures",
        "installed_not_locked",
        "packages",
        "policy",
        "profile_manifest",
        "profiles",
        "project",
        "repository_inputs",
        "structural_issues",
        "summary",
        "surface",
        "target",
        "unrepresented_lock_package_dispositions",
        "unrepresented_lock_packages",
        "candidate_binding",
    ):
        if report.get(field) != canonical.get(field):
            issues.append(
                f"dependency receipt strict report {field} differs from canonical inventory"
            )
    return issues


def _receipt_contract_issues(  # noqa: C901, PLR0912, PLR0915
    receipt: dict[str, Any],
    policy: dict[str, Any],
    policy_binding: dict[str, Any],
    repo_root: Path,
) -> list[str]:
    """Validate receipt summaries against the immutable policy and bound files."""
    issues: list[str] = []
    expected_claim_boundary = (
        "This receipt records reproducible package/archive and candidate-binding evidence. "
        "It is not a legal opinion, redistribution authorization, release approval, or "
        "independent review marker."
    )
    if receipt.get("claim_boundary") != expected_claim_boundary:
        issues.append("dependency receipt claim_boundary is not the exact non-approval boundary")
    review = receipt.get("review")
    if not isinstance(review, dict):
        issues.append("dependency receipt has no review status")
    else:
        if review.get("status") != "pending_independent_maintainer_review":
            issues.append("dependency receipt review status must remain pending")
        if review.get("reviewer") is not None or review.get("reviewed_at") is not None:
            issues.append("dependency receipt review identity must remain null while pending")
        if review.get("legal_or_redistribution_approval") is not False:
            issues.append("dependency receipt cannot claim legal_or_redistribution_approval")
    batch_rows = _policy_batch_rows(policy)
    expected_profiles = sorted(
        {
            profile
            for row in batch_rows
            for profile in row.get("profiles", [])
            if isinstance(profile, str)
        }
    )
    manifest = _read_json(repo_root / CANONICAL_PROFILE_MANIFEST)
    manifest_target = manifest.get("target", {})
    expected_target = {
        "os": manifest_target.get("os"),
        "architecture": manifest_target.get("architecture"),
        "python": {
            "implementation": manifest_target.get("python", {}).get("implementation"),
            "version": manifest_target.get("python", {}).get("version"),
        },
    }
    expected_scope = {
        "package_count": len(batch_rows),
        "artifact_count": sum(len(row.get("artifacts", [])) for row in batch_rows),
        "profile_ids": expected_profiles,
        "target": expected_target,
    }
    expected_status = (
        "blocked_diagnostic_only"
        if any(row.get("status") != "reviewed" for row in batch_rows)
        else "complete"
    )
    if receipt.get("status") != expected_status:
        issues.append(
            f"dependency receipt status does not match bound policy: expected {expected_status}"
        )
    scope = receipt.get("scope")
    if not isinstance(scope, dict):
        issues.append("dependency receipt has no scope summary")
    else:
        for field in ("package_count", "artifact_count", "profile_ids", "target"):
            if _normalise_json(scope.get(field)) != _normalise_json(expected_scope[field]):
                issues.append(f"dependency receipt scope {field} differs from bound policy")

    archive = receipt.get("archive_audit")
    if not isinstance(archive, dict):
        issues.append("dependency receipt has no archive_audit summary")
    else:
        if archive.get("schema_version") != "robot-sf.issue-8163-archive-audit.v1":
            issues.append("dependency receipt archive_audit has an unsupported schema_version")
        for field in ("package_count", "artifact_count"):
            if archive.get(field) != expected_scope[field]:
                issues.append(f"dependency receipt archive_audit {field} differs from scope")
        if archive.get("failures") != []:
            issues.append("dependency receipt archive_audit failures must be empty")
        for field in ("sha256", "upstream_tags_sha256"):
            value = archive.get(field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                issues.append(f"dependency receipt archive_audit {field} is not a valid SHA-256")
        archive_file: dict[str, Any] | None = None
        archive_path = archive.get("path")
        if isinstance(archive_path, str) and archive_path:
            resolved = _receipt_path(repo_root, archive_path)
            if _receipt_path_has_symlink(resolved) or not resolved.is_file():
                issues.append(f"dependency receipt archive audit is missing: {archive_path}")
            else:
                if archive.get("sha256") != _sha256_file(resolved):
                    issues.append(
                        "dependency receipt archive_audit SHA-256 differs from bound file"
                    )
                try:
                    archive_file = _read_json(resolved)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"dependency receipt archive audit is invalid: {exc}")
                else:
                    issues.extend(_archive_audit_semantic_issues(archive_file, batch_rows))
                    if archive_file.get("schema_version") != archive.get("schema_version"):
                        issues.append(
                            "dependency receipt archive_audit schema_version differs from bound file"
                        )
                    packages = archive_file.get("packages")
                    if not isinstance(packages, list):
                        issues.append("dependency receipt archive audit has no package rows")
                    else:
                        package_count = len(packages)
                        artifact_count = sum(
                            len(row.get("artifacts", []))
                            for row in packages
                            if isinstance(row, dict) and isinstance(row.get("artifacts"), list)
                        )
                        if package_count != archive.get("package_count"):
                            issues.append(
                                "dependency receipt archive_audit package_count differs from bound file"
                            )
                        if artifact_count != archive.get("artifact_count"):
                            issues.append(
                                "dependency receipt archive_audit artifact_count differs from bound file"
                            )
                    if archive_file.get("failures") != archive.get("failures"):
                        issues.append(
                            "dependency receipt archive_audit failures differ from bound file"
                        )
        else:
            issues.append("dependency receipt archive_audit.path is missing or unverifiable")
        tags_path = archive.get("upstream_tags_path")
        if isinstance(tags_path, str) and tags_path:
            resolved_tags = _receipt_path(repo_root, tags_path)
            if _receipt_path_has_symlink(resolved_tags) or not resolved_tags.is_file():
                issues.append(f"dependency receipt upstream tags file is missing: {tags_path}")
            else:
                if archive.get("upstream_tags_sha256") != _sha256_file(resolved_tags):
                    issues.append(
                        "dependency receipt upstream_tags SHA-256 differs from bound file"
                    )
                try:
                    tags_file = json.loads(resolved_tags.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"dependency receipt upstream tags evidence is invalid: {exc}")
                else:
                    issues.extend(
                        _upstream_tags_semantic_issues(tags_file, batch_rows, archive_file)
                    )
        else:
            issues.append(
                "dependency receipt archive_audit.upstream_tags_path is missing or unverifiable"
            )

    candidate = receipt.get("candidate_binding")
    if not isinstance(candidate, dict):
        issues.append("dependency receipt has no candidate_binding summary")
    else:
        if candidate.get("status") != "bound":
            issues.append("dependency receipt candidate_binding must be bound")
        if candidate.get("profile_ids") != expected_profiles:
            issues.append("dependency receipt candidate profile_ids differ from scope")
        package = candidate.get("package")
        if not isinstance(package, dict) or not all(
            isinstance(package.get(field), str) and package.get(field)
            for field in ("name", "version")
        ):
            issues.append("dependency receipt candidate package identity is incomplete")
        sbom = candidate.get("sbom")
        expected_components = candidate.get("expected_component_count")
        if not isinstance(expected_components, int) or expected_components <= 0:
            issues.append("dependency receipt candidate expected_component_count is invalid")
        if not isinstance(sbom, dict) or sbom.get("component_count") != expected_components:
            issues.append("dependency receipt candidate SBOM count differs from candidate summary")
        materialization = candidate.get("materialization")
        if not isinstance(materialization, dict):
            issues.append("dependency receipt candidate materialization summary is missing")
        for field in ("manifest_sha256",):
            value = candidate.get(field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                issues.append(f"dependency receipt candidate {field} is not a valid SHA-256")
        members = candidate.get("members")
        if not isinstance(members, list) or not members:
            issues.append("dependency receipt candidate members summary is missing")
        elif len({member.get("filename") for member in members if isinstance(member, dict)}) != len(
            members
        ):
            issues.append("dependency receipt candidate members contain duplicate filenames")
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict) or not isinstance(member.get("filename"), str):
                    issues.append("dependency receipt candidate member is incomplete")
                    continue
                value = member.get("sha256")
                if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                    issues.append(
                        f"dependency receipt candidate member {member['filename']} has invalid SHA-256"
                    )
        manifest_path = candidate.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            issues.append("dependency receipt candidate manifest_path is missing or unverifiable")
        else:
            resolved_manifest = Path(manifest_path.removeprefix("operator-local:"))
            if not resolved_manifest.is_absolute():
                resolved_manifest = _resolve_path(repo_root, manifest_path)
            if not resolved_manifest.is_file():
                issues.append(f"dependency receipt candidate manifest is missing: {manifest_path}")
            else:
                if candidate.get("manifest_sha256") != _sha256_file(resolved_manifest):
                    issues.append(
                        "dependency receipt candidate manifest SHA-256 differs from bound file"
                    )
                try:
                    manifest_file = _read_json(resolved_manifest)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"dependency receipt candidate manifest is invalid: {exc}")
                else:
                    for field in ("repository", "source_sha", "package"):
                        if manifest_file.get(field) != candidate.get(field):
                            issues.append(
                                f"dependency receipt candidate {field} differs from bound manifest"
                            )
                    if manifest_file.get("members") != [
                        {key: member.get(key) for key in ("filename", "kind", "sha256", "size")}
                        for member in members
                        if isinstance(member, dict)
                    ]:
                        issues.append(
                            "dependency receipt candidate members differ from bound manifest"
                        )
                    for member in members if isinstance(members, list) else []:
                        member_path = member.get("path") if isinstance(member, dict) else None
                        if not isinstance(member_path, str) or not member_path:
                            issues.append(
                                "dependency receipt candidate member path is missing or unverifiable"
                            )
                            continue
                        resolved_member = Path(member_path.removeprefix("operator-local:"))
                        if resolved_member.name != member.get("filename"):
                            issues.append(
                                "dependency receipt candidate member path does not match "
                                f"filename {member.get('filename')}"
                            )
                        if not resolved_member.is_absolute():
                            resolved_member = _resolve_path(repo_root, member_path)
                        if not resolved_member.is_file():
                            issues.append(
                                f"dependency receipt candidate member is missing: {member_path}"
                            )
                        elif member.get("sha256") != _sha256_file(resolved_member):
                            issues.append(
                                f"dependency receipt candidate member {member.get('filename')} SHA-256 differs from bound file"
                            )

    candidate_bundle: Path | None = None
    canonical_candidate: dict[str, Any] | None = None
    if isinstance(candidate, dict):
        candidate_issues, candidate_bundle, canonical_candidate = (
            _candidate_receipt_semantic_issues(
                candidate,
                repo_root=repo_root,
                expected_profiles=expected_profiles,
            )
        )
        issues.extend(candidate_issues)

    strict = receipt.get("strict_report")
    if not isinstance(strict, dict):
        issues.append("dependency receipt has no strict_report binding")
    else:
        if strict.get("candidate_bound") != (
            isinstance(candidate, dict) and candidate.get("status") == "bound"
        ):
            issues.append("dependency receipt strict_report candidate_bound differs from candidate")
        if strict.get("surface_profile_ids") != expected_profiles:
            issues.append("dependency receipt strict_report surface differs from scope")
        summary = strict.get("summary")
        if not isinstance(summary, dict):
            issues.append("dependency receipt strict_report has no summary")
        else:
            if summary.get("policy_exact_disposition_count") != len(
                policy.get("package_dispositions", [])
            ):
                issues.append("dependency receipt strict_report policy count differs from policy")
            if summary.get("status") != (
                "blocked" if receipt.get("status") != "complete" else "complete"
            ):
                issues.append("dependency receipt strict_report status differs from receipt status")
            pending = summary.get("policy_pending_package_count")
            unresolved = summary.get("unresolved_count")
            if not isinstance(pending, int) or pending < 0:
                issues.append("dependency receipt strict_report pending count is invalid")
            if not isinstance(unresolved, int) or unresolved < 0:
                issues.append("dependency receipt strict_report unresolved count is invalid")
            elif isinstance(pending, int) and unresolved != pending:
                issues.append(
                    "dependency receipt strict_report unresolved count differs from pending count"
                )
            expected_exit = 2 if summary.get("status") == "blocked" else 0
            if strict.get("exit_code") != expected_exit:
                issues.append("dependency receipt strict_report exit_code differs from status")
        report_sha = strict.get("sha256")
        if not isinstance(report_sha, str) or _SHA256_RE.fullmatch(report_sha) is None:
            issues.append("dependency receipt strict_report.sha256 is not a valid SHA-256")
        report_path = strict.get("path")
        if isinstance(report_path, str) and report_path:
            resolved = _receipt_path(repo_root, report_path)
            if _receipt_path_has_symlink(resolved) or not resolved.is_file():
                issues.append(
                    "dependency receipt strict report SHA-256 cannot be verified because "
                    f"the report is missing: {report_path}"
                )
            else:
                if report_sha != _sha256_file(resolved):
                    issues.append(
                        "dependency receipt strict report SHA-256 differs from report bytes"
                    )
                try:
                    report_file = _read_json(resolved)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"dependency receipt strict report is invalid: {exc}")
                else:
                    expected_report_keys = {
                        "candidate_binding",
                        "environment",
                        "failures",
                        "installed_not_locked",
                        "packages",
                        "policy",
                        "profile_manifest",
                        "profiles",
                        "project",
                        "repository_inputs",
                        "schema_version",
                        "structural_issues",
                        "summary",
                        "surface",
                        "target",
                        "unrepresented_lock_package_dispositions",
                        "unrepresented_lock_packages",
                        _REPORT_CONTENT_DIGEST_FIELD,
                    }
                    if set(report_file) - {"review_marker"} != expected_report_keys:
                        issues.append(
                            "dependency receipt strict report has missing or unclassified fields"
                        )
                    content_digest = report_file.get(_REPORT_CONTENT_DIGEST_FIELD)
                    if not isinstance(content_digest, str) or not _SHA256_RE.fullmatch(
                        content_digest
                    ):
                        issues.append(
                            "dependency receipt strict report has no valid content digest"
                        )
                    elif content_digest != _report_content_digest(report_file):
                        issues.append("dependency receipt strict report content digest differs")
                    if candidate_bundle is not None:
                        issues.extend(
                            check_report_freshness(
                                repo_root,
                                resolved,
                                candidate_bundle_path=candidate_bundle,
                            )
                        )
                    issues.extend(
                        _strict_report_semantic_issues(
                            report_file,
                            repo_root=repo_root,
                            candidate_bundle=candidate_bundle,
                        )
                    )
                    if (
                        canonical_candidate is not None
                        and report_file.get("candidate_binding") != canonical_candidate
                    ):
                        issues.append(
                            "dependency receipt strict report candidate binding differs from canonical bundle"
                        )
                    report_summary = report_file.get("summary")
                    if isinstance(report_summary, dict) and isinstance(summary, dict):
                        for field in (
                            "selected_package_count",
                            "license_status_counts",
                            "structural_issue_count",
                            "policy_exact_match_count",
                            "policy_exact_disposition_count",
                            "policy_pending_package_count",
                            "unresolved_count",
                            "status",
                        ):
                            if report_summary.get(field) != summary.get(field):
                                issues.append(
                                    f"dependency receipt strict_report summary {field} differs from bound report"
                                )
                    else:
                        issues.append("dependency receipt strict report has no summary object")
        else:
            issues.append("dependency receipt strict_report.path is missing or unverifiable")
    return issues


def validate_dependency_license_receipt(  # noqa: C901, PLR0912, PLR0915
    repo_root: Path,
    receipt_path: Path,
    *,
    expected_reviewed_head: str | None = None,
) -> list[str]:
    """Validate a dependency-batch receipt's reproducible, non-approval bindings.

    The checker validates hashes, identities, summaries, and exact file contents
    but never upgrades a row to ``reviewed``. Operator-local paths are not
    trusted by label: they must resolve to the exact retained files.
    """
    issues: list[str] = []
    try:
        receipt = _read_json(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"dependency receipt could not be read: {exc}"]
    if receipt.get("schema_version") != "robot-sf.issue-8163-dependency-license-batch.receipt.v1":
        issues.append("dependency receipt has an unsupported schema_version")

    policy_path = repo_root / CANONICAL_POLICY
    try:
        policy = _read_json(policy_path)
        policy_binding, policy_issues = _issue_8163_receipt_binding(policy, repo_root)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        return [f"dependency receipt policy binding could not be computed: {exc}"]
    issues.extend(policy_issues)
    try:
        issues.extend(_receipt_contract_issues(receipt, policy, policy_binding, repo_root))
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        issues.append(f"dependency receipt contract could not be checked: {exc}")
    binding = receipt.get("review_binding")
    if not isinstance(binding, dict):
        issues.append("dependency receipt has no review_binding record")
        binding = {}
    for field in ("normalized_records_sha256", "license_files_sha256"):
        expected = policy_binding[field]
        actual = binding.get(field)
        if actual != expected:
            issues.append(
                f"dependency receipt {field} differs from the current normalized policy: "
                f"receipt={actual!r} actual={expected!r}"
            )
    if binding.get("normalized_record_count") != policy_binding["normalized_record_count"]:
        issues.append("dependency receipt normalized_record_count differs from the current policy")
    if binding.get("license_files") != policy_binding["license_files"]:
        issues.append("dependency receipt license_files manifest differs from the current policy")
    if binding.get("evidence_files") != policy_binding["evidence_files"]:
        issues.append("dependency receipt evidence_files binding differs from the current evidence")
    for evidence_file in policy_binding["evidence_files"]:
        if evidence_file["sha256"] != _sha256_file(repo_root / evidence_file["path"]):
            issues.append(f"dependency receipt evidence file changed: {evidence_file['path']}")

    source = receipt.get("source")
    candidate = receipt.get("candidate_binding")
    reviewed_head = binding.get("reviewed_head_sha")
    if (
        not isinstance(reviewed_head, str)
        or _CANDIDATE_SOURCE_SHA_RE.fullmatch(reviewed_head) is None
    ):
        issues.append(
            "dependency receipt review_binding.reviewed_head_sha is not a full commit SHA"
        )
    if expected_reviewed_head is not None and reviewed_head != expected_reviewed_head:
        issues.append(
            "dependency receipt reviewed_head_sha differs from the expected reviewed head"
        )
    if (
        expected_reviewed_head is not None
        and _CANDIDATE_SOURCE_SHA_RE.fullmatch(expected_reviewed_head) is None
    ):
        issues.append("expected reviewed head must be a full commit SHA")
    if isinstance(source, dict) and source.get("source_sha") != reviewed_head:
        issues.append("dependency receipt reviewed_head_sha is not bound to source.source_sha")
    if isinstance(candidate, dict) and candidate.get("source_sha") != reviewed_head:
        issues.append("dependency receipt reviewed_head_sha is not bound to candidate source_sha")

    policy_file_binding = binding.get("policy")
    if not isinstance(policy_file_binding, dict):
        issues.append("dependency receipt review_binding has no policy file binding")
    else:
        if policy_file_binding.get("path") != CANONICAL_POLICY:
            issues.append("dependency receipt policy binding does not name the canonical policy")
        policy_sha = policy_file_binding.get("sha256")
        if not isinstance(policy_sha, str) or _SHA256_RE.fullmatch(policy_sha) is None:
            issues.append("dependency receipt policy binding has an invalid SHA-256")
        elif policy_sha != _sha256_file(policy_path):
            issues.append(
                "dependency receipt policy binding SHA-256 differs from the current policy"
            )

    strict_report = receipt.get("strict_report")
    if not isinstance(strict_report, dict):
        issues.append("dependency receipt has no strict_report binding")
    else:
        report_sha = strict_report.get("sha256")
        if not isinstance(report_sha, str) or _SHA256_RE.fullmatch(report_sha) is None:
            issues.append("dependency receipt strict_report.sha256 is not a valid SHA-256")
        report_path = strict_report.get("path")
        if isinstance(report_path, str) and not report_path.startswith("operator-local:"):
            resolved_report = _resolve_path(repo_root, report_path)
            if not resolved_report.is_file():
                issues.append(f"dependency receipt strict report is missing: {report_path}")
            elif report_sha != _sha256_file(resolved_report):
                issues.append(
                    "dependency receipt strict report SHA-256 differs from the report bytes"
                )
        elif not isinstance(report_path, str) or not report_path:
            issues.append("dependency receipt strict_report.path is missing")
    return sorted(set(issues))


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
    failures: list[str] = []
    matching_rows = [
        row
        for row in rows
        if row.get("version") == package.get("version")
        and _policy_source_matches(package.get("source", {}), row.get("source", {}))
    ]
    if len(matching_rows) != 1:
        return None, [f"exact policy identity is ambiguous ({len(matching_rows)} matching rows)"]
    policy = matching_rows[0]
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
    actual_profiles = _effective_profile_coverage(profiles, expected_profiles)
    if not actual_profiles <= expected_profiles:
        failures.append(
            "package profile membership exceeds exact policy: "
            f"{sorted(actual_profiles - expected_profiles)}"
        )
    failures.extend(_exact_artifact_failures(package, policy.get("artifacts", [])))
    return policy, sorted(set(failures))


def _exact_policy_coverage_failures(
    package_dispositions: list[dict[str, Any]],
    package_records: list[dict[str, Any]],
    profile_ids: set[str],
    selected_profile_ids: set[str] | None = None,
) -> list[str]:
    """Ensure exact rows cover their selected profile surface.

    ``profiles`` on a lock record is the complete transitive membership graph;
    it is not the policy surface selected for this invocation.  Prefer the
    record's ``selected_profiles`` projection and collapse the aggregate
    ``all`` profile so ``profiles: [all]`` remains an intentional exact row.
    """
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
        actual_profiles = {
            profile
            for record in matches
            for profile in record.get(
                "selected_profiles" if "selected_profiles" in record else "profiles",
                [],
            )
        }
        if selected_profile_ids is not None:
            actual_profiles &= selected_profile_ids
        actual_profiles = _effective_profile_coverage(actual_profiles, expected_profiles)
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
        if candidate_bundle_path is None:
            selected_ids = set(profile_id_set)
        else:
            # A v0.0.6 software candidate is the closed supported profile union.
            # Core-only selection remains available only as an explicit diagnostic
            # and can never satisfy the separate rights-admission receipt gate.
            selected_ids = set(SUPPORTED_SOFTWARE_CANDIDATE_PROFILE_IDS)
            missing_profiles = sorted(selected_ids - profile_id_set)
            structural_issues.extend(
                f"candidate-bound inventory is missing supported profile: {profile_id}"
                for profile_id in missing_profiles
            )
            selected_ids &= profile_id_set
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
                target=manifest.get("target", {})
                if isinstance(manifest.get("target"), dict)
                else {},
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
            selected_ids,
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
    effective_generator_path = generator_path
    if effective_generator_path is None:
        effective_generator_path = (
            repo_root / CANONICAL_GENERATOR
            if candidate_bundle_path is not None
            else Path(__file__).resolve()
        )
    inputs, input_issues = _input_paths(
        repo_root,
        manifest,
        policy,
        profiles,
        manifest_path,
        policy_file,
        effective_generator_path,
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
    inventory = {
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
            "normalized_records_sha256": _sha256_value(
                _policy_identity_records(package_dispositions)
            ),
            "normalized_record_count": len(package_dispositions),
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
    inventory[_REPORT_CONTENT_DIGEST_FIELD] = _report_content_digest(inventory)
    return inventory


def check_report_freshness(  # noqa: C901, PLR0912
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
    recorded_content_digest = report.get(_REPORT_CONTENT_DIGEST_FIELD)
    if not isinstance(recorded_content_digest, str) or not _SHA256_RE.fullmatch(
        recorded_content_digest
    ):
        issues.append("report has no valid report_content_sha256")
    elif recorded_content_digest != _report_content_digest(report):
        issues.append("report content digest differs from recorded report")
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
        "--check-receipt",
        type=Path,
        help="Check a dependency-batch receipt's exact policy, evidence, and report bindings.",
    )
    parser.add_argument(
        "--expected-reviewed-head",
        help="Require --check-receipt to name this exact reviewed source commit.",
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
        if args.check_receipt:
            receipt_path = _resolve_path(repo_root, args.check_receipt)
            issues = validate_dependency_license_receipt(
                repo_root,
                receipt_path,
                expected_reviewed_head=args.expected_reviewed_head,
            )
            print(
                json.dumps(
                    {
                        "schema_version": "dependency_license_receipt_check.v1",
                        "issues": issues,
                        "status": "blocked" if issues else "complete",
                    },
                    indent=2,
                )
            )
            return 1 if issues else 0
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

    # Keep the report marker additive while retaining standalone invocation.  This
    # CLI is also run by the dependency-review workflow as ``python
    # scripts/tools/check_dependency_license_inventory.py``; importing the package
    # writer here would fail in that invocation because Python puts only the script
    # directory on ``sys.path``.  Durable checked-in evidence uses the shared writer
    # and review sidecar; this output is a local/CI artifact outside that tree.
    marked_inventory = {"review_marker": _REVIEW_MARKER_JSON, **inventory}
    rendered = json.dumps(marked_inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
