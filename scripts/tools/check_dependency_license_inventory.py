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
import sys
import tomllib
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


SCHEMA_VERSION = "robot-sf.dependency-license-inventory.v1"
PROFILE_SCHEMA_VERSION = "robot-sf.dependency-license-profiles.v1"
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
_REQUIREMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[A-Za-z0-9._,-]+)\])?"
)
_EXTRA_RE = re.compile(r"extra\s*(==|!=)\s*['\"]([^'\"]+)['\"]")
_PYTHON_RE = re.compile(
    r"(?:python_full_version|python_version)\s*(==|!=|<=|>=|<|>)\s*['\"]([0-9.]+)['\"]"
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
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                continue
            entry: dict[str, Any] = {"name": dependency["name"]}
            for key in ("marker", "version"):
                if isinstance(dependency.get(key), str):
                    entry[key] = dependency[key]
            if isinstance(dependency.get("source"), dict):
                entry["source"] = _normalise_json(dependency["source"])
            dependencies.append(entry)
        identity = {
            "lockfile": repo_relative_path,
            "name": package["name"],
            "version": version,
            "source": source,
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


def _policy_records(  # noqa: C901
    policy: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Validate policy rules and capture component evidence digests."""
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
    return by_mode, sorted(components_out, key=lambda item: item["id"]), issues


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


def _input_paths(  # noqa: C901
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


def build_inventory(  # noqa: C901
    repo_root: Path,
    *,
    distributions: Iterable[Any] | None = None,
    profile_manifest_path: Path | None = None,
    policy_path: Path | None = None,
    generator_path: Path | None = None,
) -> dict[str, Any]:
    """Build a lock/profile/environment inventory without network or writes."""
    repo_root = repo_root.resolve()
    manifest_path = profile_manifest_path or (
        repo_root / "scripts/validation/dependency_license_profiles.v1.json"
    )
    policy_file = policy_path or repo_root / "scripts/validation/dependency_license_policy.v1.json"
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
    policy_rules, components, policy_issues = _policy_records(policy, repo_root)
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
    resolution_failures: list[str] = []
    for profile in profiles:
        lockfile = profile.get("lockfile")
        lock_packages = packages_by_lockfile.get(lockfile, []) if isinstance(lockfile, str) else []
        result = _resolve_profile(profile, lock_packages, manifest, repo_root)
        profile_results.append(result)
        for package_id in result.get("package_ids", []):
            package_profiles[package_id].add(result["id"])
        resolution_failures.extend(
            f"{result['id']}: {failure}"
            for failure in [
                *result.get("missing_dependencies", []),
                *result.get("conflicting_dependencies", []),
            ]
        )

    observed = _observed_distributions(distributions)
    package_records: list[dict[str, Any]] = []
    package_failures: list[str] = []
    policy_failures: list[str] = []
    status_counts: Counter[str] = Counter()
    for package_id in sorted(all_packages):
        package = all_packages[package_id]
        mode = _distribution_mode(package)
        rule = policy_rules.get(mode)
        if rule is None:
            structural_issues.append(f"no policy rule for distribution mode: {mode}")
            rule = {"id": "missing-policy-rule", "disposition": "review_required"}
        observation, failures = _package_observation(package, observed)
        package_failures.extend(failure for failure in failures if package_id in package_profiles)
        record = {
            **package,
            **observation,
            "distribution_mode": mode,
            "policy_rule_id": rule.get("id"),
            "policy_disposition": rule.get("disposition"),
            "profiles": sorted(package_profiles.get(package_id, set())),
            "originating_extras": sorted(
                {
                    extra
                    for profile_result in profile_results
                    if profile_result["id"] in package_profiles.get(package_id, set())
                    for extra in profile_result.get("extras", [])
                }
            ),
        }
        status_counts[record["license_status"]] += 1
        if package_id in package_profiles and rule.get("disposition") == "review_required":
            policy_failures.append(
                f"{package['name']}: {mode} requires an explicit reviewed disposition"
            )
        package_records.append(record)

    component_failures = [
        f"component {component['id']}: disposition is {component['disposition']}"
        for component in components
        if component.get("status") != "reviewed" or component.get("disposition") != "approved"
    ]
    unrepresented = [
        package_id for package_id in sorted(all_packages) if not package_profiles.get(package_id)
    ]
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
        "policy": {
            "path": _relative_path(repo_root, policy_file),
            "schema_version": policy.get("schema_version"),
            "claim_boundary": policy.get("claim_boundary"),
            "rules": _normalise_json(policy.get("rules", [])),
            "components": components,
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
        "profiles": profile_results,
        "packages": package_records,
        "unrepresented_lock_packages": unrepresented,
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
            "locked_package_count": len(package_records),
            "profile_membership_edge_count": sum(
                len(result.get("package_ids", [])) for result in profile_results
            ),
            "unrepresented_lock_package_count": len(unrepresented),
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
            "structural_issue_count": len(set(structural_issues)),
            "unresolved_count": len(failures),
            "status": "blocked" if failures else "complete",
        },
    }


def check_report_freshness(repo_root: Path, report_path: Path) -> list[str]:
    """Recompute every recorded input digest for an existing report."""
    report = _read_json(report_path)
    issues: list[str] = []
    inputs = report.get("repository_inputs")
    if not isinstance(inputs, list) or not inputs:
        return ["report has no repository_inputs digest list"]
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
    return sorted(set(issues))


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

    try:
        if args.check_freshness:
            issues = check_report_freshness(repo_root, args.check_freshness.resolve())
            print(
                json.dumps(
                    {"schema_version": "dependency_license_freshness.v1", "issues": issues},
                    indent=2,
                )
            )
            return 1 if issues else 0
        inventory = build_inventory(
            repo_root,
            profile_manifest_path=(
                _resolve_path(repo_root, args.profile_manifest) if args.profile_manifest else None
            ),
            policy_path=_resolve_path(repo_root, args.policy) if args.policy else None,
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
