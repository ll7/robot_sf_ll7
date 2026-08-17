#!/usr/bin/env python3
"""Capture a fail-closed license inventory for the locked Python environment.

The report joins ``uv.lock`` package identities with the metadata observed in the
current interpreter. It records SPDX ``License-Expression`` values when a wheel
provides them, but never translates free-form license prose into an SPDX decision.
Missing, custom, proprietary, or non-normalized metadata remains review-required.

Examples:
    uv run python scripts/tools/check_dependency_license_inventory.py \
        --output output/validation/dependency-license-inventory.json
    uv run python scripts/tools/check_dependency_license_inventory.py \
        --output /tmp/dependency-license-inventory.json --fail-on-unresolved
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
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


SCHEMA_VERSION = "dependency_license_inventory.v1"
_UNKNOWN_VALUES = frozenset({"", "unknown", "unknown license", "none", "null"})
_REVIEW_MARKERS = (
    "licenseref-",
    "proprietary",
    "nvidia",
    "non-commercial",
    "non commercial",
    "no redistribution",
)


def _canonicalize_name(name: str) -> str:
    """Normalize a distribution name according to PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one repository input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_packages(path: Path) -> list[dict[str, Any]]:
    """Read and normalize the package identities resolved by ``uv.lock``."""
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
        source_type = "registry"
        if isinstance(source, dict) and isinstance(source.get("editable"), str):
            source_type = "editable"
        normalized.append(
            {
                "name": package["name"],
                "version": version,
                "source_type": source_type,
            }
        )
    return sorted(normalized, key=lambda item: (item["name"].lower(), item["version"] or ""))


def _metadata_value(metadata: Any, key: str) -> str | None:
    """Return a trimmed metadata value unless it is an explicit empty marker."""
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value.lower() in _UNKNOWN_VALUES else value


def _license_record(distribution: Any) -> dict[str, Any]:
    """Summarize one installed distribution without making a legal inference."""
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

    if not expression and not license_value and not classifiers:
        status = "unknown"
        reasons = ["distribution metadata contains no license expression, text, or classifier"]
    elif any(marker in searchable for marker in _REVIEW_MARKERS):
        status = "review_required"
        reasons = ["metadata contains a custom, proprietary, or restricted-license marker"]
    elif expression:
        status = "resolved"
        reasons = []
    else:
        status = "review_required"
        reasons = ["license metadata is not a normalized SPDX License-Expression"]

    record: dict[str, Any] = {
        "license_status": status,
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
    observed: dict[str, list[Any]] = {}
    for distribution in source:
        name = distribution.metadata.get("Name") or distribution.name
        if not isinstance(name, str) or not name.strip():
            continue
        observed.setdefault(_canonicalize_name(name), []).append(distribution)
    for values in observed.values():
        values.sort(key=lambda item: (str(item.version), str(item.name).lower()))
    return observed


def build_inventory(
    repo_root: Path,
    *,
    distributions: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic lock-to-environment license inventory."""
    pyproject_path = repo_root / "pyproject.toml"
    lock_path = repo_root / "uv.lock"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing its [project] table")
    project_name = project.get("name")
    project_license = project.get("license")
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("pyproject.toml is missing project.name")
    if not isinstance(project_license, str) or not project_license.strip():
        raise ValueError("pyproject.toml is missing a string project.license")

    locked = _lock_packages(lock_path)
    observed = _observed_distributions(distributions)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    locked_names = {_canonicalize_name(package["name"]) for package in locked}

    for package in locked:
        name = package["name"]
        key = _canonicalize_name(name)
        matches = observed.get(key, [])
        record: dict[str, Any] = {
            "name": name,
            "locked_version": package["version"],
            "source_type": package["source_type"],
        }
        if not matches:
            record.update(
                {
                    "observed_version": None,
                    "license_status": "not_installed",
                    "license_expression": None,
                    "license_classifiers": [],
                    "review_reasons": [
                        "locked package is not installed in the captured environment"
                    ],
                }
            )
            failures.append(f"{name}: locked package is not installed")
        elif len(matches) > 1:
            record.update(
                {
                    "observed_version": None,
                    "license_status": "review_required",
                    "license_expression": None,
                    "license_classifiers": [],
                    "review_reasons": ["multiple installed distributions share this name"],
                }
            )
            failures.append(f"{name}: multiple installed distributions share this name")
        else:
            distribution = matches[0]
            observed_version = str(distribution.version)
            record["observed_version"] = observed_version
            if package["version"] is not None and observed_version != package["version"]:
                record["version_match"] = False
                failures.append(
                    f"{name}: lock version {package['version']} != observed {observed_version}"
                )
            else:
                record["version_match"] = True
            record.update(_license_record(distribution))
            if record["license_status"] != "resolved":
                failures.append(f"{name}: {record['license_status']} license metadata")
        records.append(record)

    installed_not_locked = sorted(
        {
            f"{distribution.metadata.get('Name') or distribution.name}=={distribution.version}"
            for key, values in observed.items()
            if key not in locked_names
            for distribution in values
        },
        key=str.lower,
    )

    status_counts = Counter(record["license_status"] for record in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_inputs": {
            "pyproject.toml": {"sha256": _sha256_file(pyproject_path)},
            "uv.lock": {"sha256": _sha256_file(lock_path)},
        },
        "project": {
            "name": project_name,
            "license_expression": project_license,
            "note": "Repository license metadata is recorded separately from dependency rights.",
        },
        "environment": {
            "python": sys.version,
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "summary": {
            "locked_package_count": len(records),
            "installed_distribution_count": sum(len(values) for values in observed.values()),
            "installed_not_locked_count": len(installed_not_locked),
            "license_status_counts": dict(sorted(status_counts.items())),
            "unresolved_count": len(failures),
            "status": "blocked" if failures else "complete",
        },
        "installed_not_locked": installed_not_locked,
        "failures": failures,
        "packages": records,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the inventory command and optionally fail on unresolved entries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing pyproject.toml and uv.lock.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Return non-zero when any lock package lacks resolved SPDX metadata.",
    )
    args = parser.parse_args(argv)

    try:
        inventory = build_inventory(args.repo_root.resolve())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"FAIL: dependency license inventory could not be built: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")

    if args.fail_on_unresolved and inventory["summary"]["unresolved_count"]:
        print(
            "FAIL: dependency license inventory remains unresolved for "
            f"{inventory['summary']['unresolved_count']} package(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
