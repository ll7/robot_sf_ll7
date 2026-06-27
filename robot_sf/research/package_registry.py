"""Read-only registry and preflight for continuous research-engine packages (issue #3057).

This module loads a declarative registry of research-engine "packages" (the scenario
suite, planner-readiness matrix, campaign-manifest flow, canonical result store, and the
July-2026 sprint packages A/B/C plus the release package) and reports, for each one:

- which required tracked artifacts are present vs. missing on disk,
- which declared prerequisite packages are satisfied vs. blocked,
- a derived ``ready`` / ``blocked`` status.

The registry composes *existing* metadata (config files, contract docs, and the issue
numbers already declared in epic #3057). It is intentionally a read-only preflight: it
makes no benchmark or research claim, schedules nothing, and runs no campaign. A package
is reported ``ready`` only when every required artifact exists and every prerequisite is
itself ``ready``; any missing artifact or unsatisfied prerequisite fails closed to
``blocked``. Missing prerequisites are surfaced as gaps rather than silently ignored.

Example:
    >>> from pathlib import Path
    >>> from robot_sf.research.package_registry import (
    ...     load_registry,
    ...     evaluate_registry_preflight,
    ... )
    >>> registry = load_registry(Path("configs/research/research_package_registry_issue_3057.yaml"))
    >>> report = evaluate_registry_preflight(registry, repo_root=Path("."))
    >>> report["summary"]["ready_count"]  # doctest: +SKIP
    5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "research-package-registry.v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

# Fail-closed status vocabulary. ``ready`` is the only status that may be treated as a
# satisfied prerequisite; every other status is a blocking gap for downstream packages.
STATUS_READY = "ready"
STATUS_BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ResearchPackage:
    """One declared research-engine package and its preflight inputs.

    Attributes:
        package_id: Stable registry identifier referenced by prerequisites.
        title: Human-facing package name.
        issue: GitHub issue number that owns the package, or ``None``.
        resource: Declared execution resource label (e.g. ``local``, ``slurm``).
        required_artifacts: Repo-relative paths that must exist for the package to be ready.
        prerequisites: Identifiers of upstream packages that must be ready first.
    """

    package_id: str
    title: str
    issue: int | None
    resource: str | None
    required_artifacts: tuple[str, ...]
    prerequisites: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResearchPackageRegistry:
    """Parsed research-package registry document."""

    schema_version: str
    description: str
    packages: tuple[ResearchPackage, ...]

    def package_ids(self) -> set[str]:
        """Return the set of declared package identifiers."""
        return {package.package_id for package in self.packages}


def load_registry(path: Path) -> ResearchPackageRegistry:
    """Load and validate a research-package registry YAML document.

    Args:
        path: Path to the registry YAML file.

    Returns:
        The parsed :class:`ResearchPackageRegistry`.

    Raises:
        ValueError: If the document is malformed, uses an unsupported schema version,
            declares duplicate package ids, or references unknown prerequisites.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    schema_version = str(raw.get("schema_version", "")).strip()
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"unsupported registry schema_version {schema_version!r}; supported: {supported}"
        )

    raw_packages = raw.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError("registry must define a non-empty 'packages' list")

    packages = tuple(_parse_package(item, index=index) for index, item in enumerate(raw_packages))
    _validate_unique_ids(packages)
    registry = ResearchPackageRegistry(
        schema_version=schema_version,
        description=str(raw.get("description", "")).strip(),
        packages=packages,
    )
    _validate_prerequisites(registry)
    return registry


def evaluate_registry_preflight(
    registry: ResearchPackageRegistry,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate artifact presence and prerequisite readiness for every package.

    Each package is classified ``ready`` only when all required artifacts exist under
    ``repo_root`` and every prerequisite resolves to ``ready``; otherwise it fails closed
    to ``blocked`` with explicit reasons. Prerequisites are resolved transitively, so a
    package downstream of a blocked dependency is also blocked.

    Args:
        registry: Parsed registry document.
        repo_root: Directory that repo-relative artifact paths are resolved against.

    Returns:
        A JSON-serializable preflight report mapping with per-package results, an ordered
        gap list, and summary counts.
    """
    by_id = {package.package_id: package for package in registry.packages}
    statuses: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}

    # Resolve in dependency order so a prerequisite's status is known before its dependents.
    for package_id in _topological_order(registry):
        package = by_id[package_id]
        result = _evaluate_package(package, repo_root=repo_root, statuses=statuses)
        statuses[package_id] = result["status"]
        results[package_id] = result

    # Preserve the registry's declared package order in the rendered output.
    ordered_results = [results[package.package_id] for package in registry.packages]
    gaps = [gap for result in ordered_results for gap in result["gaps"]]
    ready_count = sum(1 for result in ordered_results if result["status"] == STATUS_READY)

    return {
        "schema_version": registry.schema_version,
        "view_status": "read_only_research_package_preflight",
        "claim_boundary": (
            "registry/preflight metadata only; reports declared artifacts and prerequisites "
            "and makes no benchmark, metric, or research claim"
        ),
        "description": registry.description,
        "repo_root": _display_path(repo_root),
        "packages": ordered_results,
        "gaps": gaps,
        "summary": {
            "package_count": len(ordered_results),
            "ready_count": ready_count,
            "blocked_count": len(ordered_results) - ready_count,
            "gap_count": len(gaps),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a preflight report as Markdown.

    Args:
        report: A report mapping produced by :func:`evaluate_registry_preflight`.

    Returns:
        The rendered Markdown document.
    """
    summary = report.get("summary", {})
    lines = [
        "# Research Package Registry Preflight",
        "",
        f"- Schema: `{report.get('schema_version')}`",
        f"- Status: `{report.get('view_status')}`",
        f"- Claim boundary: {report.get('claim_boundary')}",
        (
            f"- Packages: {summary.get('package_count', 0)} "
            f"({summary.get('ready_count', 0)} ready, "
            f"{summary.get('blocked_count', 0)} blocked)"
        ),
        "",
        "## Packages",
        "",
        "| package | issue | resource | status | artifacts_present | prerequisites |",
        "|---|---|---|---|---|---|",
    ]
    for package in report.get("packages", []):
        artifacts = package.get("artifacts", {})
        present = artifacts.get("present_count", 0)
        required = artifacts.get("required_count", 0)
        prereqs = package.get("prerequisites", [])
        prereq_text = ", ".join(prereqs) if prereqs else "none"
        issue = package.get("issue")
        issue_text = f"#{issue}" if issue is not None else "NA"
        lines.append(
            "| "
            f"{package['package_id']} | {issue_text} | {package.get('resource') or 'NA'} | "
            f"{package['status']} | {present}/{required} | {prereq_text} |"
        )

    lines.extend(["", "## Gaps", "", "| package | gap_type | detail |", "|---|---|---|"])
    gaps = report.get("gaps", [])
    if gaps:
        for gap in gaps:
            lines.append(f"| {gap['package_id']} | {gap['gap_type']} | {gap['detail']} |")
    else:
        lines.append("| NA | none | all declared packages are ready |")
    lines.append("")
    return "\n".join(lines)


def _evaluate_package(
    package: ResearchPackage,
    *,
    repo_root: Path,
    statuses: dict[str, str],
) -> dict[str, Any]:
    """Evaluate one package against on-disk artifacts and resolved prerequisites.

    Args:
        package: The package to evaluate.
        repo_root: Directory that repo-relative artifact paths are resolved against.
        statuses: Already-resolved statuses for upstream packages.

    Returns:
        A per-package result mapping with artifact, prerequisite, and gap details.
    """
    present_artifacts: list[str] = []
    missing_artifacts: list[str] = []
    for artifact in package.required_artifacts:
        if (repo_root / artifact).exists():
            present_artifacts.append(artifact)
        else:
            missing_artifacts.append(artifact)

    # A prerequisite counts as satisfied only when it resolved to ``ready``; anything
    # else (blocked) is a blocking gap. Unknown ids are rejected at load time.
    blocked_prerequisites = [
        prerequisite
        for prerequisite in package.prerequisites
        if statuses.get(prerequisite) != STATUS_READY
    ]

    gaps: list[dict[str, str]] = []
    for artifact in missing_artifacts:
        gaps.append(
            {
                "package_id": package.package_id,
                "gap_type": "missing_artifact",
                "detail": f"required artifact not found: {artifact}",
            }
        )
    for prerequisite in blocked_prerequisites:
        gaps.append(
            {
                "package_id": package.package_id,
                "gap_type": "blocked_prerequisite",
                "detail": f"prerequisite package not ready: {prerequisite}",
            }
        )

    status = STATUS_READY if not gaps else STATUS_BLOCKED
    return {
        "package_id": package.package_id,
        "title": package.title,
        "issue": package.issue,
        "resource": package.resource,
        "status": status,
        "artifacts": {
            "required_count": len(package.required_artifacts),
            "present_count": len(present_artifacts),
            "present": present_artifacts,
            "missing": missing_artifacts,
        },
        "prerequisites": list(package.prerequisites),
        "blocked_prerequisites": blocked_prerequisites,
        "gaps": gaps,
    }


def _parse_package(item: Any, *, index: int) -> ResearchPackage:
    """Parse and validate a single package mapping.

    Args:
        item: The raw package mapping from the registry document.
        index: Position of the package in the document, used for error messages.

    Returns:
        The parsed :class:`ResearchPackage`.
    """
    if not isinstance(item, dict):
        raise ValueError(f"packages[{index}] must be a mapping")
    package_id = str(item.get("id", "")).strip()
    if not package_id:
        raise ValueError(f"packages[{index}] must define a non-empty 'id'")
    title = str(item.get("title", "")).strip()
    if not title:
        raise ValueError(f"package {package_id!r} must define a non-empty 'title'")

    issue = item.get("issue")
    if issue is not None:
        try:
            issue = int(issue)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"package {package_id!r} 'issue' must be an integer") from exc

    resource = item.get("resource")
    resource = str(resource).strip() if resource is not None else None

    required_artifacts = _parse_str_list(
        item.get("required_artifacts", []), field="required_artifacts", package_id=package_id
    )
    prerequisites = _parse_str_list(
        item.get("prerequisites", []), field="prerequisites", package_id=package_id
    )
    return ResearchPackage(
        package_id=package_id,
        title=title,
        issue=issue,
        resource=resource,
        required_artifacts=required_artifacts,
        prerequisites=prerequisites,
    )


def _parse_str_list(value: Any, *, field: str, package_id: str) -> tuple[str, ...]:
    """Parse a list of non-empty strings, rejecting malformed entries.

    Args:
        value: The raw value to parse (``None`` or a list of scalars).
        field: Field name, used for error messages.
        package_id: Owning package id, used for error messages.

    Returns:
        A tuple of stripped, non-empty strings.
    """
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"package {package_id!r} '{field}' must be a list")
    items: list[str] = []
    for entry in value:
        text = str(entry).strip()
        if not text:
            raise ValueError(f"package {package_id!r} '{field}' contains an empty entry")
        items.append(text)
    return tuple(items)


def _validate_unique_ids(packages: tuple[ResearchPackage, ...]) -> None:
    """Reject duplicate package identifiers."""
    seen: set[str] = set()
    for package in packages:
        if package.package_id in seen:
            raise ValueError(f"duplicate package id {package.package_id!r}")
        seen.add(package.package_id)


def _validate_prerequisites(registry: ResearchPackageRegistry) -> None:
    """Reject prerequisites that reference unknown packages or form a cycle."""
    known = registry.package_ids()
    for package in registry.packages:
        for prerequisite in package.prerequisites:
            if prerequisite not in known:
                raise ValueError(
                    f"package {package.package_id!r} references unknown "
                    f"prerequisite {prerequisite!r}"
                )
            if prerequisite == package.package_id:
                raise ValueError(f"package {package.package_id!r} lists itself as a prerequisite")
    # Detecting a cycle here gives a clear error instead of a downstream ordering failure.
    _topological_order(registry)


def _topological_order(registry: ResearchPackageRegistry) -> list[str]:
    """Return package ids in dependency order, raising on cycles."""
    dependencies = {package.package_id: set(package.prerequisites) for package in registry.packages}
    resolved: list[str] = []
    resolved_set: set[str] = set()
    # Iterate until no further package can be resolved; a stall implies a cycle.
    while len(resolved) < len(dependencies):
        progressed = False
        for package in registry.packages:
            package_id = package.package_id
            if package_id in resolved_set:
                continue
            if dependencies[package_id] <= resolved_set:
                resolved.append(package_id)
                resolved_set.add(package_id)
                progressed = True
        if not progressed:
            remaining = sorted(set(dependencies) - resolved_set)
            raise ValueError(f"prerequisite cycle detected among packages: {remaining}")
    return resolved


def _display_path(path: Path) -> str:
    """Return a compact display path when possible."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix() or "."
    except ValueError:
        return path.as_posix()
