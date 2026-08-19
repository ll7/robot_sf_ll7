#!/usr/bin/env python3
"""Validate dependency update lanes and route them to existing CI evidence.

This checker owns the repository's dependency-risk classification. It does not
run a second dependency test suite. Instead, it proves that the Dependabot
groups and the existing CI aggregate still provide the focused compatibility
surfaces required by the policy manifest.

When a pull request changes a lockfile or project dependency declaration, the
checker compares the exact base and head lock rows, classifies direct packages
from the canonical policy, and rejects mixed direct-risk lanes. Unknown direct
packages are rejected until the policy is deliberately extended.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "scripts/validation/dependabot_update_policy.v1.json"
DEFAULT_DEPENDABOT_CONFIG = REPO_ROOT / ".github/dependabot.yml"
DEFAULT_CI_WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"

DEPENDENCY_FILES = (
    "pyproject.toml",
    "uv.lock",
    "fast-pysf/pyproject.toml",
    "fast-pysf/uv.lock",
)
PROJECT_FILES = ("pyproject.toml", "fast-pysf/pyproject.toml")
LOCK_FILES = ("uv.lock", "fast-pysf/uv.lock")
PACKAGE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class PolicyError(ValueError):
    """Raised when dependency-policy evidence is incomplete or inconsistent."""


def normalize_package_name(name: str) -> str:
    """Return the PEP 503 comparison form for a package name."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def requirement_package_name(value: Any) -> str | None:
    """Extract a package name from a PEP 508 requirement string."""
    if not isinstance(value, str):
        return None
    match = PACKAGE_TOKEN.match(value.strip())
    return normalize_package_name(match.group(0)) if match else None


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{label} must be an object")
    return value


def _as_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PolicyError(f"{label} must be a list of strings")
    return list(value)


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    """Load the JSON policy and perform the executable shape checks."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"unable to load policy {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PolicyError("policy root must be an object")
    validate_policy(payload)
    return dict(payload)


def validate_policy(policy: Mapping[str, Any]) -> None:  # noqa: C901, PLR0912
    """Validate policy invariants that affect classification and CI routing."""
    required = {
        "schema_version",
        "policy_name",
        "dependabot_config",
        "ci_workflow",
        "aggregate_job",
        "classes",
        "transitive_fallback",
        "security_updates",
        "rollback",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise PolicyError(f"policy is missing required keys: {', '.join(missing)}")
    if policy["schema_version"] != "robot-sf.dependabot-update-policy.v1":
        raise PolicyError(f"unsupported policy schema: {policy['schema_version']!r}")
    if policy["dependabot_config"] != ".github/dependabot.yml":
        raise PolicyError("policy must own .github/dependabot.yml")
    if policy["ci_workflow"] != ".github/workflows/ci.yml":
        raise PolicyError("policy must route through .github/workflows/ci.yml")

    classes = policy["classes"]
    if not isinstance(classes, list) or not classes:
        raise PolicyError("policy classes must be a non-empty list")
    class_ids: set[str] = set()
    package_owners: dict[str, str] = {}
    for index, raw_class in enumerate(classes):
        item = _as_mapping(raw_class, f"classes[{index}]")
        for key in ("id", "risk", "update_lane", "packages", "required_jobs", "focused_test_paths"):
            if key not in item:
                raise PolicyError(f"classes[{index}] is missing {key!r}")
        class_id = item["id"]
        if not isinstance(class_id, str) or not class_id:
            raise PolicyError(f"classes[{index}].id must be a non-empty string")
        if class_id in class_ids:
            raise PolicyError(f"duplicate dependency class id: {class_id}")
        class_ids.add(class_id)
        packages = _as_string_list(item["packages"], f"classes[{index}].packages")
        if len(set(packages)) != len(packages):
            raise PolicyError(f"classes[{index}].packages contains duplicates")
        if not _as_string_list(item["required_jobs"], f"classes[{index}].required_jobs"):
            raise PolicyError(f"classes[{index}] requires at least one CI job")
        if not _as_string_list(item["focused_test_paths"], f"classes[{index}].focused_test_paths"):
            raise PolicyError(f"classes[{index}] requires at least one focused test path")
        for package in packages:
            normalized = normalize_package_name(package)
            previous = package_owners.get(normalized)
            if previous is not None:
                raise PolicyError(
                    f"package {normalized!r} appears in both {previous!r} and {class_id!r}"
                )
            package_owners[normalized] = class_id

    fallback = _as_mapping(policy["transitive_fallback"], "transitive_fallback")
    for key in ("id", "risk", "update_lane", "required_jobs", "focused_test_paths"):
        if key not in fallback:
            raise PolicyError(f"transitive_fallback is missing {key!r}")
    if fallback["risk"] != "unknown":
        raise PolicyError("transitive_fallback.risk must remain 'unknown'")
    _as_string_list(fallback["required_jobs"], "transitive_fallback.required_jobs")
    _as_string_list(fallback["focused_test_paths"], "transitive_fallback.focused_test_paths")

    security = _as_mapping(policy["security_updates"], "security_updates")
    if security.get("independent") is not True:
        raise PolicyError("security updates must remain independently actionable")
    _as_string_list(security.get("required_jobs"), "security_updates.required_jobs")


def _workflow_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _workflow_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _workflow_strings(child)


def _root_uv_update(document: Mapping[str, Any]) -> Mapping[str, Any]:
    updates = document.get("updates")
    if not isinstance(updates, list):
        raise PolicyError("Dependabot configuration must define an updates list")
    for update in updates:
        if (
            isinstance(update, Mapping)
            and update.get("package-ecosystem") == "uv"
            and update.get("directory") == "/"
        ):
            return update
    raise PolicyError("Dependabot configuration has no root uv update entry")


def _patterns_for_group(group: Any, label: str) -> list[str]:
    mapping = _as_mapping(group, label)
    return _as_string_list(mapping.get("patterns"), f"{label}.patterns")


def _matches_group(package: str, group: Any, label: str) -> bool:
    return any(
        fnmatch.fnmatchcase(package, pattern.lower())
        for pattern in _patterns_for_group(group, label)
    )


def validate_dependabot_config(path: Path = DEFAULT_DEPENDABOT_CONFIG) -> None:  # noqa: C901
    """Ensure the root groups express risk lanes without a catch-all."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"unable to load Dependabot configuration {path}: {exc}") from exc
    document = _as_mapping(document, "Dependabot configuration")
    root_update = _root_uv_update(document)
    groups = _as_mapping(root_update.get("groups"), "root uv groups")
    if "python-minor-patch" in groups:
        raise PolicyError("the mixed-risk python-minor-patch group must be removed")
    required_groups = {"developer-tooling", "serialization-data", "experiment-integrations"}
    missing = sorted(required_groups - set(groups))
    if missing:
        raise PolicyError(f"root uv groups are missing: {', '.join(missing)}")
    for group_name, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        applies_to = group.get("applies-to")
        if applies_to == "security-updates":
            raise PolicyError("security updates must not be attached to a normal group")
        if applies_to not in (None, "version-updates"):
            raise PolicyError(
                f"root uv group {group_name!r} has unsupported applies-to value {applies_to!r}"
            )

    developer = groups["developer-tooling"]
    for package in ("ruff", "mypy", "pylint", "pre-commit", "pytest"):
        if not _matches_group(package, developer, "developer-tooling"):
            raise PolicyError(f"developer-tooling group does not cover {package}")

    serialization = groups["serialization-data"]
    experiment = groups["experiment-integrations"]
    if not _matches_group("orjson", serialization, "serialization-data"):
        raise PolicyError("serialization-data group must cover orjson")
    if not _matches_group("wandb", experiment, "experiment-integrations"):
        raise PolicyError("experiment-integrations group must cover wandb")

    high_impact = {"numba", "orjson", "wandb", "pyarrow"}
    for package in high_impact:
        matching_groups = [
            group_name
            for group_name, group in groups.items()
            if _matches_group(package, group, f"root uv groups.{group_name}")
        ]
        if package in {"numba", "pyarrow"} and matching_groups:
            raise PolicyError(
                f"{package} must remain an individual update, but matches {matching_groups}"
            )
        if package == "orjson" and matching_groups != ["serialization-data"]:
            raise PolicyError(f"orjson must only match serialization-data, got {matching_groups}")
        if package == "wandb" and matching_groups != ["experiment-integrations"]:
            raise PolicyError(
                f"wandb must only match experiment-integrations, got {matching_groups}"
            )


def validate_ci_workflow(policy: Mapping[str, Any], path: Path = DEFAULT_CI_WORKFLOW) -> None:  # noqa: C901
    """Ensure policy evidence points at jobs the CI aggregate requires."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"unable to load CI workflow {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise PolicyError("CI workflow must be an object")
    jobs = _as_mapping(document.get("jobs"), "CI jobs")
    aggregate_name = str(policy["aggregate_job"])
    aggregate = _as_mapping(jobs.get(aggregate_name), f"CI jobs.{aggregate_name}")
    aggregate_needs = aggregate.get("needs", [])
    if isinstance(aggregate_needs, str):
        aggregate_needs = [aggregate_needs]
    aggregate_needs = set(_as_string_list(aggregate_needs, f"CI jobs.{aggregate_name}.needs"))

    required_jobs: set[str] = set()
    policy_items = list(policy["classes"]) + [
        _as_mapping(policy["transitive_fallback"], "transitive_fallback"),
        _as_mapping(policy["security_updates"], "security_updates"),
    ]
    for item in policy_items:
        required_jobs.update(_as_string_list(item["required_jobs"], "required_jobs"))
    missing_jobs = sorted(required_jobs - set(jobs))
    if missing_jobs:
        raise PolicyError(f"policy references missing CI jobs: {', '.join(missing_jobs)}")
    missing_needs = sorted(required_jobs - aggregate_needs)
    if missing_needs:
        raise PolicyError(
            f"CI aggregate {aggregate_name!r} does not require policy jobs: {', '.join(missing_needs)}"
        )

    compatibility_text = "\n".join(_workflow_strings(jobs["compat-matrix"]))
    for raw_class in policy["classes"]:
        item = _as_mapping(raw_class, "dependency class")
        if "compat-matrix" not in item["required_jobs"]:
            continue
        for path_value in _as_string_list(item["focused_test_paths"], "focused_test_paths"):
            if path_value not in compatibility_text:
                raise PolicyError(
                    f"compat-matrix no longer exposes focused path {path_value!r} "
                    f"for class {item['id']!r}"
                )


def _project_dependency_names(document: Mapping[str, Any]) -> set[str]:
    project = document.get("project", {})
    if not isinstance(project, Mapping):
        return set()
    project_name = normalize_package_name(str(project.get("name", "")))
    names: set[str] = set()
    sources: list[Any] = [project.get("dependencies", [])]
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, Mapping):
        sources.extend(optional.values())
    groups = document.get("dependency-groups", {})
    if isinstance(groups, Mapping):
        sources.extend(groups.values())
    for source in sources:
        if not isinstance(source, list):
            continue
        for requirement in source:
            name = requirement_package_name(requirement)
            if name and name != project_name:
                names.add(name)
    return names


def dependency_names_from_text(text: str) -> set[str]:
    """Return direct dependency names from one pyproject.toml payload."""
    if not text:
        return set()
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"invalid pyproject.toml while classifying dependencies: {exc}") from exc
    return _project_dependency_names(document)


def direct_dependency_names(repo_root: Path = REPO_ROOT) -> set[str]:
    """Return direct names from the root and standalone fast-pysf projects."""
    names: set[str] = set()
    for relative_path in PROJECT_FILES:
        path = repo_root / relative_path
        if path.is_file():
            names.update(dependency_names_from_text(path.read_text(encoding="utf-8")))
    return names


def lock_package_rows(text: str) -> dict[str, list[str]]:
    """Return canonicalized uv lock rows grouped by normalized package name."""
    if not text:
        return {}
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"invalid uv.lock while classifying dependencies: {exc}") from exc
    rows = document.get("package", [])
    if not isinstance(rows, list):
        raise PolicyError("uv.lock package table must be a list")
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise PolicyError("uv.lock contains a package row without a name")
        name = normalize_package_name(row["name"])
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        grouped.setdefault(name, []).append(canonical)
    for values in grouped.values():
        values.sort()
    return grouped


def changed_lock_package_names(base_text: str, head_text: str) -> set[str]:
    """Return package names whose complete uv lock row set changed."""
    base_rows = lock_package_rows(base_text)
    head_rows = lock_package_rows(head_text)
    return {
        name
        for name in set(base_rows) | set(head_rows)
        if base_rows.get(name, []) != head_rows.get(name, [])
    }


def classify_package_names(
    package_names: Iterable[str],
    direct_names: set[str],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Classify changed packages and fail closed for unknown direct names."""
    package_owners = {
        normalize_package_name(package): item["id"]
        for item in policy["classes"]
        for package in item["packages"]
    }
    classes = {item["id"]: item for item in policy["classes"]}
    fallback = _as_mapping(policy["transitive_fallback"], "transitive_fallback")
    classifications: list[dict[str, Any]] = []
    for raw_name in sorted(set(package_names)):
        name = normalize_package_name(raw_name)
        direct = name in direct_names
        class_id = package_owners.get(name)
        if class_id is None:
            if direct:
                raise PolicyError(
                    f"direct dependency {name!r} is not classified in the canonical policy"
                )
            item = fallback
        else:
            item = classes[class_id]
        classifications.append(
            {
                "name": name,
                "direct": direct,
                "class": item["id"],
                "risk": item["risk"],
                "update_lane": item["update_lane"],
                "required_jobs": list(item["required_jobs"]),
            }
        )
    return classifications


def validate_direct_dependency_coverage(direct_names: set[str], policy: Mapping[str, Any]) -> None:
    """Reject a new direct dependency until its validation class is reviewed."""
    known = {
        normalize_package_name(package)
        for item in policy["classes"]
        for package in item["packages"]
    }
    missing = sorted(direct_names - known)
    if missing:
        raise PolicyError(
            "direct dependencies missing from the canonical policy: " + ", ".join(missing)
        )


def validate_direct_update_lanes(classifications: Iterable[Mapping[str, Any]]) -> list[str]:
    """Reject one update that combines multiple direct risk classes."""
    direct_class_ids = {
        str(item["class"]) for item in classifications if item.get("direct") and item.get("class")
    }
    if len(direct_class_ids) > 1:
        raise PolicyError(
            "one dependency update mixes direct risk classes: "
            + ", ".join(sorted(direct_class_ids))
        )
    return sorted(direct_class_ids)


def _git_text(repo_root: Path, args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def git_file_at_ref(repo_root: Path, ref: str, relative_path: str) -> str | None:
    """Read one exact file from a git ref, returning None when it is absent."""
    return _git_text(repo_root, ["show", f"{ref}:{relative_path}"])


def _diff_vs_head(
    repo_root: Path,
    base_ref: str,
    options: list[str],
    pathspec: str | None = None,
) -> str | None:
    """Return a base-vs-HEAD diff, tolerating shallow checkouts.

    The three-dot form ``<base>...HEAD`` requires a merge base, which is absent
    on GitHub's shallow ``pull_request`` checkout (issue #7524). When it fails,
    fall back to the two-dot form ``<base> HEAD`` which compares the two trees
    directly and works without a merge base.
    """
    suffix = [f"{base_ref}...HEAD"] if pathspec is None else [f"{base_ref}...HEAD", "--", pathspec]
    output = _git_text(repo_root, ["diff", *options, *suffix])
    if output is not None:
        return output
    suffix = [base_ref, "HEAD"] if pathspec is None else [base_ref, "HEAD", "--", pathspec]
    return _git_text(repo_root, ["diff", *options, *suffix])


def changed_files(
    repo_root: Path, base_ref: str, changed_files_path: Path | None = None
) -> list[str]:
    """Return authoritative changed paths, falling back to an exact git diff."""
    if changed_files_path is not None:
        try:
            return [
                line.strip()
                for line in changed_files_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            raise PolicyError(
                f"unable to read authoritative changed-files list {changed_files_path}: {exc}"
            ) from exc
    output = _diff_vs_head(repo_root, base_ref, ["--name-only"])
    if output is None:
        raise PolicyError(f"unable to compare HEAD with base ref {base_ref!r}")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _changed_project_names(
    repo_root: Path,
    base_ref: str,
    relative_path: str,
    direct_names: set[str],
) -> set[str]:
    diff = _diff_vs_head(repo_root, base_ref, ["--unified=0"], pathspec=relative_path)
    if diff is None:
        raise PolicyError(f"unable to inspect dependency declaration diff {relative_path}")
    changed: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        for token in PACKAGE_TOKEN.findall(line[1:]):
            normalized = normalize_package_name(token)
            if normalized in direct_names:
                changed.add(normalized)
    return changed


def changed_dependency_packages(
    repo_root: Path,
    base_ref: str,
    files: Iterable[str],
    direct_names: set[str],
) -> set[str]:
    """Derive changed direct and transitive package names from exact git rows."""
    changed: set[str] = set()
    file_set = set(files)
    for relative_path in LOCK_FILES:
        if relative_path not in file_set:
            continue
        base_text = git_file_at_ref(repo_root, base_ref, relative_path) or ""
        head_path = repo_root / relative_path
        head_text = head_path.read_text(encoding="utf-8") if head_path.is_file() else ""
        changed.update(changed_lock_package_names(base_text, head_text))
    for relative_path in PROJECT_FILES:
        if relative_path in file_set:
            changed.update(_changed_project_names(repo_root, base_ref, relative_path, direct_names))
    return changed


def validate_repository_structure(
    repo_root: Path = REPO_ROOT,
    policy_path: Path = DEFAULT_POLICY,
    dependabot_path: Path = DEFAULT_DEPENDABOT_CONFIG,
    ci_workflow_path: Path = DEFAULT_CI_WORKFLOW,
) -> dict[str, Any]:
    """Validate policy, Dependabot grouping, CI ownership, and direct coverage."""
    policy = load_policy(policy_path)
    validate_dependabot_config(dependabot_path)
    validate_ci_workflow(policy, ci_workflow_path)
    direct_names = direct_dependency_names(repo_root)
    validate_direct_dependency_coverage(direct_names, policy)
    return policy


def evaluate_update(
    repo_root: Path = REPO_ROOT,
    base_ref: str = "origin/main",
    policy: Mapping[str, Any] | None = None,
    changed_files_path: Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed report for changed dependency files."""
    policy = policy or load_policy()
    files = changed_files(repo_root, base_ref, changed_files_path)
    dependency_files = sorted(set(files) & set(DEPENDENCY_FILES))
    report: dict[str, Any] = {
        "schema_version": "robot-sf.dependabot-update-report.v1",
        "base_ref": base_ref,
        "dependency_files": dependency_files,
        "status": "not_applicable",
    }
    if not dependency_files:
        report["message"] = "no project dependency declaration or lockfile changed"
        return report

    current_direct = direct_dependency_names(repo_root)
    base_direct: set[str] = set()
    for relative_path in PROJECT_FILES:
        base_text = git_file_at_ref(repo_root, base_ref, relative_path)
        base_direct.update(dependency_names_from_text(base_text or ""))
    all_direct = current_direct | base_direct
    changed_names = changed_dependency_packages(repo_root, base_ref, dependency_files, all_direct)
    if not changed_names:
        raise PolicyError(
            "dependency files changed but no package rows or direct declarations could be identified"
        )
    classifications = classify_package_names(changed_names, all_direct, policy)
    direct_class_ids = validate_direct_update_lanes(classifications)
    required_jobs = sorted({job for item in classifications for job in item["required_jobs"]})
    report.update(
        {
            "status": "pass",
            "changed_packages": classifications,
            "required_jobs": required_jobs,
            "direct_risk_classes": sorted(direct_class_ids),
        }
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--dependabot-config", type=Path, default=DEFAULT_DEPENDABOT_CONFIG)
    parser.add_argument("--ci-workflow", type=Path, default=DEFAULT_CI_WORKFLOW)
    parser.add_argument("--changed-files-file", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run structural and exact-base dependency update validation."""
    args = _build_parser().parse_args(argv)
    try:
        policy = validate_repository_structure(
            policy_path=args.policy,
            dependabot_path=args.dependabot_config,
            ci_workflow_path=args.ci_workflow,
        )
        report = evaluate_update(
            base_ref=args.base_ref,
            policy=policy,
            changed_files_path=args.changed_files_file,
        )
    except PolicyError as exc:
        if args.as_json:
            print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"DEPENDABOT POLICY: BLOCKED: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DEPENDABOT POLICY: {report['status'].upper()}")
        print(f"Base: {report['base_ref']}")
        if report["status"] == "pass":
            packages = ", ".join(item["name"] for item in report["changed_packages"])
            print(f"Changed packages: {packages}")
            print(f"Required existing CI jobs: {', '.join(report['required_jobs'])}")
        else:
            print(report["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
