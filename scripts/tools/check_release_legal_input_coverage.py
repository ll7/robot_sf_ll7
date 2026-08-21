#!/usr/bin/env python3
"""Verify that release/legal inputs trigger and reach their declared validators."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEFAULT_OWNERSHIP_PATH = Path("scripts/validation/release_legal_inputs.v1.json")
REQUIRED_INPUT_PATTERNS = (
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "fast-pysf/**",
    "third_party/python-rvo2/**",
    "third_party/socnavbench/**",
    "model/**",
)


def _normalise_path(value: str) -> str:
    """Normalise a repository path while preserving GitHub glob syntax."""
    return value.strip().removeprefix("./").lstrip("/")


def pattern_covers(pattern: str, path: str) -> bool:
    """Return whether a GitHub workflow glob can cover a repository path."""
    pattern = _normalise_path(pattern)
    path = _normalise_path(path)
    if not pattern or pattern.startswith("!"):
        return False
    if pattern in {"**", "*"}:
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    if pattern.endswith("/"):
        pattern = pattern.rstrip("/") + "/**"
        return pattern_covers(pattern, path)
    if pattern == path:
        return True
    return fnmatch.fnmatchcase(path, pattern)


def _workflow_trigger(workflow: dict[str, Any]) -> dict[str, Any]:
    """Read a workflow's `on` mapping across PyYAML 1.1/1.2 key handling."""
    trigger = workflow.get("on")
    if trigger is None:
        trigger = workflow.get(True)
    if trigger is None:
        return {}
    if isinstance(trigger, str):
        return {trigger: {}}
    if isinstance(trigger, list):
        return {item: {} for item in trigger if isinstance(item, str)}
    return trigger if isinstance(trigger, dict) else {}


def _trigger_paths(trigger: Any) -> list[str] | None:
    """Return configured path filters; None means the event covers every path."""
    if trigger is None:
        return None
    if isinstance(trigger, dict):
        paths = trigger.get("paths")
        if paths is None:
            return None
        if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            return paths
    return []


def _git_value(repo_root: Path, *args: str) -> str | None:
    """Read a git value without making the coverage check depend on git."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    """Load a mapping YAML document or return None for malformed input."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def _workflow_job(workflow: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict) or not isinstance(jobs.get(job_id), dict):
        return None
    return jobs[job_id]


def _validator_is_invoked(validator: str, workflow_text: str) -> bool:
    """Check that the declared validator command/action is present in the workflow."""
    return bool(validator.strip()) and validator in workflow_text


def _dynamic_field_values(licensing: dict[str, Any], field_path: str) -> tuple[str, list[Any]]:
    """Return a declared registry field name and its scalar/list values."""
    field = field_path.removeprefix("models[].licensing.").removesuffix("[]")
    value = licensing.get(field)
    return field, value if field_path.endswith("[]") else [value]


def _dynamic_legal_paths(
    repo_root: Path,
    source: str,
    index: int,
    licensing: dict[str, Any],
    field_path: str,
) -> tuple[list[str], list[str]]:
    """Validate one dynamic registry field and return repository-relative paths."""
    field, values = _dynamic_field_values(licensing, field_path)
    paths: list[str] = []
    issues: list[str] = []
    for item in values:
        if item is None:
            continue
        if not isinstance(item, str) or not item.strip():
            issues.append(f"{source}: models[{index}].licensing.{field} is not a path")
            continue
        candidate = Path(os.path.abspath(str(repo_root / item)))
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            issues.append(
                f"{source}: models[{index}].licensing.{field} escapes the repository: {item}"
            )
            continue
        paths.append(candidate.relative_to(repo_root).as_posix())
    return paths, issues


def _dynamic_registry_paths(
    repo_root: Path, source: str, fields: list[str]
) -> tuple[list[str], list[str]]:
    """Read dynamic legal path fields from a registry and report malformed values."""
    payload = _load_yaml(repo_root / source)
    if payload is None:
        return [], [f"dynamic source is missing or invalid YAML: {source}"]
    models = payload.get("models")
    if not isinstance(models, list):
        return [], [f"dynamic source is missing models list: {source}"]
    paths: list[str] = []
    issues: list[str] = []
    for index, entry in enumerate(models):
        if not isinstance(entry, dict):
            issues.append(f"{source}: models[{index}] is not a mapping")
            continue
        licensing = entry.get("licensing")
        if not isinstance(licensing, dict):
            continue
        for field_path in fields:
            field_paths, field_issues = _dynamic_legal_paths(
                repo_root, source, index, licensing, field_path
            )
            paths.extend(field_paths)
            issues.extend(field_issues)
    return paths, issues


def _check_surface_job_contract(
    surface: dict[str, Any],
    surface_id: str,
    workflow: dict[str, Any],
    workflow_text: str,
) -> list[str]:
    """Check the declared job context and validator invocation."""
    issues: list[str] = []
    job_id = surface.get("job_id")
    job = _workflow_job(workflow, str(job_id)) if job_id else None
    if job is None:
        issues.append(f"{surface_id}: job is missing: {job_id}")
    check_context = str(surface.get("check_context", "")).strip()
    if not check_context:
        issues.append(f"{surface_id}: check_context is required")
    elif job is not None:
        job_text = json.dumps(job, sort_keys=True)
        context_token = check_context.split(" [", 1)[0]
        if check_context not in job_text and context_token not in job_text:
            issues.append(f"{surface_id}: check_context is not represented by the job")
    validator = surface.get("validator")
    if not isinstance(validator, str) or not _validator_is_invoked(validator, workflow_text):
        issues.append(f"{surface_id}: validator is not invoked by workflow: {validator}")
    return issues


def _surface_path_config(
    surface: dict[str, Any], repo_root: Path
) -> tuple[str, list[str], list[str] | None, list[Any], list[str]]:
    """Read and validate a surface's workflow, direct paths, and dynamic declarations."""
    surface_id = str(surface.get("id", "<missing>"))
    issues: list[str] = []
    direct_globs = surface.get("direct_globs")
    if not isinstance(direct_globs, list) or not direct_globs:
        issues.append(f"{surface_id}: direct_globs must be a non-empty list")
        direct_globs = []
    valid_direct_globs = [item for item in direct_globs if isinstance(item, str)]
    issues.extend(
        f"{surface_id}: direct_globs contains a non-string"
        for item in direct_globs
        if not isinstance(item, str)
    )
    workflow_raw = surface.get("workflow")
    workflow_path = repo_root / workflow_raw if isinstance(workflow_raw, str) else None
    if workflow_path is None or not workflow_path.is_file():
        return (
            surface_id,
            valid_direct_globs,
            [],
            [],
            issues + [f"{surface_id}: workflow is missing: {workflow_raw}"],
        )
    workflow = _load_yaml(workflow_path)
    if workflow is None:
        return (
            surface_id,
            valid_direct_globs,
            [],
            [],
            issues + [f"{surface_id}: workflow is not valid YAML: {workflow_path}"],
        )
    trigger = _workflow_trigger(workflow)
    pull_request_paths = _trigger_paths(trigger.get("pull_request"))
    if pull_request_paths is None and "pull_request" not in trigger:
        issues.append(f"{surface_id}: pull_request trigger is missing")
    workflow_text = workflow_path.read_text(encoding="utf-8")
    issues.extend(_check_surface_job_contract(surface, surface_id, workflow, workflow_text))
    dynamic_inputs = surface.get("dynamic_inputs", [])
    if not isinstance(dynamic_inputs, list):
        issues.append(f"{surface_id}: dynamic_inputs must be a list")
        dynamic_inputs = []
    return surface_id, valid_direct_globs, pull_request_paths, dynamic_inputs, issues


def _check_direct_globs(
    surface_id: str, direct_globs: list[str], pull_request_paths: list[str] | None
) -> list[str]:
    """Check that each declared direct input is selected by the workflow."""
    if pull_request_paths is None:
        return []
    return [
        f"{surface_id}: direct input is not covered by pull_request paths: {direct_glob}"
        for direct_glob in direct_globs
        if not any(pattern_covers(trigger_path, direct_glob) for trigger_path in pull_request_paths)
    ]


def _check_dynamic_inputs(
    surface_id: str,
    dynamic_inputs: list[Any],
    pull_request_paths: list[str] | None,
    repo_root: Path,
) -> list[str]:
    """Check dynamic registry sources and every path they currently enumerate."""
    issues: list[str] = []
    for dynamic in dynamic_inputs:
        if not isinstance(dynamic, dict) or not isinstance(dynamic.get("source"), str):
            issues.append(f"{surface_id}: malformed dynamic input declaration")
            continue
        source = dynamic["source"]
        fields = dynamic.get("fields")
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            issues.append(f"{surface_id}: dynamic fields must be a list of strings: {source}")
            continue
        if pull_request_paths is not None and not any(
            pattern_covers(trigger_path, source) for trigger_path in pull_request_paths
        ):
            issues.append(f"{surface_id}: dynamic source is not triggered: {source}")
        dynamic_paths, dynamic_issues = _dynamic_registry_paths(repo_root, source, fields)
        issues.extend(f"{surface_id}: {issue}" for issue in dynamic_issues)
        if pull_request_paths is not None:
            issues.extend(
                f"{surface_id}: dynamic path is not triggered: {dynamic_path}"
                for dynamic_path in dynamic_paths
                if not any(
                    pattern_covers(trigger_path, dynamic_path)
                    for trigger_path in pull_request_paths
                )
            )
    return issues


def _check_surface(surface: Any, repo_root: Path) -> tuple[str | None, list[str], list[str]]:
    """Validate one ownership surface and return its id, issues, and owned patterns."""
    if not isinstance(surface, dict):
        return None, ["ownership surface must be a mapping"], []
    surface_id, direct_globs, pull_paths, dynamic_inputs, issues = _surface_path_config(
        surface, repo_root
    )
    issues.extend(_check_direct_globs(surface_id, direct_globs, pull_paths))
    issues.extend(_check_dynamic_inputs(surface_id, dynamic_inputs, pull_paths, repo_root))
    return surface_id, issues, direct_globs


def _load_ownership(path: Path) -> tuple[list[Any], list[str]]:
    """Load the versioned ownership table and validate its top-level shape."""
    issues: list[str] = []
    try:
        ownership = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"cannot load ownership registry: {exc}"]
    if not isinstance(ownership, dict):
        return [], ["ownership registry must be a mapping"]
    if ownership.get("schema_version") != "robot_sf.release_legal_inputs.v1":
        issues.append("ownership registry has an unsupported schema_version")
    surfaces = ownership.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        return [], issues + ["ownership registry must contain a non-empty surfaces list"]
    return surfaces, issues


def validate_workflow_coverage(
    *,
    repo_root: Path,
    ownership_path: Path = DEFAULT_OWNERSHIP_PATH,
) -> dict[str, Any]:
    """Validate the versioned ownership table against workflows and registries."""
    repo_root = repo_root.resolve()
    surfaces, issues = _load_ownership(ownership_path)
    surfaces_checked: list[str] = []
    owned_patterns: list[str] = []
    for surface in surfaces:
        surface_id, surface_issues, surface_patterns = _check_surface(surface, repo_root)
        if surface_id is not None:
            surfaces_checked.append(surface_id)
        issues.extend(surface_issues)
        owned_patterns.extend(surface_patterns)
    issues.extend(
        f"required legal input is unowned: {required}"
        for required in REQUIRED_INPUT_PATTERNS
        if not any(pattern_covers(owned, required) for owned in owned_patterns)
    )
    base_ref = os.environ.get("BASE_REF")
    return {
        "schema_version": "robot-sf-release-legal-input-coverage.v1",
        "status": "blocked" if issues else "passed",
        "issues": issues,
        "surfaces_checked": surfaces_checked,
        "exact_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "base": _git_value(repo_root, "rev-parse", base_ref) if base_ref else None,
        "read_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the coverage check and print its machine-readable verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", type=Path, default=DEFAULT_OWNERSHIP_PATH)
    args = parser.parse_args(argv)
    report = validate_workflow_coverage(repo_root=Path.cwd(), ownership_path=args.ownership)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
