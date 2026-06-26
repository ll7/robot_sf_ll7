#!/usr/bin/env python3
# ruff: noqa: C901, PLR0912
"""Validate repo-local skill metadata, registry links, and generated index drift.

Subcommands:
  (no subcommand)  Run full registry validation (existing behavior).
  --preflight SKILL  Check runtime requirements declared by SKILL are available.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_skills_readme import render_readme

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
README = SKILLS_ROOT / "README.md"
REGISTRY = SKILLS_ROOT / "skills.yaml"
SCHEMAS_ROOT = SKILLS_ROOT / "schemas"
SKILLS_SCHEMA = SCHEMAS_ROOT / "skills.schema.yaml"
TESTS_ROOT = SKILLS_ROOT / "tests"
ALLOWED_NON_SKILL_DIRS = {"groups", "schemas", "tests"}
REQUIRED_FRONTMATTER = {
    "name",
    "description",
    "category",
    "kind",
    "phase",
    "requires_write",
    "requires_slurm",
    "requires_benchmark_artifacts",
    "delegates_to",
    "output_schema",
}
REQUIRED_REGISTRY_KEYS = REQUIRED_FRONTMATTER - {"name"}
REQUIRED_SECTIONS = ("## When to use", "## Guardrails", "## Output")
GENERIC_REFERENCE_PREFIXES = ("docs/config", "docs/provenance", "tests/checks")
PATH_PATTERN = re.compile(
    r"`[^`]*?((?:AGENTS\.md|code_review\.md|"
    r"(?:\.agent|\.specify|\.agents|\.codex|\.opencode|docs|scripts|configs|tests|"
    r"SLURM|\.github)/[^\s`]+))[^`]*?`"
)
CRITICAL_DUPLICATE_PATTERNS = (
    "Within the same status, prefer higher Project",
    "fallback/degraded should be treated as a caveat",
)
BACKTICK_TOKEN_PATTERN = re.compile(r"`([a-z][a-z0-9-]+)`")
# A trailing ``.ext`` (1-8 alphanumerics) marks a reference as a concrete file path.
EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,8}$")
ARTIFACT_FIRST_SKILLS = {"goal-autopilot", "goal-issue-implementation", "goal-pr-review"}
ARTIFACT_FIRST_REQUIRED_FILES = ("result.json", "RESULT.md", "diffstat.txt", "validation.json")
ARTIFACT_FIRST_REQUIRED_PHRASES = (
    "artifact-first",
    "route evidence",
    "raw logs",
    "targeted local",
)
WORKER_OUTPUT_REQUIRED_PHRASES = (
    "rg -l",
    "rg --files",
    "sed -n",
    "200 lines",
    "private artifacts",
    "rg -n .",
    "full file reads",
)
GOAL_PR_REVIEW_REQUIRED_PHRASES = (
    "snapshot_pr_queue.py",
    "watch_pr_ci_status.py",
    "status,conclusion,jobs",
    "bounded excerpts",
    "full logs",
    "private artifacts",
)
GOAL_AUTOPILOT_LEDGER_REQUIRED_PHRASES = (
    "loaded context",
    "skill/doc summaries",
    "snapshot paths",
    "freshness keys",
    "expected pr head sha",
    "worker artifact paths",
    "stale-state triggers",
    "ledger snapshot paths",
    "repeating broad state polling",
    "fresh live checks",
    "issue claim",
    "push",
    "pr publication",
    "label/project mutation",
    "merge-ready",
    "compact_worktree_snapshot.py",
    "compact_ci_snapshot.py",
)
GOAL_AUTOPILOT_SPARK_REQUIRED_PHRASES = (
    "spark sidecar routing",
    "gpt-5.3-codex-spark",
    "tiny lookup",
    "read-only review",
    "docs cross-check",
    "issue/file surface mapping",
    "files inspected",
    "exact evidence",
    "uncertainty",
    "recommended next prompt",
    "github mutation",
    "shell-executable fallback",
)


def _read_yaml(path: Path) -> Any:
    """Read a YAML file and fail with a path-qualified message."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            raise AssertionError(f"{path.relative_to(REPO_ROOT)}: YAML file is empty")
        if not isinstance(data, dict):
            raise AssertionError(f"{path.relative_to(REPO_ROOT)}: YAML top level must be a mapping")
        return data
    except yaml.YAMLError as exc:
        raise AssertionError(f"{path.relative_to(REPO_ROOT)}: invalid YAML: {exc}") from exc


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    """Return parsed frontmatter and body text."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)}: missing opening YAML frontmatter marker"
        )
    try:
        closing_idx = lines[1:].index("---") + 1
    except ValueError as exc:
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)}: missing closing YAML frontmatter marker"
        ) from exc
    raw = "\n".join(lines[1:closing_idx])
    try:
        metadata = yaml.safe_load(raw)
        if not isinstance(metadata, dict):
            raise AssertionError(
                f"{path.relative_to(REPO_ROOT)}: frontmatter must be a YAML mapping"
            )
    except yaml.YAMLError as exc:
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)}: invalid frontmatter YAML: {exc}"
        ) from exc
    return metadata, "\n".join(lines[closing_idx + 1 :])


def _reference_path(match: str) -> str:
    """Extract the filesystem path token from a matched backtick snippet."""
    return match.split(maxsplit=1)[0].split("#", maxsplit=1)[0].rstrip(".,:;)")


def _is_generic_reference(reference: str) -> bool:
    """Return true for prose placeholders that name a path category, not a concrete path."""
    return reference in GENERIC_REFERENCE_PREFIXES


def _looks_like_path(reference: str) -> bool:
    """Return true when REFERENCE has the shape of a concrete repo path.

    A reference looks path-like when it carries a file extension (``foo/bar.md``)
    or nests at least two segments below its top-level prefix (``foo/bar/baz``).
    Single-segment, extension-less tokens such as ``SLURM/data-gated`` are prose
    placeholders, not file references, so they are excluded to avoid false
    positives while genuine broken paths (which keep an extension or extra depth)
    are still validated.
    """
    if EXTENSION_PATTERN.search(reference):
        return True
    return reference.count("/") >= 2


def _find_broken_paths(path: Path, text: str) -> list[str]:
    """Return repo-relative path references from PATH_PATTERN that do not exist."""
    broken_paths: list[str] = []
    for match in PATH_PATTERN.findall(text):
        reference = _reference_path(match)
        if _is_generic_reference(reference):
            continue
        if (REPO_ROOT / reference).exists():
            continue
        if not _looks_like_path(reference):
            # Non-resolving, non-path-shaped tokens are prose placeholders.
            continue
        broken_paths.append(f"{path.relative_to(REPO_ROOT)} -> {reference}")
    return broken_paths


def _validate_registry_shape(registry: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate registry-level metadata and links."""
    errors: list[str] = []
    if registry.get("version") != 1:
        errors.append(".agents/skills/skills.yaml: expected version: 1")
    skills = registry.get("skills")
    if not isinstance(skills, dict) or not skills:
        errors.append(".agents/skills/skills.yaml: missing non-empty skills map")
        return errors
    allowed_categories = set(schema["allowed_categories"])
    allowed_kinds = set(schema["allowed_kinds"])
    allowed_phases = set(schema["allowed_phases"])
    allowed_write_scopes = set(schema["allowed_write_scopes"])
    names = set(skills)
    aliases: dict[str, str] = {}
    for name, metadata in skills.items():
        if not isinstance(metadata, dict):
            errors.append(f"{name}: metadata must be a dictionary")
            continue
        missing = REQUIRED_REGISTRY_KEYS - set(metadata)
        if missing:
            errors.append(f"{name}: registry missing keys {sorted(missing)}")
        if metadata.get("category") not in allowed_categories:
            errors.append(f"{name}: invalid category {metadata.get('category')!r}")
        if metadata.get("kind") not in allowed_kinds:
            errors.append(f"{name}: invalid kind {metadata.get('kind')!r}")
        if metadata.get("phase") not in allowed_phases:
            errors.append(f"{name}: invalid phase {metadata.get('phase')!r}")
        writes = metadata.get("writes", {})
        unknown_write_scopes = set(writes) - allowed_write_scopes
        if unknown_write_scopes:
            errors.append(f"{name}: invalid write scopes {sorted(unknown_write_scopes)}")
        non_boolean_writes = sorted(
            scope for scope, value in writes.items() if not isinstance(value, bool)
        )
        if non_boolean_writes:
            errors.append(f"{name}: write scopes must be booleans: {non_boolean_writes}")
        for delegate in metadata.get("delegates_to", []):
            if delegate not in names:
                errors.append(f"{name}: delegates_to missing skill {delegate!r}")
        output_schema_name = metadata.get("output_schema")
        if output_schema_name and not (SCHEMAS_ROOT / f"{output_schema_name}.yaml").exists():
            errors.append(f"{name}: missing output schema {output_schema_name!r}")
        if metadata.get("requires_write"):
            writes = metadata.get("writes", {})
            if not any(bool(writes.get(key)) for key in writes):
                errors.append(f"{name}: mutating skill requires a writes scope")
        elif any(bool(value) for value in metadata.get("writes", {}).values()):
            errors.append(f"{name}: declares write scopes but requires_write is false")
        for alias in metadata.get("aliases", []):
            if alias in names:
                errors.append(f"{name}: alias {alias!r} collides with a skill name")
            if alias in aliases:
                errors.append(f"{name}: alias {alias!r} duplicates alias from {aliases[alias]!r}")
            aliases[alias] = name
    return errors


def _validate_skill_file(
    path: Path,
    registry_metadata: dict[str, Any],
    readme_text: str,
) -> tuple[list[str], str]:
    """Validate one skill file and return errors plus its body for path checks."""
    errors: list[str] = []
    metadata, body = _frontmatter(path)
    rel = path.relative_to(REPO_ROOT)
    missing = REQUIRED_FRONTMATTER - set(metadata)
    if missing:
        errors.append(f"{rel}: missing frontmatter keys {sorted(missing)}")
        return errors, body
    name = metadata["name"]
    if path.parent.name != name:
        errors.append(f"{rel}: directory name {path.parent.name!r} != frontmatter name {name!r}")
    if name not in registry_metadata:
        errors.append(f"{rel}: skill missing from skills.yaml")
        return errors, body
    for key in REQUIRED_REGISTRY_KEYS:
        if metadata.get(key) != registry_metadata[name].get(key):
            errors.append(f"{rel}: frontmatter key {key!r} differs from skills.yaml")
    aliases = metadata.get("aliases", [])
    if aliases != registry_metadata[name].get("aliases", []):
        errors.append(f"{rel}: aliases differ from skills.yaml")
    if f"`{name}`" not in readme_text:
        errors.append(f"{rel}: skill missing from README index")
    for section in REQUIRED_SECTIONS:
        if section not in body:
            errors.append(f"{rel}: missing required section {section!r}")
    errors.extend(_validate_artifact_first_contract(path, metadata, body))
    if registry_metadata[name].get("requires_benchmark_artifacts"):
        policy_refs = (
            "fail-closed" in body.lower()
            or "fallback" in body.lower()
            or "docs/context/issue_691_benchmark_fallback_policy.md" in body
        )
        if not policy_refs:
            errors.append(
                f"{rel}: benchmark-related skill lacks fail-closed/fallback policy reference"
            )
    return errors, body


def _validate_non_skill_dirs() -> list[str]:
    """Ensure direct children are either skill directories or explicit support directories."""
    errors: list[str] = []
    for child in sorted(path for path in SKILLS_ROOT.iterdir() if path.is_dir()):
        if child.name in ALLOWED_NON_SKILL_DIRS:
            continue
        if not (child / "SKILL.md").exists():
            errors.append(f"{child.relative_to(REPO_ROOT)}: non-skill directory is not whitelisted")
    return errors


def _validate_generated_readme(registry: dict[str, Any], readme_text: str) -> list[str]:
    """Return drift errors when README is not generated from the registry."""
    expected = render_readme(registry)
    if readme_text != expected:
        return [
            ".agents/skills/README.md is stale; run "
            "`uv run python scripts/dev/generate_skills_readme.py`"
        ]
    return []


def _validate_artifact_first_contract(path: Path, metadata: dict[str, Any], text: str) -> list[str]:
    """Validate artifact-first delegated route contract text for selected skills."""
    if metadata.get("name") not in ARTIFACT_FIRST_SKILLS:
        return []

    rel = path.relative_to(REPO_ROOT)
    errors: list[str] = []

    for filename in ARTIFACT_FIRST_REQUIRED_FILES:
        if filename not in text:
            errors.append(f"{rel}: missing artifact-first requirement {filename!r}")

    lower = text.lower()
    for phrase in ARTIFACT_FIRST_REQUIRED_PHRASES:
        if phrase not in lower:
            errors.append(f"{rel}: missing artifact-first phrase requirement {phrase!r}")

    for phrase in WORKER_OUTPUT_REQUIRED_PHRASES:
        if phrase not in lower:
            errors.append(f"{rel}: missing worker-output limit requirement {phrase!r}")

    if metadata.get("name") == "goal-autopilot":
        for phrase in GOAL_AUTOPILOT_LEDGER_REQUIRED_PHRASES:
            if phrase not in lower:
                errors.append(f"{rel}: missing active-ledger requirement {phrase!r}")
        for phrase in GOAL_AUTOPILOT_SPARK_REQUIRED_PHRASES:
            if phrase not in lower:
                errors.append(f"{rel}: missing Spark sidecar routing requirement {phrase!r}")

    if metadata.get("name") == "goal-pr-review":
        for phrase in GOAL_PR_REVIEW_REQUIRED_PHRASES:
            if phrase not in lower:
                errors.append(f"{rel}: missing PR-review compact CI requirement {phrase!r}")

    return errors


def _validate_backticked_skill_tokens(
    path: Path, text: str, skill_names: set[str], aliases: set[str]
) -> list[str]:
    """Validate skill-looking backtick tokens in generated routing docs."""
    errors: list[str] = []
    allowed_tokens = skill_names | aliases | {"none"}
    for token in BACKTICK_TOKEN_PATTERN.findall(text):
        if token not in allowed_tokens:
            errors.append(
                f"{path.relative_to(REPO_ROOT)}: unknown backticked skill token `{token}`"
            )
    return errors


def _validate_routing_goldens(registry: dict[str, Any]) -> list[str]:
    """Execute simple routing golden assertions."""
    errors: list[str] = []
    skill_names = set(registry["skills"])
    aliases = {
        alias for metadata in registry["skills"].values() for alias in metadata.get("aliases", [])
    }
    valid_names = skill_names | aliases

    routing_cases = _read_yaml(TESTS_ROOT / "routing_cases.yaml")
    for index, case in enumerate(routing_cases.get("cases", []), start=1):
        for field in ("primary",):
            if case.get(field) not in valid_names:
                errors.append(
                    f"routing_cases.yaml case {index}: unknown {field} {case.get(field)!r}"
                )
        for field in ("secondary", "negative"):
            for skill in case.get(field, []):
                if skill not in valid_names:
                    errors.append(
                        f"routing_cases.yaml case {index}: unknown {field} skill {skill!r}"
                    )

    stacks = _read_yaml(TESTS_ROOT / "expected_skill_stacks.yaml").get("expected_stacks", {})
    for stack_name, stack in stacks.items():
        if not stack:
            errors.append(f"expected_skill_stacks.yaml {stack_name}: empty stack")
        for skill in stack:
            if skill not in valid_names:
                errors.append(f"expected_skill_stacks.yaml {stack_name}: unknown skill {skill!r}")
    return errors


def _validate_duplicate_routing(text_by_path: dict[Path, str]) -> list[str]:
    """Catch repeated routing sentences in critical docs."""
    errors: list[str] = []
    for path, text in text_by_path.items():
        for pattern in CRITICAL_DUPLICATE_PATTERNS:
            if text.count(pattern) > 1:
                errors.append(f"{path.relative_to(REPO_ROOT)}: duplicate routing line {pattern!r}")
    return errors


# -- preflight helpers ----------------------------------------------------------

_REQUIREMENT_CHECKS: dict[str, tuple[str, str, str]] = {
    "git": ("git", "--version", "Install Git and ensure `git` is on PATH."),
    "uv": ("uv", "--version", "Install uv and ensure `uv` is on PATH."),
    "gh": ("gh", "--version", "Install GitHub CLI, then run `gh auth login`."),
    "slurm": (
        "sbatch",
        "--version",
        "Run on a SLURM login node or use a non-SLURM skill for this task.",
    ),
    "project5": (
        "gh",
        "--version",
        "Install/authenticate GitHub CLI; Project #5 writes also need project permissions.",
    ),
}

PUBLICATION_SCOUT_LINTER_REQUIREMENT = "publication-scout-linter"
_PROJECT5_EXTRA = "jq"


def _check_command(cmd: str, arg: str) -> tuple[bool, str]:
    """Return (available, version_or_error) for a command."""
    resolved = shutil.which(cmd)
    if resolved is None:
        return False, f"{cmd}: not found on PATH"
    try:
        result = subprocess.run(
            [resolved, arg],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        lines = (result.stdout or result.stderr or "").splitlines()
        version = lines[0].strip() if lines else "unknown version"
        return True, version
    except FileNotFoundError:
        return False, f"{cmd}: not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f"{cmd}: timed out after 15s"


def _run_preflight_check(
    req: str,
    results: dict[str, Any] | None,
    json_output: bool,
) -> tuple[str, str]:
    """Run a single requirement check and return (status, detail).

    Returns one of (ok, version_string), (missing, error_string),
    or (unrecognized, message).
    """
    if req == PUBLICATION_SCOUT_LINTER_REQUIREMENT:
        script_path = REPO_ROOT / "scripts" / "dev" / "publication_scout_linter.py"
        if script_path.is_file():
            detail = f"{script_path.relative_to(REPO_ROOT)} present"
            if json_output and results is not None:
                results["checks"][req] = {"status": "ok", "detail": detail, "remedy": None}
            return "ok", detail

        detail = f"{script_path.relative_to(REPO_ROOT)} is required for publication-scout conformance checks"
        if json_output and results is not None:
            results["checks"][req] = {
                "status": "missing",
                "detail": detail,
                "remedy": "Restore scripts/dev/publication_scout_linter.py or remove the requirement.",
            }
        return "missing", detail

    check = _REQUIREMENT_CHECKS.get(req)
    if check is None:
        detail = (
            "no check registered for this requirement; add it to "
            "scripts/dev/check_skills.py or remove the unsupported `requires` value"
        )
        if json_output and results is not None:
            results["checks"][req] = {
                "status": "unrecognized",
                "detail": detail,
                "remedy": detail,
            }
        return "unrecognized", detail

    cmd, arg, remedy = check
    ok, detail = _check_command(cmd, arg)

    if ok:
        status = "ok"
        jq_ok: bool | None = None
        if req == "project5":
            jq_ok, _ = _check_command(_PROJECT5_EXTRA, "--version")
            if not jq_ok:
                status = "warning"
                detail = f"{cmd} found but {_PROJECT5_EXTRA} not on PATH"

        if json_output and results is not None:
            results["checks"][req] = {"status": status, "detail": detail, "remedy": None}
            if jq_ok is not None:
                results["checks"][req]["jq_available"] = jq_ok

        return status, detail

    if json_output and results is not None:
        results["checks"][req] = {"status": "missing", "detail": detail, "remedy": remedy}
    return "missing", f"{detail}; remedy: {remedy}"


def _preflight(skill_name: str, json_output: bool = False) -> int:
    """Check runtime requirements declared by SKILL_NAME are available on PATH."""
    registry = _read_yaml(REGISTRY)
    skills: dict[str, Any] = registry.get("skills", {})

    # Resolve via alias lookup
    canonical = skill_name
    if canonical not in skills:
        for name, meta in skills.items():
            if skill_name in meta.get("aliases", []):
                canonical = name
                break
        else:
            msg = f"Skill {skill_name!r} not found in skills.yaml (and no alias matches)"
            if json_output:
                print(json.dumps({"status": "error", "skill": skill_name, "error": msg}))
            else:
                print(f"ERROR: {msg}")
            return 1

    meta = skills[canonical]
    requires: list[str] = meta.get("requires", [])

    results: dict[str, Any] | None = (
        {"status": "ok", "skill": canonical, "requires": requires, "checks": {}}
        if json_output
        else None
    )

    if not requires:
        if json_output:
            assert results is not None
            results["summary"] = {"available": 0, "missing": 0, "unrecognized": 0}
            print(json.dumps(results))
        else:
            _print_preflight_header(canonical, requires)
            print("  (no requirements declared)")
            print()
            print("Result: PASS (no requirements to check)")
        return 0

    if not json_output:
        _print_preflight_header(canonical, requires)

    available = 0
    missing = 0
    unrecognized = 0

    for req in requires:
        status, detail = _run_preflight_check(req, results, json_output)

        if status == "ok":
            available += 1
            if not json_output:
                print(f"  ok            {req:12}  {detail}")
        elif status == "warning":
            # project5 edge case: found gh but missing jq
            available += 1
            if not json_output:
                print(f"  warning       {req:12}  {detail}")
        elif status == "missing":
            missing += 1
            if not json_output:
                print(f"  missing       {req:12}  {detail}")
        else:
            unrecognized += 1
            if not json_output:
                print(f"  UNRECOGNIZED  {req:12}  {detail}")

    if json_output:
        assert results is not None
        if missing > 0 or unrecognized > 0:
            results["status"] = "fail"
        results["summary"] = {
            "available": available,
            "missing": missing,
            "unrecognized": unrecognized,
        }
        print(json.dumps(results))
    else:
        print()
        _print_preflight_summary(available, missing, unrecognized)

    return 1 if missing > 0 or unrecognized > 0 else 0


def _print_preflight_header(skill: str, requires: list[str]) -> None:
    """Print a human-readable preflight header."""
    print(f"Preflight check for skill: {skill}")
    print(f"Declared requirements: {', '.join(requires)}")
    print()


def _print_preflight_summary(available: int, missing: int, unrecognized: int) -> None:
    """Print a human-readable summary."""
    parts = [f"  available: {available}"]
    if missing:
        parts.append(f"missing: {missing}")
    if unrecognized:
        parts.append(f"unrecognized: {unrecognized}")
    if missing or unrecognized:
        print("Result: FAIL (" + ", ".join(parts) + ")")
    else:
        print("Result: PASS (" + ", ".join(parts) + ")")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        metavar="SKILL",
        help="run preflight requirement check for a named skill",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="emit machine-readable JSON output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Return non-zero when skill metadata, registry links, or docs are stale.

    Can also run ``--preflight SKILL`` to check runtime requirements.
    """
    args = _parse_args(argv)

    if args.preflight:
        return _preflight(args.preflight, json_output=args.json)

    registry = _read_yaml(REGISTRY)
    schema = _read_yaml(SKILLS_SCHEMA)
    readme_text = README.read_text(encoding="utf-8")
    skill_paths = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
    if not skill_paths:
        raise AssertionError(f"No SKILL.md files found under {SKILLS_ROOT}")

    errors: list[str] = []
    errors.extend(_validate_registry_shape(registry, schema))
    errors.extend(_validate_non_skill_dirs())
    errors.extend(_validate_generated_readme(registry, readme_text))
    errors.extend(_find_broken_paths(README, readme_text))

    registry_metadata = registry.get("skills", {})
    skill_names_on_disk: set[str] = set()
    text_by_path = {README: readme_text}
    for path in skill_paths:
        metadata, _ = _frontmatter(path)
        skill_names_on_disk.add(metadata.get("name", path.parent.name))
        skill_errors, _body = _validate_skill_file(path, registry_metadata, readme_text)
        text = path.read_text(encoding="utf-8")
        text_by_path[path] = text
        errors.extend(skill_errors)
        errors.extend(_find_broken_paths(path, text))

    missing_dirs = sorted(set(registry_metadata) - skill_names_on_disk)
    if missing_dirs:
        errors.append("Skills in skills.yaml without SKILL.md: " + ", ".join(missing_dirs))
    extra_dirs = sorted(skill_names_on_disk - set(registry_metadata))
    if extra_dirs:
        errors.append("Skills on disk missing from skills.yaml: " + ", ".join(extra_dirs))

    for required in [SCHEMAS_ROOT, TESTS_ROOT]:
        if not required.exists():
            errors.append(f"{required.relative_to(REPO_ROOT)}: required support directory missing")
    for required_file in [
        SKILLS_SCHEMA,
        TESTS_ROOT / "routing_cases.yaml",
        TESTS_ROOT / "expected_skill_stacks.yaml",
        TESTS_ROOT / "overlap_allowlist.yaml",
    ]:
        if not required_file.exists():
            errors.append(f"{required_file.relative_to(REPO_ROOT)}: required golden file missing")
        else:
            _read_yaml(required_file)

    aliases = {
        alias for metadata in registry_metadata.values() for alias in metadata.get("aliases", [])
    }
    errors.extend(
        _validate_backticked_skill_tokens(README, readme_text, set(registry_metadata), aliases)
    )
    errors.extend(_validate_routing_goldens(registry))
    errors.extend(_validate_duplicate_routing(text_by_path))
    if errors:
        raise AssertionError("; ".join(sorted(errors)))

    print(
        f"Validated {len(skill_paths)} skills, typed registry, generated README, and routing tests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
