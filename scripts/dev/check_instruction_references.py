#!/usr/bin/env python3
"""Validate repository-local references in the canonical agent instruction graph.

The task-to-guidance routing owner is ``docs/ai/agent_workflow_entrypoints.md``. This checker
resolves every required repository-local file reference made by a canonical instruction surface,
distinguishes optional/non-normative references, follows symlinks, and honors an explicit
generated-target allowlist. It fails closed on a missing target that an instruction presents as
required.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical boot and scoped instruction surfaces. Adding a surface here requires an owner and
# trigger in the routing owner; this tuple is the checker's source of the instruction graph.
CANONICAL_INSTRUCTION_GRAPH = (
    "AGENTS.md",
    ".agents/README.md",
    ".agents/PLANS.md",
    "docs/maintainer_values.md",
    "docs/ai/agent_workflow_entrypoints.md",
    "SLURM/AGENTS.md",
    ".claude/CLAUDE.md",
    ".github/copilot-instructions.md",
    ".specify/memory/constitution.md",
)

ROUTING_OWNER = "docs/ai/agent_workflow_entrypoints.md"
ROUTE_TABLE_MARKER = "| Route | Purpose | Required context / evidence |"
PRECEDENCE_OWNER = "AGENTS.md"
PRECEDENCE_START_MARKER = "<!-- instruction-precedence:start -->"
PRECEDENCE_END_MARKER = "<!-- instruction-precedence:end -->"

TASK_SCOPE_MANIFEST = ".agents/task_scope_manifest.yaml"
PROFILE_IDS = ("observe", "local", "coordinated", "evidence_critical")
PROFILE_BOOL_FIELDS = (
    "plan_required",
    "environment_required",
    "worktree_required",
    "pr_required",
    "evidence_gates_required",
)
ROUTE_IDS = (
    "read-only-observation",
    "documentation-only-edit",
    "implementation-runtime-change",
    "scientific-benchmark-interpretation",
    "environment-worktree-repair",
)

REPO_ROOT_SEGMENTS = frozenset(
    {
        ".agents",
        ".claude",
        ".codex",
        ".github",
        ".opencode",
        ".specify",
        "SLURM",
        "configs",
        "docs",
        "examples",
        "fast-pysf",
        "maps",
        "memory",
        "model",
        "robot_sf",
        "scripts",
        "tests",
    }
)

PATH_EXTENSIONS = (".md", ".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg", ".ini")

OPTIONAL_MARKERS = (
    "optional",
    "for example",
    "e.g.",
    "illustrative",
    "if present",
    "when present",
    "if it exists",
    "when it exists",
    "generated",
    "instead of",
)

SKIP_FRAGMENTS = ("%", "<", ">", "*", "{", "}", "$", "://", "output/", ".venv/", "http")

MARKDOWN_LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CODE_SPAN_RE = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class Reference:
    """One repository-local reference extracted from an instruction surface."""

    source: str
    line_number: int
    path: str
    required: bool


@dataclass
class CheckResult:
    """Outcome of scanning a set of instruction surfaces."""

    errors: list[str] = field(default_factory=list)
    checked: int = 0
    optional_skipped: int = 0
    files: tuple[str, ...] = ()


def _clean_token(token: str) -> str:
    """Strip wrapping punctuation and anchors from a candidate path token."""
    cleaned = token.strip().strip("`\"'()[]{}")
    cleaned = cleaned.rstrip(".,;:")
    cleaned = cleaned.split("#", maxsplit=1)[0]
    cleaned = cleaned.split("?", maxsplit=1)[0]
    return cleaned.rstrip("/")


def _is_candidate_path(token: str) -> bool:
    """Return whether a raw token plausibly names a repository-local file or directory."""
    if not token or token.startswith("-") or token.startswith("~"):
        return False
    if any(fragment in token for fragment in SKIP_FRAGMENTS):
        return False
    if " " in token or "\t" in token:
        return False
    has_extension = token.endswith(PATH_EXTENSIONS)
    if token.startswith("./") or token.startswith("../"):
        return has_extension or "/" in token.lstrip("./")
    head = token.split("/", maxsplit=1)[0]
    if head in REPO_ROOT_SEGMENTS:
        return "/" in token or has_extension
    return token == "AGENTS.md" or (has_extension and "/" in token)


def _line_references(line: str) -> list[str]:
    """Return candidate path tokens from markdown links and inline code spans on one line."""
    candidates: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(line):
        candidates.append(_clean_token(match.group(1)))
    for match in CODE_SPAN_RE.finditer(line):
        span = match.group(1)
        if " " in span or "\t" in span:
            candidates.extend(
                _clean_token(part) for part in span.split() if _is_candidate_path(part)
            )
        else:
            candidates.append(_clean_token(span))
    return [candidate for candidate in candidates if _is_candidate_path(candidate)]


def extract_references(text: str, source: str) -> list[Reference]:
    """Extract repository-local references and their required/optional classification."""
    references: list[Reference] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        required = not any(marker in lowered for marker in OPTIONAL_MARKERS)
        for candidate in _line_references(line):
            references.append(
                Reference(
                    source=source,
                    line_number=line_number,
                    path=candidate,
                    required=required,
                )
            )
    return references


def _candidate_paths(reference: Reference, root: Path) -> list[Path]:
    """Return root-relative and source-relative resolutions for one reference."""
    source_relative = (root / reference.source).parent / reference.path
    if reference.path.startswith(("./", "../")):
        return [source_relative]
    return [root / reference.path, source_relative]


def _resolve_error(reference: Reference, root: Path) -> str | None:
    """Return an error message when a reference cannot be resolved under ``root``."""
    location = f"{reference.source}:{reference.line_number}: `{reference.path}`"
    for candidate in _candidate_paths(reference, root):
        if candidate.is_symlink():
            link_target = candidate.readlink()
            resolved = (candidate.parent / link_target).resolve()
            if not resolved.exists():
                return f"{location} is a symlink to a missing target: {resolved}"
            return None
        if candidate.exists():
            return None
    if reference.required:
        return f"{location} is presented as required but does not exist"
    return None


def check_instruction_graph(
    root: Path = REPO_ROOT,
    graph: tuple[str, ...] = CANONICAL_INSTRUCTION_GRAPH,
    generated_targets: frozenset[str] = frozenset(),
) -> CheckResult:
    """Resolve every required repository-local reference in the instruction graph."""
    result = CheckResult(files=graph)
    for source in graph:
        source_path = root / source
        if not source_path.is_file():
            result.errors.append(f"{source} is listed in the instruction graph but does not exist")
            continue
        text = source_path.read_text(encoding="utf-8")
        for reference in extract_references(text, source):
            if reference.path in generated_targets:
                continue
            result.checked += 1
            error = _resolve_error(reference, root)
            if error is not None:
                result.errors.append(error)
            elif not reference.required:
                result.optional_skipped += 1
    return result


def check_routing_ownership(root: Path = REPO_ROOT) -> list[str]:
    """Require exactly one canonical routing owner containing the route table."""
    errors: list[str] = []
    owner_path = root / ROUTING_OWNER
    if not owner_path.is_file():
        return [f"routing owner {ROUTING_OWNER} does not exist"]
    if ROUTE_TABLE_MARKER not in owner_path.read_text(encoding="utf-8"):
        errors.append(f"{ROUTING_OWNER} does not contain the canonical route table")
    for source in CANONICAL_INSTRUCTION_GRAPH:
        if source == ROUTING_OWNER:
            continue
        source_path = root / source
        if not source_path.is_file():
            continue
        if ROUTE_TABLE_MARKER in source_path.read_text(encoding="utf-8"):
            errors.append(f"{source} restates the canonical route table owned by {ROUTING_OWNER}")
    return errors


def check_precedence_contract(
    root: Path = REPO_ROOT,
    graph: tuple[str, ...] = CANONICAL_INSTRUCTION_GRAPH,
) -> list[str]:
    """Require exactly one marked normative precedence block, owned by ``AGENTS.md``."""
    errors: list[str] = []
    owners: list[str] = []
    for source in graph:
        source_path = root / source
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8")
        starts = text.count(PRECEDENCE_START_MARKER)
        ends = text.count(PRECEDENCE_END_MARKER)
        if starts == 0 and ends == 0:
            continue
        if (
            starts != 1
            or ends != 1
            or text.index(PRECEDENCE_START_MARKER) > text.index(PRECEDENCE_END_MARKER)
        ):
            errors.append(f"{source}: precedence markers must appear exactly once and in order")
        owners.append(source)
    if owners != [PRECEDENCE_OWNER]:
        found = ", ".join(owners) if owners else "no surface"
        errors.append(f"precedence contract must be owned by {PRECEDENCE_OWNER}; found in {found}")
    return errors


def load_task_scope_manifest(root: Path = REPO_ROOT) -> object:
    """Load the machine-readable execution-profile and route mapping."""
    return yaml.safe_load((root / TASK_SCOPE_MANIFEST).read_text(encoding="utf-8"))


def _check_profile(root: Path, profile_id: str, profile: object, errors: list[str]) -> None:
    """Validate one execution-profile entry in the task-scope manifest."""
    prefix = f"{TASK_SCOPE_MANIFEST}: profiles.{profile_id}"
    if not isinstance(profile, dict):
        errors.append(f"{prefix} must be a mapping")
        return
    description = profile.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{prefix}.description must be a non-empty string")
    for field_name in PROFILE_BOOL_FIELDS:
        if not isinstance(profile.get(field_name), bool):
            errors.append(f"{prefix}.{field_name} must be a boolean")
    context = profile.get("required_context")
    if not isinstance(context, list) or not context:
        errors.append(f"{prefix}.required_context must be a non-empty list")
    else:
        for entry in context:
            if not isinstance(entry, str) or not entry.strip():
                errors.append(f"{prefix}.required_context entries must be non-empty strings")
            elif not (root / entry).exists():
                errors.append(f"{prefix}.required_context path does not exist: {entry}")
    ceremony = profile.get("forbidden_ceremony", [])
    if not isinstance(ceremony, list):
        errors.append(f"{prefix}.forbidden_ceremony must be a list when present")


def _check_manifest_header(manifest: dict, errors: list[str]) -> None:
    """Validate the fixed manifest identity fields."""
    if manifest.get("version") != 1:
        errors.append(f"{TASK_SCOPE_MANIFEST}: version must be 1")
    if manifest.get("routing_owner") != ROUTING_OWNER:
        errors.append(f"{TASK_SCOPE_MANIFEST}: routing_owner must be {ROUTING_OWNER}")
    if manifest.get("manifest_owner") != PRECEDENCE_OWNER:
        errors.append(f"{TASK_SCOPE_MANIFEST}: manifest_owner must be {PRECEDENCE_OWNER}")


def _check_profiles(root: Path, profiles: object, errors: list[str]) -> None:
    """Validate the execution-profile section of the manifest."""
    if not isinstance(profiles, dict):
        errors.append(f"{TASK_SCOPE_MANIFEST}: profiles must be a mapping")
        return
    if set(profiles) != set(PROFILE_IDS):
        errors.append(
            f"{TASK_SCOPE_MANIFEST}: profiles must be exactly {list(PROFILE_IDS)}; "
            f"found {sorted(profiles)}"
        )
    for profile_id, profile in profiles.items():
        _check_profile(root, str(profile_id), profile, errors)


def _check_routes(routes: object, errors: list[str]) -> None:
    """Validate the route-to-profile section of the manifest."""
    if not isinstance(routes, dict):
        errors.append(f"{TASK_SCOPE_MANIFEST}: routes must be a mapping")
        return
    if set(routes) != set(ROUTE_IDS):
        errors.append(
            f"{TASK_SCOPE_MANIFEST}: routes must be exactly {list(ROUTE_IDS)}; "
            f"found {sorted(routes)}"
        )
    for route_id, route in routes.items():
        default = route.get("default_profile") if isinstance(route, dict) else None
        if default not in PROFILE_IDS:
            errors.append(
                f"{TASK_SCOPE_MANIFEST}: routes.{route_id}.default_profile must be one of "
                f"{list(PROFILE_IDS)}"
            )


def _check_escalation(escalation: object, errors: list[str]) -> None:
    """Validate the profile escalation rule."""
    if not isinstance(escalation, dict):
        errors.append(f"{TASK_SCOPE_MANIFEST}: escalation must be a mapping")
        return
    if escalation.get("allowed") is not True:
        errors.append(f"{TASK_SCOPE_MANIFEST}: escalation.allowed must be true")
    rule = escalation.get("rule")
    if not isinstance(rule, str) or not rule.strip():
        errors.append(f"{TASK_SCOPE_MANIFEST}: escalation.rule must be a non-empty string")


def check_task_scope_manifest(
    root: Path = REPO_ROOT,
    manifest: object | None = None,
) -> list[str]:
    """Validate the machine-readable task/profile mapping schema and references."""
    errors: list[str] = []
    if manifest is None:
        path = root / TASK_SCOPE_MANIFEST
        if not path.is_file():
            return [f"{TASK_SCOPE_MANIFEST} is missing"]
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            return [f"{TASK_SCOPE_MANIFEST} is not valid YAML: {exc}"]
    if not isinstance(manifest, dict):
        return [f"{TASK_SCOPE_MANIFEST} must be a mapping at the document root"]
    _check_manifest_header(manifest, errors)
    _check_profiles(root, manifest.get("profiles"), errors)
    _check_routes(manifest.get("routes"), errors)
    _check_escalation(manifest.get("escalation"), errors)
    return errors


def run_checks(root: Path = REPO_ROOT) -> dict[str, object]:
    """Run all instruction-reference checks and return a JSON-ready report."""
    graph_result = check_instruction_graph(root)
    ownership_errors = check_routing_ownership(root)
    precedence_errors = check_precedence_contract(root)
    task_scope_errors = check_task_scope_manifest(root)
    return {
        "schema": "instruction_references.v1",
        "root": str(root),
        "routing_owner": ROUTING_OWNER,
        "precedence_owner": PRECEDENCE_OWNER,
        "files": list(graph_result.files),
        "references_checked": graph_result.checked,
        "optional_references": graph_result.optional_skipped,
        "precedence_errors": precedence_errors,
        "task_scope_errors": task_scope_errors,
        "errors": (
            graph_result.errors + ownership_errors + precedence_errors + task_scope_errors
        ),
    }


def main() -> int:
    """Return non-zero when a canonical instruction reference does not resolve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()

    report = run_checks(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for error in report["errors"]:
            print(error)
        print(
            f"instruction references checked: {report['references_checked']} "
            f"(+{report['optional_references']} optional references resolved)"
        )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
