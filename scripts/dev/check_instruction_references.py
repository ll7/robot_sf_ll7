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


def run_checks(root: Path = REPO_ROOT) -> dict[str, object]:
    """Run all instruction-reference checks and return a JSON-ready report."""
    graph_result = check_instruction_graph(root)
    ownership_errors = check_routing_ownership(root)
    return {
        "schema": "instruction_references.v1",
        "root": str(root),
        "routing_owner": ROUTING_OWNER,
        "files": list(graph_result.files),
        "references_checked": graph_result.checked,
        "optional_references": graph_result.optional_skipped,
        "errors": graph_result.errors + ownership_errors,
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
