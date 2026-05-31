#!/usr/bin/env python3
"""Validate user-facing planner, policy-card, suite, and leaderboard docs.

The checker is intentionally metadata-only. It catches drift between public-facing
Markdown surfaces and the registries or durable evidence paths they summarize.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")
DEFAULT_LEARNED_POLICY_REGISTRY = Path("docs/context/policy_search/learned_policy_registry.md")
DEFAULT_MODEL_REGISTRY = Path("model/registry.yaml")
PLANNER_ZOO_PATH = Path("docs/planner_zoo/index.md")
POLICY_CARDS_DIR = Path("docs/policy_cards")
LEADERBOARDS_DIR = Path("docs/leaderboards")
BENCHMARK_SUITES_DIR = Path("docs/benchmark_suites")

LOCAL_ONLY_URI_PREFIXES = (
    "output/",
    "results/",
    ".git/",
    ".venv/",
    "/home/",
    "/tmp/",
    "/var/tmp/",
)
LEADERBOARD_REQUIRED_COLUMNS = {
    "planner",
    "suite",
    "success",
    "collision",
    "near_miss",
    "low_progress",
    "min_distance",
    "runtime",
    "benchmark_track",
    "evidence_uri",
    "status",
    "claim_boundary",
}
BENCHMARK_SUITE_REQUIRED_FIELDS = {
    "suite_id",
    "benchmark_track",
    "status",
}
BENCHMARK_SUITE_REQUIRED_HEADINGS = {
    "Purpose",
    "Scenarios And Seeds",
    "Eligible Planners",
    "Metrics",
    "Expected Runtime",
    "Claim Boundary",
}
BENCHMARK_SUITE_COMMAND_HEADINGS = {"Canonical Command", "Canonical Commands"}
VALID_LEADERBOARD_STATUSES = {
    "completed_smoke_not_benchmark_evidence",
    "excluded",
    "failed",
    "not_available",
    "not_yet_populated",
    "pass",
    "revise",
    "successful_evidence",
}
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_TOKEN_RE = re.compile(r"`([^`]+)`")
YAML_FENCE_RE = re.compile(r"```yaml\s*\n(.*?)\n```", re.DOTALL)
PATHLIKE_RE = re.compile(
    r"(?:configs|docs|model|scripts|tests|SLURM|memory|examples)/[A-Za-z0-9_./:-]+"
)


@dataclass(frozen=True)
class PlatformDocsIssue:
    """One platform-docs validation issue."""

    path: str
    message: str


def _load_yaml(path: Path) -> Any:
    """Load a YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _split_markdown_table_row(line: str) -> list[str]:
    """Split a simple Markdown table row into cells."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _clean_markdown_cell(value: str) -> str:
    """Normalize a Markdown table cell for validation."""
    return value.strip().strip("`").strip()


def _is_markdown_separator(cells: list[str]) -> bool:
    """Return whether table cells are the Markdown separator row."""
    return bool(cells) and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells)


def _iter_markdown_tables(markdown: str) -> list[list[dict[str, str]]]:
    """Parse pipe tables from Markdown into row dictionaries."""
    tables: list[list[dict[str, str]]] = []
    headers: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in markdown.splitlines():
        cells = _split_markdown_table_row(line)
        if not cells:
            if headers is not None:
                tables.append(rows)
                headers = None
                rows = []
            continue
        cleaned = [_clean_markdown_cell(cell) for cell in cells]
        if headers is None:
            headers = cleaned
            continue
        if _is_markdown_separator(cleaned):
            continue
        if len(cleaned) < len(headers):
            continue
        rows.append(dict(zip(headers, cleaned, strict=False)))
    if headers is not None:
        tables.append(rows)
    return tables


def _extract_table_with_columns(markdown: str, required_columns: set[str]) -> list[dict[str, str]]:
    """Return rows for the first Markdown table containing all required columns."""
    for table in _iter_markdown_tables(markdown):
        if not table:
            continue
        if required_columns <= set(table[0]):
            return table
    return []


def _extract_table_by_first_column(markdown: str, first_column: str) -> list[dict[str, str]]:
    """Return rows for the first Markdown table whose first header matches."""
    for table in _iter_markdown_tables(markdown):
        if table and next(iter(table[0]), "") == first_column:
            return table
    return []


def _strip_markdown_link(value: str) -> str:
    """Return the first Markdown link target in a cell, or the cleaned text."""
    match = MARKDOWN_LINK_RE.search(value)
    if match:
        return match.group(2).strip()
    return _clean_markdown_cell(value)


def _repo_relative(repo_root: Path, path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_doc_link(repo_root: Path, source_path: Path, target: str) -> Path | None:
    """Resolve a local Markdown link target."""
    if not target or target.startswith(("http://", "https://", "#", "mailto:")):
        return None
    path_text = target.split("#", 1)[0]
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (repo_root / source_path.parent / path).resolve()


def _is_local_only_uri(value: str) -> bool:
    """Return whether a URI points at a local, disposable artifact location."""
    stripped = _strip_markdown_link(value).strip()
    return stripped.startswith(LOCAL_ONLY_URI_PREFIXES) or ".worktrees/" in stripped


def _load_candidate_rows(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Load policy-search candidate registry rows."""
    path = repo_root / DEFAULT_CANDIDATE_REGISTRY
    payload = _load_yaml(path)
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    return dict(candidates) if isinstance(candidates, dict) else {}


def _load_learned_policy_rows(repo_root: Path) -> dict[str, dict[str, str]]:
    """Load learned-policy registry rows by policy_id."""
    path = repo_root / DEFAULT_LEARNED_POLICY_REGISTRY
    markdown = path.read_text(encoding="utf-8")
    rows = _extract_table_by_first_column(markdown, "policy_id")
    return {row["policy_id"]: row for row in rows if row.get("policy_id")}


def _load_model_ids(repo_root: Path) -> set[str]:
    """Load known model IDs from the model registry."""
    path = repo_root / DEFAULT_MODEL_REGISTRY
    payload = _load_yaml(path)
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return set()
    return {str(model.get("model_id")) for model in models if isinstance(model, dict)}


def _validate_pathlike_tokens(
    issues: list[PlatformDocsIssue],
    *,
    repo_root: Path,
    source_path: Path,
    markdown: str,
    prefix: str,
) -> None:
    """Validate repository path tokens that are meant to be durable local paths."""
    for match in MARKDOWN_LINK_RE.finditer(markdown):
        target_path = _resolve_doc_link(repo_root, source_path, match.group(2).strip())
        if target_path is not None and not target_path.exists():
            issues.append(
                PlatformDocsIssue(
                    f"{prefix}.links",
                    f"linked path does not exist: {match.group(2).strip()}",
                )
            )
    for match in CODE_TOKEN_RE.finditer(markdown):
        token = match.group(1).strip()
        if not PATHLIKE_RE.fullmatch(token):
            continue
        target_path = repo_root / token
        if not target_path.exists():
            issues.append(PlatformDocsIssue(f"{prefix}.paths", f"path does not exist: {token}"))


def validate_planner_zoo(repo_root: Path) -> list[PlatformDocsIssue]:
    """Validate planner-zoo candidate and command references."""
    path = repo_root / PLANNER_ZOO_PATH
    issues: list[PlatformDocsIssue] = []
    if not path.exists():
        return [PlatformDocsIssue(PLANNER_ZOO_PATH.as_posix(), "planner zoo page is missing")]

    markdown = path.read_text(encoding="utf-8")
    rel_path = PLANNER_ZOO_PATH.as_posix()
    candidates = _load_candidate_rows(repo_root)
    _validate_pathlike_tokens(
        issues,
        repo_root=repo_root,
        source_path=PLANNER_ZOO_PATH,
        markdown=markdown,
        prefix=rel_path,
    )
    _validate_planner_zoo_commands(issues, markdown, candidates, rel_path)
    _validate_planner_zoo_runnable_rows(issues, markdown, candidates, repo_root, rel_path)
    return issues


def _iter_candidate_stage_commands(markdown: str) -> list[tuple[str, str]]:
    """Return concrete --candidate/--stage pairs from command lines."""
    commands: list[tuple[str, str]] = []
    for line in markdown.splitlines():
        if "run_policy_search_candidate.py" not in line:
            continue
        candidate_match = re.search(r"--candidate\s+([A-Za-z0-9_:-]+)", line)
        stage_match = re.search(r"--stage\s+([A-Za-z0-9_:-]+)", line)
        if candidate_match and stage_match:
            commands.append((candidate_match.group(1), stage_match.group(1)))
    return commands


def _validate_planner_zoo_commands(
    issues: list[PlatformDocsIssue],
    markdown: str,
    candidates: dict[str, dict[str, Any]],
    rel_path: str,
) -> None:
    """Validate concrete planner-zoo runner commands."""
    candidate_ids = set(candidates)
    stage_by_candidate = {
        candidate_id: set(row.get("required_stages") or [])
        for candidate_id, row in candidates.items()
        if isinstance(row, dict)
    }
    for candidate_id, stage in _iter_candidate_stage_commands(markdown):
        if candidate_id not in candidate_ids:
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.commands",
                    f"unknown candidate in runnable command: {candidate_id}",
                )
            )
            continue
        allowed = stage_by_candidate.get(candidate_id, set())
        if allowed and stage not in allowed:
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.commands",
                    f"stage {stage} is not listed for candidate {candidate_id}",
                )
            )


def _validate_planner_zoo_runnable_rows(
    issues: list[PlatformDocsIssue],
    markdown: str,
    candidates: dict[str, dict[str, Any]],
    repo_root: Path,
    rel_path: str,
) -> None:
    """Validate the planner-zoo current runnable table."""
    candidate_ids = set(candidates)
    runnable_rows = _extract_table_with_columns(
        markdown,
        {"Candidate", "Config", "Local smoke command"},
    )
    for index, row in enumerate(runnable_rows):
        candidate_id = _clean_markdown_cell(row.get("Candidate", ""))
        if candidate_id and candidate_id not in candidate_ids:
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.current_runnable[{index}].Candidate",
                    f"unknown candidate: {candidate_id}",
                )
            )
        config_path = _strip_markdown_link(row.get("Config", ""))
        resolved_config = _resolve_doc_link(repo_root, PLANNER_ZOO_PATH, config_path)
        registry_config = candidates.get(candidate_id, {}).get("candidate_config_path")
        if resolved_config is not None and not resolved_config.exists():
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.current_runnable[{index}].Config",
                    f"config path does not exist: {config_path}",
                )
            )
        if registry_config and resolved_config is not None:
            expected = (repo_root / str(registry_config)).resolve()
            if resolved_config != expected:
                issues.append(
                    PlatformDocsIssue(
                        f"{rel_path}.current_runnable[{index}].Config",
                        f"config does not match registry path {registry_config}",
                    )
                )


def _load_first_yaml_fence(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load the first fenced YAML block from a Markdown file."""
    markdown = path.read_text(encoding="utf-8")
    match = YAML_FENCE_RE.search(markdown)
    if not match:
        return None, markdown
    payload = yaml.safe_load(match.group(1))
    return (payload if isinstance(payload, dict) else None), markdown


def _extract_field_table(markdown: str) -> dict[str, str]:
    """Extract the first Field/Value table as a mapping."""
    rows = _extract_table_with_columns(markdown, {"Field", "Value"})
    return {row["Field"].strip(): row.get("Value", "").strip() for row in rows}


def validate_policy_cards(repo_root: Path) -> list[PlatformDocsIssue]:
    """Validate learned-policy cards against registries and durable artifact fields."""
    issues: list[PlatformDocsIssue] = []
    cards_dir = repo_root / POLICY_CARDS_DIR
    if not cards_dir.exists():
        return []
    candidates = _load_candidate_rows(repo_root)
    learned_rows = _load_learned_policy_rows(repo_root)
    model_ids = _load_model_ids(repo_root)
    for path in sorted(cards_dir.glob("*.md")):
        _validate_policy_card_path(
            issues,
            path,
            repo_root=repo_root,
            candidates=candidates,
            learned_rows=learned_rows,
            model_ids=model_ids,
        )
    return issues


def _validate_policy_card_path(
    issues: list[PlatformDocsIssue],
    path: Path,
    *,
    repo_root: Path,
    candidates: dict[str, dict[str, Any]],
    learned_rows: dict[str, dict[str, str]],
    model_ids: set[str],
) -> None:
    """Validate one policy-card Markdown file."""
    rel_path = _repo_relative(repo_root, path)
    if path.name == "README.md":
        _validate_pathlike_tokens(
            issues,
            repo_root=repo_root,
            source_path=Path(rel_path),
            markdown=path.read_text(encoding="utf-8"),
            prefix=rel_path,
        )
        return
    summary, markdown = _load_first_yaml_fence(path)
    if summary is None:
        issues.append(PlatformDocsIssue(f"{rel_path}.summary", "first YAML block is required"))
        return
    _validate_pathlike_tokens(
        issues,
        repo_root=repo_root,
        source_path=Path(rel_path),
        markdown=markdown,
        prefix=rel_path,
    )
    _validate_policy_card_summary(
        issues,
        summary,
        rel_path=rel_path,
        candidates=candidates,
        learned_rows=learned_rows,
    )
    _validate_policy_card_field_table(
        issues,
        markdown,
        rel_path=rel_path,
        model_ids=model_ids,
    )


def _validate_policy_card_summary(
    issues: list[PlatformDocsIssue],
    summary: dict[str, Any],
    *,
    rel_path: str,
    candidates: dict[str, dict[str, Any]],
    learned_rows: dict[str, dict[str, str]],
) -> str:
    """Validate one policy-card YAML summary and return its policy ID."""
    for field in ("policy_id", "policy_family", "card_status", "benchmark_track"):
        if not str(summary.get(field) or "").strip():
            issues.append(PlatformDocsIssue(f"{rel_path}.summary.{field}", "is required"))
    policy_id = str(summary.get("policy_id") or "").strip()
    if policy_id and policy_id not in learned_rows and policy_id not in candidates:
        issues.append(
            PlatformDocsIssue(
                f"{rel_path}.summary.policy_id",
                f"policy ID is absent from learned-policy and candidate registries: {policy_id}",
            )
        )
    registry_status = summary.get("registry_status")
    if not isinstance(registry_status, dict):
        issues.append(PlatformDocsIssue(f"{rel_path}.summary.registry_status", "must be a mapping"))
    else:
        _validate_policy_card_registry_status(
            issues,
            registry_status,
            learned_rows.get(policy_id, {}),
            rel_path=rel_path,
        )
    not_for = summary.get("not_for")
    if not isinstance(not_for, list) or not not_for:
        issues.append(PlatformDocsIssue(f"{rel_path}.summary.not_for", "must be a list"))
    return policy_id


def _validate_policy_card_registry_status(
    issues: list[PlatformDocsIssue],
    registry_status: dict[str, Any],
    learned_row: dict[str, str],
    *,
    rel_path: str,
) -> None:
    """Validate registry-status mapping from a policy-card summary."""
    for status_field in ("integration_status", "reproducibility_status", "benchmark_status"):
        value = str(registry_status.get(status_field) or "").strip()
        if not value:
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.summary.registry_status.{status_field}",
                    "is required",
                )
            )
        elif learned_row and value != learned_row.get(status_field):
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.summary.registry_status.{status_field}",
                    f"does not match learned-policy registry value {learned_row[status_field]}",
                )
            )


def _validate_policy_card_field_table(
    issues: list[PlatformDocsIssue],
    markdown: str,
    *,
    rel_path: str,
    model_ids: set[str],
) -> None:
    """Validate policy-card Field/Value artifact metadata."""
    field_table = _extract_field_table(markdown)
    model_id = _clean_markdown_cell(field_table.get("Model id", ""))
    if model_id and model_id not in model_ids:
        issues.append(
            PlatformDocsIssue(
                f"{rel_path}.artifacts.Model id",
                f"unknown model registry ID: {model_id}",
            )
        )
    for field, value in field_table.items():
        if "checkpoint" in field.lower() and _is_local_only_uri(value):
            if "cache" not in value.lower() and "local" not in field.lower():
                issues.append(
                    PlatformDocsIssue(
                        f"{rel_path}.artifacts.{field}",
                        "checkpoint fields must use durable URIs or explicit cache wording",
                    )
                )


def validate_leaderboards(repo_root: Path) -> list[PlatformDocsIssue]:
    """Validate static leaderboard row contracts."""
    issues: list[PlatformDocsIssue] = []
    directory = repo_root / LEADERBOARDS_DIR
    if not directory.exists():
        return []
    for path in sorted(directory.glob("*.md")):
        rel_path = _repo_relative(repo_root, path)
        if path.name == "README.md":
            _validate_pathlike_tokens(
                issues,
                repo_root=repo_root,
                source_path=Path(rel_path),
                markdown=path.read_text(encoding="utf-8"),
                prefix=rel_path,
            )
            continue
        markdown = path.read_text(encoding="utf-8")
        rows = _extract_table_with_columns(markdown, LEADERBOARD_REQUIRED_COLUMNS)
        if not rows:
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.table",
                    "must contain a leaderboard table with the required row contract",
                )
            )
            continue
        for index, row in enumerate(rows):
            _validate_leaderboard_row(issues, row, repo_root, rel_path=rel_path, index=index)
    return issues


def _validate_leaderboard_row(
    issues: list[PlatformDocsIssue],
    row: dict[str, str],
    repo_root: Path,
    *,
    rel_path: str,
    index: int,
) -> None:
    """Validate one static leaderboard row."""
    for column in sorted(LEADERBOARD_REQUIRED_COLUMNS):
        if not str(row.get(column) or "").strip():
            issues.append(PlatformDocsIssue(f"{rel_path}.rows[{index}].{column}", "is required"))
    evidence_uri = row.get("evidence_uri", "")
    target = _strip_markdown_link(evidence_uri)
    if _is_local_only_uri(target):
        issues.append(
            PlatformDocsIssue(
                f"{rel_path}.rows[{index}].evidence_uri",
                "must not point at a worktree-local artifact path",
            )
        )
    target_path = _resolve_doc_link(repo_root, Path(rel_path), target)
    if target_path is not None and not target_path.exists():
        issues.append(
            PlatformDocsIssue(
                f"{rel_path}.rows[{index}].evidence_uri",
                f"evidence path does not exist: {target}",
            )
        )
    status = _clean_markdown_cell(row.get("status", ""))
    if status not in VALID_LEADERBOARD_STATUSES:
        issues.append(
            PlatformDocsIssue(f"{rel_path}.rows[{index}].status", f"unknown status: {status}")
        )
    benchmark_track = _clean_markdown_cell(row.get("benchmark_track", ""))
    if benchmark_track in {"", "unknown", "not_recorded"}:
        issues.append(
            PlatformDocsIssue(
                f"{rel_path}.rows[{index}].benchmark_track",
                "must name a benchmark track or explicit non-benchmark boundary",
            )
        )
    if _clean_markdown_cell(row.get("claim_boundary", "")) in {"", "not_recorded"}:
        issues.append(
            PlatformDocsIssue(
                f"{rel_path}.rows[{index}].claim_boundary",
                "must describe the claim boundary",
            )
        )


def _validate_benchmark_suite_page(
    repo_root: Path,
    path: Path,
    *,
    issues: list[PlatformDocsIssue],
) -> None:
    """Validate one benchmark-suite catalog page."""
    rel_path = _repo_relative(repo_root, path)
    summary, markdown = _load_first_yaml_fence(path)
    if summary is None:
        issues.append(PlatformDocsIssue(f"{rel_path}.summary", "first YAML block is required"))
        return
    for field in sorted(BENCHMARK_SUITE_REQUIRED_FIELDS):
        if not summary.get(field):
            issues.append(PlatformDocsIssue(f"{rel_path}.summary.{field}", "is required"))
    headings = set(re.findall(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    for heading in sorted(BENCHMARK_SUITE_REQUIRED_HEADINGS):
        if heading not in headings:
            issues.append(PlatformDocsIssue(f"{rel_path}.section.{heading}", "is required"))
    if not headings & BENCHMARK_SUITE_COMMAND_HEADINGS:
        issues.append(PlatformDocsIssue(f"{rel_path}.section.Canonical Command", "is required"))
    command_sections = re.findall(
        r"^##\s+Canonical Commands?\s*\n(.*?)(?:\n##\s+|\Z)",
        markdown,
        flags=re.DOTALL | re.MULTILINE,
    )
    command = command_sections[0] if command_sections else ""
    for token in PATHLIKE_RE.findall(command):
        if not (repo_root / token).exists():
            issues.append(
                PlatformDocsIssue(
                    f"{rel_path}.section.Canonical Command",
                    f"command path does not exist: {token}",
                )
            )
    _validate_pathlike_tokens(
        issues,
        repo_root=repo_root,
        source_path=Path(rel_path),
        markdown=markdown,
        prefix=rel_path,
    )


def validate_benchmark_suites(repo_root: Path) -> list[PlatformDocsIssue]:
    """Validate benchmark-suite catalog pages when the catalog exists."""
    issues: list[PlatformDocsIssue] = []
    directory = repo_root / BENCHMARK_SUITES_DIR
    if not directory.exists():
        return []
    for path in sorted(directory.glob("*.md")):
        if path.name == "README.md":
            _validate_pathlike_tokens(
                issues,
                repo_root=repo_root,
                source_path=Path(_repo_relative(repo_root, path)),
                markdown=path.read_text(encoding="utf-8"),
                prefix=_repo_relative(repo_root, path),
            )
            continue
        _validate_benchmark_suite_page(repo_root, path, issues=issues)
    return issues


def validate_platform_docs(repo_root: Path = DEFAULT_REPO_ROOT) -> list[PlatformDocsIssue]:
    """Validate all current platform documentation surfaces."""
    repo_root = repo_root.resolve()
    issues: list[PlatformDocsIssue] = []
    issues.extend(validate_planner_zoo(repo_root))
    issues.extend(validate_policy_cards(repo_root))
    issues.extend(validate_leaderboards(repo_root))
    issues.extend(validate_benchmark_suites(repo_root))
    return issues


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root to validate.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON issues.")
    return parser.parse_args()


def main() -> int:
    """Run the platform-docs validator."""
    args = _parse_args()
    issues = validate_platform_docs(args.repo_root)
    if args.json:
        print(json.dumps([issue.__dict__ for issue in issues], indent=2, sort_keys=True))
    else:
        if not issues:
            print("Platform docs validation passed.")
        for issue in issues:
            print(f"{issue.path}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
