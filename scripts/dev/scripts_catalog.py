"""Shared loader, validator, and renderer for the scripts command catalog.

The catalog (``scripts/catalog.yaml``) is the machine-readable source of truth for
root-level ``scripts/`` commands. The human-facing ``scripts/README.md`` sections
between the generated markers are rendered from it deterministically.

Owners:
- ``scripts/dev/check_scripts_catalog.py``: fail-closed catalog validation CLI.
- ``scripts/dev/render_scripts_readme.py``: README renderer with ``--check`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CATALOG_PATH = Path("scripts/catalog.yaml")
README_PATH = Path("scripts/README.md")

STATUS_CANONICAL = "canonical"
STATUS_COMPATIBILITY = "compatibility"
STATUS_DEBUG_ONLY = "debug-only"
STATUS_ARCHIVE_CANDIDATE = "archive-candidate"
ALLOWED_STATUSES = (
    STATUS_CANONICAL,
    STATUS_COMPATIBILITY,
    STATUS_DEBUG_ONLY,
    STATUS_ARCHIVE_CANDIDATE,
)

SMOKE_EXPECTED_FAIL_CLOSED = "expected_fail_closed"
ALLOWED_SMOKE_MODES = (None, SMOKE_EXPECTED_FAIL_CLOSED)

ROOT_STATUS_BEGIN = "<!-- BEGIN GENERATED:root-entry-point-status -->"
ROOT_STATUS_END = "<!-- END GENERATED:root-entry-point-status -->"
DIR_OVERVIEW_BEGIN = "<!-- BEGIN GENERATED:directory-overview -->"
DIR_OVERVIEW_END = "<!-- END GENERATED:directory-overview -->"

SUPPORTED_ROOT_SUFFIXES = (".py", ".sh")


class CatalogError(ValueError):
    """Raised when the catalog cannot be parsed or fails schema validation."""


@dataclass
class CatalogCommand:
    """One root-level command entry in the scripts catalog."""

    name: str
    path: str
    domain: str
    status: str
    purpose: str
    invocation: str
    replacement: str | None = None
    replacement_rationale: str | None = None
    smoke_mode: str | None = None
    required_extras: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class Catalog:
    """Parsed scripts catalog: versioned commands plus curated directory rows."""

    version: int
    commands: dict[str, CatalogCommand]
    directories: dict[str, dict[str, Any]] = field(default_factory=dict)
    exclusions: dict[str, dict[str, Any]] = field(default_factory=dict)


def _require_str(value: Any, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context}: '{field_name}' must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, field_name: str, context: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name, context)


def _parse_command(name: str, entry: Any, seen_paths: dict[str, str]) -> CatalogCommand:
    """Parse and schema-check one command entry."""
    context = f"command {name!r}"
    if not isinstance(entry, dict):
        raise CatalogError(f"{context}: entry must be a mapping")
    path = _require_str(entry.get("path"), "path", context)
    if path in seen_paths:
        raise CatalogError(f"{context}: duplicate path {path!r} (also {seen_paths[path]!r})")
    seen_paths[path] = name
    status = _require_str(entry.get("status"), "status", context)
    if status not in ALLOWED_STATUSES:
        allowed = ", ".join(ALLOWED_STATUSES)
        raise CatalogError(f"{context}: unknown status {status!r} (allowed: {allowed})")
    smoke_mode = entry.get("smoke_mode")
    if smoke_mode not in ALLOWED_SMOKE_MODES:
        allowed = ", ".join(str(m) for m in ALLOWED_SMOKE_MODES if m is not None)
        raise CatalogError(
            f"{context}: unknown smoke_mode {smoke_mode!r} (allowed: null, {allowed})"
        )
    extras = entry.get("required_extras", [])
    if not isinstance(extras, list) or not all(isinstance(x, str) for x in extras):
        raise CatalogError(f"{context}: 'required_extras' must be a list of strings")
    return CatalogCommand(
        name=name,
        path=path,
        domain=_require_str(entry.get("domain"), "domain", context),
        status=status,
        purpose=_require_str(entry.get("purpose"), "purpose", context),
        invocation=_require_str(entry.get("invocation"), "invocation", context),
        replacement=_optional_str(entry.get("replacement"), "replacement", context),
        replacement_rationale=_optional_str(
            entry.get("replacement_rationale"), "replacement_rationale", context
        ),
        smoke_mode=smoke_mode,
        required_extras=extras,
        notes=_optional_str(entry.get("notes"), "notes", context),
    )


def parse_catalog(raw: Any) -> Catalog:
    """Parse untyped YAML data into a :class:`Catalog` with strict schema checks."""
    if not isinstance(raw, dict):
        raise CatalogError("catalog root must be a mapping")
    version = raw.get("version")
    if version != 1:
        raise CatalogError(f"unsupported catalog version: {version!r} (expected 1)")
    raw_commands = raw.get("commands")
    if not isinstance(raw_commands, dict) or not raw_commands:
        raise CatalogError("'commands' must be a non-empty mapping")

    seen_paths: dict[str, str] = {}
    commands = {
        name: _parse_command(name, entry, seen_paths) for name, entry in raw_commands.items()
    }

    directories = raw.get("directories", {}) or {}
    exclusions = raw.get("exclusions", []) or []
    if not isinstance(directories, dict):
        raise CatalogError("'directories' must be a mapping")
    if not isinstance(exclusions, list):
        raise CatalogError("'exclusions' must be a list")
    parsed_exclusions: dict[str, dict[str, Any]] = {}
    for item in exclusions:
        if not isinstance(item, dict):
            raise CatalogError("each exclusion must be a mapping")
        name = _require_str(item.get("name"), "name", "exclusion")
        parsed_exclusions[name] = item
    return Catalog(
        version=1, commands=commands, directories=directories, exclusions=parsed_exclusions
    )


def load_catalog(repo_root: Path, catalog_path: Path | None = None) -> Catalog:
    """Load and parse the catalog from disk."""
    path = repo_root / (catalog_path or CATALOG_PATH)
    if not path.is_file():
        raise CatalogError(f"catalog file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise CatalogError(f"catalog YAML parse error: {exc}") from exc
    return parse_catalog(raw)


def inventory_root_commands(repo_root: Path) -> set[str]:
    """Return repo-relative paths of executable/importable direct children of scripts/."""
    scripts_dir = repo_root / "scripts"
    found: set[str] = set()
    for child in sorted(scripts_dir.iterdir()):
        if child.is_file() and child.suffix in SUPPORTED_ROOT_SUFFIXES:
            found.add(child.relative_to(repo_root).as_posix())
    return found


def _validate_commands(catalog: Catalog, repo_root: Path) -> list[str]:
    """Return schema/path errors for command entries."""
    errors: list[str] = []
    for cmd in catalog.commands.values():
        label = f"command {cmd.name!r}"
        if not cmd.path.startswith("scripts/") or Path(cmd.path).parent != Path("scripts"):
            errors.append(f"{label}: path {cmd.path!r} must be a direct child of scripts/")
        if not (repo_root / cmd.path).exists():
            errors.append(f"{label}: path does not exist: {cmd.path}")
        if cmd.status == STATUS_COMPATIBILITY:
            if cmd.replacement is None and not cmd.replacement_rationale:
                errors.append(
                    f"{label}: compatibility command needs an existing 'replacement' "
                    "or a 'replacement_rationale'"
                )
            if cmd.replacement is not None and not (repo_root / cmd.replacement).exists():
                errors.append(f"{label}: replacement does not exist: {cmd.replacement}")
    return errors


def _validate_exclusions(catalog: Catalog, repo_root: Path) -> list[str]:
    """Return schema/path errors for exclusion entries."""
    errors: list[str] = []
    for name, item in catalog.exclusions.items():
        rel = item.get("path")
        if not isinstance(rel, str) or not rel:
            errors.append(f"exclusion {name!r}: missing 'path'")
        elif not (repo_root / rel).exists():
            errors.append(f"exclusion {name!r}: path does not exist: {rel}")
        if not isinstance(item.get("reason"), str) or not item.get("reason"):
            errors.append(f"exclusion {name!r}: missing 'reason'")
    return errors


def validate_catalog(catalog: Catalog, repo_root: Path) -> list[str]:
    """Return fail-closed validation errors; an empty list means the catalog is valid."""
    errors = _validate_commands(catalog, repo_root)
    inventory = inventory_root_commands(repo_root)
    covered = {cmd.path for cmd in catalog.commands.values()}
    excluded_paths = {
        item.get("path") for item in catalog.exclusions.values() if isinstance(item, dict)
    }
    for rel in sorted(inventory - covered - excluded_paths):
        errors.append(f"undocumented root script absent from catalog: {rel}")
    errors.extend(_validate_exclusions(catalog, repo_root))
    for dirname, row in catalog.directories.items():
        readme = row.get("readme") if isinstance(row, dict) else None
        if isinstance(row, dict) and readme is not None and not (repo_root / str(readme)).exists():
            errors.append(f"directory {dirname!r}: readme does not exist: {readme}")
    return errors


def render_root_status_table(catalog: Catalog) -> str:
    """Render the root-level entry point status table from catalog rows."""
    lines = [
        "| Command | Status | Purpose / canonical path or action |",
        "| --- | --- | --- |",
    ]
    for name in sorted(catalog.commands):
        cmd = catalog.commands[name]
        fname = Path(cmd.path).name
        guidance = cmd.purpose
        if cmd.status == STATUS_COMPATIBILITY:
            target = cmd.replacement or cmd.replacement_rationale or ""
            guidance = f"{cmd.purpose} Prefer `{target}`." if target else guidance
        if cmd.smoke_mode == SMOKE_EXPECTED_FAIL_CLOSED:
            guidance = f"{guidance} Fails closed."
        lines.append(f"| `{fname}` | {cmd.status} | {guidance} |")
    return "\n".join(lines)


def render_directory_overview(catalog: Catalog) -> str:
    """Render the compact nested-directory overview from curated catalog rows."""
    lines = ["| Directory | Role |", "| --- | --- |"]
    for dirname in sorted(catalog.directories):
        row = catalog.directories[dirname]
        description = str(row.get("description", "")).strip()
        link = f"[`scripts/{dirname}/`]( {dirname}/ )" if row.get("readme") else f"`{dirname}/`"
        lines.append(f"| {link} | {description} |")
    return "\n".join(lines)


def _replace_section(text: str, begin: str, end: str, rendered: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1 or stop < start:
        missing = []
        if start == -1:
            missing.append(begin)
        if stop == -1:
            missing.append(end)
        raise CatalogError(f"generated-section marker(s) missing from README: {missing}")
    head, tail = text[: start + len(begin)], text[stop:]
    return f"{head}\n{rendered}\n{tail}"


def render_readme(readme_text: str, catalog: Catalog) -> str:
    """Return README text with both generated sections refreshed from the catalog."""
    updated = _replace_section(
        readme_text, ROOT_STATUS_BEGIN, ROOT_STATUS_END, render_root_status_table(catalog)
    )
    updated = _replace_section(
        updated, DIR_OVERVIEW_BEGIN, DIR_OVERVIEW_END, render_directory_overview(catalog)
    )
    return updated
