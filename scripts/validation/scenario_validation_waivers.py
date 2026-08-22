"""Load and match exact waivers for pinned scenario validation findings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


class WaiverValidationError(ValueError):
    """Raised when a scenario-validation waiver document is not fail-closed."""


def load_waiver_rows(path: Path, section: str) -> list[dict[str, object]]:
    """Load one typed waiver section and require shared provenance metadata."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WaiverValidationError(f"waiver file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise WaiverValidationError(f"invalid YAML in waiver file {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise WaiverValidationError(f"waiver file {path} must contain a mapping")
    if document.get("schema") != "scenario_validation_waivers.v1":
        raise WaiverValidationError(
            f"waiver file {path} must declare schema scenario_validation_waivers.v1"
        )

    rows = document.get(section)
    if not isinstance(rows, list):
        raise WaiverValidationError(f"waiver section {section!r} in {path} must be a list")

    parsed: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise WaiverValidationError(f"{section}[{index}] in {path} must be a mapping")
        for field in ("rationale", "decision_ref"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise WaiverValidationError(
                    f"{section}[{index}] in {path} requires non-empty {field}"
                )
        parsed.append(dict(row))
    return parsed


def canonical_repo_path(value: str) -> str:
    """Render an input path relative to the repository when possible."""

    path = Path(value)
    resolved = path if path.is_absolute() else Path.cwd() / path
    try:
        return resolved.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_exact_waivers(
    actual_rows: Sequence[Mapping[str, object]],
    waiver_rows: Sequence[Mapping[str, object]],
    *,
    identity_fields: Sequence[str],
    evidence_matches: Callable[[Mapping[str, object], Mapping[str, object]], bool],
    label: str,
) -> None:
    """Require a one-to-one identity and evidence match for every finding."""

    actual_by_key, duplicate_actual = _index_rows(actual_rows, identity_fields)
    waiver_by_key, duplicate_waivers = _index_rows(waiver_rows, identity_fields)

    errors: list[str] = []
    for kind, duplicates in (
        ("duplicate actual", duplicate_actual),
        ("duplicate waiver", duplicate_waivers),
    ):
        if duplicates:
            errors.append(f"{kind} {label} identities: {_format_keys(duplicates)}")

    missing = sorted(set(actual_by_key) - set(waiver_by_key))
    stale = sorted(set(waiver_by_key) - set(actual_by_key))
    if missing:
        errors.append(f"missing {label} waivers: {_format_keys(missing)}")
    if stale:
        errors.append(f"stale {label} waivers: {_format_keys(stale)}")

    mismatched = [
        key
        for key in sorted(set(actual_by_key) & set(waiver_by_key))
        if not evidence_matches(actual_by_key[key], waiver_by_key[key])
    ]
    if mismatched:
        errors.append(f"changed expected evidence for {label}: {_format_keys(mismatched)}")

    if errors:
        raise WaiverValidationError("; ".join(errors))


def _index_rows(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> tuple[dict[tuple[str, ...], Mapping[str, object]], list[tuple[str, ...]]]:
    """Index rows by identity and return duplicate identities separately."""

    indexed: dict[tuple[str, ...], Mapping[str, object]] = {}
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = _identity_key(row, fields)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    return indexed, duplicates


def _identity_key(row: Mapping[str, object], fields: Sequence[str]) -> tuple[str, ...]:
    """Create a stable identity key while treating omitted optional fields as null."""

    return tuple(_canonical_value(row.get(field)) for field in fields)


def _canonical_value(value: object) -> str:
    """Serialize identity values deterministically for comparisons and diagnostics."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _format_keys(keys: Sequence[tuple[str, ...]]) -> str:
    """Render identity keys without depending on Python tuple formatting."""

    return "[" + ", ".join("(" + ", ".join(key) + ")" for key in keys) + "]"
