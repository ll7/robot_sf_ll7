"""Check-only and optional write-backfill for research manifest lineage fields.

This module scans manifest files for lineage contract completeness, classifies
each missing field as missing / inferred / ambiguous / blocked, and produces a
reviewable backfill plan.  It never scans or rewrites the repository by default.

Usage::

    # Check-only (default): report what needs attention
    python -m robot_sf.benchmark.manifest_lineage_backfill manifest.json

    # Check-only with explicit paths
    python -m robot_sf.benchmark.manifest_lineage_backfill manifest.json other.json

    # Write-backfill (requires explicit paths + flag)
    python -m robot_sf.benchmark.manifest_lineage_backfill --write-backfill manifest.json

The module reuses the shared ``MANDATORY_LINEAGE_FIELDS`` and
``validate_lineage_contract`` from :mod:`robot_sf.benchmark.manifest_lineage`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from robot_sf.benchmark.manifest_lineage import (
    MANDATORY_LINEAGE_FIELDS,
    validate_lineage_contract,
)

# Field status classification


class FieldStatus(StrEnum):
    """Classification for a lineage field's state in a manifest."""

    PRESENT = "present"
    MISSING = "missing"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"


# Inference rules

InferenceValue = str | dict[str, Any]

# Mapping: lineage field -> nearby proxy paths that may already carry the same
# lineage fact in old manifests. Each path is explicit so a single proxy source
# is counted once, and conflicting sources remain review-visible.
INFERENCE_PATHS: dict[str, tuple[str, ...]] = {
    "source": ("metadata.source", "provenance"),
    "generator_id": ("metadata.generator_id", "config.generator_id"),
    "validator_version": (
        "metadata.validator_version",
        "metadata.version",
        "config.validator_version",
    ),
    "schema_version": ("metadata.schema_version",),
    "claim_boundary": ("metadata.claim_boundary", "claim.claim_boundary"),
    "evidence_tier": ("metadata.evidence_tier", "claim.evidence_tier"),
    "denominator_policy": ("metadata.denominator_policy",),
    "execution_gate": ("metadata.execution_gate",),
}


_ABSENT = object()


def _lookup_path(manifest: dict[str, Any], path: str) -> Any:
    """Return a dotted-path value from a manifest, or a sentinel when absent."""
    current: Any = manifest
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _ABSENT
        current = current[part]
    return current


def _valid_candidate(field_name: str, value: Any) -> InferenceValue | None:
    """Normalize a proxy value when it has the required lineage field type.

    Returns:
        Normalized proxy value, or None when the candidate is absent or invalid.
    """
    if field_name in {
        "generator_id",
        "validator_version",
        "schema_version",
        "evidence_tier",
        "denominator_policy",
        "execution_gate",
    }:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
    if field_name == "source":
        return value if isinstance(value, dict) and value else None
    if field_name == "claim_boundary":
        if isinstance(value, str) and value.strip():
            return value.strip()
        return value if isinstance(value, dict) and value else None
    return None


def _candidate_values(
    manifest: dict[str, Any], field_name: str
) -> list[tuple[str, InferenceValue]]:
    """Return labeled candidate lineage values from explicit nearby paths.

    Args:
        manifest: The parsed manifest payload.
        field_name: The lineage field to collect candidates for.

    Returns:
        List of ``(source_label, normalized_value)`` tuples for valid candidates.
    """
    candidates: list[tuple[str, InferenceValue]] = []
    for source_label in INFERENCE_PATHS.get(field_name, ()):
        value = _valid_candidate(field_name, _lookup_path(manifest, source_label))
        if value is not None:
            candidates.append((source_label, value))
    return candidates


def _invalid_candidate_labels(manifest: dict[str, Any], field_name: str) -> list[str]:
    """Return labels for present nearby values that cannot safely infer a field.

    Returns:
        Dotted proxy labels whose values are present but invalid for the field.
    """
    invalid: list[str] = []
    for source_label in INFERENCE_PATHS.get(field_name, ()):
        raw_value = _lookup_path(manifest, source_label)
        if raw_value is _ABSENT:
            continue
        if _valid_candidate(field_name, raw_value) is None:
            invalid.append(source_label)
    return invalid


# Backfill plan data model


@dataclass(frozen=True, slots=True)
class FieldBackfillEntry:
    """One lineage field needing attention in a manifest."""

    field_name: str
    status: FieldStatus
    inferred_value: str | dict[str, Any] | None = None
    inferred_from: str | None = None
    reason: str = ""
    candidate_sources: tuple[str, ...] = ()
    conflicting_sources: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManifestBackfillPlan:
    """Backfill plan for a single manifest file."""

    path: str
    validation_errors: list[str]
    fields: list[FieldBackfillEntry]
    has_inferred: bool = False
    has_ambiguous: bool = False
    has_blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping.

        Returns:
            Dictionary with path, validation_errors, and field details.
        """
        return {
            "path": self.path,
            "validation_errors": self.validation_errors,
            "fields": [asdict(f) for f in self.fields],
            "has_inferred": self.has_inferred,
            "has_ambiguous": self.has_ambiguous,
            "has_blocked": self.has_blocked,
        }


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    """Full backfill plan across all scanned manifests."""

    manifests: list[ManifestBackfillPlan]
    total_missing: int = 0
    total_inferred: int = 0
    total_ambiguous: int = 0
    total_blocked: int = 0
    write_mode: bool = False
    written_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping.

        Returns:
            Dictionary with summary counts and per-manifest details.
        """
        return {
            "manifests": [m.to_dict() for m in self.manifests],
            "total_missing": self.total_missing,
            "total_inferred": self.total_inferred,
            "total_ambiguous": self.total_ambiguous,
            "total_blocked": self.total_blocked,
            "write_mode": self.write_mode,
            "written_paths": self.written_paths,
        }


# Core analysis


def _classify_field(
    manifest: dict[str, Any],
    field_name: str,
) -> FieldBackfillEntry:
    """Classify the status of a single lineage field in a manifest.

    Args:
        manifest: The parsed manifest payload.
        field_name: The lineage field to classify.

    Returns:
        FieldBackfillEntry with status, inferred value, and reason.
    """
    # Already present?
    if field_name in manifest:
        return FieldBackfillEntry(
            field_name=field_name,
            status=FieldStatus.PRESENT,
        )

    # Try nearby proxy fields.
    values = _candidate_values(manifest, field_name)
    invalid_labels = _invalid_candidate_labels(manifest, field_name)

    if len(values) == 0:
        if invalid_labels:
            return FieldBackfillEntry(
                field_name=field_name,
                status=FieldStatus.BLOCKED,
                blocked_by=tuple(invalid_labels),
                reason=(
                    f"nearby field has incompatible value for {field_name}: "
                    f"{', '.join(invalid_labels)}"
                ),
            )
        return FieldBackfillEntry(
            field_name=field_name,
            status=FieldStatus.MISSING,
            reason=f"no nearby field available to infer {field_name}",
        )
    if invalid_labels:
        source_labels = [label for label, _ in values]
        return FieldBackfillEntry(
            field_name=field_name,
            status=FieldStatus.AMBIGUOUS,
            candidate_sources=tuple(source_labels),
            blocked_by=tuple(invalid_labels),
            reason=(
                f"valid and invalid inference candidates for {field_name}: "
                f"{', '.join(source_labels + invalid_labels)}"
            ),
        )
    if len(values) == 1:
        source_label, val = values[0]
        return FieldBackfillEntry(
            field_name=field_name,
            status=FieldStatus.INFERRED,
            inferred_value=val,
            inferred_from=source_label,
            candidate_sources=(source_label,),
            reason=f"inferred from {source_label}",
        )
    first_label, first_value = values[0]
    labels = tuple(label for label, _ in values)
    if all(value == first_value for _, value in values[1:]):
        return FieldBackfillEntry(
            field_name=field_name,
            status=FieldStatus.INFERRED,
            inferred_value=first_value,
            inferred_from=first_label,
            candidate_sources=labels,
            reason=f"inferred from matching candidates: {', '.join(labels)}",
        )

    # Multiple conflicting candidates: ambiguous.
    return FieldBackfillEntry(
        field_name=field_name,
        status=FieldStatus.AMBIGUOUS,
        conflicting_sources=labels,
        reason=f"multiple inference candidates: {', '.join(labels)}",
    )


def analyze_manifest(
    manifest: dict[str, Any],
    *,
    path: str | None = None,
) -> ManifestBackfillPlan:
    """Analyze a single manifest for lineage field completeness.

    Args:
        manifest: The parsed manifest payload (must be a dict).
        path: Optional path label for the manifest (for reporting).

    Returns:
        ManifestBackfillPlan with per-field classification.

    Raises:
        ValueError: If manifest is not a dictionary mapping.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Manifest contract payload must be a dictionary mapping.")
    validation_errors = validate_lineage_contract(manifest)
    fields: list[FieldBackfillEntry] = []
    for field_name in MANDATORY_LINEAGE_FIELDS:
        entry = _classify_field(manifest, field_name)
        fields.append(entry)

    has_inferred = any(f.status == FieldStatus.INFERRED for f in fields)
    has_ambiguous = any(f.status == FieldStatus.AMBIGUOUS for f in fields)
    has_blocked = any(f.status == FieldStatus.BLOCKED for f in fields)

    return ManifestBackfillPlan(
        path=path or "<inline>",
        validation_errors=validation_errors,
        fields=fields,
        has_inferred=has_inferred,
        has_ambiguous=has_ambiguous,
        has_blocked=has_blocked,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed manifest dictionary.

    Raises:
        SystemExit: If the file cannot be read or parsed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        sys.stderr.write(f"Error reading {path}: {exc}\n")
        raise SystemExit(1) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Error parsing {path}: {exc}\n")
        raise SystemExit(1) from exc
    if not isinstance(payload, dict):
        sys.stderr.write(f"Error: {path} must contain a JSON object\n")
        raise SystemExit(1)
    return payload


def _validate_write_paths(paths: list[str]) -> list[Path]:
    """Validate that write-backfill paths are explicit file paths.

    Refuses broad directory paths (like repo root) to prevent accidental
    repository-wide rewrites.

    Args:
        paths: List of path strings to validate.

    Returns:
        List of validated Path objects.

    Raises:
        SystemExit: If any path is too broad or not an explicit file path.
    """
    resolved: list[Path] = []
    for p in paths:
        path = Path(p).resolve()
        # Refuse directory paths
        if path.is_dir():
            sys.stderr.write(
                f"Error: --write-backfill refuses directory path '{p}'. "
                "Provide explicit file paths.\n"
            )
            raise SystemExit(1)
        resolved.append(path)
    return resolved


# Write-backfill


def _apply_backfill(
    manifest: dict[str, Any],
    plan: ManifestBackfillPlan,
) -> dict[str, Any]:
    """Apply inferred lineage fields to a manifest payload.

    Only INFERRED fields are written.  AMBIGUOUS and BLOCKED fields are left
    untouched so the reviewer can resolve them.

    Args:
        manifest: The mutable manifest payload.
        plan: The backfill plan for this manifest.

    Returns:
        The updated manifest (same dict, mutated in place).
    """
    for entry in plan.fields:
        if entry.status == FieldStatus.INFERRED and entry.inferred_value is not None:
            manifest[entry.field_name] = entry.inferred_value
    return manifest


# Public API


def run_backfill_check(
    paths: list[str],
    *,
    write_backfill: bool = False,
    json_output: bool = False,
) -> BackfillPlan:
    """Run lineage backfill analysis on the given manifest paths.

    In check-only mode (default), this never modifies files.  In write mode,
    it requires explicit file paths and only writes inferred fields.

    Args:
        paths: List of manifest file paths to analyze.
        write_backfill: If True, apply inferred backfill and write files.
        json_output: If True, return JSON output; otherwise print human-readable.

    Returns:
        BackfillPlan with full analysis results.
    """
    resolved_paths: list[Path]
    if write_backfill:
        resolved_paths = _validate_write_paths(paths)
    else:
        resolved_paths = [Path(p).resolve() for p in paths]

    plans: list[ManifestBackfillPlan] = []
    written: list[str] = []

    for path in resolved_paths:
        manifest = _load_manifest(path)
        plan = analyze_manifest(
            manifest,
            path=str(path.relative_to(Path.cwd()))
            if path.is_relative_to(Path.cwd())
            else str(path),
        )
        plans.append(plan)

        if write_backfill and plan.has_inferred:
            manifest = _apply_backfill(manifest, plan)
            path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            written.append(str(path))

    total_missing = sum(1 for p in plans for f in p.fields if f.status == FieldStatus.MISSING)
    total_inferred = sum(1 for p in plans for f in p.fields if f.status == FieldStatus.INFERRED)
    total_ambiguous = sum(1 for p in plans for f in p.fields if f.status == FieldStatus.AMBIGUOUS)
    total_blocked = sum(1 for p in plans for f in p.fields if f.status == FieldStatus.BLOCKED)

    plan = BackfillPlan(
        manifests=plans,
        total_missing=total_missing,
        total_inferred=total_inferred,
        total_ambiguous=total_ambiguous,
        total_blocked=total_blocked,
        write_mode=write_backfill,
        written_paths=written,
    )

    if json_output:
        sys.stdout.write(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n")
    else:
        _print_human_readable(plan)

    return plan


def _print_human_readable(plan: BackfillPlan) -> None:
    """Print a human-readable summary of the backfill plan.

    Args:
        plan: The full backfill plan to summarize.
    """
    mode_label = "WRITE-BACKFILL" if plan.write_mode else "CHECK-ONLY"
    sys.stdout.write(f"=== Manifest Lineage Backfill ({mode_label}) ===\n")
    sys.stdout.write(
        f"Manifests: {len(plan.manifests)} | "
        f"Missing: {plan.total_missing} | "
        f"Inferred: {plan.total_inferred} | "
        f"Ambiguous: {plan.total_ambiguous} | "
        f"Blocked: {plan.total_blocked}\n"
    )
    if plan.written_paths:
        sys.stdout.write(f"Written: {', '.join(plan.written_paths)}\n")
    sys.stdout.write("\n")

    for mp in plan.manifests:
        sys.stdout.write(f"--- {mp.path} ---\n")
        if mp.validation_errors:
            sys.stdout.write(f"  Validation errors: {'; '.join(mp.validation_errors)}\n")
        for f in mp.fields:
            if f.status == FieldStatus.PRESENT:
                continue
            marker = {
                FieldStatus.MISSING: "MISSING",
                FieldStatus.INFERRED: f"INFERRED from {f.inferred_from}",
                FieldStatus.AMBIGUOUS: "AMBIGUOUS",
                FieldStatus.BLOCKED: "BLOCKED",
            }[f.status]
            reason = f" ({f.reason})" if f.reason else ""
            sys.stdout.write(f"  {f.field_name}: {marker}{reason}\n")
        sys.stdout.write("\n")


# CLI entry point


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for manifest lineage backfill.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        prog="manifest_lineage_backfill",
        description="Check or backfill lineage fields in research manifests.",
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        help="Manifest JSON files to analyze.",
    )
    parser.add_argument(
        "--write-backfill",
        action="store_true",
        default=False,
        help="Apply inferred backfill and write files. Requires explicit paths.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output machine-readable JSON.",
    )
    args = parser.parse_args(argv)
    run_backfill_check(
        args.manifests,
        write_backfill=args.write_backfill,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    main()
