#!/usr/bin/env python3
"""Classify dissertation and release-bundle artifact freshness.

This helper is intentionally conservative. It records whether an artifact is
safe to cite as current evidence, historically valid evidence, stale evidence
that needs refresh, or blocked for manuscript use. Local-only paths such as
``output/`` are never treated as durable manuscript proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is present in repo env
    yaml = None

SCHEMA_VERSION = "stale_artifact_detector.v1"
CURRENT_ARTIFACT_SCHEMA = "artifact_catalog.v1"
BLOCKING_MARKERS = {
    "blocked",
    "blocked-for-manuscript-use",
    "diagnostic",
    "future-work only",
    "not benchmark evidence",
}
LOCAL_ONLY_PREFIXES = ("output", "results", ".git", ".venv")


class ArtifactState(StrEnum):
    """Freshness state for manuscript/release artifact reuse."""

    CURRENT = "current"
    HISTORICAL_VALID = "historical-valid"
    STALE_NEEDS_REFRESH = "stale-needs-refresh"
    BLOCKED_FOR_MANUSCRIPT_USE = "blocked-for-manuscript-use"


@dataclass(frozen=True, slots=True)
class ArtifactResult:
    """Classification result for one manifest entry."""

    artifact_id: str
    state: ArtifactState
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checksum_expected: str | None = None
    checksum_actual: str | None = None
    schema_version: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactFreshnessReport:
    """Machine-readable artifact freshness report."""

    generated_at_utc: str
    manifest_path: str
    results: list[ArtifactResult]
    schema: str = SCHEMA_VERSION

    @property
    def exit_code(self) -> int:
        """Return non-zero when any artifact is not current or historical-valid."""
        return int(
            any(
                result.state
                in {
                    ArtifactState.STALE_NEEDS_REFRESH,
                    ArtifactState.BLOCKED_FOR_MANUSCRIPT_USE,
                }
                for result in self.results
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "schema": self.schema,
            "generated_at_utc": self.generated_at_utc,
            "manifest_path": self.manifest_path,
            "summary": _summarize(self.results),
            "results": [
                {
                    "artifact_id": result.artifact_id,
                    "state": result.state.value,
                    "reasons": result.reasons,
                    "warnings": result.warnings,
                    "checksum_expected": result.checksum_expected,
                    "checksum_actual": result.checksum_actual,
                    "schema_version": result.schema_version,
                }
                for result in self.results
            ],
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(results: list[ArtifactResult]) -> dict[str, int]:
    """Count results by state."""
    counts = {state.value: 0 for state in ArtifactState}
    for result in results:
        counts[result.state.value] += 1
    return counts


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _artifact_id(entry: dict[str, Any]) -> str:
    return _as_text(entry.get("artifact_id") or entry.get("id") or entry.get("name")) or "unnamed"


class OutputSpec(NamedTuple):
    """Normalized artifact output metadata."""

    name: str
    path: str | None
    expected_checksum: str | None


def _output_specs(entry: dict[str, Any]) -> list[OutputSpec]:
    outputs = entry.get("outputs")
    if isinstance(outputs, dict):
        specs: list[OutputSpec] = []
        for name, value in outputs.items():
            if not isinstance(value, dict):
                specs.append(OutputSpec(str(name), None, None))
                continue
            specs.append(
                OutputSpec(
                    str(name),
                    _as_text(value.get("path")),
                    _as_text(
                        value.get("checksum")
                        or value.get("sha256")
                        or entry.get("checksum")
                        or entry.get("sha256")
                    ),
                )
            )
        if specs:
            return specs

    return [
        OutputSpec(
            "artifact",
            _as_text(entry.get("path")),
            _as_text(entry.get("checksum") or entry.get("sha256")),
        )
    ]


def _is_local_only(path_text: str, *, manifest_dir: Path | None = None) -> bool:
    path = Path(path_text)
    if path.is_absolute():
        return path.parts[:2] in {("/", "tmp"), ("/", "var")} or path.resolve(strict=False).parts[
            :2
        ] in {("/", "tmp"), ("/", "var")}

    raw_parts = tuple(part for part in path.parts if part not in {"", "."})
    if raw_parts and raw_parts[0] in LOCAL_ONLY_PREFIXES:
        return True

    normalized_parts = tuple(
        part for part in Path(posixpath.normpath(path_text)).parts if part not in {"", "."}
    )
    if normalized_parts and normalized_parts[0] in LOCAL_ONLY_PREFIXES:
        return True

    if manifest_dir is None:
        return False

    candidate = _resolve_output_path(path_text, manifest_dir=manifest_dir)
    try:
        resolved_candidate = candidate.resolve(strict=True)
        resolved_manifest = manifest_dir.resolve(strict=True)
    except OSError:
        return False
    for prefix in LOCAL_ONLY_PREFIXES:
        try:
            resolved_candidate.relative_to((resolved_manifest / prefix).resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _evaluate_outputs(
    specs: list[OutputSpec],
    *,
    manifest_dir: Path | None,
) -> tuple[list[str], list[str], list[str], list[str], bool, bool]:
    reasons: list[str] = []
    result_warnings: list[str] = []
    expected_values: list[str] = []
    actual_values: list[str] = []
    has_failure = False
    has_local_only = False

    for spec in specs:
        output_label = f"output {spec.name}"
        if not spec.path:
            reasons.append(f"{output_label} missing output path")
            has_failure = True
            continue
        if _is_local_only(spec.path, manifest_dir=manifest_dir):
            result_warnings.append(
                f"{output_label} path is local-only and not durable: {spec.path}"
            )
            has_local_only = True

        if not spec.expected_checksum:
            reasons.append(f"{output_label} missing checksum")
            has_failure = True
            continue

        expected_values.append(spec.expected_checksum)
        candidate = _resolve_output_path(spec.path, manifest_dir=manifest_dir)
        if candidate.is_file():
            actual = sha256_file(candidate)
            actual_values.append(actual)
            if actual == spec.expected_checksum:
                reasons.append(f"{output_label} checksum matches")
            else:
                reasons.append(f"{output_label} checksum mismatch")
                has_failure = True
        else:
            reasons.append(f"{output_label} referenced artifact file is missing")
            has_failure = True

    return reasons, result_warnings, expected_values, actual_values, has_failure, has_local_only


def _resolve_output_path(path_text: str, *, manifest_dir: Path | None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if manifest_dir is None:
        return path
    return manifest_dir / path


def _explicit_blocker(entry: dict[str, Any]) -> str | None:
    for marker_field in ("readiness", "review_state", "status", "result_classification"):
        marker = _as_text(entry.get(marker_field))
        if marker and marker.lower() in BLOCKING_MARKERS:
            return f"{marker_field}={marker}"
    blocker = _as_text(entry.get("blocked_reason") or entry.get("provenance_blocker"))
    if blocker:
        return f"blocked_reason={blocker}"
    return None


def classify_artifact(
    entry: dict[str, Any],
    *,
    manifest_dir: Path | None = None,
    current_schema: str = CURRENT_ARTIFACT_SCHEMA,
) -> ArtifactResult:
    """Classify one artifact manifest entry without raising on malformed data."""
    artifact_id = _artifact_id(entry)
    schema_version = _as_text(entry.get("schema_version"))

    blocker = _explicit_blocker(entry)
    if blocker:
        return ArtifactResult(
            artifact_id=artifact_id,
            state=ArtifactState.BLOCKED_FOR_MANUSCRIPT_USE,
            reasons=[f"explicit manuscript blocker: {blocker}"],
            schema_version=schema_version,
        )

    reasons, result_warnings, expected_values, actual_values, has_failure, has_local_only = (
        _evaluate_outputs(_output_specs(entry), manifest_dir=manifest_dir)
    )

    if has_local_only:
        return ArtifactResult(
            artifact_id=artifact_id,
            state=ArtifactState.STALE_NEEDS_REFRESH,
            reasons=[*reasons, "artifact is only represented by disposable local output"],
            warnings=result_warnings,
            checksum_expected=", ".join(expected_values) or None,
            checksum_actual=", ".join(actual_values) or None,
            schema_version=schema_version,
        )

    if not has_failure and expected_values and len(actual_values) == len(expected_values):
        if schema_version == current_schema:
            state = ArtifactState.CURRENT
        else:
            state = ArtifactState.HISTORICAL_VALID
            reasons.append(f"schema is not current: {schema_version or 'missing'}")
    else:
        state = ArtifactState.STALE_NEEDS_REFRESH

    return ArtifactResult(
        artifact_id=artifact_id,
        state=state,
        reasons=reasons,
        warnings=result_warnings,
        checksum_expected=", ".join(expected_values) or None,
        checksum_actual=", ".join(actual_values) or None,
        schema_version=schema_version,
    )


def load_manifest(path: Path) -> list[Any]:
    """Load JSON or YAML manifest entries."""
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML manifests")
        payload = yaml.safe_load(text)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"manifest {path} is not a mapping or list")


def scan_manifest(
    manifest_path: Path,
    *,
    current_schema: str = CURRENT_ARTIFACT_SCHEMA,
) -> ArtifactFreshnessReport:
    """Build a freshness report for every entry in ``manifest_path``."""
    results: list[ArtifactResult] = []
    try:
        entries = load_manifest(manifest_path)
    except Exception as exc:
        entries = [
            {
                "artifact_id": manifest_path.name,
                "scan_error": f"cannot load manifest: {exc}",
            }
        ]

    for entry in entries:
        if not isinstance(entry, dict):
            results.append(
                ArtifactResult(
                    artifact_id="unnamed",
                    state=ArtifactState.STALE_NEEDS_REFRESH,
                    reasons=["manifest entry is not a mapping"],
                )
            )
            continue
        scan_error = _as_text(entry.get("scan_error"))
        if scan_error:
            results.append(
                ArtifactResult(
                    artifact_id=_artifact_id(entry),
                    state=ArtifactState.STALE_NEEDS_REFRESH,
                    reasons=[scan_error],
                )
            )
            continue
        results.append(
            classify_artifact(
                entry, manifest_dir=manifest_path.parent, current_schema=current_schema
            )
        )

    return ArtifactFreshnessReport(
        generated_at_utc=datetime.now(UTC).isoformat(),
        manifest_path=str(manifest_path),
        results=results,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify dissertation/release artifacts for manuscript freshness."
    )
    parser.add_argument("manifest", type=Path, help="Artifact manifest JSON/YAML file.")
    parser.add_argument("--current-schema", default=CURRENT_ARTIFACT_SCHEMA)
    parser.add_argument("--json-out", type=Path, help="Optional JSON report output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    report = scan_manifest(args.manifest, current_schema=args.current_schema)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
