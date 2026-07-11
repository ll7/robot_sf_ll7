"""Versioned artifact catalog contract for reusable benchmark figures and tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

ARTIFACT_CATALOG_SCHEMA_VERSION = "artifact_catalog.v1"
ARTIFACT_CATALOG_SCHEMA_FILE = Path(__file__).with_name("schemas") / "artifact_catalog.v1.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_ONLY_PREFIXES = (
    "output/",
    "results/",
    ".git/",
    ".venv/",
    "/tmp/",
    "/var/tmp/",
    "/home/",
)


@dataclass(frozen=True, slots=True)
class ArtifactCatalogIssue:
    """One artifact catalog validation issue."""

    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ArtifactFileRef:
    """Tracked file reference with checksum provenance."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactCatalogEntry:
    """One reusable figure or table artifact entry."""

    artifact_id: str
    artifact_kind: str
    source_kind: str
    source_files: list[ArtifactFileRef]
    outputs: dict[str, ArtifactFileRef]
    generation_command: str
    generation_commit: str
    claim_boundary: str
    caption_file: ArtifactFileRef | None = None


@dataclass(frozen=True, slots=True)
class ArtifactCatalog:
    """Typed ``artifact_catalog.v1`` payload."""

    schema_version: str
    catalog_id: str
    artifacts: list[ArtifactCatalogEntry]

    def to_dict(self) -> dict[str, Any]:
        """Convert the catalog to JSON-safe primitives.

        Returns:
            Dictionary representation of the catalog.
        """

        return asdict(self)


class ArtifactCatalogValidationError(RobotSfError, ValueError):
    """Raised when an artifact catalog fails validation."""

    def __init__(self, issues: list[ArtifactCatalogIssue], *, source: str | Path | None = None):
        """Build an actionable validation error from catalog issues."""

        self.issues = tuple(issues)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(f"{issue.path}: {issue.message}" for issue in issues))


def load_artifact_catalog_schema() -> dict[str, Any]:
    """Load the public ``artifact_catalog.v1`` JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(ARTIFACT_CATALOG_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_artifact_catalog(path: Path) -> ArtifactCatalog:
    """Load and validate a YAML or JSON artifact catalog.

    Returns:
        Typed artifact catalog metadata.
    """

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ArtifactCatalogValidationError(
            [ArtifactCatalogIssue("/", "expected a mapping payload")],
            source=path,
        )
    return artifact_catalog_from_dict(payload, catalog_path=path)


def artifact_catalog_from_dict(
    payload: Mapping[str, Any],
    *,
    catalog_path: Path,
) -> ArtifactCatalog:
    """Validate and convert a catalog mapping into typed metadata.

    Returns:
        Typed artifact catalog metadata.
    """

    issues = validate_artifact_catalog_payload(payload, catalog_path=catalog_path)
    if issues:
        raise ArtifactCatalogValidationError(issues, source=catalog_path)
    return _catalog_from_payload(payload)


def validate_artifact_catalog(path: Path) -> list[ArtifactCatalogIssue]:
    """Validate an artifact catalog path and return all issues.

    Returns:
        List of validation issues. Empty means valid.
    """

    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:  # pragma: no cover - defensive CLI path
        return [ArtifactCatalogIssue("/", f"failed to load catalog: {exc}")]
    if not isinstance(payload, Mapping):
        return [ArtifactCatalogIssue("/", "expected a mapping payload")]
    return validate_artifact_catalog_payload(payload, catalog_path=path)


def validate_artifact_catalog_payload(
    payload: Mapping[str, Any],
    *,
    catalog_path: Path,
) -> list[ArtifactCatalogIssue]:
    """Validate schema, identity, path, checksum, and durability rules.

    Returns:
        List of validation issues. Empty means valid.
    """

    issues = _schema_validation_issues(payload)
    issues.extend(_semantic_validation_issues(payload, catalog_path=catalog_path))
    return issues


def _schema_validation_issues(payload: Mapping[str, Any]) -> list[ArtifactCatalogIssue]:
    """Return JSON Schema validation issues."""

    validator = Draft202012Validator(load_artifact_catalog_schema())
    return [
        ArtifactCatalogIssue(json_pointer(error.absolute_path), error.message)
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_validation_issues(
    payload: Mapping[str, Any],
    *,
    catalog_path: Path,
) -> list[ArtifactCatalogIssue]:
    """Return cross-field and filesystem validation issues."""

    issues: list[ArtifactCatalogIssue] = []
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return issues

    seen_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        prefix = f"/artifacts/{index}"
        artifact_id = artifact.get("artifact_id")
        if isinstance(artifact_id, str):
            if artifact_id in seen_ids:
                issues.append(
                    ArtifactCatalogIssue(
                        f"{prefix}/artifact_id",
                        f"duplicate artifact_id '{artifact_id}'",
                    )
                )
            seen_ids.add(artifact_id)

        for file_index, file_ref in enumerate(_as_list(artifact.get("source_files"))):
            issues.extend(
                _validate_file_ref(
                    file_ref,
                    catalog_path=catalog_path,
                    pointer=f"{prefix}/source_files/{file_index}",
                )
            )
        outputs = artifact.get("outputs")
        if isinstance(outputs, Mapping):
            for output_key, file_ref in outputs.items():
                issues.extend(
                    _validate_file_ref(
                        file_ref,
                        catalog_path=catalog_path,
                        pointer=f"{prefix}/outputs/{output_key}",
                    )
                )
        caption_file = artifact.get("caption_file")
        if caption_file is not None:
            issues.extend(
                _validate_file_ref(
                    caption_file,
                    catalog_path=catalog_path,
                    pointer=f"{prefix}/caption_file",
                )
            )
    return issues


def _validate_file_ref(
    file_ref: Any,
    *,
    catalog_path: Path,
    pointer: str,
) -> list[ArtifactCatalogIssue]:
    """Validate one path/checksum pair.

    Returns:
        List of validation issues for the file reference.
    """

    issues: list[ArtifactCatalogIssue] = []
    if not isinstance(file_ref, Mapping):
        return issues
    raw_path = file_ref.get("path")
    raw_sha = file_ref.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return issues
    path_text = raw_path.strip()
    if _is_local_only_path(path_text):
        issues.append(
            ArtifactCatalogIssue(
                f"{pointer}/path",
                f"local-only artifact reference is not durable: {path_text}",
            )
        )
        return issues
    if Path(path_text).is_absolute() or ".." in Path(path_text).parts:
        issues.append(
            ArtifactCatalogIssue(
                f"{pointer}/path",
                "path must be repository-relative or catalog-relative without '..'",
            )
        )
        return issues

    resolved = _resolve_catalog_path(catalog_path, path_text)
    if not resolved.exists():
        issues.append(ArtifactCatalogIssue(f"{pointer}/path", f"path does not exist: {path_text}"))
        return issues
    if not resolved.is_file():
        issues.append(ArtifactCatalogIssue(f"{pointer}/path", f"path is not a file: {path_text}"))
        return issues

    if not isinstance(raw_sha, str) or _SHA256_RE.fullmatch(raw_sha.strip()) is None:
        issues.append(ArtifactCatalogIssue(f"{pointer}/sha256", "must be a 64-character SHA-256"))
        return issues
    actual_sha = sha256_file(resolved)
    if actual_sha != raw_sha.strip():
        issues.append(
            ArtifactCatalogIssue(
                f"{pointer}/sha256",
                f"checksum mismatch for {path_text}: expected {raw_sha.strip()}, got {actual_sha}",
            )
        )
    return issues


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_catalog_path(catalog_path: Path, path_text: str) -> Path:
    """Resolve a catalog file reference relative to catalog dir or repository root.

    Returns:
        Absolute resolved path.
    """

    catalog_relative = (catalog_path.parent / path_text).resolve()
    if catalog_relative.exists():
        return catalog_relative
    return (_repo_root_for(catalog_path) / path_text).resolve()


def _repo_root_for(path: Path) -> Path:
    """Return the nearest repository root, falling back to the catalog directory.

    Returns:
        Repository root or catalog parent when no Git root marker exists.
    """

    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return parent
    return resolved.parent


def _is_local_only_path(value: str) -> bool:
    """Return whether a path points at disposable local state."""

    path = Path(value.strip())
    parts = path.parts
    local_roots = {prefix.strip("/") for prefix in _LOCAL_ONLY_PREFIXES}
    local_roots.discard("")
    if path.is_absolute():
        return len(parts) > 1 and parts[1] in local_roots
    return bool(parts) and (parts[0] in local_roots or any(".worktrees" in part for part in parts))


def _as_list(value: Any) -> list[Any]:
    """Return value when it is a list, otherwise an empty list."""

    return value if isinstance(value, list) else []


def _catalog_from_payload(payload: Mapping[str, Any]) -> ArtifactCatalog:
    """Build typed catalog metadata from a validated payload.

    Returns:
        Typed artifact catalog metadata.
    """

    return ArtifactCatalog(
        schema_version=str(payload["schema_version"]),
        catalog_id=str(payload["catalog_id"]),
        artifacts=[
            ArtifactCatalogEntry(
                artifact_id=str(artifact["artifact_id"]),
                artifact_kind=str(artifact["artifact_kind"]),
                source_kind=str(artifact["source_kind"]),
                source_files=[
                    ArtifactFileRef(path=str(item["path"]), sha256=str(item["sha256"]))
                    for item in artifact["source_files"]
                ],
                outputs={
                    str(key): ArtifactFileRef(
                        path=str(file_ref["path"]),
                        sha256=str(file_ref["sha256"]),
                    )
                    for key, file_ref in artifact["outputs"].items()
                },
                generation_command=str(artifact["generation_command"]),
                generation_commit=str(artifact["generation_commit"]),
                claim_boundary=str(artifact["claim_boundary"]),
                caption_file=(
                    ArtifactFileRef(
                        path=str(artifact["caption_file"]["path"]),
                        sha256=str(artifact["caption_file"]["sha256"]),
                    )
                    if artifact.get("caption_file") is not None
                    else None
                ),
            )
            for artifact in payload["artifacts"]
        ],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the catalog validator parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Validate an artifact_catalog.v1 file.")
    parser.add_argument("catalog", type=Path, help="Artifact catalog YAML/JSON path.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON validation report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate one artifact catalog and return a shell-friendly exit code.

    Returns:
        ``0`` when valid, otherwise ``2``.
    """

    args = build_arg_parser().parse_args(argv)
    issues = validate_artifact_catalog(args.catalog)
    if args.json:
        sys.stdout.write(
            json.dumps(
                {
                    "schema": "artifact_catalog_validation.v1",
                    "catalog": str(args.catalog),
                    "ok": not issues,
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    elif issues:
        for issue in issues:
            sys.stdout.write(f"{issue.path}: {issue.message}\n")
    else:
        sys.stdout.write(f"artifact catalog valid: {args.catalog}\n")
    return 0 if not issues else 2


__all__ = [
    "ARTIFACT_CATALOG_SCHEMA_VERSION",
    "ArtifactCatalog",
    "ArtifactCatalogEntry",
    "ArtifactCatalogIssue",
    "ArtifactCatalogValidationError",
    "ArtifactFileRef",
    "artifact_catalog_from_dict",
    "load_artifact_catalog",
    "load_artifact_catalog_schema",
    "sha256_file",
    "validate_artifact_catalog",
    "validate_artifact_catalog_payload",
]
