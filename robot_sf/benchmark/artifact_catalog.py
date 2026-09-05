"""Versioned artifact catalog contract for reusable benchmark figures and tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

ARTIFACT_CATALOG_SCHEMA_V1 = "artifact_catalog.v1"
ARTIFACT_CATALOG_SCHEMA_V2 = "artifact_catalog.v2"
# Keep the historical constant as the default for callers that create v1
# catalogs. New catalogs opt in to v2 explicitly.
ARTIFACT_CATALOG_SCHEMA_VERSION = ARTIFACT_CATALOG_SCHEMA_V1
ARTIFACT_CATALOG_SCHEMA_FILE = Path(__file__).with_name("schemas") / "artifact_catalog.v1.json"
_ARTIFACT_CATALOG_SCHEMA_FILES = {
    ARTIFACT_CATALOG_SCHEMA_V1: Path(__file__).with_name("schemas") / "artifact_catalog.v1.json",
    ARTIFACT_CATALOG_SCHEMA_V2: Path(__file__).with_name("schemas") / "artifact_catalog.v2.json",
}
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
class FigureSemantics:
    """Declared publication semantics for an ``artifact_catalog.v2`` figure."""

    metric_id: str
    unit: str
    desirability: str
    support: int
    denominator: int
    comparison: bool
    uncertainty_declared: bool
    uncertainty_method: str | None
    tie_policy: str
    legend_series: list[str]
    legend_complete: bool
    accessibility_palette_contract: str | None = None


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
    figure_semantics: FigureSemantics | None = None


@dataclass(frozen=True, slots=True)
class ArtifactCatalog:
    """Typed ``artifact_catalog.v1`` payload."""

    schema_version: str
    catalog_id: str
    artifacts: list[ArtifactCatalogEntry]
    claim_identity: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the catalog to JSON-safe primitives.

        Returns:
            Dictionary representation of the catalog.
        """

        payload = asdict(self)
        if self.claim_identity is None:
            payload.pop("claim_identity", None)
        return payload


class ArtifactCatalogValidationError(RobotSfError, ValueError):
    """Raised when an artifact catalog fails validation."""

    def __init__(self, issues: list[ArtifactCatalogIssue], *, source: str | Path | None = None):
        """Build an actionable validation error from catalog issues."""

        self.issues = tuple(issues)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(f"{issue.path}: {issue.message}" for issue in issues))


def load_artifact_catalog_schema(
    schema_version: str = ARTIFACT_CATALOG_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Load the public artifact catalog JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    try:
        schema_file = _ARTIFACT_CATALOG_SCHEMA_FILES[schema_version]
    except KeyError as exc:
        raise ValueError(f"unsupported artifact catalog schema: {schema_version}") from exc
    return json.loads(schema_file.read_text(encoding="utf-8"))


def load_artifact_catalog(
    path: Path,
    *,
    repository_root: Path | None = None,
    approved_durable_roots: Iterable[Path] = (),
) -> ArtifactCatalog:
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
    return artifact_catalog_from_dict(
        payload,
        catalog_path=path,
        repository_root=repository_root,
        approved_durable_roots=approved_durable_roots,
    )


def artifact_catalog_from_dict(
    payload: Mapping[str, Any],
    *,
    catalog_path: Path,
    repository_root: Path | None = None,
    approved_durable_roots: Iterable[Path] = (),
) -> ArtifactCatalog:
    """Validate and convert a catalog mapping into typed metadata.

    Returns:
        Typed artifact catalog metadata.
    """

    issues = validate_artifact_catalog_payload(
        payload,
        catalog_path=catalog_path,
        repository_root=repository_root,
        approved_durable_roots=approved_durable_roots,
    )
    if issues:
        raise ArtifactCatalogValidationError(issues, source=catalog_path)
    return _catalog_from_payload(payload)


def validate_artifact_catalog(
    path: Path,
    *,
    repository_root: Path | None = None,
    approved_durable_roots: Iterable[Path] = (),
) -> list[ArtifactCatalogIssue]:
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
    return validate_artifact_catalog_payload(
        payload,
        catalog_path=path,
        repository_root=repository_root,
        approved_durable_roots=approved_durable_roots,
    )


def validate_artifact_catalog_payload(
    payload: Mapping[str, Any],
    *,
    catalog_path: Path,
    repository_root: Path | None = None,
    approved_durable_roots: Iterable[Path] = (),
) -> list[ArtifactCatalogIssue]:
    """Validate schema, identity, path, checksum, and durability rules.

    Returns:
        List of validation issues. Empty means valid.
    """

    approved_roots = tuple(approved_durable_roots)
    issues = _schema_validation_issues(payload)
    try:
        catalog_path_resolved = catalog_path.resolve()
        containment_roots = _containment_roots(
            catalog_path,
            repository_root=repository_root,
            approved_durable_roots=approved_roots,
        )
    except (OSError, RuntimeError) as exc:
        issues.append(
            ArtifactCatalogIssue(
                "/",
                f"could not resolve catalog path safely: {exc}",
            )
        )
    else:
        if not any(_is_within_root(catalog_path_resolved, root) for root in containment_roots):
            issues.append(
                ArtifactCatalogIssue(
                    "/",
                    "resolved catalog path escapes the repository and approved durable roots",
                )
            )
    issues.extend(
        _semantic_validation_issues(
            payload,
            catalog_path=catalog_path,
            repository_root=repository_root,
            approved_durable_roots=approved_roots,
        )
    )
    return issues


def _schema_validation_issues(payload: Mapping[str, Any]) -> list[ArtifactCatalogIssue]:
    """Return JSON Schema validation issues."""

    schema_version = payload.get("schema_version", ARTIFACT_CATALOG_SCHEMA_VERSION)
    schema = load_artifact_catalog_schema(str(schema_version))
    validator = Draft202012Validator(schema)
    return [
        ArtifactCatalogIssue(json_pointer(error.absolute_path), error.message)
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_validation_issues(
    payload: Mapping[str, Any],
    *,
    catalog_path: Path,
    repository_root: Path | None,
    approved_durable_roots: Iterable[Path],
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
                    repository_root=repository_root,
                    approved_durable_roots=approved_durable_roots,
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
                        repository_root=repository_root,
                        approved_durable_roots=approved_durable_roots,
                        pointer=f"{prefix}/outputs/{output_key}",
                    )
                )
        caption_file = artifact.get("caption_file")
        if caption_file is not None:
            issues.extend(
                _validate_file_ref(
                    caption_file,
                    catalog_path=catalog_path,
                    repository_root=repository_root,
                    approved_durable_roots=approved_durable_roots,
                    pointer=f"{prefix}/caption_file",
                )
            )
    return issues


def _validate_file_ref(  # noqa: C901
    file_ref: Any,
    *,
    catalog_path: Path,
    repository_root: Path | None,
    approved_durable_roots: Iterable[Path],
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

    try:
        resolved = _resolve_catalog_path(
            catalog_path,
            path_text,
            repository_root=repository_root,
        )
        containment_roots = _containment_roots(
            catalog_path,
            repository_root=repository_root,
            approved_durable_roots=approved_durable_roots,
        )
    except (OSError, RuntimeError) as exc:
        issues.append(
            ArtifactCatalogIssue(
                f"{pointer}/path",
                f"could not resolve artifact reference safely: {exc}",
            )
        )
        return issues
    if not any(_is_within_root(resolved, root) for root in containment_roots):
        issues.append(
            ArtifactCatalogIssue(
                f"{pointer}/path",
                "resolved artifact reference escapes the repository and approved durable roots",
            )
        )
        return issues
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


def _resolve_catalog_path(
    catalog_path: Path,
    path_text: str,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Resolve a catalog file reference relative to catalog dir or repository root.

    Returns:
        Absolute resolved path.
    """

    catalog_candidate = catalog_path.parent / path_text
    catalog_relative = catalog_candidate.resolve()
    if catalog_candidate.exists() or catalog_candidate.is_symlink():
        return catalog_relative
    fallback_root = (
        repository_root.resolve() if repository_root is not None else _repo_root_for(catalog_path)
    )
    return (fallback_root / path_text).resolve()


def _containment_roots(
    catalog_path: Path,
    *,
    repository_root: Path | None,
    approved_durable_roots: Iterable[Path],
) -> tuple[Path, ...]:
    """Return resolved roots allowed to contain catalog file references."""
    default_root = (
        repository_root.resolve() if repository_root is not None else _repo_root_for(catalog_path)
    )
    roots = [default_root]
    roots.extend(Path(root).resolve() for root in approved_durable_roots)
    return tuple(dict.fromkeys(roots))


def _is_within_root(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _repo_root_for(path: Path) -> Path:
    """Return the nearest repository root, falling back to the catalog directory.

    Returns:
        Repository root or catalog parent when no Git root marker exists.
    """

    absolute = path.absolute()
    for parent in (absolute.parent, *absolute.parents):
        if (parent / ".git").exists():
            return parent.resolve()
    return absolute.parent.resolve()


def _is_local_only_path(value: str) -> bool:
    """Return whether a path points at disposable local state."""

    path = Path(value.strip())
    parts = path.parts
    if path.is_absolute():
        for prefix in _LOCAL_ONLY_PREFIXES:
            prefix_path = Path(prefix)
            if prefix_path.is_absolute() and (path == prefix_path or prefix_path in path.parents):
                return True
        return False
    local_roots = {prefix.strip("/") for prefix in _LOCAL_ONLY_PREFIXES}
    local_roots.discard("")
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
                figure_semantics=(
                    FigureSemantics(
                        metric_id=str(artifact["figure_semantics"]["metric_id"]),
                        unit=str(artifact["figure_semantics"]["unit"]),
                        desirability=str(artifact["figure_semantics"]["desirability"]),
                        support=int(artifact["figure_semantics"]["support"]),
                        denominator=int(artifact["figure_semantics"]["denominator"]),
                        comparison=bool(artifact["figure_semantics"]["comparison"]),
                        uncertainty_declared=bool(
                            artifact["figure_semantics"]["uncertainty_declared"]
                        ),
                        uncertainty_method=(
                            str(artifact["figure_semantics"]["uncertainty_method"])
                            if artifact["figure_semantics"].get("uncertainty_method") is not None
                            else None
                        ),
                        tie_policy=str(artifact["figure_semantics"]["tie_policy"]),
                        legend_series=[
                            str(item) for item in artifact["figure_semantics"]["legend_series"]
                        ],
                        legend_complete=bool(artifact["figure_semantics"]["legend_complete"]),
                        accessibility_palette_contract=(
                            str(artifact["figure_semantics"]["accessibility_palette_contract"])
                            if artifact["figure_semantics"].get("accessibility_palette_contract")
                            is not None
                            else None
                        ),
                    )
                    if artifact.get("figure_semantics") is not None
                    else None
                ),
            )
            for artifact in payload["artifacts"]
        ],
        claim_identity=(
            {str(key): str(value) for key, value in payload["claim_identity"].items()}
            if isinstance(payload.get("claim_identity"), Mapping)
            else None
        ),
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
    "ARTIFACT_CATALOG_SCHEMA_V1",
    "ARTIFACT_CATALOG_SCHEMA_V2",
    "ARTIFACT_CATALOG_SCHEMA_VERSION",
    "ArtifactCatalog",
    "ArtifactCatalogEntry",
    "ArtifactCatalogIssue",
    "ArtifactCatalogValidationError",
    "ArtifactFileRef",
    "FigureSemantics",
    "artifact_catalog_from_dict",
    "load_artifact_catalog",
    "load_artifact_catalog_schema",
    "sha256_file",
    "validate_artifact_catalog",
    "validate_artifact_catalog_payload",
]
