"""Fail-closed prerequisite checks for checkpoint-dependent examples.

Plain-language summary: ``--check`` answers "can this example run here?" before
any simulation starts. It resolves the example through the canonical manifest,
verifies the prerequisite files, consults the model registry for pinned
checksums and acquisition pointers, and reports one stable status
(``ready``/``missing_model``/``missing_map``/``missing_extra``/
``invalid_checksum``/``unsupported_legacy_reference``/...).

The check path is deliberately import-light: it imports only the standard
library, ``yaml`` (through the manifest loader), and this module. It never
imports or instantiates a model, environment, renderer, or training framework
and it never performs network access. The model registry YAML is read directly
instead of importing :mod:`robot_sf.models.registry` so an optional
experiment-tracking import cannot leak into check-only mode.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.examples.manifest_loader import ExampleManifest, load_manifest

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Sequence

__all__ = [
    "CHECK_SCHEMA",
    "STATUS_INVALID_CHECKSUM",
    "STATUS_MISSING_EXTRA",
    "STATUS_MISSING_FILE",
    "STATUS_MISSING_MAP",
    "STATUS_MISSING_MODEL",
    "STATUS_NOT_VERIFIABLE",
    "STATUS_OPERATOR_INPUT_REQUIRED",
    "STATUS_READY",
    "STATUS_UNSUPPORTED_LEGACY_REFERENCE",
    "PrerequisiteCheck",
    "PrerequisiteReport",
    "add_prerequisite_check_arguments",
    "check_example_prerequisites",
    "check_script_prerequisites",
    "format_report_text",
    "report_to_dict",
    "run_prerequisite_check",
]

#: Stable JSON schema identifier for check-only output.
CHECK_SCHEMA = "example_prerequisites.v1"

#: Stable per-prerequisite and aggregate statuses.
STATUS_READY = "ready"
STATUS_MISSING_MODEL = "missing_model"
STATUS_MISSING_MAP = "missing_map"
STATUS_MISSING_FILE = "missing_file"
STATUS_MISSING_EXTRA = "missing_extra"
STATUS_INVALID_CHECKSUM = "invalid_checksum"
STATUS_UNSUPPORTED_LEGACY_REFERENCE = "unsupported_legacy_reference"
STATUS_OPERATOR_INPUT_REQUIRED = "operator_input_required"
STATUS_NOT_VERIFIABLE = "not_verifiable"

_ORDERED_STATUSES = (
    STATUS_READY,
    STATUS_MISSING_MODEL,
    STATUS_MISSING_MAP,
    STATUS_MISSING_FILE,
    STATUS_MISSING_EXTRA,
    STATUS_INVALID_CHECKSUM,
    STATUS_UNSUPPORTED_LEGACY_REFERENCE,
    STATUS_OPERATOR_INPUT_REQUIRED,
    STATUS_NOT_VERIFIABLE,
)

#: Example model prerequisites that are now backed by registry artifacts.
#: ``legacy_reference`` marks the pre-registry in-tree paths whose binaries were
#: removed by the phase B cutover (#6268); examples still loading those paths
#: must be updated to resolve the registry model id instead.
_MODEL_REFERENCES: dict[str, tuple[str, bool]] = {
    "model/run_023": ("legacy_ppo_run_023", True),
    "model/run_023.zip": ("legacy_ppo_run_023", True),
    "model/run_043": ("legacy_ppo_run_043", True),
    "model/run_043.zip": ("legacy_ppo_run_043", True),
    "model/pedestrian/ppo_ped_02.zip": ("legacy_ppo_pedestrian_ped_02", False),
    "model/pedestrian/ppo_intersection.zip": ("legacy_ppo_pedestrian_intersection", False),
}

#: Import probes per recognized dependency-sync prerequisite. ``find_spec``
#: resolves availability without importing the framework.
_EXTRA_IMPORT_PROBES: dict[str, tuple[str, ...]] = {
    "uv sync --all-extras": ("stable_baselines3", "torch"),
}

#: File suffixes treated as model artifacts when no registry alias applies.
_MODEL_SUFFIXES = frozenset({".zip", ".pt", ".pth", ".onnx", ".pkl", ".model"})

_SYNC_PREFIXES = ("uv sync", "uv pip install", "pip install", "uv add")


def _sha256(path: Path) -> str:
    """Return the lowercase-hex SHA256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry_entries(registry_path: Path | None) -> dict[str, dict[str, Any]]:
    """Read registry entries indexed by ``model_id`` without importing the registry module.

    Args:
        registry_path: Path to ``model/registry.yaml`` or ``None`` when unavailable.

    Returns:
        A mapping of ``model_id`` to registry entry; empty when the registry is
        absent or malformed. Check-only mode never fails because the registry is
        unavailable, it only loses checksum and acquisition detail.
    """

    # The registry is optional provenance: absence degrades detail, never readiness.

    if registry_path is None or not registry_path.is_file():
        return {}
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    entries = data.get("models") if isinstance(data, Mapping) else None
    if not isinstance(entries, list):
        return {}
    registry: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model_id = entry.get("model_id")
        if isinstance(model_id, str) and model_id:
            registry[model_id] = dict(entry)
    return registry


def _registry_status(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build the registry sub-payload for one model prerequisite.

    Returns:
        The registry sub-payload mapping, or ``None`` when no entry matched.
    """

    if entry is None:
        return None
    release = entry.get("github_release")
    release = release if isinstance(release, Mapping) else {}
    local_path = entry.get("local_path")
    expected_sha = release.get("sha256")
    model_id = entry.get("model_id")
    acquisition = entry.get("acquisition")
    if not isinstance(acquisition, str) or not acquisition.strip():
        acquisition = f"robot-sf models download {model_id}" if isinstance(model_id, str) else None
    return {
        "model_id": model_id,
        "registered": True,
        "local_path": local_path if isinstance(local_path, str) else None,
        "expected_sha256": expected_sha if isinstance(expected_sha, str) else None,
        "release_url": release.get("url") if isinstance(release.get("url"), str) else None,
        "acquisition": acquisition,
    }


@dataclass(frozen=True, slots=True)
class PrerequisiteCheck:
    """Result for a single declared prerequisite."""

    prerequisite: str
    kind: str
    status: str
    detail: str
    path: str | None = None
    acquisition: str | None = None
    observed_sha256: str | None = None
    expected_sha256: str | None = None
    model_registry: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PrerequisiteReport:
    """Aggregate check-only result for one example."""

    example_id: str
    example_path: str
    status: str
    checks: tuple[PrerequisiteCheck, ...]

    @property
    def ready(self) -> bool:
        """Return whether every declared prerequisite is satisfied."""

        return self.status == STATUS_READY


def _overall_status(checks: Sequence[PrerequisiteCheck]) -> str:
    """Return the first non-ready status in stable declaration order.

    Returns:
        The first non-ready status, or ``ready`` when every check passed.
    """

    observed = {check.status for check in checks}
    for status in _ORDERED_STATUSES:
        if status != STATUS_READY and status in observed:
            return status
    return STATUS_READY


def _is_extra_prerequisite(prerequisite: str) -> bool:
    """Return whether a prerequisite declares a dependency-sync command.

    Returns:
        ``True`` when the prerequisite is a dependency-sync command.
    """

    lowered = prerequisite.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _SYNC_PREFIXES)


def _check_extra(prerequisite: str) -> PrerequisiteCheck:
    """Probe a dependency-sync prerequisite without importing frameworks.

    Returns:
        The check result for the dependency-sync prerequisite.
    """

    probes = _EXTRA_IMPORT_PROBES.get(prerequisite.strip().lower(), ())
    if not probes:
        return PrerequisiteCheck(
            prerequisite=prerequisite,
            kind="extra",
            status=STATUS_NOT_VERIFIABLE,
            detail=(
                "dependency-sync prerequisite is not recognized by the check-only "
                "probe table; run it manually before the example"
            ),
            acquisition=prerequisite,
        )
    missing = [name for name in probes if importlib.util.find_spec(name) is None]
    if missing:
        return PrerequisiteCheck(
            prerequisite=prerequisite,
            kind="extra",
            status=STATUS_MISSING_EXTRA,
            detail="missing optional dependency module(s): " + ", ".join(missing),
            acquisition=prerequisite,
        )
    return PrerequisiteCheck(
        prerequisite=prerequisite,
        kind="extra",
        status=STATUS_READY,
        detail="optional dependency module(s) importable: " + ", ".join(probes),
    )


def _classify_path_kind(prerequisite: str, *, model_alias: bool) -> str:
    """Classify a filesystem prerequisite as model, map, or generic file.

    Returns:
        The prerequisite kind (``model``, ``map``, or ``file``).
    """

    if model_alias:
        return "model"
    path = Path(prerequisite)
    if path.suffix.lower() == ".svg" or path.parts[:1] == ("maps",):
        return "map"
    if path.parts[:1] == ("model",) or path.suffix.lower() in _MODEL_SUFFIXES:
        return "model"
    return "file"


def _missing_status(kind: str) -> str:
    """Map a missing path kind to its stable status.

    Returns:
        The status for the missing path kind.
    """

    if kind == "map":
        return STATUS_MISSING_MAP
    if kind == "model":
        return STATUS_MISSING_MODEL
    return STATUS_MISSING_FILE


def _registry_by_local_path(
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Index registry entries by their normalized ``local_path``.

    Returns:
        A mapping of posix ``local_path`` to registry entry.
    """

    index: dict[str, Mapping[str, Any]] = {}
    for entry in registry.values():
        raw_path = entry.get("local_path")
        if isinstance(raw_path, str) and raw_path.strip():
            index[Path(raw_path).as_posix()] = entry
    return index


def _legacy_reference_check(
    prerequisite: str,
    *,
    kind: str,
    model_id: str | None,
    registry_status: Mapping[str, Any] | None,
    exists: bool,
) -> PrerequisiteCheck:
    """Build the check for a pre-registry in-tree model reference.

    Returns:
        The unsupported-legacy-reference check result.
    """

    acquisition = None
    if registry_status is not None:
        acquisition = (
            f"update the example to resolve_model_path('{model_id}') "
            f"(or run `{registry_status['acquisition']}`)"
        )
    return PrerequisiteCheck(
        prerequisite=prerequisite,
        kind=kind,
        status=STATUS_UNSUPPORTED_LEGACY_REFERENCE,
        detail=(
            "in-tree model binary was replaced by a registry-backed stub; "
            "the example still loads the legacy path"
        ),
        path=prerequisite if exists else None,
        acquisition=acquisition,
        expected_sha256=(registry_status.get("expected_sha256") if registry_status else None),
        model_registry=dict(registry_status) if registry_status is not None else None,
    )


def _missing_path_check(
    prerequisite: str,
    *,
    kind: str,
    registry_status: Mapping[str, Any] | None,
    expected_sha: str | None,
) -> PrerequisiteCheck:
    """Build the check for a missing filesystem prerequisite.

    Returns:
        The missing-path check result with registry acquisition guidance.
    """

    acquisition = None
    if registry_status is not None:
        local_path = registry_status.get("local_path")
        acquisition = (
            f"{registry_status['acquisition']} (hydrates {local_path})"
            if local_path
            else registry_status["acquisition"]
        )
    return PrerequisiteCheck(
        prerequisite=prerequisite,
        kind=kind,
        status=_missing_status(kind),
        detail="path not found on disk",
        acquisition=acquisition,
        expected_sha256=expected_sha,
        model_registry=dict(registry_status) if registry_status is not None else None,
    )


def _check_path_prerequisite(
    prerequisite: str,
    *,
    repo_root: Path,
    registry: Mapping[str, Mapping[str, Any]],
) -> PrerequisiteCheck:
    """Check one filesystem prerequisite with optional registry provenance.

    Returns:
        The check result for the filesystem prerequisite.
    """

    model_reference = _MODEL_REFERENCES.get(prerequisite)
    model_alias = model_reference is not None
    model_id = model_reference[0] if model_reference else None
    legacy_reference = bool(model_reference and model_reference[1])
    if model_id is not None:
        registry_entry = registry.get(model_id)
    else:
        registry_entry = _registry_by_local_path(registry).get(Path(prerequisite).as_posix())
    registry_status = _registry_status(registry_entry)
    kind = _classify_path_kind(prerequisite, model_alias=model_alias)

    if "<" in prerequisite and ">" in prerequisite:
        return PrerequisiteCheck(
            prerequisite=prerequisite,
            kind=kind,
            status=STATUS_OPERATOR_INPUT_REQUIRED,
            detail="prerequisite contains an operator-supplied placeholder path",
            acquisition=prerequisite,
        )

    if "*" in prerequisite or "?" in prerequisite:
        matches = sorted(repo_root.glob(prerequisite))
        if matches:
            first = matches[0].relative_to(repo_root).as_posix()
            return PrerequisiteCheck(
                prerequisite=prerequisite,
                kind=kind,
                status=STATUS_READY,
                detail=f"glob matches {len(matches)} path(s), first: {first}",
                path=first,
            )
        return PrerequisiteCheck(
            prerequisite=prerequisite,
            kind=kind,
            status=STATUS_MISSING_FILE,
            detail="glob pattern matches no files",
            acquisition=prerequisite,
        )

    path = repo_root / prerequisite
    exists = path.exists()

    if legacy_reference:
        return _legacy_reference_check(
            prerequisite,
            kind=kind,
            model_id=model_id,
            registry_status=registry_status,
            exists=exists,
        )

    expected_sha = registry_status.get("expected_sha256") if registry_status else None
    if not exists:
        return _missing_path_check(
            prerequisite,
            kind=kind,
            registry_status=registry_status,
            expected_sha=expected_sha,
        )

    if expected_sha and path.is_file():
        observed = _sha256(path)
        if observed == expected_sha:
            return PrerequisiteCheck(
                prerequisite=prerequisite,
                kind=kind,
                status=STATUS_READY,
                detail="path present and SHA256 matches the registry pin",
                path=prerequisite,
                observed_sha256=observed,
                expected_sha256=expected_sha,
                model_registry=registry_status,
            )
        return PrerequisiteCheck(
            prerequisite=prerequisite,
            kind=kind,
            status=STATUS_INVALID_CHECKSUM,
            detail="path present but SHA256 does not match the registry pin",
            path=prerequisite,
            acquisition=registry_status["acquisition"] if registry_status else None,
            observed_sha256=observed,
            expected_sha256=expected_sha,
            model_registry=registry_status,
        )

    return PrerequisiteCheck(
        prerequisite=prerequisite,
        kind=kind,
        status=STATUS_READY,
        detail="path present",
        path=prerequisite,
        expected_sha256=expected_sha,
        model_registry=registry_status,
    )


def _check_prerequisite(
    prerequisite: str,
    *,
    repo_root: Path,
    registry: Mapping[str, Mapping[str, Any]],
) -> PrerequisiteCheck:
    """Dispatch one prerequisite string to the matching check.

    Returns:
        The check result for the prerequisite string.
    """

    stripped = prerequisite.strip()
    if not stripped:
        return PrerequisiteCheck(
            prerequisite=prerequisite,
            kind="unknown",
            status=STATUS_NOT_VERIFIABLE,
            detail="empty prerequisite entry",
        )
    if _is_extra_prerequisite(stripped):
        return _check_extra(stripped)
    if stripped.startswith("(") or " subtree (" in stripped:
        return PrerequisiteCheck(
            prerequisite=stripped,
            kind="descriptive",
            status=STATUS_NOT_VERIFIABLE,
            detail="descriptive prerequisite cannot be verified by check-only mode",
        )
    return _check_path_prerequisite(stripped, repo_root=repo_root, registry=registry)


def check_example_prerequisites(
    manifest: ExampleManifest,
    query: str,
    *,
    repo_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> PrerequisiteReport:
    """Resolve one manifest example and check all of its declared prerequisites.

    Args:
        manifest: The loaded examples manifest.
        query: The example id/path/stem to resolve.
        repo_root: Repository root used to resolve relative prerequisite paths.
            Defaults to the manifest's parent directory.
        registry_path: Optional ``model/registry.yaml`` override.

    Returns:
        A :class:`PrerequisiteReport` with one stable status per prerequisite.

    Raises:
        ExampleIdentityError: If the id cannot be resolved (re-exported from
            :mod:`robot_sf.examples_cli`).
    """

    from robot_sf.examples_cli import example_id, find_example  # noqa: PLC0415

    example = find_example(manifest, query)
    root = Path(repo_root) if repo_root is not None else manifest.examples_root.parent
    registry_file = (
        Path(registry_path) if registry_path is not None else root / "model" / "registry.yaml"
    )
    registry = _load_registry_entries(registry_file)
    checks = tuple(
        _check_prerequisite(prerequisite, repo_root=root, registry=registry)
        for prerequisite in example.prerequisites
    )
    return PrerequisiteReport(
        example_id=example_id(example),
        example_path=example.path.as_posix(),
        status=_overall_status(checks),
        checks=checks,
    )


def _manifest_for_script(script_path: Path) -> ExampleManifest:
    """Return the manifest for a script located under the repository ``examples/`` tree."""

    examples_root = script_path.resolve().parent
    while examples_root.name != "examples" and examples_root.parent != examples_root:
        examples_root = examples_root.parent
    manifest_path = examples_root / "examples_manifest.yaml"
    return load_manifest(manifest_path, validate_paths=True)


def check_script_prerequisites(
    script_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> PrerequisiteReport:
    """Check the prerequisites declared for one example script by path.

    Args:
        script_path: Path to the example script (usually ``__file__``).
        repo_root: Optional repository root override.
        registry_path: Optional ``model/registry.yaml`` override.

    Returns:
        The prerequisite report for the script's manifest entry.
    """

    script = Path(script_path).resolve()
    manifest = _manifest_for_script(script)
    examples_root = manifest.examples_root
    relative = script.relative_to(examples_root.resolve()).as_posix()
    return check_example_prerequisites(
        manifest,
        relative,
        repo_root=repo_root,
        registry_path=registry_path,
    )


def report_to_dict(report: PrerequisiteReport) -> dict[str, Any]:
    """Return the stable JSON payload for one prerequisite report."""

    return {
        "schema": CHECK_SCHEMA,
        "example_id": report.example_id,
        "example_path": report.example_path,
        "status": report.status,
        "ready": report.ready,
        "checks": [
            {
                "prerequisite": check.prerequisite,
                "kind": check.kind,
                "status": check.status,
                "detail": check.detail,
                "path": check.path,
                "acquisition": check.acquisition,
                "observed_sha256": check.observed_sha256,
                "expected_sha256": check.expected_sha256,
                "model_registry": check.model_registry,
            }
            for check in report.checks
        ],
    }


def format_report_text(report: PrerequisiteReport) -> str:
    """Render a compact human-readable prerequisite report.

    Returns:
        The rendered multi-line report text.
    """

    lines = [f"{report.example_id}: {report.status}"]
    for check in report.checks:
        lines.append(f"  - {check.prerequisite}: {check.status} ({check.detail})")
        if check.acquisition:
            lines.append(f"    acquire: {check.acquisition}")
    return "\n".join(lines)


def add_prerequisite_check_arguments(parser: ArgumentParser) -> None:
    """Add ``--check``/``--format`` to an example script parser."""

    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate prerequisites without starting a simulation (no model imports).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for --check (default: text).",
    )


def run_prerequisite_check(
    script_path: str | Path,
    *,
    output_format: str = "text",
    stream: Any | None = None,
) -> int:
    """Run check-only mode for one script and print its report.

    Args:
        script_path: Path to the example script.
        output_format: ``text`` or ``json``.
        stream: Optional text stream override (defaults to ``sys.stdout``).

    Returns:
        ``0`` when every prerequisite is ready, ``1`` otherwise.
    """

    report = check_script_prerequisites(script_path)
    target = stream if stream is not None else sys.stdout
    if output_format == "json":
        target.write(json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n")
    else:
        target.write(format_report_text(report) + "\n")
    return 0 if report.ready else 1
