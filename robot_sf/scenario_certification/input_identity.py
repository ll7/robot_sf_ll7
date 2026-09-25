"""Content identity for scenario manifests and their runtime-referenced inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_sf.nav.map_config import DEFAULT_MAPS_FOLDER
from robot_sf.training.scenario_loader import (
    ScenarioValidationReport,
    load_scenarios_for_validation,
)

SCENARIO_INPUT_IDENTITY_SCHEMA = "scenario_runtime_input_identity.v1"


def scenario_input_identity(
    scenario_path: str | Path,
    *,
    scenario_id: str | None = None,
    validation_report: ScenarioValidationReport | None = None,
) -> dict[str, Any]:
    """Hash a scenario manifest, its includes, and referenced map/route files.

    Returns:
        JSON-safe identity with both the root manifest digest and a digest for the
        runtime-referenced input closure. An incomplete or unreadable closure has no
        effective digest, so consumers must retain the candidate as unknown.
    """
    loaded = _load_scenario_report(scenario_path, validation_report=validation_report)
    if isinstance(loaded, dict):
        return loaded
    root, root_digest, report = loaded
    if report.load_error is not None or report.entry_issues or report.load_issues:
        legacy_identity = _legacy_single_row_identity(
            root, root_digest=root_digest, scenario_id=scenario_id
        )
        if legacy_identity is not None:
            return legacy_identity
        return _unavailable_with_root(root, root_digest, "scenario_input_expansion_incomplete")

    scenarios = _select_scenarios(report.scenarios, scenario_id, root=root)
    if isinstance(scenarios, dict):
        return _with_root_digest(scenarios, root_digest)
    manifest_records = _manifest_records(root, root_digest, report.manifest_sources)
    if isinstance(manifest_records, dict):
        return _with_root_digest(manifest_records, root_digest)
    resource_records = _runtime_resource_records(root, root_digest, scenarios)
    if isinstance(resource_records, dict):
        return _with_root_digest(resource_records, root_digest)
    records = sorted(
        [*manifest_records, *resource_records],
        key=lambda record: (
            record["role"] or "",
            record["scenario_id"] or "",
            record["sha256"] or "",
            record["path"] or "",
        ),
    )
    effective_digest, requires_closure = _effective_digest(records, root_digest)
    return {
        "status": "available",
        "path": root.as_posix(),
        "source_artifact_sha256": root_digest,
        "effective_input_sha256": effective_digest,
        "requires_effective_input_binding": requires_closure,
        "files": records,
        "reason_code": None,
    }


def runtime_input_records_match(
    identity: Mapping[str, Any],
    consumed_records: list[Mapping[str, Any]],
    *,
    scenario_id: str,
) -> bool:
    """Check exact parser-consumed map/route snapshots against the declared closure.

    Returns:
        ``True`` only when every declared runtime input was consumed with the same
        resolved path, content digest, parser, role, and scenario attribution.
    """
    if identity.get("status") != "available":
        return False
    root_value = identity.get("path")
    if not isinstance(root_value, str) or not root_value:
        return False
    root = Path(root_value).resolve()
    files = identity.get("files")
    if not isinstance(files, list):
        return False
    expected_records = [
        record
        for record in files
        if isinstance(record, Mapping)
        and record.get("role") != "scenario_manifest"
        and record.get("scenario_id") == scenario_id
    ]
    actual_records = [
        record for record in consumed_records if record.get("scenario_id") == scenario_id
    ]
    if len(expected_records) != len(actual_records):
        return False

    def normalized(record: Mapping[str, Any], *, expected: bool) -> tuple[Any, ...] | None:
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            return None
        path = Path(path_value)
        if expected and not path.is_absolute():
            path = root.parent / path
        values = (
            record.get("role"),
            record.get("scenario_id"),
            record.get("sha256"),
            path.resolve().as_posix(),
            record.get("parser"),
        )
        if not all(isinstance(value, str) and value for value in values):
            return None
        map_id = record.get("map_id") if "map_id" in record else None
        return (*values, map_id)

    expected = [normalized(record, expected=True) for record in expected_records]
    actual = [normalized(record, expected=False) for record in actual_records]
    if any(record is None for record in [*expected, *actual]):
        return False
    return sorted(expected) == sorted(actual)


def _load_scenario_report(
    scenario_path: str | Path,
    *,
    validation_report: ScenarioValidationReport | None = None,
) -> tuple[Path, str, Any] | dict[str, Any]:
    """Resolve and verify a scenario parse against the exact bytes it consumed.

    Returns:
        The resolved path, parser-consumed root digest, and validation report, or
        an unavailable payload. A supplied report lets producers bind identity to
        the exact parse they use instead of reparsing an include graph.
    """
    try:
        root = Path(scenario_path).expanduser().resolve(strict=True)
        if not root.is_file():
            return _unavailable(str(root), "scenario_manifest_not_a_file")
        report = validation_report or load_scenarios_for_validation(root)
        root_source = next(
            (item for item in report.manifest_sources if item.path.resolve() == root),
            None,
        )
        root_digest = root_source.content_sha256 if root_source is not None else None
        if root_digest is None:
            return _unavailable(str(root), "scenario_manifest_read_digest_missing")
        if _file_sha256(root) != root_digest:
            return _unavailable_with_root(
                root, root_digest, "scenario_manifest_changed_during_load"
            )
        for source in report.manifest_sources:
            if source.content_sha256 is None:
                return _unavailable(source.path.as_posix(), "scenario_manifest_read_digest_missing")
            if _file_sha256(source.path) != source.content_sha256:
                return _unavailable_with_root(
                    root,
                    root_digest,
                    "included_scenario_manifest_changed_during_load",
                )
        return root, root_digest, report
    except (OSError, RuntimeError, ValueError, TypeError):
        return _unavailable(str(scenario_path), "scenario_manifest_unavailable")


def _select_scenarios(
    scenarios: list[Mapping[str, Any]], scenario_id: str | None, *, root: Path
) -> list[Mapping[str, Any]] | dict[str, Any]:
    """Select one named scenario, or retain the full manifest for report-level identity.

    Returns:
        The selected scenario rows, or an unavailable payload when identity is ambiguous.
    """
    if scenario_id is None:
        return scenarios
    selected = [scenario for scenario in scenarios if _scenario_id(scenario) == scenario_id]
    return (
        selected
        if len(selected) == 1
        else _unavailable(root.as_posix(), "scenario_identity_missing_or_ambiguous")
    )


def _manifest_records(
    root: Path, root_digest: str, manifest_sources: list[Any]
) -> list[dict[str, str | None]] | dict[str, Any]:
    """Hash the root and all recursively included scenario manifests.

    Returns:
        One identity record per manifest, or an unavailable payload.
    """
    records: list[dict[str, str | None]] = []
    source_digests = {
        item.path.resolve(): item.content_sha256
        for item in manifest_sources
        if getattr(item, "content_sha256", None) is not None
    }
    manifests = {root, *(item.path.resolve() for item in manifest_sources)}
    for manifest in sorted(manifests, key=lambda value: value.as_posix()):
        digest = root_digest if manifest == root else source_digests.get(manifest)
        if digest is None:
            return _unavailable_with_root(
                manifest, root_digest, "included_scenario_manifest_unreadable"
            )
        records.append(
            {
                "role": "scenario_manifest",
                "scenario_id": None,
                "sha256": digest,
                "path": _portable_path(manifest, root=root),
            }
        )
    return records


def _runtime_resource_records(  # noqa: C901 - closure failures are explicit per resource role.
    root: Path, root_digest: str, scenarios: list[Mapping[str, Any]]
) -> list[dict[str, str | None]] | dict[str, Any]:
    """Hash the resolved map and route-override bytes for selected scenario rows.

    Returns:
        One identity record per referenced runtime resource, or an unavailable payload.
    """
    records: list[dict[str, str | None]] = []
    for scenario in scenarios:
        sid = _scenario_id(scenario)
        source_file = Path(getattr(scenario, "_scenario_source_file", root)).resolve()
        for role, key in (
            ("map_file", "map_file"),
            ("route_overrides_file", "route_overrides_file"),
        ):
            reference = scenario.get(key)
            if reference is None:
                if key == "map_file" and scenario.get("map_id"):
                    return _unavailable_with_root(
                        root, root_digest, "scenario_map_reference_unresolved"
                    )
                if key == "map_file":
                    default_records = _default_map_pool_records(root, root_digest, sid)
                    if isinstance(default_records, dict):
                        return default_records
                    records.extend(default_records)
                continue
            resolved = _resolve_reference(root, source_file, reference, role, key)
            if isinstance(resolved, dict):
                return _with_root_digest(resolved, root_digest)
            digest = _file_sha256(resolved)
            if digest is None:
                return _unavailable_with_root(resolved, root_digest, f"{role}_reference_unreadable")
            record: dict[str, str | None] = {
                "role": role,
                "scenario_id": sid,
                "sha256": digest,
                "path": _portable_path(resolved, root=root),
            }
            if key == "map_file":
                map_id = scenario.get("map_id")
                if isinstance(map_id, str) and map_id.strip():
                    record["map_id"] = map_id.strip()
                record["parser"] = _map_parser_for_path(resolved)
            else:
                record["parser"] = "yaml"
            records.append(record)
    return records


def _default_map_pool_records(
    root: Path, root_digest: str, scenario_id: str | None
) -> list[dict[str, str | None]] | dict[str, Any]:
    """Hash the SVG files loaded into the default map pool for a scenario.

    Returns:
        Resource records, or an unavailable identity payload if the pool cannot be read.
    """
    maps_folder = Path(DEFAULT_MAPS_FOLDER).resolve()
    try:
        map_paths = sorted(
            (
                path
                for path in maps_folder.iterdir()
                if path.is_file() and path.suffix.lower() == ".svg"
            ),
            key=lambda path: path.stem,
        )
    except OSError:
        return _unavailable_with_root(root, root_digest, "default_map_pool_unavailable")
    if not map_paths:
        return _unavailable_with_root(root, root_digest, "default_map_pool_empty")
    records: list[dict[str, str | None]] = []
    for map_path in map_paths:
        digest = _file_sha256(map_path)
        if digest is None:
            return _unavailable_with_root(map_path, root_digest, "default_map_pool_map_unreadable")
        records.append(
            {
                "role": "default_map_pool",
                "scenario_id": scenario_id,
                "sha256": digest,
                "path": map_path.resolve().as_posix(),
                "map_id": map_path.stem,
                "parser": "svg",
            }
        )
    return records


def _resolve_reference(
    root: Path, source_file: Path, reference: Any, role: str, key: str
) -> Path | dict[str, Any]:
    """Resolve one declared external path without guessing when it is malformed or missing.

    Returns:
        Existing resolved file path, or an unavailable payload.
    """
    if not isinstance(reference, str) or not reference.strip():
        return _unavailable(root.as_posix(), f"{role}_reference_malformed")
    referenced_path = Path(reference).expanduser()
    if not referenced_path.is_absolute():
        base = root.parent if key == "map_file" else source_file.parent
        referenced_path = base / referenced_path
    try:
        resolved = referenced_path.resolve(strict=True)
    except (OSError, RuntimeError):
        return _unavailable(str(referenced_path), f"{role}_reference_unavailable")
    if not resolved.is_file():
        return _unavailable(resolved.as_posix(), f"{role}_reference_not_a_file")
    return resolved


def _effective_digest(records: list[dict[str, str | None]], root_digest: str) -> tuple[str, bool]:
    """Build a location-independent digest of the complete content input closure.

    Returns:
        Effective digest and whether more than the root manifest was required.
    """
    requires_closure = len(records) > 1
    if not requires_closure:
        return root_digest, False
    canonical = json.dumps(
        {
            "schema_version": SCENARIO_INPUT_IDENTITY_SCHEMA,
            "files": [
                {
                    key: record[key]
                    for key in ("role", "scenario_id", "sha256", "path", "map_id", "parser")
                    if key in record
                }
                for record in records
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), True


def _map_parser_for_path(path: Path) -> str:
    """Name the runtime parser selected by the resolved map file suffix.

    Returns:
        Parser identity matching the scenario loader's suffix dispatch.
    """
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return "svg"
    if suffix in {".json", ".yaml", ".yml"}:
        return "legacy_serialized_map"
    return "unsupported"


def _unavailable_with_root(path: Path, root_digest: str, reason_code: str) -> dict[str, Any]:
    """Return unavailable closure evidence while retaining the known root digest."""
    return _with_root_digest(_unavailable(path.as_posix(), reason_code), root_digest)


def _with_root_digest(payload: dict[str, Any], root_digest: str) -> dict[str, Any]:
    """Attach the root manifest digest to a fail-closed result.

    Returns:
        Payload with the known root digest preserved.
    """
    return {**payload, "source_artifact_sha256": root_digest}


def _scenario_id(scenario: Mapping[str, Any]) -> str | None:
    value = scenario.get("name") or scenario.get("scenario_id") or scenario.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _legacy_single_row_identity(
    root: Path,
    *,
    root_digest: str,
    scenario_id: str | None,
) -> dict[str, Any] | None:
    """Support a legacy single-row scenario file with a complete runtime input closure.

    Returns:
        Bound single-row identity, or ``None`` when its references cannot be resolved.
    """
    try:
        root_bytes = root.read_bytes()
        if hashlib.sha256(root_bytes).hexdigest() != root_digest:
            return None
        payload = yaml.safe_load(root_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(payload, Mapping) or "scenarios" in payload:
        return None
    external_reference_keys = {
        "includes",
        "include",
        "scenario_files",
        "map_id",
        "map_file",
        "map_search_paths",
        "route_overrides_file",
    }
    if any(
        key in payload and _has_external_reference_value(payload[key])
        for key in external_reference_keys
    ):
        return None
    candidate_id = next(
        (
            value.strip()
            for key in ("name", "scenario_id", "id")
            if isinstance((value := payload.get(key)), str) and value.strip()
        ),
        None,
    )
    if scenario_id is not None and scenario_id != candidate_id:
        return None
    runtime_records = _runtime_resource_records(root, root_digest, [payload])
    if isinstance(runtime_records, dict):
        return None
    records: list[dict[str, str | None]] = [
        {
            "role": "scenario_manifest",
            "scenario_id": None,
            "sha256": root_digest,
            "path": root.name,
        },
        *runtime_records,
    ]
    effective_digest, requires_closure = _effective_digest(records, root_digest)
    return {
        "status": "available",
        "path": root.as_posix(),
        "source_artifact_sha256": root_digest,
        "effective_input_sha256": effective_digest,
        "requires_effective_input_binding": requires_closure,
        "files": records,
        "reason_code": None,
    }


def _has_external_reference_value(value: Any) -> bool:
    """Return whether an optional loader reference declares an external input."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    return True


def _file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _portable_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.parent).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _unavailable(path: str, reason_code: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "path": path,
        "source_artifact_sha256": None,
        "effective_input_sha256": None,
        "requires_effective_input_binding": True,
        "files": [],
        "reason_code": reason_code,
    }
