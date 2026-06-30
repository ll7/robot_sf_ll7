"""Read-only readiness preflight for the compact CARLA native↔aligned parity bundle.

Issue #1510 asks to *run* the compact multi-scenario native↔aligned CARLA parity bundle, but the
run itself requires a capable CARLA host and is explicitly **out of scope** for laptop-local agents
(see ``docs/context/issue_1511_carla_native_aligned_parity_claim_boundary.md`` and
``docs/context/issue_1509_carla_native_fixture_certification.md``). This module implements the
agent-executable local slice only: a bounded, fail-closed preflight that checks whether the
*prerequisites* for that bundle are present before a CARLA host is asked to execute it.

The preflight is deliberately conservative:

* It never imports CARLA, starts a simulator, or executes a replay.
* It never asserts metric parity. A "ready" verdict means the static T0 export inputs and output
  location are in place, **not** that native↔aligned parity holds. Runtime eligibility (for
  example the #1444 coordinate-alignment outcome) can only be decided on a CARLA host and remains a
  caveat recorded in the report.
* It fails closed: any missing manifest, unreadable/invalid payload, ineligible or absent scenario
  certificate, missing provenance commit, or unsafe output path marks the affected scenario
  ``blocked`` and the whole bundle ``not-ready``.

It composes the canonical T0 export readers
(:func:`robot_sf_carla_bridge.export.resolve_export_manifest_payload_paths`,
:func:`robot_sf_carla_bridge.export.read_export_payload`) and the canonical parity metric list
(:data:`robot_sf_carla_bridge.parity.DEFAULT_PARITY_METRICS`) rather than re-deriving them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema

from robot_sf_carla_bridge.export import (
    read_export_manifest,
    read_export_payload,
    resolve_export_manifest_payload_paths,
)
from robot_sf_carla_bridge.parity import DEFAULT_PARITY_METRICS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PREFLIGHT_SCHEMA_VERSION = "carla-parity-bundle-preflight.v1"
"""Version tag for the readiness report emitted by :func:`check_parity_bundle_readiness`."""

READINESS_MANIFEST_SCHEMA_VERSION = "carla-parity-readiness-manifest.v1"
"""Version tag for issue #1491 native/aligned bundle readiness manifests."""

READINESS_MANIFEST_MODES = frozenset({"native", "aligned"})
"""Replay modes that must both be declared before the issue #1491 bundle is ready."""

# Certificate statuses the scenario certifier may stamp on a certified fixture. Mirrors the
# ``status`` enum in ``schemas/carla_replay_export.v1.json``; kept here so a scenario whose
# certificate is structurally valid but not certified still fails the readiness gate explicitly.
ELIGIBLE_CERTIFICATE_STATUSES = frozenset({"passed", "valid", "hard_but_solvable", "knife_edge"})


def check_parity_bundle_readiness(
    manifest_paths: Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    metric_names: Iterable[str] = DEFAULT_PARITY_METRICS,
) -> dict[str, Any]:
    """Check whether the compact CARLA parity bundle's prerequisites are present.

    Parameters
    ----------
    manifest_paths:
        One or more T0 export manifest paths, one per candidate scenario bundle. Each manifest may
        reference several scenario payloads; every payload becomes its own readiness row.
    output_dir:
        Optional intended output directory for the parity-bundle run. When provided it is checked
        for path safety and overwrite risk; it is **never created** (this preflight is read-only).
    metric_names:
        Parity metrics the downstream bundle intends to compare. Used only to report which metrics
        are not present in a payload's recorded ``trajectory_fields`` (informational, not blocking).

    Returns
    -------
    dict[str, Any]
        A JSON-safe readiness report tagged with :data:`PREFLIGHT_SCHEMA_VERSION`. The top-level
        ``status`` is ``"ready"`` only when at least one scenario row is present, every row is
        ``ready``, and the output location is not unsafe.
    """

    metric_list = list(metric_names)
    scenario_rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        scenario_rows.extend(_rows_for_manifest(Path(manifest_path), metric_list))

    output_location = _check_output_location(output_dir)

    ready_rows = [row for row in scenario_rows if row["status"] == "ready"]
    blocking_reasons: list[str] = []
    if not scenario_rows:
        blocking_reasons.append("no scenario manifests were provided")
    for row in scenario_rows:
        label = row.get("scenario_id") or row["manifest"]
        blocking_reasons.extend(f"{label}: {reason}" for reason in row["reasons"])
    if output_location["status"] == "unsafe":
        blocking_reasons.append(f"output location: {output_location['reason']}")

    status = "ready" if scenario_rows and not blocking_reasons else "not-ready"
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "scenario_count": len(scenario_rows),
        "ready_count": len(ready_rows),
        "metric_names": metric_list,
        "output_location": output_location,
        "scenarios": scenario_rows,
        "blocking_reasons": blocking_reasons,
        "claim_boundary": (
            "Readiness checks static T0 export prerequisites only. It does not run CARLA and does "
            "not assert native↔aligned metric parity; runtime eligibility must be confirmed on a "
            "capable CARLA host."
        ),
    }


def check_native_aligned_bundle_manifest_readiness(  # noqa: C901
    readiness_manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    metric_names: Iterable[str] = DEFAULT_PARITY_METRICS,
) -> dict[str, Any]:
    """Check a declared issue #1491 native/aligned multi-scenario bundle manifest.

    The manifest is intentionally local and static: it names the intended native/aligned
    scenario export manifests, then this checker reuses the existing T0 export preflight for
    fixture certificate and provenance checks. It never imports CARLA, starts a simulator, or
    treats readiness as replay parity evidence.

    Returns:
        JSON-safe readiness report with explicit blockers and claim boundary.
    """

    metric_list = list(metric_names)
    manifest_path = Path(readiness_manifest_path)
    manifest_label = manifest_path.as_posix()
    try:
        readiness_manifest = _read_readiness_manifest(manifest_path)
    except FileNotFoundError:
        return _blocked_readiness_manifest_report(
            manifest_label,
            output_dir=output_dir,
            metric_names=metric_list,
            reason="readiness manifest file not found",
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _blocked_readiness_manifest_report(
            manifest_label,
            output_dir=output_dir,
            metric_names=metric_list,
            reason=f"invalid readiness manifest: {exc}",
        )

    manifest_dir = manifest_path.parent
    entries = _readiness_entries(readiness_manifest)
    scenario_rows: list[dict[str, Any]] = []
    manifest_blockers: list[str] = []

    if not entries:
        manifest_blockers.append("readiness manifest declares no scenarios")

    declared_modes: set[str] = set()
    for index, entry in enumerate(entries):
        label = _readiness_entry_label(entry, index)
        mode = entry.get("mode")
        if isinstance(mode, str):
            mode = mode.strip().lower()
        if mode not in READINESS_MANIFEST_MODES:
            scenario_rows.append(
                _blocked_manifest_row(
                    label,
                    f"readiness entry mode must be one of {sorted(READINESS_MANIFEST_MODES)}",
                )
            )
            continue
        declared_modes.add(mode)

        export_manifest_value = entry.get("manifest")
        if not isinstance(export_manifest_value, str) or not export_manifest_value.strip():
            scenario_rows.append(
                _blocked_manifest_row(label, "readiness entry manifest is missing")
            )
            continue
        export_manifest_path = _resolve_manifest_reference(manifest_dir, export_manifest_value)
        rows = _rows_for_manifest(export_manifest_path, metric_list)
        for row in rows:
            row["readiness_manifest"] = manifest_label
            row["readiness_mode"] = mode
            expected_scenario_id = entry.get("scenario_id")
            if (
                row["status"] == "ready"
                and isinstance(expected_scenario_id, str)
                and expected_scenario_id.strip()
                and row.get("scenario_id") != expected_scenario_id
            ):
                row["status"] = "blocked"
                row["reasons"].append(
                    "readiness entry scenario_id does not match export manifest payload"
                )
            scenario_rows.append(row)

    missing_modes = sorted(READINESS_MANIFEST_MODES - declared_modes)
    if missing_modes:
        manifest_blockers.append(
            f"readiness manifest missing mode coverage: {', '.join(missing_modes)}"
        )

    return _build_readiness_report(
        scenario_rows,
        output_dir=output_dir,
        metric_names=metric_list,
        extra_blockers=manifest_blockers,
        readiness_manifest=manifest_label,
        bundle_id=readiness_manifest.get("bundle_id"),
        issue=readiness_manifest.get("issue"),
    )


def _rows_for_manifest(manifest_path: Path, metric_names: list[str]) -> list[dict[str, Any]]:
    """Return per-scenario readiness rows for one export manifest.

    A manifest that cannot be read or validated yields a single blocked row with no ``scenario_id``
    so the failure is surfaced rather than silently dropped.
    """

    manifest_label = manifest_path.as_posix()
    try:
        read_export_manifest(manifest_path)
        payload_refs = resolve_export_manifest_payload_paths(manifest_path)
    except FileNotFoundError:
        return [_blocked_manifest_row(manifest_label, "export manifest file not found")]
    except (ValueError, OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return [_blocked_manifest_row(manifest_label, f"invalid export manifest: {exc}")]

    if not payload_refs:
        return [_blocked_manifest_row(manifest_label, "export manifest references no scenarios")]

    rows: list[dict[str, Any]] = []
    for ref in payload_refs:
        rows.append(
            _row_for_payload(
                manifest_label,
                scenario_id=str(ref["scenario_id"]),
                payload_path=Path(ref["path"]),
                metric_names=metric_names,
            )
        )
    return rows


def _row_for_payload(
    manifest_label: str,
    *,
    scenario_id: str,
    payload_path: Path,
    metric_names: list[str],
) -> dict[str, Any]:
    """Evaluate one scenario payload file and return its readiness row.

    Returns
    -------
    dict[str, Any]
        A readiness row that is ``ready`` only when the payload reads, validates, and carries the
        required certificate/provenance/trajectory metadata; otherwise ``blocked`` with reasons.
    """

    try:
        payload = read_export_payload(payload_path)
    except FileNotFoundError:
        return _blocked_payload_row(
            manifest_label, scenario_id, payload_path, "export payload file not found"
        )
    except (ValueError, OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return _blocked_payload_row(
            manifest_label, scenario_id, payload_path, f"invalid export payload: {exc}"
        )

    reasons, checks = evaluate_payload_metadata(payload, metric_names=metric_names)
    return {
        "manifest": manifest_label,
        "scenario_id": scenario_id,
        "payload": payload_path.as_posix(),
        "status": "ready" if not reasons else "blocked",
        "reasons": reasons,
        "checks": checks,
    }


def evaluate_payload_metadata(
    payload: dict[str, Any],
    *,
    metric_names: Iterable[str] = DEFAULT_PARITY_METRICS,
) -> tuple[list[str], dict[str, Any]]:
    """Check a single export payload's fixture/provenance metadata for parity-bundle readiness.

    This is the pure, file-system-free core of the preflight so callers (and tests) can exercise
    missing-certificate / missing-provenance cases directly against a payload mapping.

    Returns
    -------
    tuple[list[str], dict[str, Any]]
        ``(blocking_reasons, checks)``. ``blocking_reasons`` is empty when the payload is ready.
        ``checks`` records the observed certificate status, provenance commit, declared trajectory
        fields, and which requested parity metrics are not present in those fields (informational).
    """

    reasons: list[str] = []
    scenario = payload.get("scenario")
    scenario = scenario if isinstance(scenario, dict) else {}

    certificate = scenario.get("certificate")
    certificate = certificate if isinstance(certificate, dict) else {}
    cert_status = certificate.get("status")
    if not isinstance(cert_status, str) or not cert_status.strip():
        reasons.append("scenario certificate status is missing")
        cert_status = None
    elif cert_status not in ELIGIBLE_CERTIFICATE_STATUSES:
        reasons.append(f"scenario certificate status {cert_status!r} is not eligible")

    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    robot_sf_commit = provenance.get("robot_sf_commit")
    if not isinstance(robot_sf_commit, str) or not robot_sf_commit.strip():
        reasons.append("provenance robot_sf_commit is missing")
        robot_sf_commit = None

    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    trajectory_fields = metrics.get("trajectory_fields")
    if not isinstance(trajectory_fields, list) or not trajectory_fields:
        reasons.append("metrics.trajectory_fields is missing or empty")
        trajectory_fields = []

    declared = {str(field) for field in trajectory_fields}
    missing_parity_metrics = [name for name in metric_names if name not in declared]

    checks = {
        "certificate_status": cert_status,
        "robot_sf_commit": robot_sf_commit,
        "trajectory_fields": [str(field) for field in trajectory_fields],
        # Informational only: parity metrics such as ``snqi`` are computed downstream and need not
        # appear in the recorded trajectory fields, so this never blocks readiness on its own.
        "missing_parity_metrics": missing_parity_metrics,
    }
    return reasons, checks


def _check_output_location(output_dir: str | Path | None) -> dict[str, Any]:
    """Inspect the intended parity-bundle output directory without creating it.

    Returns
    -------
    dict[str, Any]
        A location record whose ``status`` is ``not-specified``, ``ready`` (writable, empty or
        absent), ``not-empty`` (exists with contents — overwrite risk, informational), or
        ``unsafe`` (path traversal or a non-directory collision — blocking).
    """

    if output_dir is None:
        return {
            "path": None,
            "status": "not-specified",
            "reason": "no output directory was provided; the bundle run must choose one",
        }

    raw = str(output_dir)
    path = Path(raw)
    if not raw.strip():
        return {"path": raw, "status": "unsafe", "reason": "output directory path is blank"}
    if any(part == ".." for part in path.parts):
        return {
            "path": path.as_posix(),
            "status": "unsafe",
            "reason": "output directory path must not contain parent-relative segments",
        }
    if path.exists() and not path.is_dir():
        return {
            "path": path.as_posix(),
            "status": "unsafe",
            "reason": "output path exists and is not a directory",
        }
    if path.is_dir() and any(path.iterdir()):
        return {
            "path": path.as_posix(),
            "status": "not-empty",
            "reason": "output directory already contains files; a run would risk overwriting them",
        }
    return {
        "path": path.as_posix(),
        "status": "ready",
        "reason": "output directory is empty or absent and can be created by the bundle run",
    }


def _read_readiness_manifest(path: Path) -> dict[str, Any]:
    """Read and minimally validate an issue #1491 readiness manifest.

    Returns:
        Parsed readiness manifest mapping.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("readiness manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != READINESS_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"readiness manifest schema_version must be {READINESS_MANIFEST_SCHEMA_VERSION!r}"
        )
    return payload


def _readiness_entries(readiness_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return object-valued scenario entries from a readiness manifest.

    Returns:
        Scenario entry mappings; non-object entries are ignored.
    """

    scenarios = readiness_manifest.get("scenarios")
    if not isinstance(scenarios, list):
        return []
    return [entry for entry in scenarios if isinstance(entry, dict)]


def _readiness_entry_label(entry: dict[str, Any], index: int) -> str:
    """Return a stable label for blocker rows from a readiness manifest entry.

    Returns:
        Scenario identifier when present, otherwise an index label.
    """

    scenario_id = entry.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id.strip():
        return scenario_id
    return f"readiness_entry[{index}]"


def _resolve_manifest_reference(manifest_dir: Path, manifest_value: str) -> Path:
    """Resolve a readiness-manifest export reference relative to the manifest file.

    Returns:
        Absolute path or manifest-directory-relative path object.
    """

    path = Path(manifest_value)
    return path if path.is_absolute() else manifest_dir / path


def _build_readiness_report(
    scenario_rows: list[dict[str, Any]],
    *,
    output_dir: str | Path | None,
    metric_names: list[str],
    extra_blockers: list[str] | None = None,
    readiness_manifest: str | None = None,
    bundle_id: Any = None,
    issue: Any = None,
) -> dict[str, Any]:
    """Build the shared readiness report envelope.

    Returns:
        JSON-safe readiness report.
    """

    output_location = _check_output_location(output_dir)
    ready_rows = [row for row in scenario_rows if row["status"] == "ready"]
    blocking_reasons: list[str] = list(extra_blockers or [])
    if not scenario_rows:
        blocking_reasons.append("no scenario manifests provided")
    for row in scenario_rows:
        label = row.get("scenario_id") or row["manifest"]
        blocking_reasons.extend(f"{label}: {reason}" for reason in row["reasons"])
    if output_location["status"] == "unsafe":
        blocking_reasons.append(f"output location: {output_location['reason']}")

    status = "ready" if scenario_rows and not blocking_reasons else "not-ready"
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "scenario_count": len(scenario_rows),
        "ready_count": len(ready_rows),
        "metric_names": metric_names,
        "output_location": output_location,
        "scenarios": scenario_rows,
        "blocking_reasons": blocking_reasons,
        "claim_boundary": (
            "Readiness checks static T0 export prerequisites only. It does not run CARLA and does "
            "not assert native↔aligned metric parity; runtime eligibility must be confirmed on a "
            "capable CARLA host."
        ),
    }
    if readiness_manifest is not None:
        report["readiness_manifest"] = readiness_manifest
        report["readiness_manifest_schema_version"] = READINESS_MANIFEST_SCHEMA_VERSION
    if isinstance(bundle_id, str) and bundle_id.strip():
        report["bundle_id"] = bundle_id
    if isinstance(issue, int):
        report["issue"] = issue
    return report


def _blocked_readiness_manifest_report(
    readiness_manifest: str,
    *,
    output_dir: str | Path | None,
    metric_names: list[str],
    reason: str,
) -> dict[str, Any]:
    """Return a fail-closed report for an unreadable readiness manifest.

    Returns:
        Not-ready report containing a manifest-level blocker row.
    """

    return _build_readiness_report(
        [_blocked_manifest_row(readiness_manifest, reason)],
        output_dir=output_dir,
        metric_names=metric_names,
        readiness_manifest=readiness_manifest,
    )


def _blocked_manifest_row(manifest_label: str, reason: str) -> dict[str, Any]:
    """Return a blocked readiness row for a manifest-level failure."""

    return {
        "manifest": manifest_label,
        "scenario_id": None,
        "payload": None,
        "status": "blocked",
        "reasons": [reason],
        "checks": {},
    }


def _blocked_payload_row(
    manifest_label: str, scenario_id: str, payload_path: Path, reason: str
) -> dict[str, Any]:
    """Return a blocked readiness row for a payload-level failure."""

    return {
        "manifest": manifest_label,
        "scenario_id": scenario_id,
        "payload": payload_path.as_posix(),
        "status": "blocked",
        "reasons": [reason],
        "checks": {},
    }
