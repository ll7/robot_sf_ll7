"""Compose deterministic, diagnostic-only trace dossier packages.

This module composes the already-versioned representative selector, pinned trace exporter,
cell-binding metadata, and four-panel renderer.  It consumes existing campaign artifacts only;
it never runs a simulator, submits compute, or promotes a trace to benchmark evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.trace_dossier_renderer import (
    TraceDossierRenderError,
    render_trace_dossier,
    validate_trace_dossier_manifest,
)
from robot_sf.benchmark.candidate_trace_resolution import resolve_episode_source
from robot_sf.benchmark.trace_dossier_cell_binding import build_trace_dossier_cell_binding
from robot_sf.benchmark.trace_dossier_selection import select_representative
from scripts.tools.export_trace_dossier import export_trace_dossier

TRACE_DOSSIER_PACKAGE_SCHEMA_VERSION = "trace_dossier_package.v1"
TRACE_DOSSIER_PACKAGE_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "trace_dossier_package.v1.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_CHECKSUMS = "SHA256SUMS"


class TraceDossierPackageError(ValueError):
    """Raised when a trace dossier package cannot be composed defensibly."""


@dataclass(frozen=True, slots=True)
class TraceDossierPackageResult:
    """Paths and deterministic manifest returned by package composition."""

    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def build_trace_dossier_package(
    *,
    candidates: Sequence[Mapping[str, Any]],
    release_manifest_path: Path,
    campaign_store_dir: Path,
    output_dir: Path,
    trace_search_roots: tuple[Path, ...] = (),
    command: str = "scripts/tools/build_trace_dossier_package.py",
) -> TraceDossierPackageResult:
    """Build one cell-bound trace dossier package from existing artifacts.

    Candidate rows must contain the selector fields plus campaign identity and the expected
    ``episode_id``.  ``trace_artifact_uri``/``trace_sha256`` are optional compatibility fields;
    when present, the resolved exporter source must agree with them.

    Returns:
        A package result containing the output directory and canonical manifest.
    """

    rows = _normalize_candidates(candidates)
    selection = select_representative(rows)
    selected = _selected_row(rows, selection.selected_seed_id)
    _reject_output_overlap(output_dir, campaign_store_dir)
    resolution = resolve_episode_source(
        scenario_id=selected["scenario_id"],
        planner_id=selected["planner_id"],
        seed=selected["seed"],
        campaign_store_dir=campaign_store_dir,
        trace_search_roots=trace_search_roots,
    )
    _validate_campaign_resolution(resolution, selected)
    _reject_source_overlap(output_dir, resolution)
    _prepare_output_dir(output_dir)

    export_dir = output_dir / "export"
    render_dir = output_dir / "render"
    export_manifest = export_trace_dossier(
        scenario_id=selected["scenario_id"],
        planner_id=selected["planner_id"],
        seed=selected["seed"],
        release_manifest_path=release_manifest_path,
        campaign_store_dir=campaign_store_dir,
        output_dir=export_dir,
        trace_search_roots=trace_search_roots,
    )
    _validate_export_identity(export_manifest, selected)

    trace_path = export_dir / "trace.json"
    render_path = render_dir / "dossier.png"
    render_manifest_path = render_dir / "renderer_manifest.json"
    render_trace_dossier(
        trace_path,
        output_png=render_path,
        manifest_path=render_manifest_path,
        command=command,
    )
    render_manifest = _normalize_render_manifest(render_manifest_path)

    trace_sha256 = str(export_manifest["artifacts"]["trace"]["sha256"])
    binding = build_trace_dossier_cell_binding(
        cell={
            "campaign_id": selected["campaign_id"],
            "cell_id": selected["cell_id"],
            "scenario_id": selected["scenario_id"],
            "planner_id": selected["planner_id"],
            "release_arm_id": selected.get("release_arm_id"),
            "scenario_family": selected.get("scenario_family"),
        },
        selected_trace={
            "cell_id": selected["cell_id"],
            "episode_id": selected["episode_id"],
            "seed": selected["seed"],
            "trace_artifact_uri": "export/trace.json",
            "trace_sha256": trace_sha256,
            "terminal_verdict": selected["verdict"],
        },
        terminal_verdict_counts=Counter(row["verdict"] for row in rows),
    )

    artifact_paths = (
        "export/trace.json",
        "export/normalization_receipt.json",
        "export/manifest.json",
        "export/SHA256SUMS",
        "render/dossier.png",
        "render/renderer_manifest.json",
    )
    artifacts = [
        {"path": path, "sha256": _sha256_file(output_dir / path)} for path in artifact_paths
    ]
    manifest = {
        "schema_version": TRACE_DOSSIER_PACKAGE_SCHEMA_VERSION,
        "evidence_boundary": "diagnostic_only",
        "claim_boundary": (
            "one existing release-pinned trace rendered for analysis; no benchmark, statistical, "
            "safety, or paper-facing claim"
        ),
        "composition": {"name": "trace_dossier_package", "version": "issue_7086.v1"},
        "command": command,
        "campaign_cell": {
            "campaign_id": selected["campaign_id"],
            "cell_id": selected["cell_id"],
            "scenario_id": selected["scenario_id"],
            "planner_id": selected["planner_id"],
            "release_arm_id": selected.get("release_arm_id"),
            "scenario_family": selected.get("scenario_family"),
        },
        "selection": selection.to_dict(),
        "cell_binding": binding.to_dict(),
        "trace_export": {
            "manifest": "export/manifest.json",
            "trace": "export/trace.json",
            "checksums": "export/SHA256SUMS",
        },
        "render": {
            "dossier": "render/dossier.png",
            "manifest": "render/renderer_manifest.json",
            "evidence_boundary": render_manifest["evidence_boundary"],
        },
        "artifacts": artifacts,
        "package_checksums": _PACKAGE_CHECKSUMS,
        "limitations": [
            "existing-artifact composition only; no simulation or new trace acquisition",
            "diagnostic-only renderer output is not benchmark or paper-facing evidence",
            "candidate cell identity and verdict counts are supplied metadata; the campaign store validates the study identity, tuple, episode, and artifact provenance",
        ],
    }
    validate_trace_dossier_package_manifest(manifest)
    manifest_path = output_dir / "package_manifest.json"
    _write_canonical_json(manifest_path, manifest)
    _write_package_checksums(output_dir)
    return TraceDossierPackageResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def validate_trace_dossier_package_manifest(payload: Mapping[str, Any]) -> None:
    """Validate one package manifest against the versioned package schema."""

    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in sorted(
            Draft202012Validator(_load_package_schema()).iter_errors(dict(payload)),
            key=lambda item: list(item.absolute_path),
        )
    ]
    if errors:
        raise TraceDossierPackageError("; ".join(errors))


@lru_cache(maxsize=1)
def _load_package_schema() -> dict[str, Any]:
    try:
        return json.loads(TRACE_DOSSIER_PACKAGE_SCHEMA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceDossierPackageError(f"package schema is unreadable: {exc}") from exc


def _normalize_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TraceDossierPackageError("candidates must be a non-empty sequence of mappings")
    if not candidates:
        raise TraceDossierPackageError("candidates must be non-empty")
    rows = [_normalize_candidate(raw, index) for index, raw in enumerate(candidates)]
    _validate_consistent_candidate_fields(rows)
    return rows


def _normalize_candidate(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TraceDossierPackageError(f"candidate {index} is not a mapping")
    row = dict(raw)
    if "planner_id" not in row and "planner" in row:
        row["planner_id"] = row["planner"]
    for field in ("campaign_id", "cell_id", "scenario_id", "planner_id", "episode_id"):
        row[field] = _required_text(row.get(field), f"candidate {index}.{field}")
    seed = row.get("seed")
    if type(seed) is not int or seed < 0:
        raise TraceDossierPackageError(f"candidate {index}.seed must be a non-negative integer")
    row["seed_id"] = _required_text(row.get("seed_id"), f"candidate {index}.seed_id")
    row["verdict"] = _required_text(row.get("verdict"), f"candidate {index}.verdict")
    _normalize_candidate_provenance(row, index)
    return row


def _normalize_candidate_provenance(row: dict[str, Any], index: int) -> None:
    for field in ("release_arm_id", "scenario_family", "trace_artifact_uri", "artifact_uri"):
        if field in row and row[field] is not None:
            row[field] = _required_text(row[field], f"candidate {index}.{field}")
    if "trace_artifact_uri" not in row and "artifact_uri" in row:
        row["trace_artifact_uri"] = row["artifact_uri"]
    if "trace_sha256" not in row and "artifact_sha256" in row:
        row["trace_sha256"] = row["artifact_sha256"]
    if row.get("trace_sha256") is not None:
        trace_sha256 = _required_text(row["trace_sha256"], f"candidate {index}.trace_sha256")
        if _SHA256_RE.fullmatch(trace_sha256.lower()) is None:
            raise TraceDossierPackageError(
                f"candidate {index}.trace_sha256 must be lowercase SHA-256"
            )
        row["trace_sha256"] = trace_sha256.lower()


def _validate_consistent_candidate_fields(rows: Sequence[Mapping[str, Any]]) -> None:
    for field in ("campaign_id", "cell_id", "scenario_id", "planner_id"):
        if len({row[field] for row in rows}) != 1:
            raise TraceDossierPackageError(f"candidates must agree on {field}")
    for field in ("release_arm_id", "scenario_family"):
        values = {row.get(field) for row in rows}
        if len(values) > 1:
            raise TraceDossierPackageError(f"candidates must agree on {field}")


def _selected_row(rows: Sequence[Mapping[str, Any]], seed_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["seed_id"] == seed_id]
    if len(selected) != 1:
        raise TraceDossierPackageError(
            f"selection seed_id {seed_id!r} does not identify exactly one candidate"
        )
    return dict(selected[0])


def _validate_export_identity(
    export_manifest: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    identity = export_manifest.get("identity")
    source = export_manifest.get("source")
    if not isinstance(identity, Mapping) or not isinstance(source, Mapping):
        raise TraceDossierPackageError("export manifest is missing identity or source metadata")
    for field in ("scenario_id", "planner_id", "seed", "episode_id"):
        if identity.get(field) != selected[field]:
            raise TraceDossierPackageError(
                f"export identity mismatch for {field}: expected {selected[field]!r}, "
                f"observed {identity.get(field)!r}"
            )
    expected_uri = selected.get("trace_artifact_uri")
    observed_uri = source.get("artifact_uri")
    if expected_uri is not None and not _same_path_or_text(str(expected_uri), str(observed_uri)):
        raise TraceDossierPackageError(
            "export source artifact URI does not match selected candidate"
        )
    expected_sha256 = selected.get("trace_sha256")
    observed_sha256 = source.get("artifact_sha256")
    if expected_sha256 is not None and str(expected_sha256) != str(observed_sha256).lower():
        raise TraceDossierPackageError("export source artifact SHA-256 does not match candidate")


def _validate_campaign_resolution(
    resolution: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    """Require candidate identity to agree with the authoritative campaign row."""

    if resolution.get("resolution_status") != "resolved":
        return
    observed_campaign_id = resolution.get("campaign_id")
    if not isinstance(observed_campaign_id, str) or not observed_campaign_id.strip():
        raise TraceDossierPackageError(
            "resolved campaign row is missing an authoritative campaign_id"
        )
    if selected["campaign_id"] != observed_campaign_id:
        raise TraceDossierPackageError(
            "candidate campaign_id does not match campaign result store study_id: "
            f"expected {observed_campaign_id!r}, observed {selected['campaign_id']!r}"
        )
    for field in ("scenario_id", "planner_id", "seed", "episode_id"):
        observed = resolution.get(field)
        if observed != selected[field]:
            raise TraceDossierPackageError(
                f"candidate {field} does not match authoritative campaign row: "
                f"expected {observed!r}, observed {selected[field]!r}"
            )
    observed_family = resolution.get("scenario_family")
    selected_family = selected.get("scenario_family")
    if (
        selected_family is not None
        and observed_family is not None
        and selected_family != observed_family
    ):
        raise TraceDossierPackageError(
            "candidate scenario_family does not match authoritative campaign row: "
            f"expected {observed_family!r}, observed {selected_family!r}"
        )


def _normalize_render_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceDossierPackageError(f"renderer manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceDossierPackageError("renderer manifest must be a JSON object")
    try:
        source_trace = payload["source_trace"]
        outputs = payload["outputs"]
        if not isinstance(source_trace, dict) or not isinstance(outputs, dict):
            raise TraceDossierPackageError(
                "renderer manifest source_trace and outputs must be JSON objects"
            )
        png_output = outputs["png"]
        if not isinstance(png_output, dict):
            raise TraceDossierPackageError("renderer manifest outputs.png must be a JSON object")
        source_trace["path"] = "export/trace.json"
        png_output["path"] = "render/dossier.png"
        validate_trace_dossier_manifest(payload)
    except (KeyError, TypeError, TraceDossierRenderError, TraceDossierPackageError) as exc:
        raise TraceDossierPackageError(f"renderer manifest cannot be normalized: {exc}") from exc
    _write_canonical_json(path, payload)
    return payload


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"export", "render", "package_manifest.json", _PACKAGE_CHECKSUMS}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise TraceDossierPackageError(
            "package output directory contains unexpected entries: " + ", ".join(unexpected)
        )
    for name, allowed_files in (
        ("export", {"trace.json", "normalization_receipt.json", "manifest.json", "SHA256SUMS"}),
        ("render", {"dossier.png", "renderer_manifest.json"}),
    ):
        child = output_dir / name
        if not child.exists():
            continue
        if not child.is_dir():
            raise TraceDossierPackageError(f"package output {name!r} is not a directory")
        unexpected_child = sorted(
            path.name for path in child.iterdir() if path.name not in allowed_files
        )
        if unexpected_child:
            raise TraceDossierPackageError(
                f"package output {name!r} contains unexpected entries: "
                + ", ".join(unexpected_child)
            )


def _reject_output_overlap(output_dir: Path, campaign_store_dir: Path) -> None:
    output = output_dir.resolve()
    store = campaign_store_dir.resolve()
    try:
        output.relative_to(store)
    except ValueError:
        pass
    else:
        raise TraceDossierPackageError("package output directory must not be inside campaign store")
    try:
        store.relative_to(output)
    except ValueError:
        pass
    else:
        raise TraceDossierPackageError("campaign store must not be inside package output directory")


def _reject_source_overlap(output_dir: Path, resolution: Mapping[str, Any]) -> None:
    """Reject a resolved source artifact inside the package output before writing any files."""
    if resolution.get("resolution_status") != "resolved":
        return
    raw_source_path = resolution.get("source_path")
    if not isinstance(raw_source_path, str) or not raw_source_path:
        raise TraceDossierPackageError("resolved source artifact path is missing")
    output = output_dir.resolve()
    source = Path(raw_source_path).resolve()
    try:
        source.relative_to(output)
    except ValueError:
        return
    raise TraceDossierPackageError(
        f"source artifact must not be inside package output directory: {source}"
    )


def _write_package_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / _PACKAGE_CHECKSUMS
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != checksum_path.resolve()
    )
    lines = ["# AI-GENERATED NEEDS-REVIEW"]
    lines.extend(
        f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in files
    )
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path_or_text(expected: str, observed: str) -> bool:
    try:
        expected_path = Path(expected)
        observed_path = Path(observed)
        if expected_path.exists() and observed_path.exists():
            return expected_path.resolve() == observed_path.resolve()
    except OSError:
        pass
    return expected == observed


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceDossierPackageError(f"{field} must be non-empty text")
    return value.strip()


__all__ = [
    "TRACE_DOSSIER_PACKAGE_SCHEMA_VERSION",
    "TraceDossierPackageError",
    "TraceDossierPackageResult",
    "build_trace_dossier_package",
    "validate_trace_dossier_package_manifest",
]
