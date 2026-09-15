"""Recorded scenario-state scene rendering over review contracts (SREV-09, issue #9278).

This module is the review-scene contract consumer: it turns explicitly
selected simulation trace exports into numbered per-frame scene figures
(SVG/PDF/PNG) plus a per-frame source map. It reuses the SREV-01
shared-contract surface
(:mod:`robot_sf.analysis_workbench.review_contracts`) for envelopes,
validation, and digests, and the canonical trace owner
(:mod:`robot_sf.analysis_workbench.simulation_trace_export`) to validate
every consumed trace. Rendering uses only Matplotlib primitives over recorded
geometry: it never constructs or advances a simulator, invents measurements,
or normalizes anything. Required geometry missing from a frame fails closed;
schema-optional layers (such as radii) fall back to documented defaults that
are recorded in the source map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    simulation_trace_export_from_dict,
)

COMPONENT_ID = "srev09-review-scene"
COMPONENT_VERSION = "1.0.0"

FORMAT_TRACE_EXPORT = "simulation_trace_export.v1"

SOURCE_MAP_FILENAME = "frame-source-map.json"
DESCRIPTOR_FILENAME = "component-descriptor.json"

_DESCRIBE_OUTPUT_TYPES = ("review-scene.v1",)

_DESCRIPTOR_DOC: dict[str, Any] = {
    "schema_version": "component-descriptor.v1",
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": [],
    "optional_capabilities": [],
    "output_types": list(_DESCRIBE_OUTPUT_TYPES),
}

# Fail fast on descriptor drift: the shipped descriptor must stay schema-valid.
DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOC)

SCENE_FORMATS = ("svg", "png", "pdf")
DEFAULT_SCENE_FORMATS = ("svg", "png", "pdf")
DEFAULT_ROBOT_RADIUS_M = 0.3
DEFAULT_PEDESTRIAN_RADIUS_M = 0.3
DEFAULT_FIGURE = {"width_in": 3.2, "height_in": 2.4, "dpi": 80}

# Pinned savefig behavior: other suites mutate global rcParams (notably
# savefig.bbox=tight via the latex style helper), which would silently crop
# our fixed-canvas figures under pytest-xdist worker reuse. Rendering stays
# hermetic by restoring these keys around every save.
_HERMETIC_SAVEFIG_RCPARAMS = {
    "savefig.bbox": None,
    "savefig.dpi": "figure",
    "figure.constrained_layout.use": False,
}


def descriptor() -> dict[str, Any]:
    """Return the versioned capability descriptor for this component.

    Returns:
        Descriptor document declaring exact required/optional capabilities
        (none) and versioned result artifact types.
    """
    return json.loads(json.dumps(_DESCRIPTOR_DOC))


def _canonical_sha256(value: Any) -> str:
    """Hash logical content deterministically (location-independent).

    Returns:
        Hex SHA-256 digest of the canonical encoding.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Write JSON atomically and return the hex SHA-256 of the file bytes.

    Returns:
        Hex SHA-256 digest of the written file bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    """List requested capabilities this component does not provide.

    Returns:
        Requested capability names absent from the descriptor.
    """
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _major(version: str) -> str | None:
    """Return the ``v<major>`` marker of a dotted version string, if parseable."""
    head = version.strip().split(".", 1)[0].lstrip("vV")
    return f"v{head}" if head.isdigit() else None


def _positive_int(value: Any) -> int | None:
    """Return a non-negative int for index-like inputs."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validate_figure(figure: Any) -> tuple[dict[str, float] | None, list[str]]:
    """Validate the optional figure preset, falling back to defaults.

    Returns:
        Tuple of (figure spec, errors); exactly one side is meaningful.
    """
    if figure is None:
        figure = {}
    if not isinstance(figure, dict):
        return None, ["corrupt-scene: 'figure' must be a mapping"]
    figure_spec: dict[str, float] = {}
    errors: list[str] = []
    for key in ("width_in", "height_in", "dpi"):
        raw = figure.get(key, DEFAULT_FIGURE[key])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not raw > 0:
            errors.append(f"corrupt-scene: figure '{key}' must be a positive number")
            continue
        figure_spec[key] = float(raw)
    if errors:
        return None, errors
    return figure_spec, []


def _validate_scene(scene: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the scene mapping inside a request config.

    Returns:
        Tuple of (normalized scene, errors); exactly one side is meaningful.
    """
    if not isinstance(scene, dict):
        return None, ["corrupt-scene: config must carry a 'scene' mapping"]
    errors: list[str] = []
    indices = scene.get("frame_indices")
    if indices is not None:
        if not isinstance(indices, list) or not indices:
            errors.append("corrupt-scene: 'frame_indices' must be a non-empty list")
        else:
            for index in indices:
                if _positive_int(index) is None:
                    errors.append("corrupt-scene: frame indices must be non-negative integers")
                    break
            if len(set(indices)) != len(indices):
                errors.append("corrupt-scene: frame indices must be distinct")
    formats = scene.get("formats", list(DEFAULT_SCENE_FORMATS))
    if (
        not isinstance(formats, list)
        or not formats
        or any(item not in SCENE_FORMATS for item in formats)
    ):
        errors.append(f"corrupt-scene: 'formats' must be a non-empty subset of {SCENE_FORMATS}")
    figure_spec, figure_errors = _validate_figure(scene.get("figure"))
    errors.extend(figure_errors)
    if errors or figure_spec is None:
        return None, errors or ["corrupt-scene: figure preset could not be normalized"]
    return {
        "frame_indices": list(indices) if indices is not None else None,
        "formats": list(dict.fromkeys(formats)),
        "figure": figure_spec,
    }, []


def _finite_vector2(value: Any) -> list[float] | None:
    """Extract a finite 2D point from a trace position payload.

    Returns:
        The ``[x, y]`` point, or ``None`` when the payload is not finite 2D.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = value[0], value[1]
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    point = [float(x), float(y)]
    if not all(math.isfinite(number) for number in point):
        return None
    return point


def _finite_scalar(value: Any) -> float | None:
    """Extract a finite scalar from a trace field.

    Returns:
        The scalar, or ``None`` when the field is not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probe = float(value)
    if not math.isfinite(probe):
        return None
    return probe


def _frame_geometry(frame: Any, frame_index: int) -> tuple[dict[str, Any] | None, str | None]:
    """Extract required scene geometry from one validated trace frame.

    Returns:
        Tuple of (geometry, error); exactly one side is meaningful.
    """
    robot = frame.robot if isinstance(frame.robot, dict) else {}
    position = _finite_vector2(robot.get("position"))
    if position is None:
        return None, f"missing robot position at frame index {frame_index}"
    heading = _finite_scalar(robot.get("heading"))
    if heading is None:
        return None, f"missing robot heading at frame index {frame_index}"
    velocity = _finite_vector2(robot.get("velocity"))
    if velocity is None:
        return None, f"missing robot velocity at frame index {frame_index}"
    radius = _finite_scalar(robot.get("radius"))
    robot_entry: dict[str, Any] = {
        "position_m": position,
        "heading_rad": heading,
        "velocity_mps": velocity,
        "radius_m": radius if radius is not None else DEFAULT_ROBOT_RADIUS_M,
        "radius_defaulted": radius is None,
    }
    pedestrians = []
    if not isinstance(frame.pedestrians, list):
        return None, f"missing pedestrian list at frame index {frame_index}"
    for pedestrian in frame.pedestrians:
        if not isinstance(pedestrian, dict):
            return None, f"corrupt pedestrian entry at frame index {frame_index}"
        ped_position = _finite_vector2(pedestrian.get("position"))
        if ped_position is None:
            return None, f"missing pedestrian position at frame index {frame_index}"
        ped_radius = _finite_scalar(pedestrian.get("radius"))
        pedestrians.append(
            {
                "id": str(pedestrian.get("id", "")),
                "position_m": ped_position,
                "radius_m": ped_radius if ped_radius is not None else DEFAULT_PEDESTRIAN_RADIUS_M,
                "radius_defaulted": ped_radius is None,
            }
        )
    time_s = _finite_scalar(frame.time_s)
    if time_s is None:
        return None, f"missing frame time at frame index {frame_index}"
    return {
        "step": int(frame.step),
        "time_s": time_s,
        "robot": robot_entry,
        "pedestrians": pedestrians,
    }, None


def _render_scene_figure(geometry: dict[str, Any], figure_spec: dict[str, float]) -> Any:
    """Render one scene frame with canonical Matplotlib primitives.

    Returns:
        The Matplotlib figure (caller saves and closes it).
    """
    figure, ax = plt.subplots(
        figsize=(figure_spec["width_in"], figure_spec["height_in"]),
        dpi=int(figure_spec["dpi"]),
    )
    robot = geometry["robot"]
    ax.add_patch(
        Circle(
            robot["position_m"],
            robot["radius_m"],
            facecolor="tab:blue",
            edgecolor="black",
            label="robot",
        )
    )
    radius = robot["radius_m"]
    heading = robot["heading_rad"]
    ax.annotate(
        "",
        xy=(
            robot["position_m"][0] + radius * 2 * math.cos(heading),
            robot["position_m"][1] + radius * 2 * math.sin(heading),
        ),
        xytext=tuple(robot["position_m"]),
        arrowprops={"facecolor": "black", "width": 1.0, "headwidth": 4.0},
    )
    for pedestrian in geometry["pedestrians"]:
        ax.add_patch(
            Circle(
                pedestrian["position_m"],
                pedestrian["radius_m"],
                facecolor="tab:orange",
                edgecolor="black",
                label=f"pedestrian {pedestrian['id']}",
            )
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"step {geometry['step']} t={geometry['time_s']}s")
    return figure


def _render_trace_scenes(
    trace: SimulationTraceExport,
    artifact_id: str,
    scene: dict[str, Any],
    staging_dir: Path,
    output_directory: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render the selected frames of one trace plus their source-map rows.

    Returns:
        Tuple of (artifacts, source-map rows) for one trace.
    """
    indices = scene["frame_indices"]
    if indices is None:
        indices = list(range(len(trace.frames)))
    unknown = [index for index in indices if index >= len(trace.frames)]
    if unknown:
        raise IndexError(f"frame indices out of range for {len(trace.frames)} frames: {unknown}")
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    with matplotlib.rc_context(_HERMETIC_SAVEFIG_RCPARAMS):
        for ordinal, frame_index in enumerate(indices):
            geometry, error = _frame_geometry(trace.frames[frame_index], frame_index)
            if error is not None or geometry is None:
                raise ValueError(error or "unreadable frame geometry")
            figure = _render_scene_figure(geometry, scene["figure"])
            try:
                for scene_format in scene["formats"]:
                    filename = f"{artifact_id}-scene_{ordinal:06d}.{scene_format}"
                    figure.savefig(staging_dir / filename, format=scene_format)
                    artifacts.append(
                        {
                            "artifact_id": filename,
                            "uri": str(Path(output_directory) / filename),
                            "sha256": hashlib.sha256(
                                (staging_dir / filename).read_bytes()
                            ).hexdigest(),
                        }
                    )
            finally:
                plt.close(figure)
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "trace_id": trace.trace_id,
                    "ordinal": ordinal,
                    "step": geometry["step"],
                    "time_s": geometry["time_s"],
                    "units": dict(trace.units),
                    "robot": geometry["robot"],
                    "pedestrians": geometry["pedestrians"],
                    "files": [
                        artifact["artifact_id"] for artifact in artifacts[-len(scene["formats"]) :]
                    ],
                }
            )
    return artifacts, rows


def _build_source_map(
    request: ComponentRequest,
    scene: dict[str, Any],
    rows: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose the per-frame source map with a logical content digest.

    Returns:
        Source-map payload ready for staging and digesting.
    """
    document = {
        "schema_version": "review-scene-sourcemap.v1",
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "figure": scene["figure"],
        "formats": scene["formats"],
        "frames": rows,
        "provenance": {
            "source_artifact_ids": sorted(record["artifact_id"] for record in source_records),
            "source_artifacts": source_records,
            "source_identity_status": "embedded-observed",
            "note": "observed_sha256 is the digest of the exact bytes parsed by the "
            "canonical trace owner; embedded trace metadata is observed, not an "
            "independent identity attestation.",
        },
    }
    document["sourcemap_sha256"] = _canonical_sha256(
        {
            "figure": document["figure"],
            "formats": document["formats"],
            "frames": rows,
            "provenance": document["provenance"],
        }
    )
    return document


def _stage_outputs(
    output_dir: Path,
    request: ComponentRequest,
    scene: dict[str, Any],
    pairs: list[tuple[Any, str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Render every trace and publish all artifacts atomically.

    Returns:
        Tuple of (artifacts, source-map document, descriptor digest).
    """
    staging_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        artifacts: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        source_records = [source_record for _, _, source_record in pairs]
        for trace, artifact_id, _source_record in pairs:
            trace_artifacts, trace_rows = _render_trace_scenes(
                trace, artifact_id, scene, staging_dir, request.output_directory
            )
            artifacts.extend(trace_artifacts)
            rows.extend(trace_rows)
        source_map = _build_source_map(request, scene, rows, source_records)
        sourcemap_digest = _write_json(staging_dir / SOURCE_MAP_FILENAME, source_map)
        descriptor_digest = _write_json(staging_dir / DESCRIPTOR_FILENAME, _DESCRIPTOR_DOC)
        artifacts.append(
            {
                "artifact_id": SOURCE_MAP_FILENAME,
                "uri": str(Path(request.output_directory) / SOURCE_MAP_FILENAME),
                "sha256": sourcemap_digest,
            }
        )
        artifacts.append(
            {
                "artifact_id": DESCRIPTOR_FILENAME,
                "uri": str(Path(request.output_directory) / DESCRIPTOR_FILENAME),
                "sha256": descriptor_digest,
            }
        )
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {request.output_directory}")
        staging_dir.replace(output_dir)
        staging_dir = None
        return artifacts, source_map, descriptor_digest
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir)


def _resolve_output_directory(root: Path, output_directory: str) -> tuple[Path | None, str | None]:
    """Resolve an output path and reject symlink escapes from the base directory.

    Returns:
        A resolved output path and no error, or ``(None, reason)`` for an escape.
    """
    if not _is_strict_relative_path(output_directory):
        return None, "unsafe-output-path: output directory must be a relative path"
    try:
        resolved_root = root.resolve(strict=False)
        output_dir = (resolved_root / output_directory).resolve(strict=False)
        output_dir.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-output-path: output directory must resolve within base"
    return output_dir, None


def _unavailable(request: ComponentRequest, reason: str) -> ComponentResult:
    """Build an unavailable result without artifacts.

    Returns:
        Unavailable component result carrying only the reason.
    """
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="unavailable",
        reason=reason,
    )


def _failed(
    request: ComponentRequest, reason: str, diagnostics: tuple[dict[str, Any], ...] = ()
) -> ComponentResult:
    """Build a failed result without artifacts.

    Returns:
        Failed component result carrying the reason and diagnostics.
    """
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="failed",
        diagnostics=diagnostics,
        reason=reason,
    )


def _safe_result_identity(request: Any) -> tuple[str, str]:
    """Return valid result identifiers for a possibly malformed request."""
    if isinstance(request, Mapping):
        request_id = request.get("request_id")
        component_id = request.get("component_id")
    else:
        request_id = getattr(request, "request_id", None)
        component_id = getattr(request, "component_id", None)
    if not isinstance(request_id, str) or not request_id.strip():
        request_id = "invalid-request"
    if not isinstance(component_id, str) or not component_id.strip():
        component_id = COMPONENT_ID
    return request_id, component_id


def _invalid_request_result(request: Any, reason: str) -> ComponentResult:
    """Build a schema-valid failed result for malformed API/CLI input.

    Returns:
        A failed component result with valid identifiers and no artifacts.
    """
    request_id, component_id = _safe_result_identity(request)
    return ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status="failed",
        reason=f"invalid-request: {reason}",
    )


def _source_ref_shape_errors(index: int, ref: Any) -> list[str]:
    """Return shape errors for one manually constructed source reference."""
    if not isinstance(ref, SourceRef):
        return [f"sources[{index}] must be a SourceRef"]
    errors: list[str] = []
    if not isinstance(ref.artifact_id, str) or not ref.artifact_id.strip():
        errors.append(f"sources[{index}].artifact_id must be a non-empty string")
    if not isinstance(ref.uri, str) or not ref.uri.strip():
        errors.append(f"sources[{index}].uri must be a non-empty string")
    if not isinstance(ref.format, str) or not ref.format.strip():
        errors.append(f"sources[{index}].format must be a non-empty string")
    return errors


def _request_shape_errors(request: ComponentRequest) -> list[str]:
    """Return stable shape errors for a manually constructed API request.

    Returns:
        Shape errors, or an empty list for a structurally valid request.
    """

    errors: list[str] = []
    if not isinstance(request.request_id, str) or not request.request_id.strip():
        errors.append("request_id must be a non-empty string")
    if not isinstance(request.component_id, str) or not request.component_id.strip():
        errors.append("component_id must be a non-empty string")
    if not isinstance(request.output_directory, str) or not request.output_directory.strip():
        errors.append("output_directory must be a non-empty string")
    if not isinstance(request.config, dict):
        errors.append("config must be an object")
    if not isinstance(request.sources, (tuple, list)):
        errors.append("sources must be an array")
    else:
        for index, ref in enumerate(request.sources):
            errors.extend(_source_ref_shape_errors(index, ref))
    if not isinstance(request.required_capabilities, (tuple, list)) or any(
        not isinstance(capability, str) or not capability.strip()
        for capability in request.required_capabilities
    ):
        errors.append("required_capabilities must be an array of non-empty strings")
    return errors


def _normalize_request(
    request: ComponentRequest | Mapping[str, Any] | Any,
) -> ComponentRequest | ComponentResult:
    """Normalize API input and convert malformed shapes into a stable result.

    Returns:
        A normalized request, or a failed result for malformed input.
    """
    if isinstance(request, ComponentRequest):
        errors = _request_shape_errors(request)
        if errors:
            return _invalid_request_result(request, "; ".join(errors))
        return request
    if isinstance(request, Mapping):
        try:
            normalized = component_request_from_dict(request)
        except (ReviewContractsValidationError, TypeError, ValueError) as error:
            reason = (
                "; ".join(error.errors)
                if isinstance(error, ReviewContractsValidationError)
                else str(error)
            )
            return _invalid_request_result(request, reason)
        return normalized
    return _invalid_request_result(request, "expected a component request object")


def _availability_gate(request: ComponentRequest) -> ComponentResult | None:
    """Reject unsupported components, capabilities, and versions.

    Returns:
        An unavailable result, or ``None`` when invocation may proceed.
    """
    if request.component_id != COMPONENT_ID:
        return _unavailable(request, f"unsupported component: {request.component_id}")
    missing = _unsupported_capabilities(request)
    if missing:
        return _unavailable(request, f"missing capabilities: {', '.join(sorted(missing))}")
    required_version = request.config.get("required_component_version")
    if required_version is not None and _major(str(required_version)) != _major(COMPONENT_VERSION):
        return _unavailable(
            request,
            f"incompatible-required-version: {required_version!r}; "
            f"this component implements {COMPONENT_VERSION}",
        )
    return None


def _is_strict_relative_path(value: Any) -> bool:
    """Return whether a user-supplied path is relative and traversal-free."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":
        return False
    return ".." not in normalized.split("/")


def _resolve_source_file(source_base: Path, uri: str) -> Path | None:
    """Resolve a source URI under the base without allowing escapes.

    Returns:
        The resolved path, or ``None`` when it escapes the base.
    """
    if not _is_strict_relative_path(uri):
        return None
    try:
        resolved_base = source_base.resolve(strict=False)
        resolved = (resolved_base / uri).resolve(strict=False)
        resolved.relative_to(resolved_base)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _load_trace_source(
    artifact_id: str, path: Path
) -> tuple[SimulationTraceExport | None, str | None, str | None]:
    """Validate one trace export through its canonical owner.

    Returns:
        Tuple of (trace, error, observed byte digest); exactly one of trace or
        error is meaningful.
    """
    try:
        raw = path.read_bytes()
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise SimulationTraceExportValidationError(["expected a mapping payload"], source=path)
        return simulation_trace_export_from_dict(payload, source=path), None, observed_sha256
    except (
        SimulationTraceExportValidationError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        return None, f"corrupt-source: {artifact_id}: {error}", None


def _source_declarations(ref: SourceRef) -> dict[str, Any]:
    """Collect optional future contract declarations without widening the owner.

    Returns:
        Non-empty declaration fields exposed by the current or a compatible
        future SourceRef contract.
    """
    declarations: dict[str, Any] = {}
    for field_name in (
        "schema",
        "sha256",
        "source_commit",
        "config_identity",
        "units",
        "coordinate_frame",
    ):
        if hasattr(ref, field_name):
            value = getattr(ref, field_name)
            if value not in (None, ""):
                declarations[field_name] = value
    return declarations


def _verify_sha256_declaration(
    declared_sha256: Any, observed_sha256: str
) -> tuple[str, str | None]:
    """Verify an optional source byte declaration.

    Returns:
        A declaration status and an optional stable failure reason.
    """
    if declared_sha256 is None:
        return "not-provided", None
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in declared_sha256)
    ):
        return "invalid", "source-integrity-declaration-invalid: sha256 must be 64 hex characters"
    if declared_sha256.lower() != observed_sha256:
        return "mismatch", "source-integrity-mismatch: declared sha256 does not match source bytes"
    return "verified", None


def _verify_metadata_declarations(
    declarations: Mapping[str, Any], trace: SimulationTraceExport
) -> tuple[dict[str, str], list[str]]:
    """Verify optional declarations with a direct canonical trace comparison.

    Returns:
        Declaration statuses and stable failures for mismatches or unverifiable
        fields.
    """
    statuses: dict[str, str] = {}
    failures: list[str] = []
    if "schema" in declarations:
        statuses["schema"] = (
            "verified" if declarations["schema"] == trace.schema_version else "mismatch"
        )
        if statuses["schema"] != "verified":
            failures.append("source-schema-declaration-mismatch")
    if "coordinate_frame" in declarations:
        statuses["coordinate_frame"] = (
            "verified" if declarations["coordinate_frame"] == trace.coordinate_frame else "mismatch"
        )
        if statuses["coordinate_frame"] != "verified":
            failures.append("source-coordinate-frame-mismatch")
    if "units" in declarations:
        declared_units = declarations["units"]
        statuses["units"] = (
            "verified"
            if isinstance(declared_units, Mapping) and dict(declared_units) == trace.units
            else "unverified"
        )
        if statuses["units"] != "verified":
            failures.append("source-unverified-declaration: units")
    for field_name in ("source_commit", "config_identity"):
        if field_name in declarations:
            statuses[field_name] = "unverified"
            failures.append(f"source-unverified-declaration: {field_name}")
    return statuses, failures


def _verify_source_provenance(
    ref: SourceRef,
    trace: SimulationTraceExport,
    observed_sha256: str,
) -> tuple[dict[str, Any], str | None]:
    """Build explicit source provenance and verify declarations when available.

    Returns:
        A source provenance record and an optional stable failure reason.
    """
    declarations = _source_declarations(ref)
    declaration_status: dict[str, str] = {
        "format": "verified" if ref.format == trace.schema_version else "mismatch"
    }
    failures: list[str] = []
    if ref.format != trace.schema_version:
        failures.append(
            f"source-schema-mismatch: declared format {ref.format!r} != "
            f"loaded schema {trace.schema_version!r}"
        )

    sha_status, sha_failure = _verify_sha256_declaration(
        declarations.get("sha256"), observed_sha256
    )
    declaration_status["sha256"] = sha_status
    integrity_status = "verified" if sha_status == "verified" else "observed-only"
    if sha_failure is not None:
        failures.append(sha_failure)
        integrity_status = "mismatch" if sha_status == "mismatch" else "unverified"
    metadata_status, metadata_failures = _verify_metadata_declarations(declarations, trace)
    declaration_status.update(metadata_status)
    failures.extend(metadata_failures)

    source_record = {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "format": ref.format,
        "observed_sha256": observed_sha256,
        "declared": declarations,
        "declaration_status": declaration_status,
        "integrity_status": integrity_status,
        "identity_status": "embedded-observed",
        "trace": {
            "schema_version": trace.schema_version,
            "trace_id": trace.trace_id,
            "source": asdict(trace.source),
            "evidence_boundary": trace.evidence_boundary,
            "coordinate_frame": trace.coordinate_frame,
            "units": dict(trace.units),
        },
    }
    return source_record, "; ".join(failures) if failures else None


def _collect_traces(
    request: ComponentRequest, sources_root: Path
) -> tuple[
    list[tuple[Any, str, dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    ComponentResult | None,
]:
    """Load and validate every trace source, diagnosing skipped ones.

    Returns:
        Tuple of ((trace, artifact id, provenance) pairs, source records,
        diagnostics, terminal result).
        The terminal result is set when no trace is usable.
    """
    pairs: list[tuple[Any, str, dict[str, Any]]] = []
    source_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for ref in request.sources:
        source_file = _resolve_source_file(sources_root, ref.uri)
        if source_file is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": "unsafe-source-uri"})
            continue
        if ref.format != FORMAT_TRACE_EXPORT:
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason": f"unsupported-evidence-format: {ref.format}",
                }
            )
            continue
        trace, error, observed_sha256 = _load_trace_source(ref.artifact_id, source_file)
        if error is not None or trace is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": error})
            continue
        if observed_sha256 is None:
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason": "corrupt-source: missing observed byte digest",
                }
            )
            continue
        if not trace.frames:
            diagnostics.append(
                {"artifact_id": ref.artifact_id, "reason": "empty trace has no scenes"}
            )
            continue
        source_record, provenance_error = _verify_source_provenance(ref, trace, observed_sha256)
        if provenance_error is not None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": provenance_error})
            continue
        source_records.append(source_record)
        pairs.append((trace, ref.artifact_id, source_record))
    if not request.sources:
        return (
            pairs,
            source_records,
            diagnostics,
            _failed(request, "missing-evidence: request carries no sources"),
        )
    if not pairs:
        return (
            pairs,
            source_records,
            diagnostics,
            ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                diagnostics=tuple(diagnostics),
                reason="unsupported-evidence: no source could be rendered",
            ),
        )
    return pairs, source_records, diagnostics, None


def run(
    request: ComponentRequest | Mapping[str, Any] | Any,
    *,
    base: Path | None = None,
    source_base: Path | None = None,
) -> ComponentResult:
    """Render numbered scene frames over validated trace exports.

    Args:
        request: Validated component request, or a JSON-like mapping that is
            normalized through the shared request contract.
        base: Base directory the request output directory resolves under.
        source_base: Base directory source URIs resolve under (CLI: the request
            file's directory). Defaults to the current working directory.

    Returns:
        Component result: ``complete`` with scene/source-map/descriptor
        artifacts, ``unavailable`` for unsupported components, capabilities,
        versions, or evidence, or ``failed`` for corrupt inputs, missing
        geometry, and output collisions.
    """
    normalized_request = _normalize_request(request)
    if isinstance(normalized_request, ComponentResult):
        return normalized_request
    request = normalized_request
    root = base if base is not None else Path.cwd()
    sources_root = source_base if source_base is not None else Path.cwd()
    gate = _availability_gate(request)
    if gate is not None:
        return gate
    if not isinstance(request.config, dict):
        return _failed(request, "corrupt-config: request config must be a mapping")
    scene, scene_errors = _validate_scene(request.config.get("scene"))
    if scene_errors or scene is None:
        return _failed(request, "; ".join(scene_errors))
    pairs, source_records, diagnostics, terminal = _collect_traces(request, sources_root)
    if terminal is not None:
        return terminal
    output_dir, path_error = _resolve_output_directory(root, request.output_directory)
    if path_error is not None or output_dir is None:
        return _failed(
            request, path_error or "unsafe-output-path: output directory must resolve within base"
        )
    if output_dir.exists():
        return _failed(
            request,
            f"output-collision: output directory already exists: {request.output_directory}",
        )
    try:
        artifacts, source_map, _descriptor_digest = _stage_outputs(
            output_dir, request, scene, pairs
        )
    except (OSError, ValueError, IndexError) as error:
        return _failed(request, f"output-write-failed: {error}", tuple(diagnostics))
    except Exception as error:  # noqa: BLE001 - rendering backend failures fail closed
        return _failed(request, f"render-failed: {error}", tuple(diagnostics))
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="complete" if not diagnostics else "partial",
        artifacts=tuple(artifacts),
        diagnostics=tuple(diagnostics),
        provenance={
            "output_directory": request.output_directory,
            "sourcemap_sha256": source_map["sourcemap_sha256"],
            "component_version": COMPONENT_VERSION,
            "source_artifacts": source_records,
            "source_identity_status": "embedded-observed",
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the review-scene component.

    Returns:
        Argument parser with input/config/output/base options.
    """
    parser = argparse.ArgumentParser(description="Render SREV-09 recorded scene frames.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-scene component.

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result = _invalid_request_result(None, f"cannot read request: {error}")
        print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201
        return 1
    if not isinstance(payload, dict):
        result = _invalid_request_result(payload, "request document must be a JSON object")
        print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201
        return 1
    request_config = payload.get("config", {})
    if not isinstance(request_config, dict):
        result = _invalid_request_result(payload, "request config must be a JSON object")
        print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201
        return 1
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            result = _invalid_request_result(payload, f"cannot read config: {error}")
            print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201
            return 1
        if not isinstance(config, dict):
            result = _invalid_request_result(payload, "config document must be a JSON object")
            print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201
            return 1
        payload = {**payload, "config": {**request_config, **config}}
    payload = {**payload, "output_directory": args.output}
    try:
        request = component_request_from_dict(payload, source=args.input)
    except (ReviewContractsValidationError, TypeError, ValueError) as error:
        reason = (
            "; ".join(error.errors)
            if isinstance(error, ReviewContractsValidationError)
            else str(error)
        )
        result = _invalid_request_result(payload, reason)
        print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201
        return 1
    result = run(
        request,
        base=Path(args.base) if args.base is not None else None,
        source_base=input_path.parent,
    )
    print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


def _result_payload(result: ComponentResult) -> dict[str, Any]:
    """Serialize a result with the required shared component-result version.

    Returns:
        JSON-ready component-result.v1 envelope.
    """
    payload = asdict(result)
    payload["artifacts"] = list(payload["artifacts"])
    payload["diagnostics"] = list(payload["diagnostics"])
    return {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **payload}


if __name__ == "__main__":
    raise SystemExit(main())
