"""Typed ``review-bundle.v1`` / ``visualization-spec.v1`` / component / recipe contracts.

This module owns the SREV-01 shared-contract surface: versioned interfaces,
JSON schemas, validation with stable reason codes, canonical content digests,
and the frozen ``run(request)`` invocation with inspect and capability-report
behavior. It reuses the canonical trace/timeline/annotation owners for
computation and never replaces them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from jsonschema import Draft202012Validator

from robot_sf.errors import RobotSfError

SCHEMA_DIR = Path(__file__).with_name("schemas")

REVIEW_BUNDLE_SCHEMA_VERSION = "review-bundle.v1"
VISUALIZATION_SPEC_SCHEMA_VERSION = "visualization-spec.v1"
COMPONENT_DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
COMPONENT_REQUEST_SCHEMA_VERSION = "component-request.v1"
COMPONENT_RESULT_SCHEMA_VERSION = "component-result.v1"
EXPERIMENT_RECIPE_SCHEMA_VERSION = "experiment-recipe.v1"

SCHEMA_FILES = {
    REVIEW_BUNDLE_SCHEMA_VERSION: "review_bundle.v1.json",
    VISUALIZATION_SPEC_SCHEMA_VERSION: "visualization_spec.v1.json",
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION: "component_descriptor.v1.json",
    COMPONENT_REQUEST_SCHEMA_VERSION: "component_request.v1.json",
    COMPONENT_RESULT_SCHEMA_VERSION: "component_result.v1.json",
    EXPERIMENT_RECIPE_SCHEMA_VERSION: "experiment_recipe.v1.json",
}

SUPPORTED_MAJOR_VERSIONS = {"v1"}
RESULT_STATUSES = ("complete", "partial", "unavailable", "failed", "cancelled")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_TEST_PRESET = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}


class ReviewContractsValidationError(RobotSfError, ValueError):
    """Raised when a review-contract payload fails schema or semantic validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@cache
def load_review_contracts_schema(version: str) -> dict[str, Any]:
    """Load one public review-contract JSON schema.

    Returns:
        Parsed JSON Schema document.


    Args:
        version: Schema version such as ``review-bundle.v1``.

    Returns:
        Parsed JSON Schema document.

    Raises:
        ReviewContractsValidationError: For an unknown schema version.
    """

    try:
        filename = SCHEMA_FILES[version]
    except KeyError:
        raise ReviewContractsValidationError([f"unknown schema version: {version}"]) from None
    return json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))


def _schema_errors(version: str, payload: Mapping[str, Any]) -> list[str]:
    validator = Draft202012Validator(load_review_contracts_schema(version))
    return [
        f"{_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _pointer(path: Any) -> str:
    return "/" + "/".join(str(token) for token in path)


def _require_schema(version: str, payload: Mapping[str, Any], *, source: Any = None) -> None:
    errors = _schema_errors(version, payload)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)


def _check_version_supported(version: str, *, source: Any = None) -> None:
    major = version.rsplit("v", 1)[-1].split(".", 1)[0]
    if f"v{major}" not in SUPPORTED_MAJOR_VERSIONS:
        raise ReviewContractsValidationError(
            [f"unsupported major version: {version}"], source=source
        )


def _check_finite(value: Any, *, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(value):
        errors.append(f"{path}: non-finite number is not strict-JSON safe")


def _check_no_traversal(value: str, *, path: str, errors: list[str]) -> None:
    pure = Path(value)
    if pure.is_absolute() or ".." in pure.parts:
        errors.append(f"{path}: path traversal or absolute path is rejected: {value}")


def _check_safe_artifact_id(value: Any, *, path: str, errors: list[str]) -> None:
    """Reject source identifiers that cannot be used as one output filename.

    ``artifact_id`` is an identifier rather than a relative path.  Rejecting
    both POSIX and Windows separators keeps the contract safe when a request
    created on one platform is consumed on another.
    """

    if not isinstance(value, str):
        return
    windows = PureWindowsPath(value)
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or windows.is_absolute()
        or bool(windows.drive)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        errors.append(f"{path}: unsafe artifact id for a filename: {value!r}")


def _check_sha256(value: Any, *, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected 64-hex SHA-256")


def _check_sha40(value: Any, *, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected 40-hex commit SHA")


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Identity and integrity pointer to one source artifact."""

    artifact_id: str
    uri: str
    format: str
    schema: str = ""
    sha256: str = ""
    source_commit: str = ""
    config_identity: str = ""
    units: str = ""
    coordinate_frame: str = ""


@dataclass(frozen=True, slots=True)
class ReviewBundle:
    """Index of episode/trace/geometry/media/diagnostic references."""

    bundle_id: str
    episodes: tuple[dict[str, Any], ...]
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VisualizationSpec:
    """Renderer-neutral visualization specification."""

    spec_id: str
    sources: tuple[dict[str, Any], ...]
    document: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Self-contained capability descriptor for one component."""

    component_id: str
    component_version: str
    supported_input_versions: tuple[str, ...]
    output_types: tuple[str, ...]
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentRequest:
    """Fixture request envelope for a component invocation."""

    request_id: str
    component_id: str
    sources: tuple[SourceRef, ...]
    output_directory: str
    config: dict[str, Any] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Result envelope for a component invocation."""

    request_id: str
    component_id: str
    status: str
    artifacts: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentRecipe:
    """Finite candidate-intervention recipe."""

    recipe_id: str
    document: dict[str, Any] = field(default_factory=dict)


def review_bundle_from_dict(payload: Mapping[str, Any], *, source: Any = None) -> ReviewBundle:
    """Validate and build a review bundle, checking semantic boundaries.

    Returns:
        Validated review bundle.
    """
    _require_schema(REVIEW_BUNDLE_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    seen: set[str] = set()
    for episode in payload["episodes"]:
        for ref in episode["references"]:
            artifact_id = ref["artifact_id"]
            if artifact_id in seen:
                errors.append(f"/episodes: duplicate scoped artifact id: {artifact_id}")
            seen.add(artifact_id)
            _check_sha256(ref["sha256"], path=f"/episodes/{artifact_id}/sha256", errors=errors)
            _check_sha40(
                ref["source_commit"], path=f"/episodes/{artifact_id}/source_commit", errors=errors
            )
            _check_no_traversal(ref["uri"], path=f"/episodes/{artifact_id}/uri", errors=errors)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    known = {"schema_version", "bundle_id", "episodes"}
    return ReviewBundle(
        bundle_id=str(payload["bundle_id"]),
        episodes=tuple(dict(episode) for episode in payload["episodes"]),
        extensions={k: payload[k] for k in payload if k not in known},
    )


def visualization_spec_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> VisualizationSpec:
    """Validate and build a visualization spec, checking time and unit boundaries.

    Returns:
        Validated visualization spec.
    """
    _require_schema(VISUALIZATION_SPEC_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    for index, interval in enumerate(payload.get("source_intervals", [])):
        start, end = interval["start_s"], interval["end_s"]
        _check_finite(start, path=f"/source_intervals/{index}/start_s", errors=errors)
        _check_finite(end, path=f"/source_intervals/{index}/end_s", errors=errors)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if not end > start:
                errors.append(f"/source_intervals/{index}: end_s must exceed start_s")
    units = payload.get("units", "")
    if units and units != "seconds/metres/radians" and not payload.get("unit_transforms"):
        errors.append("/units: non-canonical units require explicit unit_transforms")
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    known = {"schema_version", "spec_id", "sources"}
    return VisualizationSpec(
        spec_id=str(payload["spec_id"]),
        sources=tuple(dict(source_ref) for source_ref in payload["sources"]),
        document={k: payload[k] for k in payload if k not in known},
    )


def component_descriptor_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ComponentDescriptor:
    """Validate and build a component descriptor.

    Returns:
        Validated component descriptor.
    """
    _require_schema(COMPONENT_DESCRIPTOR_SCHEMA_VERSION, payload, source=source)
    return ComponentDescriptor(
        component_id=str(payload["component_id"]),
        component_version=str(payload["component_version"]),
        supported_input_versions=tuple(payload["supported_input_versions"]),
        output_types=tuple(payload["output_types"]),
        required_capabilities=tuple(payload.get("required_capabilities", [])),
        optional_capabilities=tuple(payload.get("optional_capabilities", [])),
    )


def component_request_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ComponentRequest:
    """Validate and build a component request, rejecting unsafe output paths.

    Returns:
        Validated component request.
    """
    _require_schema(COMPONENT_REQUEST_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    _check_no_traversal(str(payload["output_directory"]), path="/output_directory", errors=errors)
    seen_artifact_ids: set[str] = set()
    for index, ref in enumerate(payload["sources"]):
        artifact_id = ref["artifact_id"]
        _check_safe_artifact_id(artifact_id, path=f"/sources/{index}/artifact_id", errors=errors)
        if artifact_id in seen_artifact_ids:
            errors.append(f"/sources: duplicate scoped artifact id: {artifact_id}")
        seen_artifact_ids.add(artifact_id)
        _check_no_traversal(ref["uri"], path=f"/sources/{index}/uri", errors=errors)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return ComponentRequest(
        request_id=str(payload["request_id"]),
        component_id=str(payload["component_id"]),
        sources=tuple(
            SourceRef(
                artifact_id=str(ref["artifact_id"]),
                uri=str(ref["uri"]),
                format=str(ref["format"]),
                schema=str(ref.get("schema", "")),
                sha256=str(ref.get("sha256", "")),
                source_commit=str(ref.get("source_commit", "")),
                config_identity=str(ref.get("config_identity", "")),
                units=str(ref.get("units", "")),
                coordinate_frame=str(ref.get("coordinate_frame", "")),
            )
            for ref in payload["sources"]
        ),
        output_directory=str(payload["output_directory"]),
        config=dict(payload.get("config", {})),
        required_capabilities=tuple(payload.get("required_capabilities", [])),
    )


def component_result_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ComponentResult:
    """Validate and build a component result, enforcing status semantics.

    Returns:
        Validated component result.
    """
    _require_schema(COMPONENT_RESULT_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    if payload["status"] != "complete" and payload.get("artifacts"):
        errors.append(f"/status: {payload['status']} outputs cannot carry complete status")
    for index, artifact in enumerate(payload.get("artifacts", [])):
        _check_sha256(artifact["sha256"], path=f"/artifacts/{index}/sha256", errors=errors)
        _check_no_traversal(artifact["uri"], path=f"/artifacts/{index}/uri", errors=errors)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return ComponentResult(
        request_id=str(payload["request_id"]),
        component_id=str(payload["component_id"]),
        status=str(payload["status"]),
        artifacts=tuple(dict(artifact) for artifact in payload.get("artifacts", [])),
        diagnostics=tuple(dict(item) for item in payload.get("diagnostics", [])),
        provenance=dict(payload.get("provenance", {})),
        reason=str(payload.get("reason", "")),
    )


def experiment_recipe_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ExperimentRecipe:
    """Validate and build an experiment recipe.

    Returns:
        Validated experiment recipe.
    """
    _require_schema(EXPERIMENT_RECIPE_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    seen: set[str] = set()
    for intervention in payload["interventions"]:
        intervention_id = intervention["intervention_id"]
        if intervention_id in seen:
            errors.append(f"/interventions: duplicate scoped id: {intervention_id}")
        seen.add(intervention_id)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return ExperimentRecipe(recipe_id=str(payload["recipe_id"]), document=dict(payload))


def review_bundle_canonical_digest(bundle: ReviewBundle) -> str:
    """Bind logical bundle content with a stable digest (location-independent).

    Returns:
        Hex SHA-256 digest of the canonical encoding.
    """
    return _canonical_digest(
        {
            "schema_version": REVIEW_BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle.bundle_id,
            "episodes": [dict(episode) for episode in bundle.episodes],
        }
    )


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


INSPECT_COMPONENT_ID = "srev01-inspect"
CAPABILITY_COMPONENT_ID = "srev01-capability-report"

_INSPECT_DESCRIPTOR = ComponentDescriptor(
    component_id=INSPECT_COMPONENT_ID,
    component_version="1.0.0",
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("inspect-report.v1",),
)

_CAPABILITY_DESCRIPTOR = ComponentDescriptor(
    component_id=CAPABILITY_COMPONENT_ID,
    component_version="1.0.0",
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("capability-report.v1",),
    optional_capabilities=("video-frames",),
)


def capability_report() -> dict[str, Any]:
    """Describe the components this module executes offline without AI or Rerun.

    Returns:
        Capability report document.
    """
    return {
        "schema_version": "capability-report.v1",
        "components": [asdict(_INSPECT_DESCRIPTOR), asdict(_CAPABILITY_DESCRIPTOR)],
    }


def _resolve_output_dir(request: ComponentRequest, base: Path) -> Path:
    output_dir = base / request.output_directory
    if output_dir.exists():
        raise ReviewContractsValidationError(
            [f"/output_directory: output collision, already exists: {request.output_directory}"]
        )
    return output_dir


def _unsupported_capabilities(
    request: ComponentRequest, descriptor: ComponentDescriptor
) -> list[str]:
    supported = set(descriptor.required_capabilities) | set(descriptor.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute one inspect or capability-report request.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.

    Returns:
        Component result with artifacts, diagnostics, and provenance.
    """
    root = base if base is not None else Path.cwd()
    try:
        output_dir = _resolve_output_dir(request, root)
        if request.component_id == INSPECT_COMPONENT_ID:
            descriptor = _INSPECT_DESCRIPTOR
        elif request.component_id == CAPABILITY_COMPONENT_ID:
            descriptor = _CAPABILITY_DESCRIPTOR
        else:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                reason=f"unsupported component: {request.component_id}",
            )
        missing = _unsupported_capabilities(request, descriptor)
        if missing:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                reason=f"missing capabilities: {', '.join(sorted(missing))}",
            )
        if request.component_id == INSPECT_COMPONENT_ID:
            report = {
                "schema_version": "inspect-report.v1",
                "request_id": request.request_id,
                "sources": [asdict(ref) for ref in request.sources],
                "config": dict(request.config),
            }
            filename = "inspect-report.json"
        else:
            report = capability_report()
            filename = "capability-report.json"
        digest = _write_json(output_dir / filename, report)
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
            artifacts=(
                {
                    "artifact_id": filename,
                    "uri": str(Path(request.output_directory) / filename),
                    "sha256": digest,
                },
            ),
            provenance={"output_directory": request.output_directory},
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(error.errors),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute SREV-01 review-contract components.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for inspect and capability-report components.

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
