"""Component registry and extension development kit (SREV-29, issue #9290).

This module owns the ``srev29-review-registry`` component surface: discovery
of installed trusted component descriptors through the
``robot_sf.scenario_review`` Python entry-point group, validated invocation of
one discovered component per request, and a reusable conformance battery
(:func:`check_component_conformance`) that other leaves can run against their
own components.

Trust model (fail-closed):

* entry-point targets must match ``module:attribute`` shape and live under an
  allowlisted module prefix; anything else is recorded as untrusted and never
  imported;
* descriptor payloads must validate against ``component-descriptor.v1``;
* two entries claiming one ``component_id`` fail the whole registry;
* required capabilities and input versions are checked before invocation;
* missing optional imports disable only the affected component;
* artifacts can never nominate imports or shell execution: only relative,
  traversal-free artifact paths inside the requested output directory are
  accepted.

Evidence boundary: diagnostic tooling only. Discovery, invocation, and
conformance reports describe installed fixture components and their
deterministic smoke behavior; they admit no scientific claim, benchmark
result, or planner/simulator behavior change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.errors import RobotSfError
from robot_sf.render.helper_catalog import deterministic_seed_from_name

COMPONENT_ID = "srev29-review-registry"
COMPONENT_VERSION = "1.0.0"
ENTRY_POINT_GROUP = "robot_sf.scenario_review"

REGISTRY_INDEX_SCHEMA_VERSION = "registry-index.v1"
CONFORMANCE_REPORT_SCHEMA_VERSION = "conformance-report.v1"

SUPPORTED_INPUT_VERSIONS = (COMPONENT_REQUEST_SCHEMA_VERSION,)
REQUIRED_CAPABILITIES = ("bounded-execution",)
OUTPUT_TYPES = (
    REGISTRY_INDEX_SCHEMA_VERSION,
    CONFORMANCE_REPORT_SCHEMA_VERSION,
)

ALLOWED_ENTRY_POINT_PREFIXES = (
    "examples.scenario_review.components.",
    "robot_sf.analysis_workbench.",
    "robot_sf.render.",
)
_ENTRY_POINT_VALUE_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
_EXECUTE_MODES = ("execute", "discover", "conformance")


class ReviewRegistryError(RobotSfError, ValueError):
    """Raised when registry input, discovery state, or config is unusable."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable registry error."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class RegistryConfig:
    """Validated registry execution config (closed allowlist)."""

    mode: str = "execute"
    target_component_id: str | None = None
    target_config: dict[str, Any] = field(default_factory=dict)
    conformance_probe: dict[str, Any] = field(default_factory=dict)
    required_component_version: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveredComponent:
    """One trusted, importable component entry."""

    component_id: str
    entry_point_name: str
    descriptor: ComponentDescriptor
    run_callable: Callable[..., ComponentResult]


@dataclass
class ConformanceCheck:
    """Outcome of one conformance battery check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConformanceReport:
    """Full conformance battery outcome for one component."""

    component_id: str
    passed: bool
    checks: list[ConformanceCheck] = field(default_factory=list)


def descriptor() -> dict[str, Any]:
    """Describe the review-registry component and its capability contract.

    Returns:
        Component descriptor document honoring ``component-descriptor.v1``.
    """
    return {
        "schema_version": "component-descriptor.v1",
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "supported_input_versions": list(SUPPORTED_INPUT_VERSIONS),
        "required_capabilities": list(REQUIRED_CAPABILITIES),
        "optional_capabilities": [],
        "output_types": list(OUTPUT_TYPES),
    }


def _validated_target_fields(
    raw: dict[str, Any], mode: str, errors: list[str]
) -> tuple[str | None, dict[str, Any], dict[str, Any], str | None]:
    """Validate target-related config fields.

    Returns:
        Tuple of (target ID or None, target config, conformance probe,
        required version) with safe defaults; errors are recorded, not raised.
    """
    target_component_id = raw.get("target_component_id")
    if mode in ("execute", "conformance"):
        if not isinstance(target_component_id, str) or not target_component_id.strip():
            errors.append(f"mode {mode!r} requires a non-empty target_component_id")
    elif target_component_id is not None and not isinstance(target_component_id, str):
        errors.append("target_component_id must be a string")
    target_config = raw.get("target_config", {})
    if not isinstance(target_config, dict):
        errors.append("target_config must be a mapping")
        target_config = {}
    conformance_probe = raw.get("conformance_probe", {})
    if not isinstance(conformance_probe, dict):
        errors.append("conformance_probe must be a mapping")
        conformance_probe = {}
    required_version = raw.get("required_component_version")
    if required_version is not None and not isinstance(required_version, str):
        errors.append("required_component_version must be a string")
        required_version = None
    normalized_target = (
        str(target_component_id).strip()
        if isinstance(target_component_id, str) and target_component_id.strip()
        else None
    )
    return normalized_target, dict(target_config), dict(conformance_probe), required_version


def validate_registry_config(raw: Any, *, source: Any = None) -> RegistryConfig:
    """Validate raw registry config against the closed allowlist.

    Args:
        raw: Raw config mapping from the component request.
        source: Optional source label for error messages.

    Returns:
        Validated registry config with defaults applied.

    Raises:
        ReviewRegistryError: For unknown keys or invalid values.
    """
    if not isinstance(raw, dict):
        raise ReviewRegistryError(["config must be a mapping"], source=source)
    allowed = {
        "mode",
        "target_component_id",
        "target_config",
        "conformance_probe",
        "required_component_version",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ReviewRegistryError(
            [f"unknown config keys are rejected: {', '.join(unknown)}"], source=source
        )
    errors: list[str] = []
    mode = raw.get("mode", "execute")
    if mode not in _EXECUTE_MODES:
        errors.append(f"mode must be one of {list(_EXECUTE_MODES)}")
    normalized_target, target_config, conformance_probe, required_version = (
        _validated_target_fields(raw, str(mode), errors)
    )
    if errors:
        raise ReviewRegistryError(errors, source=source)
    return RegistryConfig(
        mode=str(mode),
        target_component_id=normalized_target,
        target_config=target_config,
        conformance_probe=conformance_probe,
        required_component_version=required_version,
    )


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _commit_provenance() -> dict[str, Any]:
    import subprocess  # noqa: PLC0415 - lazy: provenance only

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown"}
    digest = completed.stdout.strip()
    return {"commit": digest if digest else "unknown"}


def _installed_entry_points() -> list[Any]:
    """Return installed entry points for the registry group (monkeypatch seam)."""
    return list(importlib_metadata.entry_points(group=ENTRY_POINT_GROUP))


def _entry_point_target(entry: Any) -> tuple[str, str] | None:
    value = getattr(entry, "value", "")
    if not isinstance(value, str) or _ENTRY_POINT_VALUE_RE.fullmatch(value) is None:
        return None
    module_name, _, attribute = value.partition(":")
    if not any(module_name.startswith(prefix) for prefix in ALLOWED_ENTRY_POINT_PREFIXES):
        return None
    return module_name, attribute


def _load_descriptor(entry: Any) -> tuple[ComponentDescriptor | None, str]:
    """Import one entry point and validate its descriptor payload.

    Returns:
        Tuple of (validated descriptor or None, status detail).
    """
    target = _entry_point_target(entry)
    if target is None:
        return None, f"untrusted entry-point target: {getattr(entry, 'value', '')!r}"
    module_name, attribute = target
    try:
        module = __import__(module_name, fromlist=[attribute])
    except ImportError as error:
        return None, f"unavailable: optional import failed for {module_name}: {error}"
    try:
        candidate = getattr(module, attribute)
    except AttributeError:
        return None, f"unavailable: {module_name} has no attribute {attribute!r}"
    payload = candidate() if callable(candidate) else candidate
    if not isinstance(payload, dict):
        return None, f"unavailable: descriptor payload of {module_name} is not a mapping"
    try:
        return component_descriptor_from_dict(payload), "available"
    except ReviewContractsValidationError as error:
        return None, f"unavailable: invalid descriptor: {'; '.join(error.errors)}"


def _load_run_callable(entry: Any) -> tuple[Callable[..., ComponentResult] | None, str]:
    target = _entry_point_target(entry)
    if target is None:
        return None, "untrusted entry-point target"
    module_name, _ = target
    try:
        module = __import__(module_name, fromlist=["run"])
    except ImportError as error:
        return None, f"unavailable: optional import failed for {module_name}: {error}"
    runner = getattr(module, "run", None)
    if not callable(runner):
        return None, f"unavailable: {module_name} exposes no run(request) callable"
    return runner, "available"


def discover_components() -> tuple[dict[str, DiscoveredComponent], list[dict[str, Any]]]:
    """Discover installed trusted components without importing untrusted code.

    Returns:
        Tuple of (components by ID, index rows describing every entry point,
        including unavailable/untrusted ones with stable reasons).

    Raises:
        ReviewRegistryError: When two entries claim one component ID.
    """
    components: dict[str, DiscoveredComponent] = {}
    claimants: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for entry in _installed_entry_points():
        name = str(getattr(entry, "name", ""))
        descriptor_or_none, detail = _load_descriptor(entry)
        if descriptor_or_none is None:
            rows.append(
                {
                    "entry_point": name,
                    "status": "unavailable",
                    "reason": detail,
                }
            )
            continue
        component_id = descriptor_or_none.component_id
        if component_id in claimants:
            raise ReviewRegistryError(
                [
                    "conflicting_component_id: "
                    f"{component_id!r} claimed by {claimants[component_id]!r} and {name!r}"
                ]
            )
        claimants[component_id] = name
        runner, run_detail = _load_run_callable(entry)
        if runner is None:
            rows.append(
                {
                    "entry_point": name,
                    "component_id": component_id,
                    "status": "unavailable",
                    "reason": run_detail,
                }
            )
            continue
        components[component_id] = DiscoveredComponent(
            component_id=component_id,
            entry_point_name=name,
            descriptor=descriptor_or_none,
            run_callable=runner,
        )
        rows.append(
            {
                "entry_point": name,
                "component_id": component_id,
                "component_version": descriptor_or_none.component_version,
                "required_capabilities": list(descriptor_or_none.required_capabilities),
                "optional_capabilities": list(descriptor_or_none.optional_capabilities),
                "output_types": list(descriptor_or_none.output_types),
                "status": "available",
            }
        )
    return components, rows


def _artifact_within_output(output_dir: Path, uri: str) -> bool:
    """Return whether an artifact URI stays inside the output directory.

    Symlinks are resolved, so links pointing outside fail the check.
    """
    if Path(uri).is_absolute() or ".." in Path(uri).parts:
        return False
    try:
        (output_dir / uri).resolve(strict=False).relative_to(output_dir.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def check_component_conformance(
    *,
    component_id: str,
    descriptor_payload: dict[str, Any],
    run_callable: Callable[..., ComponentResult],
    probe_sources: list[dict[str, Any]],
    probe_config: dict[str, Any],
    base_dir: Path | str,
    case_name: str = "conformance",
    source_root: Path | str | None = None,
) -> ConformanceReport:
    """Run the isolated conformance battery against one component.

    The battery exercises descriptor validity, capability refusal, output
    collision refusal, determinism, envelope validity, and output-namespace
    containment using only fixture probe inputs under ``base_dir``. No check
    writes outside the case directories.

    Args:
        component_id: Expected component ID.
        descriptor_payload: Raw descriptor mapping under test.
        run_callable: The component ``run(request, base=...)`` callable.
        probe_sources: Fixture sources for probe requests.
        probe_config: Fixture config for probe requests.
        base_dir: Base directory probe outputs resolve under.
        case_name: Stable case prefix for probe output directories.
        source_root: Directory probe source URIs resolve under (defaults to
            ``base_dir``).

    Returns:
        Conformance report; ``passed`` is True only when every check passes.
    """
    battery = _Battery(
        component_id=component_id,
        descriptor_payload=dict(descriptor_payload),
        run_callable=run_callable,
        probe_sources=[dict(source) for source in probe_sources],
        probe_config=dict(probe_config),
        base=Path(base_dir),
        case_name=case_name,
        source_root=Path(source_root) if source_root is not None else None,
    )
    try:
        battery.check_descriptor()
        battery.stage_probe_inputs()
        battery.check_capability_refusal()
        first = battery.check_probe_completion()
        battery.check_collision_refusal()
        battery.check_determinism(first)
    except _BatteryHalted:
        return battery.report
    battery.report.passed = all(check.passed for check in battery.report.checks)
    return battery.report


class _BatteryHalted(Exception):
    """Internal signal: stop the battery after a failed check."""


@dataclass
class _Battery:
    """State for one conformance battery execution."""

    component_id: str
    descriptor_payload: dict[str, Any]
    run_callable: Callable[..., ComponentResult]
    probe_sources: list[dict[str, Any]]
    probe_config: dict[str, Any]
    base: Path
    case_name: str
    source_root: Path | None = None
    staged_sources: list[dict[str, Any]] = field(default_factory=list)
    report: ConformanceReport = field(init=False)

    def __post_init__(self) -> None:
        """Attach the result report for this battery run."""
        self.report = ConformanceReport(component_id=self.component_id, passed=False)

    def stage_probe_inputs(self) -> None:
        """Copy probe sources into the battery inputs directory.

        Staged sources replace the probe sources for all checks.

        Raises:
            _BatteryHalted: When a probe source is missing or unreadable.
        """

        inputs_dir = self.base / f"{self.case_name}-inputs"
        try:
            inputs_dir.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            self.record("probe_inputs_staged", False, f"cannot stage probe inputs: {error}")
            raise _BatteryHalted("inputs not staged") from error
        staged: list[dict[str, Any]] = []
        for index, source in enumerate(self.probe_sources):
            uri = source.get("uri", "")
            if not isinstance(uri, str) or not uri:
                self.record("probe_inputs_staged", False, "probe source uri must be set")
                raise _BatteryHalted("inputs not staged")
            candidate = (self.source_root if self.source_root is not None else self.base) / uri
            filename = f"{index}-{Path(uri).name}"
            try:
                (inputs_dir / filename).write_bytes(candidate.read_bytes())
            except OSError as error:
                self.record(
                    "probe_inputs_staged", False, f"cannot read probe source {uri!r}: {error}"
                )
                raise _BatteryHalted("inputs not staged") from error
            staged.append({**source, "uri": f"{self.case_name}-inputs/{filename}"})
        self.staged_sources = staged
        self.record("probe_inputs_staged", True)

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        """Append one battery outcome."""
        self.report.checks.append(ConformanceCheck(name=name, passed=passed, detail=detail))

    def probe_request(self, directory: str, **overrides: Any) -> ComponentRequest:
        """Build one probe request for the battery.

        Returns:
            Validated probe request for the given output directory.
        """
        payload: dict[str, Any] = {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": f"{self.case_name}-probe",
            "component_id": self.component_id,
            "sources": [dict(source) for source in self.staged_sources],
            "config": dict(self.probe_config),
            "output_directory": directory,
        }
        payload.update(overrides)
        return component_request_from_dict(payload)

    def invoke(self, check_name: str, directory: str, **overrides: Any) -> ComponentResult:
        """Invoke the component, recording a battery halt on raises.

        Returns:
            The component result for the probe request.

        Raises:
            _BatteryHalted: When the component raises instead of reporting.
        """
        try:
            return self.run_callable(self.probe_request(directory, **overrides), base=self.base)
        except Exception as error:
            detail = f"raised {type(error).__name__}: {error}"
            self.record(check_name, False, detail)
            raise _BatteryHalted(detail) from error

    def check_descriptor(self) -> ComponentDescriptor:
        """Validate the descriptor and supported input version.

        Returns:
            Validated component descriptor.

        Raises:
            _BatteryHalted: When descriptor or version checks fail.
        """
        try:
            validated = component_descriptor_from_dict(self.descriptor_payload)
        except ReviewContractsValidationError as error:
            self.record("descriptor_validates", False, "; ".join(error.errors))
            raise _BatteryHalted("descriptor invalid") from error
        self.record("descriptor_validates", True)
        if COMPONENT_REQUEST_SCHEMA_VERSION not in validated.supported_input_versions:
            self.record("version_supported", False, "component-request.v1 is not supported")
            raise _BatteryHalted("version unsupported")
        self.record("version_supported", True)
        return validated

    def check_capability_refusal(self) -> None:
        """Require unavailable for an unsatisfiable capability.

        Raises:
            _BatteryHalted: When the component does not refuse cleanly.
        """
        refused = self.invoke(
            "missing_capability_refused",
            f"{self.case_name}-capability",
            required_capabilities=["__never_supported_capability__"],
        )
        if refused.status == "unavailable" and "__never_supported_capability__" in refused.reason:
            self.record("missing_capability_refused", True)
            return
        self.record(
            "missing_capability_refused",
            False,
            f"expected unavailable, got {refused.status}: {refused.reason}",
        )
        raise _BatteryHalted("capability not refused")

    def check_probe_completion(self) -> ComponentResult:
        """Run the probe and validate envelope plus namespace containment.

        Returns:
            The completed probe result.

        Raises:
            _BatteryHalted: When the probe fails envelope or namespace checks.
        """
        first = self.invoke("probe_completes", f"{self.case_name}-run-a")
        if first.status != "complete":
            self.record(
                "probe_completes",
                False,
                f"probe ended {first.status}: {first.reason}; remaining checks skipped",
            )
            raise _BatteryHalted("probe did not complete")
        self.record("probe_completes", True)
        return self.check_completed_probe(first)

    def check_completed_probe(self, first: ComponentResult) -> ComponentResult:
        """Validate envelope and namespace of a completed probe.

        Returns:
            The validated probe result.

        Raises:
            _BatteryHalted: When envelope or namespace checks fail.
        """
        try:
            component_result_from_dict(
                {
                    "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
                    "request_id": first.request_id,
                    "component_id": first.component_id,
                    "status": first.status,
                    "artifacts": [dict(entry) for entry in first.artifacts],
                    "diagnostics": [dict(entry) for entry in first.diagnostics],
                    "provenance": dict(first.provenance),
                    "reason": first.reason,
                }
            )
        except ReviewContractsValidationError as error:
            self.record("envelope_valid", False, "; ".join(error.errors))
            raise _BatteryHalted("envelope invalid") from error
        self.record("envelope_valid", True)
        output_dir = self.base / f"{self.case_name}-run-a"
        if any(
            not _artifact_within_output(output_dir, str(entry.get("uri", "")))
            for entry in first.artifacts
        ):
            self.record("namespace_contained", False, "an artifact escapes the output directory")
            raise _BatteryHalted("namespace escape")
        self.record("namespace_contained", True)
        return first

    def check_collision_refusal(self) -> None:
        """Require failed/collision when rerunning into the same directory.

        Raises:
            _BatteryHalted: When the component does not refuse cleanly.
        """
        second = self.invoke("output_collision_refused", f"{self.case_name}-run-a")
        if second.status == "failed" and "collision" in second.reason:
            self.record("output_collision_refused", True)
            return
        self.record(
            "output_collision_refused",
            False,
            f"expected failed/collision, got {second.status}: {second.reason}",
        )
        raise _BatteryHalted("collision not refused")

    def check_determinism(self, first: ComponentResult) -> None:
        """Require identical artifact digests from a fresh rerun.

        Raises:
            _BatteryHalted: When the rerun diverges or fails.
        """
        rerun = self.invoke("deterministic_rerun", f"{self.case_name}-run-b")
        if rerun.status != "complete":
            self.record("deterministic_rerun", False, f"rerun ended {rerun.status}: {rerun.reason}")
            raise _BatteryHalted("rerun did not complete")
        first_digests = sorted(str(entry.get("sha256", "")) for entry in first.artifacts)
        rerun_digests = sorted(str(entry.get("sha256", "")) for entry in rerun.artifacts)
        if first_digests == rerun_digests and first_digests:
            self.record("deterministic_rerun", True)
            return
        self.record("deterministic_rerun", False, "artifact digests differ between runs")
        raise _BatteryHalted("rerun diverged")


def _final_result(
    request: ComponentRequest,
    *,
    status: str,
    reason: str,
    artifacts: tuple[dict[str, Any], ...] = (),
    diagnostics: tuple[dict[str, Any], ...] = (),
    provenance: dict[str, Any] | None = None,
) -> ComponentResult:
    payload = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "component_id": request.component_id,
        "status": status,
        "reason": reason,
    }
    if artifacts:
        payload["artifacts"] = [dict(entry) for entry in artifacts]
    if diagnostics:
        payload["diagnostics"] = [dict(entry) for entry in diagnostics]
    if provenance:
        payload["provenance"] = dict(provenance)
    try:
        return component_result_from_dict(payload)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason=f"internal_result_invalid: {'; '.join(error.errors)}",
        )


def _commit_environment() -> dict[str, Any]:
    provenance = _commit_provenance()
    provenance.update(
        {
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "python": sys.version.split()[0],
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    return provenance


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    supported = set(REQUIRED_CAPABILITIES)
    return [name for name in request.required_capabilities if name not in supported]


def _run_discover(
    request: ComponentRequest, output_dir: Path, provenance: dict[str, Any]
) -> ComponentResult:
    try:
        _, rows = discover_components()
    except ReviewRegistryError as error:
        return _final_result(request, status="failed", reason="; ".join(error.errors))
    index = {
        "schema_version": REGISTRY_INDEX_SCHEMA_VERSION,
        "request_id": request.request_id,
        "entry_point_group": ENTRY_POINT_GROUP,
        "components": rows,
        "provenance": provenance,
    }
    digest = _write_json(output_dir / "registry-index.json", index)
    prefix = Path(request.output_directory)
    return _final_result(
        request,
        status="complete",
        reason="component discovery completed",
        artifacts=(
            {
                "artifact_id": "registry-index.json",
                "uri": str(prefix / "registry-index.json"),
                "sha256": digest,
            },
        ),
        diagnostics=tuple(
            {
                "entry_point": row.get("entry_point", ""),
                "status": row.get("status", ""),
                "reason": row.get("reason", ""),
            }
            for row in rows
        ),
        provenance=provenance,
    )


def _run_conformance(
    request: ComponentRequest,
    config: RegistryConfig,
    output_dir: Path,
    provenance: dict[str, Any],
    root: Path,
) -> ComponentResult:
    try:
        components, _ = discover_components()
    except ReviewRegistryError as error:
        return _final_result(request, status="failed", reason="; ".join(error.errors))
    target_id = str(config.target_component_id)
    if target_id not in components:
        return _final_result(
            request,
            status="unavailable",
            reason=f"unknown component: {target_id}",
            provenance=provenance,
        )
    component = components[target_id]
    probe = config.conformance_probe
    probe_sources = probe.get("sources", [])
    probe_config = probe.get("config", {})
    if not isinstance(probe_sources, list) or not probe_sources:
        return _final_result(
            request,
            status="failed",
            reason="conformance mode requires conformance_probe.sources",
            provenance=provenance,
        )
    if not isinstance(probe_config, dict):
        return _final_result(
            request,
            status="failed",
            reason="conformance mode requires conformance_probe.config mapping",
            provenance=provenance,
        )
    case_seed = deterministic_seed_from_name(f"{request.request_id}/{target_id}")
    report = check_component_conformance(
        component_id=target_id,
        descriptor_payload={
            "schema_version": "component-descriptor.v1",
            "component_id": component.descriptor.component_id,
            "component_version": component.descriptor.component_version,
            "supported_input_versions": list(component.descriptor.supported_input_versions),
            "required_capabilities": list(component.descriptor.required_capabilities),
            "optional_capabilities": list(component.descriptor.optional_capabilities),
            "output_types": list(component.descriptor.output_types),
        },
        run_callable=component.run_callable,
        probe_sources=[dict(source) for source in probe_sources],
        probe_config=dict(probe_config),
        base_dir=output_dir / f"conformance-{case_seed}",
        case_name=f"conform-{case_seed}",
        source_root=root,
    )
    document = {
        "schema_version": CONFORMANCE_REPORT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "component_id": target_id,
        "passed": report.passed,
        "checks": [asdict(check) for check in report.checks],
        "provenance": provenance,
    }
    if not report.passed:
        return _final_result(
            request,
            status="failed",
            reason="conformance battery failed: "
            + "; ".join(
                f"{check.name}: {check.detail}" for check in report.checks if not check.passed
            ),
            diagnostics=tuple(
                {"check": check.name, "passed": check.passed, "detail": check.detail}
                for check in report.checks
            ),
            provenance=provenance,
        )
    digest = _write_json(output_dir / "conformance-report.json", document)
    prefix = Path(request.output_directory)
    return _final_result(
        request,
        status="complete",
        reason="conformance battery passed",
        artifacts=(
            {
                "artifact_id": "conformance-report.json",
                "uri": str(prefix / "conformance-report.json"),
                "sha256": digest,
            },
        ),
        diagnostics=tuple({"check": check.name, "passed": check.passed} for check in report.checks),
        provenance=provenance,
    )


def _run_execute(
    request: ComponentRequest,
    config: RegistryConfig,
    output_dir: Path,
    provenance: dict[str, Any],
    root: Path,
) -> ComponentResult:
    try:
        components, _ = discover_components()
    except ReviewRegistryError as error:
        return _final_result(request, status="failed", reason="; ".join(error.errors))
    target_id = str(config.target_component_id)
    if target_id not in components:
        return _final_result(
            request,
            status="unavailable",
            reason=f"unknown component: {target_id}",
            provenance=provenance,
        )
    component = components[target_id]
    if COMPONENT_REQUEST_SCHEMA_VERSION not in component.descriptor.supported_input_versions:
        return _final_result(
            request,
            status="unavailable",
            reason=f"incompatible_version: {target_id} does not support component-request.v1",
            provenance=provenance,
        )
    supported = set(component.descriptor.required_capabilities) | set(
        component.descriptor.optional_capabilities
    )
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return _final_result(
            request,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(missing))}",
            provenance=provenance,
        )
    target_output = f"{request.output_directory}/{target_id}"
    sub_request = ComponentRequest(
        request_id=request.request_id,
        component_id=target_id,
        sources=request.sources,
        output_directory=target_output,
        config=dict(config.target_config),
        required_capabilities=tuple(request.required_capabilities),
    )
    try:
        target_result = component.run_callable(sub_request, base=root)
    except Exception as error:  # noqa: BLE001 - registry boundary must report, never raise
        return _final_result(
            request,
            status="failed",
            reason=f"execution_error: {target_id} raised {type(error).__name__}: {error}",
            provenance=provenance,
        )
    if not isinstance(target_result, ComponentResult):
        return _final_result(
            request,
            status="failed",
            reason=f"execution_error: {target_id} returned a non-result payload",
            provenance=provenance,
        )
    routed_provenance = dict(target_result.provenance)
    routed_provenance["routed_by"] = COMPONENT_ID
    routed_provenance["registry_component_version"] = COMPONENT_VERSION
    if target_result.status != "complete":
        return _final_result(
            request,
            status=target_result.status,
            reason=f"{target_id}: {target_result.reason}",
            diagnostics=tuple(dict(entry) for entry in target_result.diagnostics),
            provenance={**provenance, "target": routed_provenance},
        )
    return _final_result(
        request,
        status="complete",
        reason=f"invoked {target_id} through the component registry",
        artifacts=tuple(dict(entry) for entry in target_result.artifacts),
        diagnostics=tuple(dict(entry) for entry in target_result.diagnostics),
        provenance={**provenance, "target": routed_provenance},
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute one registry request (execute, discover, or conformance).

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.

    Returns:
        Component result with artifacts (complete only), diagnostics, and
        provenance.
    """
    root = base if base is not None else Path.cwd()
    if request.component_id != COMPONENT_ID:
        return _final_result(
            request,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )
    missing = _unsupported_capabilities(request)
    if missing:
        return _final_result(
            request,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(missing))}",
        )
    try:
        config = validate_registry_config(request.config, source="config")
    except ReviewRegistryError as error:
        return _final_result(
            request, status="failed", reason=f"invalid_config: {'; '.join(error.errors)}"
        )
    if (
        config.required_component_version is not None
        and config.required_component_version != COMPONENT_VERSION
    ):
        return _final_result(
            request,
            status="unavailable",
            reason=(
                "incompatible_version: required "
                f"{config.required_component_version} != {COMPONENT_VERSION}"
            ),
        )
    output_dir = root / request.output_directory
    if output_dir.exists():
        return _final_result(
            request,
            status="failed",
            reason=f"output_collision: output already exists: {request.output_directory}",
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        return _final_result(
            request, status="failed", reason=f"unwritable output directory: {error}"
        )
    provenance = _commit_environment()
    provenance["request_id"] = request.request_id
    if config.mode == "discover":
        return _run_discover(request, output_dir, provenance)
    if config.mode == "conformance":
        return _run_conformance(request, config, output_dir, provenance, root)
    return _run_execute(request, config, output_dir, provenance, root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and invoke review components.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-registry component.

    Args:
        argv: Command-line arguments (defaults to process arguments).

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewRegistryError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewRegistryError([f"cannot read config: {error}"]) from error
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
