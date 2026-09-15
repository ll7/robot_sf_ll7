"""Experiment-recipe composer from diagnostic hypotheses (SREV-21, issue #9292).

This module is the review-hypotheses contract consumer: it turns one explicit
hypothesis (or one supported deterministic finding) plus source/config
references into a finite ``experiment-recipe.v1`` candidate set with an
unchanged control, measurement and evaluation rule.

It reuses the SREV-01 shared-contract surface
(:mod:`robot_sf.analysis_workbench.review_contracts`) for envelopes,
validation, and digests, and follows the survived/falsified/inconclusive
evaluation vocabulary of
:mod:`robot_sf.benchmark.counterfactual_pair` without importing simulator
behavior. Recipe creation launches nothing: no simulation, training, remote
scheduling, or scientific publication. Diagnostic tooling only — the recipe is
not campaign or evidence-admission authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
    experiment_recipe_from_dict,
)

COMPONENT_ID = "srev21-review-hypotheses"
COMPONENT_VERSION = "1.0.0"

TEMPLATE_SINGLE_PEDESTRIAN_SPEED = "single-pedestrian-speed"
TEMPLATE_SINGLE_PEDESTRIAN_START_DELAY = "single-pedestrian-start-delay"
SUPPORTED_TEMPLATES = (
    TEMPLATE_SINGLE_PEDESTRIAN_SPEED,
    TEMPLATE_SINGLE_PEDESTRIAN_START_DELAY,
)

DEFAULT_MAX_CANDIDATES = 3
DEFAULT_MAX_SIMULATOR_EXECUTIONS = 6
DEFAULT_MAX_ELAPSED_SECONDS = 600
DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES = 1
RESERVED_CONTROL_TREATMENT_PAIR = 2

EXPECTED_DIRECTIONS = ("increase", "decrease")

RECIPE_FILENAME = "experiment-recipe.json"
DESCRIPTOR_FILENAME = "component-descriptor.json"

_DESCRIBE_OUTPUT_TYPES = ("experiment-recipe.v1",)

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


def _validate_hypothesis_identity(raw: dict[str, Any], errors: list[str]) -> None:
    """Append identity-field errors (priority, pedestrian, terminal) to ``errors``."""
    priority = raw.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
        errors.append("corrupt-hypothesis: 'priority' must be a non-negative integer")
    pedestrian_id = raw.get("pedestrian_id", "ped-unknown")
    if not isinstance(pedestrian_id, str) or not pedestrian_id:
        errors.append("corrupt-hypothesis: 'pedestrian_id' must be a non-empty string")
    terminal = raw.get("terminal_condition", "episode_end_or_timeout")
    if not isinstance(terminal, str) or not terminal:
        errors.append("corrupt-hypothesis: 'terminal_condition' must be a non-empty string")


def _validate_factor_value(template: Any, factor: Any, errors: list[str]) -> None:
    """Reject invalid or non-finite candidate factor values before recipe creation."""
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        errors.append("corrupt-hypothesis: 'factor_value' must be a finite number")
        return
    if not math.isfinite(factor):
        errors.append("corrupt-hypothesis: 'factor_value' must be a finite number")
        return
    if template == TEMPLATE_SINGLE_PEDESTRIAN_SPEED and factor <= 0:
        errors.append("corrupt-hypothesis: pedestrian speed must be positive")
        return
    if template == TEMPLATE_SINGLE_PEDESTRIAN_START_DELAY and factor < 0:
        errors.append("corrupt-hypothesis: start delay must be non-negative")
        return
    if template not in SUPPORTED_TEMPLATES:
        return
    probe = float(factor)
    candidates = (
        (0.8 * probe, probe, 1.2 * probe)
        if template == TEMPLATE_SINGLE_PEDESTRIAN_SPEED
        else (max(0.0, probe - max(0.5, 0.25 * probe)), probe, probe + max(0.5, 0.25 * probe))
    )
    if not all(math.isfinite(value) for value in candidates):
        errors.append("corrupt-hypothesis: candidate factor values must be finite")


def _validate_hypothesis(config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the ``hypothesis`` mapping inside a request config.

    Returns:
        Tuple of (normalized hypothesis, errors). Exactly one side is meaningful:
        a normalized hypothesis when errors is empty, else the failure reasons.
    """
    raw = config.get("hypothesis")
    if not isinstance(raw, dict):
        return None, ["missing-hypothesis: config must carry a 'hypothesis' mapping"]
    errors: list[str] = []
    template = raw.get("template")
    if "template" not in raw:
        errors.append("missing-hypothesis-template: config hypothesis must name a 'template'")
    elif template not in SUPPORTED_TEMPLATES:
        errors.append(
            "unsupported-hypothesis-template: "
            f"{template!r}; supported templates are {', '.join(SUPPORTED_TEMPLATES)}; "
            "provide an explicit single-pedestrian speed or start-delay hypothesis"
        )
    factor = raw.get("factor_value")
    _validate_factor_value(template, factor, errors)
    direction = raw.get("expected_direction")
    if direction not in EXPECTED_DIRECTIONS:
        errors.append(
            f"corrupt-hypothesis: 'expected_direction' must be one of {EXPECTED_DIRECTIONS}"
        )
    priority = raw.get("priority", 0)
    _validate_hypothesis_identity(raw, errors)
    if not isinstance(config.get("source_config_identity", ""), str):
        errors.append("corrupt-source-config-identity: expected a string")
    if errors:
        return None, errors
    return {
        "template": str(template),
        "factor_value": float(factor),
        "expected_direction": str(direction),
        "priority": int(priority),
        "pedestrian_id": str(raw.get("pedestrian_id", "ped-unknown")),
        "terminal_condition": str(raw.get("terminal_condition", "episode_end_or_timeout")),
    }, []


def _candidate_factor_values(hypothesis: dict[str, Any]) -> list[float]:
    """Derive the finite deterministic candidate factor values for a hypothesis.

    Returns:
        Up to three candidate factor values around the hypothesis base value.
    """
    base = hypothesis["factor_value"]
    if hypothesis["template"] == TEMPLATE_SINGLE_PEDESTRIAN_SPEED:
        values = [0.8 * base, base, 1.2 * base]
    else:
        step = max(0.5, 0.25 * abs(base))
        values = [max(0.0, base - step), base, base + step]
    return [round(value, 6) for value in values[:DEFAULT_MAX_CANDIDATES]]


def _build_interventions(hypothesis: dict[str, Any]) -> list[dict[str, Any]]:
    """Build deterministically ordered candidate interventions (priority, stable ID).

    Returns:
        Candidate interventions sorted by (priority, intervention ID).
    """
    template = hypothesis["template"]
    factor_label = (
        "pedestrian_speed_mps"
        if template == TEMPLATE_SINGLE_PEDESTRIAN_SPEED
        else ("pedestrian_start_delay_s")
    )
    candidates = [
        {
            "intervention_id": f"{template}-candidate-{index:02d}",
            "factor": f"{factor_label}={value}",
            "priority": hypothesis["priority"],
        }
        for index, value in enumerate(_candidate_factor_values(hypothesis))
    ]
    return sorted(candidates, key=lambda item: (item["priority"], item["intervention_id"]))


def _build_recipe(request: ComponentRequest, hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Compose the ``experiment-recipe.v1`` document (no execution side effects).

    Returns:
        Validated recipe payload ready for schema validation and writing.
    """
    hypothesis_digest = _canonical_sha256(hypothesis)
    source_ids = [ref.artifact_id for ref in request.sources]
    source_refs = [
        {
            "artifact_id": ref.artifact_id,
            "uri": ref.uri,
            "format": ref.format,
        }
        for ref in request.sources
    ]
    config_identity = request.config.get("source_config_identity", "")
    template = hypothesis["template"]
    factor_label = "speed" if template == TEMPLATE_SINGLE_PEDESTRIAN_SPEED else "start delay"
    return {
        "schema_version": "experiment-recipe.v1",
        "recipe_id": f"{request.request_id}-recipe",
        "hypothesis": (
            f"Varying single-pedestrian {factor_label} for {hypothesis['pedestrian_id']} "
            f"changes the outcome metric in the {hypothesis['expected_direction']} direction "
            f"(template {template}, hypothesis sha256 {hypothesis_digest})."
        ),
        "source_identity": {
            "source_artifact_ids": source_ids,
            "source_refs": source_refs,
            "hypothesis_template": template,
            "hypothesis_sha256": hypothesis_digest,
            "note": "Source identities are copied from the request, not verified against source bytes.",
        },
        "interventions": _build_interventions(hypothesis),
        "control_conditions": {
            "mode": "unchanged-control",
            "source_config_identity": config_identity,
            "source_config_identity_status": (
                "provided_unverified" if config_identity else "unavailable"
            ),
            "note": (
                "Replay the preserved source configuration unchanged as the control; "
                "execution requires a verified source configuration identity."
            ),
        },
        "measurements": [
            {
                "name": "minimum_robot_pedestrian_separation",
                "units": "metres",
                "expected_direction": hypothesis["expected_direction"],
            },
            {
                "name": "mechanism_activation",
                "units": "boolean",
                "expected_direction": "increase",
            },
        ],
        "evaluation_rule": (
            "Evaluate each candidate once in (priority, intervention_id) order against the "
            "unchanged control; a hypothesis survives only when the mechanism activates, "
            "else it is falsified or inconclusive. Inconclusive candidates are recorded "
            "and never silently retried."
        ),
        "activation_checks": ["mechanism_activation_measured_before_verdict"],
        "fidelity_checks": ["verified_source_config_identity_required_before_execution"],
        "budget": {
            "max_candidate_interventions": DEFAULT_MAX_CANDIDATES,
            "max_simulator_executions": DEFAULT_MAX_SIMULATOR_EXECUTIONS,
            "max_elapsed_seconds": DEFAULT_MAX_ELAPSED_SECONDS,
            "max_concurrent_local_cpu_processes": DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES,
            "accounting": (
                "Controls, failed attempts, and retries consume budget; reserve at least "
                f"{RESERVED_CONTROL_TREATMENT_PAIR} executions for one control/treatment "
                "pair before starting."
            ),
        },
        "stop_rules": [
            "no unevaluated candidates remain",
            "remaining execution budget cannot cover a control/treatment pair",
            "invalid source identity",
            "failed control fidelity",
            "cancellation requested",
            f"terminal condition reached: {hypothesis['terminal_condition']}",
        ],
        "preservation_destination": request.output_directory,
        "admission_reference": "diagnostic-only; not campaign or evidence-admission authority",
    }


def _stage_recipe(
    output_dir: Path, request: ComponentRequest, recipe: dict[str, Any]
) -> tuple[str, str]:
    """Publish both artifacts together or leave the requested directory absent.

    Returns:
        SHA-256 digests for the recipe and descriptor bytes.
    """
    staging_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        recipe_digest = _write_json(staging_dir / RECIPE_FILENAME, recipe)
        descriptor_digest = _write_json(staging_dir / DESCRIPTOR_FILENAME, _DESCRIPTOR_DOC)
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {request.output_directory}")
        staging_dir.replace(output_dir)
        staging_dir = None
        return recipe_digest, descriptor_digest
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir)


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Compose one experiment recipe from a validated component request.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.

    Returns:
        Component result: ``complete`` with recipe/descriptor artifacts,
        ``unavailable`` for unsupported components, capabilities, templates, or
        versions, or ``failed`` for corrupt inputs and output collisions.
    """
    root = base if base is not None else Path.cwd()
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )
    missing = _unsupported_capabilities(request)
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(missing))}",
        )
    required_version = request.config.get("required_component_version")
    if required_version is not None and _major(str(required_version)) != _major(COMPONENT_VERSION):
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=(
                f"incompatible-required-version: {required_version!r}; "
                f"this component implements {COMPONENT_VERSION}"
            ),
        )
    hypothesis, errors = _validate_hypothesis(request.config)
    if errors or hypothesis is None:
        raw = request.config.get("hypothesis")
        if (
            isinstance(raw, dict)
            and "template" in raw
            and raw.get("template") not in SUPPORTED_TEMPLATES
        ):
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                reason=(
                    "unsupported-hypothesis-template: "
                    f"{raw.get('template')!r}; supported templates are "
                    f"{', '.join(SUPPORTED_TEMPLATES)}; provide an explicit "
                    "single-pedestrian speed or start-delay hypothesis"
                ),
            )
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(errors),
        )
    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason=(
                f"output-collision: output directory already exists: {request.output_directory}"
            ),
        )
    try:
        recipe = _build_recipe(request, hypothesis)
        experiment_recipe_from_dict(recipe)
        recipe_digest, descriptor_digest = _stage_recipe(output_dir, request, recipe)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(error.errors),
        )
    except OSError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason=f"output-write-failed: {error}",
        )
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="complete",
        artifacts=(
            {
                "artifact_id": RECIPE_FILENAME,
                "uri": str(Path(request.output_directory) / RECIPE_FILENAME),
                "sha256": recipe_digest,
            },
            {
                "artifact_id": DESCRIPTOR_FILENAME,
                "uri": str(Path(request.output_directory) / DESCRIPTOR_FILENAME),
                "sha256": descriptor_digest,
            },
        ),
        provenance={
            "output_directory": request.output_directory,
            "recipe_id": recipe["recipe_id"],
            "component_version": COMPONENT_VERSION,
            "hypothesis_sha256": recipe["source_identity"]["hypothesis_sha256"],
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the review-hypotheses component.

    Returns:
        Argument parser with input/config/output/base options.
    """
    parser = argparse.ArgumentParser(description="Compose SREV-21 experiment recipes.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-hypotheses component.

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
