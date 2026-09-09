"""Fail-closed answerability checks for research campaign contracts.

The contract answers whether a planned campaign can resolve its declared
question. It does not run a campaign, admit evidence, or replace the existing
research-campaign manifest and figure-quality contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ANSWERABILITY_SCHEMA = "research_answerability.v1"
ANSWERABILITY_STATES = (
    "answerable",
    "diagnostic_only",
    "blocked_missing_producer",
    "blocked_underpowered",
    "blocked_analysis_contract",
    "blocked_noncomparable_rows",
    "blocked_artifact_plan",
    "invalid_contract",
)
_DECISION_VOCABULARY = {"continue", "stop", "inconclusive", "invalid"}
_REQUIRED_SECTIONS = ("question", "estimand", "producers", "analysis", "design", "artifacts")
_REQUIRED_TEXT_FIELDS = {
    "question": (
        "research_question",
        "bounded_claim",
        "negative_result_meaning",
    ),
    "estimand": (
        "primary",
        "reference_or_null",
        "decision_predicates",
        "minimally_important_effect",
    ),
    "analysis": (
        "analysis_unit",
        "resampling_unit",
        "command",
        "multiplicity",
        "sensitivity_plan",
        "dry_run_status",
        "comparability_status",
    ),
    "design": ("mode", "power_status", "budget", "sample_size"),
    "artifacts": (
        "raw_owner",
        "durable_path",
        "durability_status",
        "incomplete_policy",
    ),
}
_PRODUCER_TEXT_FIELDS = (
    "field",
    "producer",
    "source",
    "unit",
    "direction",
    "denominator",
    "pairing_key",
    "missingness_rule",
    "status",
    "execution_mode",
)
_VALID_EXECUTION_MODES = {"native", "adapter", "fallback", "degraded", "unavailable"}
_VALID_PRODUCER_STATUSES = {"available", "unavailable", "missing", "blocked"}
_VALID_DRY_RUN_STATUSES = {"passed", "not_required", "failed", "blocked", "unknown"}
_VALID_COMPARABILITY_STATUSES = {
    "passed",
    "not_required",
    "failed",
    "blocked",
    "unknown",
    "mismatched",
}

ADVERSARIAL_FALSIFICATION_PACKET_SCHEMA = "adversarial_falsification_answerability.v1"
ADVERSARIAL_FALSIFICATION_ANSWERABILITY_SCHEMA = ADVERSARIAL_FALSIFICATION_PACKET_SCHEMA
ADVERSARIAL_FALSIFICATION_PACKET_BASE_COMMIT = "c61b0f93683e1f9d2c83125f1d830b3d0162f3d1"
ADVERSARIAL_FALSIFICATION_PACKET_CLAIM_BOUNDARY = (
    "diagnostic-only preparation packet: source-bound adversarial falsification design; "
    "no simulator, planner, optimizer, campaign, benchmark, safety, or scientific claim"
)
ADVERSARIAL_FALSIFICATION_PACKET_OUTCOMES = (
    "result",
    "null",
    "inconclusive",
    "invalid",
    "unavailable",
    "blocked",
)
_PACKET_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "adversarial_falsification_answerability.v1.json"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKET_VARIABLE_ORDER = (
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
    "spawn_time_s",
    "pedestrian_speed_mps",
    "pedestrian_delay_s",
    "scenario_seed",
)
_PACKET_PREDICATE_ORDER = (
    "kinematic_reachability",
    "behavioral_consistency",
    "geometry_traffic",
    "simulator_validity",
)
_PACKET_OBJECTIVE_ORDER = (
    "feasibility",
    "kinematic_criticality",
    "controllability_risk",
    "diversity",
    "normalized_perturbation_cost",
    "candidate_digest",
)
_PACKET_OBJECTIVE_DIRECTIONS = (
    "maximize",
    "maximize",
    "maximize",
    "maximize",
    "minimize",
    "minimize",
)
_PACKET_VARIABLE_BINDINGS = {
    "start_x": {
        "actor": "robot",
        "field": "route_overrides.robot_routes[].waypoints[0][0]",
        "unit": "m",
        "kind": "continuous",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.start_x",
    },
    "start_y": {
        "actor": "robot",
        "field": "route_overrides.robot_routes[].waypoints[0][1]",
        "unit": "m",
        "kind": "continuous",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.start_y",
    },
    "goal_x": {
        "actor": "robot",
        "field": "route_overrides.robot_routes[].waypoints[1][0]",
        "unit": "m",
        "kind": "continuous",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.goal_x",
    },
    "goal_y": {
        "actor": "robot",
        "field": "route_overrides.robot_routes[].waypoints[1][1]",
        "unit": "m",
        "kind": "continuous",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.goal_y",
    },
    "spawn_time_s": {
        "actor": "pedestrian:p2",
        "field": "single_pedestrians[p2].start_delay_s",
        "unit": "s",
        "kind": "continuous",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.spawn_time_s",
    },
    "pedestrian_speed_mps": {
        "actor": "pedestrian:p2",
        "field": "single_pedestrians[p2].speed_m_s",
        "unit": "m/s",
        "kind": "continuous",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.pedestrian_speed_mps",
    },
    "pedestrian_delay_s": {
        "actor": "pedestrian:p2",
        "field": "wait_at[0].wait_s",
        "unit": "s",
        "kind": "continuous",
        "runtime_effective": False,
        "binding_status": "provenance_only",
        "source_path": "variables.pedestrian_delay_s",
    },
    "scenario_seed": {
        "actor": "scenario",
        "field": "seeds[] + simulation_config.route_spawn_seed",
        "unit": "seed",
        "kind": "integer",
        "runtime_effective": True,
        "binding_status": "runtime_effective",
        "source_path": "variables.scenario_seed",
    },
}


class AnswerabilityContractError(ValueError):
    """Raised when a research-answerability contract is structurally invalid."""


@dataclass(frozen=True)
class AnswerabilityResult:
    """Machine-readable answerability state and conservative reasons."""

    state: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result payload."""
        return {
            "schema_version": ANSWERABILITY_SCHEMA,
            "state": self.state,
            "decision_capable": self.state == "answerable",
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnswerabilityContractError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerabilityContractError(f"{field} must be a non-empty string")
    return value.strip()


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise AnswerabilityContractError(f"{field} must be a non-empty list")
    return value


def _validate_question(question: Mapping[str, Any]) -> None:
    vocabulary = _list(
        question.get("decision_vocabulary"),
        "answerability.question.decision_vocabulary",
    )
    if not all(isinstance(item, str) and item.strip() for item in vocabulary):
        raise AnswerabilityContractError(
            "answerability.question.decision_vocabulary must contain only non-empty strings"
        )
    unknown_vocabulary = set(vocabulary) - _DECISION_VOCABULARY
    if unknown_vocabulary:
        raise AnswerabilityContractError(
            "answerability.question.decision_vocabulary contains unsupported values: "
            f"{sorted(unknown_vocabulary)}"
        )
    for field in _REQUIRED_TEXT_FIELDS["question"]:
        _text(question.get(field), f"answerability.question.{field}")


def _validate_estimand(estimand: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["estimand"]:
        _text(estimand.get(field), f"answerability.estimand.{field}")


def _validate_producers(producers: list[Any]) -> None:
    for index, producer_value in enumerate(producers):
        producer = _mapping(producer_value, f"answerability.producers[{index}]")
        for field in _PRODUCER_TEXT_FIELDS:
            _text(producer.get(field), f"answerability.producers[{index}].{field}")
        if producer["status"] not in _VALID_PRODUCER_STATUSES:
            raise AnswerabilityContractError(
                f"answerability.producers[{index}].status must be one of "
                f"{sorted(_VALID_PRODUCER_STATUSES)}"
            )
        if producer["execution_mode"] not in _VALID_EXECUTION_MODES:
            raise AnswerabilityContractError(
                f"answerability.producers[{index}].execution_mode must be one of "
                f"{sorted(_VALID_EXECUTION_MODES)}"
            )
        if not isinstance(producer.get("required", True), bool):
            raise AnswerabilityContractError(
                f"answerability.producers[{index}].required must be a boolean"
            )


def _validate_analysis(analysis: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["analysis"]:
        _text(analysis.get(field), f"answerability.analysis.{field}")
    if analysis["dry_run_status"] not in _VALID_DRY_RUN_STATUSES:
        raise AnswerabilityContractError(
            "answerability.analysis.dry_run_status must be passed, not_required, failed, blocked, or unknown"
        )
    if analysis["comparability_status"] not in _VALID_COMPARABILITY_STATUSES:
        raise AnswerabilityContractError(
            "answerability.analysis.comparability_status must be passed, not_required, failed, blocked, unknown, or mismatched"
        )


def _validate_design(design: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["design"]:
        _text(design.get(field), f"answerability.design.{field}")
    if design["mode"] not in {"decision_capable", "diagnostic"}:
        raise AnswerabilityContractError(
            "answerability.design.mode must be decision_capable or diagnostic"
        )
    if design["power_status"] not in {"adequate", "not_required", "underpowered", "unknown"}:
        raise AnswerabilityContractError(
            "answerability.design.power_status must be adequate, not_required, underpowered, or unknown"
        )


def _validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TEXT_FIELDS["artifacts"]:
        _text(artifacts.get(field), f"answerability.artifacts.{field}")
    checksums = artifacts["checksums"]
    if not isinstance(checksums, list) or not all(
        isinstance(item, str) and item.strip() for item in checksums
    ):
        raise AnswerabilityContractError(
            "answerability.artifacts.checksums must be a non-empty list of strings"
        )
    if artifacts["durability_status"] not in {"ready", "planned", "missing", "blocked"}:
        raise AnswerabilityContractError(
            "answerability.artifacts.durability_status must be ready, planned, missing, or blocked"
        )


def validate_answerability_contract(contract: Mapping[str, Any]) -> None:
    """Validate the structural contract without interpreting campaign results."""
    if not isinstance(contract, Mapping):
        raise AnswerabilityContractError("answerability must be a mapping")
    if contract.get("schema_version") != ANSWERABILITY_SCHEMA:
        raise AnswerabilityContractError(
            f"answerability.schema_version must be {ANSWERABILITY_SCHEMA}"
        )
    for section in _REQUIRED_SECTIONS:
        if section == "producers":
            _validate_producers(_list(contract.get(section), f"answerability.{section}"))
        else:
            _mapping(contract.get(section), f"answerability.{section}")
    _validate_question(_mapping(contract["question"], "answerability.question"))
    _validate_estimand(_mapping(contract["estimand"], "answerability.estimand"))
    _validate_analysis(_mapping(contract["analysis"], "answerability.analysis"))
    _validate_design(_mapping(contract["design"], "answerability.design"))
    _validate_artifacts(_mapping(contract["artifacts"], "answerability.artifacts"))


def evaluate_answerability(contract: Mapping[str, Any]) -> AnswerabilityResult:
    """Return the most conservative state supported by *contract*.

    Structural defects return ``invalid_contract``. Semantic blockers are
    ordered from missing producers through artifact durability. A diagnostic
    design may be valid without being decision-capable; it returns
    ``diagnostic_only`` and never ``answerable``.
    """
    try:
        validate_answerability_contract(contract)
    except AnswerabilityContractError as exc:
        return AnswerabilityResult("invalid_contract", (str(exc),))

    producers = [dict(_mapping(value, "producer")) for value in contract["producers"]]
    required_producers = [producer for producer in producers if producer.get("required", True)]
    missing_producers = [
        producer["field"]
        for producer in required_producers
        if producer["status"] != "available"
        or producer["execution_mode"] in {"fallback", "degraded", "unavailable"}
    ]
    optional_unavailable = [
        producer["field"]
        for producer in producers
        if not producer.get("required", True) and producer["status"] == "unavailable"
    ]
    if missing_producers:
        return AnswerabilityResult(
            "blocked_missing_producer",
            (
                "required producers are missing, unavailable, blocked, or fallback/degraded: "
                + ", ".join(sorted(missing_producers)),
            ),
        )

    analysis = _mapping(contract["analysis"], "answerability.analysis")
    if analysis["dry_run_status"] not in {"passed", "not_required"}:
        return AnswerabilityResult(
            "blocked_analysis_contract",
            (f"analysis dry-run status is {analysis['dry_run_status']!r}",),
        )
    if analysis["comparability_status"] not in {"passed", "not_required"}:
        return AnswerabilityResult(
            "blocked_noncomparable_rows",
            (f"row comparability status is {analysis['comparability_status']!r}",),
        )

    design = _mapping(contract["design"], "answerability.design")
    if design["power_status"] == "underpowered":
        return AnswerabilityResult(
            "blocked_underpowered",
            ("declared executable budget is underpowered for the minimally important effect",),
        )
    if design["power_status"] == "unknown":
        return AnswerabilityResult(
            "blocked_underpowered",
            ("design power or diagnostic-budget classification is unknown",),
        )

    artifacts = _mapping(contract["artifacts"], "answerability.artifacts")
    durability_status = artifacts["durability_status"]
    if durability_status in {"missing", "blocked"}:
        return AnswerabilityResult(
            "blocked_artifact_plan",
            (f"durable evidence plan is {durability_status}",),
        )

    warnings = ()
    if optional_unavailable:
        warnings = (
            "optional unavailable producers remain explicit and cannot be interpreted as zero: "
            + ", ".join(sorted(optional_unavailable)),
        )
    if design["mode"] == "diagnostic" or durability_status == "planned":
        return AnswerabilityResult(
            "diagnostic_only",
            (
                "contract is executable only as a bounded diagnostic, not a decision-capable campaign",
            ),
            warnings,
        )
    return AnswerabilityResult("answerable", (), warnings)


def answerability_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the optional answerability section of a campaign manifest.

    Returns:
        JSON-safe answerability state, reasons, and warnings.
    """
    contract = manifest.get("answerability")
    if contract is None:
        return {
            "schema_version": ANSWERABILITY_SCHEMA,
            "state": "not_declared",
            "decision_capable": False,
            "reasons": ["manifest does not declare answerability.v1"],
            "warnings": [],
        }
    if not isinstance(contract, Mapping):
        return AnswerabilityResult(
            "invalid_contract", ("answerability must be a mapping",)
        ).as_dict()
    return evaluate_answerability(contract).as_dict()


class AdversarialFalsificationPacketError(ValueError):
    """Raised when the bounded adversarial answerability packet is not trustworthy."""


def _packet_canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a packet payload using the stable digest profile.

    Returns:
        UTF-8 encoded canonical JSON bytes.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_adversarial_falsification_packet_digest(
    packet: Mapping[str, Any],
) -> str:
    """Compute the packet self-digest, excluding only its stored digest field.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    if not isinstance(packet, Mapping):
        raise AdversarialFalsificationPacketError("packet must be a mapping")
    digest_payload = dict(packet)
    digest_payload.pop("self_digest", None)
    try:
        return hashlib.sha256(_packet_canonical_bytes(digest_payload)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise AdversarialFalsificationPacketError(
            f"packet cannot be canonically serialized for digest: {exc}"
        ) from exc


def load_adversarial_falsification_packet_schema() -> dict[str, Any]:
    """Load the committed JSON Schema for the v1 falsification packet.

    Returns:
        Parsed JSON Schema mapping.
    """
    try:
        schema = json.loads(_PACKET_SCHEMA_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdversarialFalsificationPacketError(
            f"cannot load adversarial falsification packet schema: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise AdversarialFalsificationPacketError("packet schema must be a JSON object")
    return schema


def _packet_schema_errors(packet: Mapping[str, Any]) -> list[str]:
    """Return deterministic JSON-Schema error messages for one packet."""
    validator = Draft202012Validator(load_adversarial_falsification_packet_schema())
    errors = sorted(
        validator.iter_errors(packet),
        key=lambda error: (tuple(str(part) for part in error.path), error.message),
    )
    return [
        "packet schema at "
        + (".".join(str(part) for part in error.path) or "<root>")
        + f": {error.message}"
        for error in errors
    ]


def _resolve_packet_source_path(value: object, *, repo_root: Path, field: str) -> Path:
    """Resolve a repository-relative packet input without allowing path escape.

    Returns:
        Resolved source file path.
    """
    if not isinstance(value, str) or not value.strip():
        raise AdversarialFalsificationPacketError(f"{field} must be a non-empty path")
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise AdversarialFalsificationPacketError(f"{field} must be repository-relative")
    root = repo_root.resolve()
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AdversarialFalsificationPacketError(f"{field} escapes the repository root") from exc
    if not resolved.is_file():
        raise AdversarialFalsificationPacketError(f"{field} does not resolve to a file: {value}")
    return resolved


def _packet_source_semantics(path: Path) -> dict[str, Any]:
    """Return the canonical parsed semantics of the existing #7340 search-space config."""
    try:
        raw_payload = yaml.safe_load(path.read_bytes()) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AdversarialFalsificationPacketError(
            f"cannot load source search-space config {path}: {exc}"
        ) from exc
    if not isinstance(raw_payload, Mapping):
        raise AdversarialFalsificationPacketError("source search-space config must be a mapping")
    if raw_payload.get("schema_version") != "adversarial-search-space.v1":
        raise AdversarialFalsificationPacketError(
            "source search-space schema_version must be adversarial-search-space.v1"
        )
    try:
        from robot_sf.adversarial.config import SearchSpaceConfig  # noqa: PLC0415

        parsed = SearchSpaceConfig.from_mapping(raw_payload)
    except (TypeError, ValueError) as exc:
        raise AdversarialFalsificationPacketError(
            f"source search-space config is not loader-valid: {exc}"
        ) from exc
    return {"schema_version": raw_payload["schema_version"], **parsed.to_json()}


def _validate_packet_sources(  # noqa: C901
    packet: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify source bytes and the parsed #7340 search-space binding.

    Returns:
        Canonical parsed search-space semantics for downstream binding checks.
    """
    source = packet["source"]
    inputs = source["inputs"]
    input_paths: dict[str, Path] = {}
    for input_id in ("search_space", "scenario_template", "map"):
        reference = inputs[input_id]
        path = _resolve_packet_source_path(
            reference["path"], repo_root=repo_root, field=f"source.inputs.{input_id}.path"
        )
        declared_digest = reference["sha256"]
        if not isinstance(declared_digest, str) or _SHA256_RE.fullmatch(declared_digest) is None:
            raise AdversarialFalsificationPacketError(
                f"source.inputs.{input_id}.sha256 must be a lowercase SHA-256 digest"
            )
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != declared_digest:
            raise AdversarialFalsificationPacketError(
                f"source.inputs.{input_id}.sha256 does not match source bytes"
            )
        input_paths[input_id] = path

    expected_semantics = _packet_source_semantics(input_paths["search_space"])
    declared_semantics = source["search_space"]["semantic"]
    if declared_semantics != expected_semantics:
        raise AdversarialFalsificationPacketError(
            "source.search_space.semantic does not equal the parsed #7340 search-space config"
        )

    try:
        template_payload = yaml.safe_load(input_paths["scenario_template"].read_bytes()) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AdversarialFalsificationPacketError(
            f"cannot load source scenario template: {exc}"
        ) from exc
    if not isinstance(template_payload, Mapping):
        raise AdversarialFalsificationPacketError("source scenario template must be a mapping")
    scenarios = template_payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or not isinstance(scenarios[0], Mapping):
        raise AdversarialFalsificationPacketError(
            "source scenario template must contain a first scenario mapping"
        )
    scenario = scenarios[0]
    template_binding = source["scenario_template"]
    if scenario.get("name") != template_binding["scenario_name"]:
        raise AdversarialFalsificationPacketError(
            "source.scenario_template.scenario_name does not match source bytes"
        )
    pedestrians = scenario.get("single_pedestrians")
    if not isinstance(pedestrians, list) or not any(
        isinstance(entry, Mapping) and entry.get("id") == template_binding["pedestrian_id"]
        for entry in pedestrians
    ):
        raise AdversarialFalsificationPacketError(
            "source scenario template does not contain the declared pedestrian binding"
        )
    map_file = scenario.get("map_file")
    if not isinstance(map_file, str) or not map_file.strip():
        raise AdversarialFalsificationPacketError("source scenario template map_file is missing")
    resolved_map = (input_paths["scenario_template"].parent / map_file).resolve()
    if resolved_map != input_paths["map"]:
        raise AdversarialFalsificationPacketError(
            "source.inputs.map.path does not match the scenario template map_file"
        )
    if template_binding["pedestrian_route_mode"] != expected_semantics["pedestrian"]["route_mode"]:
        raise AdversarialFalsificationPacketError(
            "source.scenario_template.pedestrian_route_mode does not match search-space semantics"
        )
    for index, owner in enumerate(source["owners"]):
        owner_path = _resolve_packet_source_path(
            owner["path"], repo_root=repo_root, field=f"source.owners[{index}].path"
        )
        declared_owner_digest = owner["sha256"]
        if (
            not isinstance(declared_owner_digest, str)
            or _SHA256_RE.fullmatch(declared_owner_digest) is None
        ):
            raise AdversarialFalsificationPacketError(
                f"source.owners[{index}].sha256 must be a lowercase SHA-256 digest"
            )
        actual_owner_digest = hashlib.sha256(owner_path.read_bytes()).hexdigest()
        if actual_owner_digest != declared_owner_digest:
            raise AdversarialFalsificationPacketError(
                f"source.owners[{index}].sha256 does not match source bytes"
            )
    return expected_semantics


def _validate_packet_variable_map(  # noqa: C901
    packet: Mapping[str, Any],
    *,
    search_space_semantics: Mapping[str, Any],
) -> None:
    """Check units, actor bindings, effectiveness, and bounds against #7340."""
    if tuple(packet["variable_order"]) != _PACKET_VARIABLE_ORDER:
        raise AdversarialFalsificationPacketError(
            "variable_order must preserve the frozen #7340 declaration order"
        )
    variable_map = packet["variable_map"]
    if set(variable_map) != set(_PACKET_VARIABLE_ORDER):
        raise AdversarialFalsificationPacketError(
            "variable_map keys must exactly match the frozen #7340 variables"
        )
    source_variables = search_space_semantics["variables"]
    for name in _PACKET_VARIABLE_ORDER:
        entry = variable_map[name]
        expected = _PACKET_VARIABLE_BINDINGS[name]
        for field in ("actor", "field", "unit", "kind", "binding_status", "source_path"):
            if entry[field] != expected[field]:
                raise AdversarialFalsificationPacketError(
                    f"variable_map.{name}.{field} does not match the approved binding"
                )
        if entry["runtime_effective"] != expected["runtime_effective"]:
            raise AdversarialFalsificationPacketError(
                f"variable_map.{name}.runtime_effective contradicts the approved binding"
            )
        if entry["bounds"] != source_variables[name]:
            raise AdversarialFalsificationPacketError(
                f"variable_map.{name}.bounds does not match the #7340 source bounds"
            )
        if (
            not isinstance(entry.get("effectiveness_note"), str)
            or not entry["effectiveness_note"].strip()
        ):
            raise AdversarialFalsificationPacketError(
                f"variable_map.{name}.effectiveness_note must be non-empty"
            )
    if search_space_semantics["pedestrian"].get("route_mode") == "template":
        delay = variable_map["pedestrian_delay_s"]
        if delay["runtime_effective"] or delay["binding_status"] != "provenance_only":
            raise AdversarialFalsificationPacketError(
                "pedestrian_delay_s must remain provenance-only in template mode"
            )
        note = delay["effectiveness_note"].lower()
        if "provenance-only" not in note or "runtime" not in note:
            raise AdversarialFalsificationPacketError(
                "pedestrian_delay_s must explain its missing runtime effectiveness"
            )


def _validate_packet_feasibility(packet: Mapping[str, Any]) -> None:
    """Keep the packet attached to the existing feasibility-first vocabulary."""
    from robot_sf.adversarial.feasibility_first import (  # noqa: PLC0415
        CHECK_NAMES,
        SCENARIO_FEASIBILITY_CONTRACT_VERSION,
    )

    feasibility = packet["feasibility"]
    if tuple(CHECK_NAMES) != _PACKET_PREDICATE_ORDER:
        raise AdversarialFalsificationPacketError(
            "feasibility_first.CHECK_NAMES no longer matches the v1 packet contract"
        )
    if feasibility["contract_version"] != SCENARIO_FEASIBILITY_CONTRACT_VERSION:
        raise AdversarialFalsificationPacketError(
            "feasibility.contract_version must use the existing scenario feasibility contract"
        )
    if tuple(feasibility["predicate_order"]) != tuple(CHECK_NAMES):
        raise AdversarialFalsificationPacketError(
            "feasibility.predicate_order must match feasibility_first.CHECK_NAMES"
        )
    if tuple(item["name"] for item in feasibility["predicates"]) != tuple(CHECK_NAMES):
        raise AdversarialFalsificationPacketError(
            "feasibility.predicates must match feasibility_first.CHECK_NAMES"
        )
    accounting = feasibility["rejection_accounting"]
    if (
        accounting["owner"]
        != "robot_sf.adversarial.feasibility_first.build_scenario_feasibility_ledger"
    ):
        raise AdversarialFalsificationPacketError(
            "feasibility.rejection_accounting must reuse the existing ledger owner"
        )
    if (
        not accounting["pre_simulation"]
        or not accounting["rejected_excluded_from_safety_denominator"]
    ):
        raise AdversarialFalsificationPacketError(
            "feasibility rejection accounting must remain pre-simulation and denominator-safe"
        )


def _validate_packet_design(packet: Mapping[str, Any]) -> None:  # noqa: C901, PLR0912
    """Check the declared lexicographic design and equal-budget arms."""
    objective = packet["objective"]
    objective_signature = tuple(
        (item["rank"], item["name"], item["direction"]) for item in objective["ordering"]
    )
    expected_objective_signature = tuple(
        (rank, name, direction)
        for rank, (name, direction) in enumerate(
            zip(_PACKET_OBJECTIVE_ORDER, _PACKET_OBJECTIVE_DIRECTIONS, strict=True),
            start=1,
        )
    )
    if objective_signature != expected_objective_signature:
        raise AdversarialFalsificationPacketError(
            "objective.ordering, ranks, and directions do not match the approved lexicographic order"
        )
    if objective["scalarization"] != "none" or not objective["deterministic_tie_break"]:
        raise AdversarialFalsificationPacketError(
            "objective must remain lexicographic with deterministic tie breaks"
        )

    seeds = packet["seed_policy"]
    search_seeds = seeds["search_seeds"]
    confirmation_seeds = seeds["confirmation_seeds"]
    if len(search_seeds) != 3 or len(confirmation_seeds) != 5:
        raise AdversarialFalsificationPacketError(
            "seed_policy must declare three search and five confirmation seeds"
        )
    if len(set(search_seeds)) != len(search_seeds) or len(set(confirmation_seeds)) != len(
        confirmation_seeds
    ):
        raise AdversarialFalsificationPacketError("seed_policy seed lists must be unique")
    if set(search_seeds) & set(confirmation_seeds):
        raise AdversarialFalsificationPacketError(
            "seed_policy search and confirmation seeds must be disjoint"
        )
    if seeds["candidate_seed_mode"] != "index_derived":
        raise AdversarialFalsificationPacketError(
            "seed_policy.candidate_seed_mode must reuse index_derived semantics"
        )

    budget = packet["budget"]
    per_seed = budget["candidate_budget_per_arm_per_seed"]
    expected_per_arm = per_seed * len(search_seeds)
    expected_all_arms = expected_per_arm * 3
    if budget["search_candidate_rows_per_arm"] != expected_per_arm:
        raise AdversarialFalsificationPacketError(
            "budget.search_candidate_rows_per_arm does not match seed accounting"
        )
    if budget["search_candidate_rows_all_arms"] != expected_all_arms:
        raise AdversarialFalsificationPacketError(
            "budget.search_candidate_rows_all_arms does not match equal-arm accounting"
        )
    ceiling = packet["compute_ceiling"]
    if ceiling["max_search_candidate_rows"] != expected_all_arms:
        raise AdversarialFalsificationPacketError(
            "compute_ceiling.max_search_candidate_rows does not match budget accounting"
        )
    if ceiling["max_confirmation_seeds"] != len(confirmation_seeds):
        raise AdversarialFalsificationPacketError(
            "compute_ceiling.max_confirmation_seeds does not match seed accounting"
        )
    if ceiling["max_steps_per_rollout"] != budget["max_steps_per_rollout"]:
        raise AdversarialFalsificationPacketError(
            "compute_ceiling.max_steps_per_rollout does not match budget accounting"
        )
    methods = packet["search_methods"]
    method_ids = [methods["primary"]["id"], *(item["id"] for item in methods["controls"])]
    if method_ids != ["cma_es", "random", "halton"]:
        raise AdversarialFalsificationPacketError(
            "search_methods must contain CMA-ES, random, and Halton in canonical order"
        )
    for method in (methods["primary"], *methods["controls"]):
        if method["budget_per_seed"] != per_seed or method["seed_group"] != "search":
            raise AdversarialFalsificationPacketError(
                f"search method {method['id']!r} is not equal-budget"
            )
    if not methods["equal_budget"]:
        raise AdversarialFalsificationPacketError("search_methods.equal_budget must be true")
    if budget["rollouts_per_candidate"] != 1 or budget["max_steps_per_rollout"] <= 0:
        raise AdversarialFalsificationPacketError("budget rollout accounting is invalid")


def _validate_packet_stop_rule(packet: Mapping[str, Any]) -> None:
    """Require a fixed budget stop rule with no adaptive replacement rows."""
    stop_rule = packet["stop_rule"]
    search = stop_rule["search"]
    budget = packet["budget"]
    search_seeds = packet["seed_policy"]["search_seeds"]
    if (
        search["action"] != "stop"
        or search["candidates_per_arm_per_seed"] != budget["candidate_budget_per_arm_per_seed"]
    ):
        raise AdversarialFalsificationPacketError(
            "stop_rule.search must stop at the declared per-seed candidate budget"
        )
    if search["search_seed_count"] != len(search_seeds) or not search["no_early_stop_on_objective"]:
        raise AdversarialFalsificationPacketError(
            "stop_rule.search must cover all search seeds without objective-based early stopping"
        )

    confirmation = stop_rule["confirmation"]
    confirmation_seeds = packet["seed_policy"]["confirmation_seeds"]
    if (
        confirmation["action"] != "stop"
        or confirmation["held_out_seed_count"] != len(confirmation_seeds)
        or confirmation["execution_status"] != "not_authorized"
    ):
        raise AdversarialFalsificationPacketError(
            "stop_rule.confirmation must stop at the five declared held-out seeds and remain unauthorized"
        )

    contract_failure = stop_rule["contract_failure"]
    if contract_failure["action"] != "stop_before_compute" or tuple(
        contract_failure["outcomes"]
    ) != ("invalid", "unavailable", "inconclusive", "blocked"):
        raise AdversarialFalsificationPacketError(
            "stop_rule.contract_failure must preserve all explicit no-result outcomes"
        )
    if stop_rule["replacement_rows_allowed"]:
        raise AdversarialFalsificationPacketError(
            "stop_rule.replacement_rows_allowed must be false"
        )


def _validate_packet_outcomes(packet: Mapping[str, Any]) -> None:
    """Require the complete no-result vocabulary without adding aliases."""
    vocabulary = packet["outcome_vocabulary"]
    if set(vocabulary) != set(ADVERSARIAL_FALSIFICATION_PACKET_OUTCOMES):
        raise AdversarialFalsificationPacketError(
            "outcome_vocabulary must contain exactly result, null, inconclusive, invalid, "
            "unavailable, and blocked"
        )
    if any(not isinstance(value, str) or not value.strip() for value in vocabulary.values()):
        raise AdversarialFalsificationPacketError(
            "outcome_vocabulary meanings must be non-empty strings"
        )


def _validate_packet_compute_gate(packet: Mapping[str, Any]) -> None:
    """Derive compute authorization from the existing answerability evaluator."""
    result = evaluate_answerability(packet["answerability"])
    gate = packet["compute_authorization"]
    if gate["gate"] != ANSWERABILITY_SCHEMA:
        raise AdversarialFalsificationPacketError(
            "compute_authorization.gate must be research_answerability.v1"
        )
    if gate["evaluated_state"] != result.state:
        raise AdversarialFalsificationPacketError(
            "compute_authorization.evaluated_state does not match research_answerability.v1"
        )
    runtime_effective = all(
        packet["variable_map"][name]["runtime_effective"] for name in _PACKET_VARIABLE_ORDER
    )
    expected_authorized = result.state == "answerable" and runtime_effective
    if gate["authorized"] != expected_authorized:
        raise AdversarialFalsificationPacketError(
            "compute_authorization.authorized is not the derived fail-closed gate result"
        )
    expected_status = "authorized" if expected_authorized else "blocked"
    if gate["status"] != expected_status:
        raise AdversarialFalsificationPacketError(
            "compute_authorization.status does not match the derived gate result"
        )
    if not expected_authorized:
        if not gate["blocking_reasons"]:
            raise AdversarialFalsificationPacketError(
                "blocked compute authorization requires explicit blocking_reasons"
            )
        if not runtime_effective and not any(
            "pedestrian_delay_s" in reason and "runtime" in reason
            for reason in gate["blocking_reasons"]
        ):
            raise AdversarialFalsificationPacketError(
                "compute authorization must name the non-runtime-effective pedestrian_delay_s"
            )
    if (
        not isinstance(gate["future_adapter_condition"], str)
        or not gate["future_adapter_condition"].strip()
    ):
        raise AdversarialFalsificationPacketError(
            "compute_authorization.future_adapter_condition must be explicit"
        )


def validate_adversarial_falsification_packet(
    packet: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Validate a v1 packet, its self-digest, source bytes, and semantic bindings."""
    if not isinstance(packet, Mapping):
        raise AdversarialFalsificationPacketError("packet must be a mapping")
    schema_errors = _packet_schema_errors(packet)
    if schema_errors:
        raise AdversarialFalsificationPacketError("; ".join(schema_errors))
    expected_digest = compute_adversarial_falsification_packet_digest(packet)
    if packet["self_digest"] != expected_digest:
        raise AdversarialFalsificationPacketError(
            "self_digest does not match canonical packet bytes"
        )
    if packet["claim_boundary"] != ADVERSARIAL_FALSIFICATION_PACKET_CLAIM_BOUNDARY:
        raise AdversarialFalsificationPacketError("packet claim_boundary is not diagnostic-only")
    if packet["source"]["base_commit"] != ADVERSARIAL_FALSIFICATION_PACKET_BASE_COMMIT:
        raise AdversarialFalsificationPacketError(
            "packet source.base_commit does not match the frozen #7340/#7382 base"
        )
    semantics = _validate_packet_sources(
        packet,
        repo_root=Path(repo_root) if repo_root is not None else _REPO_ROOT,
    )
    _validate_packet_variable_map(packet, search_space_semantics=semantics)
    _validate_packet_feasibility(packet)
    _validate_packet_design(packet)
    _validate_packet_stop_rule(packet)
    _validate_packet_outcomes(packet)
    _validate_packet_compute_gate(packet)


def load_adversarial_falsification_packet(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate a YAML/JSON v1 adversarial falsification packet.

    Returns:
        Validated packet mapping.
    """
    packet_path = Path(path)
    try:
        payload = yaml.safe_load(packet_path.read_bytes()) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AdversarialFalsificationPacketError(
            f"cannot load adversarial falsification packet {packet_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AdversarialFalsificationPacketError(
            "adversarial falsification packet must be a mapping"
        )
    validate_adversarial_falsification_packet(payload, repo_root=repo_root)
    return dict(payload)


# Long-form aliases make the packet API discoverable without creating a second contract owner.
validate_adversarial_falsification_answerability_packet = validate_adversarial_falsification_packet
load_adversarial_falsification_answerability_packet = load_adversarial_falsification_packet
compute_adversarial_falsification_answerability_packet_digest = (
    compute_adversarial_falsification_packet_digest
)


__all__ = [
    "ADVERSARIAL_FALSIFICATION_ANSWERABILITY_SCHEMA",
    "ADVERSARIAL_FALSIFICATION_PACKET_BASE_COMMIT",
    "ADVERSARIAL_FALSIFICATION_PACKET_CLAIM_BOUNDARY",
    "ADVERSARIAL_FALSIFICATION_PACKET_OUTCOMES",
    "ADVERSARIAL_FALSIFICATION_PACKET_SCHEMA",
    "ANSWERABILITY_SCHEMA",
    "ANSWERABILITY_STATES",
    "AdversarialFalsificationPacketError",
    "AnswerabilityContractError",
    "AnswerabilityResult",
    "answerability_from_manifest",
    "compute_adversarial_falsification_answerability_packet_digest",
    "compute_adversarial_falsification_packet_digest",
    "evaluate_answerability",
    "load_adversarial_falsification_answerability_packet",
    "load_adversarial_falsification_packet",
    "load_adversarial_falsification_packet_schema",
    "validate_adversarial_falsification_answerability_packet",
    "validate_adversarial_falsification_packet",
    "validate_answerability_contract",
]
