"""Default-disabled preflight for the admitted issue #8571 falsification slice.

This module composes the source-bound answerability packet with the existing
adversarial search-harness and candidate-materialization owners.  It prepares
equal-budget random and Halton candidate ledgers, records immutable source and
overlay identities, and emits explicit no-result outcomes.  It deliberately
does not instantiate an optimizer, invoke a simulator, run a planner, or
perform replay.  Those actions remain behind the packet's answerability and
replay gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.bundle import validate_template_pedestrian_binding
from robot_sf.adversarial.config import SearchSpaceConfig
from robot_sf.adversarial.materialize import ImmutableScenarioOverlay
from robot_sf.adversarial.search_harness import (
    CandidateSpecOverlayAdapter,
    FiniteSearchSpaceManifest,
    SearchCandidate,
    prepare_equal_budget_baselines,
)
from robot_sf.benchmark.research_answerability import (
    load_adversarial_falsification_packet,
)

VERTICAL_SLICE_SCHEMA_VERSION = "adversarial_falsification_vertical_slice.v1"
CLAIM_BOUNDARY = (
    "diagnostic-only default-disabled falsification preflight: immutable source and candidate "
    "preparation ledgers; no simulator, planner, optimizer, replay, campaign, benchmark, or "
    "scientific result"
)
OUTCOME_STATUSES = ("result", "null", "inconclusive", "invalid", "unavailable", "blocked")
CONTROL_ARMS = ("random", "halton")
PRIMARY_ARM = "cma_es"
CANONICAL_PACKET_RELATIVE_PATH = (
    "configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml"
)
CANONICAL_PACKET_SELF_DIGEST = "060fcd95d00bdeeabfc1697bb378e606188a4fa678b8375f937831914ca65395"
CANONICAL_PACKET_FILE_SHA256 = "7a6d3d7eb25b7de47e735630c2f47c0ccff131f9301f6eb07f4fa694e73d0197"
_ROUTE_PROPOSAL_REASON = (
    "This diagnostic preflight does not write candidate route files; materialize a "
    "route_overrides.yaml artifact through robot_sf.adversarial.bundle.write_candidate_inputs "
    "before scenario loading or replay."
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "issue",
        "packet",
        "claim_boundary",
        "source",
        "zero_overlay_equivalence",
        "arms",
        "feasibility",
        "execution",
        "gate",
        "outcome_vocabulary",
        "outcome_rows",
        "outcome_summary",
        "native_outcomes",
        "replay",
    }
)
_PACKET_FIELDS = frozenset({"issue", "packet_id", "path", "self_digest"})
_SOURCE_FIELDS = frozenset(
    {"base_ref", "base_commit", "inputs", "scenario_template_digest", "variable_order"}
)
_SOURCE_INPUT_FIELDS = frozenset({"id", "path", "role", "sha256"})
_ZERO_OVERLAY_FIELDS = frozenset(
    {"status", "source_digest", "materialized_digest", "patch_digest", "simulation_executed"}
)
_FEASIBILITY_FIELDS = frozenset(
    {
        "owner",
        "pre_simulation_rejection_ledger",
        "native_predicates_executed",
        "simulator_validity",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "compute_authorized_by_packet",
        "simulator_executed",
        "planner_executed",
        "optimizer_instantiated",
        "campaign_launched",
        "default_disabled",
    }
)
_GATE_FIELDS = frozenset({"status", "authorized", "blocking_reasons"})
_ARM_FIELDS = frozenset(
    {
        "search_seed",
        "manifest_digest",
        "candidate_budget",
        "prepared_count",
        "rejected_count",
        "preparation",
    }
)
_CMA_ES_FIELDS = frozenset(
    {
        "role",
        "algorithm",
        "owner",
        "implementation_owner",
        "search_seeds",
        "candidate_budget_per_seed",
        "execution_status",
        "reason",
    }
)
_OUTCOME_ROW_FIELDS = frozenset(
    {
        "arm",
        "candidate_id",
        "candidate",
        "search_seed",
        "status",
        "reason",
        "simulation_executed",
        "overlay_materialized_digest",
        "native_outcome_digest",
        "replay_digest",
        "rejection",
    }
)
_NATIVE_OUTCOMES_FIELDS = frozenset({"status", "rows", "digest"})
_REPLAY_FIELDS = frozenset({"status", "rows", "digest", "reason"})


class BoundedFalsificationError(ValueError):
    """Raised when the bounded falsification preflight cannot stay source-faithful."""


@dataclass(frozen=True, slots=True)
class _PreflightSource:
    """Validated packet and immutable source state used by build and validation."""

    packet: Mapping[str, Any]
    packet_file: Path
    repo_root: Path
    inputs: tuple[dict[str, Any], ...]
    scenario_template: Mapping[str, Any]
    source_overlay: ImmutableScenarioOverlay


def _assert_exact_fields(
    value: Any,
    expected_fields: frozenset[str],
    *,
    path: str,
) -> None:
    """Reject missing, unknown, or non-string fields at one report mapping boundary."""
    if not isinstance(value, Mapping):
        raise BoundedFalsificationError(f"{path} must be a mapping")
    actual_fields = set(value)
    if any(not isinstance(field, str) for field in actual_fields):
        raise BoundedFalsificationError(f"{path} fields must be strings")
    missing = expected_fields - actual_fields
    unknown = actual_fields - expected_fields
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {sorted(missing)}")
        if unknown:
            details.append(f"unknown fields {sorted(unknown)}")
        raise BoundedFalsificationError(f"{path} schema: " + "; ".join(details))


def _strictly_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int or int/float coercion."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        actual_fields = set(actual)
        if any(not isinstance(field, str) for field in actual_fields):
            return False
        if actual_fields != set(expected):
            return False
        return all(_strictly_equal(actual[field], expected[field]) for field in expected)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strictly_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if expected is None:
        return actual is None
    return type(actual) is type(expected) and actual == expected


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one source-controlled input."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BoundedFalsificationError(f"cannot read source input {path}: {exc}") from exc
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a stable repository-relative path for a report."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    """Load one YAML mapping and reject malformed source inputs."""
    try:
        payload = yaml.safe_load(path.read_bytes()) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BoundedFalsificationError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BoundedFalsificationError(f"{label} must be a mapping: {path}")
    return dict(payload)


def _thaw_source(value: Any) -> Any:
    """Convert the overlay snapshot back to YAML-shaped containers at the adapter boundary."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_source(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_thaw_source(nested) for nested in value]
    return value


def _canonical_value_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for one JSON-shaped proposal payload."""
    try:
        encoded = json.dumps(
            _thaw_source(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BoundedFalsificationError(
            f"proposal payload cannot be canonically serialized for digest: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


class _FrozenSourceCandidateSpecAdapter:
    """Reuse the bundle adapter without changing its frozen packet-owned source bytes."""

    adapter_id = CandidateSpecOverlayAdapter.adapter_id

    def __init__(self, delegate: CandidateSpecOverlayAdapter) -> None:
        """Store the existing materializer behind a frozen-source compatibility boundary."""
        self._delegate = delegate

    def validate(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> Sequence[str]:
        """Validate against mutable YAML-shaped containers, without mutating the snapshot."""
        return self._delegate.validate(_thaw_source(source_scenario), candidate)

    def materialize(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> ImmutableScenarioOverlay:
        """Materialize through the existing pure bundle seam after the compatibility conversion."""
        overlay = self._delegate.materialize(_thaw_source(source_scenario), candidate)
        patch = _thaw_source(overlay.patch)
        if not isinstance(patch, dict):
            raise TypeError("candidate overlay patch must be a mapping")
        route_payload = patch.pop("route_overrides", None)
        scenarios = patch.get("scenarios")
        if not isinstance(route_payload, Mapping):
            raise ValueError("candidate overlay route proposal is missing")
        if (
            not isinstance(scenarios, list)
            or len(scenarios) != 1
            or not isinstance(scenarios[0], Mapping)
        ):
            raise ValueError("candidate overlay must contain one specialized scenario")

        specialized = dict(scenarios[0])
        specialized.pop("route_overrides_file", None)
        specialized["route_overrides"] = _thaw_source(route_payload)
        patch["scenarios"] = [specialized]
        provenance = _thaw_source(overlay.provenance)
        if not isinstance(provenance, dict):
            raise TypeError("candidate overlay provenance must be a mapping")
        provenance["route_file_name"] = None
        provenance["route_overrides_contract"] = {
            "status": "proposal_only",
            "loadable": False,
            "payload_sha256": _canonical_value_digest(route_payload),
            "materialization_owner": "robot_sf.adversarial.bundle.write_candidate_inputs",
            "reason": _ROUTE_PROPOSAL_REASON,
        }
        return ImmutableScenarioOverlay(
            source=overlay.source,
            patch=patch,
            candidate_id=overlay.candidate_id,
            adapter_id=overlay.adapter_id,
            provenance=provenance,
        )


def _source_inputs(packet: Mapping[str, Any], *, repo_root: Path) -> list[dict[str, Any]]:
    """Recompute and report every packet input digest after packet validation."""
    inputs = packet["source"]["inputs"]
    records: list[dict[str, Any]] = []
    for input_id in ("search_space", "scenario_template", "map"):
        descriptor = inputs[input_id]
        path = repo_root / str(descriptor["path"])
        actual_digest = _sha256_file(path)
        expected_digest = str(descriptor["sha256"])
        if actual_digest != expected_digest:
            raise BoundedFalsificationError(
                f"source input {input_id} digest drifted: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        records.append(
            {
                "id": input_id,
                "path": _repo_relative(path, repo_root),
                "role": str(descriptor["role"]),
                "sha256": actual_digest,
            }
        )
    return records


def _assert_bound_match(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, variable: str
) -> None:
    """Reject drift between the typed packet and the canonical search-space file."""
    for field in ("min", "max"):
        if not math.isclose(
            float(actual[field]), float(expected[field]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise BoundedFalsificationError(
                f"search-space bound drift for {variable}.{field}: "
                f"packet={expected[field]!r}, source={actual[field]!r}"
            )


def _validate_source_contract(
    packet: Mapping[str, Any],
    *,
    search_space: SearchSpaceConfig,
    scenario_template: Mapping[str, Any],
) -> None:
    """Cross-check packet semantics against the parsed #7340 source files."""
    variable_map = packet["variable_map"]
    source_variables = packet["source"]["search_space"]["semantic"]["variables"]
    for name in packet["variable_order"]:
        _assert_bound_match(
            search_space.to_json()["variables"][name],
            variable_map[name]["bounds"],
            variable=name,
        )
        _assert_bound_match(source_variables[name], variable_map[name]["bounds"], variable=name)

    pedestrian = packet["source"]["search_space"]["semantic"]["pedestrian"]
    if search_space.pedestrian_id != pedestrian["id"]:
        raise BoundedFalsificationError(
            "search-space pedestrian identity disagrees with packet: "
            f"{search_space.pedestrian_id!r} != {pedestrian['id']!r}"
        )
    if search_space.pedestrian_route_mode != pedestrian["route_mode"]:
        raise BoundedFalsificationError(
            "search-space pedestrian route mode disagrees with packet: "
            f"{search_space.pedestrian_route_mode!r} != {pedestrian['route_mode']!r}"
        )
    template = scenario_template.get("scenarios")
    if not isinstance(template, list) or not template or not isinstance(template[0], Mapping):
        raise BoundedFalsificationError("scenario template must contain one mapping scenario")
    binding_error = validate_template_pedestrian_binding(
        template[0],
        packet["source"]["scenario_template"]["pedestrian_id"],
    )
    if binding_error is not None:
        raise BoundedFalsificationError(binding_error)


def _load_preflight_source(
    packet_path: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> _PreflightSource:
    """Load the canonical packet and freeze the source inputs used by a preflight."""
    root = (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )
    canonical_packet_file = (root / CANONICAL_PACKET_RELATIVE_PATH).resolve()
    packet_file = canonical_packet_file if packet_path is None else Path(packet_path)
    if not packet_file.is_absolute():
        packet_file = root / packet_file
    packet_file = packet_file.resolve()
    if packet_file != canonical_packet_file:
        raise BoundedFalsificationError(
            "preflight packet must use the committed canonical packet path: "
            f"{CANONICAL_PACKET_RELATIVE_PATH}"
        )
    actual_file_sha256 = _sha256_file(packet_file)
    if actual_file_sha256 != CANONICAL_PACKET_FILE_SHA256:
        raise BoundedFalsificationError(
            "canonical #8570 packet file digest does not match the trusted committed digest"
        )
    packet = load_adversarial_falsification_packet(packet_file, repo_root=root)
    if packet.get("self_digest") != CANONICAL_PACKET_SELF_DIGEST:
        raise BoundedFalsificationError(
            "canonical #8570 packet self_digest does not match the trusted committed digest"
        )
    if packet["issue"] != 8570:
        raise BoundedFalsificationError("preflight packet must be the issue #8570 packet")

    inputs = tuple(_source_inputs(packet, repo_root=root))
    search_space_path = root / packet["source"]["inputs"]["search_space"]["path"]
    template_path = root / packet["source"]["inputs"]["scenario_template"]["path"]
    search_space = SearchSpaceConfig.from_file(search_space_path)
    scenario_template = _load_mapping(template_path, label="scenario template")
    _validate_source_contract(
        packet,
        search_space=search_space,
        scenario_template=scenario_template,
    )
    source_overlay = ImmutableScenarioOverlay(
        source=scenario_template,
        patch={},
        candidate_id="source",
        adapter_id="source_snapshot.v1",
    )
    return _PreflightSource(
        packet=packet,
        packet_file=packet_file,
        repo_root=root,
        inputs=inputs,
        scenario_template=scenario_template,
        source_overlay=source_overlay,
    )


def _constraint_expression(min_distance_m: float) -> str:
    """Encode the Euclidean minimum-distance rule in the safe harness grammar."""
    squared_distance = format(float(min_distance_m) ** 2, ".17g")
    return (
        "((goal_x - start_x) * (goal_x - start_x) + "
        f"(goal_y - start_y) * (goal_y - start_y)) >= {squared_distance}"
    )


def _build_manifest(packet: Mapping[str, Any], *, search_seed: int) -> FiniteSearchSpaceManifest:
    """Translate one packet search seed into the typed preparation manifest."""
    variable_map = packet["variable_map"]
    variables = {
        name: {
            "unit": variable_map[name]["unit"],
            "kind": variable_map[name]["kind"],
            "bounds": variable_map[name]["bounds"],
        }
        for name in packet["variable_order"]
    }
    source_semantics = packet["source"]["search_space"]["semantic"]
    minimum_distance = float(source_semantics["constraints"]["min_start_goal_distance_m"])
    objective = {
        "components": [
            {
                "name": component["name"],
                "direction": component["direction"],
                "unit": component["unit"],
            }
            for component in packet["objective"]["ordering"]
        ]
    }
    return FiniteSearchSpaceManifest.from_mapping(
        {
            "schema_version": "adversarial_search_harness.v1",
            "name": f"issue_8571_bounded_falsification_seed_{search_seed}",
            "description": CLAIM_BOUNDARY,
            "source_scenario": packet["source"]["scenario_template"]["input_id"],
            "variables": variables,
            "constraints": [
                {
                    "name": "min_start_goal_distance_m",
                    "expression": _constraint_expression(minimum_distance),
                }
            ],
            "objective_vector": objective,
            "seed_policy": {
                "search_seed": int(search_seed),
                "held_out_replay_seeds": [
                    int(seed) for seed in packet["seed_policy"]["confirmation_seeds"]
                ],
                "candidate_seed_mode": packet["seed_policy"]["candidate_seed_mode"],
            },
            "rollout_budget": {
                "candidate_budget": packet["budget"]["candidate_budget_per_arm_per_seed"],
                "rollouts_per_candidate": packet["budget"]["rollouts_per_candidate"],
                "max_steps": packet["budget"]["max_steps_per_rollout"],
            },
        }
    )


def _outcome_row(
    *,
    arm: str,
    search_seed: int,
    prepared: Mapping[str, Any],
    gate_blocked: bool,
) -> dict[str, Any]:
    """Convert one preparation row into an explicit no-result outcome row."""
    candidate = prepared["candidate"]
    rejection = prepared.get("rejection")
    if rejection is not None:
        status = "invalid"
        reason = "pre_simulation_rejection"
        overlay_digest = None
    else:
        status = "blocked" if gate_blocked else "unavailable"
        reason = (
            "compute_authorization_blocked" if gate_blocked else "default_disabled_native_execution"
        )
        overlay = prepared.get("overlay") or {}
        overlay_digest = overlay.get("materialized_digest")
    return {
        "arm": arm,
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "search_seed": int(search_seed),
        "status": status,
        "reason": reason,
        "simulation_executed": False,
        "overlay_materialized_digest": overlay_digest,
        "native_outcome_digest": None,
        "replay_digest": None,
        "rejection": rejection,
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count every declared outcome vocabulary member, including zero counts."""
    counts = dict.fromkeys(OUTCOME_STATUSES, 0)
    for row in rows:
        status = row["status"]
        if status not in counts:
            raise BoundedFalsificationError(f"unsupported outcome status: {status!r}")
        counts[status] += 1
    return counts


def _cma_es_arm(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical declaration for the intentionally unexecuted primary arm."""
    primary = packet["search_methods"]["primary"]
    return {
        "role": primary["role"],
        "algorithm": primary["algorithm"],
        "owner": primary["owner"],
        "implementation_owner": "robot_sf.adversarial.samplers.CmaEsCandidateSampler",
        "search_seeds": [int(seed) for seed in packet["seed_policy"]["search_seeds"]],
        "candidate_budget_per_seed": packet["budget"]["candidate_budget_per_arm_per_seed"],
        "execution_status": "declared_not_executed",
        "reason": (
            "compute_ceiling.optimizer_allowed=false; the default-disabled preflight "
            "must not instantiate an optimizer"
        ),
    }


def _prepare_control_ledgers(
    source: _PreflightSource,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Build canonical zero-overlay, control-arm, and outcome ledgers from source bytes."""
    packet = source.packet
    source_overlay = source.source_overlay
    zero_overlay = {
        "status": "passed"
        if source_overlay.source_digest == source_overlay.materialized_digest
        else "failed",
        "source_digest": source_overlay.source_digest,
        "materialized_digest": source_overlay.materialized_digest,
        "patch_digest": source_overlay.patch_digest,
        "simulation_executed": False,
    }
    if zero_overlay["status"] != "passed":
        raise BoundedFalsificationError("zero overlay changed the source scenario")

    adapter = _FrozenSourceCandidateSpecAdapter(
        CandidateSpecOverlayAdapter(
            pedestrian_id=packet["source"]["scenario_template"]["pedestrian_id"],
            pedestrian_route_mode=packet["source"]["scenario_template"]["pedestrian_route_mode"],
        )
    )
    gate_blocked = not bool(packet["compute_authorization"]["authorized"])
    arm_records: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROL_ARMS}
    outcome_rows: list[dict[str, Any]] = []
    for raw_seed in packet["seed_policy"]["search_seeds"]:
        search_seed = int(raw_seed)
        manifest = _build_manifest(packet, search_seed=search_seed)
        preparations = prepare_equal_budget_baselines(
            manifest,
            source.scenario_template,
            adapter,
            baselines=("random", "quasi_random"),
        )
        for harness_name, preparation in preparations.items():
            arm = "halton" if harness_name == "quasi_random" else harness_name
            preparation_payload = preparation.to_dict()
            arm_records[arm].append(
                {
                    "search_seed": search_seed,
                    "manifest_digest": manifest.digest,
                    "candidate_budget": manifest.rollout_budget.candidate_budget,
                    "prepared_count": preparation.prepared_count,
                    "rejected_count": preparation.rejected_count,
                    "preparation": preparation_payload,
                }
            )
            outcome_rows.extend(
                _outcome_row(
                    arm=arm,
                    search_seed=search_seed,
                    prepared=row.to_dict(),
                    gate_blocked=gate_blocked,
                )
                for row in preparation.candidates
            )
    return zero_overlay, arm_records, outcome_rows


def _expected_native_status(packet: Mapping[str, Any]) -> str:
    """Return the canonical status for the disabled native-outcome section."""
    return "blocked" if not bool(packet["compute_authorization"]["authorized"]) else "not_requested"


def _canonical_feasibility(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the report-level feasibility declaration from the validated packet."""
    rejection_accounting = packet["feasibility"]["rejection_accounting"]
    return {
        "owner": rejection_accounting["owner"],
        "pre_simulation_rejection_ledger": rejection_accounting["pre_simulation"],
        "native_predicates_executed": False,
        "simulator_validity": "unavailable",
    }


def _validate_bounded_falsification_report(  # noqa: C901, PLR0912, PLR0915
    report: Mapping[str, Any],
    *,
    source: _PreflightSource,
    expected_zero_overlay: Mapping[str, Any],
    expected_control_arms: Mapping[str, list[dict[str, Any]]],
    expected_outcome_rows: list[dict[str, Any]],
) -> None:
    """Validate a report against recomputed packet-owned ledgers and no-compute invariants."""
    packet = source.packet
    if not isinstance(report, Mapping):
        raise BoundedFalsificationError("preflight report must be a mapping")
    _assert_exact_fields(report, _REPORT_FIELDS, path="preflight report")
    if not _strictly_equal(report["schema_version"], VERTICAL_SLICE_SCHEMA_VERSION):
        raise BoundedFalsificationError("preflight schema_version is unsupported")
    if not _strictly_equal(report["issue"], 8571):
        raise BoundedFalsificationError("preflight issue must be 8571")
    if not _strictly_equal(report["claim_boundary"], CLAIM_BOUNDARY):
        raise BoundedFalsificationError("preflight claim boundary drifted")

    expected_packet = {
        "issue": packet["issue"],
        "packet_id": packet["packet_id"],
        "path": _repo_relative(source.packet_file, source.repo_root),
        "self_digest": packet["self_digest"],
    }
    _assert_exact_fields(report["packet"], _PACKET_FIELDS, path="preflight packet")
    if not _strictly_equal(report["packet"], expected_packet):
        raise BoundedFalsificationError("preflight packet identity is not bound to canonical #8570")
    expected_source = {
        "base_ref": packet["source"]["base_ref"],
        "base_commit": packet["source"]["base_commit"],
        "inputs": list(source.inputs),
        "scenario_template_digest": source.source_overlay.source_digest,
        "variable_order": list(packet["variable_order"]),
    }
    _assert_exact_fields(report["source"], _SOURCE_FIELDS, path="preflight source")
    source_inputs = report["source"]["inputs"]
    if not isinstance(source_inputs, list):
        raise BoundedFalsificationError("preflight source.inputs must be a list")
    for index, source_input in enumerate(source_inputs):
        _assert_exact_fields(
            source_input,
            _SOURCE_INPUT_FIELDS,
            path=f"preflight source.inputs[{index}]",
        )
    if not _strictly_equal(report["source"], expected_source):
        raise BoundedFalsificationError(
            "preflight source identity is not bound to canonical #8570 inputs"
        )

    feasibility = report["feasibility"]
    _assert_exact_fields(feasibility, _FEASIBILITY_FIELDS, path="preflight feasibility")
    expected_feasibility = _canonical_feasibility(packet)
    if not _strictly_equal(feasibility, expected_feasibility):
        raise BoundedFalsificationError("preflight feasibility is not canonical")

    execution = report["execution"]
    _assert_exact_fields(execution, _EXECUTION_FIELDS, path="preflight execution")
    for field in (
        "simulator_executed",
        "planner_executed",
        "optimizer_instantiated",
        "campaign_launched",
    ):
        if execution.get(field) is not False:
            raise BoundedFalsificationError(f"preflight execution.{field} must be false")
    if execution.get("default_disabled") is not True:
        raise BoundedFalsificationError("preflight execution.default_disabled must be true")
    expected_authorized = bool(packet["compute_authorization"]["authorized"])
    if execution.get("compute_authorized_by_packet") is not expected_authorized:
        raise BoundedFalsificationError(
            "preflight execution.compute_authorized_by_packet is not bound to canonical packet"
        )

    gate = report["gate"]
    _assert_exact_fields(gate, _GATE_FIELDS, path="preflight gate")
    if gate.get("authorized") is not expected_authorized:
        raise BoundedFalsificationError(
            "preflight gate.authorized is not bound to canonical packet"
        )
    expected_gate_status = "blocked" if not expected_authorized else "not_requested"
    if gate.get("status") != expected_gate_status:
        raise BoundedFalsificationError(
            f"preflight gate.status must be {expected_gate_status!r} for canonical packet"
        )
    expected_blocking_reasons = list(packet["compute_authorization"]["blocking_reasons"])
    if not _strictly_equal(gate.get("blocking_reasons"), expected_blocking_reasons):
        raise BoundedFalsificationError(
            "preflight gate.blocking_reasons are not bound to canonical packet"
        )

    zero_overlay = report["zero_overlay_equivalence"]
    _assert_exact_fields(
        zero_overlay,
        _ZERO_OVERLAY_FIELDS,
        path="preflight zero_overlay_equivalence",
    )
    if not _strictly_equal(zero_overlay, expected_zero_overlay):
        raise BoundedFalsificationError(
            "preflight zero_overlay_equivalence disagrees with canonical source"
        )

    arms = report["arms"]
    _assert_exact_fields(
        arms,
        frozenset({PRIMARY_ARM, *CONTROL_ARMS}),
        path="preflight arms",
    )
    expected_cma_es = _cma_es_arm(packet)
    _assert_exact_fields(arms[PRIMARY_ARM], _CMA_ES_FIELDS, path="preflight arms.cma_es")
    if not _strictly_equal(arms[PRIMARY_ARM], expected_cma_es):
        raise BoundedFalsificationError("preflight CMA-ES arm is not canonical")
    for arm in CONTROL_ARMS:
        actual_entries = arms[arm]
        expected_entries = expected_control_arms[arm]
        if not isinstance(actual_entries, list):
            raise BoundedFalsificationError(
                f"preflight {arm} candidate ledger is empty, altered, or not source-bound"
            )
        if len(actual_entries) != len(expected_entries):
            raise BoundedFalsificationError(
                f"preflight {arm} candidate ledger is empty, altered, or not source-bound"
            )
        for index, entry in enumerate(actual_entries):
            _assert_exact_fields(entry, _ARM_FIELDS, path=f"preflight arms.{arm}[{index}]")
        if not _strictly_equal(actual_entries, expected_entries):
            raise BoundedFalsificationError(
                f"preflight {arm} candidate ledger is empty, altered, or not source-bound"
            )

    for section in ("native_outcomes", "replay"):
        outputs = report[section]
        _assert_exact_fields(
            outputs,
            _NATIVE_OUTCOMES_FIELDS if section == "native_outcomes" else _REPLAY_FIELDS,
            path=f"preflight {section}",
        )
        expected_status = (
            _expected_native_status(packet) if section == "native_outcomes" else "blocked"
        )
        if outputs.get("status") != expected_status:
            raise BoundedFalsificationError(
                f"preflight {section}.status must be canonical {expected_status!r}"
            )
        rows_count = outputs.get("rows")
        if not isinstance(rows_count, int) or isinstance(rows_count, bool) or rows_count != 0:
            raise BoundedFalsificationError(f"preflight {section}.rows must be zero")
        if "digest" not in outputs or outputs["digest"] is not None:
            raise BoundedFalsificationError(f"preflight {section}.digest must be null")
        if section == "replay" and outputs.get("reason") != (
            "same-fixture replay/no-op fidelity is a prerequisite for native admission"
        ):
            raise BoundedFalsificationError("preflight replay.reason is not canonical")

    rows = report["outcome_rows"]
    if not isinstance(rows, list):
        raise BoundedFalsificationError("preflight outcome_rows must be a list")
    for index, row in enumerate(rows):
        _assert_exact_fields(row, _OUTCOME_ROW_FIELDS, path=f"preflight outcome_rows[{index}]")
        if row.get("simulation_executed") is not False:
            raise BoundedFalsificationError(
                f"preflight outcome row {index} simulation_executed must be false"
            )
        for field in ("native_outcome_digest", "replay_digest"):
            if field not in row or row[field] is not None:
                raise BoundedFalsificationError(
                    f"preflight outcome row {index} {field} must be null"
                )
        status = row.get("status")
        if not isinstance(status, str) or status not in OUTCOME_STATUSES:
            raise BoundedFalsificationError(
                f"preflight outcome row {index} has unsupported status {status!r}"
            )
        rejection = row.get("rejection")
        if status == "invalid" and not isinstance(rejection, Mapping):
            raise BoundedFalsificationError(
                f"preflight invalid outcome row {index} requires rejection metadata"
            )
        if status != "invalid" and rejection is not None:
            raise BoundedFalsificationError(
                f"preflight non-invalid outcome row {index} cannot carry rejection metadata"
            )
    expected_summary = _summary(rows)
    _assert_exact_fields(
        report["outcome_summary"],
        frozenset(OUTCOME_STATUSES),
        path="preflight outcome_summary",
    )
    if not _strictly_equal(report["outcome_summary"], expected_summary):
        raise BoundedFalsificationError("preflight outcome summary disagrees with outcome rows")
    if any(row["status"] in {"result", "null"} for row in rows):
        raise BoundedFalsificationError("preflight cannot emit result or null outcomes")
    if not _strictly_equal(rows, expected_outcome_rows):
        raise BoundedFalsificationError(
            "preflight outcome ledger is empty, altered, or not source-bound"
        )

    expected_vocabulary = {key: packet["outcome_vocabulary"][key] for key in OUTCOME_STATUSES}
    _assert_exact_fields(
        report["outcome_vocabulary"],
        frozenset(OUTCOME_STATUSES),
        path="preflight outcome_vocabulary",
    )
    if not _strictly_equal(report["outcome_vocabulary"], expected_vocabulary):
        raise BoundedFalsificationError("preflight outcome vocabulary is not canonical")


def validate_bounded_falsification_preflight(
    report: Mapping[str, Any],
    *,
    packet_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> None:
    """Validate a report against the canonical packet without running compute."""
    source = _load_preflight_source(packet_path, repo_root=repo_root)
    zero_overlay, control_arms, outcome_rows = _prepare_control_ledgers(source)
    _validate_bounded_falsification_report(
        report,
        source=source,
        expected_zero_overlay=zero_overlay,
        expected_control_arms=control_arms,
        expected_outcome_rows=outcome_rows,
    )


def build_bounded_falsification_preflight(
    packet_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the issue #8571 default-disabled preparation and blocked ledger.

    The returned report is deterministic for fixed source bytes.  It contains 64 candidate
    records per control arm, per declared search seed, plus any pre-simulation rejection
    records.  Feasible here means only that the immutable overlay adapter accepted the
    candidate; it is not a native feasibility or simulator result.
    """
    source = _load_preflight_source(packet_path, repo_root=repo_root)
    packet = source.packet
    zero_overlay, arm_records, outcome_rows = _prepare_control_ledgers(source)

    gate = packet["compute_authorization"]
    gate_blocked = not bool(gate["authorized"])
    report: dict[str, Any] = {
        "schema_version": VERTICAL_SLICE_SCHEMA_VERSION,
        "issue": 8571,
        "packet": {
            "issue": packet["issue"],
            "packet_id": packet["packet_id"],
            "path": _repo_relative(source.packet_file, source.repo_root),
            "self_digest": packet["self_digest"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "source": {
            "base_ref": packet["source"]["base_ref"],
            "base_commit": packet["source"]["base_commit"],
            "inputs": list(source.inputs),
            "scenario_template_digest": source.source_overlay.source_digest,
            "variable_order": list(packet["variable_order"]),
        },
        "zero_overlay_equivalence": zero_overlay,
        "arms": {
            "cma_es": _cma_es_arm(packet),
            "random": arm_records["random"],
            "halton": arm_records["halton"],
        },
        "feasibility": _canonical_feasibility(packet),
        "execution": {
            "compute_authorized_by_packet": bool(gate["authorized"]),
            "simulator_executed": False,
            "planner_executed": False,
            "optimizer_instantiated": False,
            "campaign_launched": False,
            "default_disabled": True,
        },
        "gate": {
            "status": "blocked" if gate_blocked else "not_requested",
            "authorized": bool(gate["authorized"]),
            "blocking_reasons": list(gate["blocking_reasons"]),
        },
        "outcome_vocabulary": {key: packet["outcome_vocabulary"][key] for key in OUTCOME_STATUSES},
        "outcome_rows": outcome_rows,
        "outcome_summary": _summary(outcome_rows),
        "native_outcomes": {
            "status": "blocked" if gate_blocked else "not_requested",
            "rows": 0,
            "digest": None,
        },
        "replay": {
            "status": "blocked",
            "rows": 0,
            "digest": None,
            "reason": "same-fixture replay/no-op fidelity is a prerequisite for native admission",
        },
    }
    _validate_bounded_falsification_report(
        report,
        source=source,
        expected_zero_overlay=zero_overlay,
        expected_control_arms=arm_records,
        expected_outcome_rows=outcome_rows,
    )
    return report


def write_bounded_falsification_preflight(
    packet_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build and persist one deterministic preflight report as JSON."""
    report = build_bounded_falsification_preflight(packet_path, repo_root=repo_root)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "CLAIM_BOUNDARY",
    "CONTROL_ARMS",
    "OUTCOME_STATUSES",
    "PRIMARY_ARM",
    "VERTICAL_SLICE_SCHEMA_VERSION",
    "BoundedFalsificationError",
    "build_bounded_falsification_preflight",
    "validate_bounded_falsification_preflight",
    "write_bounded_falsification_preflight",
]
