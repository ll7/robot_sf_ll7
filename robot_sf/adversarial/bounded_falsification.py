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


class BoundedFalsificationError(ValueError):
    """Raised when the bounded falsification preflight cannot stay source-faithful."""


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
        return self._delegate.materialize(_thaw_source(source_scenario), candidate)


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


def validate_bounded_falsification_preflight(report: Mapping[str, Any]) -> None:
    """Validate the fail-closed invariants of one generated preflight report."""
    if report.get("schema_version") != VERTICAL_SLICE_SCHEMA_VERSION:
        raise BoundedFalsificationError("preflight schema_version is unsupported")
    if report.get("claim_boundary") != CLAIM_BOUNDARY:
        raise BoundedFalsificationError("preflight claim boundary drifted")
    if report.get("execution", {}).get("simulator_executed") is not False:
        raise BoundedFalsificationError("preflight cannot claim simulator execution")
    if report.get("execution", {}).get("optimizer_instantiated") is not False:
        raise BoundedFalsificationError("preflight cannot instantiate an optimizer")
    rows = report.get("outcome_rows")
    if not isinstance(rows, list):
        raise BoundedFalsificationError("preflight outcome_rows must be a list")
    expected_summary = _summary(rows)
    if report.get("outcome_summary") != expected_summary:
        raise BoundedFalsificationError("preflight outcome summary disagrees with outcome rows")
    if any(row["status"] in {"result", "null"} for row in rows):
        raise BoundedFalsificationError("preflight cannot emit result or null outcomes")
    if report.get("replay", {}).get("digest") is not None:
        raise BoundedFalsificationError("preflight cannot invent a replay digest")


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
    packet_file = Path(packet_path)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    packet = load_adversarial_falsification_packet(packet_file, repo_root=root)
    if packet["issue"] != 8570:
        raise BoundedFalsificationError("preflight packet must be the issue #8570 packet")

    inputs = _source_inputs(packet, repo_root=root)
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
    gate = packet["compute_authorization"]
    gate_blocked = not bool(gate["authorized"])
    arm_records: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROL_ARMS}
    outcome_rows: list[dict[str, Any]] = []
    for raw_seed in packet["seed_policy"]["search_seeds"]:
        search_seed = int(raw_seed)
        manifest = _build_manifest(packet, search_seed=search_seed)
        preparations = prepare_equal_budget_baselines(
            manifest,
            scenario_template,
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

    primary = packet["search_methods"]["primary"]
    report: dict[str, Any] = {
        "schema_version": VERTICAL_SLICE_SCHEMA_VERSION,
        "issue": 8571,
        "packet": {
            "issue": packet["issue"],
            "packet_id": packet["packet_id"],
            "path": _repo_relative(packet_file, root),
            "self_digest": packet["self_digest"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "source": {
            "base_ref": packet["source"]["base_ref"],
            "base_commit": packet["source"]["base_commit"],
            "inputs": inputs,
            "scenario_template_digest": source_overlay.source_digest,
            "variable_order": list(packet["variable_order"]),
        },
        "zero_overlay_equivalence": zero_overlay,
        "arms": {
            "cma_es": {
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
            },
            "random": arm_records["random"],
            "halton": arm_records["halton"],
        },
        "feasibility": {
            "owner": packet["feasibility"]["rejection_accounting"]["owner"],
            "pre_simulation_rejection_ledger": True,
            "native_predicates_executed": False,
            "simulator_validity": "unavailable",
        },
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
    validate_bounded_falsification_preflight(report)
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
