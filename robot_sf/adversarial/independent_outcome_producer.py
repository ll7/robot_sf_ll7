"""Build the issue #6105 v2 outcome packet from explicit execution records.

The producer is deliberately a lineage bridge, not a planner runner.  A runner
must write one explicit envelope for the deterministic replay and five
confirmation executions for every selected candidate.  This module validates
that envelope against the frozen candidate binding, inspects the benchmark
episode's canonical planner metadata, and emits the row-level
``adversarial_independent_outcomes.v2`` packet.

The frozen #6105 contract admits only the canonical ``SocialForcePlannerAdapter``
identity.  Native, fallback, degraded, mixed, unavailable, or self-reported
aliases fail closed before a row is emitted.  The historical planner/reference
commit remains in ``execution_commit`` while the producing code's merged
commit is carried separately as ``producer_commit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.independent_outcomes import (
    OUTCOME_OBJECTIVE,
    OUTCOME_SCHEMA_VERSION,
    AdmissionSpec,
    build_independent_outcome_evaluation,
    payload_sha256,
)
from robot_sf.adversarial.proposal_model import load_issue_3275_contract
from robot_sf.benchmark.termination_reason import (
    TERMINATION_REASONS,
    TIMEOUT_TERMINATION_REASONS,
)

EXECUTION_RECORD_SCHEMA_VERSION = "issue_7066_execution_record.v1"
_ARMS = ("proposal", "random")
_IDENTITY_FIELDS = (
    "policy_semantics",
    "adapter_name",
    "upstream_command_space",
    "benchmark_command_space",
    "projection_policy",
)
_SHA1_LENGTH = 40
_CANONICAL_OUTCOME_FIELDS = frozenset({"route_complete", "collision_event", "timeout_event"})


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a complete SHA-256 hexadecimal digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_sha1(value: Any) -> bool:
    """Return whether a value is a complete Git SHA-1 hexadecimal digest."""
    return (
        isinstance(value, str)
        and len(value) == _SHA1_LENGTH
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _require_integer(value: Any, *, label: str) -> int:
    """Require a JSON integer without accepting booleans or numeric lookalikes."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _raw_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and fail closed on malformed input."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required JSON file is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file {path} must contain an object")
    return payload


def load_execution_records(path: Path) -> list[dict[str, Any]]:
    """Load explicit execution envelopes from JSON Lines.

    Empty lines are ignored, but every non-empty line must be one JSON object.
    A JSON array, malformed line, or empty input is rejected instead of being
    treated as a partial empirical packet.
    """
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"execution-record JSONL is missing or empty: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"execution-record JSONL line {line_number} is malformed: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"execution-record JSONL line {line_number} must be an object")
                records.append(payload)
    except OSError as exc:
        raise ValueError(f"failed to read execution-record JSONL {path}: {exc}") from exc
    if not records:
        raise ValueError(f"execution-record JSONL is empty: {path}")
    return records


def _load_contract_identity(contract_path: Path) -> dict[str, Any]:
    """Load and validate the frozen canonical-adapter contract identity."""
    contract = load_issue_3275_contract(contract_path)
    planner = contract.get("target_planner")
    if not isinstance(planner, dict):
        raise ValueError("contract target_planner must be an object")
    if planner.get("id") != "social_force":
        raise ValueError("issue #6105 producer only accepts target planner social_force")
    if not _is_sha256(planner.get("config_sha256")):
        raise ValueError("contract target planner config_sha256 must be SHA-256 hex")
    if not _is_sha1(planner.get("execution_commit")):
        raise ValueError("contract historical execution_commit must be a Git SHA-1")
    identity = planner.get("execution_identity")
    if not isinstance(identity, dict) or identity.get("execution_mode") != "adapter":
        raise ValueError("contract must declare adapter execution identity")
    for key in _IDENTITY_FIELDS:
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"contract execution identity is missing {key!r}")
    outcome_contract = contract.get("outcome_contract")
    required_fields = {"execution_identity", "producer_commit", "episode_record_sha256"}
    if (
        not isinstance(outcome_contract, dict)
        or outcome_contract.get("schema") != OUTCOME_SCHEMA_VERSION
    ):
        raise ValueError("contract must declare adversarial_independent_outcomes.v2")
    admitted_fields = outcome_contract.get("admitted_row_fields")
    if not isinstance(admitted_fields, list) or not required_fields.issubset(set(admitted_fields)):
        raise ValueError("contract outcome fields do not declare producer lineage")
    return {
        "contract": contract,
        "planner": planner,
        "identity": {key: str(identity[key]) for key in _IDENTITY_FIELDS},
        "contract_sha256": _raw_sha256(contract_path),
    }


def _load_binding(  # noqa: C901, PLR0912
    binding_path: Path, *, budget_per_arm: int
) -> dict[str, Any]:
    """Load the external selected-arm binding with exact key coverage."""
    binding = _read_json(binding_path)
    if binding.get("schema_version") != "adversarial_candidate_manifest_bindings.v2":
        raise ValueError("execution binding must use adversarial_candidate_manifest_bindings.v2")
    ids_by_arm = binding.get("candidate_manifest_ids_by_arm")
    if not isinstance(ids_by_arm, dict) or set(ids_by_arm) != set(_ARMS):
        raise ValueError("execution binding must define exactly proposal and random arms")
    normalized_ids: dict[str, tuple[str, ...]] = {}
    for arm in _ARMS:
        raw_ids = ids_by_arm[arm]
        if not isinstance(raw_ids, list) or len(raw_ids) != budget_per_arm:
            raise ValueError(f"execution binding {arm} arm does not match frozen budget")
        if any(not isinstance(value, str) or not value for value in raw_ids):
            raise ValueError(f"execution binding {arm} IDs must be non-empty strings")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError(f"execution binding {arm} IDs are not unique")
        normalized_ids[arm] = tuple(raw_ids)
    selected_ids = set(normalized_ids["proposal"]) | set(normalized_ids["random"])
    if len(selected_ids) != 2 * budget_per_arm:
        raise ValueError("execution binding reuses a candidate across arms")

    map_fields = (
        "candidate_manifest_sha256_by_id",
        "candidate_pool_index_by_manifest_id",
        "scenario_seed_by_manifest_id",
        "record_sha256_by_manifest_id",
        "execution_seeds_by_manifest_id",
    )
    for field in map_fields:
        values = binding.get(field)
        if not isinstance(values, dict) or set(values) != selected_ids:
            raise ValueError(f"execution binding {field} must cover exactly selected IDs")

    manifest_hashes = binding["candidate_manifest_sha256_by_id"]
    record_hashes = binding["record_sha256_by_manifest_id"]
    pool_indices = binding["candidate_pool_index_by_manifest_id"]
    scenario_seeds = binding["scenario_seed_by_manifest_id"]
    execution_seeds = binding["execution_seeds_by_manifest_id"]
    if any(not _is_sha256(value) for value in manifest_hashes.values()):
        raise ValueError("execution binding contains an invalid manifest SHA-256")
    if any(not _is_sha256(value) for value in record_hashes.values()):
        raise ValueError("execution binding contains an invalid record SHA-256")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in pool_indices.values()
    ):
        raise ValueError("execution binding contains an invalid candidate pool index")
    if len(set(pool_indices.values())) != len(selected_ids):
        raise ValueError("execution binding candidate pool indices are not unique")
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in scenario_seeds.values()
    ):
        raise ValueError("execution binding contains an invalid scenario seed")
    normalized_execution_seeds: dict[str, tuple[int, ...]] = {}
    for manifest_id, raw_seeds in execution_seeds.items():
        if not isinstance(raw_seeds, list) or len(raw_seeds) != 5:
            raise ValueError(f"execution binding seeds for {manifest_id} must contain five values")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds):
            raise ValueError(f"execution binding seeds for {manifest_id} must be integers")
        if len(set(raw_seeds)) != len(raw_seeds):
            raise ValueError(f"execution binding seeds for {manifest_id} are not unique")
        normalized_execution_seeds[manifest_id] = tuple(raw_seeds)
    pool_seed = binding.get("candidate_pool_seed")
    if not isinstance(pool_seed, int) or isinstance(pool_seed, bool):
        raise ValueError("execution binding candidate_pool_seed must be an integer")
    return {
        "raw": binding,
        "raw_sha256": _raw_sha256(binding_path),
        "candidate_manifest_ids_by_arm": normalized_ids,
        "candidate_manifest_sha256_by_id": dict(manifest_hashes),
        "candidate_pool_index_by_manifest_id": dict(pool_indices),
        "scenario_seed_by_manifest_id": dict(scenario_seeds),
        "record_sha256_by_manifest_id": dict(record_hashes),
        "execution_seeds_by_manifest_id": normalized_execution_seeds,
        "candidate_pool_seed": pool_seed,
    }


def _admission_spec(contract_data: dict[str, Any], binding: dict[str, Any]) -> AdmissionSpec:
    """Build the strict v2 admission spec for the canonical adapter contract."""
    contract = contract_data["contract"]
    planner = contract_data["planner"]
    threshold = "4_of_5" if contract["failure_admission"]["four_of_five_required"] else "3_of_5"
    return AdmissionSpec(
        expected_target_planner_id=planner["id"],
        expected_eval_family=contract["evaluation"]["scenario_family"],
        confirmation_threshold=threshold,
        expected_target_planner_config_sha256=planner["config_sha256"],
        expected_candidate_manifest_sha256_by_id=binding["candidate_manifest_sha256_by_id"],
        expected_candidate_pool_index_by_manifest_id=binding["candidate_pool_index_by_manifest_id"],
        expected_scenario_seed_by_manifest_id=binding["scenario_seed_by_manifest_id"],
        expected_record_sha256_by_manifest=binding["record_sha256_by_manifest_id"],
        expected_candidate_manifest_ids_by_arm=binding["candidate_manifest_ids_by_arm"],
        expected_execution_seeds_by_manifest_id=binding["execution_seeds_by_manifest_id"],
        expected_candidate_pool_seed=binding["candidate_pool_seed"],
        expected_execution_commit=planner["execution_commit"],
        expected_execution_mode=planner["execution_identity"]["execution_mode"],
        expected_execution_identity=contract_data["identity"],
        require_producer_commit=True,
        require_episode_record_sha256=True,
    )


def _validate_adapter_status(metadata: dict[str, Any]) -> None:
    """Reject unavailable, degraded, or otherwise ineligible adapter metadata."""
    if metadata.get("status") != "ok":
        raise ValueError(
            "episode record algorithm_metadata status is not ok "
            "(fallback/degraded/unavailable fail closed)"
        )
    for key, label in (
        ("fallback_or_degraded", "fallback_or_degraded"),
        ("evidence_eligible", "evidence_eligible"),
    ):
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, bool):
            raise ValueError(f"episode record {label} must be boolean when present")
        if key == "fallback_or_degraded" and value:
            raise ValueError(f"episode record {label} marks the execution ineligible")
        if key == "evidence_eligible" and not value:
            raise ValueError(f"episode record {label} marks the execution ineligible")
    if "availability_status" in metadata and metadata["availability_status"] != "available":
        raise ValueError("episode record availability_status is not available")
    if "readiness_status" in metadata and metadata["readiness_status"] not in {
        "adapter",
        "ok",
    }:
        raise ValueError("episode record readiness_status is not adapter-ready")
    if "preflight_status" in metadata and metadata["preflight_status"] not in {
        "ok",
        "pass",
        "passed",
        "ready",
    }:
        raise ValueError("episode record preflight_status is not successful")


def _validate_adapter_kinematics(
    metadata: dict[str, Any], expected: dict[str, str]
) -> dict[str, str]:
    """Extract and verify the frozen adapter command-space identity."""
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError("episode record is missing planner_kinematics diagnostics")
    observed: dict[str, str] = {}
    for key in _IDENTITY_FIELDS[1:]:
        value = kinematics.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"episode planner_kinematics is missing {key!r}")
        observed[key] = value
    if kinematics.get("execution_mode") != "adapter":
        raise ValueError("episode planner_kinematics execution_mode is not adapter")
    adapter_active = kinematics.get("adapter_active")
    if adapter_active is not True:
        raise ValueError("episode planner_kinematics adapter_active must be true for the adapter")
    if any(observed[key] != expected[key] for key in observed):
        raise ValueError("episode planner_kinematics canonical adapter identity mismatch")
    return {"policy_semantics": expected["policy_semantics"], **observed}


def _episode_metadata(record: dict[str, Any], expected: dict[str, str]) -> dict[str, str]:
    """Extract and verify canonical planner identity from one episode record."""
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("episode record is missing algorithm_metadata")
    if metadata.get("canonical_algorithm") != "social_force":
        raise ValueError("episode record canonical_algorithm is not social_force")
    if metadata.get("policy_semantics") != expected["policy_semantics"]:
        raise ValueError("episode record policy_semantics does not match the frozen adapter")
    _validate_adapter_status(metadata)
    return _validate_adapter_kinematics(metadata, expected)


def _record_producer_commit(record: dict[str, Any]) -> str:
    """Resolve the actual producing commit from canonical episode provenance."""
    candidates: list[Any] = []
    result_provenance = record.get("result_provenance")
    if isinstance(result_provenance, dict):
        candidates.append(result_provenance.get("repo_commit"))
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        candidates.append(provenance.get("git_hash"))
    commits = [value for value in candidates if value is not None]
    if not commits or any(not _is_sha1(value) for value in commits):
        raise ValueError("episode record must carry a 40-character producer Git commit")
    if len(set(commits)) != 1:
        raise ValueError("episode provenance producer commits disagree")
    return str(commits[0])


def _validate_episode_outcome(record: dict[str, Any], *, manifest_id: str) -> None:
    """Require the episode's canonical outcome and termination semantics.

    The shared episode schema deliberately does not accept legacy ``collision``
    or ``timeout`` aliases in the outcome object.  Enforcing that boundary here
    is important because :func:`attribution_from_episode_record` supports those
    aliases for older diagnostic callers and uses Python truthiness.  Without
    this check, strings such as ``"false"`` could become empirical failures.
    """
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or set(outcome) != _CANONICAL_OUTCOME_FIELDS:
        observed = sorted(outcome) if isinstance(outcome, dict) else []
        missing = sorted(_CANONICAL_OUTCOME_FIELDS - set(observed))
        extra = sorted(set(observed) - _CANONICAL_OUTCOME_FIELDS)
        raise ValueError(
            f"episode record {manifest_id} outcome must use exactly the canonical fields; "
            f"missing={missing} extra={extra}"
        )
    invalid_fields = [
        field for field in sorted(_CANONICAL_OUTCOME_FIELDS) if not isinstance(outcome[field], bool)
    ]
    if invalid_fields:
        raise ValueError(
            f"episode record {manifest_id} outcome fields must be boolean: {invalid_fields}"
        )

    termination_reason = record.get("termination_reason")
    if termination_reason not in TERMINATION_REASONS:
        raise ValueError(
            f"episode record {manifest_id} termination_reason is not canonical: "
            f"{termination_reason!r}"
        )
    if termination_reason == "error":
        raise ValueError(f"episode record {manifest_id} runtime error is not an outcome")

    route_complete = outcome["route_complete"]
    collision = outcome["collision_event"]
    timeout = outcome["timeout_event"]
    if sum((route_complete, collision, timeout)) > 1:
        raise ValueError(
            f"episode record {manifest_id} outcome contains contradictory terminal flags"
        )
    if route_complete != (termination_reason == "success"):
        raise ValueError(
            f"episode record {manifest_id} route_complete disagrees with termination_reason"
        )
    if collision != (termination_reason == "collision"):
        raise ValueError(
            f"episode record {manifest_id} collision_event disagrees with termination_reason"
        )
    if timeout != (termination_reason in TIMEOUT_TERMINATION_REASONS):
        raise ValueError(
            f"episode record {manifest_id} timeout_event disagrees with termination_reason"
        )


def _validate_replay_lineage(value: Any) -> dict[str, Any]:
    """Validate the deterministic replay signature block supplied by the runner."""
    if not isinstance(value, dict):
        raise ValueError("replay execution is missing replay_lineage")
    original = value.get("original_signature_sha256")
    replay = value.get("replay_signature_sha256")
    if not _is_sha256(original) or not _is_sha256(replay) or original != replay:
        raise ValueError("replay signatures are missing, malformed, or do not match")
    if value.get("exact_signature_match") is not True:
        raise ValueError("replay_lineage.exact_signature_match must be true")
    return {
        "exact_signature_match": True,
        "original_signature_sha256": original,
        "replay_signature_sha256": replay,
    }


def _validate_envelope_common(  # noqa: C901, PLR0912
    envelope: dict[str, Any],
    *,
    contract_data: dict[str, Any],
    binding: dict[str, Any],
    producer_commit: str,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate common execution-envelope and raw-record provenance."""
    if envelope.get("schema_version") != EXECUTION_RECORD_SCHEMA_VERSION:
        raise ValueError("execution envelope schema version is unsupported")
    manifest_id = envelope.get("candidate_manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("execution envelope candidate_manifest_id is missing")
    ids = {
        manifest_id
        for arm in _ARMS
        for manifest_id in binding["candidate_manifest_ids_by_arm"][arm]
    }
    if manifest_id not in ids:
        raise ValueError(f"execution envelope candidate {manifest_id} is not selected")
    record = envelope.get("episode_record")
    if not isinstance(record, dict):
        raise ValueError(f"execution envelope {manifest_id} is missing episode_record")
    command = envelope.get("execution_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part.strip() for part in command)
    ):
        raise ValueError(f"execution envelope {manifest_id} has no execution_command")
    config_lineage = envelope.get("execution_config_lineage")
    if not isinstance(config_lineage, dict) or not config_lineage:
        raise ValueError(f"execution envelope {manifest_id} has no execution_config_lineage")
    if not _is_sha256(config_lineage.get("contract_sha256")):
        raise ValueError(f"execution envelope {manifest_id} has no contract SHA-256")
    if config_lineage["contract_sha256"] != contract_data["contract_sha256"]:
        raise ValueError(f"execution envelope {manifest_id} contract SHA-256 mismatch")
    if not _is_sha256(config_lineage.get("bindings_sha256")):
        raise ValueError(f"execution envelope {manifest_id} has no bindings SHA-256")
    if config_lineage["bindings_sha256"] != binding["raw_sha256"]:
        raise ValueError(f"execution envelope {manifest_id} bindings SHA-256 mismatch")
    planner = contract_data["planner"]
    if config_lineage.get("target_planner_config_sha256") != planner["config_sha256"]:
        raise ValueError(f"execution envelope {manifest_id} target config hash mismatch")
    if config_lineage.get("planner_reference_commit") != planner["execution_commit"]:
        raise ValueError(f"execution envelope {manifest_id} planner reference commit mismatch")
    if config_lineage.get("producer_commit") != producer_commit:
        raise ValueError(f"execution envelope {manifest_id} producer commit mismatch")
    expected_identity = _episode_metadata(record, contract_data["identity"])
    observed_producer_commit = _record_producer_commit(record)
    if observed_producer_commit != producer_commit:
        raise ValueError(f"episode record {manifest_id} producer commit mismatch")
    if record.get("scenario_id") is None or record.get("scenario_params") is None:
        raise ValueError(f"episode record {manifest_id} is missing scenario provenance")
    scenario_family = contract_data["contract"]["evaluation"]["scenario_family"]
    if record.get("scenario_id") != scenario_family:
        raise ValueError(f"episode record {manifest_id} scenario_id does not match scenario family")
    scenario_params = record.get("scenario_params")
    if not isinstance(scenario_params, dict):
        raise ValueError(f"episode record {manifest_id} scenario_params must be an object")
    if scenario_params.get("candidate_manifest_id") != manifest_id:
        raise ValueError(
            f"episode record {manifest_id} candidate_manifest_id does not match selected candidate"
        )
    expected_scenario_seed = binding["scenario_seed_by_manifest_id"][manifest_id]
    if (
        _require_integer(
            scenario_params.get("scenario_seed"),
            label=f"episode record {manifest_id} scenario_seed",
        )
        != expected_scenario_seed
    ):
        raise ValueError(
            f"episode record {manifest_id} scenario_seed does not match selected scenario"
        )
    _validate_episode_outcome(record, manifest_id=manifest_id)
    if envelope.get("scenario_family") != scenario_family:
        raise ValueError(f"execution envelope {manifest_id} scenario family mismatch")
    if envelope.get("scenario_certification_status") != "passed":
        raise ValueError(f"execution envelope {manifest_id} scenario certification is not passed")
    identity = {"execution_mode": "adapter", **expected_identity}
    return manifest_id, record, config_lineage, identity


def _selection_info(binding: dict[str, Any], manifest_id: str) -> tuple[str, int, int]:
    """Return arm, one-based rank, and candidate pool index for a selected ID."""
    for arm in _ARMS:
        ids = binding["candidate_manifest_ids_by_arm"][arm]
        if manifest_id in ids:
            return (
                arm,
                ids.index(manifest_id) + 1,
                binding["candidate_pool_index_by_manifest_id"][manifest_id],
            )
    raise ValueError(f"candidate {manifest_id} is not selected")


def _row_id(arm: str, rank: int, seed_offset: int) -> str:
    """Return the frozen row identity used by the preflight packet."""
    return f"{arm}_rank_{rank}_seed_{seed_offset}"


def _build_candidate_rows(  # noqa: C901
    manifest_id: str,
    envelopes: list[dict[str, Any]],
    *,
    contract_data: dict[str, Any],
    binding: dict[str, Any],
    producer_commit: str,
) -> list[dict[str, Any]]:
    """Validate one candidate's replay/confirmation set and build five rows."""
    arm, rank, pool_index = _selection_info(binding, manifest_id)
    replay_envelopes = [item for item in envelopes if item.get("execution_stage") == "replay"]
    confirmations = [item for item in envelopes if item.get("execution_stage") == "confirmation"]
    if len(replay_envelopes) != 1:
        raise ValueError(f"candidate {manifest_id} must have exactly one deterministic replay")
    expected_seeds = binding["execution_seeds_by_manifest_id"][manifest_id]
    if len(confirmations) != len(expected_seeds):
        raise ValueError(
            f"candidate {manifest_id} requires five confirmation executions; "
            f"observed {len(confirmations)}"
        )
    replay_envelope = replay_envelopes[0]
    replay_id, replay_record, _replay_lineage, _replay_identity = _validate_envelope_common(
        replay_envelope,
        contract_data=contract_data,
        binding=binding,
        producer_commit=producer_commit,
    )
    if replay_id != manifest_id:
        raise ValueError("replay candidate ID drifted within the execution envelope")
    if replay_envelope.get("execution_seed") is not None:
        raise ValueError(f"candidate {manifest_id} replay execution_seed must be null")
    replay_seed = _require_integer(
        replay_record.get("seed"), label=f"candidate {manifest_id} replay seed"
    )
    expected_scenario_seed = binding["scenario_seed_by_manifest_id"][manifest_id]
    if replay_seed != expected_scenario_seed:
        raise ValueError(f"candidate {manifest_id} replay seed does not match scenario seed")
    replay_lineage = _validate_replay_lineage(replay_envelope.get("replay_lineage"))
    replay_attr = attribution_from_episode_record(replay_record)
    replay_pair = (str(replay_attr.primary_failure), str(replay_attr.details["termination_reason"]))

    by_seed: dict[int, tuple[dict[str, Any], dict[str, Any], str, str]] = {}
    for envelope in confirmations:
        envelope_id, record, config_lineage, identity = _validate_envelope_common(
            envelope,
            contract_data=contract_data,
            binding=binding,
            producer_commit=producer_commit,
        )
        if envelope_id != manifest_id:
            raise ValueError("confirmation candidate ID drifted within the execution envelope")
        seed = envelope.get("execution_seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed not in expected_seeds:
            raise ValueError(f"candidate {manifest_id} has an unbound confirmation seed {seed!r}")
        if seed in by_seed:
            raise ValueError(f"candidate {manifest_id} has duplicate confirmation seed {seed}")
        if (
            _require_integer(record.get("seed"), label=f"candidate {manifest_id} confirmation seed")
            != seed
        ):
            raise ValueError(f"candidate {manifest_id} record seed does not match execution seed")
        by_seed[int(seed)] = (
            envelope,
            record,
            identity,
            json.dumps(config_lineage, sort_keys=True),
        )
    if set(by_seed) != set(expected_seeds):
        raise ValueError(f"candidate {manifest_id} confirmation seed set is incomplete")

    attributions: dict[int, Any] = {
        seed: attribution_from_episode_record(item[1]) for seed, item in by_seed.items()
    }
    failure_seeds = [
        seed
        for seed, attribution in attributions.items()
        if attribution.primary_failure != "success"
    ]
    failure_pairs = {
        (
            str(attributions[seed].primary_failure),
            str(attributions[seed].details["termination_reason"]),
        )
        for seed in failure_seeds
    }
    confirmed_count = len(failure_seeds)
    if confirmed_count >= 3 and (
        len(failure_pairs) != 1 or next(iter(failure_pairs)) != replay_pair
    ):
        raise ValueError(f"candidate {manifest_id} has unstable replay/confirmation attribution")
    stable_attribution = len(failure_pairs) <= 1 and (
        confirmed_count < 3 or next(iter(failure_pairs)) == replay_pair
    )
    confirmation_lineage = {
        "confirmed_count": confirmed_count,
        "attempt_count": len(expected_seeds),
        "stable_attribution": stable_attribution,
        "execution_seeds": list(expected_seeds),
    }
    rows: list[dict[str, Any]] = []
    for seed_offset, seed in enumerate(expected_seeds):
        envelope, record, identity, config_lineage_json = by_seed[seed]
        config_lineage = json.loads(config_lineage_json)
        attribution = attributions[seed]
        episode_hash = payload_sha256(record)
        rows.append(
            {
                "row_id": _row_id(arm, rank, seed_offset),
                "candidate_manifest_id": manifest_id,
                "candidate_manifest_sha256": binding["candidate_manifest_sha256_by_id"][
                    manifest_id
                ],
                "selection_arm": arm,
                "selection_rank": rank,
                "candidate_pool_seed": binding["candidate_pool_seed"],
                "candidate_pool_index": pool_index,
                "target_planner_id": contract_data["planner"]["id"],
                "target_planner_config_sha256": contract_data["planner"]["config_sha256"],
                "scenario_family": contract_data["contract"]["evaluation"]["scenario_family"],
                "scenario_seed": expected_scenario_seed,
                "execution_seed": seed,
                "execution_commit": contract_data["planner"]["execution_commit"],
                "execution_command": list(envelope["execution_command"]),
                "execution_config_lineage": config_lineage,
                "execution_mode": identity["execution_mode"],
                "execution_identity": {key: identity[key] for key in _IDENTITY_FIELDS},
                "producer_commit": producer_commit,
                "episode_record_sha256": episode_hash,
                "outcome": dict(record["outcome"]),
                "primary_failure": str(attribution.primary_failure),
                "termination_reason": str(attribution.details["termination_reason"]),
                "independent_failure_outcome": attribution.primary_failure != "success",
                "scenario_certification_status": "passed",
                "candidate_certification_status": "passed",
                "replay_lineage": {
                    **replay_lineage,
                    "replay_record_sha256": payload_sha256(replay_record),
                    "replay_seed": expected_scenario_seed,
                },
                "confirmation_lineage": confirmation_lineage,
                "record_sha256": binding["record_sha256_by_manifest_id"][manifest_id],
                "admission_status": "admitted",
                "exclusion_reason": None,
            }
        )
    return rows


def _validated_row_map(rows: Any, *, packet_label: str) -> dict[str, dict[str, Any]]:
    """Return a row map after validating the resumable packet row boundary."""
    if not isinstance(rows, list):
        raise ValueError(f"{packet_label} rows are malformed")
    row_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{packet_label} contains a malformed row")
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{packet_label} contains a row with an invalid row_id")
        if row_id in row_map:
            raise ValueError(f"{packet_label} contains duplicate row {row_id}")
        row_map[row_id] = row
    return row_map


def _merge_existing_packet(existing: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Merge an idempotent rerun without allowing row replacement."""
    if existing.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        raise ValueError("existing output packet has an unsupported schema")
    if existing.get("producer_commit") != current.get("producer_commit"):
        raise ValueError("existing output packet producer_commit differs from current run")
    existing_by_id = _validated_row_map(existing.get("rows"), packet_label="existing output packet")
    current_by_id = _validated_row_map(current.get("rows"), packet_label="current output packet")
    for row_id, row in existing_by_id.items():
        if row_id not in current_by_id:
            raise ValueError(f"existing output packet contains stale row {row_id}")
        if row != current_by_id[row_id]:
            raise ValueError(f"row {row_id} changed during resumable producer rerun")
    merged = dict(current)
    merged["rows"] = [current_by_id[row_id] for row_id in sorted(current_by_id)]
    merged["resumed_from_existing"] = True
    return merged


def build_outcome_packet(
    execution_records: Iterable[dict[str, Any]],
    *,
    contract_path: Path,
    binding_path: Path,
    producer_commit: str,
) -> dict[str, Any]:
    """Build a complete v2 packet from all selected candidate executions."""
    if not _is_sha1(producer_commit):
        raise ValueError("producer_commit must be a 40-character Git SHA-1")
    contract_data = _load_contract_identity(contract_path)
    contract = contract_data["contract"]
    budget = contract["budget"]["candidate_budget_per_arm"]
    binding = _load_binding(binding_path, budget_per_arm=budget)
    expected_ids = {
        manifest_id
        for arm in _ARMS
        for manifest_id in binding["candidate_manifest_ids_by_arm"][arm]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for envelope in execution_records:
        manifest_id = envelope.get("candidate_manifest_id") if isinstance(envelope, dict) else None
        if not isinstance(manifest_id, str) or not manifest_id:
            raise ValueError("every execution envelope needs a candidate_manifest_id")
        if envelope.get("execution_stage") not in {"replay", "confirmation"}:
            raise ValueError(f"execution envelope {manifest_id} has an unsupported execution_stage")
        grouped[manifest_id].append(envelope)
    if set(grouped) != expected_ids:
        missing = sorted(expected_ids - set(grouped))
        extra = sorted(set(grouped) - expected_ids)
        raise ValueError(
            f"execution record candidate coverage mismatch: missing={missing} extra={extra}"
        )
    rows: list[dict[str, Any]] = []
    for manifest_id in sorted(expected_ids):
        rows.extend(
            _build_candidate_rows(
                manifest_id,
                grouped[manifest_id],
                contract_data=contract_data,
                binding=binding,
                producer_commit=producer_commit,
            )
        )
    rows.sort(key=lambda row: str(row["row_id"]))
    packet = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "source": "issue_7066_adapter_outcome_producer",
        "outcome_source": "planner_execution",
        "objective": OUTCOME_OBJECTIVE,
        "target_planner_id": contract_data["planner"]["id"],
        "target_planner_config_sha256": contract_data["planner"]["config_sha256"],
        "execution_commit": contract_data["planner"]["execution_commit"],
        "execution_commit_role": "historical_planner_reference_lineage",
        "execution_mode": "adapter",
        "execution_identity": dict(contract_data["identity"]),
        "producer_commit": producer_commit,
        "contract_path": str(contract_path),
        "contract_sha256": contract_data["contract_sha256"],
        "binding_path": str(binding_path),
        "binding_sha256": binding["raw_sha256"],
        "production_status": "complete",
        "raw_execution_count": len(rows) + len(expected_ids),
        "confirmation_row_count": len(rows),
        "replay_execution_count": len(expected_ids),
        "rows": rows,
    }
    validation = _evaluate_packet(packet, contract_data=contract_data, binding=binding)
    if validation.get("status") != "complete":
        raise ValueError(
            f"generated outcome packet failed v2 admission: {validation.get('reason')}"
        )
    return packet


def _evaluate_packet(
    packet: dict[str, Any], *, contract_data: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    """Run the shared v2 evaluator over the newly generated packet."""
    budget = contract_data["contract"]["budget"]["candidate_budget_per_arm"]
    return build_independent_outcome_evaluation(
        packet,
        budget_per_arm=budget,
        minimally_important=contract_data["contract"]["power_sensitivity"][
            "minimally_important_absolute_yield_improvement"
        ],
        admission_spec=_admission_spec(contract_data, binding),
    )


def _git_head(repo_root: Path) -> str:
    """Resolve the producer commit used by the CLI."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not _is_sha1(result.stdout.strip()):
        raise ValueError("failed to resolve producer Git commit")
    return result.stdout.strip()


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON packet through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    """Parse the producer CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build issue #6105 v2 outcomes from explicit canonical adapter execution records."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--execution-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the fail-closed outcome producer."""
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        producer_commit = _git_head(repo_root)
        records = load_execution_records(args.execution_records)
        packet = build_outcome_packet(
            records,
            contract_path=args.contract,
            binding_path=args.bindings,
            producer_commit=producer_commit,
        )
        packet["execution_records_path"] = str(args.execution_records)
        packet["execution_records_sha256"] = _raw_sha256(args.execution_records)
        if args.output.exists():
            existing = _read_json(args.output)
            packet = _merge_existing_packet(existing, packet)
        _write_json_atomically(args.output, packet)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stdout.write(
            json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True) + "\n"
        )
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "producer_commit": producer_commit,
                "row_count": len(packet["rows"]),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
