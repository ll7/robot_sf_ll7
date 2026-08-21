"""Tests for the issue #7066 canonical-adapter execution-to-v2 outcome bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.independent_outcome_producer import (
    EXECUTION_RECORD_SCHEMA_VERSION,
    _merge_existing_packet,
    _validate_episode_outcome,
    build_outcome_packet,
    load_execution_records,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/adversarial/issue_3275_same_planner_contract.json"
PRODUCER_COMMIT = "a" * 40
REFERENCE_COMMIT = "ecf997d392a4f2c1a4fb5a56e8101acb030b7e2f"
CONFIG_SHA = "dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735"


def _sha256(value: str) -> str:
    """Return a test fixture SHA-256."""
    return hashlib.sha256(value.encode()).hexdigest()


def _contract_and_binding(tmp_path: Path) -> tuple[Path, Path]:
    """Write a one-candidate-per-arm frozen contract and binding fixture."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["budget"] = {
        **contract["budget"],
        "candidate_budget_per_arm": 1,
        "candidate_pool_size": 2,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    ids = {"proposal": ["proposal_0"], "random": ["random_0"]}
    selected = [*ids["proposal"], *ids["random"]]
    binding = {
        "schema_version": "adversarial_candidate_manifest_bindings.v2",
        "candidate_manifest_ids_by_arm": ids,
        "candidate_manifest_sha256_by_id": {
            candidate_id: _sha256(f"manifest:{candidate_id}") for candidate_id in selected
        },
        "candidate_pool_index_by_manifest_id": {selected[0]: 0, selected[1]: 1},
        "scenario_seed_by_manifest_id": {selected[0]: 11, selected[1]: 12},
        "record_sha256_by_manifest_id": {
            candidate_id: _sha256(f"record:{candidate_id}") for candidate_id in selected
        },
        "execution_seeds_by_manifest_id": {
            selected[0]: [100, 101, 102, 103, 104],
            selected[1]: [200, 201, 202, 203, 204],
        },
        "candidate_pool_seed": 42,
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    return contract_path, binding_path


def _episode_record(
    *,
    candidate_id: str,
    scenario_seed: int,
    seed: int,
    failure: bool,
    producer_commit: str = PRODUCER_COMMIT,
) -> dict[str, Any]:
    """Build a minimal benchmark-shaped record with explicit adapter metadata."""
    return {
        "version": "v1",
        "scenario_id": "classic_cross_trap_medium",
        "scenario_params": {
            "candidate_manifest_id": candidate_id,
            "scenario_seed": scenario_seed,
        },
        "seed": seed,
        "algorithm_metadata": {
            "canonical_algorithm": "social_force",
            "policy_semantics": "social_force_adapter",
            "status": "ok",
            "fallback_or_degraded": False,
            "evidence_eligible": True,
            "availability_status": "available",
            "readiness_status": "adapter",
            "preflight_status": "passed",
            "planner_kinematics": {
                "execution_mode": "adapter",
                "adapter_active": True,
                "adapter_name": "SocialForcePlannerAdapter",
                "upstream_command_space": "velocity_vector_xy",
                "benchmark_command_space": "unicycle_vw",
                "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            },
        },
        "provenance": {"git_hash": producer_commit},
        "result_provenance": {"repo_commit": producer_commit},
        "outcome": {
            "route_complete": not failure,
            "collision_event": failure,
            "timeout_event": False,
        },
        "termination_reason": "collision" if failure else "success",
    }


def _envelope(
    candidate_id: str,
    *,
    scenario_seed: int,
    execution_seed: int | None,
    failure: bool,
    stage: str,
    contract_path: Path,
    binding_path: Path,
) -> dict[str, Any]:
    """Build one explicit replay or confirmation envelope."""
    record_seed = scenario_seed if stage == "replay" else int(execution_seed)
    lineage = {
        "contract_path": str(contract_path),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "bindings_path": str(binding_path),
        "bindings_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "target_planner_config_sha256": CONFIG_SHA,
        "planner_reference_commit": REFERENCE_COMMIT,
        "producer_commit": PRODUCER_COMMIT,
    }
    envelope: dict[str, Any] = {
        "schema_version": EXECUTION_RECORD_SCHEMA_VERSION,
        "candidate_manifest_id": candidate_id,
        "execution_stage": stage,
        "scenario_family": "classic_cross_trap_medium",
        "scenario_certification_status": "passed",
        "execution_seed": execution_seed,
        "execution_command": ["uv", "run", "robot_sf_bench", "run"],
        "execution_config_lineage": lineage,
        "episode_record": _episode_record(
            candidate_id=candidate_id,
            scenario_seed=scenario_seed,
            seed=record_seed,
            failure=failure,
        ),
    }
    if stage == "replay":
        signature = _sha256(f"replay:{candidate_id}")
        envelope["replay_lineage"] = {
            "exact_signature_match": True,
            "original_signature_sha256": signature,
            "replay_signature_sha256": signature,
        }
    return envelope


def _execution_records(contract_path: Path, binding_path: Path) -> list[dict[str, Any]]:
    """Build two complete candidates: proposal fails and random succeeds."""
    records: list[dict[str, Any]] = []
    for candidate_id, scenario_seed, seeds, failure in (
        ("proposal_0", 11, [100, 101, 102, 103, 104], True),
        ("random_0", 12, [200, 201, 202, 203, 204], False),
    ):
        records.append(
            _envelope(
                candidate_id,
                scenario_seed=scenario_seed,
                execution_seed=None,
                failure=failure,
                stage="replay",
                contract_path=contract_path,
                binding_path=binding_path,
            )
        )
        records.extend(
            _envelope(
                candidate_id,
                scenario_seed=scenario_seed,
                execution_seed=seed,
                failure=failure,
                stage="confirmation",
                contract_path=contract_path,
                binding_path=binding_path,
            )
            for seed in seeds
        )
    return records


def test_producer_emits_complete_adapter_packet_with_separate_commits(tmp_path: Path) -> None:
    """Valid canonical-adapter records pass the shared v2 admission evaluator."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    packet = build_outcome_packet(
        _execution_records(contract_path, binding_path),
        contract_path=contract_path,
        binding_path=binding_path,
        producer_commit=PRODUCER_COMMIT,
    )

    assert packet["production_status"] == "complete"
    assert len(packet["rows"]) == 10
    assert packet["execution_commit"] == REFERENCE_COMMIT
    assert packet["producer_commit"] == PRODUCER_COMMIT
    assert all(row["execution_mode"] == "adapter" for row in packet["rows"])
    assert all(
        row["execution_identity"]["adapter_name"] == "SocialForcePlannerAdapter"
        for row in packet["rows"]
    )
    assert all(row["execution_commit"] != row["producer_commit"] for row in packet["rows"])


def test_producer_cli_reads_jsonl_and_writes_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public producer entry point validates JSON Lines and writes the packet."""
    from robot_sf.adversarial import independent_outcome_producer as producer

    contract_path, binding_path = _contract_and_binding(tmp_path)
    records_path = tmp_path / "execution-records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in _execution_records(contract_path, binding_path)
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "independent_outcomes.json"
    monkeypatch.setattr(producer, "_git_head", lambda _repo_root: PRODUCER_COMMIT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize_issue_6105_outcomes.py",
            "--contract",
            str(contract_path),
            "--bindings",
            str(binding_path),
            "--execution-records",
            str(records_path),
            "--output",
            str(output_path),
        ],
    )

    assert producer.main() == 0
    packet = json.loads(output_path.read_text(encoding="utf-8"))
    assert packet["production_status"] == "complete"
    assert (
        packet["execution_records_sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("contents", "expected_fragment"),
    [
        ("", "missing or empty"),
        ("not-json\n", "line 1 is malformed"),
        ("[]\n", "line 1 must be an object"),
        ("\n", "JSONL is empty"),
    ],
)
def test_execution_record_loader_rejects_partial_or_malformed_input(
    tmp_path: Path, contents: str, expected_fragment: str
) -> None:
    """The JSON Lines boundary rejects incomplete or non-object input."""
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_fragment):
        load_execution_records(records_path)


def test_producer_rejects_common_envelope_provenance_drift(tmp_path: Path) -> None:
    """Every envelope must carry exact schema, command, and configuration lineage."""
    mutations = (
        ("schema_version", "schema version is unsupported"),
        ("episode_record", "missing episode_record"),
        ("execution_command", "no execution_command"),
        ("execution_config_lineage", "no execution_config_lineage"),
        ("contract_sha256", "contract SHA-256 mismatch"),
        ("bindings_sha256", "bindings SHA-256 mismatch"),
        ("target_planner_config_sha256", "target config hash mismatch"),
        ("planner_reference_commit", "planner reference commit mismatch"),
        ("producer_commit", "producer commit mismatch"),
        ("scenario_family", "scenario family mismatch"),
        ("scenario_certification_status", "scenario certification is not passed"),
    )
    for field, expected_fragment in mutations:
        case_dir = tmp_path / field
        case_dir.mkdir()
        contract_path, binding_path = _contract_and_binding(case_dir)
        records = _execution_records(contract_path, binding_path)
        if field in {"contract_sha256", "bindings_sha256", "target_planner_config_sha256"}:
            records[0]["execution_config_lineage"][field] = "b" * 64
        elif field in {"planner_reference_commit", "producer_commit"}:
            records[0]["execution_config_lineage"][field] = "b" * 40
        elif field == "scenario_family":
            records[0][field] = "other_family"
        elif field == "scenario_certification_status":
            records[0][field] = "not_passed"
        elif field == "schema_version":
            records[0][field] = "unsupported"
        elif field == "execution_command":
            records[0][field] = []
        else:
            records[0][field] = None

        with pytest.raises(ValueError, match=expected_fragment):
            build_outcome_packet(
                records,
                contract_path=contract_path,
                binding_path=binding_path,
                producer_commit=PRODUCER_COMMIT,
            )


def test_producer_rejects_episode_record_provenance_drift(tmp_path: Path) -> None:
    """Episode metadata and producer provenance are independently fail-closed."""
    mutations = (
        ("algorithm_metadata", None, "missing algorithm_metadata"),
        ("canonical_algorithm", "other", "canonical_algorithm is not social_force"),
        ("planner_kinematics", None, "missing planner_kinematics"),
        ("scenario_id", None, "missing scenario provenance"),
        ("outcome", None, "outcome must use exactly the canonical fields"),
        ("termination_reason", None, "termination_reason is not canonical"),
    )
    for field, value, expected_fragment in mutations:
        case_dir = tmp_path / field
        case_dir.mkdir()
        contract_path, binding_path = _contract_and_binding(case_dir)
        records = _execution_records(contract_path, binding_path)
        episode = records[0]["episode_record"]
        if field in {"canonical_algorithm", "policy_semantics", "planner_kinematics"}:
            if field == "planner_kinematics":
                episode["algorithm_metadata"][field] = value
            else:
                episode["algorithm_metadata"][field] = value
        else:
            episode[field] = value

        with pytest.raises(ValueError, match=expected_fragment):
            build_outcome_packet(
                records,
                contract_path=contract_path,
                binding_path=binding_path,
                producer_commit=PRODUCER_COMMIT,
            )


@pytest.mark.parametrize(
    ("field", "value", "expected_fragment"),
    [
        ("scenario_id", "other_scenario", "scenario_id does not match scenario family"),
        (
            "candidate_manifest_id",
            "other_candidate",
            "candidate_manifest_id does not match selected candidate",
        ),
        ("scenario_seed", 999, "scenario_seed does not match selected scenario"),
    ],
)
def test_producer_rejects_selected_candidate_scenario_drift(
    tmp_path: Path, field: str, value: Any, expected_fragment: str
) -> None:
    """Episode records must bind to the selected candidate and frozen scenario."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    episode = records[0]["episode_record"]
    if field == "scenario_id":
        episode[field] = value
    else:
        episode["scenario_params"][field] = value

    with pytest.raises(ValueError, match=expected_fragment):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ("scenario_seed_float", "scenario_seed must be an integer"),
        ("scenario_seed_bool", "scenario_seed must be an integer"),
        ("replay_record_seed_float", "seed must be an integer"),
        ("confirmation_record_seed_float", "seed must be an integer"),
    ],
)
def test_producer_rejects_non_integer_seed_lineage(
    tmp_path: Path, mutation: str, expected_fragment: str
) -> None:
    """JSON numeric lookalikes cannot bypass integer seed bindings."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    if mutation == "scenario_seed_float":
        records[0]["episode_record"]["scenario_params"]["scenario_seed"] = 11.0
    elif mutation == "scenario_seed_bool":
        records[0]["episode_record"]["scenario_params"]["scenario_seed"] = True
    elif mutation == "replay_record_seed_float":
        records[0]["episode_record"]["seed"] = 11.0
    else:
        records[1]["episode_record"]["seed"] = 100.0

    with pytest.raises(ValueError, match=expected_fragment):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


def test_producer_rejects_legacy_outcome_aliases(tmp_path: Path) -> None:
    """Legacy outcome aliases cannot bypass the canonical episode contract."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    records[0]["episode_record"]["outcome"]["collision"] = True

    with pytest.raises(ValueError, match="outcome must use exactly the canonical fields"):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ({"route_complete": "false"}, "outcome fields must be boolean"),
        ({"collision_event": 1}, "outcome fields must be boolean"),
        ({"timeout_event": None}, "outcome fields must be boolean"),
        ({"termination_reason": "goal_reached"}, "termination_reason is not canonical"),
        ({"termination_reason": "error"}, "runtime error is not an outcome"),
    ],
)
def test_producer_rejects_noncanonical_outcome_values(
    tmp_path: Path, mutation: dict[str, Any], expected_fragment: str
) -> None:
    """Non-boolean flags and legacy termination labels fail closed."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    episode = records[0]["episode_record"]
    if "termination_reason" in mutation:
        episode["termination_reason"] = mutation["termination_reason"]
    else:
        field, value = next(iter(mutation.items()))
        episode["outcome"][field] = value

    with pytest.raises(ValueError, match=expected_fragment):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


@pytest.mark.parametrize(
    ("termination_reason", "outcome", "expected_failure"),
    [
        (
            "success",
            {"route_complete": True, "collision_event": False, "timeout_event": False},
            "success",
        ),
        (
            "collision",
            {"route_complete": False, "collision_event": True, "timeout_event": False},
            "collision",
        ),
        (
            "terminated",
            {"route_complete": False, "collision_event": False, "timeout_event": True},
            "timeout",
        ),
        (
            "truncated",
            {"route_complete": False, "collision_event": False, "timeout_event": True},
            "timeout",
        ),
        (
            "max_steps",
            {"route_complete": False, "collision_event": False, "timeout_event": True},
            "timeout",
        ),
    ],
)
def test_validate_episode_outcome_matches_attribution_policy(
    termination_reason: str, outcome: dict[str, bool], expected_failure: str
) -> None:
    """Every accepted canonical reason maps to the same attribution category."""
    record = {"termination_reason": termination_reason, "outcome": outcome}

    _validate_episode_outcome(record, manifest_id="candidate_0")

    assert attribution_from_episode_record(record).primary_failure == expected_failure


def test_producer_rejects_terminated_without_timeout_flag() -> None:
    """A terminal episode cannot silently become an unclassified incomplete row."""
    record = {
        "termination_reason": "terminated",
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": False},
    }

    with pytest.raises(ValueError, match="timeout_event disagrees with termination_reason"):
        _validate_episode_outcome(record, manifest_id="candidate_0")


def test_producer_rejects_replay_and_confirmation_seed_drift(tmp_path: Path) -> None:
    """Replay signatures and confirmation seeds must match the frozen design."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    mutations = (
        ("replay_seed", "replay execution_seed must be null"),
        ("replay_record_seed", "replay seed does not match scenario seed"),
        ("replay_lineage", "missing replay_lineage"),
        ("confirmation_seed", "unbound confirmation seed"),
        ("confirmation_record_seed", "record seed does not match execution seed"),
    )
    for mutation, expected_fragment in mutations:
        records = _execution_records(contract_path, binding_path)
        if mutation == "replay_seed":
            records[0]["execution_seed"] = 999
        elif mutation == "replay_record_seed":
            records[0]["episode_record"]["seed"] = 999
        elif mutation == "replay_lineage":
            records[0].pop("replay_lineage")
        elif mutation == "confirmation_seed":
            records[1]["execution_seed"] = 999
        else:
            records[1]["episode_record"]["seed"] = 999

        with pytest.raises(ValueError, match=expected_fragment):
            build_outcome_packet(
                records,
                contract_path=contract_path,
                binding_path=binding_path,
                producer_commit=PRODUCER_COMMIT,
            )


@pytest.mark.parametrize(
    ("metadata_change", "expected_fragment"),
    [
        ({"execution_mode": "native"}, "execution_mode is not adapter"),
        ({"execution_mode": "fallback"}, "execution_mode is not adapter"),
        ({"adapter_active": False}, "adapter_active must be true"),
        ({"adapter_name": "OtherAdapter"}, "canonical adapter identity mismatch"),
        ({"upstream_command_space": "world_velocity"}, "canonical adapter identity mismatch"),
        ({"policy_semantics": "social_force_native"}, "policy_semantics does not match"),
    ],
)
def test_producer_rejects_noncanonical_execution(
    tmp_path: Path, metadata_change: dict[str, Any], expected_fragment: str
) -> None:
    """Native, fallback, and mismatched adapter rows never enter the packet."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    metadata = records[1]["episode_record"]["algorithm_metadata"]
    if "policy_semantics" in metadata_change:
        metadata.update(metadata_change)
    else:
        metadata["planner_kinematics"].update(metadata_change)

    with pytest.raises(ValueError, match=expected_fragment):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


@pytest.mark.parametrize(
    ("metadata_field", "metadata_value", "expected_fragment"),
    [
        ("status", "fallback", "status is not ok"),
        ("fallback_or_degraded", True, "fallback_or_degraded"),
        ("evidence_eligible", False, "evidence_eligible"),
        ("availability_status", "unavailable", "availability_status is not available"),
        ("readiness_status", "native", "readiness_status is not adapter-ready"),
        ("preflight_status", "failed", "preflight_status is not successful"),
    ],
)
def test_producer_rejects_unavailable_or_ineligible_adapter_metadata(
    tmp_path: Path,
    metadata_field: str,
    metadata_value: Any,
    expected_fragment: str,
) -> None:
    """Availability and evidence-eligibility metadata fail closed before admission."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    records[1]["episode_record"]["algorithm_metadata"][metadata_field] = metadata_value

    with pytest.raises(ValueError, match=expected_fragment):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


def test_producer_rejects_partial_or_duplicate_confirmation_lineage(tmp_path: Path) -> None:
    """A missing or duplicate execution seed blocks the complete packet."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    records.pop(2)
    with pytest.raises(ValueError, match="requires five confirmation"):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )

    records = _execution_records(contract_path, binding_path)
    records[2]["execution_seed"] = records[1]["execution_seed"]
    with pytest.raises(ValueError, match="duplicate confirmation seed"):
        build_outcome_packet(
            records,
            contract_path=contract_path,
            binding_path=binding_path,
            producer_commit=PRODUCER_COMMIT,
        )


def test_resumable_merge_is_idempotent_and_rejects_row_replacement(tmp_path: Path) -> None:
    """Rerunning the producer cannot silently replace an existing row."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    packet = build_outcome_packet(
        _execution_records(contract_path, binding_path),
        contract_path=contract_path,
        binding_path=binding_path,
        producer_commit=PRODUCER_COMMIT,
    )
    merged = _merge_existing_packet(packet, packet)
    assert merged["resumed_from_existing"] is True
    assert merged["rows"] == packet["rows"]

    changed = json.loads(json.dumps(packet))
    changed["rows"][0]["producer_commit"] = "b" * 40
    with pytest.raises(ValueError, match="changed during resumable"):
        _merge_existing_packet(packet, changed)

    malformed = json.loads(json.dumps(packet))
    malformed["rows"][0]["row_id"] = None
    with pytest.raises(ValueError, match="invalid row_id"):
        _merge_existing_packet(malformed, packet)

    stale = json.loads(json.dumps(packet))
    stale["rows"].append({"row_id": "stale-row"})
    with pytest.raises(ValueError, match="stale row"):
        _merge_existing_packet(stale, packet)
