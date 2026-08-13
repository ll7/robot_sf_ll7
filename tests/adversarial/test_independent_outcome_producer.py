"""Tests for the issue #7066 adapter execution-to-v2 outcome bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.independent_outcome_producer import (
    EXECUTION_RECORD_SCHEMA_VERSION,
    _merge_existing_packet,
    build_outcome_packet,
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
    *, seed: int, failure: bool, producer_commit: str = PRODUCER_COMMIT
) -> dict[str, Any]:
    """Build a minimal benchmark-shaped record with canonical adapter metadata."""
    return {
        "version": "v1",
        "scenario_id": "classic_cross_trap_medium",
        "scenario_params": {"candidate_manifest_id": "fixture"},
        "seed": seed,
        "algorithm_metadata": {
            "canonical_algorithm": "social_force",
            "policy_semantics": "social_force_adapter",
            "planner_kinematics": {
                "execution_mode": "adapter",
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
            "collision": failure,
            "timeout": False,
        },
        "termination_reason": "collision" if failure else "goal_reached",
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
        "episode_record": _episode_record(seed=record_seed, failure=failure),
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
    """Valid adapter records pass the shared v2 admission evaluator."""
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
    ("metadata_change", "expected_fragment"),
    [
        ({"execution_mode": "native"}, "execution_mode is not adapter"),
        ({"execution_mode": "fallback"}, "execution_mode is not adapter"),
        ({"adapter_name": "OtherAdapter"}, "canonical adapter identity mismatch"),
    ],
)
def test_producer_rejects_noncanonical_execution(
    tmp_path: Path, metadata_change: dict[str, str], expected_fragment: str
) -> None:
    """Native aliases, fallback, and identity drift never enter the adapter packet."""
    contract_path, binding_path = _contract_and_binding(tmp_path)
    records = _execution_records(contract_path, binding_path)
    kinematics = records[1]["episode_record"]["algorithm_metadata"]["planner_kinematics"]
    kinematics.update(metadata_change)

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
