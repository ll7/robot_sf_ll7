"""Focused outcome-free contract tests for issue #8891."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.matched_budget_packet import (
    PacketError,
    build_canary_packet,
    build_expected_identities,
    canonical_sha256,
    validate_call_ledger,
    validate_packet,
    validate_result_rows,
    validate_temporal_sidecar,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "configs/adversarial/issue_8891_temporal_robustness_packet.yaml"


@pytest.fixture()
def packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def test_packet_is_source_bound_and_diagnostic_only(packet: dict) -> None:
    result = validate_packet(packet, repo_root=ROOT)
    assert result["status"] == "ok" and not result["campaign_execution_allowed"]
    assert packet["source"]["base_commit"] == "1b94a1a780e4fa4788ed27ca0a0cecb567c75841"


def test_identity_builder_is_deterministic_and_complete(packet: dict) -> None:
    first = build_expected_identities(packet, repo_root=ROOT)
    assert first == build_expected_identities(packet, repo_root=ROOT)
    assert (first["run_count"], first["candidate_slot_count"]) == (54, 2016)
    candidate_ids = [
        slot["candidate_id"] for run in first["runs"] for slot in run["candidate_slots"]
    ]
    assert len(candidate_ids) == 2016
    seeds = [
        seed
        for run in first["runs"]
        for slot in run["candidate_slots"]
        for seed in [slot["replay_seed"], *slot["confirmation_seeds"]]
    ]
    assert len(seeds) == len(set(seeds)) and not set(seeds) & set(
        packet["seed_policy"]["search_seeds"]
    )


def test_canary_is_disjoint_planned_data_and_uses_every_gate(packet: dict) -> None:
    canary = build_canary_packet(packet, repo_root=ROOT)
    assert canary == build_canary_packet(packet, repo_root=ROOT)
    assert (
        canary["claim_eligible"],
        canary["candidate_count"],
        canary["ledger"]["simulator_invocations"],
        {item["search_family"] for item in canary["candidates"]},
    ) == (False, 6, 42, {"random", "optuna", "cmaes"})
    seeds = [
        seed
        for item in canary["candidates"]
        for seed in [item["search_seed"], item["replay_seed"], *item["confirmation_seeds"]]
    ]
    assert len(seeds) == len(set(seeds))


def _sidecar(packet: dict, candidate_id: str, *, artifact_only: bool = False) -> dict:
    sidecar = copy.deepcopy(
        next(
            item["temporal_sidecar"]
            for item in build_canary_packet(packet, repo_root=ROOT)["candidates"]
            if "temporal_sidecar" in item
        )
    )
    sidecar.update(status="observed", candidate_id=candidate_id, admission_status="not_admitted")
    for item in sidecar["properties"]:
        item.update(
            signed_margin=-0.1,
            activation_time_s=0.2,
            execution_mode="native",
            certification_state="passed",
            replay_state="passed",
            independent_seed_state="passed",
            provenance={"source": "fixture", "packet_digest": canonical_sha256(packet)},
        )
    sidecar["monitor"].update(sample_count=3, artifact_only=artifact_only)
    return sidecar


def _bad(call, match: str) -> None:
    with pytest.raises(PacketError, match=match):
        call()


def test_sidecar_rejects_property_drift_monitor_artifact_and_weak_admission(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    broken = copy.deepcopy(sidecar)
    broken["property_ids"][-1] = "different_property"
    _bad(lambda: validate_temporal_sidecar(broken, packet, candidate_id="fixture"), "property IDs")
    _bad(
        lambda: validate_temporal_sidecar(
            _sidecar(packet, "fixture", artifact_only=True), packet, candidate_id="fixture"
        ),
        "monitor-only",
    )
    broken = copy.deepcopy(sidecar)
    broken["properties"][0]["provenance"]["packet_digest"] = "a" * 64
    _bad(
        lambda: validate_temporal_sidecar(broken, packet, candidate_id="fixture"),
        "packet provenance",
    )
    broken = copy.deepcopy(sidecar)
    broken["properties"][0]["execution_mode"] = "fallback"
    broken["admission_status"] = "confirmed_failure"
    _bad(
        lambda: validate_temporal_sidecar(broken, packet, candidate_id="fixture"),
        "fallback/degraded",
    )
    sidecar["admission_status"], sidecar["failure_basis"] = "confirmed_failure", "objective_value"
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id="fixture"),
        "cannot establish failure",
    )


def test_ledger_rejects_hidden_retry_duplicate_and_non_native_admission(packet: dict) -> None:
    rows = build_canary_packet(packet, repo_root=ROOT)["rows"]
    retry = copy.deepcopy(rows)
    retry[0]["retry_of"] = "prior"
    _bad(lambda: validate_call_ledger(packet, retry), "hidden retries")
    _bad(lambda: validate_call_ledger(packet, [*rows, copy.deepcopy(rows[0])]), "duplicate search")
    weak = copy.deepcopy(rows)
    weak[0].update(execution_mode="fallback", status="observed", admission_status="confirmed")
    _bad(lambda: validate_call_ledger(packet, weak), "must be excluded")


def _result_row(packet: dict, identities: dict) -> dict:
    run = next(item for item in identities["runs"] if item["objective_id"] == "temporal_robustness")
    slot = run["candidate_slots"][0]
    return {
        "schema_version": "temporal-robustness-call-ledger.v1",
        "candidate_id": slot["candidate_id"],
        "packet_digest": identities["packet_digest"],
        "identity_sha256": identities["identity_sha256"],
        "objective_id": run["objective_id"],
        "run_id": run["run_id"],
        "attempt_index": 0,
        "phase": "search",
        "call_class": "search_evaluation",
        "simulator_invocations": 1,
        "simulator_call_id": "fixture-call",
        "seed": run["search_seed"],
        "seed_role": "search",
        "execution_mode": "native",
        "retry_of": None,
        "post_outcome_change": False,
        "temporal_sidecar": _sidecar(packet, slot["candidate_id"]),
    }


def test_result_lineage_rejects_post_outcome_identity_change(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    row = _result_row(packet, identities)
    row["post_outcome_change"] = True
    _bad(lambda: validate_result_rows(packet, [row], identities), "post-outcome")
    row["post_outcome_change"] = False
    row.pop("temporal_sidecar")
    _bad(lambda: validate_result_rows(packet, [row], identities), "temporal_sidecar")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("budget", "budgets"), [16, 32, 63], "budget grid"),
        (("seed_policy", "search_seeds"), [1101, 1101, 3303], "search seeds"),
        (("monitor_contract", "property_ids"), ["clearance"], "property IDs"),
        (("objectives",), [{"id": "new_objective"}], "roster"),
    ],
)
def test_packet_mutations_fail_closed(
    packet: dict, path: tuple[str, ...], value: object, message: str
) -> None:
    broken = copy.deepcopy(packet)
    target = broken
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _bad(lambda: validate_packet(broken, repo_root=ROOT), message)
