"""Focused outcome-free contract tests for issue #8891."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.matched_budget_packet import (
    TEMPORAL_SIDECAR_SCHEMA_VERSION,
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
    assert (
        result["status"] == "ok"
        and result["simulator_call_budget"] == 14112
        and not result["campaign_execution_allowed"]
    )
    assert packet["source"]["base_commit"] == "69580e4837ac96e9f658da92eb19e8b1a3a76950"


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


def test_temporal_sidecar_uses_a_distinct_versioned_schema(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    assert sidecar["schema_version"] == TEMPORAL_SIDECAR_SCHEMA_VERSION
    runtime_shape = copy.deepcopy(sidecar)
    runtime_shape["schema_version"] = "robustness-report.v1"
    _bad(
        lambda: validate_temporal_sidecar(runtime_shape, packet, candidate_id="fixture"),
        "temporal sidecar schema",
    )


def test_confirmed_failure_rejects_all_positive_signed_margins(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    sidecar["admission_status"] = "confirmed_failure"
    sidecar["failure_basis"] = "independent_confirmation"
    for property_item in sidecar["properties"]:
        property_item["signed_margin"] = 0.1
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id="fixture"),
        "negative signed margin",
    )


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
    over_budget = copy.deepcopy(packet)
    over_budget["budget"]["simulator_call_budget"] = 41
    _bad(lambda: validate_call_ledger(over_budget, rows), "simulator call budget")


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


def _result_lineage_rows(
    packet: dict,
    identities: dict,
    *,
    confirmed: bool = False,
    confirmation_states: list[str] | None = None,
) -> list[dict]:
    search = _result_row(packet, identities)
    run = next(item for item in identities["runs"] if item["run_id"] == search["run_id"])
    slot = next(
        item for item in run["candidate_slots"] if item["candidate_id"] == search["candidate_id"]
    )
    sidecar = search["temporal_sidecar"]
    sidecar["admission_status"] = "confirmed_failure" if confirmed else "not_admitted"
    if confirmed:
        sidecar["failure_basis"] = "independent_confirmation"
    rows = [search]
    certification = copy.deepcopy(search)
    certification.update(
        phase="certification",
        call_class="certification",
        seed=None,
        seed_role="none",
        simulator_invocations=0,
        simulator_call_id=None,
        certification_state="passed",
    )
    rows.append(certification)
    replay = copy.deepcopy(search)
    replay.update(
        phase="replay",
        call_class="deterministic_replay",
        seed=slot["replay_seed"],
        seed_role="replay",
        simulator_invocations=1,
        simulator_call_id="fixture-replay-call",
        replay_state="passed",
    )
    rows.append(replay)
    states = confirmation_states or ["passed"] * 5
    for offset, (seed, state) in enumerate(zip(slot["confirmation_seeds"], states, strict=True)):
        confirmation = copy.deepcopy(search)
        confirmation.update(
            phase="confirmation",
            call_class="independent_confirmation",
            seed=seed,
            seed_role="confirmation",
            simulator_invocations=1,
            simulator_call_id=f"fixture-confirmation-call-{offset}",
            independent_seed_state=state,
        )
        rows.append(confirmation)
    return rows


def test_result_lineage_requires_all_gate_records(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    search_only = _result_row(packet, identities)
    _bad(
        lambda: validate_result_rows(packet, [search_only], identities),
        "certification lineage",
    )
    rows = _result_lineage_rows(packet, identities)
    result = validate_result_rows(packet, rows, identities)
    assert (result["row_count"], result["simulator_invocations"]) == (8, 7)


def test_confirmed_result_lineage_enforces_three_of_five(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    rows = _result_lineage_rows(
        packet,
        identities,
        confirmed=True,
        confirmation_states=["passed", "passed", "failed", "failed", "failed"],
    )
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        "3-of-5 threshold",
    )


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda packet: packet.update(status="untrusted_status"), "packet metadata.status"),
        (lambda packet: packet.update(unreviewed_claim="failure"), "unsupported fields"),
        (lambda packet: packet["source"].update(base_commit="0" * 40), "cannot be resolved"),
        (
            lambda packet: packet["budget"].update(simulator_call_budget=1),
            "budget grid",
        ),
        (
            lambda packet: packet["analysis_contract"].update(
                denominator="scheduled_search_attempt_slots"
            ),
            "analysis",
        ),
    ],
)
def test_packet_metadata_budget_and_source_are_fail_closed(
    packet: dict, mutation, message: str
) -> None:
    broken = copy.deepcopy(packet)
    mutation(broken)
    _bad(lambda: validate_packet(broken, repo_root=ROOT), message)


def test_cli_requires_one_explicit_operation_and_uses_check() -> None:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/validation/check_issue_8891_temporal_robustness_packet.py"),
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    checked = run("--check", "--format", "json")
    assert checked.returncode == 0
    assert json.loads(checked.stdout)["operation"] == "check"
    missing = run("--format", "json")
    assert missing.returncode == 2
    assert "one of the arguments --check --identities --canary is required" in missing.stderr
    conflicting = run("--check", "--identities", "--format", "json")
    assert conflicting.returncode == 2
    assert "not allowed with argument" in conflicting.stderr
