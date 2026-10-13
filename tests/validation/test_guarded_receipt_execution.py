"""Authenticated source-defined guarded mixed mode and exact diagnostic leaves."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from scripts.dev.merge_queue_gate import metadata_digest, metadata_trailer
from tests.dev.test_merge_queue_gate import queue_equivalence as _queue_equivalence
from tests.validation.receipt_guarded_fixture import guarded_episode
from tests.validation.test_default_path_equivalence import (
    _episode,
    committed_control_body,
    rewrite_source_archive,
)

queue_equivalence = _queue_equivalence

NATIVE_IDENTITY_PATHS = (
    ("algorithm_metadata", "planner_runtime", "planner_type"),
    ("algorithm_metadata", "policy_semantics"),
    ("event_ledger", "planner"),
    ("scenario_params", "algo"),
    ("algorithm_metadata", "paired_effect_metric_producer", "pairing", "key", "planner"),
    ("algorithm_metadata", "planner_runtime", "fallback_diagnostics", "planner_type"),
)


def _at(row, path):
    for key in path:
        row = row[key]
    return row


def _row():
    return guarded_episode(_episode())


def test_native_guarded_mixed_execution_preserves_mode_and_bytes():
    row = _row()
    before = copy.deepcopy(row)
    evidence = producer._execution_evidence(row)
    assert evidence["execution_mode"] == "mixed"
    assert producer._audit_status([evidence]) == "pass"
    producer.require_valid_execution(
        evidence, "guarded_ppo", enrich_algorithm_metadata(algo="guarded_ppo")["planner_kinematics"]
    )
    assert row == before


def test_native_guarded_exact_calibration_contract_is_not_degraded():
    row = _row()
    assert (
        producer._controller_runtime_marker(row, row["algorithm_metadata"], "guarded_ppo") is None
    )


def test_guarded_mixed_comparison_keeps_all_native_outcome_bytes():
    scope = {"arms": ["guarded_ppo"], "maps": ["open"]}
    rows = {("guarded_ppo", "open", seed): _row() for seed in eq.SEEDS}
    before = copy.deepcopy(rows)
    compared = eq.compare_rows(rows, copy.deepcopy(rows), scope)
    assert all(row["head_execution"]["execution_mode"] == "mixed" for row in compared)
    assert rows == before
    changed = copy.deepcopy(rows)
    changed[next(iter(changed))]["algorithm_metadata"]["guard_stats"] = {"fallback_best_effort": 1}
    with pytest.raises(ValueError, match="changed behavioral byte"):
        eq.compare_rows(changed, rows, scope)


@pytest.mark.parametrize("path", NATIVE_IDENTITY_PATHS)
@pytest.mark.parametrize("fault", ["changed", "missing"])
def test_guarded_native_identity_leaves_are_required_and_consistent(path, fault):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    owner = _at(row, path[:-1])
    if fault == "missing":
        del owner[path[-1]]
    else:
        owner[path[-1]] = "ppo"
    evidence = producer._execution_evidence(row)
    assert evidence["execution_mode"] == "unknown"
    assert producer._audit_status([evidence]) == "fail"


@pytest.mark.parametrize(
    "label",
    [
        "ppo_clear",
        "ppo_safe",
        "goal_reached",
        "fallback_safe",
        "fallback_best_effort",
        "stop_safe",
        "stop_best_effort",
    ],
)
def test_native_guarded_decision_policy_is_bound_to_standard_source_labels(label):
    row = guarded_episode(_episode(), label=label)
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    for owner in ("planner_runtime", "shield_stats"):
        state = row["algorithm_metadata"][owner]["last_decision"]["fallback_controller_state"]
        assert state["policy"] == ("stop" if label.startswith("stop_") else "RiskDWAPlannerAdapter")


@pytest.mark.parametrize("owner", ["planner_runtime", "shield_stats"])
@pytest.mark.parametrize(
    "fault",
    [
        "wrong_policy",
        "missing_policy",
        "prior",
        "unknown_label",
        "missing_label",
        "optional_prior_label",
    ],
)
def test_guarded_decision_policy_requires_exact_present_native_source_relation(owner, fault):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    decision = row["algorithm_metadata"][owner]["last_decision"]
    state = decision["fallback_controller_state"]
    if fault == "wrong_policy":
        state["policy"] = "ppo"
    elif fault == "missing_policy":
        del state["policy"]
    elif fault == "prior":
        state["prior_available"] = True
    elif fault == "missing_label":
        del decision["decision_label"]
    else:
        decision["decision_label"] = "prior_safe" if fault == "optional_prior_label" else "unknown"
    assert producer._execution_evidence(row)["execution_mode"] == "unknown"


def test_guarded_last_decision_mirrors_must_agree_even_if_individually_source_defined():
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    decision = row["algorithm_metadata"]["shield_stats"]["last_decision"]
    decision["decision_label"] = "stop_safe"
    decision["fallback_controller_state"]["policy"] = "stop"
    assert producer._execution_evidence(row)["execution_mode"] == "unknown"


@pytest.mark.parametrize("path", NATIVE_IDENTITY_PATHS)
@pytest.mark.parametrize("fault", ["changed", "missing"])
def test_live_pr_and_queue_refuse_coherently_resealed_guarded_identity_conflicts(
    queue_equivalence, tmp_path, monkeypatch, capsys, path, fault
):
    snapshot, _state, invoke, base, _head, header, payload = queue_equivalence
    assert invoke() == 0
    capsys.readouterr()

    def mutate(root):
        raw = next(root.glob("campaigns/**/guarded_ppo__differential_drive/episodes.jsonl"))
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        owner = _at(rows[0], path[:-1])
        if fault == "missing":
            del owner[path[-1]]
        else:
            owner[path[-1]] = "ppo"
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))

    for side in ("head", "baseline"):
        claims = [copy.deepcopy(row[f"{side}_execution"]) for row in payload["rows"]]
        rewrite_source_archive(
            tmp_path, payload, side, mutate, synchronize=True, suffix="identity-" + fault
        )
        for row, claim in zip(payload["rows"], claims, strict=True):
            row[f"{side}_execution"] = claim
    assert all(row["head_sha256"] == row["baseline_sha256"] for row in payload["rows"])
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = producer.behaviour_receipt.check_receipt(
        body, ["robot_sf/benchmark/camera_ready/_run_state.py"], "test/repo", base
    )
    head = json.loads(body.split("\n")[1])["head_sha"]
    metadata = metadata_digest("merge queue test PR", body)
    snapshot.update(
        body=body,
        head_sha=head,
        gate_verdicts=[f"gate-verdict: accepted @ {head}"],
        changed_coverage={"status": "success", "head_sha": head},
        metadata_digest=metadata,
        metadata_verdicts=[metadata_trailer(metadata)],
    )
    queue_exit = invoke()
    queue_output = capsys.readouterr().out
    assert blockers and "behavioral/execution source projection differs" in blockers[0], (
        f"PR blockers={blockers!r}, queue_exit={queue_exit}"
    )
    assert queue_exit == 1
    assert "behavioral/execution source projection differs" in queue_output


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("algo",), "ppo"),
        (("algorithm",), "ppo"),
        (("algorithm_metadata", "canonical_algorithm"), "ppo"),
        (("algorithm_metadata", "algorithm"), "guarded_ppo"),
        (("algorithm_metadata", "planner_contract", "planner_id"), "ppo"),
        (("algorithm_metadata", "planner_kinematics", "supports_native_commands"), False),
        (("algorithm_metadata", "planner_kinematics", "supports_adapter_commands"), False),
        (("algorithm_metadata", "planner_kinematics", "adapter_active"), False),
        (("algorithm_metadata", "planner_kinematics", "adapter_name"), "other_adapter"),
        (("algorithm_metadata", "planner_kinematics", "supports_native_commands"), 1),
        (("execution_mode",), "native"),
        (("algorithm_metadata", "execution_mode"), "adapter"),
    ],
)
def test_mixed_profile_requires_consistent_guarded_identity_and_capabilities(path, value):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _at(row, path[:-1])[path[-1]] = value
    assert producer._execution_evidence(row)["execution_mode"] == "unknown"


@pytest.mark.parametrize("owner", ["planner_runtime", "shield_stats"])
@pytest.mark.parametrize("value", ["failed", "degraded", "fallback", "unavailable", None])
def test_exact_calibration_status_does_not_hide_other_emitted_values(owner, value):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"][owner]["last_decision"]["prediction"]["calibration"]["status"] = value
    marker = producer._controller_runtime_marker(row, row["algorithm_metadata"], "guarded_ppo")
    assert marker is not None and marker[1] == "unclassified_degraded"
    assert producer._audit_status([producer._execution_evidence(row)]) != "pass"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("safety_shield_contract", "calibration_status"), "failed"),
        (("safety_shield_contract", "fallback_policy"), "ORCAPlannerAdapter"),
        (("safety_shield_contract", "schema_version"), "unknown"),
        (("safety_shield_contract", "shield_name"), "OtherAdapter"),
        (("planner_runtime", "last_decision", "schema_version"), "unknown"),
        (("shield_stats", "last_decision", "prediction", "source"), "unknown"),
    ],
)
def test_guarded_declarations_require_the_exact_source_context(path, value):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _at(row["algorithm_metadata"], path[:-1])[path[-1]] = value
    assert producer._execution_evidence(row)["execution_mode"] == "unknown"


@pytest.mark.parametrize("name", ["selected_action", "applied_environment_action"])
@pytest.mark.parametrize("fault", ["missing", "empty", "axis", "bool", "nan", "infinity"])
def test_guarded_mixed_requires_complete_finite_commands(name, fault):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    planner = row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"]
    if fault == "missing":
        del planner[name]
    elif fault == "empty":
        planner[name] = {}
    elif fault == "axis":
        del planner[name]["angular_velocity"]
    else:
        planner[name]["linear_velocity"] = {
            "bool": True,
            "nan": float("nan"),
            "infinity": float("inf"),
        }[fault]
    assert producer._execution_evidence(row)["controller_executed"] is False


@pytest.mark.parametrize("field", ["schema_version", "kind"])
def test_guarded_conversion_contract_is_required_at_every_step(field):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"]["action_conversion"][
        field
    ] = "unknown"
    assert producer._execution_evidence(row)["controller_executed"] is False


@pytest.mark.parametrize(
    "path",
    [
        ("non_runtime_provenance",),
        ("algorithm_metadata", "safety_shield_contract"),
        ("algorithm_metadata", "planner_runtime", "last_decision", "prediction"),
    ],
)
def test_guarded_context_never_exempts_unknown_fault_subtrees(path):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    if path == ("non_runtime_provenance",):
        row[path[0]] = {}
    _at(row, path)["unknown_status"] = "degraded"
    evidence = producer._execution_evidence(row)
    assert evidence["execution_mode"] == "mixed"
    assert evidence["degraded"] is True


@pytest.mark.parametrize("fault", ["fallback_used", "degraded", "status"])
def test_guarded_mode_preserves_actual_controller_failure_refusals(fault):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"]["planner_runtime"][fault] = "failed" if fault == "status" else True
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


@pytest.mark.parametrize("field", ["fallback_best_effort", "stop_best_effort"])
def test_existing_guarded_native_decisions_stay_present_and_byte_compared(field):
    row = _row()
    row["metrics"]["success"] = False
    row["algorithm_metadata"]["guard_stats"] = {field: 1}
    before = copy.deepcopy(row)
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    assert row == before
    changed = copy.deepcopy(row)
    changed["algorithm_metadata"]["guard_stats"][field] += 1
    assert eq.episode_bytes(row) != eq.episode_bytes(changed)


@pytest.mark.parametrize("algorithm", ["guarded_ppo", "ppo", "goal"])
def test_equivalence_transport_mixed_enum_is_guarded_only(algorithm):
    schema = json.loads(
        Path(producer.__file__).with_name("behaviour_receipt.schema.json").read_text()
    )
    execution = schema["$defs"]["equivalence_receipt"]["properties"]["rows"]["items"]["properties"][
        "head_execution"
    ]
    value = {
        "algorithm": algorithm,
        "execution_mode": "mixed",
        "controller_executed": True,
        "fallback": False,
        "degraded": False,
    }
    errors = list(jsonschema.Draft202012Validator(execution).iter_errors(value))
    assert bool(errors) is (algorithm != "guarded_ppo")


@pytest.mark.parametrize("fault", ["wrong_mode", "runtime_status", "missing_command"])
def test_live_queue_rederives_guarded_proof_from_committed_native_sources(
    queue_equivalence, tmp_path, monkeypatch, capsys, fault
):
    snapshot, _state, invoke, _base, _head, header, payload = queue_equivalence
    guarded = [row for row in payload["rows"] if row["arm"] == "guarded_ppo"]
    assert guarded and all(row["head_execution"]["execution_mode"] == "mixed" for row in guarded)
    assert invoke() == 0
    capsys.readouterr()
    if fault == "wrong_mode":
        guarded[0]["head_execution"]["execution_mode"] = "native"
    else:

        def mutate(root):
            path = next(root.glob("campaigns/**/guarded_ppo__differential_drive/episodes.jsonl"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if fault == "runtime_status":
                rows[0]["algorithm_metadata"]["planner_runtime"]["status"] = "failed"
            else:
                del rows[0]["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"][
                    "applied_environment_action"
                ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        # Fresh synthetic producer seals keep all unrelated source/custody guards valid.
        if fault == "missing_command":
            # The existing byte reader refuses structurally incomplete traces
            # before computing a digest, so no production hash can be refreshed.
            rewrite_source_archive(tmp_path, payload, "head", mutate, suffix="guarded-" + fault)
        else:
            for side in ("head", "baseline"):
                claims = [copy.deepcopy(row[f"{side}_execution"]) for row in payload["rows"]]
                rewrite_source_archive(
                    tmp_path, payload, side, mutate, synchronize=True, suffix="guarded-" + fault
                )
                for row, claim in zip(payload["rows"], claims, strict=True):
                    row[f"{side}_execution"] = claim
            # Digests are coherently equal and correct. Only the falsely healthy
            # execution claim disagrees with the authenticated raw proof.
            assert all(row["head_sha256"] == row["baseline_sha256"] for row in payload["rows"])
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    head = json.loads(body.split("\n")[1])["head_sha"]
    metadata = metadata_digest("merge queue test PR", body)
    snapshot.update(
        body=body,
        head_sha=head,
        gate_verdicts=[f"gate-verdict: accepted @ {head}"],
        changed_coverage={"status": "success", "head_sha": head},
        metadata_digest=metadata,
        metadata_verdicts=[metadata_trailer(metadata)],
    )
    assert invoke() == 1
    output = capsys.readouterr().out
    reason = (
        "missing requested/applied action trace"
        if fault == "missing_command"
        else "behavioral/execution source projection differs"
    )
    assert reason in output
