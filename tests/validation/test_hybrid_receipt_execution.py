"""Only the source-bound hybrid adapter may use its registered adapter default."""

import copy
import json

import pytest

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from scripts.dev.merge_queue_gate import metadata_digest, metadata_trailer
from tests.dev.test_merge_queue_gate import queue_equivalence as _queue_equivalence
from tests.validation import test_default_path_equivalence as committed_fixture
from tests.validation.receipt_hybrid_fixture import hybrid_episode
from tests.validation.test_default_path_equivalence import _episode

ALGORITHM = "hybrid_rule_local_planner"
queue_equivalence = _queue_equivalence
ARMS = (
    "hybrid_rule_v4_fast_progress_static_escape",
    "hybrid_rule_v4_fast_progress_static_escape_continuous",
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
)
IDENTITIES = (
    ("algo",),
    ("algorithm_metadata", "algorithm"),
    ("algorithm_metadata", "canonical_algorithm"),
    ("algorithm_metadata", "planner_contract", "planner_id"),
    ("algorithm_metadata", "policy_semantics"),
    ("event_ledger", "planner"),
    ("scenario_params", "algo"),
    ("algorithm_metadata", "paired_effect_metric_producer", "pairing", "key", "planner"),
)


def _row():
    return hybrid_episode(_episode())


def _at(row, path):
    for key in path:
        row = row[key]
    return row


@pytest.fixture(autouse=True)
def _hybrid_scope(monkeypatch):
    original = committed_fixture._build_admitted
    scope = {
        "arms": ["ppo", "guarded_ppo", *ARMS, *[f"arm-{i}" for i in range(8)]],
        "maps": [f"map-{i}" for i in range(51)],
    }
    scope["arm_algorithms"] = {
        arm: ALGORITHM if arm in ARMS else arm if arm in {"ppo", "guarded_ppo"} else "goal"
        for arm in scope["arms"]
    }

    def build(root, patcher):
        return original(root, patcher, scope_override=scope)

    monkeypatch.setattr(committed_fixture, "_build_admitted", build)


def test_native_hybrid_factory_satisfies_registered_adapter_without_mutation():
    row = _row()
    before = copy.deepcopy(row)
    evidence = producer._execution_evidence(row)
    assert evidence == {
        "algorithm": ALGORITHM,
        "execution_mode": "adapter",
        "controller_executed": True,
        "fallback": False,
        "degraded": False,
    }
    producer.require_valid_execution(
        evidence, ALGORITHM, enrich_algorithm_metadata(algo=ALGORITHM)["planner_kinematics"]
    )
    assert row == before


def test_equivalence_compares_source_hybrid_adapter_and_all_behavioral_bytes():
    rows = {(ALGORITHM, "open", seed): _row() for seed in eq.SEEDS}
    scope = {"arms": [ALGORITHM], "maps": ["open"]}
    assert len(eq.compare_rows(rows, copy.deepcopy(rows), scope)) == 3
    changed = copy.deepcopy(rows)
    changed[next(iter(changed))]["metrics"]["success"] = False
    with pytest.raises(ValueError, match="changed behavioral byte"):
        eq.compare_rows(changed, rows, scope)


@pytest.mark.parametrize("path", IDENTITIES)
@pytest.mark.parametrize("fault", ["missing", "contradictory"])
def test_hybrid_adapter_requires_every_native_identity(path, fault):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    owner = _at(row, path[:-1])
    if fault == "missing":
        del owner[path[-1]]
    else:
        owner[path[-1]] = "orca"
    evidence = producer._execution_evidence(row)
    if evidence["algorithm"] == ALGORITHM:
        assert evidence["execution_mode"] == "unknown"
    with pytest.raises(ValueError, match="raw execution invalid"):
        producer.require_valid_execution(
            evidence, ALGORITHM, enrich_algorithm_metadata(algo=ALGORITHM)["planner_kinematics"]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("supports_native_commands", False),
        ("supports_adapter_commands", False),
        ("supports_native_commands", 1),
        ("adapter_active", False),
        ("adapter_name", "OtherAdapter"),
        ("execution_mode", "native"),
    ],
)
def test_hybrid_adapter_requires_exact_capabilities_and_default(field, value):
    row = _row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"]["planner_kinematics"][field] = value
    assert producer._execution_evidence(row)["execution_mode"] == "unknown"


@pytest.mark.parametrize("identity", [None, "HybridRuleLocalPlannerAdapter", "ForeignPlanner", 1])
def test_optional_hybrid_runtime_type_is_not_manufactured(identity):
    row = _row()
    if identity is not None:
        row["algorithm_metadata"]["planner_runtime"] = {"planner_type": identity}
    evidence = producer._execution_evidence(row)
    assert (producer._audit_status([evidence]) == "pass") is (
        identity in (None, "HybridRuleLocalPlannerAdapter")
    )
    if identity is None:
        assert "planner_runtime" not in row["algorithm_metadata"]


@pytest.mark.parametrize("action", ["selected_action", "applied_environment_action"])
@pytest.mark.parametrize("fault", ["missing", "axis", "nonfinite", "boolean"])
def test_hybrid_adapter_requires_complete_finite_commands(action, fault):
    row = _row()
    planner = row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"]
    if fault == "missing":
        del planner[action]
    elif fault == "axis":
        del planner[action]["angular_velocity"]
    else:
        planner[action]["linear_velocity"] = True if fault == "boolean" else float("inf")
    assert producer._execution_evidence(row)["controller_executed"] is False


@pytest.mark.parametrize("fault", ["fallback_used", "degraded", "status", "unknown_status"])
def test_hybrid_adapter_does_not_waive_runtime_faults(fault):
    row = _row()
    row["algorithm_metadata"]["planner_runtime"] = {fault: "failed" if "status" in fault else True}
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


def test_registered_hybrid_default_does_not_relax_other_dual_capability_algorithms():
    profile = enrich_algorithm_metadata(algo="ppo")["planner_kinematics"]
    assert producer.required_execution_mode("ppo", profile) == "native"


@pytest.mark.parametrize("path", [IDENTITIES[1], IDENTITIES[5]])
@pytest.mark.parametrize("fault", ["missing", "contradictory"])
def test_committed_pr_and_queue_refuse_coherently_resealed_hybrid_identity(
    queue_equivalence, tmp_path, monkeypatch, capsys, path, fault
):
    snapshot, _state, invoke, base, _head, header, payload = queue_equivalence
    for arm in ARMS:
        assert all(
            row["head_execution"]["execution_mode"] == "adapter"
            for row in payload["rows"]
            if row["arm"] == arm
        )
    assert invoke() == 0
    capsys.readouterr()

    def mutate(root):
        raw = next(root.glob(f"campaigns/**/{ARMS[0]}__differential_drive/episodes.jsonl"))
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        parent = _at(rows[0], path[:-1])
        if fault == "missing":
            del parent[path[-1]]
        else:
            parent[path[-1]] = "orca"
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))

    for side in ("head", "baseline"):
        claims = [copy.deepcopy(row[f"{side}_execution"]) for row in payload["rows"]]
        committed_fixture.rewrite_source_archive(
            tmp_path, payload, side, mutate, synchronize=True, suffix="hybrid-" + fault
        )
        for row, claim in zip(payload["rows"], claims, strict=True):
            row[f"{side}_execution"] = claim
    assert all(row["head_sha256"] == row["baseline_sha256"] for row in payload["rows"])
    body = committed_fixture.committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = producer.behaviour_receipt.check_receipt(
        body, [committed_fixture.PATH], "test/repo", base
    )
    head = json.loads(body.split("\n")[1])["head_sha"]
    digest = metadata_digest("merge queue test PR", body)
    snapshot.update(
        body=body,
        head_sha=head,
        gate_verdicts=[f"gate-verdict: accepted @ {head}"],
        changed_coverage={"status": "success", "head_sha": head},
        metadata_digest=digest,
        metadata_verdicts=[metadata_trailer(digest)],
    )
    queue_exit = invoke()
    output = capsys.readouterr().out
    assert blockers and "behavioral/execution source projection differs" in blockers[0]
    assert queue_exit == 1 and "behavioral/execution source projection differs" in output
