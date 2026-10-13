"""Equal partial traces must refuse collection, committed admission and queue admission."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from scripts.dev.merge_queue_gate import metadata_digest, metadata_trailer
from tests.dev.test_merge_queue_gate import queue_equivalence as _queue_equivalence
from tests.validation.test_default_path_equivalence import (
    PATH,
    _episode,
    _refresh_equivalence_binding,
    committed_control_body,
    rewrite_source_archive,
    scheduler_endpoint,  # noqa: F401
)
from tests.validation.test_default_path_equivalence import (
    admitted as _admitted,
)
from tests.validation.test_default_path_equivalence import (
    exact_source_sweeps as _exact_source_sweeps,
)
from tests.validation.test_receipt_outcomes import _synchronize_claims

admitted = _admitted
exact_source_sweeps = _exact_source_sweeps
queue_equivalence = _queue_equivalence

# Each observation is required in the producer's simulation-step-trace schema.
OBSERVATIONS = (
    ("rl",),
    ("rl", "reward"),
    ("rl", "terminated"),
    ("rl", "truncated"),
    ("collision",),
    ("collision", "pedestrian"),
    ("collision", "obstacle"),
    ("collision", "robot"),
)
ACTIONS = ("selected_action", "applied_environment_action")
INVALID_ACTIONS = (
    None,
    {},
    {"linear_velocity": 1.0},
    {"angular_velocity": 0.0},
    {"linear_velocity": True, "angular_velocity": 0.0},
    {"linear_velocity": 1.0, "angular_velocity": False},
    {"linear_velocity": "1.0", "angular_velocity": 0.0},
    {"linear_velocity": [1.0], "angular_velocity": 0.0},
    {"linear_velocity": float("nan"), "angular_velocity": 0.0},
    {"linear_velocity": 1.0, "angular_velocity": float("inf")},
    {"command_kind": "holonomic_vxy_world", "vx": 1.0, "vy": 0.0},
    [1.0, 0.0],
    {"linear_velocity": 1.0, "angular_velocity": 0.0, "vx": 1.0},
)


def _steps(row):
    return row["algorithm_metadata"]["simulation_step_trace"]["steps"]


@pytest.mark.parametrize("side", ["head", "baseline", "both"])
@pytest.mark.parametrize("path", OBSERVATIONS, ids=".".join)
@pytest.mark.parametrize("fault", ["missing", "null"])
def test_comparison_requires_each_trace_observation(side, path, fault):
    """A valid false/zero control does not authorize equally unavailable observations."""
    scope = {"arms": ["goal"], "maps": ["open"]}
    rows = {("goal", "open", seed): _episode() for seed in eq.SEEDS}
    assert len(eq.compare_rows(rows, rows, scope)) == 3
    head, baseline = copy.deepcopy(rows), copy.deepcopy(rows)
    for selected in [head, baseline] if side == "both" else [head if side == "head" else baseline]:
        for row in selected.values():
            parent = _steps(row)[0]
            for part in path[:-1]:
                parent = parent[part]
            if fault == "missing":
                parent.pop(path[-1])
            else:
                parent[path[-1]] = None
    with pytest.raises(ValueError, match="incomplete action/pose/event/reward trace"):
        eq.compare_rows(head, baseline, scope)


@pytest.mark.parametrize("algorithm", ["goal", "ppo", "orca", "predictive_mppi"])
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("invalid", INVALID_ACTIONS)
def test_projection_requires_typed_two_component_actions(algorithm, action, invalid):
    """The same malformed command on both sources cannot receive a behavioral digest."""
    row = _episode()
    row["algo"] = algorithm
    assert eq.episode_bytes(row)
    _steps(row)[0]["planner"][action] = invalid
    with pytest.raises(ValueError, match="invalid requested/applied action schema"):
        eq.episode_bytes(row)


# Exercise the real sealed-source entrypoints, including PPO ordinary-arm commands.
SOURCE_FAULTS = (
    "missing_reward",
    "null_reward",
    "null_rl",
    "null_collision",
    "missing_event",
    "empty_action",
    "null_action",
    "boolean_action",
    "short_action",
)


def _mutate_trace(root, fault):
    for path in root.glob("campaigns/**/episodes.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for step in _steps(row):
                if fault == "missing_reward":
                    step["rl"].pop("reward")
                elif fault == "null_reward":
                    step["rl"]["reward"] = None
                elif fault == "null_rl":
                    step["rl"] = None
                elif fault == "null_collision":
                    step["collision"] = None
                elif fault == "missing_event":
                    step["collision"].pop("pedestrian")
                elif row["algo"] not in {"guarded_ppo", "hybrid_rule_local_planner"}:
                    step["planner"]["applied_environment_action"] = {
                        "empty_action": {},
                        "null_action": None,
                        "boolean_action": {"linear_velocity": True, "angular_velocity": 0.0},
                        "short_action": {"linear_velocity": 1.0},
                    }[fault]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _reason(fault):
    return (
        "invalid requested/applied action schema"
        if fault.endswith("action")
        else "incomplete action/pose/event/reward trace"
    )


def _archive_invalid_trace(repo, payload, side, fault):
    def mutate(root):
        _mutate_trace(root, fault)
        # Coherent digests are constructed independently of production validation.
        _synchronize_claims(root, payload, side)

    rewrite_source_archive(repo, payload, side, mutate, suffix=fault)


@pytest.mark.parametrize("fault", SOURCE_FAULTS)
@pytest.mark.parametrize("side", ["head", "baseline", "both"])
def test_private_comparison_refuses_incomplete_trace(exact_source_sweeps, capsys, fault, side):
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    for root in [head, baseline] if side == "both" else [head if side == "head" else baseline]:
        _mutate_trace(root, fault)
        _refresh_equivalence_binding(root)
    with pytest.raises(ValueError, match=_reason(fault)):
        producer.main(args)


@pytest.mark.parametrize("fault", SOURCE_FAULTS)
@pytest.mark.parametrize("side", ["head", "baseline", "both"])
def test_committed_admission_refuses_incomplete_trace(admitted, tmp_path, monkeypatch, fault, side):
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    for selected in ("head", "baseline") if side == "both" else (side,):
        _archive_invalid_trace(tmp_path, payload, selected, fault)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert blockers and _reason(fault) in blockers[0]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_reward",
        "null_rl",
        "null_collision",
        "empty_action",
        "null_action",
        "boolean_action",
    ],
)
def test_live_queue_refuses_equally_incomplete_trace(
    queue_equivalence, tmp_path, monkeypatch, capsys, fault
):
    snapshot, _, invoke, _, _, header, payload = queue_equivalence
    assert invoke() == 0
    capsys.readouterr()
    for side in ("head", "baseline"):
        _archive_invalid_trace(tmp_path, payload, side, fault)
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
    assert _reason(fault) in capsys.readouterr().out


@pytest.mark.parametrize("duplicate", ["arms", "maps", "both"])
def test_producer_refuses_duplicate_partial_scope(exact_source_sweeps, capsys, duplicate):
    """Nominal list lengths cannot turn a reduced matrix into full equality."""
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    scope_path = Path(args[args.index("--scope-path") + 1])
    scope = json.loads(scope_path.read_text())
    if duplicate in {"arms", "both"}:
        scope["arms"] = ["arm-0"] * 14
    if duplicate in {"maps", "both"}:
        scope["maps"] = ["map-0"] * 51
    scope_path.write_text(json.dumps(scope))
    for root in (baseline, head):
        summary_path = root / "episodes_main.jsonl"
        summaries = [json.loads(line) for line in summary_path.read_text().splitlines()]
        summaries = [
            row
            for row in summaries
            if row["arm"] in scope["arms"] and row["scenario"] in scope["maps"]
        ]
        summary_path.write_text("".join(json.dumps(row) + "\n" for row in summaries))
        for raw_path in root.glob("campaigns/**/episodes.jsonl"):
            arm = raw_path.parent.name.split("__")[0]
            if arm not in scope["arms"]:
                shutil.rmtree(raw_path.parent)
                continue
            raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            raw_rows = [row for row in raw_rows if row["scenario_id"] in scope["maps"]]
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
        metadata_path = root / "execution_main.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            expected_slots=len(summaries),
            written_rows=len(summaries),
            total_episodes=len(summaries),
        )
        metadata_path.write_text(json.dumps(metadata))
        _refresh_equivalence_binding(root)
    with pytest.raises(ValueError, match="equivalence refused: duplicate scope"):
        producer.main(args)


def test_producer_requires_expected_episode_count_before_equality(
    exact_source_sweeps, monkeypatch, capsys
):
    """The public equality report independently enforces its 2142-episode contract."""
    args, _, _ = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    comparison_rows = eq.comparison_rows
    monkeypatch.setattr(eq, "comparison_rows", lambda *a, **kw: comparison_rows(*a, **kw)[:3])
    with pytest.raises(ValueError, match="equivalence refused: expected 2142 paired episodes"):
        producer.main(args)
    assert not capsys.readouterr().out
