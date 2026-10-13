"""Reviewed native slot coverage is mandatory at every receipt admission boundary."""

import json
from types import SimpleNamespace

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer
from scripts.ci import receipt_native_slots as slots
from scripts.dev.merge_queue_gate import metadata_digest, metadata_trailer
from scripts.validation import run_empty_world_sweep as native
from tests.dev.test_merge_queue_gate import queue_equivalence as _queue_equivalence
from tests.validation.test_default_path_equivalence import (
    PATH,
    _refresh_equivalence_binding,
    committed_control_body,
    rewrite_source_archive,
    scheduler_endpoint,  # noqa: F401
)
from tests.validation.test_default_path_equivalence import admitted as _admitted
from tests.validation.test_default_path_equivalence import (
    exact_source_sweeps as _exact_source_sweeps,
)
from tests.validation.test_receipt_canonical_ids import identity_sweep as _identity_sweep
from tests.validation.test_receipt_outcomes import _synchronize_claims

admitted = _admitted
exact_source_sweeps = _exact_source_sweeps
queue_equivalence = _queue_equivalence
identity_sweep = _identity_sweep


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _set_row_kinematics(row):
    for field in ("kinematics", "_sweep_kinematics"):
        if field in row:
            row[field] = "holonomic"
    if "_source_file" in row:
        row["_source_file"] = row["_source_file"].replace("__differential_drive", "__holonomic")
    robot = (row.get("scenario_params") or {}).get("robot_config") or {}
    if "type" in robot:
        robot["type"] = "holonomic"


def _holonomic(root, ppo_only):
    original, attacked = [], []
    arms, scenarios = set(), set()
    for path in list(root.glob("campaigns/**/episodes.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            context = {**row, "_source_file": str(path.relative_to(root))}
            original.append(context)
            arms.add(native._arm_key(context))
            scenarios.add(row["scenario_id"])
        target = path.parent
        if not ppo_only or target.name.startswith("ppo__"):
            for row in rows:
                _set_row_kinematics(row)
            _write(path, rows)
            target = target.with_name(target.name.replace("__differential_drive", "__holonomic"))
            path.parent.rename(target)
        attacked.extend(
            {**row, "_source_file": str((target / path.name).relative_to(root))} for row in rows
        )
    for path in root.glob("episodes_*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if not ppo_only or row["arm"] == "ppo":
                row["kinematics"] = "holonomic"
                if "_sweep_kinematics" in row:
                    row["_sweep_kinematics"] = "holonomic"
                row["source_file"] = row["source_file"].replace(
                    "__differential_drive", "__holonomic"
                )
        _write(path, rows)
    cfg = SimpleNamespace(
        planners=[SimpleNamespace(key=arm) for arm in sorted(arms)],
        kinematics_matrix=["differential_drive"],
    )
    matrix = [{"name": name, "seeds": [1001, 1002, 1003]} for name in sorted(scenarios)]
    result = {
        "campaign_execution_status": "completed",
        "exit_code": 0,
        "unexpected_failed_runs": 0,
        "total_episodes": 2142,
    }
    assert (
        native.reconcile_execution(cfg, matrix, original, root, result, step_trace=True)[1][
            "complete"
        ]
        is True
    )
    audit = native.reconcile_execution(cfg, matrix, attacked, root, result, step_trace=True)[1]
    assert audit["complete"] is False
    assert (
        len(audit["missing_slots"]) == len(audit["unexpected_slots"]) == (153 if ppo_only else 2142)
    )


def _refresh_metadata_seeds(root):
    for metadata in root.glob("execution_*.json"):
        if metadata.name in {
            "execution_request.json",
            "execution_scheduler.json",
            "execution_binding.json",
        }:
            continue
        suite = metadata.stem.removeprefix("execution_")
        summary = root / f"episodes_{suite}.jsonl"
        if summary.is_file():
            values = json.loads(metadata.read_text())
            values["seeds"] = sorted(
                {json.loads(line)["seed"] for line in summary.read_text().splitlines()}
            )
            metadata.write_text(json.dumps(values))


@pytest.mark.parametrize("ppo_only", [False, True], ids=["all-arms", "ppo-only"])
def test_producer_refuses_resealed_holonomic_slots(exact_source_sweeps, capsys, ppo_only):
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 2142
    for root in (baseline, head):
        _holonomic(root, ppo_only)
        _refresh_metadata_seeds(root)
        _refresh_equivalence_binding(root, validate_identity=False)
    with pytest.raises(ValueError, match="identity|scope|coverage|matrix"):
        producer.main(args)
    assert not capsys.readouterr().out


def _attack(tmp_path, payload, ppo_only):
    for side in ("head", "baseline"):

        def edit(root):
            _holonomic(root, ppo_only)
            _synchronize_claims(root, payload, side)

        rewrite_source_archive(tmp_path, payload, side, edit, suffix="holonomic-slots")
    for row in payload["rows"]:
        if not ppo_only or row["arm"] == "ppo":
            for field in ("head_path", "baseline_path"):
                row[field] = row[field].replace("__differential_drive", "__holonomic")


@pytest.mark.parametrize("ppo_only", [False, True], ids=["all-arms", "ppo-only"])
def test_committed_admission_refuses_resealed_holonomic_slots(
    admitted, tmp_path, monkeypatch, ppo_only
):
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    _attack(tmp_path, payload, ppo_only)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert blockers and any(
        word in blockers[0] for word in ("identity", "scope", "coverage", "matrix")
    ), blockers


@pytest.mark.parametrize("ppo_only", [False, True], ids=["all-arms", "ppo-only"])
def test_queue_admission_refuses_resealed_holonomic_slots(
    queue_equivalence, tmp_path, monkeypatch, capsys, ppo_only
):
    snapshot, _, invoke, _, _, header, payload = queue_equivalence
    assert invoke() == 0
    capsys.readouterr()
    _attack(tmp_path, payload, ppo_only)
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
    assert any(word in output for word in ("identity", "scope", "coverage", "matrix"))
    assert '"admission_status": "accepted"' not in output


@pytest.mark.parametrize("field", slots.EpisodeSlot._fields)
def test_every_runtime_slot_field_refuses_coherent_producer_forgery(
    exact_source_sweeps, capsys, field
):
    """New runtime slot fields automatically join this source-forgery probe."""
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    for root in (baseline, head):
        replacements = {}
        for path in list(root.glob("campaigns/**/episodes.jsonl")):
            if not path.parent.name.startswith("ppo__"):
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                slot = slots.episode_slot({**row, "_source_file": path.as_posix()})
                value = getattr(slot, field)
                replacement = value + 3 if type(value) is int else value + "_forged"
                forged = slot._replace(**{field: replacement})
                row.update(slots.episode_slot_row(forged))
                replacements[slot] = forged
            target = path.parent.with_name(f"{forged.arm}__{forged.kinematics}")
            _write(path, rows)
            if target != path.parent:
                path.parent.rename(target)
        for path in root.glob("episodes_*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if row["arm"] != "ppo":
                    continue
                original = slots.episode_slot(
                    {
                        **row,
                        "_sweep_arm": row["arm"],
                        "scenario_id": row["scenario"],
                        "_source_file": row["source_file"],
                    }
                )
                forged = replacements[original]
                row.update(slots.episode_slot_row(forged))
                row.update(
                    arm=forged.arm,
                    scenario=forged.scenario,
                    seed=forged.seed,
                    source_file=f"runs/{forged.arm}__{forged.kinematics}/episodes.jsonl",
                )
            _write(path, rows)
        _refresh_metadata_seeds(root)
        _refresh_equivalence_binding(root, validate_identity=False)
    with pytest.raises(ValueError, match="identity|scope|coverage|matrix|development seed slice"):
        producer.main(args)
    assert not capsys.readouterr().out


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
@pytest.mark.parametrize("source", ["directory", "row", "summary"])
def test_kinematics_sources_must_agree(identity_sweep, reader, source):
    """A masked lower-priority kinematics claim must also refuse locally."""
    raw = next(identity_sweep.glob("campaigns/**/episodes.jsonl"))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in rows:
        row["_sweep_kinematics"] = "differential_drive"
    _write(raw, rows)
    summary = identity_sweep / "episodes_main.jsonl"
    summaries = [json.loads(line) for line in summary.read_text().splitlines()]
    for row in summaries:
        row["kinematics"] = "differential_drive"
    _write(summary, summaries)

    def read():
        if reader == "metadata":
            return producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        return {
            "rows": producer._load_sweep_rows,
            "paths": producer._sweep_episode_paths,
            "statuses": producer._sweep_execution_statuses,
        }[reader](identity_sweep)

    read()
    if source == "directory":
        raw.parent.rename(raw.parent.with_name("goal__holonomic"))
        for row in summaries:
            row["source_file"] = row["source_file"].replace("__differential_drive", "__holonomic")
        _write(summary, summaries)
    elif source == "row":
        rows[0]["_sweep_kinematics"] = "holonomic"
        _write(raw, rows)
    else:
        summaries[0]["kinematics"] = "holonomic"
        _write(summary, summaries)
    with pytest.raises(ValueError, match="identity"):
        read()


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
def test_native_future_kinematics_priority_cannot_hide_directory(
    identity_sweep, monkeypatch, reader
):
    original = native._kinematics
    monkeypatch.setattr(
        native, "_kinematics", lambda row: row.get("_future_kinematics") or original(row)
    )
    raw = next(identity_sweep.glob("campaigns/**/episodes.jsonl"))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in rows:
        row["_future_kinematics"] = "holonomic"
    _write(raw, rows)
    with pytest.raises(ValueError, match="identity"):
        if reader == "metadata":
            producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        else:
            {
                "rows": producer._load_sweep_rows,
                "paths": producer._sweep_episode_paths,
                "statuses": producer._sweep_execution_statuses,
            }[reader](identity_sweep)
