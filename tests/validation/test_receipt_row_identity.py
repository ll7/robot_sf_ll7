"""Native identity on the actual row must agree with directory and summary claims."""

import json
from types import SimpleNamespace

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer
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
from tests.validation.test_default_path_equivalence import (
    admitted as _admitted,
)
from tests.validation.test_default_path_equivalence import (
    exact_source_sweeps as _exact_source_sweeps,
)
from tests.validation.test_receipt_canonical_ids import identity_sweep as _identity_sweep
from tests.validation.test_receipt_outcomes import _synchronize_claims

admitted = _admitted
exact_source_sweeps = _exact_source_sweeps
queue_equivalence = _queue_equivalence
identity_sweep = _identity_sweep


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _plant_roster_override(root):
    """The native reconciler accepts the original matrix and refuses its resealed attack."""
    healthy, attacked = [], []
    arms, scenarios = set(), set()
    for path in sorted(root.glob("campaigns/**/episodes.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            context = {**row, "_source_file": str(path.relative_to(root))}
            healthy.append(context)
            arms.add(native._arm_key(context))
            scenarios.add(row["scenario_id"])
            row["_sweep_arm"] = "ppo"
            attacked.append({**context, "_sweep_arm": "ppo"})
        _write_rows(path, rows)
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
    _, original = native.reconcile_execution(cfg, matrix, healthy, root, result, step_trace=True)
    assert original["complete"] is True
    _, forged = native.reconcile_execution(cfg, matrix, attacked, root, result, step_trace=True)
    keys = {(native._arm_key(row), row["scenario_id"], row["seed"]) for row in attacked}
    assert len(attacked) == 2142 and len(keys) == 153
    assert forged["complete"] is False
    assert len(forged["missing_slots"]) == len(forged["duplicate_slots"]) == 1989


def test_producer_refuses_resealed_actual_row_roster_override(exact_source_sweeps, capsys):
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 2142
    for root in (baseline, head):
        _plant_roster_override(root)
        _refresh_equivalence_binding(root, validate_identity=False)
    before = {
        path: path.read_bytes()
        for root in (baseline, head)
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="identity"):
        producer.main(args)
    assert not capsys.readouterr().out
    assert {path: path.read_bytes() for path in before} == before


def _attack_archives(tmp_path, payload):
    for side in ("head", "baseline"):

        def edit(root):
            _plant_roster_override(root)
            _synchronize_claims(root, payload, side)

        rewrite_source_archive(tmp_path, payload, side, edit, suffix="native-row-roster")


def test_committed_admission_refuses_resealed_actual_row_roster_override(
    admitted, tmp_path, monkeypatch
):
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    _attack_archives(tmp_path, payload)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert blockers and "identity" in blockers[0], blockers


def test_queue_admission_refuses_resealed_actual_row_roster_override(
    queue_equivalence, tmp_path, monkeypatch, capsys
):
    snapshot, _, invoke, _, _, header, payload = queue_equivalence
    assert invoke() == 0
    capsys.readouterr()
    _attack_archives(tmp_path, payload)
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
    assert "identity" in output
    assert '"admission_status": "accepted"' not in output


def _agreeing_rows(root):
    """All explicit native inputs are present and agree before a one-source forgery."""
    raw = next(root.glob("campaigns/**/episodes.jsonl"))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in rows:
        row.update(
            _sweep_arm="goal",
            _source_file="runs/goal__differential_drive/episodes.jsonl",
            name="open",
            id="open",
        )
    _write_rows(raw, rows)
    return raw


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
@pytest.mark.parametrize(
    "source",
    [
        "directory",
        "row_arm",
        "row_source",
        "row_name",
        "row_id",
        "row_scenario",
        "row_seed",
        "summary_arm",
        "summary_scenario",
        "summary_seed",
    ],
)
def test_every_identity_source_must_agree(identity_sweep, reader, source):
    raw = _agreeing_rows(identity_sweep)

    def read():
        if reader == "metadata":
            return producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        return {
            "rows": producer._load_sweep_rows,
            "paths": producer._sweep_episode_paths,
            "statuses": producer._sweep_execution_statuses,
        }[reader](identity_sweep)

    read()
    summary = identity_sweep / "episodes_main.jsonl"
    summaries = [json.loads(line) for line in summary.read_text().splitlines()]
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if source == "directory":
        target = raw.parent.with_name("orca__differential_drive")
        raw.parent.rename(target)
        # Follow the observed file's transport location, retaining every declared ID.
        for row in summaries:
            row["source_file"] = "runs/orca__differential_drive/episodes.jsonl"
        _write_rows(summary, summaries)
    elif source.startswith("row_"):
        field, value = {
            "row_arm": ("_sweep_arm", "orca"),
            "row_source": ("_source_file", "runs/orca__differential_drive/episodes.jsonl"),
            "row_name": ("name", "other"),
            "row_id": ("id", "other"),
            "row_scenario": ("scenario_id", "other"),
            "row_seed": ("seed", 1004),
        }[source]
        rows[0][field] = value
        _write_rows(raw, rows)
    else:
        field, value = {
            "summary_arm": ("arm", "orca"),
            "summary_scenario": ("scenario", "other"),
            "summary_seed": ("seed", 1004),
        }[source]
        summaries[0][field] = value
        _write_rows(summary, summaries)
    paths = [summary, *identity_sweep.glob("campaigns/**/episodes.jsonl")]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="identity"):
        read()
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
def test_native_extractor_receives_every_actual_row_field(identity_sweep, monkeypatch, reader):
    """A newly introduced native priority input must not be lost to a synthetic projection."""
    raw = _agreeing_rows(identity_sweep)
    original = native._arm_key

    def with_new_priority(row):
        return row.get("_future_native_arm") or original(row)

    monkeypatch.setattr(native, "_arm_key", with_new_priority)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in rows:
        row["_future_native_arm"] = "orca"
    _write_rows(raw, rows)
    with pytest.raises(ValueError, match="identity"):
        if reader == "metadata":
            producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        else:
            {
                "rows": producer._load_sweep_rows,
                "paths": producer._sweep_episode_paths,
                "statuses": producer._sweep_execution_statuses,
            }[reader](identity_sweep)


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
@pytest.mark.parametrize(
    "source", ["runs/goal__differential_drive/other.jsonl", "runs/goal__holonomic/episodes.jsonl"]
)
def test_row_source_path_must_match_observed_file(identity_sweep, reader, source):
    """An agreeing arm name cannot hide a contradictory native source location."""
    raw = _agreeing_rows(identity_sweep)

    def read():
        if reader == "metadata":
            return producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        return {
            "rows": producer._load_sweep_rows,
            "paths": producer._sweep_episode_paths,
            "statuses": producer._sweep_execution_statuses,
        }[reader](identity_sweep)

    read()
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[0]["_source_file"] = source
    _write_rows(raw, rows)
    before = raw.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        read()
    assert raw.read_bytes() == before
