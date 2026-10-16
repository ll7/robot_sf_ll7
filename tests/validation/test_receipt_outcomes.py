"""Complete outcome evidence is mandatory even after coherent producer resealing."""

import copy
import hashlib
import json

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from scripts.dev.merge_queue_gate import metadata_digest, metadata_trailer
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

# Register imported fixtures without shadowing imported names in test arguments.
admitted = _admitted
exact_source_sweeps = _exact_source_sweeps
queue_equivalence = _queue_equivalence

FIELDS = ("success", "collisions", "steps")
RAW_FIELDS = (*FIELDS, "total_collision_count")
SIDES = ("head", "baseline", "both")


def _edit_outcomes(root, missing=None):
    """Use false/zero controls so the old defaults equal genuine recorded outcomes."""
    paths = [*root.glob("campaigns/**/episodes.jsonl"), *root.glob("episodes_*.jsonl")]
    for path in paths:
        raw = path.relative_to(root).parts[0] == "campaigns"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            outcomes = row["metrics"] if raw else row
            outcomes["success"] = False
            if missing is not None:
                (row if missing == "steps" else outcomes).pop(missing, None)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _sides(side):
    return ("head", "baseline") if side == "both" else (side,)


def _synchronize_claims(root, payload, side):
    """Reseal independently, without asking production admission to validate the mutation."""
    job = root.name
    # Forge claims independently of admission; retain reviewed native-slot claims
    # when relabeling sources to an unreviewed kinematics type.
    compact_rows = {(row["arm"], row["map"], row["seed"]): row for row in payload["rows"]}
    for path in root.glob("campaigns/**/episodes.jsonl"):
        arm = path.parent.name.split("__")[0]
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            projected = copy.deepcopy(raw)
            for excluded in eq.EXCLUDED_LEAVES:
                parent = projected
                for part in excluded[:-1]:
                    parent = parent.get(part, {}) if isinstance(parent, dict) else {}
                if isinstance(parent, dict):
                    parent.pop(excluded[-1], None)
            digest = hashlib.sha256(
                json.dumps(projected, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            key = (arm, raw["scenario_id"], raw["seed"])
            compact = compact_rows[key]
            compact[f"{side}_sha256"] = digest
            assert compact[("baseline_" if side == "baseline" else "") + "scheduler_job_id"] == job


def _archive_outcomes(repo, payload, side, missing=None, suffix="outcome"):
    def mutate(root):
        _edit_outcomes(root, missing)
        _synchronize_claims(root, payload, side)

    rewrite_source_archive(repo, payload, side, mutate, suffix=suffix)


def _valid_false_receipt(repo, payload, header, monkeypatch, base):
    for side in ("head", "baseline"):
        _archive_outcomes(repo, payload, side, suffix="false-control")
    body = committed_control_body(repo, payload, header, monkeypatch)
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    return body


@pytest.mark.parametrize("field", RAW_FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_collection_requires_each_raw_outcome(exact_source_sweeps, field, side):
    _, baseline, head = exact_source_sweeps
    for root in (head, baseline):
        _edit_outcomes(root)
        _refresh_equivalence_binding(root)
        assert producer._load_sweep_rows(root, preserve_raw=True)
    for selected in _sides(side):
        root = head if selected == "head" else baseline
        _edit_outcomes(root, field)
        _refresh_equivalence_binding(root)
        with pytest.raises(ValueError, match=rf"outcome schema refused: missing raw .*{field}"):
            producer._load_sweep_rows(root, preserve_raw=True)


@pytest.mark.parametrize("field", RAW_FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_private_comparison_requires_each_raw_outcome(exact_source_sweeps, capsys, field, side):
    args, baseline, head = exact_source_sweeps
    for root in (head, baseline):
        _edit_outcomes(root)
        _refresh_equivalence_binding(root)
    assert producer.main(args) == 0
    capsys.readouterr()
    for selected in _sides(side):
        root = head if selected == "head" else baseline
        _edit_outcomes(root, field)
        _refresh_equivalence_binding(root)
    with pytest.raises(ValueError, match=rf"outcome schema refused: missing raw .*{field}"):
        producer.main(args)


@pytest.mark.parametrize("field", RAW_FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_committed_pr_requires_each_raw_outcome(admitted, tmp_path, monkeypatch, field, side):
    payload, header, _, _, base, _ = admitted
    _valid_false_receipt(tmp_path, payload, header, monkeypatch, base)
    for selected in _sides(side):
        _archive_outcomes(tmp_path, payload, selected, field, suffix="missing-" + field)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert (
        blockers
        and f"outcome schema refused: missing raw {'metrics.' if field != 'steps' else ''}{field}"
        in blockers[0]
    )


@pytest.mark.parametrize("field", RAW_FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_live_queue_requires_each_raw_outcome(
    queue_equivalence, tmp_path, monkeypatch, capsys, field, side
):
    snapshot, _, invoke, base, _, header, payload = queue_equivalence

    def select(body):
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

    select(_valid_false_receipt(tmp_path, payload, header, monkeypatch, base))
    assert invoke() == 0
    capsys.readouterr()
    for selected in _sides(side):
        _archive_outcomes(tmp_path, payload, selected, field, suffix="missing-" + field)
    select(committed_control_body(tmp_path, payload, header, monkeypatch))
    assert invoke() == 1
    assert (
        f"outcome schema refused: missing raw {'metrics.' if field != 'steps' else ''}{field}"
        in capsys.readouterr().out
    )


def _archive_missing_summary(repo, payload, side, field):
    def mutate(root):
        path = root / "episodes_main.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop(field)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    rewrite_source_archive(repo, payload, side, mutate, suffix="missing-summary-" + field)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_committed_pr_requires_each_summary_outcome(admitted, tmp_path, monkeypatch, field, side):
    payload, header, _, _, base, _ = admitted
    _valid_false_receipt(tmp_path, payload, header, monkeypatch, base)
    for selected in _sides(side):
        _archive_missing_summary(tmp_path, payload, selected, field)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert blockers and f"outcome schema refused: missing summary {field}" in blockers[0]
