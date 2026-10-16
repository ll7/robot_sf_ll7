"""Redundant summary robot declarations must agree with verified raw slots."""

import json

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer
from tests.validation.test_default_path_equivalence import (
    PATH,
    committed_control_body,
    rewrite_source_archive,
    scheduler_endpoint,  # noqa: F401
)
from tests.validation.test_default_path_equivalence import admitted as _admitted
from tests.validation.test_receipt_canonical_ids import identity_sweep as _identity_sweep
from tests.validation.test_receipt_outcomes import _synchronize_claims

admitted = _admitted
identity_sweep = _identity_sweep


def _declare_summary_robot_type(root, robot_type):
    for path in root.glob("episodes_*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.setdefault("scenario_params", {}).setdefault("robot_config", {})["type"] = (
                robot_type
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
def test_producer_readers_refuse_contradictory_summary_robot_type(identity_sweep, reader):
    def read():
        if reader == "metadata":
            return producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        return {
            "rows": producer._load_sweep_rows,
            "paths": producer._sweep_episode_paths,
            "statuses": producer._sweep_execution_statuses,
        }[reader](identity_sweep)

    _declare_summary_robot_type(identity_sweep, "differential_drive")
    read()
    raw = {path: path.read_bytes() for path in identity_sweep.glob("campaigns/**/episodes.jsonl")}
    _declare_summary_robot_type(identity_sweep, "holonomic")
    before = {path: path.read_bytes() for path in identity_sweep.glob("episodes_*.jsonl")}
    with pytest.raises(ValueError, match="identity.*robot kinematics differs"):
        read()
    assert all(path.read_bytes() == content for path, content in {**raw, **before}.items())


def test_committed_admission_refuses_resealed_summary_robot_type(admitted, tmp_path, monkeypatch):
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []

    def alter(root):
        _declare_summary_robot_type(root, "holonomic")
        _synchronize_claims(root, payload, "head")

    rewrite_source_archive(tmp_path, payload, "head", alter, suffix="summary-robot-type")
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert blockers and "identity" in blockers[0] and "robot kinematics differs" in blockers[0], (
        blockers
    )
