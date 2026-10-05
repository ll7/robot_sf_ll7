"""Behaviour-changing PRs must reach the real contract gate, without simulation."""

import hashlib
import json

import pytest

from scripts.ci import pr_contract_check as checker

HEAD = "a" * 40


def _checks(monkeypatch, body, paths):
    for name in (
        "check_closes_discipline",
        "check_github_closing_parity",
        "check_closure_declaration",
        "check_state_refresh_only",
        "check_evidence_tree_hygiene",
        "check_evidence_writer_usage",
        "check_successor_discipline",
        "check_placeholder_docstrings",
        "check_line_budget_discipline",
    ):
        monkeypatch.setattr(checker, name, lambda *a, **k: [])
    monkeypatch.setattr(checker, "check_worker_lane_provenance", lambda *a, **k: ("tooling", False))
    return checker.run_all_checks("change", body, paths, "ll7/robot_sf_ll7", "origin/main", None)[0]


@pytest.mark.parametrize(
    "path",
    [
        "robot_sf/planner/guarded_ppo.py",
        "robot_sf/benchmark/map_runner/map_runner_episode.py",
        "maps/renamed.svg",
    ],
)
def test_behaviour_changes_without_receipt_are_blocked(monkeypatch, path):
    blockers = _checks(monkeypatch, "", [path])
    assert any("behaviour receipt" in blocker.lower() for blocker in blockers), blockers


def _receipt():
    rows = [
        {
            "arm": arm,
            "map": map_id,
            "seed": seed,
            "success": True,
            "collisions": 0,
            "fallback": False,
            "execution_mode": "native",
            "baseline_success": True,
            "baseline_collisions": 0,
        }
        for arm in ["goal", "orca"]
        for map_id in ["open", "door"]
        for seed in range(1001, 1031)
    ]
    return {
        "schema_version": "behaviour-change-receipt.v1",
        "scope_sha256": hashlib.sha256(
            json.dumps(
                {
                    "arms": ["goal", "orca"],
                    "maps": ["open", "door"],
                    "vehicle_id": "t60",
                    "exceptions": [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "head_sha": HEAD,
        "scheduler": {"kind": "slurm", "job_id": "12345", "source_sha": HEAD},
        "artifact": {"uri": "https://example.org/sweep.json", "sha256": "b" * 64},
        "baseline": {
            "release": "0.0.7",
            "source_sha": "c" * 40,
            "artifact_uri": "https://example.org/baseline.json",
            "artifact_sha256": "d" * 64,
            "comparison": "archived_rows",
            "body_id": "previous",
            "config_sha256": "e" * 64,
            "differences": ["vehicle body differs; diagnostic comparison, no causal claim"],
        },
        "vehicle": {"id": "t60", "body_sha256": "f" * 64},
        "rows": rows,
        "classifications": [],
        "exceptions": [],
        "totals": {"episodes": len(rows), "new_failures": 0, "new_collisions": 0},
        "interaction_audit": {
            "uri": "https://example.org/real-row-audit.json",
            "sha256": "1" * 64,
            "source_sha": HEAD,
        },
        "refute_review": {
            "head_sha": HEAD,
            "verdict": "accepted",
            "uri": "https://github.com/ll7/robot_sf_ll7/pull/123#issuecomment-456",
        },
    }


@pytest.mark.parametrize(
    "fault",
    [
        "missing_classification",
        "stale_head",
        "contradictory_totals",
        "undeclared_exception",
        "fallback",
        "omitted_arm",
        "omitted_map",
        "missing_job",
        "frozen_baseline",
        "bad_seed",
    ],
)
def test_invalid_receipt_is_blocked_by_process_contract(monkeypatch, tmp_path, fault):  # noqa: C901
    from scripts.ci import behaviour_receipt as adapter

    scope = {
        "arms": ["goal", "orca"],
        "maps": ["open", "door"],
        "vehicle_id": "t60",
        "exceptions": [],
    }
    policy = tmp_path / "scope.json"
    policy.write_text(json.dumps(scope))
    monkeypatch.setattr(adapter, "SCOPE_PATH", policy)
    monkeypatch.setattr(adapter, "current_head", lambda: HEAD)
    monkeypatch.setattr(adapter, "latest_release", lambda repo: "0.0.7")
    monkeypatch.setattr(adapter, "release_source", lambda release: "c" * 40)
    receipt = _receipt()
    if fault == "missing_classification":
        receipt["rows"][0]["collisions"] = 1
    elif fault == "stale_head":
        receipt["head_sha"] = "9" * 40
    elif fault == "contradictory_totals":
        receipt["totals"]["episodes"] += 1
    elif fault == "undeclared_exception":
        receipt["exceptions"] = [
            {
                "map": "door",
                "vehicle": "t60",
                "kind": "infeasible_by_design",
                "evidence": "https://example.org/exception.json",
            }
        ]
    elif fault == "fallback":
        receipt["rows"][0]["fallback"] = True
    elif fault == "omitted_arm":
        receipt["rows"] = [r for r in receipt["rows"] if r["arm"] != "orca"]
    elif fault == "omitted_map":
        receipt["rows"] = [r for r in receipt["rows"] if r["map"] != "door"]
    elif fault == "missing_job":
        receipt["scheduler"]["job_id"] = ""
    elif fault == "frozen_baseline":
        receipt["baseline"]["release"] = "0.0.8"
    elif fault == "bad_seed":
        receipt["rows"][0]["seed"] = 999
    body = "<!-- behaviour-change-receipt:v1\n" + json.dumps(receipt) + "\n-->"
    assert any(
        "behaviour receipt" in blocker.lower()
        for blocker in _checks(monkeypatch, body, ["robot_sf/planner/guarded_ppo.py"])
    )


def test_complete_receipt_and_tooling_exemption(monkeypatch, tmp_path):
    from scripts.ci import behaviour_receipt as adapter

    policy = tmp_path / "scope.json"
    policy.write_text(
        json.dumps(
            {
                "arms": ["goal", "orca"],
                "maps": ["open", "door"],
                "vehicle_id": "t60",
                "exceptions": [],
            }
        )
    )
    monkeypatch.setattr(adapter, "SCOPE_PATH", policy)
    monkeypatch.setattr(adapter, "current_head", lambda: HEAD)
    monkeypatch.setattr(adapter, "latest_release", lambda repo: "0.0.7")
    monkeypatch.setattr(adapter, "release_source", lambda release: "c" * 40)
    receipt = _receipt()
    receipt["rows"][0]["success"] = False
    receipt["totals"]["new_failures"] = 1
    receipt["classifications"] = [
        {
            "arm": "goal",
            "map": "open",
            "seed": 1001,
            "kind": "success_to_failure",
            "class": "known_limitation",
            "count": 1,
            "evidence": "https://example.org/trace.json",
        }
    ]
    body = "<!-- behaviour-change-receipt:v1\n" + json.dumps(receipt) + "\n-->"
    assert _checks(monkeypatch, body, ["robot_sf/planner/guarded_ppo.py"]) == []
    assert _checks(monkeypatch, "", ["scripts/dev/affected_test_selection.py"]) == []


def test_rename_out_of_behaviour_scope_keeps_old_endpoint(monkeypatch, tmp_path):
    """A real rename must not hide the deleted production path from the gate."""
    import subprocess

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, text=True, capture_output=True, check=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    source = tmp_path / "robot_sf/planner/old.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    git("add", "robot_sf/planner/old.py")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "docs").mkdir()
    source.rename(tmp_path / "docs/renamed.py")
    git("add", "robot_sf/planner/old.py", "docs/renamed.py")
    git("commit", "-m", "rename")
    monkeypatch.chdir(tmp_path)
    paths = checker.get_changed_files(None, base)
    assert "robot_sf/planner/old.py" in paths and "docs/renamed.py" in paths
    assert any("behaviour receipt" in b.lower() for b in _checks(monkeypatch, "", paths))
