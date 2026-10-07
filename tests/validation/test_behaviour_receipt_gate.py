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
        "robot_sf/benchmark/runner.py",
        "robot_sf/benchmark/types.py",
        "robot_sf/benchmark/schemas/episode.schema.v1.json",
        "robot_sf/benchmark/map_runner_jsonl.py",
        "robot_sf/benchmark/schema_loader.py",
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
            "execution_mode": "native" if arm == "goal" else "adapter",
            "algorithm": arm,
            "controller_executed": True,
            "degraded": False,
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


def test_contract_checkout_uses_source_identity_and_hydrates_release_tags(monkeypatch, tmp_path):
    """A source receipt survives a workflow-shaped shallow synthetic-merge checkout."""
    import subprocess
    from pathlib import Path

    import yaml

    from scripts.ci import behaviour_receipt as adapter

    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/pr-contract-check.yml"
        ).read_text()
    )
    checkout = next(
        s for s in workflow["jobs"]["pr-contract-check"]["steps"] if s.get("name") == "Checkout"
    )
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["with"]["fetch-depth"] == 0

    source = tmp_path / "source"
    source.mkdir()

    def git(root, *args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.invalid")
    (source / "file").write_text("baseline")
    git(source, "add", "file")
    git(source, "commit", "-m", "baseline")
    baseline = git(source, "rev-parse", "HEAD")
    git(source, "tag", "0.0.7")
    (source / "file").write_text("source")
    git(source, "add", "file")
    git(source, "commit", "-m", "PR source")
    head = git(source, "rev-parse", "HEAD")
    git(source, "checkout", "-b", "synthetic-merge")
    git(source, "commit", "--allow-empty", "-m", "synthetic merge identity")
    merge = git(source, "rev-parse", "HEAD")
    clone = tmp_path / "checkout"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "--no-tags",
            "--branch",
            "synthetic-merge",
            source.as_uri(),
            str(clone),
        ],
        check=True,
    )
    assert git(clone, "rev-parse", "HEAD") == merge != head
    assert git(clone, "tag") == ""
    # checkout@ fetch-depth: 0 hydrates history/tags before resolving the event source.
    git(clone, "fetch", "--unshallow", "--tags", "origin")
    git(clone, "checkout", "--detach", head)
    monkeypatch.setattr(adapter, "ROOT", clone)
    receipt = _receipt()
    receipt["head_sha"] = receipt["scheduler"]["source_sha"] = head
    receipt["interaction_audit"]["source_sha"] = receipt["refute_review"]["head_sha"] = head
    receipt["baseline"]["source_sha"] = baseline
    scope = {
        "arms": ["goal", "orca"],
        "maps": ["open", "door"],
        "vehicle_id": "t60",
        "exceptions": [],
    }
    adapter.validate_receipt(
        receipt, scope, adapter.current_head(), "0.0.7", adapter.release_source("0.0.7")
    )
    with pytest.raises(ValueError, match="receipt head is stale"):
        adapter.validate_receipt(receipt, scope, merge, "0.0.7", baseline)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "solver_skipped",
        "solver_fallback",
        "degraded",
        "native_capable_adapter",
        "algorithm_substitution",
    ],
)
def test_adapter_receipt_requires_registry_mode_and_real_controller(fault):
    """Command adaptation is valid only for the bound adapter-only controller."""
    from scripts.ci import behaviour_receipt as adapter

    receipt = _receipt()
    scope = {
        "arms": ["goal", "orca"],
        "maps": ["open", "door"],
        "vehicle_id": "t60",
        "exceptions": [],
        "arm_algorithms": {"goal": "prediction_mpc", "orca": "orca"},
    }
    receipt["scope_sha256"] = hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for row in receipt["rows"]:
        if row["arm"] == "goal":
            row["algorithm"] = "prediction_mpc"
            row["execution_mode"] = "adapter"
    row = receipt["rows"][0]
    if fault == "solver_skipped":
        row["controller_executed"] = False
    elif fault == "solver_fallback":
        row["fallback"] = True
    elif fault == "degraded":
        row["degraded"] = True
    elif fault == "native_capable_adapter":
        scope["arm_algorithms"]["goal"] = row["algorithm"] = "goal"
        receipt["scope_sha256"] = hashlib.sha256(
            json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif fault == "algorithm_substitution":
        row["algorithm"] = "learned_prediction_mpc"
    if fault is None:
        adapter.validate_receipt(receipt, scope, HEAD, "0.0.7", "c" * 40)
    else:
        reason = {
            "solver_skipped": "intended solver/controller did not execute",
            "solver_fallback": "fallback or degraded execution",
            "degraded": "fallback or degraded execution",
            "native_capable_adapter": "command execution mode",
            "algorithm_substitution": "row algorithm differs",
        }[fault]
        with pytest.raises(ValueError, match=reason):
            adapter.validate_receipt(receipt, scope, HEAD, "0.0.7", "c" * 40)


def test_missing_owner_inventory_is_explicit_blocker(monkeypatch, tmp_path):
    """No fixture or inferred roster can activate the integration-owned gate."""
    from scripts.ci import behaviour_receipt as adapter

    monkeypatch.setattr(adapter, "SCOPE_PATH", tmp_path / "missing.json")
    body = "<!-- behaviour-change-receipt:v1\n" + json.dumps(_receipt()) + "\n-->"
    blockers = adapter.check_receipt(body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7")
    assert any("reviewed scope inventory" in b for b in blockers), blockers
