"""Exercise receipt admission, Git source binding and immutable-base CI policy."""

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
    from scripts.ci import behaviour_receipt as adapter

    blockers = adapter.check_receipt(body, paths, "ll7/robot_sf_ll7")
    return (
        blockers
        + checker.run_all_checks(
            "change", body, paths, "ll7/robot_sf_ll7", checker.PRDiffBases("origin/main"), None
        )[0]
    )


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


def _header(receipt, raw, path="receipts/behaviour/sweep.json"):
    """Build the independently specified compact body around payload bytes."""
    header = {
        key: value for key, value in receipt.items() if key not in {"rows", "classifications"}
    }
    header["schema_version"] = "behaviour-change-receipt-header.v2"
    header["rows_artifact"] = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    header["classifications"] = {
        "count": len(receipt["classifications"]),
        "sha256": hashlib.sha256(
            json.dumps(receipt["classifications"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"


def _commit_receipt(monkeypatch, tmp_path, receipt):
    """Commit only rows/classifications, then bind event metadata to the resulting source."""
    import subprocess

    from scripts.ci import behaviour_receipt as adapter

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("commit", "--allow-empty", "-m", "executed run source")
    run_source = git("rev-parse", "HEAD")
    path = "receipts/behaviour/sweep.json"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    raw = json.dumps(
        {
            "schema_version": "behaviour-change-rows.v1",
            "rows": receipt["rows"],
            "classifications": receipt["classifications"],
        }
    ).encode()
    file.write_bytes(raw)
    git("add", path)
    git("commit", "-m", "receipt payload")
    head = git("rev-parse", "HEAD")
    if receipt["head_sha"] == HEAD:
        receipt["head_sha"] = head
    receipt["scheduler"]["source_sha"] = run_source
    receipt["interaction_audit"]["source_sha"] = run_source
    receipt["refute_review"]["head_sha"] = head
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    return _header(receipt, raw)


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
    body = _commit_receipt(monkeypatch, tmp_path, receipt)
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
    body = _commit_receipt(monkeypatch, tmp_path, receipt)
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
    assert "ref" not in checkout["with"]
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
    # Full history/tag hydration retains the default merge identity.
    git(clone, "fetch", "--unshallow", "--tags", "origin")
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    monkeypatch.setattr(adapter, "ROOT", clone)
    assert git(clone, "rev-parse", "HEAD") == merge
    assert adapter.current_head() == head
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
    body = _header(_receipt(), b"unread payload")
    blockers = adapter.check_receipt(body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7")
    assert any("reviewed scope inventory" in b for b in blockers), blockers


@pytest.mark.parametrize(
    ("paths", "blocked"),
    [
        (["docs/receipt_notes.md"], False),
        (["configs/benchmarks/releases/behaviour_gate_0_1_0.json"], False),
        (["robot_sf/planner/guarded_ppo.py"], True),
        (
            [
                "configs/benchmarks/releases/behaviour_gate_0_1_0.json",
                "robot_sf/planner/guarded_ppo.py",
            ],
            True,
        ),
        (["configs/benchmarks/paper_experiment_matrix_v1.yaml"], True),
        (["configs/algos/policy.yaml"], True),
        (["configs/baselines/ppo.yaml"], True),
        (["configs/planners/goal.yaml"], True),
        (["configs/robots/body.yaml"], True),
        (["model/policy.zip"], True),
        (["model/registry.yaml"], True),
        (["robot_sf/models/registry.py"], True),
        (["robot_sf/sensor/raycast.py"], True),
        (["robot_sf/ped_npc/force.py"], True),
        (["robot_sf/common/seed.py"], True),
        (["robot_sf/training/scenario_loader.py"], True),
        (["robot_sf/prediction/model.py"], True),
        (["robot_sf/feature_extractors/grid.py"], True),
        (["fast-pysf/accelerator.py"], True),
        (["scripts/benchmark/runner.py"], True),
        (["scripts/benchmark_planner.py"], True),
        (["scripts/tools/run_camera_ready_benchmark.py"], True),
        (["scripts/tools/run_benchmark_release.py"], True),
        (["scripts/tools/run_split_camera_ready_campaign.py"], True),
        (["scripts/tools/benchmark_feature_extractors.py"], True),
        (["robot_sf/feature_extractor.py"], True),
        (["robot_sf/ped_ego/unicycle_drive.py"], True),
        (["robot_sf/core/time.py"], False),
        (["robot_sf/planner/README.md"], False),
        (["configs/algos/README.md"], False),
        (["model/README.md"], False),
        (["fast-pysf/README.md"], False),
        (["scripts/benchmark/README.md"], False),
    ],
)
def test_real_diff_scope_controls(monkeypatch, tmp_path, paths, blocked):
    """Actual Git diffs exempt prose/inventory while retaining planner/campaign admission."""
    import subprocess

    from scripts.ci import behaviour_receipt as adapter

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("commit", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")
    git("add", *paths)
    git("commit", "-m", "change")
    monkeypatch.chdir(tmp_path)
    # Exempt paths must not consult inventory/release state even if absent.
    monkeypatch.setattr(adapter, "SCOPE_PATH", tmp_path / "absent.json")

    def unexpected_release_query(*args):
        pytest.fail("scope controls should not query release metadata")

    monkeypatch.setattr(adapter, "latest_release", unexpected_release_query)
    changed = checker.get_changed_files(None, base)
    assert set(changed) == set(paths)
    blockers = _checks(monkeypatch, "", changed)
    if blocked:
        assert blockers == [
            "BLOCKER: behaviour receipt missing or duplicated for a behaviour-changing PR"
        ]
    else:
        assert blockers == []


def test_installed_owner_inventory_matches_sweep_inputs():
    """The shipped inventory covers both sweep suites and preserves the declared probe."""
    from pathlib import Path

    import yaml

    from robot_sf.robot.differential_drive import DifferentialDriveSettings
    from robot_sf.training.scenario_loader import load_scenarios
    from scripts.validation.run_empty_world_sweep import SUITES

    root = Path(__file__).resolve().parents[2]
    policy = root / "configs/benchmarks/releases/behaviour_gate_0_1_0.json"
    assert policy.is_file(), "behaviour gate owner inventory must ship with the gate"
    from scripts.ci import behaviour_receipt as adapter

    scope = json.loads(policy.read_text())
    assert scope["owner"] == "release-integration"
    assert scope["vehicle_id"] == "differential_drive_r1m"
    assert DifferentialDriveSettings().radius == 1.0
    arms = {}
    maps = {}
    probes = []
    for suite, config_path in SUITES.items():
        config_file = root / config_path
        config = yaml.safe_load(config_file.read_text())
        matrix = root / config["scenario_matrix"]
        provenance = scope["sources"][suite]
        assert provenance["campaign"] == config_path
        assert provenance["campaign_sha256"] == hashlib.sha256(config_file.read_bytes()).hexdigest()
        assert provenance["matrix"] == config["scenario_matrix"]
        assert provenance["matrix_sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()
        arms.update({arm["key"]: arm["algo"] for arm in config["planners"]})
        for row in load_scenarios(matrix, base_dir=matrix.parent):
            assert not row.get("map_id"), "inventory map paths must follow resolved map authority"
            map_file = (matrix.parent / row["map_file"]).resolve()
            maps[row["name"]] = map_file.relative_to(root).as_posix()
            assert (
                scope["map_sha256"][row["name"]]
                == hashlib.sha256(map_file.read_bytes()).hexdigest()
            )
            if row.get("infeasibility_probe"):
                probes.append(row["name"])
    assert len(arms) == 14 and len(maps) == 51
    assert set(scope["arms"]) == set(arms)
    assert scope["arm_algorithms"] == arms
    assert set(scope["maps"]) == set(maps)
    assert scope["map_files"] == maps
    assert probes == ["francis2023_narrow_doorway"]
    assert [(e["map"], e["vehicle"], e["kind"]) for e in scope["exceptions"]] == [
        ("francis2023_narrow_doorway", "differential_drive_r1m", "infeasible_by_design")
    ]
    # Exercise the actual validator with the installed roster, not only a template check.
    adapter.validate_receipt(_inventory_receipt(scope), scope, HEAD, "0.0.7", "c" * 40)


def _inventory_receipt(scope):
    """Generate deterministic execution rows for all independently checked scope slots."""
    from robot_sf.benchmark.algorithm_metadata import (
        canonical_algorithm_name,
        enrich_algorithm_metadata,
    )

    receipt = _receipt()
    receipt["scope_sha256"] = hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt["vehicle"]["id"] = scope["vehicle_id"]
    receipt["exceptions"] = scope["exceptions"]
    prototype = receipt["rows"][0]
    receipt["rows"] = []
    for arm in scope["arms"]:
        algo = canonical_algorithm_name(scope["arm_algorithms"][arm])
        profile = enrich_algorithm_metadata(algo=algo)["planner_kinematics"]
        mode = "native" if profile["supports_native_commands"] else "adapter"
        for map_id in scope["maps"]:
            for seed in range(1001, 1031):
                receipt["rows"].append(
                    {
                        **prototype,
                        "arm": arm,
                        "algorithm": algo,
                        "execution_mode": mode,
                        "map": map_id,
                        "seed": seed,
                    }
                )
    receipt["totals"]["episodes"] = 21420
    return receipt
