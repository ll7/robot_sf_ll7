"""Producer provenance, real-gate admission, and pre-fix negative controls."""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer

REVIEWED_HEAD = "e7fd36f98fab243ce95cf45bc50ca8a9f7c30893"
ORIGINAL_MAIN = "de3e37774049b2c6d017116426ae522dc232f679"
RELEASE_TAG = "paper-matrix-v2-h600-s30-synthetic"


def _git_repo(root: Path):
    root.mkdir(exist_ok=True)

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    return git


def _release(tag="0.0.7", name="Robot SF 0.0.7", published="2026-09-17T16:20:07Z"):
    return {
        "draft": False,
        "prerelease": False,
        "tag_name": tag,
        "name": name,
        "published_at": published,
    }


def _release_inventory():
    return [
        [
            _release(
                RELEASE_TAG, "Robot SF benchmark data 0.0.8 - main only", "2026-10-09T17:39:53Z"
            ),
            _release(),
        ]
    ]


def _scope():
    return {
        "arms": ["goal"],
        "maps": ["open"],
        "vehicle_id": "t60",
        "exceptions": [],
        "arm_algorithms": {"goal": "goal"},
    }


def _fake_release_api(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\nprint({json.dumps(json.dumps(_release_inventory()))})\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])


def _raw_row(seed, *, failing=False):
    return {
        "algo": "goal",
        "scenario_id": "open",
        "seed": seed,
        "steps": 1,
        "metrics": {"success": not (failing and seed == 1001), "total_collision_count": 0},
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "native"},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v2",
                "dt": 0.1,
                "reset": {"robot": {"position": [0.0, 0.0]}},
                "steps": [
                    {
                        "robot": {"position": [0.1, 0.0]},
                        "pedestrians": [],
                        "planner": {
                            "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0}
                        },
                    }
                ],
            },
        },
    }


def _sweep_rows(root, label, *, source_sha="a" * 40, failing=False):
    out = root / label
    out.mkdir()
    raw_path = out / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_rows = [_raw_row(seed, failing=failing) for seed in range(1001, 1031)]
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    rows = [
        {
            "arm": "goal",
            "scenario": "open",
            "seed": row["seed"],
            "success": row["metrics"]["success"],
            "collisions": 0,
            "execution_status": "written",
            "trace_complete": True,
            "source_file": "runs/goal__differential_drive/episodes.jsonl",
        }
        for row in raw_rows
    ]
    (out / "episodes_main.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = {
        "suite": "main",
        "head_sha": source_sha,
        "seeds": list(range(1001, 1031)),
        "complete": True,
        "trace_requested": True,
        "campaign_execution_status": "completed",
        "exit_code": 0,
        "unexpected_failed_runs": 0,
        "expected_slots": 30,
        "written_rows": 30,
        "total_episodes": 30,
        "failed_slots": [],
        "missing_slots": [],
        "duplicate_slots": [],
        "unexpected_slots": [],
        "incomplete_trace_slots": [],
    }
    (out / "execution_main.json").write_text(json.dumps(meta))
    (out / "README.md").write_text(label + "\n")
    return out


def _existing(source_sha, out, digest=None):
    return producer._existing_sweep(
        source_sha,
        out,
        artifact_uri="https://example.org/sweep",
        artifact_sha256=digest,
        job_id="123",
    )


@pytest.mark.parametrize("claimed_sha", ["a" * 40, "c" * 40], ids=["head", "baseline"])
def test_existing_sweep_rejects_relabelled_source(tmp_path, claimed_sha):
    """Foreign recorded source cannot become either claimed execution source."""
    out = _sweep_rows(tmp_path, "foreign", source_sha="b" * 40)
    with pytest.raises(ValueError, match="recorded SHA differs"):
        _existing(claimed_sha, out)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_metadata",
        "incomplete",
        "wrong_suite",
        "wrong_seeds",
        "wrong_counts",
        "second_suite",
        "digest",
    ],
)
def test_existing_sweep_requires_complete_identity_and_actual_digest(tmp_path, fault):
    out = _sweep_rows(tmp_path, "sweep")
    meta_path = out / "execution_main.json"
    meta = json.loads(meta_path.read_text())
    if fault == "missing_metadata":
        meta_path.unlink()
    elif fault == "second_suite":
        shutil.copyfile(out / "episodes_main.jsonl", out / "episodes_width.jsonl")
    elif fault != "digest":
        key, value = {
            "incomplete": ("complete", False),
            "wrong_suite": ("suite", "width"),
            "wrong_seeds": ("seeds", [1001]),
            "wrong_counts": ("total_episodes", 29),
        }[fault]
        meta[key] = value
        meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        _existing("a" * 40, out, "f" * 64 if fault == "digest" else None)


@pytest.mark.parametrize(
    "fault",
    ["missing", "tag_name_conflict", "multiple_name_versions", "duplicate_highest", "latest_tie"],
)
def test_release_resolution_rejects_ambiguous_metadata(monkeypatch, fault):
    """Bad newest publication must never silently select 0.0.7 or an arbitrary tag."""
    new = _release(RELEASE_TAG, "Robot SF 0.0.8", "2026-10-09T17:39:53Z")
    rows = [_release(), new]
    if fault == "missing":
        new["name"] = "Robot SF publication"
    elif fault == "tag_name_conflict":
        new["tag_name"] = "0.0.7"
    elif fault == "multiple_name_versions":
        new["name"] = "Robot SF 0.0.7 and 0.0.8"
    else:
        duplicate = {**new, "tag_name": "baseline-b"}
        if fault == "duplicate_highest":
            duplicate["published_at"] = "2026-10-08T17:39:53Z"
        rows.append(duplicate)
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *a, **kw: json.dumps([rows]))
    with pytest.raises(ValueError, match="version|ambiguous"):
        gate.latest_release("ll7/robot_sf_ll7")


def test_latest_release_selects_published_0_0_8_from_release_name(monkeypatch):
    monkeypatch.setattr(
        gate.subprocess, "check_output", lambda *a, **kw: json.dumps(_release_inventory())
    )
    assert gate.latest_release("ll7/robot_sf_ll7") == RELEASE_TAG


def test_latest_release_keeps_publication_order_and_ignores_older_unversioned(monkeypatch):
    rows = [
        _release("0.0.8", "Robot SF 0.0.8"),
        _release("0.0.6", "Robot SF 0.0.6", "2026-10-09T17:39:53Z"),
        _release("old-data", "Benchmark archive", "2026-08-01T17:39:53Z"),
    ]
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *a, **kw: json.dumps([rows]))
    assert gate.latest_release("ll7/robot_sf_ll7") == "0.0.6"


@pytest.mark.parametrize("fault", ["no_controller", "fallback", "degraded", "missing"])
def test_audit_refuses_absent_or_tainted_controller_evidence(tmp_path, fault):
    """Audit status reflects execution faults rather than unconditional pass."""
    row = {
        "controller_executed": True,
        "fallback": False,
        "degraded": False,
        "algorithm": "goal",
        "execution_mode": "native",
    }
    if fault == "missing":
        row.pop("controller_executed")
    else:
        row[{"no_controller": "controller_executed"}.get(fault, fault)] = fault != "no_controller"
    audit = tmp_path / "audit.json"
    producer.write_real_row_audit(audit, source_sha="a" * 40, rows=[row], classifications=[])
    assert json.loads(audit.read_text())["status"] == (
        "fail" if fault in {"no_controller", "missing"} else "degraded"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_metadata",
        "missing_trace",
        "missing_action",
        "fallback_counter",
        "degraded_status",
        "contradictory_mode",
    ],
)
def test_row_construction_audits_recorded_trace_evidence(tmp_path, fault):
    head = {
        ("goal", "open", seed): {**_raw_row(seed), "execution_status": "written"}
        for seed in range(1001, 1031)
    }
    row = head[("goal", "open", 1001)]
    meta = row["algorithm_metadata"]
    if fault == "missing_metadata":
        row.pop("algorithm_metadata")
    elif fault == "missing_trace":
        meta.pop("simulation_step_trace")
    elif fault == "missing_action":
        meta["simulation_step_trace"]["steps"][0]["planner"].pop("selected_action")
    elif fault == "fallback_counter":
        meta["planner_diagnostics"] = {"fallback_count": 1}
    elif fault == "degraded_status":
        meta["status"] = "degraded"
    else:
        row["execution_mode"] = "adapter"
    rows, classes, _ = producer.build_rows_and_classifications(
        head,
        head,
        _scope(),
        classification_class="known_limitation",
        evidence_base_uri="https://example.org/classifications",
    )
    audit = tmp_path / "audit.json"
    producer.write_real_row_audit(audit, source_sha="a" * 40, rows=rows, classifications=classes)
    assert json.loads(audit.read_text())["status"] in {"fail", "degraded"}


def _executed_goal_with_optional_metrics():
    """Mirror #10313's seed-1001 goal row with 203 complete controller actions."""
    row = {**_raw_row(1001), "scenario_id": "classic_bottleneck_low", "execution_status": "written"}
    row["steps"] = 203
    metadata = row["algorithm_metadata"]
    step = metadata["simulation_step_trace"]["steps"][0]
    metadata["simulation_step_trace"]["steps"] = [copy.deepcopy(step) for _ in range(203)]
    metadata["planner_diagnostics"] = {"fallback_count": 0, "degraded_count": 0}
    metadata["paired_effect_metric_producer"] = {
        "schema_version": "paired_effect_metric_producer.v1",
        "status": "unavailable",
        "reason": "one_or_more_fields_unavailable",
        "metric_values": {},
        "fields": {
            "false_positive_stop_rate": {
                "status": "unavailable",
                "reason": "missing_safety_wrapper_summary",
            }
        },
    }
    return row


def test_optional_metrics_do_not_degrade_executed_controller_or_mutate_row():
    row = _executed_goal_with_optional_metrics()
    original = copy.deepcopy(row)
    evidence = producer._execution_evidence(row)
    assert evidence == {
        "algorithm": "goal",
        "controller_executed": True,
        "execution_mode": "native",
        "fallback": False,
        "degraded": False,
    }
    assert producer._audit_status([evidence]) == "pass"
    assert row == original


@pytest.mark.parametrize(
    "fault, axis, expected, audit_status",
    [
        ("row_fallback", "fallback", True, "degraded"),
        ("controller_fallback", "fallback", True, "degraded"),
        ("positive_counter", "fallback", True, "degraded"),
        ("negative_counter", "degraded", True, "degraded"),
        ("string_counter", "degraded", True, "degraded"),
        ("nonfinite_counter", "degraded", True, "degraded"),
        ("boolean_counter", "degraded", True, "degraded"),
        ("controller_status", "controller_executed", False, "fail"),
        ("nested_controller_status", "degraded", True, "degraded"),
        ("unavailable_controller_status", "degraded", True, "degraded"),
        ("missing_action", "controller_executed", False, "fail"),
        ("empty_action", "controller_executed", False, "fail"),
        ("missing_trace", "controller_executed", False, "fail"),
        ("incomplete_trace", "controller_executed", False, "fail"),
        ("wrong_mode", "execution_mode", "unknown", "fail"),
        ("contradictory_mode", "execution_mode", "unknown", "fail"),
        ("no_controller", "controller_executed", False, "fail"),
    ],
)
def test_optional_metrics_do_not_excuse_runtime_faults(fault, axis, expected, audit_status):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    metadata = row["algorithm_metadata"]
    trace = metadata["simulation_step_trace"]
    counters = metadata["planner_diagnostics"]
    actions = trace["steps"][0]["planner"]
    target, key, value = {
        "row_fallback": (row, "fallback", True),
        "controller_fallback": (metadata, "fallback_used", True),
        "positive_counter": (counters, "fallback_count", 1),
        "negative_counter": (counters, "fallback_count", -1),
        "string_counter": (counters, "fallback_count", "0"),
        "nonfinite_counter": (counters, "fallback_count", float("nan")),
        "boolean_counter": (counters, "fallback_count", False),
        "controller_status": (metadata, "status", "degraded"),
        "nested_controller_status": (metadata, "controller", {"status": "degraded"}),
        "unavailable_controller_status": (metadata, "controller", {"status": "unavailable"}),
        "missing_action": (actions, "selected_action", None),
        "empty_action": (actions, "selected_action", {}),
        "missing_trace": (metadata, "simulation_step_trace", None),
        "incomplete_trace": (trace, "steps", trace["steps"][:-1]),
        "wrong_mode": (metadata["planner_kinematics"], "execution_mode", "unknown"),
        "contradictory_mode": (row, "execution_mode", "adapter"),
        "no_controller": (row, "controller_executed", False),
    }[fault]
    if value is None:
        target.pop(key)
    else:
        target[key] = value
    evidence = producer._execution_evidence(row)
    assert evidence[axis] == expected
    assert producer._audit_status([evidence]) == audit_status


@pytest.mark.parametrize("location", ["row", "metadata", "controller"])
def test_metric_subtree_cannot_mask_controller_degradation(location):
    row = _executed_goal_with_optional_metrics()
    metadata = row["algorithm_metadata"]
    metrics = metadata["paired_effect_metric_producer"]
    metrics["fields"]["false_positive_stop_rate"]["status"] = "degraded"
    metrics["fields"]["false_positive_stop_rate"]["fallback_used"] = True
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    if location == "row":
        row["controller"] = {"status": "degraded"}
    elif location == "metadata":
        metadata["status"] = "degraded"
    else:
        metadata["controller"] = {"status": "degraded"}
    evidence = producer._execution_evidence(row)
    assert evidence["degraded"] is True
    assert producer._audit_status([evidence]) == ("fail" if location == "metadata" else "degraded")


def _payload_fixture(tmp_path, monkeypatch):
    git = _git_repo(tmp_path)
    _fake_release_api(tmp_path, monkeypatch)
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(_scope()))
    git("add", "scope.json")
    git("commit", "-m", "baseline")
    baseline_source = git("rev-parse", "HEAD")
    git("tag", RELEASE_TAG)
    git("tag", "0.0.7")
    git("commit", "--allow-empty", "-m", "execution source")
    source = git("rev-parse", "HEAD")
    scheduler = _existing(source, _sweep_rows(tmp_path, "head", source_sha=source, failing=True))
    baseline = _existing(
        baseline_source, _sweep_rows(tmp_path, "baseline", source_sha=baseline_source)
    )
    rows, classes, totals = producer.build_rows_and_classifications(
        producer._load_sweep_rows(scheduler.output_dir),
        producer._load_sweep_rows(baseline.output_dir),
        _scope(),
        classification_class="known_limitation",
        evidence_base_uri="https://example.org/e",
    )
    kwargs = {
        "repo_root": tmp_path,
        "receipt_id": "synthetic",
        "head_sha": source,
        "scheduler": scheduler,
        "baseline": baseline,
        "baseline_release": RELEASE_TAG,
        "baseline_body_id": "t60",
        "baseline_config_sha256": "d" * 64,
        "baseline_differences": ["synthetic behaviour-change fixture"],
        "scope": _scope(),
        "rows": rows,
        "classifications": classes,
        "totals": totals,
        "audit_uri": "https://example.org/audit",
        "audit_sha256": "e" * 64,
        "audit_source_sha": source,
        "refute_review_uri": producer.PLACEHOLDER_REVIEW_URI,
        "output_dir": tmp_path / "receipts/behaviour",
    }
    producer.write_receipt_and_header(**kwargs)
    git("add", "receipts/behaviour/synthetic.json")
    git("commit", "-m", "receipt payload")
    kwargs["head_sha"] = git("rev-parse", "HEAD")
    _, header = producer.write_receipt_and_header(**kwargs)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "SCOPE_PATH", scope_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", kwargs["head_sha"])
    return header


def test_placeholder_review_fails_real_gate(monkeypatch, tmp_path):
    """An untouched placeholder fails on review verdict, even with a valid URI."""
    header = _payload_fixture(tmp_path, monkeypatch)
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    assert gate.check_receipt(body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7")
    with pytest.raises(jsonschema.ValidationError) as error:
        gate.load_receipt(header, header["head_sha"])
    assert list(error.value.absolute_path) == ["refute_review", "verdict"]


def _revision_module(tmp_path, revision, filename):
    name = f"receipt_{revision[:8]}_{filename.removesuffix('.py')}"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:scripts/ci/{filename}"], cwd=producer.ROOT, text=True
    )
    path = tmp_path / filename
    path.write_text(source)
    schema = subprocess.check_output(
        ["git", "show", f"{revision}:scripts/ci/behaviour_receipt.schema.json"],
        cwd=producer.ROOT,
    )
    (tmp_path / "behaviour_receipt.schema.json").write_bytes(schema)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("finding", ["source", "review", "release", "audit"])
def test_p1_regression_controls_fail_on_reviewed_head(monkeypatch, tmp_path, finding):
    """Run regression assertions on e7fd36f9 and observe the specific defect."""
    old = _revision_module(
        tmp_path,
        REVIEWED_HEAD,
        "behaviour_receipt.py" if finding == "release" else "make_behaviour_receipt.py",
    )
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    if finding == "release":
        monkeypatch.setattr(sys.modules[__name__], "gate", old)
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            test_release_resolution_rejects_ambiguous_metadata(monkeypatch, "missing")
    else:
        monkeypatch.setattr(sys.modules[__name__], "producer", old)
        if finding == "source":
            with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
                test_existing_sweep_rejects_relabelled_source(fixture_dir, "a" * 40)
        elif finding == "review":
            header = _payload_fixture(fixture_dir, monkeypatch)
            body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
            assert (
                gate.check_receipt(body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7")
                == []
            )
            with pytest.raises(AssertionError):
                assert gate.check_receipt(
                    body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7"
                )
        else:
            with pytest.raises(AssertionError, match="pass"):
                test_audit_refuses_absent_or_tainted_controller_evidence(
                    fixture_dir, "no_controller"
                )


def _install_cli_fixture(tmp_path, monkeypatch):
    git = _git_repo(tmp_path)
    for filename in (
        "make_behaviour_receipt.py",
        "behaviour_receipt.py",
        "behaviour_receipt.schema.json",
    ):
        destination = tmp_path / "scripts/ci" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(producer.ROOT / "scripts/ci" / filename, destination)
    shutil.copyfile(producer.ROOT / "scripts/__init__.py", tmp_path / "scripts/__init__.py")
    validation = tmp_path / "scripts/validation"
    validation.mkdir()
    shutil.copyfile(
        producer.ROOT / "scripts/validation/run_empty_world_sweep.py",
        validation / "run_empty_world_sweep.py",
    )
    scope_path = tmp_path / gate.SCOPE_FILE
    scope_path.parent.mkdir(parents=True)
    scope_path.write_text(json.dumps(_scope()))
    git("add", "scripts", gate.SCOPE_FILE)
    git("commit", "-m", "baseline launcher")
    baseline_source = git("rev-parse", "HEAD")
    git("tag", RELEASE_TAG)
    git("tag", "0.0.7")
    marker = tmp_path / "robot_sf/planner/synthetic.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("BEHAVIOUR = 'changed'\n")
    git("add", "robot_sf/planner/synthetic.py")
    git("commit", "-m", "synthetic behaviour change")
    source = git("rev-parse", "HEAD")
    head = _sweep_rows(tmp_path, "head", source_sha=source, failing=True)
    baseline = _sweep_rows(tmp_path, "baseline", source_sha=baseline_source)
    _fake_release_api(tmp_path, monkeypatch)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "SCOPE_PATH", scope_path)
    return git, source, head, baseline


def test_file_cli_and_real_gate_cover_name_only_baseline(monkeypatch, tmp_path):
    """File CLI and actual release resolution distinguish origin/main without PYTHONPATH."""
    git, source, head, baseline = _install_cli_fixture(tmp_path, monkeypatch)
    script = tmp_path / "scripts/ci/make_behaviour_receipt.py"
    command = [
        sys.executable,
        str(script),
        "--head-sha",
        source,
        "--baseline",
        "latest",
        "--receipt-id",
        "synthetic",
        "--mode",
        "existing",
        "--head-sweep-dir",
        str(head),
        "--baseline-sweep-dir",
        str(baseline),
        "--job-id",
        "123",
        "--baseline-job-id",
        "456",
    ]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert (
        json.loads(first.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])["baseline"]["release"]
        == RELEASE_TAG
    )
    git("add", "receipts/behaviour/synthetic.json")
    git("commit", "-m", "receipt payload")
    final_head = git("rev-parse", "HEAD")
    command[command.index("--head-sha") + 1] = final_head
    result = subprocess.run(
        [*command, "--head-source-sha", source],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    header = json.loads(result.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
    assert header["scheduler"]["source_sha"] == source
    assert header["interaction_audit"]["source_sha"] == source
    assert header["refute_review"]["verdict"] == "pending_independent_review"
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", final_head)
    changed = ["robot_sf/planner/synthetic.py"]
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    assert gate.check_receipt(body, changed, "ll7/robot_sf_ll7")
    # Acceptance belongs to a separate reviewer, represented by this test actor.
    header["refute_review"] = {
        "head_sha": final_head,
        "verdict": "accepted",
        "uri": "https://example.org/independent-review",
    }
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    assert gate.check_receipt(body, changed, "ll7/robot_sf_ll7") == []
    original = _revision_module(tmp_path, ORIGINAL_MAIN, "behaviour_receipt.py")
    original.ROOT, original.SCOPE_PATH = tmp_path, gate.SCOPE_PATH
    assert original.latest_release("ll7/robot_sf_ll7") == "0.0.7"
    receipt = original.load_receipt(header, final_head)
    with pytest.raises(ValueError, match="baseline is not the latest published software release"):
        original.validate_receipt(
            receipt, _scope(), final_head, "0.0.7", original.release_source("0.0.7")
        )
    assert original.check_receipt(body, changed, "ll7/robot_sf_ll7")


def test_seed_guard_names_wider_dev_range_but_keeps_current_gate_contract():
    assert producer._check_seed_range([1001, 1030]) == [1001, 1030]
    with pytest.raises(ValueError, match="current behaviour gate admits only seeds"):
        producer._check_seed_range([1031])
    with pytest.raises(ValueError, match="outside development range"):
        producer._check_seed_range([1201])
