"""Producer provenance, real-gate admission, and pre-fix negative controls."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
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
REAL_ROWS = Path(__file__).parents[1] / "fixtures/behaviour_receipt_baseline_31302"
REAL_ROW_MANIFEST = json.loads((REAL_ROWS / "manifest.json").read_text())


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
    """Use the untouched #10313 record, with only the loader's written attachment."""
    return {
        **json.loads((REAL_ROWS / "goal__differential_drive.jsonl").read_bytes()),
        "execution_status": "written",
    }


@pytest.mark.parametrize("record", REAL_ROW_MANIFEST["rows"], ids=lambda row: row["algorithm"])
def test_byte_exact_baseline_controller_rows_are_admitted(record):
    raw = (REAL_ROWS / record["path"]).read_bytes()
    assert len(raw) == record["bytes"]
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    if record["algorithm"] == "goal":
        assert len(raw) == 397583
        assert (
            hashlib.sha256(raw).hexdigest()
            == "6084b29551a667654d511dad222a96b777910d42e1672aad4617daff707b10ab"
        )
    row = json.loads(raw)
    assert row["seed"] == 1001
    original = copy.deepcopy(row)
    evidence = producer._execution_evidence({**row, "execution_status": "written"})
    assert evidence["controller_executed"] is True
    assert evidence["algorithm"] == record["algorithm"]
    assert producer._audit_status([evidence]) == "pass"
    assert row == original


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


def _executed_orca_row():
    return {
        **json.loads((REAL_ROWS / "orca__differential_drive.jsonl").read_bytes()),
        "execution_status": "written",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "cbf_counter",
        "nmpc_counter",
        "guard_counter",
        "cbf_decision",
        "shield_decision",
        "cbf_state_only",
        "cbf_episode_steps",
        "cbf_histogram",
    ],
)
def test_real_row_emitted_runtime_fallback_reports_are_degraded(fault):
    """The reviewed projection admitted these actual emitter-shaped fault reports."""
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    metadata = row["algorithm_metadata"]
    if fault == "cbf_counter":
        metadata["planner_runtime"]["cbf_safety_filter"] = {
            "schema_version": "cbf-safety-filter-stats.v1",
            "fallback_count": 1,
        }
    elif fault == "nmpc_counter":
        row["algo"] = "nmpc_social"
        metadata.update(algorithm="nmpc_social", canonical_algorithm="nmpc_social")
        metadata["planner_contract"]["planner_id"] = "nmpc_social"
        metadata["planner_runtime"] = {"calls": 1, "solver_failures": 1, "fallback_stop_count": 1}
    elif fault == "guard_counter":
        metadata["guard_stats"] = {"fallback_count": 1}
    elif fault == "cbf_episode_steps":
        metadata["cbf_safety_filter"] = {
            "fallback_step_count": 0,
            "steps": [{"fallback_applied": True}],
        }
    elif fault == "cbf_histogram":
        metadata["shield_stats"] = {"decision_counts": {"cbf_best_effort": 1}}
    else:
        decision = {
            "schema_version": "shield-decision.v1",
            "decision_label": "cbf_best_effort",
            "fallback_controller_state": {
                "filter": "CollisionConeCbfSafetyFilter",
                "variant": "collision_cone",
                "fallback": True,
            },
        }
        if fault == "cbf_state_only":
            decision["decision_label"] = "cbf_feasible"
        if fault == "cbf_decision":
            metadata["planner_runtime"]["cbf_safety_filter"] = {"last_decision": decision}
        else:
            metadata["shield_stats"] = {"last_decision": decision}
    evidence = producer._execution_evidence(row)
    assert evidence["controller_executed"] is True
    assert evidence["degraded"] is True
    assert producer._audit_status([evidence]) == "degraded"


@pytest.mark.parametrize("location", ["cbf", "nmpc", "guard"])
@pytest.mark.parametrize("counter", [1, "0", True, -1, float("nan"), float("inf"), {}, []])
def test_real_row_emitted_fallback_counters_fail_closed(location, counter):
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    metadata = row["algorithm_metadata"]
    if location == "cbf":
        metadata["planner_runtime"]["cbf_safety_filter"] = {"fallback_count": counter}
    elif location == "nmpc":
        metadata["planner_runtime"]["fallback_stop_count"] = counter
    else:
        metadata["guard_stats"] = {"fallback_count": counter}
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


def test_real_row_live_cbf_reports_distinguish_feasible_and_fallback():
    from robot_sf.planner.cbf_safety_filter import (
        CbfSafetyFilterConfig,
        CollisionConeCbfSafetyFilter,
    )
    from robot_sf.planner.safety_shield import new_shield_stats, update_shield_stats

    def audit(agents):
        filter_ = CollisionConeCbfSafetyFilter(CbfSafetyFilterConfig(enabled=True))
        decision = filter_.filter_command(
            {
                "robot": {
                    "position": [0.0, 0.0],
                    "velocity": [0.0, 0.0],
                    "heading": 0.0,
                    "radius": 0.3,
                },
                "agents": agents,
            },
            (0.8, 0.0),
        )
        row = _executed_orca_row()
        row["algorithm_metadata"]["planner_runtime"]["cbf_safety_filter"] = filter_.diagnostics()
        row["algorithm_metadata"]["shield_stats"] = update_shield_stats(
            new_shield_stats(), decision
        )
        return producer._audit_status([producer._execution_evidence(row)]), filter_.diagnostics()

    healthy, clean_report = audit([])
    assert clean_report["fallback_count"] == 0
    assert healthy == "pass"
    faulty, fault_report = audit(
        [
            {"position": [0.1, 0.0], "velocity": [-0.6, 0.0], "radius": 0.3},
            {"position": [-0.1, 0.0], "velocity": [0.6, 0.0], "radius": 0.3},
        ]
    )
    assert fault_report["fallback_count"] == 1
    assert fault_report["last_decision"]["decision_label"] == "cbf_best_effort"
    assert faulty == "degraded"


_EMITTER_KEY = re.compile(
    r"fallback|degrad|(?:^|_)status(?:$|_)|stop_count|stop_safe|stop_best_effort|safe_stop|execution_mode|decision_label",
    re.IGNORECASE,
)


def _serialized_dataclass_keys(node, owner):
    if not isinstance(node.target, ast.Name) or not isinstance(owner, ast.ClassDef):
        return []
    dataclass_decorated = any(
        (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
        or (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "dataclass"
        )
        for decorator in owner.decorator_list
    )
    return [node.target.id] if dataclass_decorated else []


def _emitted_fault_keys(source):
    """Scan writes, not reads/comments: dict literals, store subscripts, dict/update/setdefault."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    emitted = set()
    for node in ast.walk(tree):
        keys = []
        if isinstance(node, ast.Dict):
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
        ):
            keys = [node.slice.value] if isinstance(node.slice.value, str) else []
        elif isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == "dict") or (
                isinstance(node.func, ast.Attribute) and node.func.attr in {"update", "setdefault"}
            ):
                keys = [kw.arg for kw in node.keywords if kw.arg]
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and node.args
                ):
                    if isinstance(node.args[0], ast.Constant) and isinstance(
                        node.args[0].value, str
                    ):
                        keys.append(node.args[0].value)
        elif isinstance(node, ast.AnnAssign):
            # asdict serializes these fields without literal emitted keys.
            keys = _serialized_dataclass_keys(node, parents.get(node))
        parent = node
        while parent in parents and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parents[parent]
        function = (
            parent.name
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "<module>"
        )
        emitted.update((function, key) for key in keys if _EMITTER_KEY.search(key))
    return emitted


def _runtime_emitter_inventory():
    root = Path(__file__).resolve().parents[2]
    directories = (
        "robot_sf/planner",
        "robot_sf/baselines",
        "robot_sf/sim",
        "robot_sf/benchmark/map_runner",
        "robot_sf/benchmark/map_runner_policies",
        "robot_sf/benchmark/safety",
        "fast-pysf/pysocialforce",
    )
    paths = set()
    for directory in directories:
        files = set((root / directory).rglob("*.py"))
        assert files, f"missing emitter source tree: {directory}"
        paths.update(files)
    paths.update((root / "robot_sf").rglob("*adapter*.py"))
    return {
        (str(path.relative_to(root)), function, key)
        for path in paths
        for function, key in _emitted_fault_keys(path.read_text())
    }


def _unclassified_emitter_keys(inventory):
    locations = producer.EXECUTION_EVIDENCE_LOCATIONS
    runtime = (
        set(locations["runtime_fields"])
        | set(locations["counter_fields"])
        | set(locations.get("runtime_aliases", {}))
    )
    non_runtime = set(locations.get("non_runtime_fields", {}))
    exceptions = locations.get("non_runtime_emissions", {})
    return {
        site
        for site in inventory
        if site[2] not in runtime | non_runtime and site not in exceptions
    }


def test_planner_adapter_guard_emitted_fault_keys_are_classified():
    inventory = _runtime_emitter_inventory()
    assert any(key == "fallback_stop_count" for _, _, key in inventory)
    assert any(key == "fallback_count" for _, _, key in inventory)
    assert not _unclassified_emitter_keys(inventory), sorted(_unclassified_emitter_keys(inventory))


def test_emitter_catcher_distinguishes_writes_from_reads_and_detects_new_keys():
    emitted = _emitted_fault_keys("""
from dataclasses import dataclass
@dataclass
class FutureReport:
    new_degraded_status: str = "none"
def report(out):
    ignored = out.get("unseen_read_fallback")
    out["new_fallback_count"] = 1
    out.update(new_degraded_flag=True)
    out.setdefault("new_safe_stop_count", 0)
    return dict(new_used_fallback=True, **{"new_status": "fallback"})
""")
    assert {key for _, key in emitted} == {
        "new_fallback_count",
        "new_degraded_flag",
        "new_safe_stop_count",
        "new_used_fallback",
        "new_status",
        "new_degraded_status",
    }
    inventory = {("robot_sf/planner/future_planner.py", function, key) for function, key in emitted}
    assert _unclassified_emitter_keys(inventory) == inventory


@pytest.mark.parametrize(
    "key,value",
    [
        ("ever_degraded", True),
        ("degradation_reasons", ["missing_controller_input"]),
        ("fallback_reasons", {"numerical_force_fallback": 1}),
        ("fallback_status", "goal_fallback"),
        ("fallback_from", "failed_planner_head"),
        ("runtime_status", "failed"),
        ("last_step_status", "failed"),
        ("observation_validation_status", "invalid"),
        ("guard_or_fallback_reason", "fallback_to_stop"),
    ],
)
def test_real_row_inventory_runtime_aliases_cannot_hide_faults(key, value):
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"]["planner_runtime"][key] = value
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


@pytest.mark.parametrize(
    "binding_fault",
    ["wrong_filter", "wrong_variant", "wrong_schema", "non_boolean", "non_string_filter"],
)
def test_real_row_cbf_state_binding_rejects_malformed_reports(binding_fault):
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    decision = {
        "schema_version": "shield-decision.v1",
        "decision_label": "cbf_feasible",
        "fallback_controller_state": {
            "filter": "CollisionConeCbfSafetyFilter",
            "variant": "collision_cone",
            "fallback": False,
        },
    }
    if binding_fault == "wrong_schema":
        decision["schema_version"] = "unbound"
    else:
        field, value = {
            "wrong_filter": ("filter", "unknown_filter"),
            "wrong_variant": ("variant", "unknown_variant"),
            "non_boolean": ("fallback", "false"),
            "non_string_filter": ("filter", []),
        }[binding_fault]
        decision["fallback_controller_state"][field] = value
    row["algorithm_metadata"]["shield_stats"] = {"last_decision": decision}
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


@pytest.mark.parametrize("location", ["nmpc", "cbf", "wrapper"])
def test_real_row_zero_emitted_fallback_counters_remain_healthy(location):
    row = _executed_orca_row()
    if location == "nmpc":
        row["algorithm_metadata"]["planner_runtime"]["fallback_stop_count"] = 0
    elif location == "cbf":
        row["algorithm_metadata"]["planner_runtime"]["cbf_safety_filter"] = {"fallback_count": 0}
    else:
        row["algorithm_metadata"]["planner_runtime"]["fast_pysf_wrapper"] = {
            "fallback": False,
            "fallback_count": 0,
            "fallback_reason": None,
            "fallback_reasons": {},
        }
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"


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


def _runtime_object_for_fault(row, path):
    value = row
    for key in path:
        value = value[-1] if key == "*" else value.setdefault(key, {})
    return value


@pytest.mark.parametrize("status", ["degraded", "fallback", "unavailable"])
@pytest.mark.parametrize(
    "path",
    [
        (),
        ("controller",),
        ("algorithm_metadata",),
        ("algorithm_metadata", "controller"),
        ("algorithm_metadata", "planner_diagnostics"),
        ("algorithm_metadata", "planner_runtime"),
        ("algorithm_metadata", "planner_runtime", "last_decision"),
        ("algorithm_metadata", "planner_runtime", "checkpoint_provenance"),
        ("algorithm_metadata", "foresight_prediction"),
        ("algorithm_metadata", "planner_runtime", "foresight_prediction"),
        ("fallback_diagnostics",),
        ("algorithm_metadata", "fallback_diagnostics"),
        ("algorithm_metadata", "planner_runtime", "fallback_diagnostics"),
        ("algorithm_metadata", "fallback_controller_state"),
        ("algorithm_metadata", "planner_runtime", "fallback_controller_state"),
        ("algorithm_metadata", "simulation_step_trace", "steps", "*", "planner"),
    ],
)
def test_real_row_rejects_fault_status_at_each_runtime_location(path, status):
    row = _executed_goal_with_optional_metrics()
    if "fallback_controller_state" in path:
        row["algo"] = "guarded_ppo"
        row["algorithm_metadata"].update(algorithm="ppo", canonical_algorithm="guarded_ppo")
        row["algorithm_metadata"]["planner_contract"]["planner_id"] = "guarded_ppo"
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _runtime_object_for_fault(row, path)["status"] = status
    evidence = producer._execution_evidence(row)
    assert evidence["degraded"] is True
    assert producer._audit_status([evidence]) == (
        "fail" if path == ("algorithm_metadata",) else "degraded"
    )


@pytest.mark.parametrize(
    "counter",
    [1, -1, "0", False, None, float("nan"), float("inf"), [], {}, 10**1000],
    ids=[
        "positive",
        "negative",
        "string",
        "boolean",
        "null",
        "nan",
        "infinite",
        "list",
        "mapping",
        "oversized",
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        (),
        ("algorithm_metadata",),
        ("algorithm_metadata", "planner_diagnostics"),
        ("algorithm_metadata", "planner_runtime"),
        ("algorithm_metadata", "planner_runtime", "last_decision"),
        ("algorithm_metadata", "simulation_step_trace", "steps", "*", "planner"),
    ],
)
def test_real_row_rejects_bad_runtime_counter_shapes(path, counter):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _runtime_object_for_fault(row, path)["fallback_count"] = counter
    evidence = producer._execution_evidence(row)
    assert evidence["fallback"] is True
    assert evidence["degraded"] is True
    assert producer._audit_status([evidence]) == "degraded"


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "written",
        "metadata",
        "status",
        "mode",
        "step_count",
        "trace",
        "empty_trace",
        "malformed_trace",
        "schema",
        "dt",
        "reset",
        "empty_steps",
        "last_action",
        "empty_last_action",
        "malformed_last_action",
        "malformed_last_planner",
        "controller_flag",
    ],
)
def test_real_row_missing_required_execution_evidence_fails_closed(fault):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    metadata = row["algorithm_metadata"]
    trace = metadata["simulation_step_trace"]
    action = trace["steps"][-1]["planner"]
    target, key, replacement = {
        "identity": (row, "algo", None),
        "written": (row, "execution_status", None),
        "metadata": (row, "algorithm_metadata", None),
        "status": (metadata, "status", None),
        "mode": (metadata["planner_kinematics"], "execution_mode", None),
        "step_count": (row, "steps", None),
        "trace": (metadata, "simulation_step_trace", None),
        "empty_trace": (metadata, "simulation_step_trace", {}),
        "malformed_trace": (metadata, "simulation_step_trace", "not-a-trace"),
        "schema": (trace, "schema_version", None),
        "dt": (trace, "dt", None),
        "reset": (trace, "reset", None),
        "empty_steps": (trace, "steps", []),
        "last_action": (action, "selected_action", None),
        "empty_last_action": (action, "selected_action", {}),
        "malformed_last_action": (action, "selected_action", "not-an-action"),
        "malformed_last_planner": (trace["steps"][-1], "planner", "not-a-planner"),
        "controller_flag": (row, "controller_executed", "true"),
    }[fault]
    if replacement is None:
        target.pop(key)
    else:
        target[key] = replacement
    assert producer._audit_status([producer._execution_evidence(row)]) == "fail"


@pytest.mark.parametrize("mode", ["adapter", "unknown", "fallback", "degraded"])
@pytest.mark.parametrize("location", ["legacy", "adapter"])
def test_real_row_rejects_secondary_mode_contradictions(location, mode):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    metadata = row["algorithm_metadata"]
    target = metadata if location == "legacy" else metadata.setdefault("adapter_impact", {})
    target["execution_mode"] = mode
    evidence = producer._execution_evidence(row)
    assert evidence["execution_mode"] == "unknown"
    assert producer._audit_status([evidence]) == "fail"


def test_real_row_rejects_unsupported_primary_command_space():
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = "adapter"
    evidence = producer._execution_evidence(row)
    assert evidence["execution_mode"] == "unknown"
    assert producer._audit_status([evidence]) == "fail"


@pytest.mark.parametrize(
    "marker",
    ["fallback_used", "degraded", "degraded_reason", "degraded_statuses", "decision_label"],
)
def test_real_row_rejects_typed_runtime_fault_reports(marker):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    value = {
        "fallback_used": True,
        "degraded": True,
        "degraded_reason": "controller_failure",
        "degraded_statuses": ["degraded"],
        "decision_label": "fallback",
    }[marker]
    row["algorithm_metadata"].setdefault("planner_runtime", {})[marker] = value
    evidence = producer._execution_evidence(row)
    assert evidence["degraded"] is True
    assert producer._audit_status([evidence]) == "degraded"


@pytest.mark.parametrize("counter", [False, None, -1, float("inf"), {}])
def test_real_row_rejects_malformed_guarded_native_counters(counter):
    row = _executed_goal_with_optional_metrics()
    row["algo"] = "guarded_ppo"
    metadata = row["algorithm_metadata"]
    metadata.update(algorithm="ppo", canonical_algorithm="guarded_ppo")
    metadata["planner_contract"]["planner_id"] = "guarded_ppo"
    metadata["guard_stats"] = {"fallback_safe": 2}
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    metadata["guard_stats"]["fallback_safe"] = counter
    evidence = producer._execution_evidence(row)
    assert evidence["degraded"] is True
    assert producer._audit_status([evidence]) == "degraded"


@pytest.mark.parametrize("status", ["unavailable", "fallback", "degraded"])
@pytest.mark.parametrize(
    "path",
    [
        ("metrics", "social_compliance"),
        ("metrics", "distributional_disruption"),
        ("algorithm_metadata", "simulation_step_trace", "reset", "routes"),
        ("algorithm_metadata", "simulation_step_trace", "reset", "spawn"),
        ("algorithm_metadata", "paired_effect_metric_producer"),
        ("algorithm_metadata", "controller", "optional_metric"),
        ("non_runtime_provenance",),
    ],
)
def test_real_row_non_runtime_markers_cannot_taint_execution(path, status):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _runtime_object_for_fault(row, path).update(status=status, fallback=True, degraded=True)
    original = copy.deepcopy(row)
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    assert row == original


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
