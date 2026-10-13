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
from tests.validation.receipt_scheduler_fixture import install_scheduler


@pytest.fixture(autouse=True)
def _scheduler_authority(tmp_path, monkeypatch):
    install_scheduler(tmp_path, monkeypatch)


REVIEWED_HEAD = "e7fd36f98fab243ce95cf45bc50ca8a9f7c30893"
ORIGINAL_MAIN = "de3e37774049b2c6d017116426ae522dc232f679"
RELEASE_TAG = "paper-matrix-v2-h600-s30-synthetic"
REAL_ROWS = Path(__file__).parents[1] / "analysis/fixtures/behaviour_receipt_baseline_31302"
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
        "scheduler_job_id": "123",
        "scheduler_node": "auxme-imech142",
        "algo": "goal",
        "scenario_id": "open",
        "seed": seed,
        "steps": 1,
        "metrics": {
            "success": not (failing and seed == 1001),
            "collisions": 0,
            "total_collision_count": 0,
        },
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


def _sweep_rows(root, label, *, source_sha="a" * 40, failing=False, job="123"):
    out = root / label / job
    out.mkdir(parents=True)
    raw_path = out / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_rows = [_raw_row(seed, failing=failing) for seed in range(1001, 1031)]
    for row in raw_rows:
        row["scheduler_job_id"] = job
        row["git_hash"] = source_sha
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    rows = [
        {
            "scheduler_job_id": job,
            "scheduler_node": "auxme-imech142",
            "arm": "goal",
            "scenario": "open",
            "seed": row["seed"],
            "success": row["metrics"]["success"],
            "collisions": 0,
            "steps": row["steps"],
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
    from tests.validation.test_behaviour_receipt_gate import _hardware

    for name, entry in _hardware(source_sha, job).items():
        (out / name).write_text(entry["content"])
    request_path = out / "execution_request.json"
    request = json.loads(request_path.read_text())
    request["arm_role"] = "baseline" if label == "baseline" else "head"
    request_path.write_text(json.dumps(request))
    (out / "README.md").write_text(label + "\n")
    from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding

    refresh_sweep_binding(out)
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
    # Keep hardware valid for the claimed source to isolate the execution-metadata guard.
    request = out / "execution_request.json"
    hardware_request = json.loads(request.read_text())
    hardware_request["source_sha"] = claimed_sha
    request.write_text(json.dumps(hardware_request))
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


@pytest.mark.parametrize("identity", ["missing", "null"])
def test_missing_algorithm_is_invalid_execution_evidence(identity):
    row = _executed_goal_with_optional_metrics()
    row.pop("algo", None)
    row.pop("algorithm", None)
    if identity == "null":
        row.update(algo=None, algorithm=None)
    original = copy.deepcopy(row)

    evidence = producer._execution_evidence(row)

    assert evidence == {
        "algorithm": "unknown",
        "controller_executed": False,
        "execution_mode": "native",
        "fallback": False,
        "degraded": True,
    }
    assert producer._audit_status([evidence]) == "fail"
    assert row == original


def test_algorithm_alias_preserves_valid_execution_evidence():
    row = _executed_goal_with_optional_metrics()
    row["algorithm"] = row.pop("algo")

    evidence = producer._execution_evidence(row)

    assert evidence["algorithm"] == "goal"
    assert evidence["controller_executed"] is True
    assert evidence["degraded"] is False
    assert producer._audit_status([evidence]) == "pass"


def _executed_orca_row():
    return {
        **json.loads((REAL_ROWS / "orca__differential_drive.jsonl").read_bytes()),
        "execution_status": "written",
    }


def _attach_real_controller_fault_export(row, metadata):
    from types import SimpleNamespace

    from robot_sf.benchmark.ped_model_sensitivity import (
        attach_pedestrian_model_fields,
        build_pedestrian_model_provenance,
    )

    report = build_pedestrian_model_provenance(
        sim_config=SimpleNamespace(pedestrian_model="social_force_default"),
        policy_cfg={},
        algorithm_metadata=metadata,
    )
    attach_pedestrian_model_fields(row, report)
    return report


@pytest.mark.parametrize(
    "status,consistent",
    [("fallback", False), ("degraded", False), ("failed", False), ("failed", True)],
)
def test_real_writer_robot_fault_export_rejects_faults(status, consistent):
    row = _executed_orca_row()
    metadata = row["algorithm_metadata"]
    clean = _attach_real_controller_fault_export(row, metadata)
    assert clean["fallback_degraded_status"] == "native"
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    faulty = copy.deepcopy(metadata)
    faulty["planner_runtime"]["status"] = status
    if consistent:
        row["algorithm_metadata"] = faulty
    report = _attach_real_controller_fault_export(row, faulty)
    assert report["fallback_degraded_status"] == status
    evidence = producer._execution_evidence(row)
    assert evidence["controller_executed"] is True
    assert producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"]) == (
        "pedestrian_model.fallback_degraded_status",
        status,
    )
    assert producer._audit_status([evidence]) == "degraded"


@pytest.mark.parametrize("status", ["failed", "error", "not_available", "unavailable"])
def test_real_row_planner_runtime_fault_status_rejects_without_export(status):
    row = _executed_orca_row()
    assert row["pedestrian_model"]["fallback_degraded_status"] == "native"
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["algorithm_metadata"]["planner_runtime"]["status"] = status
    assert producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"]) == (
        "algorithm_metadata.planner_runtime.status",
        status,
    )
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


@pytest.mark.parametrize("status", ["native", None])
def test_real_row_robot_fault_export_exact_clean_vocabulary(status):
    row = _executed_orca_row()
    row["pedestrian_model"]["fallback_degraded_status"] = status
    original = copy.deepcopy(row)
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    assert row == original


@pytest.mark.parametrize(
    "status",
    [
        "fallback",
        "degraded",
        "not_available",
        "unavailable",
        "failed",
        "error",
        "bad",
        "NATIVE",
        " native ",
        "null",
        "",
        False,
        True,
        0,
        1,
        {},
        [],
        ["native"],
    ],
)
def test_real_row_robot_fault_export_rejects_malformed_or_fault_values(status):
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    row["pedestrian_model"]["fallback_degraded_status"] = status
    evidence = producer._execution_evidence(row)
    assert evidence["controller_executed"] is True
    marker = producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"])
    assert marker is not None
    assert marker[0] == "pedestrian_model.fallback_degraded_status"
    assert producer._audit_status([evidence]) == "degraded"


@pytest.mark.parametrize("status", ["failed", "error", "not_available", "unavailable"])
@pytest.mark.parametrize(
    "field",
    [
        "status",
        "row_status",
        "readiness_status",
        "availability_status",
        "execution_mode",
        "decision_label",
        "runtime_status",
        "last_step_status",
        "observation_validation_status",
        "load_status",
        "qp_status",
    ],
)
def test_all_named_runtime_status_locations_reject_fault_vocabulary(field, status):
    for path in producer.EXECUTION_EVIDENCE_LOCATIONS["runtime_objects"]:
        name = ".".join(path)
        if f"{name}.{field}" in producer.EXECUTION_EVIDENCE_LOCATIONS["non_runtime_locations"]:
            continue  # Exact optional availability paths are not runtime statuses.
        row = _executed_orca_row()
        if "fallback_controller_state" in path:
            row["algo"] = "guarded_ppo"
            row["algorithm_metadata"].update(algorithm="ppo", canonical_algorithm="guarded_ppo")
            row["algorithm_metadata"]["planner_contract"]["planner_id"] = "guarded_ppo"
        if "*" in path and path[:2] == ("algorithm_metadata", "cbf_safety_filter"):
            row["algorithm_metadata"]["cbf_safety_filter"] = {"steps": [{}]}
        report = _runtime_object_for_fault(row, path)
        assert producer._audit_status([producer._execution_evidence(row)]) == "pass", path
        report[field] = status
        evidence = producer._execution_evidence(row)
        assert evidence["degraded"] is True, (path, field, status)
        assert producer._audit_status([evidence]) in {"degraded", "fail"}, (path, field, status)


@pytest.mark.parametrize(
    "mode,label",
    [
        ("fallback", "uncertainty_fallback_configured"),
        ("stop", "uncertainty_fallback_stop"),
        ("slow_down", "uncertainty_fallback_slow_down"),
    ],
)
@pytest.mark.parametrize("transport", ["guard_stats", "algorithm_metadata.guard_stats"])
@pytest.mark.parametrize("value,marker", [(1, "1"), ("bad", "invalid")])
def test_real_row_executed_uncertainty_guard_decisions_are_runtime_counters(
    mode, label, transport, value, marker
):
    from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, GuardedPPOConfig

    class CountingFallback:
        calls = 0

        def plan(self, observation):
            self.calls += 1
            return (0.1, 0.2)

    fallback = CountingFallback()
    guard = GuardedPPOAdapter(
        config=GuardedPPOConfig(uncertainty_fallback_mode=mode), fallback_adapter=fallback
    )
    command, decision, _ = guard._uncertainty_fallback_command({}, (0.8, 0.2))
    assert decision == label
    assert fallback.calls == (1 if mode == "fallback" else 0)
    assert command == (
        (0.1, 0.2)
        if mode == "fallback"
        else (0.0, 0.0)
        if mode == "stop"
        else (guard.config.uncertainty_slow_down_speed_m_s, 0.2)
    )
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    counter = _runtime_object_for_fault(row, tuple(transport.split(".")))
    counter[decision] = 0
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    counter[decision] = value
    actual = producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"])
    assert actual == (f"{transport}.{decision}", marker)
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


def test_real_row_controller_observation_integrity_uses_actual_transport():
    from types import SimpleNamespace

    from robot_sf.benchmark.map_runner.map_runner_view_integrity import (
        evaluate_effective_view_integrity,
    )

    class Extractor:
        def __init__(self, count):
            self.count = count

        def _socnav_fields(self, observation):
            return (
                {},
                {},
                {"positions": [[1.0, 0.0]] if self.count else None, "count": [self.count]},
            )

    row = _executed_orca_row()
    for count, expected in [(1, "pass"), (0, "degraded")]:
        report = evaluate_effective_view_integrity(
            policy_fn=SimpleNamespace(_planner_adapter=Extractor(count)),
            observation={"pedestrians": {"count": [1], "positions": [[1.0, 0.0]]}},
            algo_meta=row["algorithm_metadata"],
        ).to_metadata()
        assert report["probed"] is True
        assert report["degraded"] is (count == 0)
        row["integrity"]["effective_view"] = report
        assert producer._audit_status([producer._execution_evidence(row)]) == expected
    assert report["degraded_reason"] == "degenerate_planner_view"


@pytest.mark.parametrize("earlier_fallback", [True, False])
def test_real_row_topology_cumulative_actual_use_survives_clean_last_decision(earlier_fallback):
    from robot_sf.benchmark.map_runner.map_runner_episode import (
        _topology_guided_episode_diagnostics,
    )

    steps = [
        {
            "topology_guided": {"status": "insufficient_hypotheses", "hypothesis_count": 1},
            "topology_lane_status": "fallback_only",
            "topology_fallback_reason": "missing_candidate_route_geometry",
            "topology_candidate_availability": {
                "status": "unavailable",
                "outcome": "configured_fallback" if earlier_fallback else "candidate_unavailable",
                "fallback_used": earlier_fallback,
            },
        },
        {
            "topology_guided": {"status": "ok", "hypothesis_count": 2},
            "topology_candidate_availability": {"status": "available", "fallback_used": False},
        },
    ]
    summary = _topology_guided_episode_diagnostics(steps)
    assert summary["configured_fallback_steps"] == int(earlier_fallback)
    # Both arms have an unavailable optional lane; only one executed a substitute.
    assert summary["fallback_used"] is True
    assert summary["fallback_steps"] == 1
    row = _executed_orca_row()
    metadata = row["algorithm_metadata"]
    metadata["topology_guided_episode"] = summary
    metadata["planner_runtime"]["last_decision"] = {
        "topology_candidate_availability": steps[-1]["topology_candidate_availability"]
    }
    assert producer._audit_status([producer._execution_evidence(row)]) == (
        "degraded" if earlier_fallback else "pass"
    )
    if earlier_fallback:
        assert producer._controller_runtime_marker(row, metadata, row["algo"]) == (
            "algorithm_metadata.topology_guided_episode.configured_fallback_steps",
            "1",
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("future_used_fallback", True),
        ("future_degraded_count", 1),
        ("fallback_used", False),
        ("uncertainty_fallback_enabled", False),
        ("future_status", "ok"),
        ("qp_status", "fallback_infeasible"),
    ],
)
def test_real_row_unknown_fault_transports_fail_closed_even_for_catalogued_keys(key, value):
    row = _executed_orca_row()
    row["algorithm_metadata"]["new_controller_report"] = {key: value}
    expected = f"algorithm_metadata.new_controller_report.{key}"
    assert producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"]) == (
        expected,
        "unclassified_degraded",
    )
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


def test_non_runtime_exemptions_are_exact_source_cited_paths_not_subtrees():
    locations = producer.EXECUTION_EVIDENCE_LOCATIONS["non_runtime_locations"]
    root = Path(__file__).resolve().parents[2]
    for path, reason in locations.items():
        assert not path.endswith(".*")
        source, line = reason.split(" ", 1)[0].rsplit(":", 1)
        assert (root / source).is_file()
        assert 0 < int(line) <= len((root / source).read_text().splitlines())
    row = _executed_orca_row()
    producer._execution_evidence(row)
    # An exempt status does not exempt a new field in that same real object.
    row["algorithm_metadata"]["paired_effect_metric_producer"]["new_degraded_flag"] = True
    assert producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"]) == (
        "algorithm_metadata.paired_effect_metric_producer.new_degraded_flag",
        "unclassified_degraded",
    )


@pytest.mark.parametrize("collision", ["dotted_key", "literal_wildcard"])
def test_real_row_non_runtime_path_exemptions_cannot_be_spoofed_by_dictionary_keys(collision):
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    if collision == "dotted_key":
        row["algorithm_metadata.config.fallback_to_goal"] = True
    else:
        row["metrics"]["social_mini_game"]["rows"] = {"*": {"status": "degraded"}}
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


@pytest.mark.parametrize(
    "path,key,value",
    [
        (
            ("algorithm_metadata", "cbf_safety_filter", "last_step"),
            "qp_status",
            "fallback_infeasible",
        ),
        (
            ("algorithm_metadata", "planner_runtime", "checkpoint_provenance"),
            "load_status",
            "fallback",
        ),
        (("algorithm_metadata", "foresight_prediction"), "load_status", "failed"),
    ],
)
def test_real_row_writer_statuses_cannot_lose_actual_runtime_faults(path, key, value):
    row = _executed_orca_row()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _runtime_object_for_fault(row, path)[key] = value
    assert producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"]) == (
        ".".join((*path, key)),
        "degraded",
    )
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


@pytest.mark.parametrize(
    "path,key",
    [
        (("integrity", "effective_view"), "degraded"),
        (("integrity", "effective_view"), "degraded_reason"),
        (("algorithm_metadata", "topology_guided_episode"), "configured_fallback_steps"),
    ],
)
def test_real_row_new_runtime_transports_require_complete_reports(path, key):
    row = _executed_orca_row()
    if path[-1] == "topology_guided_episode":
        row["algorithm_metadata"]["topology_guided_episode"] = {"configured_fallback_steps": 0}
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    del _runtime_object_for_fault(row, path)[key]
    assert producer._controller_runtime_marker(row, row["algorithm_metadata"], row["algo"]) == (
        ".".join((*path, key)),
        "invalid",
    )
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"


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


_EMITTER_KEY = re.compile(producer.EXECUTION_EVIDENCE_LOCATIONS["fault_key_pattern"], re.IGNORECASE)


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
    non_runtime = set(locations.get("non_runtime_writers", {}))
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
    # This newly injected key has no writer-backed transport exemption. The
    # stronger default rejects it; derived status alone remains non-runtime.
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"
    del metrics["fields"]["false_positive_stop_rate"]["fallback_used"]
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
def test_real_row_unclassified_fault_fields_fail_closed_even_in_optional_objects(path, status):
    row = _executed_goal_with_optional_metrics()
    assert producer._audit_status([producer._execution_evidence(row)]) == "pass"
    _runtime_object_for_fault(row, path).update(status=status, fallback=True, degraded=True)
    original = copy.deepcopy(row)
    assert producer._audit_status([producer._execution_evidence(row)]) == "degraded"
    assert row == original


def _payload_fixture(tmp_path, monkeypatch, *, legacy_projection=False):
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
    if legacy_projection:
        # Historical policy controls retain their historical wire schema; the
        # optional native-slot presentation field is unrelated to those defects.
        for row in [*rows, *classes]:
            row.pop("native_slot", None)
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
    git("add", "receipts/behaviour")
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
        ["git", "show", f"{revision}:scripts/ci/{filename}"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )
    path = tmp_path / filename
    path.write_text(source)
    schema = subprocess.check_output(
        ["git", "show", f"{revision}:scripts/ci/behaviour_receipt.schema.json"],
        cwd=Path(__file__).resolve().parents[2],
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
            old_gate = _revision_module(tmp_path, REVIEWED_HEAD, "behaviour_receipt.py")
            monkeypatch.setattr(sys.modules[__name__], "gate", old_gate)
            header = _payload_fixture(fixture_dir, monkeypatch, legacy_projection=True)
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
        "behaviour_receipt_nodes.json",
        "receipt_provenance.py",
        "receipt_outcomes.py",
        "receipt_arm_execution.py",
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
    from scripts.validation import run_empty_world_sweep as native

    # The isolated CLI must carry the same reviewed configuration authority as
    # acquisition, rather than inventing a test-only kinematics default.
    for template in native.SUITES.values():
        destination = tmp_path / template
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(producer.ROOT / template, destination)
    scope_path = tmp_path / gate.SCOPE_FILE
    scope_path.parent.mkdir(parents=True, exist_ok=True)
    scope_path.write_text(json.dumps(_scope()))
    git("add", "scripts", "configs", gate.SCOPE_FILE)
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
    baseline = _sweep_rows(tmp_path, "baseline", source_sha=baseline_source, job="456")
    request = baseline / "execution_request.json"
    metadata = json.loads(request.read_text())
    metadata["job_id"] = "456"
    request.write_text(json.dumps(metadata))
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
    git("add", "receipts/behaviour")
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
    legacy_payload = json.loads((tmp_path / header["rows_artifact"]["path"]).read_text())
    legacy_payload.pop("hardware")
    legacy_payload.pop("source_artifacts")
    for row in [*legacy_payload["rows"], *legacy_payload["classifications"]]:
        row.pop("native_slot", None)
    for row in legacy_payload["rows"]:
        for key in (
            "scheduler_job_id",
            "scheduler_node",
            "baseline_scheduler_job_id",
            "baseline_scheduler_node",
        ):
            row.pop(key)
    legacy_raw = json.dumps(legacy_payload).encode()
    (tmp_path / header["rows_artifact"]["path"]).write_bytes(legacy_raw)
    git("add", header["rows_artifact"]["path"])
    git("commit", "-m", "historical transport control")
    legacy_head = git("rev-parse", "HEAD")
    legacy_header = copy.deepcopy(header)
    legacy_header["rows_artifact"]["sha256"] = hashlib.sha256(legacy_raw).hexdigest()
    legacy_header["classifications"] = {
        "count": len(legacy_payload["classifications"]),
        "sha256": hashlib.sha256(
            json.dumps(
                legacy_payload["classifications"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    legacy_header["head_sha"] = legacy_head
    legacy_header["refute_review"]["head_sha"] = legacy_head
    receipt = original.load_receipt(legacy_header, legacy_head)
    with pytest.raises(ValueError, match="baseline is not the latest published software release"):
        original.validate_receipt(
            receipt, _scope(), legacy_head, "0.0.7", original.release_source("0.0.7")
        )
    assert original.check_receipt(body, changed, "ll7/robot_sf_ll7")


def test_seed_guard_names_wider_dev_range_but_keeps_current_gate_contract():
    assert producer._check_seed_range([1001, 1030]) == [1001, 1030]
    with pytest.raises(ValueError, match="current behaviour gate admits only seeds"):
        producer._check_seed_range([1031])
    with pytest.raises(ValueError, match="outside development range"):
        producer._check_seed_range([1201])


def test_raw_trace_file_is_decoded_once_per_invocation(tmp_path, monkeypatch):
    """Receipt assembly scales by raw bytes, rather than rereading per summary row."""
    out = _sweep_rows(tmp_path, "head")
    original = Path.read_text
    reads = []

    def track_reads(path, *args, **kwargs):
        if path.name == "episodes.jsonl":
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", track_reads)
    rows = producer._load_sweep_rows(out)
    assert len(rows) == 30
    assert rows[("goal", "differential_drive", "open", 1001)]["metrics"]["success"] is True
    assert len(reads) == 1


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("missing_raw", "raw episode identity differs"),
        ("duplicate_raw", "raw episode identity differs"),
        ("wrong_arm", "raw episode identity differs"),
        ("success", "raw episode outcome differs"),
        ("collision", "raw episode outcome differs"),
        ("duplicate_summary", "duplicate sweep row"),
        ("escape", "raw controller trace path escapes"),
        ("missing_source", "missing raw controller trace source"),
    ],
)
def test_raw_index_preserves_planted_regression_detection(tmp_path, fault, reason):
    """Reject tampering even after a previous successful read of the same files."""
    out = _sweep_rows(tmp_path, "head")
    assert len(producer._load_sweep_rows(out)) == 30
    summary_path = out / "episodes_main.jsonl"
    raw_path = out / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl"
    summary = [json.loads(line) for line in summary_path.read_text().splitlines()]
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    # Plant on the last slot: earlier successful rows must not hide a later fault.
    if fault == "missing_raw":
        raw.pop()
    elif fault == "duplicate_raw":
        raw.append(copy.deepcopy(raw[-1]))
    elif fault == "wrong_arm":
        wrong_path = raw_path.parent.with_name("orca__differential_drive") / "episodes.jsonl"
        wrong_path.parent.mkdir()
        wrong_path.write_text(raw_path.read_text())
        summary[-1]["source_file"] = "runs/orca__differential_drive/episodes.jsonl"
    elif fault == "success":
        raw[-1]["metrics"]["success"] = False
    elif fault == "collision":
        raw[-1]["metrics"]["collisions"] = 1
        raw[-1]["metrics"]["total_collision_count"] = 1
    elif fault == "duplicate_summary":
        summary.append(copy.deepcopy(summary[-1]))
    elif fault == "escape":
        summary[-1]["source_file"] = "../outside.jsonl"
    elif fault == "missing_source":
        summary[-1].pop("source_file")
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw))
    summary_path.write_text("".join(json.dumps(row) + "\n" for row in summary))
    with pytest.raises(ValueError, match=reason):
        producer._load_sweep_rows(out)
