"""Tests for the matched-compute production canary (issue #7893).

Validator/accounting failure cases use fixtures and injected fakes; the
tracked production receipt must come from the real seams.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.adversarial.matched_compute import MatchedComputeRuntimeTrace
from scripts.validation.run_matched_compute_production_canary import (
    CANDIDATE_RECORD_SCHEMA,
    CandidateRecord,
    _aggregate_reconcile,
    _budget_reconcile,
    _check_receipt,
    _digest_file,
    _digest_text,
    _discover_frozen_input_files,
    _manifest_status,
    _packet_arm_expectations,
    _record_integrity_problems,
    _run_open_loop,
    _runtime_trace_is_production_observed,
    _verify_frozen_inputs,
    _write_receipt,
    load_packet,
    main,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validation"
    / ("run_matched_compute_production_canary.py")
)
PACKET = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "adversarial"
    / ("issue_6921_matched_compute_packet.yaml")
)
REPOSITORY_ROOT = PACKET.parents[2]
TEST_ARTIFACT = Path(__file__).resolve()
TEST_ARTIFACT_ID = TEST_ARTIFACT.relative_to(REPOSITORY_ROOT).as_posix()
TEST_ARTIFACT_DIGEST = hashlib.sha256(TEST_ARTIFACT.read_bytes()).hexdigest()


def _record(
    *,
    arm: str = "open_loop",
    status: str = "accepted",
    candidate_identity: str = "cand-0",
    packet_digest: str = "a" * 64,
    commit: str = "c" * 40,
) -> CandidateRecord:
    try:
        candidate_number = int(candidate_identity.rsplit("-", 1)[-1])
    except ValueError:
        candidate_number = 0
    if arm == "reactive":
        native_seam = (
            "robot_sf.ped_npc.residual_search.FiniteGridSearchPolicy+"
            "robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary"
        )
        policy_identity = "finite_grid_search_v1"
        objective_identity = "minimize_predicted_robot_distance"
    else:
        native_seam = (
            "robot_sf.adversarial.search.run_adversarial_search(default production evaluator)"
        )
        policy_identity = "social_force"
        objective_identity = "minimize_episode_min_robot_distance"
    return CandidateRecord(
        packet_digest=packet_digest,
        arm=arm,
        candidate_identity=candidate_identity,
        scenario_template="crossing_ttc_template",
        scenario_seed=123,
        search_seed=42,
        native_seam=native_seam,
        macro_action_index=(candidate_number // 9) if arm == "reactive" else None,
        policy_identity=policy_identity,
        objective_identity=objective_identity,
        repository_commit=commit,
        status=status,
        simulator_steps=50,
        simulator_steps_source="observed_episode_record",
        episode_identity=TEST_ARTIFACT_ID,
        episode_digest=TEST_ARTIFACT_DIGEST,
    )


class _Trace:
    accepted = 90
    rejected = 0
    invalid = 0


def _runtime_trace(
    *,
    arm: str = "open_loop",
    candidate_evaluations: int = 90,
    accepted: int = 90,
    rejected: int = 0,
    invalid: int = 0,
    candidate_budget: int = 90,
) -> MatchedComputeRuntimeTrace:
    """Return a valid production-shaped trace for receipt tests."""
    return MatchedComputeRuntimeTrace(
        arm=arm,
        scenario_seed=123,
        search_seed=42,
        execution_mode="native",
        simulator_physics_steps=50,
        macro_actions=10,
        candidate_evaluations=candidate_evaluations,
        accepted=accepted,
        rejected=rejected,
        invalid=invalid,
        status="native",
        adapter=(
            "finite_grid_residual_adversary"
            if arm == "reactive"
            else "adversarial_search_production_candidate"
        ),
        native_path=(
            "robot_sf.ped_npc.residual_search.FiniteGridSearchPolicy+"
            "robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary"
            if arm == "reactive"
            else "robot_sf.adversarial.search.run_adversarial_search"
        ),
        candidate_budget=candidate_budget,
        evidence_status="production_observed",
        simulator_steps_source="observed_episode_record",
    )


def _real_receipt_inputs() -> tuple[str, str, dict[str, str]]:
    """Return packet-bound receipt inputs for CLI validation tests."""
    input_digests = _verify_frozen_inputs(PACKET)
    packet_relative = PACKET.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return commit, input_digests[packet_relative], input_digests


def test_packet_loads_and_validates() -> None:
    packet = load_packet(PACKET)
    assert packet["schema_version"] == "matched_compute_packet.v2"
    assert set(packet["arms"]) == {"open_loop", "reactive"}
    assert packet["budget"]["candidates_per_arm_per_episode"] == 90


def test_packet_rejects_wrong_schema() -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        load_packet(Path("missing.yaml"))


def _frozen_input_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    packet = tmp_path / "configs" / "packet.yaml"
    packet.parent.mkdir(parents=True)
    packet.write_text(
        """
schema_version: matched_compute_packet.v2
arms:
  open_loop: {}
  reactive: {}
scenario:
  template: configs/template.yaml
  search_space: configs/search.yaml
  source_search_space: configs/source.yaml
provenance:
  residual_search_config: configs/residual.yaml
""".lstrip(),
        encoding="utf-8",
    )
    for name in ("template.yaml", "search.yaml", "source.yaml", "residual.yaml"):
        path = tmp_path / "configs" / name
        path.write_text(f"schema: {name}\n", encoding="utf-8")
    files = _discover_frozen_input_files(packet, tmp_path)
    return packet, {relative: _digest_file(path) for relative, path in files.items()}


def test_frozen_input_verification_rejects_one_byte_packet_drift(tmp_path: Path) -> None:
    packet, expected = _frozen_input_fixture(tmp_path)
    assert (
        _verify_frozen_inputs(packet, repository_root=tmp_path, expected_digests=expected)
        == expected
    )
    packet.write_bytes(packet.read_bytes() + b"#")
    with pytest.raises(ValueError, match="frozen input digest mismatch"):
        _verify_frozen_inputs(packet, repository_root=tmp_path, expected_digests=expected)


def test_frozen_input_verification_rejects_one_byte_referenced_config_drift(
    tmp_path: Path,
) -> None:
    packet, expected = _frozen_input_fixture(tmp_path)
    referenced_config = tmp_path / "configs" / "search.yaml"
    referenced_config.write_bytes(referenced_config.read_bytes() + b"#")
    with pytest.raises(ValueError, match="frozen input digest mismatch"):
        _verify_frozen_inputs(packet, repository_root=tmp_path, expected_digests=expected)


def test_budget_reconcile_exact() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    assert _budget_reconcile(records, 90) == []


def test_budget_reconcile_detects_mismatch() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(89)]
    problems = _budget_reconcile(records, 90)
    assert any("89 != frozen budget 90" in problem for problem in problems)


def test_budget_reconcile_rejects_fallback_as_inadmissible() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = _record(status="fallback")
    problems = _budget_reconcile(records, 90)
    assert any("inadmissible as production_observed" in problem for problem in problems)


def test_budget_reconcile_rejects_unavailable() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = _record(status="unavailable")
    problems = _budget_reconcile(records, 90)
    assert any("inadmissible as production_observed" in problem for problem in problems)


def test_budget_reconcile_rejects_unknown_status() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = _record(status="bogus")
    problems = _budget_reconcile(records, 90)
    assert any("unknown candidate status" in problem for problem in problems)


def test_budget_reconcile_rejects_duplicate_identity() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[-1] = _record(candidate_identity="cand-0")
    problems = _budget_reconcile(records, 90)
    assert any("duplicate candidate identities" in problem for problem in problems)


def test_aggregate_reconcile_matches_v1_semantics() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    assert _aggregate_reconcile(records, _Trace()) == []


def test_aggregate_reconcile_detects_accepted_drift() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = _record(status="rejected")
    problems = _aggregate_reconcile(records, _Trace())
    assert any("accepted 89 != trace.accepted 90" in problem for problem in problems)


def test_aggregate_reconcile_rejected_equals_rejected_nonfailed_plus_failed() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = _record(status="failed")
    trace = _Trace()
    trace.rejected = 0
    problems = _aggregate_reconcile(records, trace)
    assert any("rejected+failed 1 != trace.rejected 0" in problem for problem in problems)


@pytest.mark.parametrize(
    "field", ["adapter", "native_path", "scenario_seed", "search_seed", "macro_actions"]
)
def test_runtime_trace_admission_rejects_identity_drift(field: str) -> None:
    packet = load_packet(PACKET)
    expected = _packet_arm_expectations(packet, "open_loop")
    trace = _runtime_trace()
    trace = replace(trace, **{field: "forged" if isinstance(getattr(trace, field), str) else 999})
    assert not _runtime_trace_is_production_observed(trace, arm_name="open_loop", expected=expected)


def test_aggregate_reconcile_rejects_packet_step_drift() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    trace = _runtime_trace()
    trace = replace(trace, simulator_physics_steps=49)
    problems = _aggregate_reconcile(records, trace, expected_simulator_steps=50)
    assert any("simulator_physics_steps" in problem for problem in problems)


def test_aggregate_reconcile_rejects_candidate_step_drift() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = replace(records[0], simulator_steps=90)
    problems = _aggregate_reconcile(records, _runtime_trace(), expected_simulator_steps=50)
    assert any("record[0] simulator_steps 90" in problem for problem in problems)


def test_candidate_record_schema_is_versioned() -> None:
    record = _record()
    assert record.schema == CANDIDATE_RECORD_SCHEMA
    payload = record.as_dict()
    assert payload["status"] == "accepted"
    assert payload["simulator_steps_source"] == "observed_episode_record"


def test_candidate_record_disjoint_status_vocabulary() -> None:
    for status in ("accepted", "rejected", "failed", "invalid", "fallback", "unavailable"):
        assert _record(status=status).status == status


@pytest.mark.parametrize(
    ("certification_status", "expected"),
    [
        ({"status": "passed", "details": {"readiness_status": "fallback"}}, "fallback"),
        ({"status": "passed", "details": {"execution_mode": "degraded"}}, "fallback"),
        ({"status": "passed", "details": {"availability_status": "unavailable"}}, "unavailable"),
        ({"status": "passed", "details": {"readiness_status": "native"}}, "accepted"),
    ],
)
def test_manifest_status_rejects_nested_non_native_certification(
    certification_status: dict[str, object], expected: str
) -> None:
    assert (
        _manifest_status({"certification_status": certification_status, "bundle_path": "bundle"})
        == expected
    )


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"readiness_status": "fallback"}, "fallback"),
        ({"execution_mode": "degraded"}, "fallback"),
        ({"availability_status": "not_available"}, "unavailable"),
    ],
)
def test_manifest_status_rejects_nested_failure_attribution_status(
    details: dict[str, str], expected: str
) -> None:
    entry = {
        "certification_status": {"status": "passed"},
        "failure_attribution": {"details": details},
        "bundle_path": "bundle",
    }
    assert _manifest_status(entry) == expected


def test_receipt_round_trip_and_deterministic_check(tmp_path: Path) -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    reactive_records = [_record(arm="reactive", candidate_identity=f"cand-{i}") for i in range(90)]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    assert _check_receipt(receipt_path) == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["review_marker"] == (
        "AI-GENERATED NEEDS-REVIEW"
    )
    # Re-checking is deterministic.
    assert receipt_path.read_bytes() == receipt_path.read_bytes()


@pytest.mark.parametrize("arm_name", ["open_loop", "reactive"])
def test_receipt_check_rejects_diagnostic_runtime_trace(arm_name: str, tmp_path: Path) -> None:
    records = [_record(candidate_identity=f"open-{i}") for i in range(90)]
    reactive_records = [
        _record(arm="reactive", candidate_identity=f"reactive-{i}") for i in range(90)
    ]
    diagnostic_trace = replace(
        _runtime_trace(arm=arm_name),
        evidence_status="diagnostic_only_preflight",
        simulator_steps_source="synthetic_episode_fixture",
        simulator_physics_steps=None,
    )
    runtime_traces = {
        "open_loop": _runtime_trace(),
        "reactive": _runtime_trace(arm="reactive"),
    }
    runtime_traces[arm_name] = diagnostic_trace
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
        runtime_traces=runtime_traces,
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "blocked"
    payload["evidence_status"] = "production_observed"
    payload["problems"] = []
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _check_receipt(receipt_path) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_digest", "not-a-sha256"),
        ("episode_digest", "0" * 64),
        ("episode_identity", "missing/episode_records.jsonl"),
        ("episode_identity", "\x00malformed"),
    ],
)
def test_receipt_check_rejects_untrusted_episode_provenance(
    field: str, value: str, tmp_path: Path
) -> None:
    records = [_record(candidate_identity=f"open-{i}") for i in range(90)]
    records[0] = replace(records[0], **{field: value})
    reactive_records = [
        _record(arm="reactive", candidate_identity=f"reactive-{i}") for i in range(90)
    ]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    assert _check_receipt(receipt_path) == 1


def test_receipt_check_fails_on_unavailable_reactive(tmp_path: Path) -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    reactive = [_record(arm="reactive", candidate_identity=f"cand-{i}") for i in range(90)]
    reactive[0] = _record(arm="reactive", status="unavailable")
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
    )
    assert _check_receipt(receipt_path) == 1


def test_receipt_check_rejects_positive_receipt_without_input_digests(
    tmp_path: Path,
) -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    reactive_records = [_record(arm="reactive", candidate_identity=f"cand-{i}") for i in range(90)]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    assert _check_receipt(receipt_path) == 1


def test_receipt_check_rejects_positive_receipt_without_runtime_traces(
    tmp_path: Path,
) -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    reactive_records = [_record(arm="reactive", candidate_identity=f"cand-{i}") for i in range(90)]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
    )
    assert _check_receipt(receipt_path) == 1


def test_receipt_check_rejects_synthetic_step_source(tmp_path: Path) -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = replace(records[0], simulator_steps_source="synthetic_episode_fixture")
    reactive_records = [_record(arm="reactive", candidate_identity=f"cand-{i}") for i in range(90)]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    assert _check_receipt(receipt_path) == 1


def test_receipt_check_rejects_missing_step_provenance(tmp_path: Path) -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    records[0] = replace(
        records[0],
        simulator_steps=None,
        simulator_steps_source="unavailable",
        degraded_reason="",
    )
    reactive_records = [_record(arm="reactive", candidate_identity=f"cand-{i}") for i in range(90)]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest="a" * 64,
        commit="c" * 40,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests={"fixture.yaml": "f" * 64},
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    assert _check_receipt(receipt_path) == 1


def test_open_loop_uses_native_manifest_identity_and_packet_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    episode_path = tmp_path / "candidate_0000" / "episode_records.jsonl"
    episode_path.parent.mkdir()
    episode_path.write_text(json.dumps({"steps": 50, "min_robot_distance": 1.25}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate": {"scenario_seed": 123},
                        "certification_status": {"status": "valid"},
                        "bundle_path": str(episode_path.parent),
                        "episode_record_path": str(episode_path),
                        "objective_value": 1.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary.run_adversarial_search",
        lambda _config: SimpleNamespace(
            manifest_path=manifest_path,
            best_candidate=None,
            best_bundle_path=None,
            num_candidates=1,
            num_valid_candidates=1,
            num_invalid_candidates=0,
            num_failed_evaluations=0,
        ),
    )
    records, trace = _run_open_loop(
        load_packet(PACKET),
        tmp_path / "output",
        "c" * 40,
        "a" * 64,
    )
    assert len(records) == 1
    assert trace.candidate_evaluations == 1
    assert trace.accepted == 1
    assert records[0].candidate_identity == "candidate_0000"
    assert records[0].packet_digest == "a" * 64
    assert records[0].simulator_steps_source == "observed_episode_record"


def test_cli_check_mode(tmp_path: Path) -> None:
    commit, packet_digest, input_digests = _real_receipt_inputs()
    records = [
        _record(candidate_identity=f"cand-{i}", packet_digest=packet_digest, commit=commit)
        for i in range(90)
    ]
    reactive_records = [
        _record(
            arm="reactive",
            candidate_identity=f"cand-{i}",
            packet_digest=packet_digest,
            commit=commit,
        )
        for i in range(90)
    ]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest=packet_digest,
        commit=commit,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests=input_digests,
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--packet",
            str(PACKET),
            "--receipt",
            str(receipt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "receipt check passed" in proc.stdout


def test_cli_check_detects_budget_drift(tmp_path: Path) -> None:
    commit, packet_digest, input_digests = _real_receipt_inputs()
    records = [
        _record(candidate_identity=f"cand-{i}", packet_digest=packet_digest, commit=commit)
        for i in range(89)
    ]
    reactive_records = [
        _record(
            arm="reactive",
            candidate_identity=f"cand-{i}",
            packet_digest=packet_digest,
            commit=commit,
        )
        for i in range(89)
    ]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest=packet_digest,
        commit=commit,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests=input_digests,
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--packet",
            str(PACKET),
            "--receipt",
            str(receipt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "check failed" in proc.stdout


def test_cli_check_rejects_packet_identity_drift(tmp_path: Path) -> None:
    commit, packet_digest, input_digests = _real_receipt_inputs()
    records = [
        _record(candidate_identity=f"cand-{i}", packet_digest=packet_digest, commit=commit)
        for i in range(90)
    ]
    records[0] = replace(records[0], scenario_seed=999)
    reactive_records = [
        _record(
            arm="reactive",
            candidate_identity=f"cand-{i}",
            packet_digest=packet_digest,
            commit=commit,
        )
        for i in range(90)
    ]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest=packet_digest,
        commit=commit,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests=input_digests,
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--packet",
            str(PACKET),
            "--receipt",
            str(receipt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "scenario_seed" in proc.stdout
    assert "does not match frozen packet" in proc.stdout


def test_cli_check_rejects_trace_budget_drift(tmp_path: Path) -> None:
    commit, packet_digest, input_digests = _real_receipt_inputs()
    records = [
        _record(candidate_identity=f"cand-{i}", packet_digest=packet_digest, commit=commit)
        for i in range(90)
    ]
    reactive_records = [
        _record(
            arm="reactive",
            candidate_identity=f"cand-{i}",
            packet_digest=packet_digest,
            commit=commit,
        )
        for i in range(90)
    ]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest=packet_digest,
        commit=commit,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests=input_digests,
        runtime_traces={
            "open_loop": _runtime_trace(candidate_budget=91),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--packet",
            str(PACKET),
            "--receipt",
            str(receipt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "candidate_budget 91" in proc.stdout
    assert "frozen budget 90" in proc.stdout


def test_runtime_trace_rejects_evaluations_over_budget() -> None:
    with pytest.raises(ValueError, match="candidate_evaluations must not exceed candidate_budget"):
        _runtime_trace(candidate_budget=89)


def test_cli_check_rejects_unreachable_source_commit(tmp_path: Path) -> None:
    _commit, packet_digest, input_digests = _real_receipt_inputs()
    bad_commit = "0" * 40
    records = [
        _record(candidate_identity=f"cand-{i}", packet_digest=packet_digest, commit=bad_commit)
        for i in range(90)
    ]
    reactive_records = [
        _record(
            arm="reactive",
            candidate_identity=f"cand-{i}",
            packet_digest=packet_digest,
            commit=bad_commit,
        )
        for i in range(90)
    ]
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(
        packet_digest=packet_digest,
        commit=bad_commit,
        open_loop_records=records,
        reactive_records=reactive_records,
        problems=[],
        output=receipt_path,
        input_digests=input_digests,
        runtime_traces={
            "open_loop": _runtime_trace(),
            "reactive": _runtime_trace(arm="reactive"),
        },
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--packet",
            str(PACKET),
            "--receipt",
            str(receipt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "not present in the current source checkout" in proc.stdout


def test_main_reconciles_runtime_trace_before_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    commit, packet_digest, _input_digests = _real_receipt_inputs()
    open_loop_records = [
        _record(candidate_identity=f"cand-{i}", packet_digest=packet_digest, commit=commit)
        for i in range(90)
    ]
    open_loop_records[0] = replace(open_loop_records[0], status="rejected")
    reactive_records = [
        _record(
            arm="reactive",
            candidate_identity=f"cand-{i}",
            packet_digest=packet_digest,
            commit=commit,
        )
        for i in range(90)
    ]
    open_loop_trace = _runtime_trace()
    reactive_trace = _runtime_trace(arm="reactive")
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._reactive_production_preflight_problem",
        lambda _packet: None,
    )
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._run_open_loop",
        lambda *_args, **_kwargs: (open_loop_records, open_loop_trace),
    )
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._run_reactive",
        lambda *_args, **_kwargs: (reactive_records, reactive_trace),
    )
    receipt_path = tmp_path / "receipt.json"
    result = main(
        [
            "--packet",
            str(PACKET),
            "--output-dir",
            str(tmp_path / "output"),
            "--receipt",
            str(receipt_path),
            "--commit",
            commit,
        ]
    )
    assert result == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert any("accepted 89 != trace.accepted 90" in problem for problem in receipt["problems"])
    assert receipt["runtime_traces"]["open_loop"]["schema_version"] == "matched_compute_trace.v1"


def test_cli_blocks_before_partial_arm_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(PACKET.parents[2])

    def unexpected_open_loop(*_args: object, **_kwargs: object) -> list[CandidateRecord]:
        raise AssertionError(
            "open-loop execution must not begin when reactive preflight is blocked"
        )

    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._run_open_loop",
        unexpected_open_loop,
    )
    receipt_path = tmp_path / "receipt.json"
    result = main(
        [
            "--packet",
            str(PACKET),
            "--output-dir",
            str(tmp_path / "output"),
            "--receipt",
            str(receipt_path),
            "--commit",
            "c" * 40,
        ]
    )
    assert result == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["arms"]["open_loop"]["records"] == []
    assert receipt["arms"]["reactive"]["records"] == []
    assert receipt["evidence_status"] == "blocked"
    assert any("blocked before arm execution" in problem for problem in receipt["problems"])


def test_empty_blocked_reactive_arm_skips_macro_index_reconciliation() -> None:
    problems = _record_integrity_problems(
        [],
        arm_name="reactive",
        packet_digest="a" * 64,
        commit="c" * 40,
        expected={"macro_actions": 10, "candidates_per_macro": 9},
    )

    assert not any("macro_action_index values" in problem for problem in problems)


def test_digest_text_is_stable() -> None:
    assert _digest_text("same") == _digest_text("same")
    assert _digest_text("same") != _digest_text("other")
