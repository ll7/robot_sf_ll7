"""Tests for the matched-compute production canary (issue #7893).

Validator/accounting failure cases use fixtures and injected fakes; the
tracked production receipt must come from the real seams.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.adversarial.config import SearchConfig
from robot_sf.adversarial.matched_compute import MatchedComputeRuntimeTrace
from scripts.validation.run_matched_compute_production_canary import (
    CANDIDATE_RECORD_SCHEMA,
    CandidateRecord,
    _aggregate_reconcile,
    _arm_evidence_status,
    _budget_reconcile,
    _canonical_episode_observation,
    _check_receipt,
    _cross_arm_episode_reuse_problems,
    _digest_file,
    _digest_text,
    _discover_frozen_input_files,
    _manifest_status,
    _packet_arm_expectations,
    _record_integrity_problems,
    _run_open_loop,
    _runtime_trace_is_production_observed,
    _validate_execution_destinations,
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
    episode_identity: str = TEST_ARTIFACT_ID,
    episode_digest: str = TEST_ARTIFACT_DIGEST,
    objective_value: float | str | None = 1.25,
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
        episode_identity=episode_identity,
        episode_digest=episode_digest,
        objective_value=objective_value,
    )


def _canonical_records(
    artifact_root: Path,
    *,
    arm: str,
    identity_prefix: str,
    packet_digest: str = "a" * 64,
    commit: str = "c" * 40,
    count: int = 90,
) -> list[CandidateRecord]:
    """Write genuine, unique canonical episode records for positive receipt tests."""
    records: list[CandidateRecord] = []
    for index in range(count):
        candidate_identity = f"{identity_prefix}-{index}"
        episode_path = artifact_root / arm / candidate_identity / "episode_records.jsonl"
        episode_path.parent.mkdir(parents=True)
        episode_payload = {
            "version": "v1",
            "episode_id": f"{arm}-episode-{index}",
            "scenario_id": "crossing_ttc_template",
            "seed": 123,
            "steps": 50,
            "metrics": {"collisions": 0},
            "termination_reason": "max_steps",
            "outcome": {
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            "integrity": {"contradictions": []},
        }
        episode_path.write_text(json.dumps(episode_payload) + "\n", encoding="utf-8")
        records.append(
            _record(
                arm=arm,
                candidate_identity=candidate_identity,
                packet_digest=packet_digest,
                commit=commit,
                episode_identity=episode_path.as_posix(),
                episode_digest=_digest_file(episode_path),
            )
        )
    return records


class _Trace:
    accepted = 90
    rejected = 0
    invalid = 0
    simulator_physics_steps = 90 * 50


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
        simulator_physics_steps=candidate_evaluations * 50,
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


def _fake_native_open_loop_runner(
    tmp_path: Path,
    *,
    scenario_seed: int = 123,
    config_overrides: dict[str, object] | None = None,
    objective_value: object = 1.25,
    include_objective: bool = True,
) -> Callable[[SearchConfig], SimpleNamespace]:
    """Return a native-manifest-shaped search fake with selectable source seed."""

    def _run(config: SearchConfig) -> SimpleNamespace:
        episode_path = tmp_path / "candidate_0000" / "episode_records.jsonl"
        episode_path.parent.mkdir(exist_ok=True)
        episode_path.write_text(
            json.dumps({"steps": 50, "min_robot_distance": 1.25}), encoding="utf-8"
        )
        manifest_path = tmp_path / "manifest.json"
        manifest_config = config.to_json()
        manifest_config.update(config_overrides or {})
        candidate_entry = {
            "candidate": {"scenario_seed": scenario_seed},
            "certification_status": {"status": "valid"},
            "bundle_path": str(episode_path.parent),
            "episode_record_path": str(episode_path),
        }
        if include_objective:
            candidate_entry["objective_value"] = objective_value
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "adversarial-search-manifest.v1",
                    "config": manifest_config,
                    "candidates": [candidate_entry],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            manifest_path=manifest_path,
            best_candidate=None,
            best_bundle_path=None,
            num_candidates=1,
            num_valid_candidates=1,
            num_invalid_candidates=0,
            num_failed_evaluations=0,
        )

    return _run


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


def test_aggregate_reconcile_requires_sum_of_all_episode_steps() -> None:
    records = [_record(candidate_identity=f"cand-{i}") for i in range(90)]
    correct_trace = replace(_runtime_trace(), simulator_physics_steps=90 * 50)

    assert _aggregate_reconcile(records, correct_trace, expected_simulator_steps=50) == []
    undersized_trace = replace(_runtime_trace(), simulator_physics_steps=50)
    problems = _aggregate_reconcile(records, undersized_trace, expected_simulator_steps=50)
    assert any("4500" in problem and "simulator_physics_steps" in problem for problem in problems)


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
    records = _canonical_records(tmp_path / "artifacts", arm="open_loop", identity_prefix="open")
    reactive_records = _canonical_records(
        tmp_path / "artifacts", arm="reactive", identity_prefix="reactive"
    )
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


@pytest.mark.parametrize(
    ("objective_value", "expected_problem"),
    [
        (None, "objective_value is required for accepted candidate"),
        ("not-a-number", "objective_value is not numeric"),
        (float("nan"), "objective_value is not finite"),
        (float("inf"), "objective_value is not finite"),
    ],
    ids=("missing", "non-numeric", "nan", "infinity"),
)
def test_receipt_check_rejects_accepted_candidate_without_finite_objective_full_180(
    objective_value: float | str | None,
    expected_problem: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All 180 records stay production-shaped while one accepted objective is invalid."""
    records = _canonical_records(tmp_path / "artifacts", arm="open_loop", identity_prefix="open")
    reactive_records = _canonical_records(
        tmp_path / "artifacts", arm="reactive", identity_prefix="reactive"
    )
    records[0] = replace(records[0], objective_value=objective_value)
    assert _arm_evidence_status(records) == "not_production_observed"
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
    assert expected_problem in capsys.readouterr().out


def test_receipt_check_binds_digest_to_single_episode_byte_buffer_full_180(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical replacement cannot be hashed as A and parsed as different bytes B."""
    records = _canonical_records(tmp_path / "artifacts", arm="open_loop", identity_prefix="open")
    reactive_records = _canonical_records(
        tmp_path / "artifacts", arm="reactive", identity_prefix="reactive"
    )
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

    target = Path(records[0].episode_identity)
    canonical_bytes_a = target.read_bytes()
    replacement = json.loads(canonical_bytes_a)
    replacement["episode_id"] = f"{replacement['episode_id']}-replacement"
    canonical_bytes_b = (json.dumps(replacement) + "\n").encode()
    assert hashlib.sha256(canonical_bytes_a).digest() != hashlib.sha256(canonical_bytes_b).digest()
    target.write_bytes(canonical_bytes_b)

    original_digest_file = _digest_file

    def _swap_between_digest_and_parse(path: Path) -> str:
        if path.resolve() == target.resolve():
            target.write_bytes(canonical_bytes_a)
            digest = original_digest_file(path)
            target.write_bytes(canonical_bytes_b)
            return digest
        return original_digest_file(path)

    _canonical_episode_observation.cache_clear()
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._digest_file",
        _swap_between_digest_and_parse,
    )

    assert _check_receipt(receipt_path) == 1
    assert target.read_bytes() == canonical_bytes_b


def test_receipt_check_rejects_arbitrary_regular_file_as_episode_provenance(
    tmp_path: Path,
) -> None:
    """A matching digest does not turn an arbitrary source file into episode evidence."""
    records = [_record(candidate_identity=f"open-{i}") for i in range(90)]
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


def test_episode_provenance_reconciles_steps_from_canonical_bytes(tmp_path: Path) -> None:
    record = _canonical_records(
        tmp_path / "artifacts", arm="open_loop", identity_prefix="open", count=1
    )[0]

    problems = _record_integrity_problems(
        [replace(record, simulator_steps=49)],
        arm_name="open_loop",
        packet_digest="a" * 64,
        commit="c" * 40,
    )

    assert any("simulator_steps does not match canonical episode artifact" in p for p in problems)


def test_episode_provenance_rejects_declared_integrity_contradictions(tmp_path: Path) -> None:
    record = _canonical_records(
        tmp_path / "artifacts", arm="open_loop", identity_prefix="open", count=1
    )[0]
    episode_path = Path(record.episode_identity)
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["integrity"]["contradictions"] = ["fixture contradiction"]
    episode_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    problems = _record_integrity_problems(
        [replace(record, episode_digest=_digest_file(episode_path))],
        arm_name="open_loop",
        packet_digest="a" * 64,
        commit="c" * 40,
    )

    assert any("declares integrity contradictions" in problem for problem in problems)


def test_episode_provenance_rejects_reuse_across_candidates(tmp_path: Path) -> None:
    records = _canonical_records(
        tmp_path / "artifacts", arm="open_loop", identity_prefix="open", count=2
    )
    records[1] = replace(
        records[1],
        episode_identity=records[0].episode_identity,
        episode_digest=records[0].episode_digest,
    )

    problems = _record_integrity_problems(
        records,
        arm_name="open_loop",
        packet_digest="a" * 64,
        commit="c" * 40,
    )

    assert any("reuse an episode artifact across candidates" in problem for problem in problems)
    assert any("reuse episode artifact bytes across candidates" in problem for problem in problems)


def test_episode_provenance_rejects_reuse_between_arms(tmp_path: Path) -> None:
    open_loop = _canonical_records(
        tmp_path / "artifacts", arm="open_loop", identity_prefix="shared", count=1
    )[0]
    reactive = replace(
        _record(arm="reactive", candidate_identity="shared-0"),
        episode_identity=open_loop.episode_identity,
        episode_digest=open_loop.episode_digest,
    )

    problems = _cross_arm_episode_reuse_problems([open_loop], [reactive], None)

    assert "receipt arms reuse an episode artifact across candidates" in problems
    assert "receipt arms reuse episode artifact bytes across candidates" in problems


def test_episode_provenance_must_stay_inside_receipt_artifact_bundle(tmp_path: Path) -> None:
    record = _canonical_records(
        tmp_path / "outside", arm="open_loop", identity_prefix="open", count=1
    )[0]

    problems = _record_integrity_problems(
        [record],
        arm_name="open_loop",
        packet_digest="a" * 64,
        commit="c" * 40,
        episode_artifact_root=tmp_path / "receipt_bundle",
    )

    assert any("outside the receipt's artifact bundle" in problem for problem in problems)


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
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary.run_adversarial_search",
        _fake_native_open_loop_runner(tmp_path),
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


def test_open_loop_rejects_manifest_scenario_seed_drift_before_record_emission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary.run_adversarial_search",
        _fake_native_open_loop_runner(tmp_path, scenario_seed=999),
    )

    with pytest.raises(ValueError, match="scenario_seed.*frozen packet"):
        _run_open_loop(
            load_packet(PACKET),
            tmp_path / "output",
            "c" * 40,
            "a" * 64,
        )


@pytest.mark.parametrize(
    ("include_objective", "objective_value", "expected_problem"),
    [
        (False, None, "objective_value is required for accepted candidate"),
        (True, "not-a-number", "objective_value is not numeric"),
        (True, float("nan"), "objective_value is not finite"),
        (True, float("inf"), "objective_value is not finite"),
    ],
    ids=("missing", "non-numeric", "nan", "infinity"),
)
def test_open_loop_rejects_accepted_manifest_without_finite_objective(
    include_objective: bool,
    objective_value: object,
    expected_problem: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary.run_adversarial_search",
        _fake_native_open_loop_runner(
            tmp_path,
            include_objective=include_objective,
            objective_value=objective_value,
        ),
    )

    with pytest.raises(ValueError, match=expected_problem):
        _run_open_loop(
            load_packet(PACKET),
            tmp_path / "output",
            "c" * 40,
            "a" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", "forged_policy"),
        ("objective", "forged_objective"),
        ("seed", 999),
        ("scenario_template", "configs/scenarios/templates/forged.yaml"),
    ],
)
def test_open_loop_rejects_native_manifest_config_drift(
    field: str, value: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary.run_adversarial_search",
        _fake_native_open_loop_runner(tmp_path, config_overrides={field: value}),
    )

    with pytest.raises(ValueError, match="manifest config.*frozen packet"):
        _run_open_loop(
            load_packet(PACKET),
            tmp_path / "output",
            "c" * 40,
            "a" * 64,
        )


def test_cli_check_mode(tmp_path: Path) -> None:
    commit, packet_digest, input_digests = _real_receipt_inputs()
    records = _canonical_records(
        tmp_path / "artifacts",
        arm="open_loop",
        identity_prefix="open",
        packet_digest=packet_digest,
        commit=commit,
    )
    reactive_records = _canonical_records(
        tmp_path / "artifacts",
        arm="reactive",
        identity_prefix="reactive",
        packet_digest=packet_digest,
        commit=commit,
    )
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
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._validate_execution_destinations",
        lambda output, receipt, _root: (output, receipt),
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
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._validate_execution_destinations",
        lambda output, receipt, _root: (output, receipt),
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


def test_main_rejects_out_of_scope_destinations_before_arm_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._reactive_production_preflight_problem",
        lambda _packet: None,
    )

    def unexpected_arm(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("an arm must not run before destination admission")

    monkeypatch.setattr(
        "scripts.validation.run_matched_compute_production_canary._run_open_loop",
        unexpected_arm,
    )
    receipt_path = tmp_path / "receipt.json"

    with pytest.raises(ValueError, match="issue-scoped"):
        main(
            [
                "--packet",
                str(PACKET),
                "--output-dir",
                str(tmp_path / "matched_compute_canary-prefix-confusable"),
                "--receipt",
                str(receipt_path),
                "--commit",
                "c" * 40,
            ]
        )

    assert not receipt_path.exists()


def _destination_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text("output/\n", encoding="utf-8")
    return repository


def test_execution_destination_gate_accepts_only_exact_ignored_scope(tmp_path: Path) -> None:
    repository = _destination_repository(tmp_path)
    output = repository / "output" / "matched_compute_canary" / "run"
    receipt = output / "receipt.json"

    assert _validate_execution_destinations(output, receipt, repository) == (output, receipt)


@pytest.mark.parametrize(
    "unsafe_output",
    [
        Path("output/matched_compute_canary-prefix-confusable"),
        Path("output/matched_compute_canary/../../elsewhere"),
        Path("elsewhere/matched_compute_canary"),
    ],
)
def test_execution_destination_gate_rejects_scope_escapes(
    unsafe_output: Path, tmp_path: Path
) -> None:
    repository = _destination_repository(tmp_path)
    receipt = repository / "output" / "matched_compute_canary" / "receipt.json"

    with pytest.raises(ValueError, match="issue-scoped"):
        _validate_execution_destinations(unsafe_output, receipt, repository)


@pytest.mark.parametrize("tracked_kind", ["output", "receipt"])
def test_execution_destination_gate_rejects_tracked_paths(
    tracked_kind: str, tmp_path: Path
) -> None:
    repository = _destination_repository(tmp_path)
    allowed = repository / "output" / "matched_compute_canary"
    output = allowed / "run"
    receipt = allowed / "receipt.json"
    tracked_path = output / "tracked.json" if tracked_kind == "output" else receipt
    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_path.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", str(tracked_path)], cwd=repository, check=True)

    with pytest.raises(ValueError, match="tracked path"):
        _validate_execution_destinations(output, receipt, repository)


@pytest.mark.parametrize("symlink_kind", ["output", "receipt"])
def test_execution_destination_gate_rejects_symlinks(symlink_kind: str, tmp_path: Path) -> None:
    repository = _destination_repository(tmp_path)
    allowed = repository / "output" / "matched_compute_canary"
    allowed.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    output = allowed / "run"
    receipt = allowed / "receipt.json"
    if symlink_kind == "output":
        output.symlink_to(outside, target_is_directory=True)
    else:
        outside_receipt = outside / "receipt.json"
        outside_receipt.write_text("{}", encoding="utf-8")
        receipt.symlink_to(outside_receipt)

    with pytest.raises(ValueError, match="symlink"):
        _validate_execution_destinations(output, receipt, repository)


def test_execution_destination_gate_rejects_nested_output_symlink(tmp_path: Path) -> None:
    repository = _destination_repository(tmp_path)
    output = repository / "output" / "matched_compute_canary" / "run"
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "candidate_0000").symlink_to(outside, target_is_directory=True)
    receipt = repository / "output" / "matched_compute_canary" / "receipt.json"

    with pytest.raises(ValueError, match="symlink member"):
        _validate_execution_destinations(output, receipt, repository)


def test_execution_destination_gate_rejects_hard_link_to_tracked_file(tmp_path: Path) -> None:
    repository = _destination_repository(tmp_path)
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    allowed = repository / "output" / "matched_compute_canary"
    allowed.mkdir(parents=True)
    receipt = allowed / "receipt.json"
    receipt.hardlink_to(repository / ".gitignore")

    with pytest.raises(ValueError, match="hard-linked"):
        _validate_execution_destinations(allowed / "run", receipt, repository)


@pytest.mark.parametrize("invalid_kind", ["output_file", "receipt_directory"])
def test_execution_destination_gate_requires_destination_file_types(
    invalid_kind: str, tmp_path: Path
) -> None:
    repository = _destination_repository(tmp_path)
    allowed = repository / "output" / "matched_compute_canary"
    allowed.mkdir(parents=True)
    output = allowed / "run"
    receipt = allowed / "receipt.json"
    if invalid_kind == "output_file":
        output.write_text("not a directory\n", encoding="utf-8")
    else:
        receipt.mkdir()

    with pytest.raises(ValueError, match="directory|regular file"):
        _validate_execution_destinations(output, receipt, repository)


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
