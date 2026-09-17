"""BA-02 queue contracts and offline selection behaviour.

These tests cover the public queue contract: lexicographic priority bands,
explainable active components, detector-independent controls, explicit review
credit, and lossless state recovery.  They intentionally do not run a browser,
agent, simulator, shell command, or network request.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    EpisodeRef,
    Finding,
    ReviewRecord,
    Signal,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_queue import (
    ACTIVE_WEIGHTS,
    ActivePolicy,
    AuditQueue,
    CoverageDeficit,
    QueueConflictError,
    QueueDataset,
    QueueInputError,
    QueueOperationConflictError,
    QueuePolicy,
    QueueState,
    QueueStateError,
    ScanSummary,
    load_queue_input,
)
from robot_sf.analysis_workbench.audit_store import AuditStore


def _episode(
    execution_id: str,
    *,
    planner_id: str = "ppo",
    scenario_id: str = "corridor",
    seed: int = 1,
    source_digest: str = "b" * 64,
    config_digest: str = "c" * 64,
) -> EpisodeRef:
    return EpisodeRef(
        campaign_digest="a" * 64,
        source_digest=source_digest,
        execution_id=execution_id,
        planner_id=planner_id,
        scenario_id=scenario_id,
        seed=seed,
        config_digest=config_digest,
        environment_digest="e" * 64,
    )


def _candidate(
    execution_id: str,
    *,
    planner_id: str = "ppo",
    metadata: dict[str, object] | None = None,
    signals: tuple[Signal, ...] = (),
    trace: bool | None = None,
    outcome: str = "success",
    stratum_id: str = "ppo|corridor|success",
):
    from robot_sf.analysis_workbench.audit_queue import QueueCandidate

    return QueueCandidate(
        _episode(execution_id, planner_id=planner_id),
        metadata=metadata or {},
        signals=signals,
        trace_available=trace,
        trace_identity=f"trace-{execution_id}" if trace else "",
        outcome=outcome,
        stratum_id=stratum_id,
    )


def _canonical_alignment(primary, peer):
    """Build the owner-shaped compatibility receipt required for peer display."""

    checks = {
        "scenario_id_equal": True,
        "coordinate_frame_equal": True,
        "units_equal": True,
        "seed_equal": True,
        "planner_id_different": True,
        "map_id_present": True,
        "map_id_equal": True,
        "horizon_present": True,
        "horizon_equal": True,
        "config_digest_present": True,
        "time_step_s_present": True,
        "time_step_s_equal": True,
    }
    availability = {
        name: {"left": "fixture", "right": "fixture", "status": "available"}
        for name in ("map_id", "horizon", "config_digest", "time_step_s")
    }
    initial = {
        "status": "available",
        "equivalent": True,
        "robot_position_delta_m": 0.0,
        "robot_velocity_delta_mps": 0.0,
        "robot_heading_delta_rad": 0.0,
        "robot_radius_delta_m": 0.0,
        "actor_id_sets_equal": True,
        "actor_position_delta_m": {},
        "actor_velocity_delta_mps": {},
        "actor_radius_delta_m": {},
        "max_actor_position_delta_m": None,
        "max_actor_velocity_delta_mps": None,
        "max_actor_radius_delta_m": None,
        "position_tolerance_m": 1e-6,
        "heading_tolerance_rad": 1e-6,
    }
    return {
        "profile_version": "pair_compatibility.deterministic.v1",
        "status": "available",
        "comparison_grain": {
            "grain_id": "matched_planner_pair",
            "left_role": "primary_trace",
            "right_role": "comparison_trace",
        },
        "provenance": {
            "left_artifact_id": f"artifact-{primary.episode.execution_id}",
            "right_artifact_id": f"artifact-{peer.episode.execution_id}",
            "left_trace_id": primary.trace_identity,
            "right_trace_id": peer.trace_identity,
        },
        "provenance_gate": {
            "status": "available",
            "compatible": True,
            "comparison_grain": "matched_planner_pair",
            "left_content_sha256": "e" * 64,
            "right_content_sha256": "f" * 64,
            "checks": checks,
            "availability": availability,
            "time_step_contracts": {
                "left": {"status": "available"},
                "right": {"status": "available"},
            },
        },
        "right_source_trace": {
            "status": "available",
            "schema_version": "simulation_trace_export.v1",
            "trace_id": peer.trace_identity,
            "content_sha256": "f" * 64,
            "content_receipt": {"content_contract": {}},
            "source": {
                "episode_id": peer.episode_id,
                "scenario_id": peer.scenario_id,
                "planner_id": peer.planner_id,
                "seed": peer.episode.seed,
            },
        },
        "initial_state_equivalence": initial,
    }


def _select_from_worker(state_path: str, start_event, result_queue) -> None:
    """Attempt one synchronized state write in a separate process."""

    queue = AuditQueue((_candidate("cross-process"),), state_path=state_path)
    start_event.wait(timeout=10)
    try:
        queue.select_next()
    except QueueConflictError:
        result_queue.put("conflict")
    else:
        result_queue.put("ok")


def test_fixed_priority_bands_are_lexicographic_before_weights() -> None:
    benchmark = _candidate(
        "benchmark",
        metadata={"benchmark_config_defect": 0.01, "safety_severity": 1.0},
    )
    safety = _candidate("safety", metadata={"safety_severity": 1.0})
    queue = AuditQueue(
        QueueDataset((safety, benchmark)),
        policy=QueuePolicy.fixed(),
    )

    ranked = queue.rank_candidates()

    assert [item.episode_id for item in ranked] == [benchmark.episode_id, safety.episode_id]
    assert ranked[0].explanation.priority_band == "benchmark_config_defect"
    assert ranked[0].explanation.band_rank < ranked[1].explanation.band_rank
    assert all(
        "probability" not in reason for item in ranked for reason in item.explanation.reasons
    )


def test_typed_global_ba01_signals_drive_band_and_score() -> None:
    candidate = _candidate("global-signal")
    signal = Signal(
        signal_id="global-config-signal",
        detector_id="config-integrity",
        episode_id=candidate.episode_id,
        status="flagged",
        reason_code="configuration defect",
        measured={"severity": 0.75},
    )

    ranked = AuditQueue(QueueDataset((candidate,), signals=(signal,)), policy=QueuePolicy.fixed())
    result = ranked.rank_candidates()[0]

    assert result.explanation.priority_band == "benchmark_config_defect"
    assert result.explanation.components["signal_strength"] == 0.75
    assert "ba-01-signals:unavailable" not in ranked.dataset.missingness


def test_active_explanation_exposes_coverage_hypothesis_novelty_and_redundancy() -> None:
    reviewed = _candidate(
        "reviewed",
        metadata={"symptoms": ["turning"], "geometry_signature": ["map-a"]},
    )
    repeated = _candidate(
        "repeated",
        metadata={"symptoms": ["turning"], "geometry_signature": ["map-a"]},
    )
    novel = _candidate(
        "novel",
        metadata={
            "symptoms": ["turning"],
            "geometry_signature": ["map-b"],
            "contradictory_evidence": True,
            "competing_hypotheses": ["goal geometry", "controller"],
            "more_evidence_requested": True,
        },
    )
    finding = Finding(
        finding_id="finding-turning",
        title="turning symptom",
        status="under_investigation",
        candidate_members=(reviewed.episode_id, repeated.episode_id),
        hypotheses=("controller instability",),
        source_revision="scan-1",
    )
    review = ReviewRecord(
        review_id="reviewed-1",
        episode_id=reviewed.episode_id,
        scope="full_episode",
        source_revision=1,
    )
    queue = AuditQueue(
        QueueDataset(
            (reviewed, repeated, novel),
            findings=(finding,),
            review_records=(review,),
            coverage_deficits=(
                CoverageDeficit("ppo|corridor|success", 0, 2, "scan-1", "protocol-1"),
            ),
            scan_summary=ScanSummary("scan-1", revision=1, accounting={"expected_rows": 3}),
            input_revision=1,
        ),
        policy=ActivePolicy(),
    )

    ranked = {item.episode_id: item for item in queue.rank_candidates()}

    assert ranked[novel.episode_id].explanation.components["hypothesis_gain"] == 1.0
    assert ranked[novel.episode_id].explanation.components["unresolved_request"] == 1.0
    assert (
        ranked[novel.episode_id].explanation.components["novelty"]
        > ranked[repeated.episode_id].explanation.components["novelty"]
    )
    assert ranked[repeated.episode_id].explanation.components["redundancy_penalty"] > 0.0
    assert any(
        "repeated evidence lowers priority" in reason
        for reason in ranked[repeated.episode_id].explanation.reasons
    )


def test_control_stream_surfaces_common_mode_flag_independently_of_anomaly_rank() -> None:
    common_mode = _candidate(
        "common-mode",
        metadata={"control_eligible": True, "ordinary": True, "trace_available": True},
        signals=(
            Signal(
                signal_id="common-signal",
                detector_id="outcome-check",
                episode_id=_episode("common-mode").episode_id,
                status="flagged",
                reason_code="common-mode anomaly",
            ),
        ),
    )
    high_anomaly = _candidate(
        "anomaly",
        metadata={"benchmark_config_defect": 1.0},
        signals=(Signal(signal_id="high-signal", detector_id="config", status="flagged"),),
    )
    dataset = QueueDataset(
        (high_anomaly, common_mode),
        signals=(common_mode.signals[0],),
        coverage_deficits=(
            CoverageDeficit(common_mode.effective_stratum_id, 0, 1, "scan-1", "protocol-1"),
        ),
        scan_summary=ScanSummary("scan-1", revision=1, accounting={"expected_rows": 2}),
        input_revision=1,
    )
    queue = AuditQueue(
        dataset, policy=ActivePolicy(control_schedule=(0,), max_control_selections=1), seed=7
    )

    result = queue.select_next()

    assert result is not None
    assert result.packet.primary.episode_id == common_mode.episode_id
    assert result.context.selection_kind == "control"
    assert result.context.control_reason
    assert result.packet.signals[0].signal_id == "common-signal"
    assert any(
        "control selection ignores detector flags" in reason
        for reason in result.packet.selection_reasons
    )


def test_under_review_controls_are_sampled_with_persisted_rng_state(tmp_path: Path) -> None:
    controls = tuple(
        _candidate(
            f"control-{index}",
            planner_id="orca",
            metadata={"control_eligible": True, "ordinary": True, "trace_available": True},
            stratum_id="orca|corridor|success",
        )
        for index in range(4)
    )
    dataset = QueueDataset(
        controls,
        coverage_deficits=(CoverageDeficit("orca|corridor|success", 0, 4, "scan-1", "protocol-1"),),
        scan_summary=ScanSummary("scan-1", revision=1),
        input_revision=1,
    )
    state_path = tmp_path / "queue-state.json"
    policy = ActivePolicy(control_schedule=(0, 1, 2, 3), max_control_selections=4)
    first = AuditQueue(dataset, policy=policy, seed=11, state_path=state_path)
    first_result = first.select_next()
    second_result = first.select_next()
    assert first_result is not None and second_result is not None

    replay_state_path = tmp_path / "queue-state-replay.json"
    replay_state_path.write_bytes(state_path.read_bytes())
    resumed = AuditQueue(dataset, policy=policy, seed=999, state_path=state_path)
    replay = AuditQueue(dataset, policy=policy, seed=123, state_path=replay_state_path)
    resumed_result = resumed.select_next()
    replay_result = replay.select_next()

    assert resumed_result is not None
    assert replay_result is not None
    assert resumed_result.context.selection_kind == "control"
    assert resumed_result.packet.primary.episode_id == replay_result.packet.primary.episode_id
    assert resumed.state.rng_state == replay.state.rng_state


def test_missing_and_incompatible_peers_are_explained_without_blocking_primary() -> None:
    primary = _candidate(
        "primary",
        metadata={"benchmark_config_defect": 1.0},
        trace=True,
    )
    compatible = _candidate(
        "compatible",
        planner_id="orca",
        trace=True,
    )
    compatible = replace_candidate(
        compatible,
        metadata={"event_alignment": _canonical_alignment(primary, compatible)},
    )
    missing = _candidate("missing", planner_id="social-force", trace=False)
    incompatible = _candidate(
        "incompatible",
        planner_id="mpc",
        trace=True,
    )
    # Source identity differs, so this row cannot be represented as a peer.
    incompatible = replace_candidate(incompatible, source_digest="d" * 64)
    queue = AuditQueue(QueueDataset((primary, compatible, missing, incompatible)))

    result = queue.select_next()

    assert result is not None
    assert result.packet.primary.episode_id == primary.episode_id
    assert [item.planner_id for item in result.packet.peers] == ["orca"]
    assert any("trace-unavailable" in item for item in result.packet.missingness)
    assert any("incompatible-identity" in item for item in result.packet.missingness)


def replace_candidate(candidate, *, source_digest: str | None = None, metadata=None):
    from robot_sf.analysis_workbench.audit_queue import QueueCandidate

    source_digest = source_digest or candidate.episode.source_digest
    return QueueCandidate(
        _episode(
            candidate.episode.execution_id,
            planner_id=candidate.planner_id,
            source_digest=source_digest,
        ),
        signals=candidate.signals,
        metadata=candidate.metadata if metadata is None else metadata,
        trace_available=candidate.trace_available,
        trace_identity=candidate.trace_identity,
        stratum_id=candidate.stratum_id,
        outcome=candidate.outcome,
    )


def test_manual_actions_review_credit_and_store_actor_kinds(tmp_path: Path) -> None:
    first = _candidate("first", metadata={"benchmark_config_defect": 1.0})
    second = _candidate("second", metadata={"safety_severity": 1.0})
    with AuditStore(tmp_path / "audit") as store:
        queue = AuditQueue(
            QueueDataset((first, second)), store=store, state_path=tmp_path / "state.json"
        )
        selected = queue.select_next()
        assert selected is not None
        queue.pin(second.episode_id, operation_id="manual-pin")
        queue.skip(operation_id="manual-skip")
        pinned = queue.select_next()
        assert pinned is not None and pinned.packet.primary.episode_id == second.episode_id
        review = queue.record_review(scope="full_episode", operation_id="human-review")
        assert review.author_kind == "human"
        actions = [item.record for item in store.list_records(record_type="action_record")]
        reviews = [item.record for item in store.list_records(record_type="review_record")]

    assert any(
        item.action_type == "queue_select" and item.actor_kind == "agent" for item in actions
    )
    assert any(item.action_type == "pin" and item.actor_kind == "human" for item in actions)
    assert any(item.action_type == "skip" and item.actor_kind == "human" for item in actions)
    assert reviews and reviews[0].author_kind == "human"


def test_previous_packet_and_resume_do_not_create_human_review_credit(tmp_path: Path) -> None:
    candidates = tuple(_candidate(f"episode-{index}") for index in range(3))
    state_path = tmp_path / "state.json"
    queue = AuditQueue(candidates, state_path=state_path, seed=4)
    first = queue.select_next()
    second = queue.select_next()
    assert first is not None and second is not None
    previous = queue.previous_packet()
    assert previous is not None and previous.packet.packet_id == first.packet.packet_id
    resumed = AuditQueue(candidates, state_path=state_path).resume()
    assert resumed is not None and resumed.packet.packet_id == first.packet.packet_id
    assert queue.dataset.review_records == ()


def test_previous_packet_replay_is_idempotent_and_collision_safe() -> None:
    queue = AuditQueue(tuple(_candidate(f"episode-{index}") for index in range(4)))
    first = queue.select_next()
    second = queue.select_next()
    third = queue.select_next()
    assert first is not None and second is not None and third is not None

    previous = queue.previous_packet(operation_id="previous-op")
    assert previous is not None
    revision = queue.state_revision
    replay = queue.previous_packet(operation_id="previous-op")
    assert replay is not None and replay.packet.packet_id == previous.packet.packet_id
    assert queue.state_revision == revision
    with pytest.raises(QueueOperationConflictError):
        queue.pin(first.packet.primary.episode_id, operation_id="previous-op")


def test_stale_scan_and_finding_updates_are_visible_on_resume() -> None:
    candidate = _candidate("episode")
    finding = Finding(
        finding_id="finding",
        title="symptom",
        source_revision="scan-1",
    )
    initial = QueueDataset(
        (candidate,),
        findings=(finding,),
        scan_summary=ScanSummary("scan-1", revision=1),
        input_revision=1,
    )
    queue = AuditQueue(initial)
    first = queue.select_next()
    assert first is not None
    queue.update_dataset(
        QueueDataset(
            (candidate,),
            findings=(Finding(finding_id="finding", title="symptom", source_revision="scan-2"),),
            scan_summary=ScanSummary("scan-1", revision=2),
            input_revision=2,
        )
    )

    resumed = queue.resume(reload=False)

    assert resumed is not None
    assert "stale-input-revision" in resumed.packet.missingness
    assert "stale-scan-summary-revision" in resumed.packet.missingness
    assert "stale-finding-update" in resumed.packet.missingness


def test_state_compare_and_swap_and_operation_collision_are_fail_closed(tmp_path: Path) -> None:
    candidate = _candidate("episode")
    state_path = tmp_path / "state.json"
    first = AuditQueue((candidate,), state_path=state_path)
    first.select_next()
    stale = AuditQueue((candidate,), state_path=state_path)
    first.select_next(force_current=True)
    with pytest.raises(QueueConflictError):
        stale.save_state(expected_revision=1)


def test_untrusted_json_and_unsupported_state_are_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "nan.json"
    input_path.write_text('{"candidates": [], "bad": NaN}', encoding="utf-8")
    with pytest.raises(QueueInputError, match="invalid JSON constant"):
        load_queue_input(input_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": "audit-queue.v99"}), encoding="utf-8")
    with pytest.raises(QueueStateError, match="unsupported queue state schema"):
        AuditQueue((), state_path=state_path)


def test_absent_ba01_ba04_inputs_are_explicit_missingness() -> None:
    queue = AuditQueue((_candidate("episode"),))

    result = queue.select_next()

    assert result is not None
    assert "ba-01-signals:unavailable" in result.packet.missingness
    assert "ba-01-scan-summary:unavailable" in result.packet.missingness
    assert "ba-04-coverage-deficits:unavailable" in result.packet.missingness


def test_checked_in_fixture_supports_offline_rank_and_select() -> None:
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "analysis_workbench"
        / "audit_queue_v1"
        / "queue.json"
    )

    dataset = load_queue_input(fixture)
    result = AuditQueue(
        dataset,
        policy=QueuePolicy.fixed(),
        seed=3,
    ).select_next()

    assert len(dataset.candidates) == 4
    assert result is not None
    assert result.packet.primary.planner_id == "ppo"
    assert result.packet.policy_version == "audit-queue.fixed.v1"
    assert result.context.accounting["expected_rows"] == 4
    assert result.context.scan_summary_id == "scan-summary-4"
    assert result.context.seed == 11


@pytest.mark.skipif(
    not hasattr(multiprocessing, "get_context"), reason="multiprocessing context unavailable"
)
def test_state_cas_serializes_synchronized_cross_process_writers(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    state_path = str(tmp_path / "state.json")
    start_event = context.Event()
    result_queue = context.Queue()
    workers = [
        context.Process(target=_select_from_worker, args=(state_path, start_event, result_queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start_event.set()
    results = [result_queue.get(timeout=10) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
    assert sorted(results) == ["conflict", "ok"]
    assert all(worker.exitcode == 0 for worker in workers)
    payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
    assert payload["state_revision"] == 1
    assert Path(f"{state_path}.lock").is_file()


def test_corrupt_current_or_history_packet_snapshots_fail_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    queue = AuditQueue((_candidate("snapshot-a"), _candidate("snapshot-b")), state_path=state_path)
    first = queue.select_next()
    assert first is not None
    second = queue.select_next()
    assert second is not None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["packet_payloads"][first.packet.packet_id]["record_id"] = "forged"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QueueStateError, match="corrupt persisted packet snapshot"):
        AuditQueue((_candidate("snapshot-a"), _candidate("snapshot-b")), state_path=state_path)

    payload["packet_payloads"][first.packet.packet_id]["record_id"] = first.packet.packet_id
    payload["current_packet_id"] = first.packet.packet_id
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    resumed = AuditQueue(
        (_candidate("snapshot-a"), _candidate("snapshot-b")), state_path=state_path
    )
    resumed.state.packet_payloads[first.packet.packet_id]["record_id"] = "forged-again"
    with pytest.raises(QueueStateError, match="corrupt current packet snapshot"):
        resumed.resume(reload=False)


def test_minimal_caller_compatibility_bit_never_exposes_peer() -> None:
    primary = _candidate("canonical-primary", trace=True)
    peer = _candidate(
        "caller-claims-compatible",
        planner_id="orca",
        trace=True,
        metadata={
            "event_alignment": {
                "status": "available",
                "provenance_gate": {"status": "available", "compatible": True},
            }
        },
    )
    result = AuditQueue((primary, peer)).select_next()
    assert result is not None
    assert result.packet.peers == ()
    assert any("alignment-profile-mismatch" in item for item in result.packet.missingness)


def test_peer_gate_rejects_incomplete_or_mismatched_owner_receipts() -> None:
    primary = _candidate("gate-primary", trace=True)
    peer = _candidate("gate-peer", planner_id="orca", trace=True)
    queue = AuditQueue((primary, peer))

    cases = [
        None,
        {"schema_version": "wrong-wrapper", "compatibility": _canonical_alignment(primary, peer)},
        {
            "schema_version": "review-alignment.v1",
            "alignment_sha256": "a" * 64,
            "compatibility": _canonical_alignment(primary, peer),
        },
        {
            "pair_compatibility": _canonical_alignment(primary, peer),
        },
    ]
    for alignment in cases:
        compatible, reasons = queue._canonical_peer_alignment(primary, peer, alignment)
        assert compatible is False
        assert reasons

    role_mismatch = _canonical_alignment(primary, peer)
    role_mismatch["comparison_grain"]["left_role"] = "comparison_trace"
    compatible, reasons = queue._canonical_peer_alignment(primary, peer, role_mismatch)
    assert compatible is False
    assert "alignment-role-mismatch" in reasons

    source_field_mismatch = _canonical_alignment(primary, peer)
    source_field_mismatch["provenance_gate"]["availability"]["map_id"] = {"status": "available"}
    compatible, reasons = queue._canonical_peer_alignment(primary, peer, source_field_mismatch)
    assert compatible is False
    assert "alignment-source-field-missing:map_id" in reasons

    source_mismatch = _canonical_alignment(primary, peer)
    source_mismatch["right_source_trace"]["source"].update(
        {"episode_id": "wrong", "scenario_id": "wrong", "planner_id": "wrong", "seed": 9}
    )
    source_mismatch_peer = replace_candidate(
        peer,
        metadata={
            "event_alignment": source_mismatch,
            "trace_content_sha256": "not-a-digest",
        },
    )
    compatible, reasons = queue._canonical_peer_alignment(
        primary, source_mismatch_peer, source_mismatch
    )
    assert compatible is False
    assert "peer-source-episode_id-mismatch" in reasons
    assert "peer-trace-content-identity-malformed" in reasons

    primary_with_content = _candidate(
        "gate-primary-content",
        trace=True,
        metadata={"trace_content_sha256": "a" * 64},
    )
    peer_with_content = _candidate("gate-peer-content", planner_id="orca", trace=True)
    content_mismatch = _canonical_alignment(primary_with_content, peer_with_content)
    content_mismatch["right_source_trace"]["content_sha256"] = "a" * 64
    content_mismatch["provenance_gate"]["right_content_sha256"] = "f" * 64
    peer_with_content = replace_candidate(
        peer_with_content,
        metadata={
            "event_alignment": content_mismatch,
            "trace_content_sha256": "b" * 64,
        },
    )
    compatible, reasons = queue._canonical_peer_alignment(
        primary_with_content, peer_with_content, content_mismatch
    )
    assert compatible is False
    assert "peer-trace-content-identity-mismatch" in reasons
    assert "primary-trace-content-identity-mismatch" in reasons

    malformed_initial = _canonical_alignment(primary, peer)
    malformed_initial["initial_state_equivalence"] = {
        **malformed_initial["initial_state_equivalence"],
        "actor_id_sets_equal": False,
        "robot_position_delta_m": "not-a-number",
        "actor_position_delta_m": {"actor": None},
        "actor_velocity_delta_mps": {"actor": "not-a-number"},
        "actor_radius_delta_m": {"actor": None},
    }
    malformed_initial_peer = replace_candidate(
        peer,
        metadata={
            "event_alignment": malformed_initial,
            "initial_state_identity": "a" * 64,
        },
    )
    compatible, reasons = queue._canonical_peer_alignment(
        primary, malformed_initial_peer, malformed_initial
    )
    assert compatible is False
    assert "initial-state-actor-identity-mismatch" in reasons
    assert "initial-state-field-malformed:robot_position_delta_m" in reasons
    assert "initial-state-identity-mismatch" in reasons


def test_packet_identity_binds_material_peer_alignment_content() -> None:
    primary = _candidate("identity-primary", trace=True)
    peer = _candidate("identity-peer", planner_id="orca", trace=True)
    aligned = replace_candidate(
        peer,
        metadata={"event_alignment": _canonical_alignment(primary, peer)},
    )
    first = AuditQueue((primary, aligned)).select_next()
    assert first is not None
    changed = _canonical_alignment(primary, peer)
    changed["right_source_trace"]["content_sha256"] = "e" * 64
    changed_peer = replace_candidate(peer, metadata={"event_alignment": changed})
    second = AuditQueue((primary, changed_peer)).select_next()
    assert second is not None
    assert first.packet.packet_id != second.packet.packet_id


def test_policy_semantics_change_marks_resume_stale(tmp_path: Path) -> None:
    state_path = tmp_path / "policy-state.json"
    candidate = _candidate("policy-stale")
    original = AuditQueue((candidate,), policy=ActivePolicy(), state_path=state_path)
    assert original.select_next() is not None
    changed_weights = dict(ACTIVE_WEIGHTS)
    changed_weights["novelty"] = 0.91
    changed = AuditQueue(
        (candidate,),
        policy=ActivePolicy(weights=changed_weights),
        state_path=state_path,
    )
    resumed = changed.resume()
    assert resumed is not None
    assert "stale-policy-semantics" in resumed.packet.missingness


def test_queue_dataset_requires_exact_root_schema_version() -> None:
    with pytest.raises(QueueInputError, match="unsupported queue input schema_version"):
        QueueDataset.from_mapping({"schema_version": "audit-queue.v99", "candidates": []})
    with pytest.raises(QueueInputError, match="unsupported queue input schema_version"):
        QueueDataset.from_mapping({"candidates": []})


def test_detector_unavailable_accounting_is_propagated() -> None:
    dataset = QueueDataset(
        (_candidate("detector-unavailable"),),
        scan_summary=ScanSummary(
            "scan-detector",
            detector_accounting={
                "video.v1": {"status": "unavailable", "reason": "not recorded"},
                "collision.v1": {"status": "error", "reason": "corrupt"},
            },
        ),
    )
    assert "ba-01-detector:video.v1:unavailable" in dataset.missingness
    assert "ba-01-detector:collision.v1:error" in dataset.missingness
    assert set(dataset.accounting["unavailable_detectors"]) == {"video.v1", "collision.v1"}

    signal_dataset = QueueDataset(
        (
            _candidate(
                "signal-detector-unavailable",
                signals=(
                    Signal(
                        signal_id="missing-signal",
                        detector_id="video.v1",
                        status="unavailable",
                        message="trace was not recorded",
                    ),
                ),
            ),
        )
    )
    assert "ba-01-detector:video.v1:unavailable" in signal_dataset.missingness


def test_scan_and_coverage_source_revision_mismatches_are_visible() -> None:
    dataset = QueueDataset(
        (_candidate("coverage-mismatch"),),
        source_digest="b" * 64,
        protocol_version="protocol-current",
        protocol_digest="p" * 64,
        accounting={
            "scan_summary_id": "scan-other",
            "scan_summary_revision": 9,
            "scan_source_revision": "scan-other",
            "scan_source_id": "source-other",
        },
        scan_summary=ScanSummary(
            "scan-current",
            revision=2,
            source_revision="scan-stale",
            source_id="source-stale",
        ),
        coverage_deficits=(
            CoverageDeficit(
                "ppo|corridor|success",
                0,
                1,
                "scan-other",
                "protocol-old",
                source_id="source-old",
                protocol_id="digest-old",
            ),
            CoverageDeficit(
                "ppo|corridor|failure",
                0,
                1,
                "scan-other-2",
                "protocol-old-2",
                source_id="source-old-2",
                protocol_id="digest-old-2",
            ),
        ),
        input_revision=3,
    )
    assert "scan-summary:accounting-id-mismatch" in dataset.missingness
    assert "scan-summary:accounting-revision-mismatch" in dataset.missingness
    assert "scan-summary:source-revision-mismatch" in dataset.missingness
    assert "scan-summary:source-id-mismatch" in dataset.missingness
    assert "scan-summary:input-revision-mismatch" in dataset.missingness
    assert "scan-summary:accounting-source-revision-mismatch" in dataset.missingness
    assert "scan-summary:accounting-source-id-mismatch" in dataset.missingness
    assert "coverage:source-revision-mismatch" in dataset.missingness
    assert "coverage:protocol-revision-mismatch" in dataset.missingness
    assert "coverage:ppo|corridor|success:source-revision-mismatch" in dataset.missingness
    assert "coverage:ppo|corridor|success:protocol-revision-mismatch" in dataset.missingness
    assert "coverage:ppo|corridor|success:source-id-mismatch" in dataset.missingness
    assert "coverage:ppo|corridor|success:protocol-id-mismatch" in dataset.missingness


def test_conflicting_global_and_local_signal_ids_fail_closed() -> None:
    candidate = _candidate(
        "signal-conflict",
        signals=(
            Signal(
                signal_id="duplicate",
                detector_id="local-detector",
                status="flagged",
                reason_code="one",
            ),
        ),
    )
    global_signal = Signal(
        signal_id="duplicate", detector_id="global-detector", status="clear", reason_code="two"
    )
    with pytest.raises(QueueInputError, match="conflicting signal payload"):
        QueueDataset((candidate,), signals=(global_signal,))


@pytest.mark.parametrize(
    "metadata",
    [
        {"finding_ids": [{"not": "an-id"}]},
        {"symptoms": {"not": "a-sequence"}},
        {"competing_hypotheses": "characters-are-not-hypotheses"},
    ],
)
def test_malformed_metadata_containers_raise_queue_input_error(metadata) -> None:
    with pytest.raises(QueueInputError):
        _candidate("malformed-metadata", metadata=metadata)


def test_metadata_validation_rejects_scalars_and_accepts_explicit_strings() -> None:
    # String conveniences are accepted only for descriptive fields; all
    # identity/control containers remain explicitly typed.
    descriptive = _candidate(
        "metadata-strings",
        metadata={
            "geometry_signature": "single-geometry",
            "tags": "single-tag",
            "symptom": "single-symptom",
        },
    )
    assert "geometry:single-geometry" in descriptive.feature_signatures
    assert "tag:single-tag" in descriptive.feature_signatures
    assert "symptom:single-symptom" in descriptive.feature_signatures

    mapping_tags = _candidate("metadata-tag-map", metadata={"tags": {"tag-key": True}})
    assert "tag:tag-key" in mapping_tags.feature_signatures

    for metadata in (
        {"alignment": "not-a-mapping"},
        {"ordinary": "not-a-boolean"},
        {"review_scope": 7},
        {"trace": "not-a-trace"},
    ):
        with pytest.raises(QueueInputError):
            _candidate("metadata-invalid", metadata=metadata)

    with pytest.raises(QueueInputError, match="finite number"):
        ActivePolicy(weights={"overflow": 10**1000})


def test_state_snapshot_shape_and_symlink_errors_fail_closed(tmp_path: Path) -> None:
    queue = AuditQueue((_candidate("state-shape"),))
    selected = queue.select_next()
    assert selected is not None
    signal_payload = record_to_dict(
        Signal(
            signal_id="not-a-packet",
            detector_id="detector",
            status="flagged",
        )
    )
    with pytest.raises(QueueStateError, match="not a review_packet"):
        QueueState(packet_payloads={"not-a-packet": signal_payload})
    with pytest.raises(QueueStateError, match="does not match packet_id"):
        QueueState(packet_payloads={"wrong-key": record_to_dict(selected.packet)})
    with pytest.raises(QueueStateError, match="current packet snapshot is missing"):
        QueueState(current_packet_id="missing-packet")
    with pytest.raises(QueueStateError, match="previous packet snapshot is missing"):
        QueueState(previous_packet_ids=("missing-packet",))

    real_state = tmp_path / "real-state.json"
    AuditQueue((_candidate("symlink-state"),), state_path=real_state).select_next()
    symlink_state = tmp_path / "symlink-state.json"
    symlink_state.symlink_to(real_state)
    with pytest.raises(QueueStateError, match="must not be a symlink"):
        AuditQueue((_candidate("symlink-state"),), state_path=symlink_state)


def test_invalid_detector_accounting_containers_fail_closed() -> None:
    with pytest.raises(QueueInputError, match="detector_accounting must be a mapping"):
        QueueDataset(
            (_candidate("bad-detector-container"),), accounting={"detector_accounting": []}
        )
    with pytest.raises(QueueInputError, match="entries must be mappings"):
        QueueDataset(
            (_candidate("bad-detector-entry"),),
            accounting={"detector_accounting": {"detector": "bad"}},
        )
    with pytest.raises(QueueInputError, match="must contain detector IDs"):
        QueueDataset(
            (_candidate("bad-detector-id"),),
            accounting={"unavailable_detectors": [1]},
        )


def test_starvation_override_eventually_lifts_lower_band() -> None:
    low = _candidate("starved-low", metadata={"safety_severity": 1.0})
    policy = ActivePolicy(control_enabled=False, max_age=2)
    queue = AuditQueue(
        (low, _candidate("high-0", metadata={"benchmark_config_defect": 1.0})), policy=policy
    )
    assert queue.select_next().packet.primary.execution_id == "high-0"  # type: ignore[union-attr]
    queue.update_dataset(
        (
            low,
            _candidate("high-0", metadata={"benchmark_config_defect": 1.0}),
            _candidate("high-1", metadata={"benchmark_config_defect": 1.0}),
        )
    )
    assert queue.select_next().packet.primary.execution_id == "high-1"  # type: ignore[union-attr]
    queue.update_dataset(
        (
            low,
            _candidate("high-0", metadata={"benchmark_config_defect": 1.0}),
            _candidate("high-1", metadata={"benchmark_config_defect": 1.0}),
            _candidate("high-2", metadata={"benchmark_config_defect": 1.0}),
        )
    )
    result = queue.select_next()
    assert result is not None
    assert result.packet.primary.execution_id == "starved-low"
    assert result.context.selection_kind == "anomaly"


def test_only_human_full_episode_reviews_grant_coverage_credit() -> None:
    control = _candidate(
        "coverage-credit",
        metadata={"control_eligible": True, "ordinary": True},
    )
    reviews = (
        ReviewRecord("detector-review", control.episode_id, "full_episode", author_kind="detector"),
        ReviewRecord("agent-review", control.episode_id, "full_episode", author_kind="agent"),
        ReviewRecord("interval-review", control.episode_id, "interval", author_kind="human"),
    )
    queue = AuditQueue(
        QueueDataset(
            (control,),
            review_records=reviews,
            coverage_deficits=(CoverageDeficit(control.effective_stratum_id, 0, 1, "s", "p"),),
        ),
        policy=ActivePolicy(control_schedule=(0,), max_control_selections=1),
    )
    result = queue.select_next()
    assert result is not None
    assert queue._reviewed_ids() == set()
    assert result.context.selection_kind == "control"


def test_controls_require_an_ordinary_success_or_failure_outcome() -> None:
    candidate = _candidate(
        "nonordinary-control",
        metadata={"control_eligible": True, "ordinary": True},
        outcome="diagnostic-only",
    )
    queue = AuditQueue(
        (candidate,),
        policy=ActivePolicy(control_schedule=(0,), max_control_selections=1),
    )
    assert candidate.control_eligible is False
    result = queue.select_next()
    assert result is not None
    assert result.context.selection_kind == "anomaly"


def test_manual_actions_reject_unknown_episode_ids() -> None:
    queue = AuditQueue((_candidate("known-action"),))
    with pytest.raises(QueueInputError, match="known episode"):
        queue.pin("unknown-episode")
    with pytest.raises(QueueInputError, match="known episode"):
        queue.skip("unknown-episode")
    with pytest.raises(QueueInputError, match="known episode"):
        queue.record_review("unknown-episode")
    with pytest.raises(QueueInputError, match="known episode"):
        queue.more_evidence("need evidence", "unknown-episode")


def test_annotation_id_scalar_is_rejected_without_character_splitting() -> None:
    candidate = _candidate("annotation-scalar")
    queue = AuditQueue((candidate,))
    assert queue.select_next() is not None
    with pytest.raises(QueueInputError, match="annotation_ids"):
        queue.record_review(annotation_ids="annotation-1")
