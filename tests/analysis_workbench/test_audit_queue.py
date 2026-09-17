"""BA-02 queue contracts and offline selection behaviour.

These tests cover the public queue contract: lexicographic priority bands,
explainable active components, detector-independent controls, explicit review
credit, and lossless state recovery.  They intentionally do not run a browser,
agent, simulator, shell command, or network request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_contracts import EpisodeRef, Finding, ReviewRecord, Signal
from robot_sf.analysis_workbench.audit_queue import (
    ActivePolicy,
    AuditQueue,
    CoverageDeficit,
    QueueConflictError,
    QueueDataset,
    QueueInputError,
    QueueOperationConflictError,
    QueuePolicy,
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
        metadata={
            "event_alignment": {"status": "available", "provenance_gate": {"compatible": True}}
        },
        trace=True,
    )
    missing = _candidate("missing", planner_id="social-force", trace=False)
    incompatible = _candidate(
        "incompatible",
        planner_id="mpc",
        metadata={
            "event_alignment": {"status": "available", "provenance_gate": {"compatible": True}}
        },
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


def replace_candidate(candidate, *, source_digest: str):
    from robot_sf.analysis_workbench.audit_queue import QueueCandidate

    return QueueCandidate(
        _episode(
            candidate.episode.execution_id,
            planner_id=candidate.planner_id,
            source_digest=source_digest,
        ),
        signals=candidate.signals,
        metadata=candidate.metadata,
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
