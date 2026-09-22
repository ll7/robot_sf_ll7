"""Acceptance proof for a repository-retained real Benchmark Auditor source."""

from __future__ import annotations

from pathlib import Path

from robot_sf.analysis_workbench.audit_contracts import Annotation
from robot_sf.analysis_workbench.audit_coverage import evaluate_coverage
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset, ScanSummary
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import (
    AuditSourceBinding,
    CoverageReadAdapter,
    QueueNextAdapter,
)
from robot_sf.render.audit_workbench import ServiceAuditWorkbenchFacade

RETAINED_CAMPAIGN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmark"
    / "scenario_flakiness_issue_4978"
    / "real_campaign_episodes.jsonl"
)
FAILURE_ID = "classic_doorway_low--112--7633cbd1b58e0be6"
CONTROL_ID = "classic_doorway_low--114--7e1e0bb0f634327f"


def test_retained_real_campaign_reaches_queue_and_fails_closed_without_native_bundle(
    tmp_path: Path,
) -> None:
    """Real rows reach durable review while absent native inputs remain unavailable."""

    report = scan_campaign(RETAINED_CAMPAIGN, root=RETAINED_CAMPAIGN.parent)
    inventory = {item.episode_id: item for item in report.inventory}
    references = {item.execution_id: item for item in report.episode_refs}
    assert report.counts["coverage"]["readable"] == 400
    assert report.audit.source is not None
    assert report.audit.source.config_identity == ""
    assert inventory[FAILURE_ID].row is not None
    assert inventory[CONTROL_ID].row is not None
    assert inventory[FAILURE_ID].row["metrics"] == {"success": False, "collisions": 1}
    assert inventory[CONTROL_ID].row["metrics"] == {"success": True, "collisions": 0}
    assert references[FAILURE_ID].config_digest != references[CONTROL_ID].config_digest
    assert references[FAILURE_ID].checkpoint_digest == ""
    assert references[CONTROL_ID].checkpoint_digest == ""

    service = AuditService(
        tmp_path / "store",
        campaign_source=RETAINED_CAMPAIGN,
        source_root=RETAINED_CAMPAIGN.parent,
    )
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="campaign"),
            actor="agent",
            actor_id="retained-real-test",
            policy=SessionPolicy(
                allowed_roots=(str(RETAINED_CAMPAIGN.parent),), compute_budget=2.0
            ),
        )
        bound = service.update_context(
            session,
            session.context.next_revision(source_identity=session.source_digest),
            expected_context_revision=0,
        )
        assert bound.status == "committed"

        candidate_ids = {FAILURE_ID, CONTROL_ID}
        dataset = QueueDataset(
            candidates=(references[FAILURE_ID], references[CONTROL_ID]),
            signals=tuple(
                signal for signal in report.signals if signal.episode_id in candidate_ids
            ),
            campaign_digest=report.audit.campaign_digest,
            source_digest=session.source_digest,
            protocol_version=report.audit.protocol_version,
            protocol_digest=report.audit.protocol_digest,
            scan_summary=ScanSummary.from_scan_report(report),
        )
        binding = AuditSourceBinding(
            "campaign",
            report.audit.campaign_digest,
            session.source_digest,
            session.source_revision,
        )
        queue_adapter = QueueNextAdapter(
            binding,
            lambda: AuditQueue(dataset, state_path=tmp_path / "queue.json", store=service.store),
        )
        service.next_adapter = queue_adapter
        service.queue_next_adapter = queue_adapter
        service.coverage_adapter = CoverageReadAdapter(
            binding,
            lambda: evaluate_coverage(scan=report, review_records=()),
            report_source_revision=report.audit.source.source_commit,
        )

        facade = ServiceAuditWorkbenchFacade(service, session, token=session.session_token)
        selected = facade.next(operation_id="retained-real-next")
        assert selected["status"] == "complete", selected.get("reason")
        assert selected["episode"]["episode_id"] == FAILURE_ID
        assert selected["packet"]["primary"]["execution_id"] == FAILURE_ID
        assert len(dataset.candidates) == 2

        artifact_status = facade.read_selected_artifact_status()
        assert artifact_status["value"]["native_diagnostic"]["status"] == "not_configured"
        assert artifact_status["value"]["native_diagnostic"]["scientific_claim_allowed"] is False
        native = facade.run_native_diagnostic(
            operation_id="retained-real-native-unavailable",
            intervention_id="goal-y",
            robot_goal=[1.0, 1.0],
        )
        assert native["status"] == "unavailable"
        assert "binding" in native["reason"]
        assert native["evidence_boundary"] == "diagnostic_only"
        assert native["scientific_claim_allowed"] is False

        selected_episode_id = selected["context"]["episode_id"]
        annotation = Annotation(
            annotation_id="retained-real-note",
            episode_id=selected_episode_id,
            classification="unclear",
            author_kind="agent",
            author_id="retained-real-test",
            source_revision=session.source_revision,
            source_identity=session.source_digest,
        )
        saved = service.write_annotation(
            session,
            annotation,
            context=session.context,
            expected_source_revision=session.source_revision,
            operation_id="retained-real-note-save",
        )
        assert saved.status == "committed", saved.reason
        finding = service.finding_from_annotation(
            session,
            annotation,
            finding_id="retained-real-finding",
            title="Retained real campaign candidate",
            context=session.context,
            operation_id="retained-real-finding-save",
        )
        assert finding.status == "committed", finding.reason
        persisted = service.store.get("retained-real-finding")
        assert persisted is not None
        assert persisted.record.status == "proposed"
        assert persisted.record.confirmed_members == ()

        coverage = facade.read_coverage(operation_id="retained-real-coverage")
        assert coverage["status"] == "complete", coverage.get("reason")
        coverage_report = coverage["value"]["value"]
        assert coverage_report["status"] == "incomplete"
        assert coverage_report["counts"]["coverage"]["readable"] == 400
        assert coverage_report["counts"]["diagnostics"]["executed"] == 0
    finally:
        service.close()
