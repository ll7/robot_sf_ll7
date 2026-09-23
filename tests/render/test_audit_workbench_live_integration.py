"""Cross-service BA-05/BA-06 proof using a retained campaign fixture."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from robot_sf.analysis_workbench.audit_contracts import ActionRecord, Annotation
from robot_sf.analysis_workbench.audit_findings import add_candidate, new_finding
from robot_sf.analysis_workbench.audit_github import GitHubIssue, SearchResult
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    NativeDiagnosticBinding,
    NativeDiagnosticCampaignRowBinding,
    NativeDiagnosticServiceConfig,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import AuditSourceBinding, QueueNextAdapter
from robot_sf.analysis_workbench.review_contracts import (
    component_request_from_dict,
    experiment_recipe_from_dict,
)
from robot_sf.benchmark.runner import run_episode
from robot_sf.render.audit_workbench import ServiceAuditWorkbenchFacade
from robot_sf.render.audit_workbench_launch import open_live_audit_workbench

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


class _LiveGitHubProvider:
    """Minimal append-only provider fake for the HTTP publication proof."""

    def __init__(self) -> None:
        self.issues: list[GitHubIssue] = []
        self.create_calls: list[dict[str, Any]] = []

    def search_issues(self, repository: str, *, marker: str) -> SearchResult:
        del marker
        return SearchResult(
            tuple(issue for issue in self.issues if issue.repository == repository),
            complete=True,
        )

    def create_issue(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        labels: tuple[str, ...],
    ) -> GitHubIssue:
        self.create_calls.append(
            {"repository": repository, "title": title, "body": body, "labels": labels}
        )
        issue = GitHubIssue(
            repository=repository,
            number=len(self.issues) + 1,
            url=f"https://github.com/{repository}/issues/{len(self.issues) + 1}",
            title=title,
            body=body,
            labels=labels,
            updated_at=str(len(self.issues) + 1),
        )
        self.issues.append(issue)
        return issue

    def create_issue_with_finding_revision(
        self,
        repository: str,
        *,
        finding_id: str,
        expected_finding_revision: int,
        title: str,
        body: str,
        labels: tuple[str, ...],
    ) -> GitHubIssue:
        del finding_id, expected_finding_revision
        return self.create_issue(repository, title=title, body=body, labels=labels)


def _browser_source_ref(source_entry: dict[str, object]) -> dict[str, str]:
    """Project a native source identity to the browser's closed SourceRef shape."""

    projected: dict[str, str] = {}
    for field_name in (
        "artifact_id",
        "uri",
        "format",
        "schema",
        "sha256",
        "source_commit",
        "config_identity",
        "units",
        "coordinate_frame",
    ):
        value = source_entry.get(field_name)
        if field_name == "sha256" and not value:
            value = source_entry.get("declared_sha256") or source_entry.get("computed_sha256")
        if isinstance(value, str) and value:
            projected[field_name] = value
    return projected


def _assert_native_annotation_provenance(
    annotation: dict[str, object], source_entry: dict[str, object]
) -> None:
    """Check the durable native visual-source bridge projection."""

    assert annotation["source_ref"]["source_commit"] == ""
    assert annotation["metadata"]["visual_source_identity"] == source_entry["sha256"]
    expected_provenance = {
        "declared_sha256": source_entry["declared_sha256"],
        "computed_sha256": source_entry["computed_sha256"],
        "integrity": source_entry["integrity"],
        "availability": source_entry["availability"],
        "admission": source_entry["admission"],
    }
    if "config_digest" in source_entry:
        expected_provenance["config_digest"] = source_entry["config_digest"]
    assert annotation["metadata"]["visual_source_provenance"] == expected_provenance
    reference = annotation["references"][0]
    assert reference["source"]["source_commit"] == ""
    assert reference["source_revision"]
    binding = annotation["metadata"]["reference_source_bindings"]["scene-robot-1"]
    assert binding["source_ref"]["artifact_id"] == source_entry["artifact_id"]
    assert binding["source_ref"]["uri"] == "<redacted-local-uri>"
    assert binding["source_identity"] == source_entry["sha256"]
    assert binding["source_revision"] == source_entry["source_commit"]
    assert binding["source_provenance"] == expected_provenance


def _native_config_for_scene(
    native_case: dict[str, Any], campaign: Path, campaign_bytes: bytes
) -> NativeDiagnosticServiceConfig:
    """Bind the admitted canonical source bundle to one literal campaign row."""

    from robot_sf.analysis_workbench import audit_native_diagnostic as native

    original = native_case["original"]
    config_identity = native_case["source_document"]["identity"]["config_identity"]
    admission = native.NativeDiagnosticAdmission.from_mapping(native_case["admission_document"])
    request = component_request_from_dict(
        native_case["request_document"], source="live integration"
    )
    recipe = experiment_recipe_from_dict(native_case["recipe_document"], source="live integration")
    source = request.sources[0]
    descriptor = NativeDiagnosticCampaignRowBinding(
        campaign_uri=campaign.name,
        campaign_sha256=hashlib.sha256(campaign_bytes).hexdigest(),
        native_uri=source.uri,
        native_sha256=source.sha256,
        episode_id=original["episode_id"],
        scenario_id=original["scenario_id"],
        seed=original["seed"],
        planner_id=original["algo"],
        source_commit=original["git_hash"],
        config_identity=config_identity,
    )
    return NativeDiagnosticServiceConfig(
        bindings=(
            NativeDiagnosticBinding(
                admission=admission,
                request=request,
                recipe=recipe,
                campaign_row=descriptor,
            ),
        ),
        compute_cost=2.0,
    )


def test_real_queue_next_reads_generated_primary_without_fabricating_media(tmp_path: Path) -> None:
    """BA-02's generated primary ID resolves through BA-05 into BA-06 inspection."""

    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=4, compute_budget=4.0)
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="audit-fixture-campaign"),
            actor="agent",
            actor_id="integration-test",
            policy=policy,
        )
        campaign = service.read_campaign(session)
        assert campaign.value is not None
        episode = campaign.value.episode("fixture-readable")
        assert episode is not None and episode.episode_ref is not None
        bound = service.update_context(
            session,
            session.context.next_revision(source_identity=session.source_digest),
            expected_context_revision=0,
        )
        assert bound.status == "committed"
        dataset = QueueDataset(
            candidates=(episode.episode_ref,),
            campaign_digest=episode.episode_ref.campaign_digest,
            source_digest=session.source_digest,
        )
        binding = AuditSourceBinding(
            "audit-fixture-campaign",
            episode.episode_ref.campaign_digest,
            session.source_digest,
            session.source_revision,
        )
        adapter = QueueNextAdapter(
            binding,
            lambda: AuditQueue(dataset, state_path=tmp_path / "queue.json", store=service.store),
        )
        service.next_adapter = adapter
        service.queue_next_adapter = adapter
        facade = ServiceAuditWorkbenchFacade(service, session, token=session.session_token)
        selected = facade.next(operation_id="cross-service-next")
        assert selected["status"] == "complete", selected.get("reason")
        artifact_status = facade.read_selected_artifact_status()
        assert artifact_status["status"] == "complete"
        assert artifact_status["value"]["materialization"]["status"] == "not_configured"
        assert artifact_status["value"]["native_diagnostic"]["status"] == "not_configured"
        assert artifact_status["value"]["native_diagnostic"]["scientific_claim_allowed"] is False
        assert str(root) not in json.dumps(artifact_status, sort_keys=True)
        assert selected["inspection_status"] == "complete", selected.get("inspection_reason")
        assert selected["packet"]["primary"]["episode_id"] == episode.episode_ref.episode_id
        assert selected["episode"]["episode_id"] == "fixture-readable"
        assert selected["presentation_status"] == "selected"
        assert "scene" not in selected["episode"]
        assert "video" not in selected["episode"]
        assert "editor_model" not in selected["packet"]
    finally:
        service.close()


def test_live_related_cases_accepts_verified_source_row_alias(tmp_path: Path) -> None:
    """A BA-05 similarity row uses its literal ID, while Next selects a generated ID."""

    root = tmp_path / "source"
    root.mkdir()
    source = root / "campaign.json"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    first = payload["episodes"][0]
    peer = {**first, "episode_id": "fixture-peer", "planner_id": "peer-planner", "seed": 8}
    payload["episodes"] = [first, peer]
    source.write_text(json.dumps(payload), encoding="utf-8")
    opened = open_live_audit_workbench(source, source_root=root, store_root=tmp_path / "store")
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urlopen(opened.url, timeout=15) as response:
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        parsed = urlsplit(opened.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        def post(operation: str, arguments: dict[str, object]) -> dict[str, object]:
            request = Request(
                origin + "/api/audit",
                data=json.dumps({"operation": operation, "arguments": arguments}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "Cookie": browser_cookie,
                },
                method="POST",
            )
            with urlopen(request, timeout=15) as response:
                return json.load(response)

        selected = post("next", {"expected_selection_revision": 0, "operation_id": "next-peer"})
        assert selected["status"] == "complete", selected.get("reason")
        generated_id = selected["packet"]["primary"]["episode_id"]
        literal_id = selected["episode"]["episode_id"]
        assert generated_id != literal_id
        related = post(
            "related_cases",
            {
                "episode_id": generated_id,
                "expected_context_revision": selected["context_revision"],
                "mode": "same_scenario_across_planners",
                "operation_id": "related-peer",
            },
        )
        assert related["status"] == "complete", related.get("reason")
        assert related["candidate_count"] == 1
        assert related["value"][0]["query_id"] == generated_id
        assert related["value"][0]["candidate_id"] in {"fixture-readable", "fixture-peer"} - {
            literal_id
        }
        assert related["membership_boundary"] == "candidates_are_unconfirmed"
    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


def test_live_http_publication_uses_canonical_finding_and_append_only_provider(
    tmp_path: Path,
) -> None:
    """The resumed BA-06 slice reaches BA-05 sync without browser authority."""

    root = tmp_path / "source"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    repository = "ll7/robot_sf_ll7"
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        allowed_repositories=(repository,),
        issue_write_budget=1,
    )
    opened = open_live_audit_workbench(
        source,
        source_root=root,
        store_root=tmp_path / "store",
        policy=policy,
    )
    provider = _LiveGitHubProvider()
    opened.service.github_provider = provider
    session = next(iter(opened.service._sessions.values()))
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urlopen(opened.url, timeout=15) as response:
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        parsed = urlsplit(opened.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        def post(operation: str, arguments: dict[str, object]) -> dict[str, object]:
            request = Request(
                origin + "/api/audit",
                data=json.dumps({"operation": operation, "arguments": arguments}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "Cookie": browser_cookie,
                },
                method="POST",
            )
            with urlopen(request, timeout=30) as response:
                return json.load(response)

        selected = post("next", {"expected_selection_revision": 0, "operation_id": "sync-next"})
        assert selected["status"] == "complete", selected.get("reason")
        episode_id = selected["packet"]["primary"]["episode_id"]
        finding = add_candidate(
            new_finding("live-http-finding", "retained source symptom"), episode_id
        )
        commit = opened.service.finding_store.create(
            finding,
            operation_id="live-http-finding-create",
            actor="human",
        )
        source_revision = session.source_revision
        sync = post(
            "sync_finding",
            {
                "finding_id": finding.finding_id,
                "repository": repository,
                "expected_finding_revision": commit.revision,
                "expected_selection_revision": selected["selection_revision"],
                "expected_context_revision": selected["context_revision"],
                "expected_source_revision": source_revision,
                "retry_ambiguous": False,
                "operation_id": "live-http-github-sync",
            },
        )
        assert sync["status"] == "committed", sync
        assert sync["value"]["status"] == "created", sync
        assert len(provider.create_calls) == 1
        assert "robot_sf_audit_finding:v1" in provider.create_calls[0]["body"]
        payload = json.dumps(sync, sort_keys=True)
        assert session.session_token not in payload
        assert str(root) not in payload
        assert sync["evidence_boundary"] == "diagnostic_only"
        assert sync["scientific_claim_allowed"] is False
    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


def test_live_next_projects_only_scanner_admitted_native_scene(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    """A real retained native trace reaches the shared editor without invented media."""

    from robot_sf.analysis_workbench import audit_native_diagnostic as native
    from tests.analysis_workbench import test_audit_native_diagnostic as native_tests

    original = run_episode(
        {"id": "ba06_live_scene", "density": "low", "flow": "uni", "obstacle": "open"},
        42,
        horizon=8,
        dt=0.1,
        robot_start=(-4.0, 0.0),
        robot_goal=(4.0, 0.0),
        record_forces=False,
        algo="simple_policy",
        telemetry={"analysis_trace": "all"},
        provenance={"test": "ba06-live-scene"},
    )
    runner_input = {
        "scenario_params": {
            "id": "ba06_live_scene",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
        },
        "seed": 42,
        "horizon": 8,
        "dt": 0.1,
        "robot_start": [-4.0, 0.0],
        "robot_goal": [4.0, 0.0],
        "algo": "simple_policy",
        "algo_config_path": None,
        "record_forces": False,
        "telemetry": {"analysis_trace": "all"},
    }
    runner_input["environment_identity"] = native._environment_identity_for_runner(runner_input)
    source_identity = {
        "config_identity": native._config_identity_for_runner(runner_input),
        "initial_state_sha256": native._initial_state_digest(runner_input),
        "environment_identity": runner_input["environment_identity"],
    }
    original["provenance"]["native_diagnostic"] = {
        "identity": "historical_original",
        # The source-bundle helper owns this receipt-bound identifier; the BA06
        # campaign row remains the literal join asserted below.
        "source_id": "ba05-native-original-42",
        "config_identity": source_identity["config_identity"],
        "source_identity": source_identity,
        "runner_input_identity": native._runner_input_identity(runner_input),
    }
    original["provenance"]["native_diagnostic"]["source_commit"] = original["git_hash"]
    trace = original["algorithm_metadata"]["analysis_trace"]
    native_case = native_tests._make_case(
        tmp_path / "native-bundle",
        original,
        runner_input_overrides={
            "scenario_params": {
                "id": "ba06_live_scene",
                "density": "low",
                "flow": "uni",
                "obstacle": "open",
            }
        },
    )
    config_identity = native_case["source_document"]["identity"]["config_identity"]
    row = {
        "episode_id": original["episode_id"],
        "scenario_id": "ba06_live_scene",
        "planner_id": "simple_policy",
        "seed": original["seed"],
        "source_commit": original["git_hash"],
        "config_identity": config_identity,
        "config_digest": trace["config_digest"],
        "retained_trace": trace,
        "metrics": {},
        "outcome": {"label": "success"},
    }
    root = tmp_path / "source"
    root.mkdir()
    source = root / "campaign.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "campaign-result.v1",
                "campaign_id": "ba06-live-scene-campaign",
                "source_commit": original["git_hash"],
                "config_identity": config_identity,
                "execution_status": "native",
                "expected_episode_ids": [original["episode_id"]],
                "episodes": [row],
            }
        ),
        encoding="utf-8",
    )
    campaign_bytes = source.read_bytes()
    native_config = _native_config_for_scene(native_case, source, campaign_bytes)
    native_binding = native_config.bindings[0]
    assert native_binding.campaign_row is not None
    policy = SessionPolicy(
        allowed_roots=(str(root), str(native_case["root"])),
        allowed_recipes=(native_binding.recipe.recipe_id,),
        compute_budget=2.0,
    )
    opened = open_live_audit_workbench(
        source,
        source_root=root,
        store_root=tmp_path / "store",
        policy=policy,
        native_diagnostic_config=native_config,
    )
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urlopen(opened.url, timeout=5) as response:
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        parsed = urlsplit(opened.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        request = Request(
            origin + "/api/audit",
            data=json.dumps(
                {
                    "operation": "next",
                    "arguments": {
                        "expected_selection_revision": 0,
                        "operation_id": "native-scene-next",
                    },
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        # The native-scene projection is intentionally an integration path;
        # allow the bounded slow lane under xdist before closing the store.
        with urlopen(request, timeout=30) as response:
            selected = json.load(response)
        assert selected["status"] == "complete" and selected["presentation_status"] == "selected", (
            selected.get("reason")
        )
        packet = selected["packet"]
        assert packet["primary"]["episode_id"] != native_binding.campaign_row.episode_id
        assert selected["context"]["episode_id"] == packet["primary"]["episode_id"]
        assert selected["episode"]["episode_id"] == native_binding.campaign_row.episode_id
        assert "projection_status" in packet, "same-run retained trace was not projected"
        assert packet["projection_status"] == "partial"
        editor = packet["editor_model"]
        assert (
            editor["schema_version"] == "review-editor.v1"
            and editor["panel_model"]["schema_version"] == "review-panels.v1"
        )
        scene = editor["panel_model"]["streams"]["scene"]
        assert [sample["time_s"] for sample in scene["samples"]] == [
            step["time_s"] for step in trace["steps"]
        ]
        assert packet["cursor"]["time_s"] == trace["steps"][0]["time_s"]
        assert editor["panel_model"]["streams"]["video"]["status"] == "unavailable"
        assert editor["panel_model"]["metrics"] == {}
        assert editor["panel_model"]["provenance"]["evidence_status"] == "diagnostic_only"
        assert str(root) not in json.dumps(packet)
        assert str(tmp_path / "store") not in json.dumps(packet)
        source_entry = next(iter(editor["source_identity"]["sources"].values()))
        source_ref = _browser_source_ref(source_entry)
        save_request = Request(
            origin + "/api/audit",
            data=json.dumps(
                {
                    "operation": "save_annotation",
                    "arguments": {
                        "annotation": {
                            "annotation_id": "native-scene-note",
                            "episode_id": packet["primary"]["episode_id"],
                            "classification": "unclear",
                            "mode": "full",
                            "author_kind": "human",
                            "author_id": "local-reviewer",
                            "review_scope": "full_episode",
                            "observed_behavior": "robot traversed the retained scene",
                            "suspected_cause": "planner yielded before clearance recovered",
                            "confidence": 0.8,
                            "evidence": [{"description": "retained actor sample"}],
                            "interval": {
                                "start_s": packet["cursor"]["time_s"],
                                "end_s": packet["cursor"]["time_s"],
                            },
                            "references": [
                                {
                                    "reference_id": "scene-robot-1",
                                    "coordinate_frame": "world",
                                    "point": [0.0, 0.0],
                                    "timestamp_s": packet["cursor"]["time_s"],
                                    "actor_id": "robot",
                                    "source": source_ref,
                                    "source_revision": source_entry.get("source_commit") or 0,
                                }
                            ],
                            "source_ref": source_ref,
                            "source_identity": source_entry["sha256"],
                            "source_revision": source_entry.get("source_commit") or 0,
                            "provenance_status": "verified",
                        },
                        "expected_selection_revision": selected["selection_revision"],
                        "expected_revision": 0,
                        "expected_context_revision": selected["context_revision"],
                        "operation_id": "native-scene-note-save",
                    },
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        with urlopen(save_request, timeout=5) as response:
            saved = json.load(response)
        assert saved["presentation_status"] == "saved", saved
        finding_request = Request(
            origin + "/api/audit",
            data=json.dumps(
                {
                    "operation": "persist_finding",
                    "arguments": {
                        "annotation": None,
                        "expected_selection_revision": selected["selection_revision"],
                        "expected_context_revision": selected["context_revision"],
                        "operation_id": "native-scene-finding-save",
                        "finding_id": "native-scene-finding",
                        "title": "Native scene candidate",
                    },
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        with urlopen(finding_request, timeout=5) as response:
            finding = json.load(response)
        assert finding["presentation_status"] == "saved", finding
        assert finding["finding"]["status"] == "proposed"
        assert finding["finding"]["confirmed_members"] == []
        assert finding["finding"].get("github_issue") is None
        native_request = Request(
            origin + "/api/audit",
            data=json.dumps(
                {
                    "operation": "run_native_diagnostic",
                    "arguments": {
                        "operation_id": "native-scene-control",
                        "expected_selection_revision": selected["selection_revision"],
                        "expected_context_revision": selected["context_revision"],
                        "intervention_id": "goal-y",
                        "robot_goal": [4.0, 4.0],
                        "activation_epsilon_m": 1e-9,
                        "deadline_s": 30.0,
                    },
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        with urlopen(native_request, timeout=60) as response:
            native_result = json.load(response)
        assert native_result["status"] == "complete", native_result
        assert native_result["diagnostic_only"] is True
        assert native_result["scientific_claim_allowed"] is False
        assert native_result["control_fidelity"] == "verified"
        assert native_result["activation"] == "verified"
        for phase in ("before", "after"):
            stored = opened.service.store.get(f"audit-operation-native-scene-control-{phase}")
            assert stored is not None
            assert isinstance(stored.record, ActionRecord)
            assert stored.record.action_type == "diagnostic.native"
            if phase == "after":
                assert stored.record.details["operation_id"] == "native-scene-control"
        after = opened.service.store.get("audit-operation-native-scene-control-after")
        assert after is not None and isinstance(after.record, ActionRecord)
        assert after.record.status == "complete"
        snapshot_request = Request(
            origin + "/api/audit",
            data=json.dumps({"operation": "snapshot", "arguments": {}}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": origin,
                "Cookie": browser_cookie,
            },
            method="POST",
        )
        # A fresh snapshot revalidates queue, coverage, episode, and records
        # through the service.  The full native-scene integration fixture can
        # take over a minute under hosted xdist, so keep this bounded timeout
        # above the observed slow path while still failing a hung server.
        with urlopen(snapshot_request, timeout=90) as response:
            reopened = json.load(response)
        assert reopened["artifact_status"]["materialization"]["status"] == "not_configured"
        assert reopened["artifact_status"]["native_diagnostic"]["status"] == "available"
        assert reopened["artifact_status"]["native_diagnostic"]["scientific_claim_allowed"] is False
        assert reopened["records_result"]["status"] == "complete"
        assert reopened["record_projection"]["authoritative"] is True
        assert [record["record_id"] for record in reopened["annotations"]] == ["native-scene-note"]
        assert [record["record_id"] for record in reopened["findings"]] == ["native-scene-finding"]
        # This retained native-trace proof keeps provider sync disabled; the
        # separate publication test covers the append-only service boundary.
        assert opened.service.github_provider is None
        _assert_native_annotation_provenance(reopened["annotations"][0], source_entry)
    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


@pytest.mark.parametrize("source_revision", [7, 0], ids=["versioned", "digest-only"])
def test_real_service_facade_reopens_ba03_records_before_and_after_reconnect(
    tmp_path: Path, source_revision: int
) -> None:
    """BA-06 reads the BA-03 projection before and after a service restart."""
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if source_revision:
        payload["source_revision"] = source_revision
    source.write_text(json.dumps(payload), encoding="utf-8")
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=12, compute_budget=12.0)
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="integration-test",
        policy=policy,
    )
    session_id = session.session_id
    session_token = session.session_token
    campaign = service.read_campaign(session)
    assert campaign.value is not None
    episode = campaign.value.episode("fixture-readable")
    assert episode is not None and episode.episode_ref is not None
    generated_id = episode.episode_ref.episode_id
    selected = session.context.next_revision(
        episode_id=generated_id,
        source_identity=session.source_digest,
        source_revision=session.source_revision,
    )
    bound = service.update_context(
        session,
        selected,
        expected_context_revision=session.context.context_revision,
    )
    assert bound.status == "committed"
    annotation = Annotation(
        annotation_id=f"annotation-reconnect-{source_revision}",
        episode_id=generated_id,
        classification="unclear",
        author_kind="agent",
        author_id="integration-test",
        source_revision=source_revision,
        source_identity=session.source_digest,
    )
    assert (
        service.write_annotation(
            session,
            annotation,
            context=session.context,
            expected_source_revision=source_revision,
            operation_id=f"annotation-reconnect-{source_revision}",
        ).status
        == "committed"
    )
    assert (
        service.finding_from_annotation(
            session,
            annotation,
            finding_id=f"finding-reconnect-{source_revision}",
            title="candidate finding",
            context=session.context,
            operation_id=f"finding-reconnect-{source_revision}",
        ).status
        == "committed"
    )
    facade = ServiceAuditWorkbenchFacade(service, session, token=session_token)
    before = facade.read_saved_records(operation_id=f"read-before-{source_revision}")
    assert before["status"] == "complete", before.get("reason")
    assert [item["record_id"] for item in before["records"]] == [
        f"annotation-reconnect-{source_revision}",
        f"finding-reconnect-{source_revision}",
    ]
    assert before["records"][0]["author_kind"] == "agent"
    assert before["records"][0]["source_identity"] == session.source_digest
    assert str(before["records"][0]["source_revision"]) == str(source_revision)
    assert before["records"][1]["status"] == "proposed"
    assert str(before["records"][1]["source_revision"]) == str(source_revision)
    assert before["records"][1]["confirmed_members"] == []
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        reopened_facade = ServiceAuditWorkbenchFacade(reopened, reconnected, token=session_token)
        after = reopened_facade.read_saved_records(
            episode_id=generated_id,
            operation_id=f"read-after-{source_revision}",
        )
        assert after["status"] == "complete", after.get("reason")
        assert [item["record_id"] for item in after["records"]] == [
            f"annotation-reconnect-{source_revision}",
            f"finding-reconnect-{source_revision}",
        ]
        snapshot = reopened_facade.snapshot()
        assert snapshot["records_result"]["status"] == "complete"
        assert snapshot["record_projection"]["authoritative"] is True
        assert snapshot["record_projection"]["fresh_process"] == "reconnectable"
        assert [item["record_id"] for item in snapshot["annotations"]] == [
            f"annotation-reconnect-{source_revision}"
        ]
        assert [item["record_id"] for item in snapshot["findings"]] == [
            f"finding-reconnect-{source_revision}"
        ]
        assert session_token not in json.dumps(snapshot, sort_keys=True)
    finally:
        reopened.close()


def test_real_service_facade_saved_record_read_rejects_wrong_context_and_source(
    tmp_path: Path,
) -> None:
    """BA-06 preserves BA-05 unavailable/conflict status for unbound reads."""
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    source.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=8, compute_budget=8.0)
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="audit-fixture-campaign"),
            actor="agent",
            actor_id="integration-test",
            policy=policy,
        )
        campaign = service.read_campaign(session)
        assert campaign.value is not None
        episode = campaign.value.episode("fixture-readable")
        assert episode is not None and episode.episode_ref is not None
        generated_id = episode.episode_ref.episode_id
        selected = session.context.next_revision(
            episode_id=generated_id,
            source_identity=session.source_digest,
        )
        assert (
            service.update_context(
                session,
                selected,
                expected_context_revision=session.context.context_revision,
            ).status
            == "committed"
        )
        foreign = Annotation(
            annotation_id="annotation-foreign-source-facade",
            episode_id=generated_id,
            classification="unclear",
            source_revision=session.source_revision,
            source_identity="foreign-source",
        )
        service.store.save(
            foreign,
            operation_id="seed-foreign-source-facade",
            actor="human",
            actor_id="fixture",
        )
        facade = ServiceAuditWorkbenchFacade(service, session, token=session.session_token)
        wrong_context = session.context.next_revision(source_identity="foreign-source")
        wrong = facade.read_saved_records(
            episode_id=generated_id,
            context=wrong_context,
            operation_id="read-wrong-source-context",
        )
        assert wrong["status"] in {"conflict", "unavailable"}
        assert "records" not in wrong
        valid = facade.read_saved_records(
            episode_id=generated_id,
            operation_id="read-valid-source-context",
        )
        assert valid["status"] == "complete", valid.get("reason")
        assert valid["records"] == []
    finally:
        service.close()
