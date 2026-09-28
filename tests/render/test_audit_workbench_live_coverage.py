"""Live BA-04 coverage reads through the BA-05/BA-06 loopback route."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from robot_sf.analysis_workbench.audit_contracts import Annotation, ReviewRecord
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_store import AuditStore
from robot_sf.render.audit_workbench_launch import open_live_audit_workbench

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


def test_live_next_launch_snapshot_recomputes_coverage_from_durable_reviews_only(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    """A browser Next route mounts a launch snapshot sourced from BA-03 only."""

    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    scan = scan_campaign(campaign, root=root)
    readable_row = next(item for item in scan.inventory if item.status == "readable")
    episode_id = readable_row.row["episode_id"]
    store_root = tmp_path / "store"
    seed_store = AuditStore(store_root)
    seed_store.save(
        Annotation(
            annotation_id="quick-note-does-not-review",
            episode_id=episode_id,
            classification="unclear",
            mode="quick",
            author_kind="human",
            author_id="local-reviewer",
            review_scope="interval",
            source_revision=0,
            source_identity=scan.audit.source_digest,
        ),
        operation_id="coverage-quick-note",
        actor="human",
        actor_id="local-reviewer",
    )
    seed_store.save(
        ReviewRecord(
            review_id="stale-full-human-receipt",
            episode_id=episode_id,
            scope="full_episode",
            author_kind="human",
            author_id="local-reviewer",
            source_revision=0,
        ),
        operation_id="coverage-stale-review",
        actor="human",
        actor_id="local-reviewer",
    )
    seed_store.close()
    opened = open_live_audit_workbench(
        campaign,
        source_root=root,
        store_root=store_root,
    )
    server_thread = threading.Thread(target=opened.server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urlopen(opened.url, timeout=30) as response:
            html = response.read().decode("utf-8")
            browser_cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        assert "Benchmark audit workbench" in html
        assert "session_token" not in html
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
            # Coverage recomputation is not this contract's latency assertion;
            # track its slow path separately in #9526.
            with urlopen(request, timeout=90) as response:
                return json.load(response)

        selected = post(
            "next",
            {"expected_selection_revision": 0, "operation_id": "coverage-next"},
        )
        assert selected["status"] == "complete", selected.get("reason")
        assert selected["presentation_status"] == "selected"

        launch_model = opened.document["extensions"]["audit_workbench"]["model"]
        launch_report = _coverage_report(launch_model["service_snapshot"])
        assert launch_report["status"] == "incomplete"
        assert launch_report["identity"]["release_digest"] == ""
        assert launch_report["identity"]["source_revision"] == scan.audit.source.source_commit
        assert launch_report["identity"]["scan_revision"] == scan.cache_key
        assert (
            str(selected["context"]["source_revision"])
            != launch_report["identity"]["source_revision"]
        )
        assert launch_report["identity"]["source_digest"] == selected["context"]["source_identity"]
        assert launch_report["counts"]["reviews"]["stale_or_mismatched"] == 1
        assert launch_report["counts"]["reviews"]["full_episode_human"] == 0
        assert launch_report["counts"]["reviews"]["full_human_ids"] == []
        assert any(
            deficit["reason"] == "missing_full_human_review"
            for deficit in launch_report["deficits"]
        )

        reviewed = post(
            "record_human_review",
            {
                "outcome": "pass",
                "expected_selection_revision": selected["selection_revision"],
                "expected_context_revision": selected["context_revision"],
                "expected_queue_state_revision": selected["queue_state_revision"],
                "expected_queue_input_revision": selected["queue_input_revision"],
                "operation_id": "explicit-browser-full-review",
            },
        )
        assert reviewed["status"] == "committed", reviewed.get("reason")
        assert reviewed["value"]["scope"] == "full_episode"
        assert reviewed["value"]["episode_id"] == selected["packet"]["episode_id"]
        assert (
            _coverage_report({"coverage_result": reviewed["coverage_result"]})["counts"]["reviews"][
                "full_episode_human"
            ]
            == 1
        )
        with pytest.raises(HTTPError) as missing_cas:
            post("record_human_review", {"outcome": "pass", "operation_id": "missing-cas"})
        assert missing_cas.value.code == 400
        with pytest.raises(HTTPError) as invalid_outcome:
            post(
                "record_human_review",
                {
                    "outcome": "auto_complete",
                    "expected_selection_revision": selected["selection_revision"],
                    "expected_context_revision": selected["context_revision"],
                    "expected_queue_state_revision": selected["queue_state_revision"],
                    "expected_queue_input_revision": selected["queue_input_revision"],
                    "operation_id": "invalid-outcome",
                },
            )
        assert invalid_outcome.value.code == 400
        for index, malformed_outcome in enumerate(([], {}, None)):
            with pytest.raises(HTTPError) as malformed:
                post(
                    "record_human_review",
                    {
                        "outcome": malformed_outcome,
                        "expected_selection_revision": selected["selection_revision"],
                        "expected_context_revision": selected["context_revision"],
                        "expected_queue_state_revision": selected["queue_state_revision"],
                        "expected_queue_input_revision": selected["queue_input_revision"],
                        "operation_id": f"malformed-outcome-{index}",
                    },
                )
            assert malformed.value.code == 400

        refreshed = post("snapshot", {})
        assert refreshed["queue_state_revision"] > selected["queue_state_revision"], refreshed[
            "queue_result"
        ]
        assert _coverage_report(refreshed)["counts"]["reviews"]["full_episode_human"] == 1
        advanced = post(
            "next",
            {
                "expected_selection_revision": selected["selection_revision"],
                "expected_context_revision": selected["context_revision"],
                "expected_queue_state_revision": refreshed["queue_state_revision"],
                "expected_queue_input_revision": refreshed["queue_input_revision"],
                "operation_id": "next-after-explicit-review",
            },
        )
        assert advanced["status"] == "complete", advanced.get("reason")
        assert advanced["selection_revision"] > selected["selection_revision"]
        assert advanced["queue_state_revision"] > refreshed["queue_state_revision"]

    finally:
        opened.server.shutdown()
        server_thread.join(timeout=5)
        opened.close()


def _coverage_report(snapshot: dict[str, object]) -> dict[str, object]:
    """Extract the authoritative BA-04 report from one service snapshot."""

    result = snapshot["coverage_result"]
    assert result["status"] == "complete", result.get("reason")
    capability = result["value"]
    assert capability["capability"] == "ba-04.coverage"
    assert capability["status"] == "complete"
    report = capability["value"]
    assert report["schema_version"] == "audit-coverage.v1"
    return report


def test_live_coverage_projects_unique_queue_review_without_rewriting_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    campaign = root / "campaign.json"
    shutil.copyfile(FIXTURE, campaign)
    scan = scan_campaign(campaign, root=root)
    queue_episode_id = scan.episode_refs[0].episode_id
    receipt = ReviewRecord(
        review_id="typed-queue-human-review",
        episode_id=queue_episode_id,
        scope="full_episode",
        author_kind="human",
        author_id="local-reviewer",
        source_identity=scan.audit.source.source_commit,
        scan_identity=scan.cache_key,
    )
    store_root = tmp_path / "store"
    store = AuditStore(store_root)
    store.save(
        receipt,
        operation_id="typed-queue-human-review-write",
        actor="human",
        actor_id="local-reviewer",
    )
    store.close()

    opened = open_live_audit_workbench(campaign, source_root=root, store_root=store_root)
    try:
        report = _coverage_report(
            opened.document["extensions"]["audit_workbench"]["model"]["service_snapshot"]
        )
        assert report["counts"]["reviews"]["full_episode_human"] == 1
        assert report["counts"]["reviews"]["full_human_ids"] == ["fixture-readable"]
        persisted = opened.service.store.get(receipt.review_id)
        assert persisted.record.episode_id == queue_episode_id
    finally:
        opened.close()
