"""Coherence guard for the AMV command-response trace acquisition issue (#2000).

Issue #2000 asks to collect a *real* AMV command-response trace so the AMV actuation envelope can be
hardware-calibrated. The maintainer decision on #2000 (2026-06-22) hard-blocks implementation at
<5% feasibility until real platform access / controller logs / ROS bags exist, and the issue's
agent-execution spec names the only agent-executable action:

    "Keep the acquisition path documented and the downstream consumer ready: the manifest/validator
     that will ingest the trace is #2415; the proxy fallback already in use is #1585. Ensure those
     two stay coherent."

with acceptance criteria:

    - "Issue remains explicitly blocked with the exact access requirement named."
    - "No synthetic/proxy trace is ever promoted as hardware-calibrated evidence."

The trace-manifest *mechanism* is owned by issue #2415
(:mod:`robot_sf.research.amv_command_response_trace_manifest` and the shipped manifest
``configs/research/amv_command_response_trace_manifest_issue_2415.yaml``). This module does not fork
that owner; it adds the missing **regression guard** that operationalizes #2000's acceptance
criteria so future edits cannot silently:

1. drop the #2000 (acquisition) / #1585 (proxy) cross-reference from the shipped consumer manifest,
2. flip the still-blocked acquisition into a "ready"/"staged" state without a real source, or
3. let the manifest preflight imply a calibrated/hardware realism claim.

The tests use both the shipped manifest (coherence) and synthetic manifests (the blocked
external-data state and required calibrated-trace field classes), matching the issue's validation
hint. They never ingest a trace, calibrate a value, or assert hardware realism.
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.synthetic_actuation import actuation_variability_fields
from robot_sf.research.amv_command_response_trace_manifest import (
    AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY,
    MANIFEST_STATUS_BLOCKED_EXTERNAL,
    MANIFEST_STATUS_READY,
    check_amv_trace_manifest,
    load_amv_trace_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_MANIFEST_PATH = (
    REPO_ROOT / "configs" / "research" / "amv_command_response_trace_manifest_issue_2415.yaml"
)

# Acquisition issue (#2000) and the proxy-fallback issue (#1585) that the consumer manifest must
# keep cross-referenced so the acquisition path stays documented and coherent.
ACQUISITION_ISSUE = 2000
PROXY_ISSUE = 1585

ALLOWED_TARGETS = set(actuation_variability_fields())


def _shipped_report():
    """Load and check the shipped #2415 consumer manifest with the live envelope vocabulary."""
    manifest = load_amv_trace_manifest(SHIPPED_MANIFEST_PATH)
    return manifest, check_amv_trace_manifest(manifest, allowed_calibration_targets=ALLOWED_TARGETS)


def test_shipped_manifest_cross_references_acquisition_and_proxy_issues() -> None:
    """The consumer manifest must name #2000 (acquisition) and #1585 (proxy) as blockers.

    This is the concrete "keep #2415 and #1585 coherent" guard from the #2000 agent-exec spec:
    if someone edits the consumer manifest and drops either cross-reference, the acquisition path
    silently loses its documented link and this test fails.
    """
    manifest, _ = _shipped_report()
    referenced_blockers = {
        issue for trace in manifest["traces"] for issue in trace.get("blocker_issues", [])
    }
    assert ACQUISITION_ISSUE in referenced_blockers, (
        "shipped #2415 manifest must keep #2000 (real-trace acquisition) as a named blocker issue"
    )
    assert PROXY_ISSUE in referenced_blockers, (
        "shipped #2415 manifest must keep #1585 (proxy fallback) as a named blocker issue"
    )


def test_shipped_manifest_sources_from_acquisition_issue() -> None:
    """At least one trace must point its provenance at the #2000 acquisition source thread."""
    manifest, _ = _shipped_report()
    source_urls = [
        str(trace.get("provenance", {}).get("source_url", "")) for trace in manifest["traces"]
    ]
    assert any(f"/issues/{ACQUISITION_ISSUE}" in url for url in source_urls), (
        "a shipped trace must source from the #2000 acquisition issue so provenance stays traceable"
    )


def test_shipped_manifest_stays_blocked_external_input() -> None:
    """The acquisition is still hard-blocked: the manifest must not be calibration-ready today.

    Operationalizes #2000 acceptance "issue remains explicitly blocked" and
    "no synthetic/proxy trace is ever promoted as hardware-calibrated evidence": with no real
    bundle staged, the consumer manifest must report blocked-external-input and forbid ingest.
    """
    _, report = _shipped_report()
    assert report.manifest_status == MANIFEST_STATUS_BLOCKED_EXTERNAL
    assert report.calibration_ingest_allowed is False
    assert report.calibration_ready_traces == []


def test_shipped_manifest_is_manifest_only_evidence_boundary() -> None:
    """A passing preflight must stay a manifest-only contract, never a hardware-calibrated claim."""
    _, report = _shipped_report()
    assert report.evidence_boundary == AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY
    # The boundary string itself must keep asserting no-ingest / no-calibration / no-claim.
    for marker in ("no_trace_ingest", "no_calibration_run", "no_calibrated_claim"):
        assert marker in AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY


def test_shipped_manifest_declares_required_calibrated_trace_field_classes() -> None:
    """The acquisition trace must declare the command / response / timing classes #2000 names.

    A real command-response trace usable for an actuation envelope needs at least one command
    channel, one measured-response channel, and a timing base that includes a timestamp, a
    command-to-response latency, and a sampling/update rate. This guards that the #2000 raw-signal
    requirements stay represented in the consumer manifest without inventing any values.
    """
    manifest, _ = _shipped_report()
    primary = manifest["traces"][0]

    assert primary["command_channels"], "trace must declare at least one command channel"
    assert primary["response_channels"], "trace must declare at least one measured-response channel"

    timing = " ".join(str(field).lower() for field in primary["timing_fields"])
    assert "timestamp" in timing, "timing fields must include a timestamp/time base"
    assert "latency" in timing, "timing fields must include a command-to-response latency"
    assert "rate" in timing or "period" in timing, (
        "timing fields must include a sampling/update rate or controller update period"
    )


def test_blocked_acquisition_trace_is_never_calibration_ready() -> None:
    """A synthetic blocked-external-input trace must not unlock calibration (fail-closed).

    Mirrors the steady state of #2000: a trace that names #2000/#1585 as blockers and is not staged
    must report blocked, never ready -- the core "no synthetic promoted as calibrated" guarantee.
    """
    trace = {
        "trace_id": "amv_command_response_primary",
        "asset_id": "amv-calibration",
        "title": "Real AMV command-response actuation trace bundle",
        "staging_status": "blocked-external-input",
        "redistribution": "none",
        "blocker_issues": [ACQUISITION_ISSUE, PROXY_ISSUE],
        "provenance": {
            "source_url": f"https://github.com/ll7/robot_sf_ll7/issues/{ACQUISITION_ISSUE}",
            "license": "Access depends on source; private traces stay local",
            "license_status": "unknown",
            "citation": "Real AMV command-response trace source to be identified via #2000/#1585.",
        },
        "command_channels": ["linear_velocity_command_mps"],
        "response_channels": ["measured_linear_velocity_mps"],
        "timing_fields": ["timestamp_s", "control_latency_s", "sample_rate_hz"],
        "calibration_targets": ["max_linear_accel_m_s2"],
    }
    manifest = {
        "schema_version": "amv_command_response_trace_manifest.v1",
        "manifest_id": "issue_2000_coherence_test",
        "issue": ACQUISITION_ISSUE,
        "claim_boundary": "staging manifest metadata only",
        "calibration_asset_id": "amv-calibration",
        "synthetic_envelope_module": "robot_sf/benchmark/synthetic_actuation.py",
        "traces": [trace],
    }
    report = check_amv_trace_manifest(manifest, allowed_calibration_targets=ALLOWED_TARGETS)
    assert report.manifest_status == MANIFEST_STATUS_BLOCKED_EXTERNAL
    assert report.manifest_status != MANIFEST_STATUS_READY
    assert report.calibration_ingest_allowed is False
    assert report.calibration_ready_traces == []
