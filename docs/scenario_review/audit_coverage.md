# Release-bound audit coverage

BA-04 provides a deterministic health report for the Benchmark Auditor. It
answers what was expected, what was indexed and detector-evaluable, and where
human review or finding disposition is still missing. It is an operational
coverage report—not a benchmark-validity badge, confidence percentage, or
claim that unknown bugs are absent.

## Python entry point

```python
from robot_sf.analysis_workbench import (
    AuditIdentity,
    DetectorRegistry,
    DetectorSpec,
    ReviewRecord,
    default_audit_protocol,
    evaluate_coverage,
)

detector_registry = DetectorRegistry(
    detectors=(DetectorSpec("telemetry", "example", "example typed detector"),)
)
protocol = default_audit_protocol()

identity = AuditIdentity(
    campaign_digest="campaign-release-2026-09",
    source_digest="campaign-source-sha256",
    release_digest="release-0.0.7",
    protocol_version=protocol.version,
    protocol_digest=protocol.digest,
    detector_registry_digest="detector-registry-digest",
    detector_config_digest="detector-config-digest",
)
report = evaluate_coverage(
    expected_rows=[
        {
            "episode_id": "episode-1",
            "planner_id": "ppo",
            "scenario_group": "corridor",
            "outcome": "success",
        }
    ],
    detector_attempts=[
        {"episode_id": "episode-1", "detector_id": "telemetry", "status": "clear"}
    ],
    detector_registry=detector_registry,
    protocol=protocol,
    review_records=[
        ReviewRecord(
            review_id="review-1",
            episode_id="episode-1",
            scope="full_episode",
            author_kind="human",
        )
    ],
    identity=identity,
)
print(report.status)
print(report.to_json())
```

The shipped `audit-protocol.v1` defaults require every expected row to be
accounted for, a typed result for every scheduled detector attempt, one full
human review per observed planner × declared scenario-group × outcome stratum,
an ordinary detector-independent control where one exists, representative and
different-context/control review for high-priority anomaly clusters, and an
explicit disposition for critical integrity findings.

## Identity and denominator rules

Every report binds campaign, source, release, protocol, detector-registry,
detector-configuration, scan, review, and receipt revisions when available.
The release/source/protocol identity is included in the completion receipt;
`receipt_matches_identity` therefore rejects a receipt from a changed release,
source, detector policy, or protocol. Missing identity components are visible as
an error deficit rather than silently completing.

Completion receipts use the closed `audit-completion-receipt.v1` envelope: all
identity fields and a 64-character report digest are required. The digest is an
unkeyed integrity check, not a signature or proof of trusted report production;
admission code must retain and validate the referenced report as well.

BA-01 `AuditScanReport` inventory and `Signal` values are accepted directly.
For the authoritative human denominator, pass BA-03 typed `ReviewRecord`
values. A compact mapping earns credit only when it is a complete BA-03 record
envelope and successfully deserializes through `record_from_dict`; a mapping
that merely contains `episode_id`, `author_kind`, and `scope` is rejected.
Only `author_kind="human"` with `scope="full_episode"` earns full human
credit. Agent records, interval records, one-click/clip annotations, replays,
duplicate rows, similarity membership, and regenerated executions do not
inflate original-episode coverage.

Detector attempts likewise require a typed BA-01 `DetectorRegistry` (or a
typed `AuditScanReport` carrying one). Attempts outside the active readable
episode × registry schedule are recorded as errors and never count as clear or
flagged evidence. If the registry is unavailable, the report remains
incomplete with an explicit missing-registry deficit.

The versioned report retains the typed readable-episode × detector schedule and
attempts, ordinary-control candidate/reviewed IDs, materialization rows, and
finding representative/context-control IDs. Validation recomputes each gate
from those episode-level records before admitting a complete status; aggregate
counts or booleans alone cannot be promoted by deleting a deficit. These
records provide internal consistency evidence only: the report digest remains
an unkeyed integrity token, not authenticated evaluator provenance.

Materialization rows must name an episode from the readable expected inventory.
Foreign episode IDs and anonymous rows are retained as unavailable/unbound
evidence with an explicit deficit, so they cannot satisfy completion. Rows are
canonically ordered by episode identity and status; no positional synthetic ID
is created. The same binding applies when materialization is supplied through a
typed source-scan summary.

When a source or scan revision is active, a BA-03 `ReviewRecord.source_revision`
must carry the active revision token. Reviews with unavailable or stale
revision provenance remain visible but cannot enter the human denominator.

Missing, unavailable, invalid, duplicate, unsupported, and error states remain
in the report. A declared exception is a closed, non-empty
requirement/reason/status envelope and is retained as exactly one matching
`CoverageDeficit` with `status="waived"`; unmatched or duplicate declarations
are discarded during evaluation and cannot produce an exception-bearing
completion status. No exception is inferred from a timestamp or the existence
of an artifact.

## Outputs and downstream handoff

`AuditHealthReport.to_dict()` and `to_json()` emit `audit-coverage.v1`; exports
are deep copies, so mutating a nested returned mapping cannot change the frozen
report or leave its digest stale.
`AuditHealthReport.from_dict()` and `validate_audit_coverage()` require the
complete nested schema, finite JSON values, matching identity/protocol/report
digests, and status/deficit consistency; a payload cannot change its status to
complete while retaining an empty identity or an unwaived deficit.
`report.deficits` uses the BA-02-compatible `CoverageDeficit` shape, with
additional release/source/detector identity fields and reason codes. The
`deficit.to_ba02_dict()` adapter emits exactly the nine BA-02 fields: its
`source_revision` is the scan revision (when available), `protocol_revision`
is the protocol version, `source_id` is the source digest, and `protocol_id` is
the protocol digest. An unavailable scan revision remains empty so BA-02 can
surface missing provenance rather than treating a source digest as a scan
revision. The adapter maps an explicitly waived BA-04 gap to BA-02's `met`
control status while retaining the reason in the handoff. The
existing SREV report renderer is used through a narrow adapter:
`report.to_markdown()` and `report.to_html()` produce readable health reports
without creating a second coverage or report journal.

The statuses have intentionally narrow meanings:

- `incomplete`: at least one unwaived accounting, detector, review, finding,
  materialization, or identity deficit remains;
- `complete_with_declared_exceptions`: all remaining gaps are explicitly
  waived and visible;
- `complete_under_protocol`: all configured targets are met with no exception.

These statuses describe implementation/audit coverage only. They do not admit
scientific evidence or alter released metrics.
