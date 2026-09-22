# Benchmark Auditor BA-05 / BA-06 acceptance report (2026-09-22)

Status: BA-05's bounded D1/D2 slices are accepted and merged. BA-06's
server-held publication slice is implemented, hosted-green, and merged with
an injected-provider diagnostic proof. The full BA-06 workflow and epic #9483
remain open.

## Evidence boundary

These are diagnostic integration and contract results, not benchmark or
scientific evidence. No fallback, degraded, unavailable, or synthetic result
is promoted to a benchmark claim. The live publication proof uses a fake
append-only provider behind the real `AuditService` and loopback HTTP route.

## Exact-head and hosted-CI receipts

| Slice | PR / exact head | Merge | Focused proof | Hosted proof |
|---|---|---|---|---|
| BA-05 D1 append-only publication | #9559 / `ec8061c4517fb3926fefd1b1a2cb6cb3cdcf0088` | `6787f5b93a1023dd243f66d81b1e6a13fe6871a3` | 102 passed | run `35655668527`, attempt 2: full matrix, CodeQL, changed coverage and merge queue green |
| BA-05 D2 route/accounting | #9558 / `78fb3c818fafe1ae0ea1522c96092e55b8460e3b` | `6982a2456aa98cabff6f5186d396d4f15eb72b48` | 113 passed, 2 skipped | run `35658827492`: full matrix, CodeQL, changed coverage and merge queue green |
| BA-06 publication boundary | #9561 / `e2c84dd8ea75bd63a62005ecbc55ae7847d06328` | `208e8a3bf02f5b2c2b3eafe39fe776da07a9dcc7` | 149 passed | run `35667772524` green; changed coverage job `106560755969` and merge-queue-gate run `35669539946` green |

D1's first hosted attempt had one transient macOS/Matplotlib native-launch
failure while the font cache was unavailable; the guarded failed-job rerun
passed. The accepted receipt is attempt 2, not the transient attempt.

## Post-merge CI reconciliation

The hosted receipts above are exact PR-head acceptance runs. They do not imply
that the later merge commits were independently hosted-green. The repository
records the later runs explicitly:

- D1 merge `6787f5b93a1023dd243f66d81b1e6a13fe6871a`: push run
  `35658730273` was cancelled after supersession; it is not a focused failure.
- D2 merge `6982a2456aa98cabff6f5186d396d4f15eb72b48`: workflow-dispatch run
  `35668217774` was cancelled and reported base-sensitive `ci`/macOS failures;
  its optional-import and #7330/#7331 inventory drift is not used as D2
  acceptance evidence.
- BA-06 merge `208e8a3bf02f5b2c2b3eafe39fe776da07a9dcc7`: push run
  `35669627895` failed in `fast-feedback (2)` on optional-import and
  #7330/#7331 inventory drift and in `fast-feedback (4)` on the timing-sensitive
  `tests/render/test_review_sessions.py::test_start_and_stop_race_serializes_the_inactive_owner_branch`.
  The failed logs did not identify a BA-06 focused-test failure, but the merged
  SHA is still not hosted-green and is not promoted as such.

This reconciliation preserves the valid exact-head focused/hosted receipts
while making the post-merge evidence boundary fail closed. A future clean
merge-SHA run can replace this limitation; until then, BA-06 remains
diagnostic-only and the epic remains open.

## BA-06 bounded proof

The loopback route now admits a closed `sync_finding` operation. It accepts
only repository, finding identity, canonical finding revision, selection and
context CAS, source revision, retry flag, and opaque operation ID. The facade
passes the server-held session context and token to BA-05; the browser cannot
supply provider credentials, paths, evidence overrides, or service context.
Focused tests cover keyword-only service mapping, stale selection/context,
closed arguments, authority redaction, no-provider `unavailable`, and a real
`AuditService` route that creates one marker-bearing issue through an injected
provider. The response is explicitly `diagnostic_only` with
`scientific_claim_allowed: false`.

The native retained-trace integration remains diagnostic-only. It exercises
queue, source-bound scene projection, annotations, findings, and native
control semantics, but does not claim full media, provider, Codex, MCP, or
recovery acceptance.

## Explicit unresolved gates

- GitHub does not provide the required cross-system CAS/atomic linkage. The
  accepted contract uses a local durable outbox, immutable initial issue, and
  append-only marker-bearing comments. Incomplete pagination, ambiguous
  remote outcomes, duplicates, human conflicts, and local link CAS conflicts
  remain visible and block unsafe automatic retry. This is not an exactly-once
  remote-delivery claim.
- The installed Codex App Server does not prove a provider-enforced token or
  compute ceiling. `local_accounting` is a finite local reservation/ledger,
  not a physical provider cap. `strict_provider_ceiling` must refuse admission
  without a verified finite ceiling; hosted evidence must report the selected
  mode and physical-cap verification separately.
- Local full-workbench collection is environment-limited by missing optional
  `imageio_ffmpeg` (#9560). The affected native-trace import is not treated as
  success locally; hosted CI is the authoritative compatibility proof.

## Remaining BA-06 / epic gates

The merged publication boundary is only one BA-06 slice. The issue still
needs the complete real-data browser workflow, external MCP/Codex same-context
smoke, reconnect/cancellation/recovery evidence, and retained media/source
coverage.
SREV #9285/#9287/#9288 are integrated at contract/test level; #9296/#9299 and
prerequisite #9417 still need the admitted-source/live proof recorded by the
epic acceptance criteria. Epic #9483 remains open and is not complete.
