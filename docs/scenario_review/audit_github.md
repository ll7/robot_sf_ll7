# Finding-level GitHub synchronization

This note describes the append-only GitHub route, which preserves the initial issue body, and what
the offline tests do not prove. It covers the bounded Benchmark Auditor issue 5 (BA-05)
GitHub synchronization slice. The renderer/outbox core and authenticated
`AuditService.sync_finding` route through the Model Context Protocol (MCP) are implemented and
tested offline.
No live GitHub client or external issue write is claimed by this slice.

## What the slice provides

`robot_sf.analysis_workbench.audit_github` renders a typed `Finding` into:

- one stable marker: `robot_sf_audit_finding:v1`;
- one delimited auditor-owned body block;
- separate candidate, confirmed, and negative-control counts; and
- source-bound campaign, source identity, revision/digest, representative-case, finding/request
  evidence, diagnostic, and reproduction fields.

`GitHubOutbox` stores requests as `ActionRecord` rows in the canonical `AuditStore` journal. The
durable key is `(repository, finding_id, operation_id)`. Reusing an operation ID with a different
rendered request is a conflict, not a new issue operation. A second finding-wide claim record
serializes different operation IDs and client processes before any issue create or update. Terminal
outbox success and claim success are committed as one canonical batch; replay repairs a legacy or
crash-interrupted entry whose claim is still in flight.

`GitHubSync` depends on the small `GitHubProvider` protocol. A provider must search by the exact
marker before creating an issue and return complete issue snapshots. Existing-issue updates require
`update_issue_if_unchanged`, implemented by a live adapter with a provider-side compare-and-swap
(CAS) via an entity tag (ETag) or equivalent conditional request. A provider without that CAS/ETag
seam is rejected before a body mutation. When a
canonical `FindingStore` is supplied, the provider must additionally implement
`create_issue_with_finding_revision` and `update_issue_with_finding_revision`; these are an explicit
service/provider reservation boundary for the expected canonical revision. A generic provider is
rejected before remote mutation, so the module does not claim to close a canonical revision
time-of-check-to-time-of-use (TOCTOU) race on behalf of a live adapter. The supplied tests use a
fake provider only; they do not require
credentials or mutate GitHub.

The REST adapter's callable `create_issue_with_finding_revision` is an explicit unsupported
capability: it raises before constructing an HTTP request. `GitHubSync` records that pre-write
boundary as `unavailable` with `remote_write: none` and a failed outbox entry. The authenticated
service releases the reserved issue-write unit, so its durable issue-write usage is unchanged.
This classification applies only while no mutating `POST` has begun. Once a provider begins a
`POST`, a timeout or unreadable outcome remains `ambiguous`, retains the ambiguous outbox/claim
state, and consumes the reserved issue-write unit even when a later read is incomplete. A known
existing canonical issue still follows the append-only comment path; the create-capability guard
does not disable revision comments.

`AuditService.sync_finding` is the service-owned entry point. It accepts a finding ID rather than a
caller-owned finding object, reads the canonical `FindingStore` revision, authenticates the session
token, checks the exact selection/source context and repository allowlist, and reserves the finite
session issue-write budget before invoking `GitHubSync`. The service and GitHub outbox share the
same `AuditStore`; no second finding or synchronization state machine is introduced. Every call
has a durable service receipt. Reusing an operation ID replays that receipt without a provider
call. An explicit `retry_ambiguous=True` uses a separate service receipt while reusing the same
GitHub outbox operation ID for marker reconciliation.

The closed `AuditMCPDispatcher` operation and stdio tool are both named `sync_finding`. They accept
the finding ID, repository, expected canonical finding revision, optional source-bound evidence,
and the explicit ambiguous-retry flag. Provider construction remains server-owned; MCP payloads
cannot nominate a client or credentials.

### Append-only GitHub publication protocol

When the injected provider exposes `append_auditor_comment`, the route keeps the initial issue
immutable and publishes later finding revisions as comments. The initial issue is identified by the
finding-wide `(repository, finding_id, initial_issue)` identity. A current-schema marker discovered
without a local receipt is adopted only after its stable finding marker, initial-publication marker,
and immutable auditor block validate. If it matches the current initial snapshot, recovery reports
`unchanged` without a synthetic comment; if it is an earlier snapshot, recovery reads the complete
comment collection and reconciles or appends the current revision without rewriting the issue.
Legacy or forged publication markers remain conflicts until their provenance is independently
established.

Each revision comment has a semantic `(repository, finding_id, revision, publication_digest)` key,
an exact request marker, and an exact publication marker. Recovery reads the complete comment
collection before posting. If the outbox was lost but the exact same-revision comment is present,
the provider fake route reconciles it without a duplicate POST. An ambiguous comment remains
ambiguous unless the caller explicitly sets `retry_ambiguous=True`.

When the canonical finding link points to that current revision comment, recovery validates the
linked issue and exact comment ID, digest, and body before adopting an older immutable initial
snapshot. The mismatched local create intent is retained as a conflict receipt; the observed issue
snapshot is stored separately under its own semantic key. This preserves the outbox's immutable
request-digest contract while allowing a later revision to reuse the verified initial issue.

The authenticated service response is a bounded projection: issue/comment identities and SHA-256
body digests are returned, while complete bodies remain in the local outbox and direct typed sync
result. This avoids duplicating large snapshots in the MCP/service envelope without weakening the
durable readback record.

The canonical `finding.github_issue` link is validated and reread before create. A link/search or
link/snapshot disagreement is a visible conflict. Every adopted or reconciled publication checks
the issue identity, finding marker, expected revision, auditor-block digest, publication provenance,
comment ID, and exact comment body. If a remote issue or comment disappears after local success,
the local receipt is retained and downgraded to a conflict; it is never silently erased.

## Safety and conflict rules

1. A create or update is never attempted before an exact-marker search whose pagination is complete.
   An expected canonical finding revision is checked before the provider is allowed to mutate. A
   canonical-linked write without the provider reservation seam fails closed rather than creating
   an orphan issue after a revision race.
2. A transport failure during create is recorded as `ambiguous`. Reopening the outbox searches
   again; an incomplete search or an unconfirmed marker never authorizes a blind retry.
3. More than one matching marker, a malformed marker, marker-shaped finding text, a
   repository/finding mismatch, or a changed auditor-owned block is retained as a visible conflict
   (user text is escaped before a create).
4. Updates send only `body`; title, labels, comments, and state are not replaced, and a provider
   response changing those unrelated fields is rejected. The issue is re-read immediately before
   an update; a body edit observed after search is preserved when a prior block digest is known,
   otherwise it is a conflict. The first update attempt durably records the observed block digest
   even when no finding link exists, so an ambiguous retry cannot overwrite a human block edit. The
   provider CAS/ETag write closes the remaining race. A success-then-timeout update is reconciled
   by observing the requested block before retrying, including human text outside that block. A
   create reconciliation similarly requires the requested auditor block and re-reads before
   terminal success; a later marker discovery with no prior digest conflicts on a changed block
   before replacement. Reread transport uncertainty is durable `ambiguous`; malformed or
   mismatched rereads are durable `conflict`. No operation closes or reopens an issue.
5. Configured private roots and generic credential-shaped values, including nested evidence values
   under keys such as `token`, `credential`, `oauth_token`, `api_key`, `accessToken`, `apiKey`,
   `APIKey`, and `oauthToken`, are redacted. Local paths, loopback/private network URLs, and
   unauthorized media hosts are rejected as public media evidence.
6. When a `FindingStore` adapter is supplied, the remote identity is linked back to the canonical
   finding with its own compare-and-swap write. Without that adapter, the result is a transport
   receipt plus a returned link candidate; callers must persist that candidate before describing it
   as a durable finding link.

The remote issue body and comments are data. They are never interpreted as authorization or
executable instructions.

## Live-provider boundary

The service accepts an injected provider-neutral seam for tests and a future authenticated adapter;
it does not construct credentials, read environment tokens, or make network calls. A live adapter
must provide complete marker and comment pagination/readback, authenticated repository scope,
canonical revision reservation, and explicit handling for provider transport uncertainty. Mutable
issue-body routes still require a provider-side CAS/ETag (or equivalent atomic reservation); a
read-then-PATCH sequence is not a CAS. The append-only fake-provider protocol does not establish
provider-side exactly-once delivery, cross-system CAS, or physical provider write caps. Those live
CAS/exactly-once/provider-cap gates remain unresolved and are not claimed by this slice. No live
GitHub application programming interface (API) write or mutation-based capability probe is part of
the offline validation below.

## Offline fake-provider route

The exercised route is the focused test module:

```text
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/analysis_workbench/test_audit_github_append_only.py \
  tests/analysis_workbench/test_audit_github.py \
  tests/analysis_workbench/test_audit_github_service.py -q
```

The append-only fake-provider tests cover same-snapshot and older-snapshot immutable-issue adoption,
canonical-link readback and link/search conflicts, same-revision outbox-loss deduplication without a
canonical link, create-reservation enforcement when a canonical `FindingStore` is present, refusal to
reuse a nonterminal old initial payload for a changed finding, exact publication provenance and
comment-body validation, explicit ambiguous-comment retry, and remote issue/comment deletion with
the local receipt retained as conflict. The general fake-provider tests
cover success-then-timeout reconciliation, block-only updates with human text and labels preserved,
human-edit conflicts after search, concurrent-marker conflict, finding-wide claim recovery,
stale-operation rollback protection, private-path/secret/media filtering, malformed responses, and
outbox request identity. A later live adapter must separately prove authenticated repository scope,
pagination completeness, provider CAS/ETag behavior, canonical revision reservation, write-budget
admission, and origin/session-token checks in `audit_service.py`; none of that is established by
these offline fakes.

## Evidence and missing artifacts

Pass `FindingEvidence` (or its mapping form) with the exact campaign/source identities, source
revision and digest, representative cases, source-bound reproduction commands, diagnostics, and
authorized artifact references. A local artifact URL is not a remotely accessible evidence link;
preserve it in an authorized artifact store first or omit it. Missing campaign/source identity is
rendered as `unavailable` and must not be treated as reproducible proof. Unsupported or divergent
materialization remains outside this module.

## Recovery and integration boundary

The outbox is the local recovery point. On process interruption, reopen the same `AuditStore` path
and reuse the same operation ID. A current-schema marker-discovered initial issue is validated and
adopted without a synthetic comment when it matches the current initial snapshot; an older immutable
snapshot proceeds through exact comment readback. A crash-left canonical claim is repaired locally.
A same-revision comment found after outbox loss is reconciled from its exact markers and body without
a duplicate POST. An `ambiguous` comment requires a complete reread and explicit
`retry_ambiguous=True` before another POST. If a previously successful issue or comment is deleted
or cannot be reread, the durable receipt is retained and reported as `conflict`; the protocol does
not erase local accounting intent or silently replace the remote publication. An older ambiguous
request cannot take over a different newer succeeded request for the same finding. If a nonterminal
initial-create receipt belongs to a different immutable finding snapshot and a complete marker search
finds no issue, the service remains `ambiguous` and refuses to send a new body under the old receipt;
reconciliation must establish whether that original create applied first.
An initial-publication receipt already marked `conflict` remains a conflict when a complete marker
search finds no issue; ordinary replay never authorizes a replacement create from that terminal
receipt. A validated existing issue may still be reconciled without replacing it.
The optional `FindingStore` adapter binds the issue URL/number, marker, source revision, and
auditor-block digest to the current finding revision.

The session service still owns actor/session policy, aggregate token/compute/issue-write budgets,
kill-switch/cancellation, credentials, and operation before/after receipts. Codex/MCP setup must
use the repository's existing authenticated client and routing resolver. This slice intentionally
does not guess provider/model IDs, add credentials, add a live network client, materialize missing
media, run diagnostics, or provide a browser/CLI launch entrypoint.
