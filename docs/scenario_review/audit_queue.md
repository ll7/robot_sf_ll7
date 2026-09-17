# Benchmark-auditor review queue (BA-02)

BA-02 chooses an explainable next review from a conforming campaign.  It is an
offline policy layer around the closed BA-03 `ReviewPacket` contract.  Ranking
components, coverage state, random state, and selection provenance are kept in
the separate `QueueSelectionContext`; `selection_reasons` remains a tuple of
human-readable strings.

## Offline launch

The checked-in fixture can be inspected without a browser, network, live agent,
shell command, simulator, or experiment runner:

```bash
uv run python -m robot_sf.analysis_workbench.audit_queue \
  rank --input tests/fixtures/analysis_workbench/audit_queue_v1/queue.json \
  --policy active --limit 4
```

`select` emits one `ReviewPacket` and its structured selection context.  Add
`--state output/audit-queue/state.json` to `select` and `resume` for durable,
lossless local state.  State is strict JSON, atomically replaced, and guarded
by a monotonic compare-and-swap revision under a stable sibling lock
(`state.json.lock`).  It contains the RNG state, validated packet snapshots,
selection history, defer/pin/request actions, and input/policy identities;
there is no pickle or executable payload.  Concurrent writers fail closed on
a stale revision, and corrupt current or historical snapshots cannot be
silently replaced by a new selection.

The Python API is equivalent:

```python
from robot_sf.analysis_workbench.audit_queue import AuditQueue, load_queue_input

dataset = load_queue_input("tests/fixtures/analysis_workbench/audit_queue_v1/queue.json")
queue = AuditQueue(dataset)  # active policy is the default
selected = queue.select_next()
assert selected is not None
print(selected.packet.primary.episode_id)
print(selected.context.priority_band, selected.context.components)
```

## Policy and explanations

Both `QueuePolicy.fixed()` and `QueuePolicy.active()` are deterministic for
fixed inputs.  Active mode is the default.  The accepted priority bands are
lexicographic and are applied before any within-band weights:

1. benchmark/configuration defect;
2. release/manuscript impact;
3. unexplained behaviour;
4. statistical unusualness;
5. planner disagreement;
6. safety severity.

The resulting score is an uncalibrated priority, never a probability, defect
prevalence estimate, or scientific-exemplar score.  Each ranked candidate
exposes the band, score, component values, weighted contributions, starvation
status, and reason strings.  Active components include coverage gain,
competing-hypothesis evidence, unresolved requests, novelty, redundancy,
presentation age, and defer pressure.  Repeated symptoms are penalized but
remain eligible; new geometry, planner, outcome, contradictory evidence, or a
request for more evidence can restore priority.

The control stream is separate from anomaly ranking.  A candidate must declare
`control_eligible` (or `ordinary`) and a stable observed stratum.  The sampler
chooses randomly within an under-reviewed `CoverageDeficit` stratum, regardless
of detector flags, and persists its `random.Random` state.  A finite schedule
of selection positions prevents a high anomaly volume from starving controls;
it is not a permanent 70/20/10 quota and cannot support an unbiased prevalence
claim.  `CoverageDeficit` records only the BA-04 input contract locally:
stratum ID, observed/target counts, source/protocol revisions, status, and
reason.  BA-02 does not calculate coverage.

The fixed policy leaves controls disabled by default so its score/order remains
anomaly-only; callers can opt into the same finite control schedule explicitly.

## Packets, peers, and missing producers

The primary `EpisodeRef` retains campaign, source, execution, configuration,
checkpoint, environment, scenario, planner, seed, and attempt identity.  Peer
episodes are included only when the BA-03 compatibility helper and an available
`event_alignment`, `review_alignment`, or pair-compatibility gate agree.  A
missing trace or incompatible source disables that peer and records
missingness; it never blocks single-episode primary review.  Similarity is a
retrieval explanation only: it does not imply review, confirmation, a matched
state, a common cause, or a scientific exemplar.

Peer matched-state display also requires the canonical
`pair_compatibility.deterministic.v1` profile, complete provenance checks,
source-trace content receipt/identity, and the full initial-state equivalence
receipt bound to both trace IDs.  A caller's `compatible: true` flag or a
minimal alignment mapping is insufficient.

The queue consumes typed BA-01 `Signal` records and a stable `ScanSummary`
identity/revision/accounting handle.  Missing BA-01 signals/scan summaries and
missing BA-04 deficits are reported explicitly as unavailable.  Detector-level
unavailable/error accounting and coverage/scan source ID or revision
mismatches remain visible in packet missingness.  The offline root document
must carry the exact `audit-queue-input.v1` schema version.  A changed
scan/finding/input/policy-semantic revision is surfaced as stale on resume;
prior selection history is retained and must not be silently rebound.

## Actions and review credit

`pin`, `skip`, `defer`, `previous_packet`, `resume`, and `more_evidence` are
available on `AuditQueue`.  If an `AuditStore` is supplied, each action is
written with `AuditStore.save` using an operation ID and compare-and-swap
revision.  Selected packets are emitted as BA-03 records.  Selection actions
are agent-authored with `human_review_granted: false`; selection never writes a
`ReviewRecord`.  Call `record_review(scope="full_episode", ...)` only for an
explicit human/agent review receipt; only a human full-episode receipt grants
coverage credit.  Replaying an operation is idempotent;
reusing it with a different payload or saving against a stale queue revision
fails closed.

BA-03 remains the annotation backup/restore owner.  Use its canonical
`AuditStore.export`/`AuditStore.restore` for annotation history; queue state is
an additional local JSON snapshot and does not replace the audit journal.

## Capability boundary

This leaf performs bounded local ranking and state transitions only.  It does
not launch autonomous diagnostics, consume Codex credentials, execute a
planner, materialize missing media, synchronize GitHub findings, or enforce
agent token/issue-write budgets.  Those operations belong to later BA service
issues.  A future service may wrap this API with a finite policy and
cancellation envelope, but source files, annotations, and queue input remain
data—not authorization or shell instructions.  No Codex setup is needed for
the offline route; use the existing local Codex/App Server integration only
when the later service slice is present.
