# Issue #3275 Rerun Closure Packet — Real Archive Fail-Closed Blocker (2026-07-03)

This is a small, reviewable copy of the consolidated rerun closure packet
produced for a **real** disjoint source/rerun failure-archive attempt. It is
the durable evidence for the issue #3275 closure-packet slice.

## Claim boundary

Consolidated readiness/leakage closure packet only. No benchmark campaign run,
no proposal-model inference, no planner execution, no held-out yield claim, and
no dissertation or paper-facing claim promotion. Evidence tier:
`diagnostic-only` (fail-closed blocker verdict).

## What this shows

The packet consolidates the accumulated issue #3275 rerun readiness gates into a
single fail-closed verdict for a real archive pair:

- source archive: `docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/archive.json` (60 real entries)
- rerun archive: `docs/context/evidence/issue_1501_adversarial_smoke_2026-05-28/archive.json` (15 real entries)

Verdict: `disposition = fail_closed_blocked` (CLI exit code 3). The archives are
real adversarial-search smoke outputs, not synthetic fixtures, and they are
**not** a valid disjoint certified pair. Consolidated blockers:

- `archive_id_overlap:15` — the two archives share archive IDs (`failure_0000`…),
- `scenario_family_overlap:2` and `seed_overlap:8` — overlapping split metadata,
- `source_missing_certification_metadata:60` / `missing_certification_metadata:15`
  — neither archive carries per-entry certification metadata,
- `missing_null_test_prerequisites:1` — no null-test prerequisite report supplied.

Next empirical action (deterministically selected, leakage category dominates):
provide a genuinely disjoint rerun archive with no shared archive IDs, no
duplicate IDs within a side, and no shared `config.source_manifests` lineage.

## Reproduce

```bash
uv run python scripts/adversarial/produce_rerun_closure_packet.py \
  --source-archive docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/archive.json \
  --rerun-archive docs/context/evidence/issue_1501_adversarial_smoke_2026-05-28/archive.json \
  --output output/adversarial/issue_3275_closure/real_archive_closure_packet.json
# exit code 3 == fail_closed_blocked
```

The archive SHA-256 values embedded in the packet
(`pair_readiness.source_archive_sha256`, `pair_readiness.rerun_archive_sha256`)
pin the exact inputs.
