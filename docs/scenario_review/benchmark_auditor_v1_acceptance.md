# Benchmark Auditor V1 acceptance report (canonical)

Status: **BA-05 D1/D2 bounded contracts are accepted and one bounded BA-06
materialization slice is merged; BA-05/#9488, BA-06/#9489, and epic #9483
remain open.** This is an evidence receipt, not a benchmark or scientific
result. Fallback, degraded, unavailable, synthetic, and diagnostic-only paths
are never promoted.

## Current author scope direction (2026-09-25)

The author explicitly declined BA-05's external acceptance gates—the
provider-enforced physical cost cap, GitHub remote-publication guarantees, and
retained failure/control bundle with media and PTS/provenance receipts—and
BA-06 workbench integration until dissertation 0.0.8 work is done. The author
states this is an explicit decline, not an open deferral. The declined work is
what the unattended autonomous service needs; it is not required for
supervised audits. BA-06 workbench integration stays closed behind BA-05 as a
workstream; issue #9489 remains open.

Supervised audit sessions are permitted: use the auditor interactively with a
person or coordinating agent reviewing its findings. The author expects it to
be most useful on corrected dissertation 0.0.8 data. Source: [#9483 author
scope decision, 2026-09-25](https://github.com/ll7/robot_sf_ll7/issues/9483#issuecomment-5832436448).

## Acceptance boundary for this packet

- Acceptance base before this documentation receipt:
  `346673a80e04888746e0b78abcccecffe623de47`.
- This is a docs-only boundary receipt. The acceptance base is intentionally
  stable; it is not a moving claim about the squash merge that carries this
  document.
- BA-05 D1/D2 remain accepted bounded contracts with exact focused and hosted
  receipts recorded in the [dated full report](./benchmark_auditor_v1_acceptance_2026-09-22.md).
- BA-06 PR #9571 exact reviewed head:
  `1aa3bb47bc60771d0511a36a201e1084c2c9c01e`.
- BA-06 PR #9571 guarded squash merge:
  `f231fc2faa729a5c96f732d4d690365882878feb`.
- PR metadata digest:
  `492a32dfe252d660757c700fd9e8e9b7dab95936204ef9fa5ac1fadef0b407f3`.
- Hosted CI run `35688989382`: success; changed coverage
  `106625885543`: success; merge-queue gate `35690704033`: success.
- Focused proof: `174 passed in 83.92 s`, Node runtime contract passed, and
  the full docs/link scan passed (`2123` Markdown files).
- Previous docs receipt PR #9573 exact reviewed head:
  `6b3bc0632027094de5d4728a5f3ac9d13ec8582a`; guarded squash merge:
  `346673a80e04888746e0b78abcccecffe623de47`; hosted CI and merge-queue
  gate passed. Its metadata digest is
  `62b90e7254f6de33ca96f0d5362c5d1416788f6742932b79657f6a65c113935e`.

## Accepted scope and explicit gates

The merged BA-06 slice exposes one revision-bound server-held
`materialize_selected` action through the browser/service seam. It carries no
source/output paths, credentials, provider authority, or benchmark claim.

GitHub's ordinary issue API still does not prove cross-system CAS or
exactly-once delivery. Complete pagination/readback, duplicate-marker
detection, timeout reconciliation, durable outbox replay, and canonical
finding-link CAS remain requirements for a future unattended autonomous
service; the author has declined this external-gate work until dissertation
0.0.8 work is done. The installed App Server still lacks a verified physical
provider compute ceiling: `offline` is provider-free, `local_accounting`
records observed overspend without physical enforcement, and
`strict_provider_ceiling` refuses without a verified finite cap.

The complete real-data/browser workflow, retained media and native-source
proof, live Codex, cancellation/reconnect recovery, and campaign completion
remain unproven. See the [dated full report](./benchmark_auditor_v1_acceptance_2026-09-22.md)
for all six package dispositions, SREV/prerequisite dispositions, the
thirty-nine decision matrix, historical exact-head receipts, and limitations.

## Decision

The bounded BA-05 D1/D2 contracts and one bounded BA-06 materialization slice
remain accepted at the recorded receipt boundary. The full BA-05 and BA-06
packages remain incomplete; #9488, #9489, and epic #9483 remain open. No issue
or receipt in this packet declares either package or the epic complete.

Machine-readable companion: [`benchmark_auditor_v1_acceptance.json`](./benchmark_auditor_v1_acceptance.json).
