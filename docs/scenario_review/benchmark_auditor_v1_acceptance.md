# Benchmark Auditor V1 acceptance report (canonical)

Status: **BA-05 bounded service acceptance and one BA-06 materialization slice
are merged; BA-06 and epic #9483 remain open.** This is an evidence receipt,
not a benchmark or scientific result. Fallback, degraded, unavailable,
synthetic, and diagnostic-only paths are never promoted.

## Current merged boundary

- Current `origin/main`: `f231fc2faa729a5c96f732d4d690365882878feb`.
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

## Accepted scope and explicit gates

The merged BA-06 slice exposes one revision-bound server-held
`materialize_selected` action through the browser/service seam. It carries no
source/output paths, credentials, provider authority, or benchmark claim.

GitHub's ordinary issue API still does not prove cross-system CAS or
exactly-once delivery. Complete pagination/readback, duplicate-marker
detection, timeout reconciliation, durable outbox replay, and canonical
finding-link CAS remain required. The installed App Server still lacks a
verified physical provider compute ceiling: `offline` is provider-free,
`local_accounting` records observed overspend without physical enforcement,
and `strict_provider_ceiling` refuses without a verified finite cap.

The complete real-data/browser workflow, retained media and native-source
proof, live Codex, cancellation/reconnect recovery, and campaign completion
remain unproven. See the [dated full report](./benchmark_auditor_v1_acceptance_2026-09-22.md)
for all six package dispositions, SREV/prerequisite dispositions, the
thirty-nine decision matrix, historical exact-head receipts, and limitations.

## Decision

BA-05's bounded D1/D2 package is accepted. BA-06 has resumed and one bounded
materialization slice is merged. #9488, #9489, and epic #9483 remain open and
gated; no issue or receipt in this packet declares the epic complete.

Machine-readable companion: [`benchmark_auditor_v1_acceptance.json`](./benchmark_auditor_v1_acceptance.json).
