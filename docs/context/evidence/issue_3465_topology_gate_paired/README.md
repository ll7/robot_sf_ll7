# Issue #3465 — Near-Parity Promotion Gate Decision Report

**Decision Status:** `blocked`
**Verdict:** `blocked`

## Rationale

Campaign data is not available or checks failed.

## Acceptance Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Paired enabled/disabled config exists and is reproducible. | `met` | PR #4246 added configs/benchmarks/issue_3465_topology_gate_paired.yaml and the fail-closed preregistration validator. |
| Same scenario/seed/planner contract is used across arms. | `met` | PR #4246 validates topology_gate_disabled and topology_gate_enabled differ only by near_parity_diversity_gate_enabled while preserving planner, scenario, seed, and horizon pairing. |
| Corrective #3463 gate completed or waived before paired interpretation. | `met` | PR #4465 integrated corrective packets #4388, #4411, #4426, #4435, and #4444 into the readiness contract and reports ready_for_paired_run. |
| Degraded or fallback rows are excluded or caveated. | `met` | PR #4487 added the paired decision builder, excludes fallback/degraded arms from promotion evidence, and defaults missing paired_significant to False in real mode. |
| Result classification is recorded conservatively. | `met` | PR #3602 added near_parity_promotion_gate.v1; PR #4597 added this decision packet. Missing campaign evidence is classified as blocked instead of promoted, and real campaign summaries are classified by the same fail-closed report builder. |

## Blockers

- Campaign summary JSON is missing; provide a durable paired campaign summary before promotion classification.
