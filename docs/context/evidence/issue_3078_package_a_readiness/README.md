# Package A decision packet (issue #3078)

Durable materialization of the fail-closed Package A decision packet, assembled
from the retained/tracked evidence surfaces available in this repository on the
execution date below. This directory is the `durable_evidence.plan.path`
declared by `configs/benchmarks/issue_3078_package_a_readiness.yaml`.

- **Issue:** [#3078](https://github.com/ll7/robot_sf_ll7/issues/3078)
- **Parent:** [#3057](https://github.com/ll7/robot_sf_ll7/issues/3057)
- **Package:** A — seed/planner-rank stability + held-out scenario-family transfer
- **Execution date:** 2026-07-05
- **Regeneration host:** `auxme-imech039`
- **Issue state at execution:** `state:ready`
- **Classification (this slice):** `blocked` — pending a canonical Package A
  campaign result store (see below).

## Plain-language summary

Package A needs three evidence surfaces before its result can be classified as
benchmark evidence: (1) a held-out-family **partition manifest**, (2) a
**seed-sufficiency analysis** derived from the frozen Package A protocol
campaign, and (3) a **canonical campaign result store** (episode rows +
summary + claim card). This slice runs the shipped fail-closed checkers against
the repository as it stands today and records the verdict as durable evidence.

Result: readiness is `ready` and the partition manifest validates, but **no
frozen Package A protocol campaign has been run and retained**, so neither a
seed-sufficiency analysis nor a canonical result store exists. The decision
packet therefore classifies Package A as
`blocked_pending_package_a_evidence`. Producing the missing surfaces requires a
held-out-family benchmark campaign run, which is **out of scope for this
assembly lane** (no compute submission, no campaign execution).

This is a fail-closed decision-packet state, not a benchmark, diagnostic, or
paper-facing claim. No ranking is interpreted and no claim boundary is moved.

## Evidence surfaces checked

| Surface | Command | Result |
| --- | --- | --- |
| Package A readiness manifest | `check_package_a_readiness.py --json` | `ready` (no missing paths, no issues) |
| Held-out-family partition manifest | `validate_heldout_transfer_partitions.py …partitions.yaml` | valid (exit 0) |
| Seed-sufficiency CLI surface | `analyze_seed_sufficiency.py --help` | present |
| Decision packet | `check_package_a_readiness.py --decision-packet --json …` | `blocked_pending_package_a_evidence` |
| Issue #3078 result classification | derived by the checker from the packet status | `blocked` |

## Acceptance-Criteria Audit

`package_a_decision_packet.json` now carries an `acceptance_criteria` array so
closure audits can compare the issue checklist against retained evidence without
promoting missing results.

| Issue #3078 criterion | Current status | Evidence | Remaining work |
| --- | --- | --- | --- |
| Runs seed-sufficiency analysis on retained campaign bundles. | `blocked` | none retained | Produce a `seed_sufficiency_analysis.v1` report from the frozen Package A protocol. |
| Runs or validates held-out-family pilot rows, leakage audit, and transfer-delta outputs. | `blocked` | Held-out-family partition manifest validates: `configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml` | Produce the canonical Package A result store from the frozen held-out-family run. |
| Produces baseline table, seed-stability figure, transfer-delta figure, and preliminary claim card. | `blocked` | preliminary blocked claim card retained; no campaign-derived table/figures retained | Render tables and figures from the canonical result store and seed analysis, then retain the reviewed Package A claim card. |
| Classifies result as benchmark, diagnostic, negative, null, invalid, or blocked durable evidence. | `satisfied` | `issue_result_classification=blocked` in `package_a_decision_packet.json` | none for the blocked state; a non-blocked classification requires the missing evidence above. |

The decision packet now emits the issue #3078 six-way result classification
directly (`issue_result_classification`), mapped mechanically from the internal
packet status by `build_decision_packet`. Previously the `blocked` verdict was
hand-copied into the claim card; the checker and claim card now share one
derived source of truth. The checker only ever emits the two conservative
members `blocked` (evidence missing/invalid) or `diagnostic` (all surfaces
validate, claim-card review still required); the interpretive members
`benchmark` / `negative` / `null` / `invalid` stay reserved for claim-card
review of an actual campaign result and are never auto-promoted here.

See [`reproduction.md`](reproduction.md) for the exact commands and captured
output, and [`package_a_decision_packet.json`](package_a_decision_packet.json)
for the machine-readable verdict.

## Why blocked (fail-closed reasons)

From `package_a_decision_packet.json`:

1. `no canonical campaign result store supplied` — no result store
   (`episodes.parquet` + `summary.json` + `analysis.json` + `claim_card.yaml`)
   exists anywhere in the tree; one can only be produced by running the frozen
   held-out-family campaign.
2. `no seed-sufficiency analysis report supplied` — no
   `seed_sufficiency_analysis.v1` report from the frozen Package A protocol
   exists.

### On the retained seed bundles

Retained bundles that contain `reports/seed_variability_by_scenario.json` do
exist (for example `docs/context/evidence/issue_1454_stage_a_fixed_h100_2026-05-22`
and `docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28`).
They are **not** associated with the frozen Package A protocol
(`configs/benchmarks/issue_3059_research_engine_suite_v0.yaml`) — they are older
cross-kinematics / candidate-scan campaigns. Per the maintainer implementation
plan (issue #3078, 2026-07-02), bundles not associated with the frozen protocol
must not be substituted for the Package A result store. This slice therefore
does **not** manufacture a Package A seed analysis from them; doing so would
misrepresent provenance. They remain available as diagnostic inputs only.

## Next empirical action (out of scope here)

1. Run the frozen held-out-family transfer pilot to produce a canonical result
   store (`run_camera_ready_benchmark.py --config
   configs/benchmarks/issue_2128_heldout_family_transfer_pilot.yaml --mode run`).
   Requires compute authorization.
2. Run `analyze_seed_sufficiency.py` on the retained frozen-protocol campaign
   roots to produce `seed_sufficiency_analysis.v1`.
3. Render the transfer tables/figures with the merged deterministic renderer
   `scripts/tools/build_package_a_transfer_report.py` (fail-closed; keeps
   fallback rows visible), which landed in PR #4262.
4. Re-run the decision packet with `--result-store` and
   `--seed-analysis-report`; it should then reach `diagnostic_review_ready`,
   after which the claim card can move to `pending_review`.

This ordering matches the maintainer gate update on issue #3078
(2026-07-03): "Remaining for #3078: actual Package A campaign / result-store +
seed-sufficiency inputs before any real (non-blocked/non-diagnostic) transfer
classification."

## Boundary

Local `output/` paths are not durable evidence. The compact artifacts in this
directory (decision packet JSON, claim card, reproduction log, checksums) are
the durable record. No paper-facing planner-superiority or generalization claim
is promoted. Fallback/degraded/failed/unavailable rows would remain visible
exclusions if and when a result store is produced.
