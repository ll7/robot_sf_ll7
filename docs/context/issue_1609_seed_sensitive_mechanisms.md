# Issue #1609 Seed-Sensitive Scenario Mechanisms

Date: 2026-05-31

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1609>
- <https://github.com/ll7/robot_sf_ll7/issues/1608>

Evidence:

- [Issue #1608 seed sensitivity analysis](issue_1608_seed_sensitivity_analysis.md)
- [Issue #1609 mechanism summary](evidence/issue_1609_seed_mechanisms_2026-05-31/mechanism_synthesis_summary.json)
- [Issue #1609 mechanism table](evidence/issue_1609_seed_mechanisms_2026-05-31/mechanism_synthesis_table.csv)
- [Issue #1454 S10/h500 candidate bundle](evidence/issue_1454_s10_h500_candidates_2026-05-23/)

## Scope

Issue #1608 identified 25 seed-sensitive scenarios from the durable compact Issue #1454 S10/h500
candidate bundle. This note provides the first mechanism-level interpretation layer over that
classification. It compares hard seeds against easier seeds using the tracked episode rows for the
same top-four planners selected by Issue #1608.

This is still diagnostic synthesis. It prioritizes trace review and scenario inspection; it does
not prove causal mechanisms, alter benchmark definitions, or create paper-facing significance
claims.

## Method

For every Issue #1608 `seed_sensitive` scenario, the derived table computes top-four-planner
averages over:

- hard seeds reported by Issue #1608,
- a sample of easier seeds,
- success,
- collision,
- near-miss count,
- normalized time-to-goal,
- and the most brittle planner named by the seed-sensitivity analyzer.

Mechanism labels are conservative scenario-family hypotheses:

| Mechanism hypothesis | Scenario cue | Status |
|---|---|---|
| `crossing_order_timing` | cross traps, crossings, intersections, perpendicular traffic | aggregate-supported, trace-limited |
| `narrow_passage_timing` | doorway and narrow hallway cases | aggregate-supported, trace-limited |
| `opposing_flow_commitment` | head-on corridor and bottleneck style cases | aggregate-supported, trace-limited |
| `group_cohesion_or_gap_selection` | group crossing or joining-group cases | aggregate-supported, trace-limited |
| `merge_gap_timing` | merging cases | aggregate-supported, trace-limited |
| `limited_lookahead_corner_timing` | blind-corner case | aggregate-supported, trace-limited |
| `artifact_limited_unresolved` | cases where aggregate metrics do not name a clear mechanism | unresolved until trace review |

## Findings

The 25 seed-sensitive scenarios fall into these mechanism buckets:

| Mechanism hypothesis | Count |
|---|---:|
| `crossing_order_timing` | 11 |
| `narrow_passage_timing` | 4 |
| `artifact_limited_unresolved` | 3 |
| `group_cohesion_or_gap_selection` | 2 |
| `merge_gap_timing` | 2 |
| `opposing_flow_commitment` | 2 |
| `limited_lookahead_corner_timing` | 1 |

The clearest aggregate-supported mechanisms are:

- `classic_cross_trap_high` and `classic_cross_trap_medium`: hard seeds have `0.0000` mean success
  and `1.0000` mean collision across the selected top planners, while easy seeds have `1.0000`
  mean success and no collisions. The likely mechanism is crossing-order timing, but trace review is
  still required to distinguish pedestrian phase from planner commitment.
- `classic_doorway_high` and `classic_doorway_medium`: hard seeds show much higher near-miss
  pressure and low success compared with easy seeds, consistent with narrow-passage timing and
  local gap-selection sensitivity.
- `classic_head_on_corridor_low` and `classic_head_on_corridor_medium`: seed `116` is a hard seed
  with `0.0000` mean success and `1.0000` mean collision, consistent with opposing-flow commitment
  sensitivity. Issue #1878 now provides one durable head-on replay determinism fixture for a related
  route, but it is fixed-seed replay evidence rather than broad seed-mechanism proof.
- `francis2023_join_group`: hard seeds `112 113 115 116` have `0.0000` mean success and high
  near-miss pressure relative to easier seeds, consistent with group-cohesion or gap-selection
  sensitivity.

The full per-scenario table is in
[mechanism_synthesis_table.csv](evidence/issue_1609_seed_mechanisms_2026-05-31/mechanism_synthesis_table.csv).

## Interpretation Boundary

The mechanism labels are hypotheses backed by aggregate hard-vs-easy seed differences. They are not
trace-level causal proof. In particular:

- the source campaign is the historical Issue #1454 S10/h500 candidate surface, not a paper-facing
  seed-budget campaign;
- the selected planners all ran in adapter mode in the source surface, so planner-specific adapter
  behavior can contribute to apparent scenario sensitivity;
- no videos or raw trajectory bundles are promoted in this slice;
- fallback/degraded rows remain limitations under the benchmark fallback policy, although the
  Issue #1608 source analysis reports complete top-four coverage for this source bundle.

## Recommended Next Actions

1. Prioritize trace review for one representative from each high-count mechanism bucket:
   `classic_cross_trap_high`, `classic_doorway_high`, `classic_head_on_corridor_low`, and
   `francis2023_join_group`.
2. For each trace review, compare at least one hard seed and one easy seed using the same planner
   family and seed rows from the Issue #1454 bundle.
3. If a future paper-facing comparison needs these mechanisms, rerun a track-aware S20/S30 or
   equivalent seed-budget campaign before upgrading the diagnostic labels.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_1609_seed_mechanisms_2026-05-31/mechanism_synthesis_summary.json
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

No new benchmark campaign was run for this issue.
