# Issue #2706 Topology Lane Synthesis After Progress-Gated Successor

Issue: [#2706](https://github.com/ll7/robot_sf_ll7/issues/2706)
Status: current diagnostic synthesis.

## Claim Boundary

This note synthesizes the recent topology-selection diagnostic lane after the progress-gated
successor in [Issue #2704](issue_2704_progress_gated_topology_successor.md). It does not establish
benchmark evidence, planner promotion, paper-facing success, or a general negative result for all
topology-guided planning.

The conclusion is narrower: on the canonical h160 double-bottleneck slice, local selector variants
that primarily move `primary_route` versus non-primary hypothesis labels have repeatedly failed to
move the blocker that matters: topology-command influence, route progress, or terminal outcome.

## Evidence Chain

| Issue | Mechanism | Result | Interpretation |
| --- | --- | --- | --- |
| [#2518](issue_2518_topology_near_parity_gate.md) | Near-parity diversity gate | `accept` as diagnostic direction | Selection diversity became possible: `primary_route=56`, non-primary selections `42`, and 7 non-primary topology-command influence steps. |
| [#2530](issue_2530_topology_near_parity_corrective_smoke.md) | Corrective-behavior smoke for the near-parity gate | `revise` | The selector reached local-command arbitration, but the episode still ended `horizon_exhausted` with only `0.16812408921843236 m` route-progress delta. |
| [#2540](issue_2540_topology_reuse_penalty_diagnostic.md) | Primary-route reuse-penalty implementation | diagnostic implementation only | Mechanism and candidate existed, but required a paired gate before any improvement claim. |
| [#2624](issue_2624_topology_reuse_penalty_gate.md) | Paired reuse-penalty diagnostic | `revise` | Reuse penalty activated and reduced primary-route selections, but non-primary command influence stayed flat at 7, route progress regressed by `-0.9514718767541766 m`, and outcome stayed `horizon_exhausted`. |
| [#2660](archive/issue_2660_topology_successor_gate.md) | Successor decision after reuse-penalty regression | `revise` with named successor | Selected progress-gated primary-route reselection as the smallest remaining discriminating selector hypothesis. |
| [#2704](issue_2704_progress_gated_topology_successor.md) | Progress-gated primary-route reselection | `revise` | The gate ran and suppressed six reuse penalties, but route progress stayed unchanged, non-primary command influence stayed flat, topology-command steps fell by one, and outcome stayed `horizon_exhausted`. |

## Lane Decision

```yaml
topology_selector_variant_synthesis:
  current_state: stop
  claim_boundary: diagnostic_only
  stopped_family:
    - topology_guided_hybrid_rule_v0 unchanged near-parity selector reruns on the canonical h160 slice
    - topology_guided_hybrid_rule_v0_reuse_penalty unchanged reruns on the canonical h160 slice
    - topology_guided_hybrid_rule_v0_progress_gated_reselection unchanged reruns on the canonical h160 slice
  reason: >
    The tested variants changed route-selector labels, but did not improve non-primary
    topology-command influence, route progress, or terminal outcome on the canonical h160
    double-bottleneck slice.
  uncertainty: medium
```

The lane should stop adding same-family selector tie-breaks for this slice. The repeated diagnostic
pattern is now strong enough to reject "more primary-route label movement" as the next useful
research step.

This does not mean topology-guided planning is exhausted. It means a future topology issue should
open a new hypothesis with a different mechanism and metric before implementation starts.

## What Not To Rerun Unchanged

Do not treat any of these as benchmark-improvement candidates on the canonical h160
double-bottleneck slice without a new issue, comparator, and stop rule:

- `topology_guided_hybrid_rule_v0` as another near-parity/selection-diversity proof;
- `topology_guided_hybrid_rule_v0_reuse_penalty`;
- `topology_guided_hybrid_rule_v0_progress_gated_reselection`.

Rerunning them can still be useful for regression checks or report-shape validation, but the result
should be labeled as reproducibility or diagnostics plumbing, not new research evidence.

## Future Hypothesis Gate

A future topology issue should be accepted only if it names at least one of these:

- a mechanism that changes topology-command feasibility or command generation, not only selected
  hypothesis labels;
- a hypothesis-availability mechanism that reduces the 70/160 insufficient-hypothesis frames on
  the canonical slice;
- a trace-level blocker analysis explaining why the 7 non-primary command-influence steps do not
  translate into route progress;
- a predeclared scenario slice where topology-command influence, route-progress delta, and terminal
  behavior are all decision metrics.

Minimum acceptance fields for a successor issue:

- target mechanism;
- comparator;
- route-progress or command-feasibility metric that is not primary-route selection count;
- stop rule for unchanged `horizon_exhausted` outcomes;
- claim boundary kept diagnostic-only unless a separate benchmark contract is opened.

## Validation

Use cheap docs validation for this synthesis:

```bash
rtk uv run python scripts/validation/check_research_lane_states.py
rtk uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/catalog.yaml \
  --path docs/context/INDEX.md \
  --path docs/context/README.md \
  --path docs/context/issue_2706_topology_lane_synthesis.md \
  --path docs/context/research_lane_states.md
rtk git diff --check
```
