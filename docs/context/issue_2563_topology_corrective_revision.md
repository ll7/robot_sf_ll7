# Issue #2563 Topology Corrective Revision Proposal

Issue: [#2563](https://github.com/ll7/robot_sf_ll7/issues/2563)
Status: current, proposal/analysis only.

## Claim Boundary

This note selects one diagnostic topology-selection revision to try after the
[#2530](issue_2530_topology_near_parity_corrective_smoke.md) near-parity corrective smoke
classified the lane as `revise`. It does not claim benchmark improvement, planner promotion, or
leaderboard movement. The selected mechanism must still pass a controlled diagnostic gate before
any implementation issue treats it as useful planner evidence.

## Selected Revision

Selected mechanism:
`primary_route_reuse_penalty_under_near_parity_alternatives`.

The next topology revision should penalize repeated reuse of `primary_route` when near-parity
alternatives remain eligible and the primary route has not shown enough route-progress evidence to
justify another selection. The penalty should stay inside the diagnostic topology-hypothesis
selection surface; it should not retune downstream command scoring or promote the
`topology_guided_hybrid_rule_v0` candidate.

Expected effect:

- reduce repeated rearming of `primary_route` under eligible near-parity alternatives;
- preserve or increase non-primary topology-command influence on the canonical slice;
- make route-progress change explicit before a primary-route reselection is treated as acceptable;
- keep the outcome classified as diagnostic-only unless route progress, terminal behavior, or a
  paired comparator improves.

## Evidence

The selection is grounded in the existing topology evidence chain:

- [#2258 primary-route audit](issue_2258_topology_primary_route_audit.md): alternatives were often
  present, but the selected topology hypothesis stayed on `primary_route` and all 33 topology-command
  wins inherited that route.
- [#2403 selection-score decision](issue_2403_topology_selection_score_decision.md): the field-mapped
  decision accepted `primary_route_overselected`; alternatives were scored and rejected by lower
  topology-selection score rather than absence or invalidity.
- [#2518 near-parity gate](issue_2518_topology_near_parity_gate.md): the near-parity diversity gate
  moved the route selector off `primary_route` on 42 rows and produced 7 non-primary
  topology-command influence steps, so non-primary arbitration is reachable.
- [#2530 corrective smoke](issue_2530_topology_near_parity_corrective_smoke.md): the same canonical
  slice still ended `horizon_exhausted`, with only `0.16812408921843236` meters of maximum
  per-rank route-progress delta and a `revise` decision.

That pattern makes "selection diversity alone" too weak, while still pointing at upstream route
selection rather than downstream command arbitration as the smallest next intervention. A
primary-route reuse penalty is the narrowest revision that directly targets the observed
overselection history without requiring a broader new planner or benchmark run.

## Deferred Alternatives

`minimum_hypothesis_sequence_diversity` is deferred because the current evidence already shows
eligible alternatives and non-primary selections in the canonical slice. It may be useful later if
future diagnostics show that the hypothesis set repeats stale masked cells rather than merely
overusing `primary_route`.

`alternative_route_selection_only_when_primary_progress_stalls` is deferred as a stricter successor
to the selected mechanism. It needs more explicit route-progress accounting before it can be the
first revision, but it can become the next refinement if the reuse penalty increases volatility
without progress.

`hysteresis_or_switch_cost` is deferred because the current failure is not primarily random
switching. A stability term should come after a revision proves that non-primary route use can
improve progress; otherwise it risks stabilizing the same stalled route choice.

## Diagnostic Gate

A follow-up implementation or launch-packet issue should use the same canonical slice as the prior
evidence before expanding scope:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2563_primary_route_reuse_penalty
```

Minimum diagnostic fields:

- `diagnostic_status`
- topology status counts
- route-selector selected-hypothesis counts
- near-parity gate reasons
- local command-source counts
- topology-command influence counts
- route-progress deltas
- hypothesis switch count
- terminal outcome

Pass/revise/stop rule:

- `continue`: diagnostic completes, non-primary topology-command influence is preserved or
  increased against #2530, route-progress evidence improves without worse terminal behavior, and the
  paired comparator does not explain the gain away.
- `revise`: the mechanism runs but progress remains weak, terminal behavior stays
  `horizon_exhausted`, or switching volatility increases without route-progress benefit.
- `stop`: non-primary topology-command influence collapses to zero, the diagnostic fails closed on
  required fields, or the mechanism only changes route labels without a plausible progress signal.

## Follow-Up Boundary

[#2540](https://github.com/ll7/robot_sf_ll7/issues/2540) is the existing paired/broadened topology
diagnostic follow-up. It should use this selected mechanism as the concrete revision hypothesis, or
open a narrower child issue if implementation requires changing the topology-selection code before
the paired diagnostic can run.

## Validation

For this proposal note, use cheap docs validation:

```bash
rtk uv run python scripts/validation/check_research_lane_states.py
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
rtk git diff --check
```

No runtime planner validation is required for #2563 because this issue selects the next hypothesis
and gate; it does not implement the mechanism.
