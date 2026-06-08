# Issue #2566 Static-Recenter Inactive Propagation

Issue: [#2566](https://github.com/ll7/robot_sf_ll7/issues/2566)
Status: current analysis-only propagation; not new benchmark evidence.

## Purpose

This note records the propagation pass after PR #2520 / Issue #2438 closed the static-recenter
activation gap for the current held-out route. The durable conclusion is scoped:

- static recentering remains a local h500 diagnostic component from the Issue #2180/#2182
  one-factor evidence;
- the Issue #2221/#2250 held-out route should not be treated as a live transfer candidate;
- the instrumented Issue #2306 / Issue #2402 / Issue #2438 evidence classifies the unsolved
  held-out row as `mechanism_inactive` because activation count was zero, command source did not
  change, and trajectory delta was `0.0 m`;
- static recentering should be revisited only through a predeclared activation-targeted slice that
  preserves the same activation and command-source fields.

## Surfaces Updated

- [issue_2453_planner_mechanism_cards.md](issue_2453_planner_mechanism_cards.md): the
  `issue_2170_static_recenter_only` card now names the Issue #2438 inactive held-out closure and changes
  transfer status from generic `slice_local` to a stop boundary for the current held-out route.
- [issue_2228_research_dashboard.md](issue_2228_research_dashboard.md): the static-recenter lane now
  separates local diagnostic support from the current held-out inactive negative.
- [mechanism_closure_status.md](mechanism_closure_status.md): the static-recenter row now links the
  Issue #2438 closure and this propagation note.
- [policy_search/candidate_registry_summary.md](policy_search/candidate_registry_summary.md): the
  one-factor static-recenter candidate is now a diagnostic-only row rather than a hidden historical
  reference.
- [INDEX.md](INDEX.md), [README.md](README.md), and [catalog.yaml](catalog.yaml): this note is
  discoverable from the static-recenter context entry points.

## Open Issue Audit

Open GitHub search for `static recenter`, `static-recenter`, and `recentering` found:

| Issue | Action | Reason |
| --- | --- | --- |
| [#2521](https://github.com/ll7/robot_sf_ll7/issues/2521) | no comment | The epic already says static recentering is diagnostic-only/non-transfer for the current held-out slice and should only be revisited with a predeclared target failure slice. |
| [#2227](https://github.com/ll7/robot_sf_ll7/issues/2227) | no comment | The issue is already blocked and asks for failure-explanation panels, not positive static-recenter promotion. The #2438 inactive state should inform case selection when it unblocks. |
| [#2159](https://github.com/ll7/robot_sf_ll7/issues/2159) | no comment | The trace-review pack issue is broad and blocked; it does not currently promote static recentering as a transfer candidate. |

## Claim Boundary

This pass updates interpretation and discoverability only. It does not rerun static-recenter
experiments, change planner behavior, or claim static recentering is globally useless. The stop
decision applies to the current held-out transfer route unless a future issue predeclares an
activation-targeted slice and records the same fields needed to distinguish inactive,
active-but-irrelevant, and successful rescue behavior.

## Validation

```bash
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
BASE_REF=origin/main rtk scripts/dev/check_docs_proof_consistency_diff.sh
rtk git diff --check
```
