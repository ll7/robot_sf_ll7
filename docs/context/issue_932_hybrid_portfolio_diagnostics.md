# Issue 932 Hybrid Portfolio Diagnostics

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/932>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/871>

## Goal

Add a first small runtime diagnostics slice for the policy-stack workstream. The existing
`HybridPortfolioAdapter` already switches between risk-DWA, ORCA, prediction, and optional MPPI
planner heads, but it did not expose episode-local diagnostics for benchmark metadata.

## Public Surface

`HybridPortfolioAdapter.diagnostics()` now returns a JSON-safe payload with:

- `steps`,
- `active_head`,
- `hold_remaining`,
- `selected_head_counts`,
- `fallback_count`,
- `last_decision`.

`last_decision` records the desired head, selected head, fallback flag, fallback source, error
string when available, active head, and hysteresis hold count.

## Scope Boundary

This is not the full `policy_stack_v1` planner from #871. It does not add proposal normalization,
risk scoring, safety shielding, route rebasing, or benchmark proof. It only gives the current
hybrid portfolio a benchmark-consumable diagnostics hook that the existing map runner can snapshot
through its generic `diagnostics()` path.

Fallback-on-exception remains an explicit degraded behavior of `HybridPortfolioAdapter`; the
diagnostics now make that visible through `fallback_count` and `last_decision`.

## Validation

Targeted TDD evidence:

```bash
uv run pytest tests/planner/test_risk_dwa_mppi_hybrid.py -q
```

The RED run failed with `AttributeError: 'HybridPortfolioAdapter' object has no attribute
'diagnostics'` for the new selection and fallback tests. The GREEN run passed:

```text
24 passed in 9.78s
```

Before PR handoff, also run:

```bash
git diff --check
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Follow-Up Boundary

Future #871 children should define the full `policy_stack_v1` proposal/risk/shield schema and then
prove the planner through the normal benchmark or policy-analysis entry point. This diagnostics
surface can be reused directly or superseded with an explicitly versioned policy-stack schema.
