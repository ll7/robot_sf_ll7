# Issue 938 Hybrid Portfolio Last Decision

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/938>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/871>

Depends on issue #932 / PR #933.

## Goal

Align `HybridPortfolioAdapter` with existing planner step-level diagnostics conventions. The
portfolio diagnostics added in #932 expose `last_decision` through `diagnostics()`, and this issue
adds the direct `last_decision()` accessor already used by nearby planners.

## Public Surface

- `HybridPortfolioAdapter.last_decision() -> dict[str, Any] | None`

The method returns a defensive copy of the latest selected-head decision, or `None` before the
first planning step and after `reset()`.

## Validation

Targeted TDD evidence:

```bash
uv run pytest tests/planner/test_risk_dwa_mppi_hybrid.py -q
```

The RED run failed with `AttributeError: 'HybridPortfolioAdapter' object has no attribute
'last_decision'`. The GREEN run passed:

```text
25 passed in 9.81s
```

Before PR handoff, run the stacked readiness gate against the #932 branch:

```bash
git diff --check
BASE_REF=origin/932-hybrid-portfolio-diagnostics scripts/dev/pr_ready_check.sh
```
