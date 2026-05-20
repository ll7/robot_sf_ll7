# Issue #1323 SocNavBench Personal-Space Velocity

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1323>

## Decision

SocNavBench personal-space cost now has a default-off `use_agent_velocity` mode. The default path
keeps the previous heading-derived unit vector, while enabled mode scales the heading vector by the
agent's stored scalar speed when that speed is finite and above `min_agent_speed`.

## Scope

This change is limited to the SocNavBench objective implementation and its parameters. It does not
make velocity-aware personal-space cost a benchmark default and does not claim planner-level
performance impact.

When speed is unavailable, zero, below threshold, or not finite, enabled mode falls back to the
heading-derived unit vector so stationary or incomplete agent states preserve the old behavior.

## Validation

Validated on 2026-05-20 from the issue worktree:

```bash
uv run pytest tests/test_socnavbench_personal_space_cost.py -q
```

Result: 4 passed. The tests cover default-off equivalence, enabled positive-speed sensitivity,
zero-speed fallback, and unavailable-speed fallback.

```bash
uv run python scripts/validation/run_issue_1323_socnav_personal_space_velocity_smoke.py
```

Result: passed. The synthetic sweep wrote:

- `output/validation/issue_1323_socnav_personal_space_velocity/latest/summary.json`
- `output/validation/issue_1323_socnav_personal_space_velocity/latest/summary.md`

The smoke shows the disabled baseline and zero-speed enabled row match, while positive-speed
enabled rows are sensitive to `agent_velocity_scale`. The local `output/` files are disposable;
the reviewable values from the passing run were:

| velocity_aware | agent_speed | velocity_scale | value | delta_from_default |
| --- | ---: | ---: | ---: | ---: |
| false | 2.000 | 1.000 | 0.135335 | 0.000000 |
| true | 0.000 | 1.000 | 0.135335 | 0.000000 |
| true | 2.000 | 0.500 | 0.135335 | 0.000000 |
| true | 2.000 | 1.000 | 0.606531 | 0.471195 |
| true | 2.000 | 2.000 | 0.882497 | 0.747162 |

## Follow-Up Boundary

The smoke is objective-level research evidence only. Planner-level SocNavBench benchmark claims
would still need a separate scenario run with explicit asset readiness and fallback/degraded-mode
reporting.
