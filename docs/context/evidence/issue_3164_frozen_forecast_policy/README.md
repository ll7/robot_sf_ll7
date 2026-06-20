# Issue #3164 Forecast Replay Fixture Suite

- Status: `passed`
- Evidence status: frozen-policy diagnostic only
- Fixture count: 4
- Scenario families: 4
- Variants: none, cv, semantic, interaction_aware, risk_filtered
- Row status summary: `{"degraded": 1, "native": 3}`
- Max native non-none closed-loop signatures: `1`

This frozen-policy summary does not claim planner superiority, safety improvement, or paper-grade evidence.

## Rows

### crossing_goal_directed

- Family: `crossing`
- Scenario: `issue_2868_goal_directed_crossing`
- Seed: `2868`
- Classification: `native`
- Reason: cv forecast produces closed-loop metrics that differ from the integrated no-forecast baseline

### corridor_route_conflict

- Family: `corridor_route_conflict`
- Scenario: `issue_2868_route_conflict_goal`
- Seed: `2868`
- Classification: `native`
- Reason: cv forecast produces closed-loop metrics that differ from the integrated no-forecast baseline

### dense_bottleneck_stress

- Family: `dense_bottleneck`
- Scenario: `dense_pedestrian_stress`
- Seed: `2765`
- Classification: `native`
- Reason: cv forecast produces closed-loop metrics that differ from the integrated no-forecast baseline

### signalized_crossing_degraded

- Family: `signalized_crossing`
- Scenario: `issue_2868_signalized_crossing`
- Seed: `2868`
- Classification: `degraded`
- Reason: native live path components are present but cv closed-loop metrics match the integrated no-forecast baseline, so cv does not affect replay behavior
