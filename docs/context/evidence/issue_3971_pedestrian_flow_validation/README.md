# Issue #3971 Pedestrian Flow Validation Evidence

Claim boundary: diagnostic-only pedestrian-flow validation; no realism thresholds, build gates, robot-vs-crowd safety claims, or benchmark-strength claims.

Evidence status: diagnostic-only smoke evidence. No pass/fail realism threshold was applied.

Major caveats: short CPU fixtures, synthetic maps, no robot inserted, no human-subject realism validation, and no benchmark-strength claim.

## Run Summary

- Schema: `pedestrian_flow_validation.report.v1`
- Scenarios: bidirectional_corridor, bottleneck, forked_route
- Pedestrian counts: [2, 6]
- Duration: 2.0 s
- Timestep: 0.1 s
- Robot inserted: False
- Thresholds applied: False

## Flow Metrics

| scenario | peds | density | avg speed | jam s |
| --- | ---: | ---: | ---: | ---: |
| bidirectional_corridor | 2 | 0.0417 | 1.3977 | 0.0000 |
| bidirectional_corridor | 6 | 0.1250 | 1.3206 | 0.0000 |
| bottleneck | 2 | 0.0417 | 1.2217 | 0.0000 |
| bottleneck | 6 | 0.1250 | 1.1065 | 0.0000 |
| forked_route | 2 | 0.0278 | 1.3928 | 0.0000 |
| forked_route | 6 | 0.0833 | 1.2816 | 0.0000 |
