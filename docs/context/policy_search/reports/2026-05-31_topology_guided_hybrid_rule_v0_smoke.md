# Candidate Report: topology_guided_hybrid_rule_v0 (smoke)

## Decision

pass

## Hypothesis

A diagnostic masked-route selector can expose up to two local topology hypotheses from the occupancy grid, select one with route length and static clearance scoring, and feed it to the existing hybrid-rule corridor-subgoal scorer. Missing topology inputs fail closed; single-hypothesis rows do not activate the topology injection and are not benchmark evidence.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `topology_guided_hybrid_rule_v0`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/topology_guided_hybrid_rule_v0/smoke/issue1804_local/summary.json`
- Git commit: `9e0c2170de35397edc7202f110d041439c1163fc`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8941 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Topology-Hypothesis Probe

A separate bottleneck diagnostic was run because the default smoke scenario is only a wiring sanity
check and should not be treated as topology evidence.

Command:

```bash
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --output-dir output/diagnostics/issue1804_topology_guided_policy/classic_realworld_double_bottleneck_high_seed111_h160_retry \
  --max-hypotheses 2 \
  --min-hypotheses 2
```

Observed result:

- `diagnostic_status`: `diagnostic_complete`
- topology status counts: `ok=95`, `insufficient_hypotheses=65`
- selected local command sources: `dynamic_window=153`, `path_follow_0.5m=3`,
  `route_guide=2`, `topology_fail_closed=2`
- candidate route-corridor diagnostics included two scored topology hypotheses on the first
  available step, but the selected local command was still usually `dynamic_window`.

## Claim Boundary

This proves that the candidate is wired into policy search, can expose and select masked-route
hypotheses on the double-bottleneck diagnostic slice, and remains diagnostic-only. It does not prove
benchmark improvement, comfort improvement, or a better local command policy; the h160 trace still
mostly selected ordinary dynamic-window commands.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
