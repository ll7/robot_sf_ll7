# Issue #2704 Progress-Gated Topology Successor Diagnostic

Issue: [#2704](https://github.com/ll7/robot_sf_ll7/issues/2704)
Status: current, diagnostic-only evidence.

## Claim Boundary

This note records the implementation and one paired smoke for the
`primary_route_reselection_requires_route_progress` successor selected by
[Issue #2660](archive/issue_2660_topology_successor_gate.md). It does not promote
`topology_guided_hybrid_rule_v0_progress_gated_reselection`, establish benchmark evidence, or claim
a planner improvement.

Compact evidence:
[evidence/issue_2704_progress_gated_topology_successor/summary.json](evidence/issue_2704_progress_gated_topology_successor/summary.json).

## Implemented Mechanism

The successor adds a bounded progress gate to the existing primary-route reuse penalty:

- records recent primary-route remaining-distance samples from selected route hypotheses;
- computes `primary_route_recent_progress_m` as the decrease from the oldest to newest recent
  primary-route distance sample;
- suppresses the reuse penalty only when
  `primary_route_progress_gate_enabled=true` and recent progress is at least the predeclared
  `primary_route_progress_gate_threshold_m`;
- emits compact diagnostic fields:
  `primary_route_recent_progress_m`, `primary_route_recent_progress_sample_count`,
  `primary_route_progress_gate_satisfied`, and `reuse_penalty_suppressed_by_progress`.

The candidate config is
`configs/policy_search/candidates/topology_guided_hybrid_rule_v0_progress_gated_reselection.yaml`.
It is registered as diagnostic-only in
`docs/context/policy_search/candidate_registry.yaml`.

## Paired Smoke

The paired smoke used the same canonical slice as the reuse-penalty decision:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0_progress_gated_reselection \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir /tmp/robot_sf_issue2704_progress_gated_successor

LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir /tmp/robot_sf_issue2704_baseline_comparator
```

Both runs completed as `diagnostic_complete`. Raw diagnostics remain disposable local artifacts; the
tracked summary preserves the compact reviewable result.

## Result

```yaml
classification: revise
claim_boundary: diagnostic_only
candidate:
  selected_source_counts:
    topology_hypothesis: 32
    dynamic_window: 122
  topology_reuse_penalty:
    applied_steps: 7
    progress_gate_satisfied_steps: 28
    progress_suppressed_steps: 6
    max_primary_route_recent_progress_m: 6.0627417901388405
comparator:
  selected_source_counts:
    topology_hypothesis: 33
    dynamic_window: 121
deltas:
  selected_hypothesis_primary_count_delta: -5
  selected_hypothesis_non_primary_count_delta: 5
  topology_command_steps_delta: -1
  non_primary_topology_command_influence_delta: 0
  route_progress_delta_m: 0.0
  terminal_outcome_delta: unchanged_horizon_exhausted
```

The mechanism ran and exposed the intended diagnostics, but it did not improve the measured blocker
on the canonical h160 double-bottleneck slice. It reduced selected primary-route hypotheses by five
and suppressed six reuse penalties when progress was sufficient, yet non-primary topology-command
influence stayed flat at seven steps, route-progress delta was unchanged, topology-command steps
fell by one, and both runs ended `horizon_exhausted`.

## Decision

Do not rerun this progress-gated candidate unchanged on the same slice as promotion or benchmark
evidence. The useful result is negative/diagnostic: the lane now has an implemented successor and a
paired smoke showing that route-progress-aware reuse suppression alone is not enough to move the
canonical blocker.

Recommended next action is synthesis before another local topology variant. A future issue may
open a new topology hypothesis, but it should name a different mechanism and a discriminating
metric beyond primary-route selection counts.

## Validation

Use the implementation and docs checks from the PR that introduced this note:

```bash
rtk uv run pytest tests/planner/test_topology_guided_local_policy.py \
  tests/validation/test_run_topology_hypothesis_diagnostics.py -q
rtk uv run ruff check robot_sf/planner/topology_guided_local_policy.py \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  tests/planner/test_topology_guided_local_policy.py \
  tests/validation/test_run_topology_hypothesis_diagnostics.py
rtk uv run ruff format --check robot_sf/planner/topology_guided_local_policy.py \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  tests/planner/test_topology_guided_local_policy.py \
  tests/validation/test_run_topology_hypothesis_diagnostics.py
rtk uv run python scripts/validation/check_research_lane_states.py
rtk uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/catalog.yaml \
  --path docs/context/INDEX.md \
  --path docs/context/README.md \
  --path docs/context/policy_search/candidate_registry.yaml \
  --path docs/context/policy_search/candidate_registry_summary.md \
  --path docs/context/issue_2704_progress_gated_topology_successor.md \
  --path docs/context/evidence/issue_2704_progress_gated_topology_successor/README.md \
  --path docs/context/evidence/issue_2704_progress_gated_topology_successor/summary.json
rtk git diff --check
```
