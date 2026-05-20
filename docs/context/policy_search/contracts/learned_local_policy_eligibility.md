# Issue #1363 Learned Local Policy Eligibility Checklist 2026-05-20

Related issues: <https://github.com/ll7/robot_sf_ll7/issues/1363>,
<https://github.com/ll7/robot_sf_ll7/issues/1355>, and
<https://github.com/ll7/robot_sf_ll7/issues/1359>

## Purpose

Use this checklist before recommending a learned local-navigation policy for Robot SF adapter work,
candidate-registry entry, or benchmark evaluation. It extends the generic planner metadata in
`robot_sf/baselines/interface.py`, `robot_sf/benchmark/algorithm_metadata.py`, and
`robot_sf/benchmark/planner_command_contract.py` with learned-policy-specific leakage,
observation, action, and logging gates.

Treat this document as the docs-first `LearnedLocalPolicySpec`: a stable review contract for
candidate intake until a later helper issue turns the checks into executable validation.

Passing this checklist means a candidate has an auditable local-policy contract. It is not
performance evidence, baseline readiness, safety certification, or a paper-facing benchmark claim.

## Required Verdict

Every learned local policy assessment should record one of these verdicts:

| Verdict | Meaning |
| --- | --- |
| `eligible_for_adapter` | Deployment observation/action contract can be satisfied without leakage. |
| `eligible_for_research_only` | Runnable for exploratory analysis, but not eligible for benchmark rows yet. |
| `training_only_or_oracle` | Uses privileged or future information at evaluation time. |
| `monitor_only` | Interesting source, but no runnable Robot SF contract yet. |
| `reject_for_benchmark` | Incompatible with local-policy, observation, or reproducibility boundaries. |

## Observation Gate

A learned local policy must define `observation_t`, the exact data available at one decision step.
The assessment must classify every input field as deployment-observable, training-only, or
forbidden at evaluation time.

Deployment-observable fields may include:

- robot pose, heading, velocity, radius, and active kinematics mode,
- current goal, route waypoint, or local subgoal,
- current pedestrian positions, velocities, radii, and visibility masks,
- bounded history ending at `t`, with no rows newer than `observation_t`,
- lidar/range observations or occupancy-grid features available to the declared observation level,
- static map or obstacle tokens when the benchmark scenario exposes them to the planner.

Training-only fields may include:

- privileged critic inputs used only during training,
- expert labels, demonstrations, or imitation targets,
- future returns, advantages, risk labels, or outcome annotations used for optimization,
- offline dataset metadata that is not passed to the deployed policy.

Forbidden evaluation-time fields include:

- future pedestrian states, future ego trajectory, or future collision/near-miss labels,
- simulator termination reason before the step has happened,
- ground-truth obstacle or pedestrian fields outside the declared observation level,
- unbounded global route or map data when the policy claims local-only observation,
- training split identity, scenario outcome, or benchmark aggregate metadata.

If a source policy needs forbidden fields to run, classify it as `training_only_or_oracle` unless a
separate deployment adapter removes those fields and is tested directly.

## Split And Provenance Gate

The assessment must state:

- training data source, if any,
- validation and test splits, including map/scenario overlap assumptions,
- checkpoint or model provenance,
- whether privileged-training inputs differ from deployment inputs,
- whether normalization statistics were fit on training data only,
- whether candidate evidence comes from source claims, Robot SF local execution, or synthesis.

Do not compare learned policies as benchmark candidates when their training/evaluation split leaks
the exact test scenarios, future trajectories, or outcome labels used by Robot SF metrics.

## Action Gate

The policy must declare one primary output contract:

| Output family | Required contract |
| --- | --- |
| Velocity command | Frame, units, bounds, kinematics compatibility, and projection policy. |
| Bounded residual command | Base planner, residual bounds, clamp/projection order, and fallback behavior. |
| Waypoint or subgoal | Frame, horizon, goal tolerance, local planner used to execute it, and timeout behavior. |
| Short trajectory | Horizon, timestep, frame, feasibility projection, collision handling, and first-action extraction. |
| Motion-primitive scores | Primitive library, score normalization, tie-breaking, selected primitive execution, and fallback. |

The action adapter must define how raw model output becomes a Robot SF action dictionary. If a guard
or kinematics projector changes the output, the changed action must be logged separately from the
raw model output.

## Required Per-Step Logging

Benchmark or smoke runs for a learned candidate must be able to emit these fields, either in
episode records or candidate-specific diagnostics:

- `raw_model_action`: the direct model output before Robot SF adaptation,
- `adapted_action`: the command after shape, frame, scale, or kinematics conversion,
- `post_guard_action`: the final command after safety guard, fallback, or projection,
- `guard_applied`: boolean,
- `guard_or_fallback_reason`: stable string or `none`,
- `observation_level`: active observation level,
- `planner_observation_mode`: planner observation mode when it differs from the global
  observation level or adds planner-specific preprocessing,
- `action_bounds`: action bounds used by the candidate,
- `action_projection_metadata`: projection metadata when a command was clipped or transformed.

For candidates without a guard, `post_guard_action` may equal `adapted_action`, but the equality
should be explicit so reviewers can distinguish unguarded success from guard-mediated success.

## Candidate Registry Boundary

`docs/context/policy_search/candidate_registry.yaml` is for implemented or concrete runnable Robot
SF candidates with config pointers. Do not add a learned method to that registry merely because a
paper, repository, or checkpoint exists.

Before registry entry, the candidate must have:

- an eligibility verdict from this checklist,
- a concrete Robot SF config or adapter path,
- a stated observation/action contract,
- a smoke or validation command that exercises the actual adapter path,
- a failure/fallback policy for missing checkpoints, unsupported observations, or guard activation.

Candidates that are source-side only, blocked by forbidden evaluation fields, or missing runnable
Robot SF configs should stay in reject/monitor notes and cite the evidence needed to reopen them.

## Review Checklist

Use this compact checklist in issue or PR reviews:

- [ ] Verdict recorded and separated from performance claims.
- [ ] `observation_t` defined.
- [ ] Deployment-observable, training-only, and forbidden fields classified.
- [ ] Privileged-training inputs do not enter evaluation-time observations.
- [ ] Train/validation/test split and normalization provenance stated.
- [ ] One action-output family selected and fully declared.
- [ ] Raw, adapted, post-guard, and fallback-reason logging specified.
- [ ] Candidate registry entry exists only when a runnable Robot SF config or adapter path exists.
- [ ] Any fallback or degraded mode is reported as a caveat, not a successful benchmark outcome.

## Validation

For docs-only updates to this checklist or candidate assessments that cite it, use:

```bash
rg -n "LearnedLocalPolicy|observation_t|post_guard_action|privileged" docs/context docs/README.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Code changes that implement a candidate or helper must add targeted tests for the adapter path and
then run the normal PR readiness gate.
