# Issue #3184 Distributional Disruption Metric Contract (2026-06-20)

Issue: [#3184](https://github.com/ll7/robot_sf_ll7/issues/3184)
Decision source: [#3070](https://github.com/ll7/robot_sf_ll7/issues/3070)

Status: proposal / metric-contract note. No benchmark result, implementation,
or paper-facing claim is created by this note.
Implementation follow-up: [#3194](https://github.com/ll7/robot_sf_ll7/issues/3194).

## Decision Boundary

Robot SF may define diagnostic distributional disruption metrics over
observable simulation-state cohorts. The contract must not infer, encode, or
name protected attributes, and it must not make real-world ethical claims.

This note resolves the #3070 follow-up by defining the allowed simulation-only
cohort keys, candidate formulas, prohibited terminology, and minimum evidence
required before a future implementation issue can emit metrics.

## Observable Cohort Keys

Allowed cohort keys are derived only from fields already present or explicitly
declared in simulation/scenario state:

| Cohort key | Source | Allowed bins | Required support |
|---|---|---|---|
| `speed_tier` | configured desired speed or initial pedestrian speed | `slow`, `typical`, `fast`, with thresholds recorded in the metric config | at least one valid pedestrian per emitted bin |
| `trajectory_role` | scenario-authored route/behavior metadata | examples: `crossing`, `head_on`, `overtaking`, `waiting`, `parallel` | scenario must author the role; no inference from identity |
| `group_membership` | scenario-authored group id | `grouped`, `ungrouped`, or group-size bins | only when group ids are explicitly available |
| `interaction_exposure` | geometric relation to robot path in the episode | `near_path`, `far_path`, with distance threshold recorded | threshold and valid sample counts required |

These keys are diagnostic stratification axes, not people categories. A report
must include support counts for every emitted cohort and must omit cohorts whose
support is below the configured minimum.

## Candidate Metrics

All metrics require paired evidence: the same scenario, seed, pedestrian route,
and non-variant parameters with a robot-present condition and a matched control
condition. The control may be robot-absent or another explicitly declared
baseline, but the baseline must be named in the output.

For pedestrian `p`, cohort `c`, and valid paired timesteps `t` with duration
`dt`:

### Speed-Loss Distance

```text
speed_loss_m(p) = sum_t max(0, speed_control(p,t) - speed_robot(p,t)) * dt
cohort_speed_loss_mean_m(c) = mean_p_in_c speed_loss_m(p)
```

Units: meters of speed-equivalent progress loss.
Denominator: valid paired pedestrians in cohort `c`.

### Wait-Time Increase

```text
wait_increase_s(p) =
  sum_t I(speed_control(p,t) >= moving_threshold_mps
          and speed_robot(p,t) < waiting_threshold_mps) * dt
cohort_wait_increase_mean_s(c) = mean_p_in_c wait_increase_s(p)
```

Units: seconds.
Denominator: valid paired pedestrians in cohort `c`.

### Detour-Ratio Increase

```text
detour_ratio_increase(p) =
  path_length_robot(p) / max(path_length_control(p), epsilon_m) - 1
cohort_detour_ratio_increase_mean(c) = mean_p_in_c detour_ratio_increase(p)
```

Units: dimensionless ratio.
Denominator: valid paired pedestrians in cohort `c`; `epsilon_m` must be
recorded.

### Distributional Spread

For each cohort metric above, reports should include:

- `count`: valid paired pedestrian count;
- `mean`;
- `variance` or `std`;
- `median`;
- `p90` when support is large enough;
- `missing_reason` when a cohort is omitted.

The primary analysis is the distribution across observable simulation cohorts,
not a headline scalar score.

## Required Output Shape For A Future Implementation

A future implementation should emit a structured block under a namespace such
as:

```text
metrics.distributional_disruption
```

Required fields:

- `schema_version`, initially `distributional-disruption.v1`;
- `claim_boundary: diagnostic_simulation_proxy`;
- `baseline_condition`;
- `cohort_definitions`;
- `units`;
- `support_counts`;
- `cohort_metrics`;
- `missing_data`;
- `non_claims`.

Flat aggregate columns may be added later for report tooling, but the structured
block is the canonical contract.

## Terminology Boundary

Allowed wording:

- distributional disruption;
- inconvenience distribution;
- observable simulation cohort;
- speed-loss distribution;
- wait-time increase;
- detour-ratio increase;
- diagnostic simulation measure.

Prohibited wording in metric names, reports, issue titles, PR titles, and
paper-facing summaries unless a future maintainer decision explicitly changes
the lane:

- fairness;
- equity;
- bias;
- protected attribute;
- demographic group;
- disparate impact;
- human-subject assessment;
- real-world ethical assessment.

The prohibited terms may appear only in a boundary section like this one, where
the text is explicitly saying what not to claim.

## Non-Claims

These metrics do not measure:

- protected-attribute outcomes;
- demographic outcomes;
- human comfort or social validity;
- real-world ethics;
- legal compliance;
- calibrated pedestrian preference;
- benchmark-strength planner ranking by themselves.

They are diagnostic simulation measures for controlled Robot SF traces. Any
promotion to benchmark-strength or paper-facing evidence requires a separate
evidence issue, durable paired artifacts, uncertainty reporting, and review
against the repository evidence policy.

## Minimum Evidence Before Implementation

Before a future implementation PR, use follow-up #3194 or another bounded issue
that provides:

1. A fixture or scenario set with explicit observable cohort keys.
2. A paired robot-present/control condition with fixed seeds and non-variant
   parameters.
3. A schema fixture that includes at least two cohorts with nonzero support.
4. Missing-data behavior for absent or under-supported cohorts.
5. Tests for formula units, denominators, and support counts.
6. A report caveat proving the prohibited terminology is absent from generated
   output except in a boundary/non-claims section.
7. A validation command that runs without depending on worktree-local `output/`
   artifacts.

## Relationship To Existing Metric Contracts

- [Issue #3061](issue_3061_social_compliance_metric_contract.md) already names
  `distributional_inconvenience` as a social-compliance family. This note
  narrows the terminology and support-count contract before emission.
- [Issue #1085](issue_1085_pedestrian_impact_metrics.md) is the model for a
  schema-backed metric block with units, sample counts, and aggregate-ready
  reductions.
- [Issue #2458](issue_2458_human_interaction_proxy_metrics.md) is the closest
  claim-boundary precedent for human-centered simulation proxies.
