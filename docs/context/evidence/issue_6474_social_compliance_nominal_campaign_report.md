<!-- AI-GENERATED (robot_sf#6474) - NEEDS-REVIEW -->
# Nominal social-compliance cross-planner report (issue #6474)

> AI-GENERATED NEEDS-REVIEW

## Claim boundary

Simulator-defined social-compliance metric-family paired effects only, for the goal, social_force, orca planner pairs. Effects are mean paired differences (comparison - reference) with percentile-bootstrap 95% CI and two-sided paired-permutation p-values under Holm step-down multiplicity control across the planner-pair-by-metric-family decisions (family-wise alpha = 0.05). Declared support counts and denominators are reported per decision; metric families with insufficient paired support are marked diagnostic-only and never zero-imputed.

No composite social-compliance ranking, fairness, deployment-ethics, legibility, social-validity, safety, welfare, universal-ranking, or real-world claim is made. Complete valid output may reach nominal benchmark evidence only for these simulator-defined metrics.

## Provenance

- campaign config: `configs/benchmarks/issue_6474_social_compliance_nominal_campaign.yaml`
- config sha256: `fed85cef7ac43817d0aa47a3ac10f9e7f4b50b4be6410e796fdf3d837e69811e`
- analysis commit sha: `3615540feab993e55fba658e8fff63e5de4b6de4`
- campaign source commit sha: `d0c8d3400f79c35b40062304b17f91598ccac98d`
- campaign manifest: `output/issue_6474_social_compliance_nominal/job-13985/campaign_manifest.json`
- campaign manifest sha256: `6a76205cf6e36b6a7cf926f3cf56c83f4de971bce1b8b65d46c51e103e65e737`
- episode rows: `output/issue_6474_social_compliance_nominal/job-13985/runs`
- episode rows sha256: `68421501a52f953cfda60983c5a22d441203d89f6a8463ee67306aeeb0de57f5`
- rows validated: 540
- benchmark-capable rows: 540
- execution modes: {"adapter": 360, "native": 180}
- rejected (fallback/degraded/unknown) rows: 0
- schema version: `social-compliance-metric-contract.v1`
- claim class: `diagnostic_proxy`

## Paired effects by planner pair and metric family

### social_force vs goal (mean difference = social_force - goal)

| metric family | units | n paired | mean diff | CI95 low | CI95 high | raw p | Holm adj. p | reject @alpha | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pedestrian_deviation | meters | 0 | nan | nan | nan | nan | nan | no | tracked_pedestrian_steps_with_baseline |
| flow_disruption | seconds | 0 | nan | nan | nan | nan | nan | no | pedestrians_with_reference_arrival |
| comfort_exposure | person_seconds | 180 | -1.24056 | -1.93667 | -0.471569 | 0.001999 | 0.025987 | yes | pedestrian_steps |
| legibility_progress | meters | 0 | nan | nan | nan | nan | nan | no | robot_steps_before_terminal |
| distributional_inconvenience | seconds | 0 | nan | nan | nan | nan | nan | no | pedestrians_with_delay_samples |

### orca vs goal (mean difference = orca - goal)

| metric family | units | n paired | mean diff | CI95 low | CI95 high | raw p | Holm adj. p | reject @alpha | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pedestrian_deviation | meters | 0 | nan | nan | nan | nan | nan | no | tracked_pedestrian_steps_with_baseline |
| flow_disruption | seconds | 0 | nan | nan | nan | nan | nan | no | pedestrians_with_reference_arrival |
| comfort_exposure | person_seconds | 180 | 2.76778 | 1.83771 | 3.69947 | 0.00049975 | 0.00749625 | yes | pedestrian_steps |
| legibility_progress | meters | 0 | nan | nan | nan | nan | nan | no | robot_steps_before_terminal |
| distributional_inconvenience | seconds | 0 | nan | nan | nan | nan | nan | no | pedestrians_with_delay_samples |

### orca vs social_force (mean difference = orca - social_force)

| metric family | units | n paired | mean diff | CI95 low | CI95 high | raw p | Holm adj. p | reject @alpha | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pedestrian_deviation | meters | 0 | nan | nan | nan | nan | nan | no | tracked_pedestrian_steps_with_baseline |
| flow_disruption | seconds | 0 | nan | nan | nan | nan | nan | no | pedestrians_with_reference_arrival |
| comfort_exposure | person_seconds | 180 | 4.00833 | 2.87981 | 5.12642 | 0.00049975 | 0.00749625 | yes | pedestrian_steps |
| legibility_progress | meters | 0 | nan | nan | nan | nan | nan | no | robot_steps_before_terminal |
| distributional_inconvenience | seconds | 0 | nan | nan | nan | nan | nan | no | pedestrians_with_delay_samples |

## Scenario-family support

- social_force vs goal / pedestrian_deviation: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- social_force vs goal / flow_disruption: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- social_force vs goal / comfort_exposure: doorway: n=30, mean diff=0.49, group_crossing: n=30, mean diff=-2.80667, head_on_corridor: n=30, mean diff=0.56, merging: n=30, mean diff=-4.18, overtaking: n=30, mean diff=0.02, station_platform: n=30, mean diff=-1.52667.
- social_force vs goal / legibility_progress: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- social_force vs goal / distributional_inconvenience: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs goal / pedestrian_deviation: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs goal / flow_disruption: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs goal / comfort_exposure: doorway: n=30, mean diff=3.72667, group_crossing: n=30, mean diff=-0.516667, head_on_corridor: n=30, mean diff=-1.16667, merging: n=30, mean diff=8.59667, overtaking: n=30, mean diff=3.06667, station_platform: n=30, mean diff=2.9.
- orca vs goal / legibility_progress: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs goal / distributional_inconvenience: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs social_force / pedestrian_deviation: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs social_force / flow_disruption: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs social_force / comfort_exposure: doorway: n=30, mean diff=3.23667, group_crossing: n=30, mean diff=2.29, head_on_corridor: n=30, mean diff=-1.72667, merging: n=30, mean diff=12.7767, overtaking: n=30, mean diff=3.04667, station_platform: n=30, mean diff=4.42667.
- orca vs social_force / legibility_progress: no paired support.
  Diagnostic-only (n_paired=0 < 5).
- orca vs social_force / distributional_inconvenience: no paired support.
  Diagnostic-only (n_paired=0 < 5).

## Campaign caveats

The following pre-declared campaign-wide caveats are recorded and do not invalidate the supported comfort_exposure decisions:

- Route-clearance warnings are unresolved (2, below the per-scenario warning threshold of 0.5 m):
  - `classic_doorway_medium`: min center-distance 1.012233 m, clearance margin 0.012233 m.
  - `classic_overtaking_medium`: min center-distance 1.326555 m, clearance margin 0.326555 m.
- SNQI is reported diagnostic-only (contract status `fail`, positioning recommendation
  `downgrade_to_appendix_or_implementation_aid`); it contributes no supported decision here.
- Fairness is explicitly excluded from any ranking or comparison; no fairness, ethics, safety,
  social-validity, real-world, universal-ranking, composite-score, or dissertation claim is made.

Supported decisions constitute nominal benchmark evidence for the stated simulator-defined metric-family estimands only. Diagnostic-only families are reported with declared support and are excluded from inference.

