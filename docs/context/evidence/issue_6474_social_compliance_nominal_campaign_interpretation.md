# Issue #6474 social-compliance campaign interpretation

**Status:** interpretation handoff; not paper-facing.  **Evidence status:** nominal benchmark
evidence for the bounded simulator-defined estimands below.  This note does not create new
benchmark rows or upgrade the campaign to paper-grade evidence.

## Claim boundary and evidence admission

The source campaign compares `goal`, `social_force`, and `orca` on six frozen medium-band
scenario families with paired seeds `111`–`140`.  The estimand is the paired mean difference
(`comparison - reference`) for each declared social-compliance metric family.  Confidence
interval (CI) estimates are percentile-bootstrap 95% intervals, and p-values use paired
permutation tests with Holm step-down multiplicity control across the 15 declared
planner-pair-by-metric-family decisions.

The admitted claim is limited to simulator-defined metric-family effects on this frozen scenario
and seed surface.  It is not a planner ranking, composite social-compliance score, fairness,
deployment-ethics, legibility, safety, welfare, social-validity, real-world, or dissertation
claim.  Fallback, degraded, unavailable, invalid, and unknown execution rows are not success
evidence and are excluded from the admitted result.

The campaign artifact manifest records 540 valid benchmark-capable rows: 180 native rows and
360 declared adapter rows, with zero rejected fallback/degraded/unknown rows.  The campaign is
marked `paper_facing: false`; its Social Navigation Quality Index (SNQI) contract remains
diagnostic-only with warn enforcement.

## Evidence and provenance synthesis

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Comfort-exposure paired effects | [#6474](https://github.com/ll7/robot_sf_ll7/issues/6474) | Nominal benchmark evidence | `configs/benchmarks/issue_6474_social_compliance_nominal_campaign.yaml` (`fed85cef…`) | 111–140, paired | Tracked report, summary, campaign manifest, and artifact manifest under `docs/context/evidence/` | `comfort_exposure_person_s`, paired mean, CI95, raw/Holm-adjusted p, support and denominator | 3/3 comfort-exposure decisions reach the preregistered support threshold and reject at Holm-adjusted α=0.05 | Effects are simulator-defined and scenario-dependent; no ranking or causal interpretation |
| Other four metric families | [#6474](https://github.com/ll7/robot_sf_ll7/issues/6474) | Diagnostic-only | Same frozen config | 111–140, paired | Same tracked report and summary | `pedestrian_deviation`, `flow_disruption`, `legibility_progress`, `distributional_inconvenience` | 12/12 decisions have zero paired support and remain diagnostic-only; they were not zero-imputed | The absence of support is an availability/denominator result, not evidence of no effect |

Provenance breadcrumbs are:

- campaign source commit: `d0c8d3400f79c35b40062304b17f91598ccac98d`;
- analysis commit: `3615540feab993e55fba658e8fff63e5de4b6de4`;
- campaign manifest SHA-256: `6a76205cf6e36b6a7cf926f3cf56c83f4de971bce1b8b65d46c51e103e65e737`;
- frozen config SHA-256: `fed85cef7ac43817d0aa47a3ac10f9e7f4b50b4be6410e796fdf3d837e69811e`;
- scenario matrix SHA-256: `8c87eac284b51108d992521b1cdbef3edc28d13a3e0c5c34933a76f076ce3d6f`;
- terminal campaign job: `13985`, integrity status `valid`.

The raw episode rows and Slurm logs remain local output.  The tracked [report](issue_6474_social_compliance_nominal_campaign_report.md),
[summary](issue_6474_social_compliance_nominal_campaign_report.summary.json), [campaign manifest](issue_6474_social_compliance_nominal_campaign_manifest.json),
and [artifact manifest](issue_6474_social_compliance_nominal_campaign_artifact_manifest.json) are the cited evidence surface; the raw `output/` paths are not durable dependencies for this interpretation note.

## Supported paired effects

All values below are in person-seconds and use `comparison - reference`.  A negative value means
the comparison planner has lower measured comfort exposure on this simulator surface; it does
not mean that planner is socially better in general.

| Reference | Comparison | Paired rows | Mean difference | CI95 | Raw p | Holm-adjusted p | Denominator |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `goal` | `social_force` | 180 | -1.24056 | [-1.93667, -0.471569] | 0.001999 | 0.025987 | `pedestrian_steps` |
| `goal` | `orca` | 180 | 2.76778 | [1.83771, 3.69947] | 0.00049975 | 0.007496 | `pedestrian_steps` |
| `social_force` | `orca` | 180 | 4.00833 | [2.87981, 5.12642] | 0.00049975 | 0.007496 | `pedestrian_steps` |

## Scenario-family support

The supported family is heterogeneous rather than monotone across scenarios.  The entries below
are the per-family mean differences from the tracked report, in this order: `doorway`,
`group_crossing`, `head_on_corridor`, `merging`, `overtaking`, `station_platform`.  Each family
has 30 paired rows.

| Reference | Comparison | Per-family mean differences |
| --- | --- | --- |
| `goal` | `social_force` | `0.49`, `-2.80667`, `0.56`, `-4.18`, `0.02`, `-1.52667` |
| `goal` | `orca` | `3.72667`, `-0.516667`, `-1.16667`, `8.59667`, `3.06667`, `2.9` |
| `social_force` | `orca` | `3.23667`, `2.29`, `-1.72667`, `12.77667`, `3.04667`, `4.42667` |

This pattern supports a bounded interpretation: comfort exposure differs across the declared
planner pairs on the frozen simulator surface, with material scenario-family heterogeneity.  It
does not support an omnibus planner ordering or a claim that one planner is universally better.

## Diagnostic-only decisions

The remaining 12 declared decisions—each of the three planner pairs crossed with
`pedestrian_deviation`, `flow_disruption`, `legibility_progress`, and
`distributional_inconvenience`—has `n_paired = 0`.  Their denominators are retained in the source
report, and their missing support is reported as diagnostic-only rather than converted to zero.
The result therefore cannot answer whether those metric families differ between planners.

## Reproduction and next decision

The frozen analysis protocol can be self-checked without campaign data:

```bash
uv run python scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py --self-test
```

Rebuilding the admitted report requires the tracked campaign manifest plus the raw episode-row
directory from job `13985`; the canonical command and fail-closed row policy are documented in
`scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py` and
`docs/context/issue_691_benchmark_fallback_policy.md`.

The bounded interpretation step for #6474 is now represented by this note and the linked tracked
report, summary, and manifests.  Any future campaign, metric-family availability change, release artifact, or
paper-facing use requires a new evidence-admission decision with its own provenance and claim
boundary.  No further execution is implied by this handoff.
