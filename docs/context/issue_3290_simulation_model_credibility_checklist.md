# Issue #3290 Simulation Model Credibility Checklist

**Status**: Current checklist/template for Robot SF simulation-evidence credibility reviews.

**Related issue**: [#3290](https://github.com/ll7/robot_sf_ll7/issues/3290)

**Policy sources**:

- [Maintainer Values And Hard Contracts](../maintainer_values.md)
- [Artifact Evidence Vocabulary](artifact_evidence_vocabulary.md)
- [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md)
- [Simulation-Evidence Safety Case Template](../simulation_evidence_safety_case.md)

## Claim Boundary

This checklist helps decide whether a Robot SF simulation result is credible for its stated use.
It does not turn simulation output into real-world validation, deployment approval, safety
certification, or paper-grade evidence by itself.

A filled checklist must separate:

- **verification**: evidence that repository code, configs, fixtures, metrics, scenario contracts,
  and artifact provenance behaved as intended;
- **validation**: evidence that the simulated model corresponds to the intended real-world
  pedestrian, robot, sensor, actuator, or operating-domain behavior.

A campaign can be well verified and still fail validation for real-world or paper-facing claims.
When validation evidence is missing, mark the row `missing` and link the issue that owns the next
proof step instead of weakening the claim boundary.

## Credibility Decision Grid

Use these rows before promoting a simulation campaign into a release, benchmark report,
paper-facing claim, or safety-case evidence map. Keep the answer compact, but do not leave blanks.

| Area | Question | Evidence type | Status values |
| --- | --- | --- | --- |
| Claim boundary | What exact claim is being made, and what is explicitly not claimed? | Verification boundary | `covered`, `partial`, `missing` |
| Artifact provenance | Are configs, commands, commits, schemas, checksums, and durable evidence paths named? | Verification | `covered`, `partial`, `missing` |
| Numerical implementation | Are integration steps, force calculations, action projections, metric formulas, and units covered by tests or fixtures? | Verification | `covered`, `partial`, `missing` |
| Fixture and regression tests | Do focused tests prove the contract used by this campaign, including fail-closed paths? | Verification | `covered`, `partial`, `missing` |
| Scenario validity | Do scenario contracts, ODD boundaries, seeds, exclusions, and feasibility checks match the intended question? | Verification plus domain judgment | `covered`, `partial`, `missing`, `excluded` |
| Model assumptions | Are pedestrian, robot, sensor, actuator, observation, and planner-interface assumptions named? | Boundary evidence | `covered`, `partial`, `missing` |
| Uncertainty and sensitivity | Are seed variance, confidence, denominator health, perturbation sensitivity, and non-identifiable comparisons reported? | Verification plus statistical evidence | `covered`, `partial`, `missing` |
| Fallback/degraded handling | Are fallback, degraded, failed, not-available, and adapter rows excluded or caveated? | Verification | `covered`, `partial`, `missing` |
| Real-world validation | Is there empirical, hardware, external-simulator, field, or human-factors evidence for the modeled behavior? | Validation | `covered`, `partial`, `missing`, `out_of_scope` |
| Known invalid domains | Are domains where the result must not be used named near the claim? | Boundary evidence | `covered`, `partial`, `missing` |

Promotion rule: the maximum claim strength is limited by the weakest relevant row. For example,
`missing` real-world validation still allows a diagnostic or internal benchmark claim when clearly
labeled, but blocks sim-to-real, deployment, and paper-facing realism language.

## Compact Template

Copy this into campaign reports, release notes, context notes, or safety-case mappings when a full
table would be too heavy.

```md
### Simulation Model Credibility

| Field | Answer |
| --- | --- |
| Campaign / evidence id | `<issue, campaign id, or evidence path>` |
| Intended claim | `<bounded claim>` |
| Evidence status | `diagnostic-only` / `smoke evidence` / `nominal benchmark evidence` / `paper-grade` |
| Non-claims | `<real-world validation, safety certification, deployment readiness, etc.>` |
| Artifact provenance | `<configs, command, commit, durable evidence path, checksums>` |
| Verified implementation behavior | `<tests, fixtures, schema checks, fail-closed proof>` |
| Scenario / ODD validity | `<contracts, seeds, feasibility, exclusions>` |
| Model assumptions | `<pedestrian, robot, sensor, actuator, observation, planner-interface assumptions>` |
| Uncertainty / sensitivity | `<seed count, confidence, denominator health, perturbation findings, non-identifiable states>` |
| Real-world validation | `<covered/partial/missing/out_of_scope + source or blocker issue>` |
| Known invalid domains | `<domains and claim wordings this evidence must not support>` |
| Decision | `continue` / `revise` / `block` / `external` |
| Follow-up issues | `<links to blockers or next proof steps>` |
```

## Pilot Mapping: Issue #3207 Fidelity Sensitivity

This pilot maps the current #3207 fidelity-sensitivity evidence through the checklist. It does not
run new experiments or change #3207's evidence tier.

| Field | Assessment |
| --- | --- |
| Campaign / evidence id | [Issue #3207](https://github.com/ll7/robot_sf_ll7/issues/3207), especially the tracked [launch packet](evidence/issue_3207_fidelity_sensitivity_launch_packet_2026-06-20/README.md), [diagnostic smoke](evidence/issue_3207_fidelity_sensitivity_smoke_2026-06-20/README.md), and [actual slice](evidence/issue_3207_fidelity_sensitivity_actual_slice_2026-06-20/README.md). |
| Intended claim | Internal simulator-fidelity sensitivity on the bounded local slice: how selected metrics and rank calculations behave under timestep, pedestrian-archetype, observation-noise, and clearance-radius perturbations. |
| Evidence status | `diagnostic-only` / bounded actual-slice evidence. |
| Non-claims | Not simulator-realism evidence, not sensor-realism evidence, not sim-to-real validation, not full #3207 acceptance evidence, not paper-facing planner-ranking evidence. |
| Artifact provenance | Tracked evidence bundles under `docs/context/evidence/issue_3207_*`; raw rows remain local `output/` regeneration material, not durable proof. |
| Verified implementation behavior | `partial`: launch-packet, manifest, rank-stability, metric-drift, and report-generation paths have focused tests and tracked summaries, but the full fixed-scope sweep has not been completed. |
| Numerical implementation | `partial`: timestep perturbations are deliberate sensitivity axes, and rank/metric helpers have tests. There is no independent numerical convergence proof for the Social Force dynamics. |
| Fixture and regression tests | `partial`: helper/report tests cover the current compact path; broader benchmark and fixed-scope campaign proof remains open under #3207. |
| Scenario / ODD validity | `partial`: uses `configs/scenarios/sets/paper_cross_kinematics_v1.yaml` on a compact two-planner local slice. It does not cover the full scenario/planner matrix needed for stronger claims. |
| Model assumptions | `partial`: Social Force, timestep, observation noise, pedestrian archetypes, and clearance radius are named as simulation assumptions and perturbation axes; they are not externally calibrated here. |
| Uncertainty / sensitivity | `partial`: three seeds and multiple perturbation axes are present in the bounded slice, but the actual slice reports rank evidence as `non-identifiable` because the primary success-rate metric has zero variance. |
| Fallback/degraded handling | `covered`: evidence language excludes fallback/degraded and raw-local-output promotion; #3207 is framed as deliberate sensitivity probing, not fallback success. |
| Real-world validation | `missing`: no empirical pedestrian, sensor, actuation, or field correspondence evidence is supplied by #3207. External simulator or sensor-realism validation is tracked separately by [#3028](https://github.com/ll7/robot_sf_ll7/issues/3028), and AMV actuation calibration by [#1559](https://github.com/ll7/robot_sf_ll7/issues/1559). |
| Known invalid domains | `covered`: the evidence must not be cited for deployment safety, paper-grade ranking, real-world realism, full benchmark ranking, hardware actuation validity, or sensor-performance validity. |
| Decision | `revise`: keep the result as bounded diagnostic/sensitivity evidence until the non-identifiable rank state and full sweep blockers are resolved. Use `external` for any real-world validation claim. |
| Follow-up issues | [#3299](https://github.com/ll7/robot_sf_ll7/issues/3299) for zero-variance rank classification, [#3207](https://github.com/ll7/robot_sf_ll7/issues/3207) for the full fidelity-sensitivity study, [#3028](https://github.com/ll7/robot_sf_ll7/issues/3028) for external simulator/sensor realism, [#1559](https://github.com/ll7/robot_sf_ll7/issues/1559) for AMV calibration, and [#3081](https://github.com/ll7/robot_sf_ll7/issues/3081) for release-package synthesis. |

## Release And Campaign Linkage

When evidence is promoted, link the filled credibility checklist from the same surface that links
the campaign report or release packet. For Robot SF reports, that usually means one of:

- the campaign context note under `docs/context/`;
- the compact evidence bundle under `docs/context/evidence/`;
- a release package or research synthesis issue such as #3081;
- the credibility row in [Simulation-Evidence Safety Case Template](../simulation_evidence_safety_case.md).

Do not cite this checklist as proof that a campaign is credible. Cite the filled assessment plus
the underlying commands, configs, artifacts, and validation evidence.
