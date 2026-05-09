# Issue #1075 Robot-SF Operating Envelope And Non-Claims

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1075>

Claim-language audit: [Issue #1052 Claim-Language Audit](issue_1052_claim_language_audit.md)

Scenario certification: [docs/scenario_certification.md](../scenario_certification.md)

CARLA transfer boundary: [Issue #928 CARLA T0/T1 Oracle Replay Contract](issue_928_carla_t0_t1_replay_contract.md)

Two-horizon roadmap: [docs/plan/plan_big_picture_2026-04-30.md](../plan/plan_big_picture_2026-04-30.md)

## Purpose

This note defines the current `robot_sf` operating envelope for dissertation-floor use. It is a
claim-boundary document: it says what current Robot-SF evidence can support, what it cannot support,
and which future work must happen before stronger empirical or transfer claims are appropriate.

The intended use is conservative. `robot_sf` can support illustrative framework evidence and
benchmark-set analysis when the command path, scenario set, planner mode, metrics, and artifacts are
traceable. It does not, by itself, license full empirical validation, physical-robot deployment
claims, CARLA transfer claims, or broad out-of-distribution generalization claims.

## Current Operating Envelope

Robot-SF evidence is currently strongest inside this envelope:

| Dimension | Current envelope | Required caveat |
| --- | --- | --- |
| Simulator | Robot-SF simulation and benchmark runner paths in this repository. | Higher-fidelity simulator transfer remains future work. |
| Scenario source | Maintained scenario manifests, certified or schema-validated scenario sets, and documented context fixtures. | Scenario validity is not the same as planner success or real-world representativeness. |
| Planner support | Implemented planners with documented native/adapter/fallback mode and dependency status. | Fallback, degraded, failed, or not-available rows are caveats or exclusions, not successful benchmark evidence. |
| Metrics | Repository benchmark metrics, episode schema, aggregate reports, SNQI diagnostics, and documented h500 sensitivity tables. | Metric conclusions require the matching schema, command, seed/bootstrap, and artifact provenance. |
| Evidence role | Dissertation-floor framework illustration, benchmark-set comparison, failure attribution, and reproducible workflow demonstration. | These are not physical deployment, field-study, or external-validity proofs. |

Within this envelope, acceptable claims should use language such as:

- benchmark-set performance,
- evaluated scenario matrix,
- scenario-certification or scenario-schema eligibility,
- planner mode and dependency availability,
- seed/bootstrap or trace-backed evidence where recorded,
- illustrative framework evidence for a dissertation-floor argument.

## Supported Evidence Types

The following evidence types can support dissertation-floor discussion when they are linked to
committed configs, commands, or durable notes:

- Scenario and benchmark contracts:
  `docs/scenario_certification.md`, scenario matrix configs under `configs/scenarios/`, and
  benchmark configs under `configs/benchmarks/`.
- Benchmark execution evidence:
  episode JSONL, aggregate reports, campaign summaries, SNQI diagnostics, and compact evidence
  bundles under `docs/context/evidence/` when retained there deliberately.
- Planner-readiness evidence:
  native/adapter mode, dependency status, fail-closed behavior, and explicit fallback/degraded
  caveats.
- Failure-attribution evidence:
  route/geometry sanity, scenario certification labels, trace-backed h500 mechanism notes, and
  reproducible counterexample or episode artifacts.
- Documentation evidence:
  issue execution notes under `docs/context/` that name command paths, validation status, and known
  limits.

These surfaces are enough to show how the framework structures evidence and where benchmark claims
are currently grounded. They are not enough to claim field validity or simulator-transfer parity.

## Unsupported Claims

Do not use Robot-SF evidence alone to claim:

- physical AMV deployment readiness,
- safety in a real pedestrian environment,
- CARLA, ROS, or external-simulator parity,
- transfer to unseen environments,
- out-of-distribution generalization,
- superiority of a planner family beyond the evaluated scenario matrix,
- architecture-causality for learned policy gains without a dedicated ablation,
- calibrated real sensor robustness from synthetic observation noise alone,
- multi-robot or multi-AMV behavior unless a dedicated scenario and metric contract exists.

If stronger wording is needed, it requires a separate study with the relevant proof surface, not a
reinterpretation of current Robot-SF benchmark evidence.

## Future-Work Boundaries

The following work remains outside the current dissertation-floor operating envelope:

- **CARLA transfer.** The current repository has T0 export and optional-runtime guard surfaces, and
  #1003 tracks the first narrow T1 smoke slice. CARLA parity still requires a CARLA-capable run,
  certified scenario replay, trajectory-level metric comparison, and explicit `oracle-replay`,
  `failed`, or `not-available` status handling.
- **Physical prototype validation.** No current Robot-SF benchmark result proves physical AMV
  execution, hardware safety, controller latency, perception performance, or deployment readiness.
- **Broader empirical campaigns.** Additional scenario matrices, real-world pedestrian datasets,
  multi-AMV support, or field data can extend the evidence base only after their data provenance,
  metrics, licensing, and reproducibility contracts are documented.
- **Full h500 interpretation.** Long-horizon h500 evidence can support route-budget sensitivity and
  mechanism hypotheses when trace-backed, but it should not be collapsed into a single winner table
  without exposure, safety, fallback, and runtime caveats.

## Routing Rule

When new work claims to expand the envelope, route it to the proof surface it actually needs:

| Proposed stronger claim | Required next proof |
| --- | --- |
| CARLA transfer or simulator parity | CARLA replay on a CARLA-capable machine, with certified T0 input and comparable trajectory metrics. |
| Physical AMV validity | Hardware/prototype execution evidence, controller and perception contracts, and field or lab protocol documentation. |
| OOD or unseen-environment generalization | Held-out scenario design, preregistered split, seed/bootstrap evidence, and artifact provenance. |
| Real-world pedestrian validity | Dataset provenance/licensing, scenario mapping, metric contract, and reproducible import path. |
| Multi-AMV behavior | Multi-robot scenario contract, inter-robot metrics, and dedicated benchmark slices. |

Until that proof exists, keep claims inside the current Robot-SF simulation and benchmark-set
envelope.

## Validation Notes

Checked on 2026-05-09:

- Read #1075 in full, including comments.
- Reviewed repo-local claim-boundary surfaces:
  `docs/code_review.md`, `docs/benchmark_camera_ready.md`,
  `docs/benchmark_artifact_publication.md`, `docs/scenario_certification.md`,
  `docs/context/issue_1052_claim_language_audit.md`,
  `docs/context/issue_868_scenario_certification.md`, and
  `docs/context/issue_928_carla_t0_t1_replay_contract.md`.
- Confirmed the issue-referenced external source path
  `ll7/diss:docs/superpowers/specs/2026-05-08-dissertation-narrative-framework-thesis-floor-design.md`
  was not present in this checkout; this note therefore relies on the issue body plus repo-local
  canonical docs.

Validation commands for this PR should include:

```bash
rtk rg -n "operating envelope|non-claims|CARLA transfer|physical AMV|dissertation-floor" docs
rtk git diff --check
```
