# Issue 1073 Robot SF Empirical-Expansion Gate

Date: 2026-05-09

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1073>

Checkpoint date: 2026-06-08

## Goal

Define the bounded Robot SF improvement plan that decides whether dissertation work should expand
from floor-level illustration into deeper empirical claims after the June 8 checkpoint.

This note is a gate, not an expansion claim. It names the work that would make expansion more
defensible, the proof required for each unit, and the decline path if the repository is not ready.

## Source Boundary

Issue #1073 cites an external dissertation-floor source spec at
`ll7/diss:docs/superpowers/specs/2026-05-08-dissertation-narrative-framework-thesis-floor-design.md`.
That file is not present in this checkout, so this gate uses the issue body plus repo-local evidence
and docs as the auditable source of truth.

The floor boundary remains:

- Robot SF can be used as the primary dissertation illustration site.
- Empirical expansion is allowed only if the improvement plan below is at least 50% implemented
  and verified by 2026-06-08.
- CARLA transfer remains dependent on #872 and #1003 and does not count as floor-scope work.
- Physical prototype work is out of scope.

## Counting Rule

The plan contains ten counted units. A unit counts at the checkpoint only when all of these are
true:

- the implementation or docs change is merged or has a PR that can be merged without unresolved
  scope gaps;
- its proof surface named below exists and matches the implemented behavior;
- validation is recorded in the PR body, a linked context note, or a compact evidence artifact;
- fallback, degraded, unavailable, or local-only output is not treated as successful evidence.

The 2026-06-08 promotion threshold is therefore at least five counted units complete. A sixth
completed unit is the recommended buffer because several units depend on pending PR review and
freshness gates.

## Current Coverage Map

| Issue or surface | Gate role | Current status | Count treatment |
|---|---|---|---|
| #691 / `docs/context/issue_691_benchmark_fallback_policy.md` | Fail-closed fallback/degraded outcome semantics. | Covered in current `main`. | Guardrail only; not a counted expansion unit by itself. |
| #1038 / `docs/context/issue_1038_h500_snqi_contract.md` | H500 SNQI contract boundary. | Covered in current `main`. | Guardrail only; prevents h500 from replacing camera-ready SNQI prematurely. |
| #1044 / `docs/context/issue_1044_h500_followup_benchmark_plan.md` | Long-horizon benchmark follow-up framing. | Covered in current `main`. | Planning predecessor; supports unit selection. |
| #1045 / `docs/context/issue_1045_h500_solvability_mechanisms.md` | Aggregate fixed-to-h500 mechanism split. | Covered in current `main`. | Selection evidence only until trace-backed by #1049 or equivalent. |
| #1049 / `docs/context/issue_1049_h500_mechanism_pilot.md` | Trace-backed h500 mechanism pilot. | Covered by retained compact traces and summary tables. | Counts toward evidence packaging only when reused by a reporting unit. |
| #1054 / `docs/context/issue_1054_planner_readiness_fallback_audit.md` | Planner readiness and fallback-mode audit. | Covered in current `main`. | Guardrail only; supports claim and exclusion rules. |
| #1055 / `docs/context/issue_1055_exposure_aware_h500_tables.md` | Exposure-aware h500 reporting tables. | Covered in current `main`. | Counted unit 2. |
| #1056 / `docs/context/issue_1056_h500_failure_classification.md` | H500 failure vocabulary and routing rules. | Covered in current `main`. | Counted unit 3. |
| #1057 / `docs/context/issue_1057_semantic_blocker_audit.md` | Semantic blocker audit before failure attribution. | Covered in current `main`; issue remains open for delivery tracking. | Supports unit 8 when linked to an accepted/merge-ready delivery PR. |
| #1058 / `docs/context/issue_1058_h500_paper_language.md` | Paper-facing h500 interpretation language. | Covered in current `main`; issue remains open for delivery tracking. | Supports unit 10 when linked to an accepted/merge-ready delivery PR. |
| #1074 | Worked-example pack for framework illustrations. | Open PR, not part of `main` at the time of this note. | Do not count until merged or accepted as merge-ready. |
| #1075 | Operating envelope and non-claims. | Open PR, not part of `main` at the time of this note. | Do not count until merged or accepted as merge-ready. |
| #1076 | Upstream AMV paper-defense backlog tracker. | Open PR, not part of `main` at the time of this note. | Do not count until merged or accepted as merge-ready. |
| #872 and CARLA T0 chain | Broad CARLA oracle replay bridge, with T0/T1 contract and T0 schema/export children. | T0 groundwork is documented in `docs/context/issue_928_carla_t0_t1_replay_contract.md` and follow-up T0 notes. | Does not count toward the floor gate. |
| #1003 | CARLA T1 oracle replay smoke slice. | Open PR, optional transfer proof only; no `main` context note yet. | May count as a transfer dependency, not as floor-scope empirical expansion. |

The CARLA T0 chain currently includes `docs/context/issue_928_carla_t0_t1_replay_contract.md`,
`docs/context/issue_930_carla_t0_export_schema.md`,
`docs/context/issue_934_carla_t0_export_builder.md`,
`docs/context/issue_940_carla_t0_export_read_helper.md`,
`docs/context/issue_942_carla_t0_map_definition_adapter.md`,
`docs/context/issue_946_carla_t0_scenario_entry_export.md`,
`docs/context/issue_948_carla_t0_scenario_file_export.md`,
`docs/context/issue_950_carla_t0_export_record_writer.md`,
`docs/context/issue_952_carla_t0_export_cli.md`,
`docs/context/issue_954_carla_t0_export_cli_packaging.md`,
`docs/context/issue_956_carla_t0_export_manifest_reader.md`,
`docs/context/issue_958_carla_t0_manifest_validation_cli.md`,
`docs/context/issue_960_carla_t0_manifest_payload_paths.md`,
`docs/context/issue_968_carla_runtime_availability_guard.md`,
`docs/context/issue_970_carla_availability_check_cli.md`, and
`docs/context/issue_972_carla_availability_cli_require_mode.md`. Those surfaces are useful
transfer prerequisites, but they stay outside the floor-scope count until CARLA replay evidence is
available and explicitly separated from Robot SF benchmark evidence.

## Counted Improvement Units

| Unit | Class | Task | Proof surface | Required validation |
|---:|---|---|---|---|
| 1 | Scenario | Canonicalize collision encoding and enforce episode schema conformance. | #1077 PR and schema/tests. | Targeted schema tests plus PR-ready gate. |
| 2 | Metric/reporting | Add exposure-aware h500 reporting tables. | `docs/context/issue_1055_exposure_aware_h500_tables.md` and retained CSVs. | `rtk column` checks on the retained tables plus `rtk git diff --check`. |
| 3 | Evidence interpretation | Classify h500 failures by scenario, time budget, and planner mechanism. | `docs/context/issue_1056_h500_failure_classification.md`. | Link/path verification against #1049 evidence plus `rtk git diff --check`. |
| 4 | Statistical reporting | Add paired bootstrap contrasts and effect sizes to aggregate outputs. | #1078 PR and aggregate-output tests. | Targeted aggregate/statistics tests plus PR-ready gate. |
| 5 | Scenario coverage | Ship a confirmation scenario matrix with semantically disjoint archetypes. | #1079 PR and config/docs coverage. | Scenario config validation plus PR-ready gate. |
| 6 | Observation contract | Declare planner observation specs and controlled observation-mode overrides. | #1080 PR and observation contract docs/tests. | Observation-spec tests plus PR-ready gate. |
| 7 | Robustness stress | Add configurable observation-noise injection for benchmark runs. | #1081 PR and benchmark/config tests. | Targeted noise/config tests plus PR-ready gate. |
| 8 | Scenario certification | Audit semantic blockers before paper failure attribution. | #1057 issue/PR and `docs/scenario_certification.md` linkage. | Scenario-certification or blocker-audit validation plus PR-ready gate. |
| 9 | Reproducibility | Publish durable paper evidence bundle archive and diagnostics pointers. | #1062 issue/PR and publication-bundle pointers. | Bundle manifest/checksum validation plus PR-ready gate. |
| 10 | Claim boundary | Define operating envelope, non-claims, and reusable h500 paper language. | #1075 and #1058 issue/PR surfaces. | Docs link verification plus PR-ready gate. |

These units are intentionally bounded to less than three weeks of focused implementation and review
time when handled as small PRs. They emphasize repo-local proof over new long-running campaigns.

## Promotion Criteria

Promote Robot SF from floor illustration to empirical-expansion candidate only if all of the
following are true on 2026-06-08:

- at least five counted units satisfy the counting rule above;
- units include at least one scenario/contract unit, one metric/reporting unit, and one
  reproducibility or claim-boundary unit;
- no counted unit depends on untracked `output/` artifacts without a durable manifest, release
  asset, W&B artifact, or compact tracked evidence copy;
- h500 language keeps completion, exposure, collision, and time-budget effects separate;
- fallback, degraded, unavailable, or adapter-blocked planner rows remain caveats or exclusions.

Recommended promotion wording:

> Robot SF has enough verified scenario, metric, and reproducibility support to justify a bounded
> empirical expansion beyond floor-level examples, while keeping simulator-transfer and physical
> deployment claims out of scope.

## Decline Criteria

Keep Robot SF at floor-level illustration only if any of these are true on 2026-06-08:

- fewer than five counted units satisfy the counting rule;
- the completed units are mostly docs-only and do not cover scenario, metric, and reproducibility
  surfaces together;
- evidence still relies on local-only `output/` contents, stale validation, or pending PRs with
  unresolved scope gaps;
- h500, CARLA, or planner-readiness wording would require fallback/degraded rows to be treated as
  successful benchmark evidence;
- CARLA transfer is the main available strengthening path but #1003 or #872 remains runtime
  unproven.

Recommended decline wording:

> Robot SF remains appropriate as the dissertation floor illustration, but empirical expansion is
> deferred because the repository has not yet verified enough scenario, metric, and reproducibility
> improvements to support stronger claims.

## Validation

This note is docs-only. Validation for the issue #1073 PR should include:

```bash
rtk bash -lc 'test -f docs/context/issue_1049_h500_mechanism_pilot.md'
rtk bash -lc 'test -f docs/context/issue_1055_exposure_aware_h500_tables.md'
rtk bash -lc 'test -f docs/context/issue_1056_h500_failure_classification.md'
rtk bash -lc 'test -f docs/context/issue_1038_h500_snqi_contract.md'
rtk bash -lc 'test -f docs/context/issue_1044_h500_followup_benchmark_plan.md'
rtk bash -lc 'test -f docs/context/issue_1045_h500_solvability_mechanisms.md'
rtk bash -lc 'test -f docs/context/issue_691_benchmark_fallback_policy.md'
rtk bash -lc 'test -f docs/context/issue_928_carla_t0_t1_replay_contract.md'
rtk bash -lc 'test -f docs/scenario_certification.md'
rtk bash -lc 'test -f docs/benchmark_artifact_publication.md'
rtk git diff --check
PYTEST_NUM_WORKERS=8 BASE_REF=origin/main rtk scripts/dev/pr_ready_check.sh
```

No helper script or benchmark campaign is introduced here, so no new runtime evidence is required
for this planning issue.
