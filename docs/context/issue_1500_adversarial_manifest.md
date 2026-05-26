# Issue #1500 Adversarial Campaign Manifest Freeze (2026-05-26)

Date: 2026-05-26

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1500> (this manifest)
- <https://github.com/ll7/robot_sf_ll7/issues/1488> (parent umbrella)
- <https://github.com/ll7/robot_sf_ll7/issues/1501> (next: one-family smoke)
- <https://github.com/ll7/robot_sf_ll7/issues/1433> (search design)
- <https://github.com/ll7/robot_sf_ll7/issues/1457> (generation protocol)
- <https://github.com/ll7/robot_sf_ll7/issues/1434> (stress-coverage schema)
- <https://github.com/ll7/robot_sf_ll7/issues/1237> (failure archive)
- <https://github.com/ll7/robot_sf_ll7/issues/691> (fallback policy)

## Goal

Freeze scenario families, seeds, search engines, budgets, replay determinism checks, and
artifact layout for the bounded adversarial comparison campaign in
[issue #1488](https://github.com/ll7/robot_sf_ll7/issues/1488). This manifest is a
**specification artifact**; it does not run evaluations, emit dry-run outputs, or produce
benchmark evidence.

## Manifest Artifact

The frozen manifest lives at:

```
configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml
```

It is tracked in git and is the durable, reviewable contract for child stages
[issue #1501](https://github.com/ll7/robot_sf_ll7/issues/1501),
[issue #1502](https://github.com/ll7/robot_sf_ll7/issues/1502), and
[issue #1503](https://github.com/ll7/robot_sf_ll7/issues/1503).

## Frozen Decisions

### Scenario Families (2)

| Family | Template/Source | Map | Search Engines |
|---|---|---|---|
| `crossing_ttc` | `configs/scenarios/templates/crossing_ttc.yaml` | `classic_cross_trap` | random, optuna_tpe |
| `classic_head_on_corridor` | `configs/scenarios/classic_interactions.yaml` (id: `classic_head_on_corridor_low`) | `classic_head_on_corridor` | guided_route_search |

**Unavailable-by-design**: random and TPE search on `classic_head_on_corridor` are
`not_available` because no CandidateSpec-compatible search space or parametric template
exists for route-level graph search on this family. Guided route search on `crossing_ttc` is
`not_available` because `generate_adversarial_routes.py` uses the route override paradigm
which does not map to the parametric crossing/TTC template.

### Search Engines (3)

| Engine | Class | Objective | Global Seed | Budget (local) | Families |
|---|---|---|---|---|---|
| `random` | `RandomCandidateSampler` | `worst_case_snqi` | 42 | 32 | crossing_ttc |
| `optuna_tpe` | `OptunaCandidateSampler` | `worst_case_snqi` | 42 | 32 | crossing_ttc |
| `guided_route_search` | `OptunaBasedRouteSearch` | `composite` | 123 | 20 | classic_head_on_corridor |

Total campaign budget (local): 84 candidate evaluations. SLURM budget: 612.

### Planner Rows (2)

| Planner | Policy | Algorithm | Readiness | Mode |
|---|---|---|---|---|
| `classic_global_theta_star` | `ClassicGlobalPlanner` | theta_star_v2 | native | native |
| `orca` | `ORCA` | - | adapter | adapter |

### Replay Determinism Checks (3)

1. **manifest_materialization**: Re-materialize scenario YAML from manifest; compare byte-for-byte.
2. **seed_determinism**: Re-evaluate archived failures with same seeds; compare objective and
   collision/timeout flags.
3. **search_trajectory**: Re-run sampler from frozen global seed; verify identical candidate
   sequence.

### Artifact Layout

All run outputs under `output/adversarial/campaign/`:

```
output/adversarial/campaign/
  crossing_ttc/
    random/                        # manifest.json + candidate bundles
    optuna_tpe/                    # manifest.json + candidate bundles
    archive.json                   # adversarial_failure_archive.v1
  classic_head_on_corridor/
    guided_route_search/           # manifest.json + candidate bundles
    archive.json                   # adversarial_failure_archive.v1
  reports/
    crossing_ttc_stress_coverage.json
    classic_head_on_corridor_stress_coverage.json
```

Artifact policy (from [issue #1433](https://github.com/ll7/robot_sf_ll7/issues/1433)):
raw outputs in `output/` are git-ignored and worktree-local.
Small reviewable evidence copies may be promoted to `docs/context/evidence/` when they
support a design decision or bug report. Do not commit raw search outputs.

## Row Classification Contract

Every candidate evaluation is classified into exactly one row type:

| Row Type | Archive Eligible | Evidence Class | Description |
|---|---|---|---|
| `valid_behavioral_failure` | yes | `development_stress_test` | Collision, near_miss, timeout, comfort_violation |
| `success` | no | `budget_audit_only` | Episode completed without failure |
| `invalid_candidate` | no | `budget_audit_only` | Search-space/constraint violation |
| `simulation_error` | no | `budget_audit_only` | Simulator exception; explicit exclusion from archive and benchmark-style interpretation |
| `fallback` | no | `not_benchmark_evidence` | Planner entered fallback mode |
| `degraded` | no | `not_benchmark_evidence` | Planner deviated from contract |
| `not_available` | no | `exclusion` | Engine x family design exclusion |

Per [issue #691](https://github.com/ll7/robot_sf_ll7/issues/691): fallback and degraded rows are
explicitly **not** benchmark evidence. Per
[issue #1433](https://github.com/ll7/robot_sf_ll7/issues/1433): generated adversarial cases are
`generated_cases_are_benchmark_evidence: false` and must carry
`evidence_class: development_stress_test`. `simulation_error` and `not_available` rows remain
explicit exclusion classes, not successful comparisons.

## Comparison Caveats

- Crossing/TTC and head-on corridor use different search paradigms (parametric CandidateSpec
  vs. route-level graph optimization). Direct cross-family comparison of absolute failure counts
  is not valid.
- Within the crossing/TTC family, random vs. TPE is the intended within-family comparison axis
  because both engines share the same search space, objective, budget, and planner rows. This
  manifest does not claim any observed superiority.
- Guided route search on head-on corridor is a single-engine baseline. Cross-engine comparison
  with random/TPE requires a parametric search space for this family (future work).
- Coverage and uncertainty reporting (`stress_uncertainty_coverage.v1`) is not yet required at
  this stage; it is deferred to
  [issue #1503](https://github.com/ll7/robot_sf_ll7/issues/1503).

## Dependency Relationship

- **Parent**: [issue #1488](https://github.com/ll7/robot_sf_ll7/issues/1488) (adversarial comparison umbrella).
- **Next child**: [issue #1501](https://github.com/ll7/robot_sf_ll7/issues/1501) (one-family smoke across all engines over crossing/TTC).
- **Blockers**: none. This manifest is self-contained.
- **Blocked by**: none. All referenced configs, maps, and code surfaces exist on `origin/main`.

## Non-Execution Guarantees

This manifest stage:
- Does **not** run simulations, search engines, or planners.
- Does **not** produce `output/` artifacts, dry-runs, or launch packets.
- Does **not** interpret or extrapolate failure-mode diversity.
- Does **not** treat any absent row, simulation error, or unavailable engine as success.
- Does **not** change benchmark semantics, metrics, runner code, seeds outside the manifest, or
  paper claims.

## Provenance

- Manifest frozen from configs, templates, and code surfaces at the current
  HEAD.
- All referenced config files and maps exist and are unmodified relative to the manifest.
- Search-space bounds, budget rules, and seed policies inherit from
  [issue #1433](https://github.com/ll7/robot_sf_ll7/issues/1433) design.
- Row classification contract aligns with
  [issue #691](https://github.com/ll7/robot_sf_ll7/issues/691),
  [issue #1433](https://github.com/ll7/robot_sf_ll7/issues/1433), and
  [issue #1237](https://github.com/ll7/robot_sf_ll7/issues/1237).
- The `generated_cases_are_benchmark_evidence: false` marker in the crossing/TTC template
  carries through to all search outputs.

## Follow-Up

The next implementation stage
([issue #1501](https://github.com/ll7/robot_sf_ll7/issues/1501)) should:
1. Load this manifest.
2. Run the crossing/TTC family with random and TPE search engines.
3. Emit `adversarial-search-manifest.v1` with explicit row classification.
4. Archive failures into `adversarial_failure_archive.v1`.
5. Report per-row counts and determinism-check results.
6. Update this context note with smoke evidence and any manifest revisions.

Coverage/stress report synthesis
([issue #1503](https://github.com/ll7/robot_sf_ll7/issues/1503)) should not begin before
[issue #1501](https://github.com/ll7/robot_sf_ll7/issues/1501) and
[issue #1502](https://github.com/ll7/robot_sf_ll7/issues/1502) complete.
