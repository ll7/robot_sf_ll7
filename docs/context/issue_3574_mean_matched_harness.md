# Issue #3574 Mean-Matched Harness

This note records the dry-run harness added for issue #3574: it builds paired
heterogeneous and homogeneous mean-matched rows before any benchmark campaign runs.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- The manifest is a pre-run contract, not a heterogeneous-population effect result.
- It attributes future rows by scenario, seed, planner, density, and population arm.
- It records the expected `pedestrian_control_trace` fields needed by the existing
  per-archetype metric primitives: surface clearance (`clearance_m`) and
  near-field exposure duration (`near_field_exposure_s`). The trace records the
  configured surface-clearance threshold alongside those fields; the manifest blocks
  near-field exposure rows when that threshold metadata is absent.
- The runtime trace path is `algorithm_metadata.pedestrian_control_trace`. Before report
  analysis, `mean_matched_episode_readiness.v1` requires every manifest row exactly once and
  checks each simulator-indexed archetype/control label, every declared trace metric field, and
  the finite episode-level `metrics.mean_clearance` value consumed by rank sensitivity.
- It does not run a full ablation campaign, response-law mixture sweep, rank-order
  sensitivity analysis, Slurm job, or paper/dissertation claim update.

## Entrypoint

```bash
uv run python scripts/benchmark/build_heterogeneity_ablation_manifest_issue_3574.py \
  --config configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml \
  --output output/issue_3574_mean_matched_harness/manifest.json
```

The output uses schema `mean_matched_heterogeneity_harness.v1`. Missing episode
trace inputs are fail-closed blockers until a future run supplies stable
per-pedestrian archetype labels and per-step metric values.

## Scenario-Matrix Extension (#5504)

The harness can now ingest the canonical robot_sf.scenario_matrix.v1 YAML format so each
generated row runs on the map and pedestrian density declared by its own matrix cell.
Configure this with scenario_matrix plus a required scenario_matrix_derivation block:

```yaml
scenario_matrix: configs/scenarios/classic_interactions_francis2023.yaml
scenario_matrix_derivation:
  population_size: 12
  composition: {cautious: 0.25, standard: 0.5, hurried: 0.25}
  archetypes: { ... }
```

The derivation block is fixed across matrix cells and must explicitly provide the population size,
composition, archetype parameters, and any seeds used by the harness. The matrix cell supplies
map_file and simulation_config.ped_density; neither value is invented or silently inherited.
Response-law fractions and the two mean-matched population arms remain the only harness treatment
axes. Legacy inline scenarios configs remain unchanged and omit additive row map fields.

The two-geometry CPU smoke uses
configs/benchmarks/issue_5504_mean_matched_harness_scenario_matrix_smoke.yaml:

```bash
uv run python scripts/benchmark/build_heterogeneity_ablation_manifest_issue_3574.py \
  --config configs/benchmarks/issue_5504_mean_matched_harness_scenario_matrix_smoke.yaml \
  --output output/issue_5504_mean_matched_harness/manifest.json
uv run python scripts/benchmark/run_heterogeneous_population_ablation_issue_3574.py \
  --manifest output/issue_5504_mean_matched_harness/manifest.json \
  --output output/issue_5504_mean_matched_harness/episode_records.jsonl
```

Rows from matrix-derived manifests carry their own map_file; the runner fails closed when a
legacy inline row lacks a map unless the caller supplies the explicit --legacy-map fallback.
This remains harness smoke/integration proof only: it is not a full benchmark campaign or a
paper-facing result.

The report command consumes both the manifest and episode records. It writes
`integration_readiness.json` and exits with status 2 before metric or rank analysis when rows are
missing, duplicated, unexpected, or lack aligned trace metadata:

```bash
uv run python scripts/benchmark/build_heterogeneous_population_ablation_report.py \
  --manifest output/issue_3574_mean_matched_harness/manifest.json \
  --records output/issue_3574_mean_matched_harness/episode_records.jsonl
```

## Integration Contract Update (2026-07-11)

The report now carries a coherent per-archetype result for every trace metric declared in the
paired manifest, currently `clearance_m` and `near_field_exposure_s`. Its additive
`per_archetype_metric_reports` summary field is keyed first by metric and then by the paired
scenario/seed/planner triplet. The existing `ablation_reports` field remains the `clearance_m`
view for compatibility.

This joins the already-landed pieces without turning fixture output into evidence:

- the dry-run manifest defines the two mean-matched population arms, three planners, and common
  seeds;
- episode readiness requires every expected row once, aligned per-pedestrian labels, finite trace
  fields, and finite `metrics.mean_clearance` before analysis;
- the report produces per-archetype summaries for all requested trace fields; and
- the separate preregistered rank-reversal CLI consumes the same readiness-checked records.

### Blocker Inventory

- **Remaining:** no attributed mean-matched paired campaign records exist, so there is no effect,
  rank-stability, or realized-distribution conclusion.
- **New:** none observed by the fixture integration check.
- **Intentional:** this slice does not run a benchmark campaign, submit Slurm/GPU work, or promote
  a paper/dissertation claim.

### Next Empirical Action

Run the 72-row tracked manifest across its common scenario, seed, planner, response-law-fraction,
and arm keys. Retain the episode records with aligned traces for both declared metrics. Then run
the report command above,
the preregistered rank-reversal command in
[the rank-reversal note](issue_3574_rank_reversal_test.md), and the configured-versus-realized
distribution audit on those same traces. Interpret results only after the paired records, durable
provenance, and confidence-bound review are complete.
