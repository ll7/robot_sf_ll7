# Scenario Zoo Index

[Back to Documentation Index](../README.md)

This index is a discoverability layer for maintained and emerging scenario families. It does not
promote a scenario into a benchmark suite by itself. Treat each row as documented scenario
surface area unless it points to a benchmark config, certification, and executed evidence.

For the evidence boundary, keep these surfaces separate:

- Authored intent: [Scenario Contracts](../scenario_contracts.md).
- Feasibility and eligibility: [Scenario Certification](../scenario_certification.md).
- Paper-facing benchmark definitions: [Benchmark Spec](../benchmark_spec.md) and
  [Camera-ready Benchmark Workflow](../benchmark_camera_ready.md).
- Fallback and degraded execution policy:
  [Issue #691 Benchmark Fallback Policy](../context/issue_691_benchmark_fallback_policy.md).

## Draft Authoring

Use the v1 authoring tools when a contributor needs a small, deterministic YAML skeleton instead of
copying a large existing scenario by hand:

```bash
uv run python scripts/tools/create_scenario.py \
  --template bottleneck \
  --name draft_bottleneck_review \
  --output configs/scenarios/single/draft_bottleneck_review.yaml

uv run python scripts/tools/validate_scenario.py \
  configs/scenarios/single/draft_bottleneck_review.yaml
```

The authoring validator checks required draft fields, map references, and seed metadata, then reuses
the maintained scenario loader and map/config builder. A passing draft is reviewable and loadable,
but it is not certified and is not benchmark evidence.

## Atlas Generation

For a generated scenario overview with thumbnails, mechanism cards, coverage
gaps, and a checksum manifest, use the scenario atlas workflow documented in
[Scenario Thumbnails and Montage](../scenario_thumbnails.md#scenario-atlas).
The atlas complements this hand-maintained zoo index. It should be read as a
discoverability artifact unless the row separately links certification and
executed benchmark evidence.

## Classic Archetype Density / Tier Index

The classic interaction archetypes parameterize pedestrian density unevenly (different tier coverage
per archetype, different density bands, and an overloaded `ped_density: 0.0` placeholder for
marker-spawn configs). For a single machine-readable description of the *current* per-archetype tier
coverage, density bands, and pedestrian spawn modes, see
[classic_density_tier_index.yaml](../../configs/scenarios/archetypes/classic_density_tier_index.yaml).

It clarifies the scenario denominator — the in-matrix graded total is **23 rows across 11 configs**,
not "12 archetypes × 3 densities" — and documents that `ped_density: 0.0` in `spawn_mode: markers`
configs is a placement-mode placeholder (pedestrians come from fixed markers), not an empty scene.
The index is a derived, documentation-only artifact and does not change scenario generation; it is
kept in sync with the configs by `tests/test_classic_archetype_density_index.py` (issue #3725).

## Families

| Family | Scenario ids / examples | Maps and configs | Agents / actors | Difficulty | Known failure modes | Recommended seed / source | Example command or links |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bottlenecks | `classic_bottleneck_low`, `classic_bottleneck_medium`, `classic_bottleneck_high`, `classic_realworld_double_bottleneck_high` | [classic_bottleneck.yaml](../../configs/scenarios/archetypes/classic_bottleneck.yaml), [classic_realworld_bottleneck.yaml](../../configs/scenarios/archetypes/classic_realworld_bottleneck.yaml), [classic_bottleneck.svg](../../maps/svg_maps/classic_bottleneck.svg), [classic_bottleneck_medium.svg](../../maps/svg_maps/classic_bottleneck_medium.svg), [classic_bottleneck_high.svg](../../maps/svg_maps/classic_bottleneck_high.svg), [classic_realworld_bottleneck.svg](../../maps/svg_maps/classic_realworld_bottleneck.svg) | Single robot plus bidirectional pedestrians or explicit opposing `single_pedestrians` in the real-world-inspired variant. | Medium to high. Narrow passage constraints and the real-world double bottleneck need longer horizons. | Stalls, conservative yielding, route-budget timeouts, static-clearance rejection, and crowd-pressure near misses. | Scenario-local seeds are `[131, 132, 133]` for the classic variants and `[201, 202, 203]` for the double bottleneck. Benchmark configs often override with `eval` seeds from [seed_sets_v1.yaml](../../configs/benchmarks/seed_sets_v1.yaml). | `uv run robot_sf_bench validate-config --matrix configs/scenarios/classic_interactions.yaml`; see [route_clearance_certifications_v1.yaml](../../configs/benchmarks/route_clearance_certifications_v1.yaml), [Issue #1344 AMV report](../context/issue_1344_paired_amv_protocol_report.md), and [h500 policy-search analysis](../context/policy_search/reports/2026-05-05_full_matrix_h500_analysis.md). |
| Crossings and cross-traps | `classic_cross_trap_low`, `classic_cross_trap_medium`, `classic_cross_trap_high`, `classic_urban_crossing_medium`, deprecated aliases `classic_crossing_*`, `single_ped_crossing_orthogonal` | [classic_cross_trap.yaml](../../configs/scenarios/archetypes/classic_cross_trap.yaml), [classic_urban_crossing.yaml](../../configs/scenarios/archetypes/classic_urban_crossing.yaml), [classic_crossing.yaml](../../configs/scenarios/archetypes/classic_crossing.yaml), [issue_596_dynamic.yaml](../../configs/scenarios/archetypes/issue_596_dynamic.yaml), [classic_crossing.svg](../../maps/svg_maps/classic_crossing.svg), [classic_urban_crossing.svg](../../maps/svg_maps/classic_urban_crossing.svg), [classic_cross_trap_subset.yaml](../../configs/scenarios/sets/classic_cross_trap_subset.yaml) | Single robot, bidirectional crossing flows, and optional single-pedestrian orthogonal crossing fixtures. | Low to high. `classic_cross_trap_high` is a stress slice; `single_ped_crossing_orthogonal` is a simpler validation fixture. | Crossing deadlock, local-minimum circulation, route-progress stalls, first-step collision when spawned too close to pedestrians, and horizon sensitivity. | Scenario-local cross-trap seeds are `[101, 102, 103]`; policy-search stress and collision slices use [stress_slice_seeds.yaml](../../configs/policy_search/stress_slice_seeds.yaml) and [leader_collision_slice_h500_seeds.yaml](../../configs/policy_search/leader_collision_slice_h500_seeds.yaml). | `uv run robot_sf_bench run --matrix configs/scenarios/sets/classic_cross_trap_subset.yaml --algo goal --out output/benchmarks/cross_trap_smoke/episodes.jsonl --workers 1 --no-resume`; see [Issue #596 atomic matrix](../context/issue_596_atomic_scenario_matrix.md), [Issue #1105 route-clearance note](../context/issue_1105_route_clearance_certification.md), and [Issue #692 difficulty analysis](../context/issue_692_scenario_difficulty_analysis.md). |
| AMV actuation diagnostics | `classic_overtaking_medium`, `classic_bottleneck_high`, `classic_cross_trap_high`, `francis2023_blind_corner`, `francis2023_intersection_wait`; policy-search smoke filters `classic_cross_trap_high` | [issue_1556_amv_actuation_stress_slice_v0.yaml](../../configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml), [funnel.yaml](../../configs/policy_search/funnel.yaml), [classic_interactions_francis2023.yaml](../../configs/scenarios/classic_interactions_francis2023.yaml), [actuation_aware_hybrid_rule_v0.yaml](../../configs/policy_search/candidates/actuation_aware_hybrid_rule_v0.yaml) | Single AMV-like robot rows with synthetic acceleration, yaw-rate, latency, and update-rate constraints; primary planner rows are `goal`, `social_force`, and `orca` in the benchmark config. | Diagnostic stress. It exercises AMV-like actuation envelopes, but is not calibrated hardware evidence. | Actuation clipping, delayed or held commands, initial overlap exclusion, conservative first-step rejection, and low stress success on primary rows. | The Issue #1556 config uses `eval` seeds from [seed_sets_v1.yaml](../../configs/benchmarks/seed_sets_v1.yaml). The policy-search smoke uses seed `[111]` from [funnel.yaml](../../configs/policy_search/funnel.yaml). | `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml --mode preflight --label issue1556-amv-preflight --output-root output/benchmarks/issue1556 --log-level WARNING`; see [Issue #1744 latency preflight](../context/issue_1744_latency_stress_preflight_contract.md), [Issue #1344 paired AMV report](../context/issue_1344_paired_amv_protocol_report.md), and [AMV smoke candidate report](../context/policy_search/reports/2026-05-31_actuation_aware_hybrid_rule_v0_amv_actuation_smoke.md). |
| Adversarial stress search | `crossing_ttc_template`, generated `crossing_ttc` candidates, `classic_head_on_corridor_low` route-search candidates | [crossing_ttc.yaml](../../configs/scenarios/templates/crossing_ttc.yaml), [crossing_ttc_space.yaml](../../configs/adversarial/crossing_ttc_space.yaml), [issue_1500_adversarial_comparison_manifest.v1.yaml](../../configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml), [default.yaml](../../configs/adversarial_routes/default.yaml), [classic_interactions.yaml](../../configs/scenarios/classic_interactions.yaml), [classic_head_on_corridor.svg](../../maps/svg_maps/classic_head_on_corridor.svg) | Scripted adversarial pedestrian candidates or route/start-state search against a planner row such as `classic_global_theta_star` or `orca`; [Issue #2468](../context/issue_2468_adversarial_generation_roadmap.md) defines the cross-method roadmap, and [Issue #2568](../context/issue_2568_adversarial_expansion_gate.md) gates RL/diffusion expansion on manifest smoke plus quality metrics. | Development stress only. Generated cases are explicitly not benchmark evidence until separately certified and promoted. | Collision, near miss, timeout, comfort violation, invalid candidate, and simulation error classifications. Fallback or degraded rows remain non-evidence. | Crossing/TTC search-space candidates draw `scenario_seed` in `[100, 999]`; the frozen comparison manifest uses global seed `42` for random/TPE and seed `123` for guided route search. | `uv run python scripts/tools/generate_adversarial_routes.py --config configs/adversarial_routes/default.yaml`; see [Issue #1433 search design](../context/issue_1433_adversarial_edge_case_search_design.md), [Issue #1457 generation protocol](../context/issue_1457_adversarial_generation_protocol.md), and [Issue #1500 manifest](../context/issue_1500_adversarial_manifest.md). |

## Benchmark Evidence Boundary

The scenario families above are easiest to find through the classic matrix, but matrix inclusion is
not the same as paper-quality evidence.

- Broad classic coverage enters through
  [classic_interactions_francis2023.yaml](../../configs/scenarios/classic_interactions_francis2023.yaml)
  and configs such as
  [paper_experiment_matrix_v1.yaml](../../configs/benchmarks/paper_experiment_matrix_v1.yaml),
  [camera_ready_baseline_safe.yaml](../../configs/benchmarks/camera_ready_baseline_safe.yaml), and
  [paper_experiment_matrix_v1_scenario_horizons_h500.yaml](../../configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml).
- Nominal, sanity, AMV, and adversarial entries are intentionally narrower. Treat their
  results according to their own `paper_facing`, `claim_scope`, `benchmark_track`, and context-note
  caveats.
- `fallback`, `degraded`, `not_available`, `simulation_error`, and generated-adversarial rows do
  not count as successful benchmark evidence unless a task explicitly exists to measure that mode.

## Adding Or Updating A Family

When adding a new row here, link the durable source files instead of copying scenario definitions:

- scenario YAML or manifest under [configs/scenarios](../../configs/scenarios/),
- benchmark or policy-search config under [configs/benchmarks](../../configs/benchmarks/) or
  [configs/policy_search](../../configs/policy_search/),
- source map or `map_id` registry entry under [maps](../../maps/),
- scenario contract, certification, or context note when the row affects benchmark interpretation,
- a validation or preflight command that can be run from the repository root.
