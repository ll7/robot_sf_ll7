# Issue #2155 Research-v1 AMMV Social-Navigation Benchmark Matrix Contract

Issue: [#2155](https://github.com/ll7/robot_sf_ll7/issues/2155)
Status: current research-v1 AMV matrix contract as of 2026-06-03.

This note defines the frozen pre-execution contract for a compact
research-v1 AMV (autonomous mobility vehicle / device) social-navigation
benchmark matrix. It is a contract freezing the scenario set, planner row
families, seed policy, artifact policy, and fail-closed row semantics. It
does **not** contain benchmark results, performance numbers, or
paper-facing claims.

## Claim Boundary

This note is a **benchmark contract / pre-execution evidence** artifact.
Do not cite it as successful benchmark evidence or paper-facing support.

- Research-v1 evidence status: `proposal` in the
  [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md)
  before execution; updates to `candidate` after this contract freeze.
- Fallback, degraded, failed, or `not_available` rows are excluded from
  successful campaign evidence under
  [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md).
- No benchmark run, output artifact, or performance claim is created by
  this note.

## Scenario Families

23 scenarios in 11 compact research families across classic interaction archetypes
and Francis 2023 singles:

| Family | Archetype | Scenarios | Rationale |
|---|---|---|---|
| `bottleneck` | classic_bottleneck | low, medium | Narrow-constraint navigation |
| `doorway` | classic_doorway | low, medium | Passage negotiation |
| `head_on_corridor` | classic_head_on_corridor | low, medium | Frontal encounter |
| `overtaking` | classic_overtaking | low | Passing slower traffic |
| `cross_trap` | classic_cross_trap | low | Crossing traffic |
| `t_intersection` | classic_t_intersection | low | Junction |
| `merging` | classic_merging | low | Merge interaction |
| `group_crossing` | classic_group_crossing | low | Group interaction |
| `urban_crossing` | classic_urban_crossing | medium | Street crossing |
| `francis_basic` | Francis 2023 singles | frontal_approach, pedestrian_obstruction, blind_corner, narrow_hallway, narrow_doorway, down_path, intersection_wait, intersection_proceed | Sparse AMV-relevant interactions |
| `francis_overtaking` | Francis 2023 singles | robot_overtaking, pedestrian_overtaking, parallel_traffic | Passing variants |

Source: `configs/scenarios/issue_2155_research_v1_ammv.yaml`

## Planner Row Families

12 planner rows in 4 families:

| Family | Key | Algo | Benchmark profile | Prereq policy |
|---|---|---|---|---|
| `core` | goal | goal | baseline-safe | - |
| `core` | social_force | social_force | baseline-safe | - |
| `core` | orca | orca | baseline-safe | fallback |
| `experimental` | ppo | ppo | experimental | - |
| `experimental` | prediction_planner | prediction_planner | experimental | - |
| `experimental` | socnav_sampling | socnav_sampling | experimental | skip-with-warning |
| `experimental` | sacadrl | sacadrl | experimental | fallback |
| `experimental` | socnav_bench | socnav_bench | experimental | skip-with-warning |
| `hybrid_rule` | hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_local_planner | experimental | - |
| `hybrid_rule` | scenario_adaptive_hybrid_orca_v1 | hybrid_rule_local_planner | experimental | - |
| `predictive_v2` | prediction_planner_v2_full | prediction_planner | experimental | - |
| `predictive_v2` | prediction_planner_v2_xl_ego | prediction_planner | experimental | - |

Source: `configs/benchmarks/issue_2155_research_v1_ammv_matrix.yaml`

## Seed Policy

- Mode: `seed-set`
- Seed set: `paper_eval_s5` → [111, 112, 113, 114, 115]
- Source: `configs/benchmarks/seed_sets_v1.yaml`
- Rationale: 5 seeds provide compact seed-sensitivity coverage while staying
  small enough for local non-SLURM runs. The S3 `eval` set is extended by
  2 seeds to S5.

## Artifact Policy

- `export_publication_bundle: false`
- `include_videos_in_publication: false`
- Campaign output is research-v1 evidence only.
- Do not promote outputs to paper release without explicit resolution.
- Any generated output stays under `output/` and is worktree-local; no
  durable artifact store upload is required at this stage.

## Fallback/Degraded Fail-Closed Handling

Per [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md):

- Rows with `socnav_missing_prereq_policy: fallback` (orca, sacadrl) are
  eligible to run but their output must be classified as `fallback` and
  **excluded from successful benchmark evidence**.
- Rows with `socnav_missing_prereq_policy: skip-with-warning` (socnav_sampling,
  socnav_bench) will be skipped when their prereqs are missing.
- Any `degraded`, `failed`, or `not_available` row is also excluded from
  success.
- Campaign-level success is anchored on `core` planner rows: all core
  planners must complete in `native` mode for the campaign to be
  classified as successful.

## Canonical Pre-Execution Validation

```bash
# Validate the scenario matrix
robot_sf_bench validate-config --matrix configs/scenarios/issue_2155_research_v1_ammv.yaml

# Preview resolved scenario list
robot_sf_bench preview-scenarios --matrix configs/scenarios/issue_2155_research_v1_ammv.yaml

# Validate the benchmark config (requires campaign runner entry point)
# Benchmark-config semantics are covered by the targeted contract tests.
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/benchmark/test_issue_2155_research_v1_matrix.py -q
```

## Downstream Update Rule

When this matrix is executed, update this note with:
- command/config paths and commit SHA
- campaign-level outcome and core success/failure
- per-planner execution modes (native / fallback / degraded / skip / failed)
- durable artifact pointer or manifest
- link to the parent issue_2153 evidence map update

## Related Documents

- [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md)
- [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)
- [configs/benchmarks/issue_2155_research_v1_ammv_matrix.yaml](../../configs/benchmarks/issue_2155_research_v1_ammv_matrix.yaml)
- [configs/scenarios/issue_2155_research_v1_ammv.yaml](../../configs/scenarios/issue_2155_research_v1_ammv.yaml)
- [configs/benchmarks/seed_sets_v1.yaml](../../configs/benchmarks/seed_sets_v1.yaml)
