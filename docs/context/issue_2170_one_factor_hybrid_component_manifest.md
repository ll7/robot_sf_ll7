# Issue #2170 One-Factor Hybrid Component Ablation Manifest

Issue: [#2170](https://github.com/ll7/robot_sf_ll7/issues/2170)
Parent: [#2104](https://github.com/ll7/robot_sf_ll7/issues/2104)
Status: proposal/pre-execution contract as of 2026-06-03.

## Scope

This note records the next implementable child after the merged
[issue_2104_component_ablation_pilot.md](issue_2104_component_ablation_pilot.md) diagnostic. The
new manifest is:

- `configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml`

It freezes a compact one-factor ablation contract for leading hybrid-rule planner components. It
does not execute the benchmark matrix, create new planner code, or claim one-factor causality.

## Manifest Shape

The manifest uses the existing h500 policy-search surface but reduces the follow-up execution slice
to six scenarios with identical seeds `111`, `112`, and `113`:

- `classic_cross_trap_high`;
- `classic_merging_low`;
- `classic_bottleneck_medium`;
- `classic_head_on_corridor_low`;
- `francis2023_leave_group`;
- `francis2023_perpendicular_traffic`.

The scenario set intentionally combines leader-collision h500 scenarios with stress or regression
guards from the #1454/#2104 evidence surface. The execution contract keeps `horizon=500`, `dt=0.1`,
`benchmark_profile=experimental`, and `paper_facing=false`.

## Component Boundary

The manifest separates existing reference rows from planned one-factor rows.

Existing reference rows:

- `hybrid_rule_v3_fast_progress`;
- `hybrid_rule_v3_fast_progress_static_escape`;
- `hybrid_rule_v3_fast_progress_static_escape_continuous`;
- `scenario_adaptive_hybrid_orca_v1`;
- `scenario_adaptive_hybrid_orca_v2_collision_guard`;
- tuned ORCA for the selector reference.

Planned one-factor rows:

- static escape only;
- static recenter only;
- static escape plus recenter without corridor-transit terms;
- continuous static checks;
- scenario-adaptive ORCA selector only;
- speed/progress pressure `2.4` sensitivity.

The current `scenario_adaptive_hybrid_orca_v2_collision_guard` config is kept as a caveated
historical row because its YAML does not expose a literal `collision_guard` parameter. The later
execution issue must either resolve the actual guard mechanism or exclude it from one-factor claims.

## Claim Boundary

Classification: `proposal_pre_execution_contract`.

This work moves #2104 from a grouped retrospective diagnostic toward a runnable one-factor
protocol. It is not benchmark evidence and should not be cited as successful planner performance,
component causality, or paper-facing support. Confidence is about 0.85 that the selected manifest is
the right next local proof step because it follows the merged #2151 pilot and references existing
scenario/candidate surfaces; confidence would drop if the later candidate-loader validation cannot
materialize the planned config rows cleanly.

## Validation

Validation for this manifest update:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

manifest = yaml.safe_load(Path("configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml").read_text())
for key in ("source_context", "execution_contract", "scenario_slice", "existing_reference_rows", "planned_one_factor_rows", "comparison_plan"):
    assert key in manifest, key
for path_key in [
    ("execution_contract", "base_config_path"),
    ("execution_contract", "scenario_matrix"),
    ("execution_contract", "scenario_horizons"),
]:
    assert Path(manifest[path_key[0]][path_key[1]]).exists(), path_key
for row in manifest["existing_reference_rows"]:
    assert Path(row["config_path"]).exists(), row["key"]
PY
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

## Next Step

Implement the planned candidate config rows, validate that the policy-search loader can resolve each
row, then open a separate execution issue or PR that runs the compact matrix and promotes durable
effect-size evidence. Keep grouped and one-factor rows separate in the interpretation.

Update 2026-06-03: [issue_2180_one_factor_h500.md](issue_2180_one_factor_h500.md) executes the
manifest at h500 with zero failed rows. The h500 result points toward recentering as the clearest
positive component and does not support static escape alone, corridor-transit terms, selector-only,
or speed/progress-2.4 as independent gains on this slice.
