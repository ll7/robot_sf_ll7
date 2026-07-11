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
