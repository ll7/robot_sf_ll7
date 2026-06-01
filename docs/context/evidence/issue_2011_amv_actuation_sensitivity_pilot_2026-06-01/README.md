# Issue #2011 AMV Actuation Sensitivity Pilot Evidence

Issue: [#2011](https://github.com/ll7/robot_sf_ll7/issues/2011)

Generated from commit `4f517eb82fc2ee7e2c1da598d80f1e636d6ad17b` plus the local Issue #2011
branch changes.

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/run_amv_actuation_sensitivity_sweep.py \
  --manifest configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml \
  --output output/issue_2011_amv_actuation_sensitivity_preflight \
  --mode preflight \
  --log-level WARNING

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/run_amv_actuation_sensitivity_sweep.py \
  --manifest configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml \
  --output output/issue_2011_amv_actuation_sensitivity_pilot_summary \
  --mode pilot \
  --log-level WARNING
```

The promoted files are compact summaries only. Raw campaign roots and episode JSONL files remain
under the ignored local output tree.

## Contents

- `resolved_sweep_manifest.json`: materialized source-status manifest with generated config
  checksums.
- `preflight_index.json`: 12 generated campaign configs passed camera-ready preflight.
- `pilot_index.json`: 12 pilot campaign roots, each with 4 executed episodes.
- `reports/effect_size_summary.{csv,json,md}`: effect-size table by field group, level, planner,
  and scenario family.
- `figures/outcome_sensitivity.svg`: compact sensitivity figure from the pilot table.
- `checksums.sha256`: SHA-256 checksums for promoted evidence files.

## Boundary

All pilot campaigns are classified `accepted_unavailable_only` because
`latency_stress_profile` is still `preflight-and-provenance-only`; runtime latency metrics are not
implemented. Treat these outputs as diagnostic plumbing evidence and not benchmark-strengthening or
paper-facing planner evidence.
