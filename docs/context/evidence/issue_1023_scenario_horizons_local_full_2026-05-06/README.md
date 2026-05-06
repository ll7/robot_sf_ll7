# Issue 1023 Scenario-Horizon Local Full Campaign Evidence

Compact, tracked evidence copied from the local non-Slurm scenario-horizon campaign run on
2026-05-06.

## Source

`output/benchmarks/issue_1023/issue1023_scenario_horizons_h500_local_2026-05-06/`

The run used:

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_h500_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

## Outcome

- 7 planners completed.
- 1008 episodes completed.
- Campaign runtime: 819.7857 seconds.
- Analyzer finding: no automated campaign inconsistencies.
- SNQI contract status: `warn`.
- Reproducibility comparison against the May 4 fixed-horizon evidence: `drift_detected`.
- Coverage gap against the May 4 fixed-horizon evidence: `socnav_bench` is present only in the
  fixed-horizon reference.

Scenario-specific horizons increased success for the seven matched planners, but collisions and
near-misses also increased across all matched planner-level rows. Treat this as horizon
sensitivity/confounding evidence, not as a replacement headline benchmark surface.

## Contents

- `campaign/`: campaign manifest, minimal manifest, and run metadata.
- `preflight/`: validate/preview artifacts for the exact run config.
- `reports/`: compact campaign, analyzer, SNQI, comparability, matrix, scenario, family, and
  fixed-vs-scenario comparison reports.
- `runs/*/summary.json`: per-planner run summaries.
- `manifest.sha256`: checksums for the copied files.

The fixed-vs-scenario comparison includes planner-level deltas plus complete machine-readable
scenario and scenario-family deltas. It reports `unfinished_mean` as `1 - success_mean`, which is a
route-incomplete comparison metric rather than raw timeout attribution. The Markdown report shows
the largest scenario/family deltas; use the JSON artifact for complete per-row review.

## Storage Decision

This bundle keeps the small, reviewable evidence needed for issue #1023 and PR #1033 in git. Raw
episode JSONL, local logs, seed-episode rows, seed-variability tables, coverage outputs, and
worktree-local model-cache symlinks remain ignored under `output/`. They are reproducible from the
tracked config, seed schedule, commit, and command, or should be archived externally only for a
release package.
