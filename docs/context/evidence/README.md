# Evidence Bundles

This directory contains small, reviewable copies of generated artifacts that are worth preserving in
git because they support durable context notes, benchmark decisions, or PR handoff.

Do not mirror `output/` here wholesale. Generated files under `output/` remain worktree-local and
may be deleted when they are reproducible from a tracked config, seed schedule, commit, and command.

## What Belongs Here

- compact `summary.json` files that anchor a promoted benchmark or policy-search decision,
- Markdown/JSON analysis reports used by tracked context notes,
- small CSV/JSON tables needed to review a campaign without rerunning it,
- manifests and checksums for the copied evidence.

## What Stays Out

- raw episode JSONL files unless a tiny curated subset is required for a regression fixture,
- large Slurm stderr/stdout logs,
- model checkpoints and model caches,
- coverage HTML,
- temporary repos, caches, and scratch outputs.

For larger artifacts that are not cheap to regenerate but are not reviewable in git, use an external
artifact store and track only a manifest or registry pointer here.

Git LFS is not the default storage type for generated benchmark artifacts. Use it only for a
deliberately versioned, non-regenerable binary fixture after an explicit maintainer decision. If an
artifact can be regenerated from tracked configs, seed schedules, commands, and commits, it is fine
to leave it ignored or delete it locally once the durable summary/report evidence is preserved.

## Current Bundles

- `policy_search_h500_2026-05-06/`: h500 policy-search leader summaries and failure reports that
  support the v1 raw-success leader and v2 strict-gate promotion decision.
- `issue_1023_scenario_horizons_preflight_2026-05-06/`: compact preflight artifacts for the
  paper-facing scenario-horizon benchmark config.
- `issue_1023_scenario_horizons_local_full_2026-05-06/`: compact local non-Slurm full-campaign
  artifacts, analyzer output, and fixed-vs-scenario comparison for issue #1023.
- `issue_1045_h500_solvability_mechanisms_2026-05-07/`: aggregate mechanism classification for
  h500 fixed-timeout relief, including explicit trace-required boundaries for wait-then-go claims.
- `issue_1111_carla_setup_smoke_2026-05-18/`: compact setup-only CARLA T1 oracle smoke evidence
  proving the optional Python API and T0 payload-selection boundary without live replay claims.
- `issue_1239_human_model_transfer_2026-05-18/`: compact human-model transfer smoke evidence with
  explicit variant/source rows and fail-closed upstream adapter availability.
- `issue_1169_carla_live_replay_2026-05-18/`: compact Docker-backed CARLA live replay summaries
  proving client/server connectivity and the fail-closed static-geometry boundary.
- `issue_1344_paired_amv_primary_2026-05-20/`: compact paired nominal/stress AMV primary-row
  campaign summaries and tables.
- `issue_1454_s10_preflight_2026-05-22/`: compact preflight evidence for the staged S10 fixed-h100
  and scenario-horizon h500 robustness configs.
- `issue_1454_stage_a_fixed_h100_2026-05-22/`: compact Stage A full-campaign, analyzer, and
  May 4 comparison evidence for the issue #1454 S10 fixed-h100 broader-baseline gate.
- `issue_1454_s10_h500_candidates_2026-05-23/`: compact exploratory S10 scenario-horizon h500
  evidence for the seven functioning Stage A rows plus local policy-search candidate rows, with a
  pointer to the non-package GitHub artifact release for the raw campaign archive.
- `issue_1462_s10_h500_failure_modes_2026-05-24/`: compact scenario, candidate-vs-core, and seed
  failure-mode tables derived from the issue #1454 S10/h500 candidate evidence.
- `issue_1318_teb_corridor_deadlock_2026-05-20/`: compact TEB/ORCA/hybrid-rule
  classic-merging corridor-deadlock comparison summary for issue #1318.
- `camera_ready_all_planners_2026-05-04/`: compact camera-ready all-planners campaign summaries and
  reports from the May 4 run.
