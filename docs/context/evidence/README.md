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
- `camera_ready_all_planners_2026-05-04/`: compact camera-ready all-planners campaign summaries and
  reports from the May 4 run.
