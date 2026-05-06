# Policy Search H500 Evidence (2026-05-06)

Small, tracked evidence copied from `output/` for the h500 policy-search decision trail.

## Source

- Worktree output root: `output/policy_search/`
- Failure reports:
  - `output/ai/autoresearch/h500_leader_repair/baseline_failure_report/`
  - `output/ai/autoresearch/h500_leader_repair/v2_full_h500_failure_report/`
- Synthesis:
  `docs/context/policy_search/reports/2026-05-05_full_matrix_h500_analysis.md`

## Contents

- `summaries/`: selected `summary.json` files for the h500 raw-success leader, strict-gate
  candidate, strict baseline, and near-leader.
- `failure_reports/`: compact failure-taxonomy reports comparing the baseline leader with the v2
  collision guard.
- `manifest.sha256`: checksums for the copied files.

## Storage Decision

These files are small enough to track directly and are useful for reviewing development progress
without depending on local `output/` contents. Raw episode JSONL and Slurm logs stay out of git
because the tracked configs, seeds, commits, reports, and summaries are enough for this decision
trail; rerun or archive raw JSONL externally only when exact per-episode replay is required.
