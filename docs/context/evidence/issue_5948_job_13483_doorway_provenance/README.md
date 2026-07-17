# Issue #5948: Job 13483 doorway provenance registration

Plain-language summary: job 13483's output tree was the sole surviving provenance of a shipped
figure, stored untracked in an ignored `output/` directory on a single disk. The config that drove
the run was created ad-hoc in a throwaway worktree and lost. This registration preserves the
provenance through the external-data-plane convention and commits the reconstructed config.

## Background

The doorway butterfly trace re-export run (job 13483) was executed with an ad-hoc config created
inside worktree `doorway-a307`. That worktree was cleaned, taking the config with it. The only
record of what ran was the 7.2 MB output tree — gitignored, untracked, one `git clean` from
permanent loss.

## Immediate action taken (pre-PR)

The output tree was preserved as a checksummed bundle before this fix:
- `job-13483-doorway-provenance.tar.gz`, sha256 `1a434946d774a5550ec3791ec6a829768fab5ce058c7a9ec4db9d43711780dff`
- 46 files individually hashed
- Copied off the originating disk to a second host under `durable-evidence/`

## Registration

- Job: `13483` (`doorway-butterfly-trace-reexport`)
- Execution commit: `a307ef276`
- Config hash: `846e99aaba7dff51`
- Private provenance URI: `private-artifact://job-13483/doorway-butterfly-trace-reexport/output`
- Bundle SHA-256: `1a434946d774a5550ec3791ec6a829768fab5ce058c7a9ec4db9d43711780dff`
- Size: 7,549,747 bytes (7.2 MB)
- File count: 46
- Reconstructed config: `configs/benchmarks/doorway_butterfly_trace_reexport.yaml` (committed in this PR)

## Run parameters (reconstructed from run_summary.yaml)

- Scenario matrix: `classic_interactions_francis2023.yaml`
- Planner: `ppo` @ `ppo_15m_grid_socnav.yaml`
- Seed set: `paper_eval_s30` = 111–140 (30 seeds)
- Horizon: 600
- dt: 0.1
- Workers: 1
- SNQI: `camera_ready_v3` enforcement `warn`
- AMV: `amv-paper-v1` status `pass`
- Observation noise: disabled
- Paper-facing: false

## Discrepancy note

The spec's minimal variant listed seeds `[113, 114]`; the run actually used the **recommended**
30-seed set (111–140). The artifact is authoritative about what ran.

## Files

- `registration.json`: external-data-plane provenance, checksums, and sufficiency disposition.
