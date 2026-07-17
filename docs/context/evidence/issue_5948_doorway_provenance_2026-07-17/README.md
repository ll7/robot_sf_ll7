<!-- AI-GENERATED (robot_sf#5948, 2026-07-17) - NEEDS-REVIEW -->

# Issue #5948: Doorway Butterfly Trace Re-export Provenance

## Summary

This directory registers the durable provenance pointer for job 13483 (doorway butterfly trace re-export), whose run config was lost when the ad-hoc worktree was cleaned.

## Incident

- **Date discovered:** 2026-07-17
- **Root cause:** Campaign config `configs/benchmarks/doorway_butterfly_trace_reexport.yaml` was created ad-hoc inside worktree `doorway-a307`, used to drive job 13483, and left uncommitted. The worktree was subsequently cleaned, and the original file is gone.
- **Impact:** The only record of what actually ran was job 13483's output tree (7.2 MB: `run_summary.yaml`, `campaign_manifest.json`, `PER_SEED_DIFF.md`, per-seed reports) — gitignored, untracked, one `git clean` from permanent loss.

## Durable Preservation

The output tree was preserved as a checksummed bundle before this fix:
- **Bundle:** `job-13483-doorway-provenance.tar.gz`
- **SHA-256:** `1a434946d774a5550ec3791ec6a829768fab5ce058c7a9ec4db9d43711780dff`
- **Files:** 46 individually hashed
- **Location:** Copied off the originating disk to a second host under `durable-evidence/`

## Reconstruction

The config was reconstructed from two independent sources that agree:
1. The spec's inline YAML
2. Job 13483's own `run_summary.yaml` (config_hash `846e99aaba7dff51`)

Reconstruction parameters:
- Scenario matrix: `classic_interactions_francis2023.yaml`
- Scenario candidate: `classic_doorway_medium`
- Planner: `ppo` @ `ppo_15m_grid_socnav.yaml`
- Seed set: `paper_eval_s30` = 111–140 (30 seeds)
- Horizon: 600
- dt: 0.1
- Workers: 1
- SNQI: `camera_ready_v3` enforcement `warn`
- AMV: `amv-paper-v1` status `pass`
- Observation noise: disabled
- `paper_facing: false`

**Note:** The spec's minimal variant lists seeds `[113, 114]`; the run actually used the **recommended** 30-seed set (111–140). The artifact is authoritative about what ran.

## Rule Established

Per the comment in issue #5948, this incident establishes the rule:

> **Anything required to reproduce a shipped artifact — its config, its input, AND the code that produced it — must be durable on main (or registered with a durable pointer) before the artifact is treated as shipped.**

This is the producer-side sibling of #5936 (result-job durability gate): #5936 requires an analysis to have a durable *input*; this requires a recorded run to have a durable *config*.

## Files in This Directory

- `provenance_manifest.json` — Machine-readable provenance pointer
- `run_summary_reconstruction.yaml` — Key fields from job 13483's run_summary.yaml
- `config_sha256.txt` — SHA-256 of the committed reconstructed config file

## Related Issues

- #5936 — Result-job durability gate (no analysis successor before input is durable)
- #5893 — Promotion decision for trace-tooling prototypes (#5834/#5839/#5840)
- #5447 — Trace comparison and figure eligibility promotion
