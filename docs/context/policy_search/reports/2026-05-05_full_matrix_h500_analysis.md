# Full-Matrix H500 Policy Search Analysis (2026-05-05)

## Scope

This note summarizes the tracked `full_matrix_h500` policy-search reports. The initial 23-candidate
matrix was produced from commit `47fecd938482949b7989f1011ec6e34237d8b45d`; the two leader rerun
reports were refreshed by the clean pinned Slurm rerun on commit
`2b796ea92104467d3bc913528801fb8bb11034dd`. A focused collision-guard follow-up added
`scenario_adaptive_hybrid_orca_v2_collision_guard` from commit
`d22c5e6c91b8f690a4ba0a1c6e32bf7aa3cf1b21`.

- Stage: `full_matrix_h500`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Candidate reports: 24 tracked reports under `docs/context/policy_search/reports/`
- Episode count: 144 per candidate
- Local source artifacts: `output/policy_search/*/full_matrix_h500/*/summary.json` and combined
  JSONL files referenced by each report

These are policy-search triage artifacts, not paper-facing benchmark claims. The output artifacts
under `output/` are worktree-local; the tracked Markdown reports and this synthesis are the durable
record for handoff.

## Clean Leader Rerun

Slurm job `12339` reran `scenario_adaptive_hybrid_orca_v1` and
`hybrid_rule_v3_fast_progress` with `--clean-pinned` under run id
`policy_search_full_matrix_h500_leaders_clean_20260505_204501`.

- Both array tasks completed successfully (`0:0`) on `auxme-imech172`.
- The clean rerun reproduced the earlier h500 metrics exactly for both candidates.
- The only tracked per-candidate report deltas are provenance updates to the clean summary paths and
  commit hash.
- Log review found repeated global-planner invalid-goal-cell messages, but no Slurm/job failure,
  traceback, integrity contradiction, or benchmark fallback/degraded execution mode. Planner
  diagnostics show adapter execution throughout; internal `EMERGENCY_STOP` command fallbacks remain
  episode-level behavior, not fallback planner substitution.

## Collision-Guard Rerun

Slurm job `12348` ran `scenario_adaptive_hybrid_orca_v2_collision_guard` with `--clean-pinned`
under run id `policy_search_full_matrix_h500_collision_guard_20260506_0800`.

- The job completed successfully (`0:0`) on `auxme-imech172` in `00:22:55`.
- The candidate deliberately trades one `classic_merging_low` success for one fewer collision:
  `0.9028` success and `0.0139` collision.
- The two remaining h500 collisions are first-step dynamic-deadlock episodes
  (`classic_cross_trap_high` seed `112`, `francis2023_circular_crossing` seed `111`) that tuned
  ORCA did not repair on the targeted micro-slice.
- The strict `nominal_sanity` promotion gate marks this candidate `promote`.

## Aggregate Ranking

| Rank | Candidate | Success | Collision | Near Miss | Classic Success | Francis Success | Main Failures |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `scenario_adaptive_hybrid_orca_v1` | 0.9097 | 0.0208 | 0.4236 | 0.8696 | 0.9467 | `timeout_low_progress` 6, `static_collision` 3 |
| 2 | `scenario_adaptive_hybrid_orca_v2_collision_guard` | 0.9028 | 0.0139 | 0.4236 | 0.8551 | 0.9467 | `timeout_low_progress` 8, `static_collision` 2 |
| 3 | `hybrid_rule_v3_fast_progress_static_escape` | 0.9028 | 0.0208 | 0.4236 | 0.8696 | 0.9333 | `timeout_low_progress` 6, `static_collision` 3 |
| 4 | `hybrid_rule_v3_fast_progress` | 0.8264 | 0.0139 | 0.4236 | 0.7391 | 0.9067 | `timeout_low_progress` 11, `static_collision` 2 |
| 5 | `hybrid_rule_v3_progress_2p4` | 0.8056 | 0.0139 | 0.4097 | 0.6957 | 0.9067 | `timeout_low_progress` 14, `static_collision` 2 |
| 6 | `hybrid_rule_v4_recovery_aware` | 0.8056 | 0.0139 | 0.4097 | 0.6957 | 0.9067 | `timeout_low_progress` 14, `static_collision` 2 |
| 7 | `hybrid_rule_v3_dynamic_relaxed` | 0.7778 | 0.0139 | 0.4167 | 0.6667 | 0.8800 | `timeout_low_progress` 15, `static_collision` 2 |
| 8 | `hybrid_rule_v3_waypoint2_route_lookahead8_static02` | 0.7778 | 0.0139 | 0.4097 | 0.6522 | 0.8933 | `timeout_low_progress` 15, `static_collision` 2 |
| 9 | `hybrid_rule_v3_waypoint2_route_lookahead8_static05` | 0.7778 | 0.0139 | 0.4097 | 0.6522 | 0.8933 | `timeout_low_progress` 15, `static_collision` 2 |
| 10 | `hybrid_rule_v3_teb_like_rollout` | 0.7708 | 0.0139 | 0.4097 | 0.6667 | 0.8667 | `timeout_low_progress` 15, `static_collision` 2 |

The remaining route-lookahead and waypoint variants mostly trade extra success for static
collisions. `mpc_clearance_sampler_v1` remains attractive only as a research component because it
has the lowest near-miss rate in this h500 set (`0.1597`) but a very high collision rate (`0.3750`).

## Horizon Effect

The h500 rerun mostly reveals that the short `full_matrix` horizon was truncating route completion,
not that a different planner family suddenly became dominant.

| Candidate | Full Matrix Success | H500 Success | Success Delta | Collision Delta |
|---|---:|---:|---:|---:|
| `scenario_adaptive_hybrid_orca_v1` | 0.2778 | 0.9097 | +0.6319 | +0.0069 |
| `hybrid_rule_v3_fast_progress_static_escape` | 0.2639 | 0.9028 | +0.6389 | +0.0069 |
| `hybrid_rule_v3_fast_progress` | 0.2569 | 0.8264 | +0.5694 | +0.0000 |
| `hybrid_rule_v3_progress_2p4` | 0.2708 | 0.8056 | +0.5347 | +0.0000 |
| `hybrid_rule_v4_recovery_aware` | 0.2500 | 0.8056 | +0.5556 | +0.0000 |

The raw-success leader and the `hybrid_rule_v3_fast_progress_static_escape` near-leader gain the
most success but add one extra collision relative to the shorter full-matrix run. This matters
because the strict `nominal_sanity` gate allows at most `0.0200` collision rate, while both finish
at `0.0208` (3 collisions in 144 episodes). The v2 collision-guard follow-up is the h500 candidate
that trades one success for a strict-gate-clean `0.0139` collision rate.

## Promotion Decision

Current h500 leader:

- Promote `scenario_adaptive_hybrid_orca_v1` as the current experimental h500 leader. It has the
  best aggregate success/collision tradeoff (`0.9097` success, `0.0208` collision), with especially
  strong Francis-family success (`0.9467`).

Strict gate-clean h500 candidates:

- Promote `scenario_adaptive_hybrid_orca_v2_collision_guard` as the primary strict-gate h500
  candidate. It preserves most of the v1 leader's success while passing success `>= 0.80`,
  collision `<= 0.02`, and the scenario-stratified collision guard (`0.9028` success, `0.0139`
  collision).
- Keep `hybrid_rule_v3_fast_progress` as the simplest strict-gate h500 baseline. It passes the same
  strict checks (`0.8264` success, `0.0139` collision), but gives up far more route completion.
- Keep `hybrid_rule_v3_progress_2p4` and `hybrid_rule_v4_recovery_aware` as backup ablations. They
  both pass the same strict gate at `0.8056` success and `0.0139` collision, but neither beats
  `hybrid_rule_v3_fast_progress` on aggregate success.

Track but do not promote as primary:

- `hybrid_rule_v3_fast_progress_static_escape`: nearly tied with the leader (`0.9028` success) but
  has the same one-collision-over-strict-gate issue and no aggregate advantage over the
  scenario-adaptive candidate.
- `hybrid_rule_v3_dynamic_relaxed`, route-lookahead static-margin variants, and
  `hybrid_rule_v3_teb_like_rollout`: useful ablations, but all fall below the `0.80` strict success
  floor.
- `mpc_clearance_sampler_v1`: do not promote; use only as a future clearance-cost/proposer idea
  behind a hard static safety filter.

Before any camera-ready or paper-facing promotion, rerun the recommended candidates from a clean,
pinned worktree and include horizon-specific evidence in the benchmark claim.

## Remaining Scenario Blockers

H500 does not solve every case. The aggregate leader still fails:

- `classic_merging_medium`: 0/3 success, 3 timeouts.
- `classic_station_platform_medium`: 0/3 success, 3 timeouts.
- `francis2023_narrow_doorway`: 0/3 success, 3 timeouts.
- `classic_merging_low`: 1/3 success for v1, with one collision and one timeout; the v2 collision
  guard changes this to 0/3 success with no collision.
- `classic_cross_trap_high`: 2/3 success, with one collision.
- `francis2023_circular_crossing`: 2/3 success, with one collision.

The no-success-at-h500 cases should be treated as planner/design blockers rather than horizon
selection problems.

## Research Directions

1. Horizon-aware benchmark contracts. H500 shows route-completion budget is a first-class variable:
   aggregate success rises by roughly 0.5-0.64 for the top rule-based family. Future reports should
   separate planner quality from time-budget sufficiency.
2. Deadlock and route-local-minimum recovery. The remaining hard cases are merging, station
   platform, and narrow doorway scenes where more horizon does not create progress. Work should move
   from local DWA scoring toward deliberate retreat/recenter/re-route actions with hysteresis.
3. Comfort-preserving high-success planning. The best h500 candidates succeed but retain high
   near-miss rates around `0.41-0.42`. Add speed-dependent near-miss/comfort terms or scenario
   affordance gates before treating high success as socially acceptable.
4. Selector with safety accounting. Scenario-adaptive selection is the aggregate leader, but its
   advantage over `hybrid_rule_v3_fast_progress_static_escape` is only one episode. Any next
   selector should require per-scenario evidence that the override improves success without adding
   static collisions.
5. MPC as a proposer, not an executor. `mpc_clearance_sampler_v1` has low near-miss exposure but
   unacceptable static-collision behavior. The promising direction is to reuse its clearance score
   under the hybrid rule planner's hard safety filter.

## Scenario Horizon Policy

Use scenario-specific horizons rather than one fixed budget for every scenario.

Recommended decision rule:

1. For each scenario, collect successful completion step counts from the safe incumbent set
   (`collision <= 0.03`, success `>= 0.75`) across pinned h500 runs.
2. Set the candidate horizon to `ceil(p95_success_steps * 1.2 + 20)` with a small floor for very
   short scenes, capped at h600. Use the route-length/nominal-speed estimate as a sanity check when
   `shortest_path_len` or equivalent path metrics are present.
3. If `p95_success_steps` is near the h600 cap or the scenario still has many h500 timeouts, treat
   the scenario as needing a separate design/probe decision rather than silently increasing the
   benchmark budget.
4. If no safe incumbent succeeds by h500, mark the scenario as `planner_blocked` under the h600 cap.
   Do not keep raising the horizon and calling the resulting failure a fair route-completion test.

Observed buckets from the h500 evidence:

| Horizon Bucket | Candidate Scenarios | Recommended Handling |
|---|---|---|
| 100-180 steps | elevator/room entry-exit, group crossing, intersections, `francis2023_leave_group` | Short-to-medium fixed budgets are enough; keep the 20% slack unless route length changes. |
| 200-350 steps | bottlenecks, corridors, T-intersections, most Francis following/traffic scenes | Use h200-h350, selected by per-scenario p95 with slack. |
| 359-498 steps | cross-trap high, overtaking, `classic_merging_low`, doorway-low, robot-crowding | Use h360-h500; treat high failure counts as planner risk even when a few successes set the p95. |
| 514-562 steps | cross-trap low/medium and `classic_realworld_double_bottleneck_high` | Use extended h515-h565 budgets under the h600 cap. |
| Planner-blocked at h600 | `classic_merging_medium`, `classic_station_platform_medium`, `francis2023_narrow_doorway` | Do not solve with horizon; require planner or scenario-contract work. |

## Validation

- Refreshed portfolio overview:
  `uv run python scripts/tools/summarize_policy_search_portfolio.py --output-md docs/context/policy_search/portfolio_overview_2026-05-05.md --output-json docs/context/policy_search/portfolio_overview_2026-05-05.json`
- Parsed the 23 h500 report summary paths and combined JSONL files referenced by those reports to
  compute aggregate ranking, h100-to-h500 deltas, scenario blockers, gate checks, and horizon
  buckets.
- Reviewed clean leader rerun job `12339` with `sacct`; both tasks completed with exit code `0:0`
  in about 24 minutes, and the rerun summaries matched the earlier h500 metrics exactly.
- Refreshed strict h500 gate reports and scenario horizon source paths so the two rerun leaders now
  reference `policy_search_full_matrix_h500_leaders_clean_20260505_204501`.
- Ran h500 collision repair micro-sweeps (`12341`, `12345`, `12347`) and full h500 validation
  (`12348`) for `scenario_adaptive_hybrid_orca_v2_collision_guard`; the full run completed with
  exit code `0:0` and strict gate decision `promote`.
- Follow-up implementation artifacts:
  - Clean pinned rerun handoff:
    `docs/context/policy_search/SLURM/004_h500_leader_clean_rerun.md`
  - Strict h500 gate reports:
    `docs/context/policy_search/reports/promotions/2026-05-05_full_matrix_h500_strict_gate/`
  - Scenario horizon recommendations:
    `docs/context/policy_search/reports/2026-05-05_h500_horizon_recommendations.md`
    and `configs/policy_search/scenario_horizons_h500.yaml`
  - Research plan:
    `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
- Tooling checks for the follow-up implementation:
  - `uv run pytest tests/tools/test_promote_policy_search_candidate.py tests/tools/test_suggest_policy_search_horizons.py`
  - `uv run ruff check scripts/tools/suggest_policy_search_horizons.py scripts/tools/promote_policy_search_candidate.py scripts/tools/summarize_policy_search_portfolio.py tests/tools/test_promote_policy_search_candidate.py tests/tools/test_suggest_policy_search_horizons.py`
  - `uv run ruff format --check scripts/tools/suggest_policy_search_horizons.py scripts/tools/promote_policy_search_candidate.py scripts/tools/summarize_policy_search_portfolio.py tests/tools/test_promote_policy_search_candidate.py tests/tools/test_suggest_policy_search_horizons.py`
  - `bash -n scripts/dev/sbatch_policy_search_sweep.sh SLURM/Auxme/policy_search_candidate_stage.sl`
  - `scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix_h500 --candidates-file configs/policy_search/candidate_sets/h500_leader_clean_rerun.txt --run-id policy_search_full_matrix_h500_leaders_clean_dryrun --throttle 1 --workers 2 --clean-pinned --no-status --dry-run`
  - `git diff --check`
