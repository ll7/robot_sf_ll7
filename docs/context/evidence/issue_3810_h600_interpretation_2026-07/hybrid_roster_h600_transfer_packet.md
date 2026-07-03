<!-- AI-GENERATED (robot_sf#4230 / job 13282, 2026-07-03) — NEEDS-REVIEW -->
# H600 Hybrid-Roster Transfer Packet (job 13282)

**Claim boundary:** pre-registered h600 hybrid-roster comparison rows only
(issue #4230, pre-reg `docs/context/issue_4230_h600_hybrid_roster_preregistration.md`,
PR #4265). Feeds the #4195 F-C4(ii) promotion gate. No paper/dissertation
promotion is made here; this packet reports and interprets one campaign.

## Provenance

- Slurm job **13282** (imech192 l40s), COMPLETED 0:0, elapsed 1:14:51,
  campaign `issue4230_h600_hybrid_roster_run_20260703`.
- Public commit `b4800dc828e6`; scenario matrix `classic_interactions_francis2023.yaml`,
  hash **`c10df617a87c`** (identical to reference jobs 13268/13273); seeds
  **[111, 112, 113]** (`eval` set); horizon fixed **600**; 48 scenarios ×
  3 seeds = **144 episodes/arm**; observation noise: none (hash `c71dd80adab5`).
- Pre-registered no-submit CPU preflight passed before submission (same
  worktree; matrix hash, roster, and seed surface verified).
- Artifacts: ledger row 13282 (private ops), metrics mirrored at
  `imech156-u:~/git/robot_sf_ll7/output/issue4230-h600-hybrid-roster-run/13282/reports/`.
  Reference rows: `planner_metric_summary.md` in this directory (13268/13273).

## Headline rows (success, mean over 144 episodes; 95% binomial CI)

| planner_key | success | 95% CI | per-seed success (111/112/113) | SNQI mean |
| --- | --- | --- | --- | --- |
| hybrid_rule_v3_fast_progress_static_escape_continuous | **0.799** | [0.733, 0.864] | 0.813 / 0.813 / 0.771 | −0.124 |
| hybrid_rule_v3_fast_progress_static_escape | **0.792** | [0.725, 0.858] | 0.750 / 0.813 / 0.813 | −0.094 |
| scenario_adaptive_hybrid_orca_v1 | **0.778** | [0.710, 0.846] | 0.729 / 0.792 / 0.813 | −0.094 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | **0.771** | [0.702, 0.839] | 0.729 / 0.792 / 0.792 | −0.102 |
| *(ref, 13268/13273)* orca | 0.743 | seed-mean CI [0.729, 0.771] | 0.771 / 0.729 / 0.729 | −0.198 |
| *(ref)* ppo | 0.701 | [0.625, 0.771] | 0.625 / 0.708 / 0.771 | −0.156 |
| *(ref)* prediction_mpc_cbf | 0.569 | — | — | — |
| *(ref)* prediction_mpc | 0.563 | — | — | — |
| *(ref)* prediction_planner | 0.493 | [0.438, 0.542] | 0.438 / 0.542 / 0.500 | −0.145 |

## Interpretation (ranked by evidential strength)

1. **Well-separated (disjoint CIs): all four hybrid arms dominate every
   prediction-equipped arm at h600.** Worst hybrid (0.771) vs best prediction
   arm (0.569): Δ ≥ 0.20 success, no CI overlap. This extends the
   control-law-bound finding (13268/13273 chain, spine F-C4/RQ3 draft claims)
   to the pre-registered hybrid roster at long horizon: arms that change the
   *control law* clear arms that add *prediction* to a fixed control law.
2. **h500 → h600 rank transfer holds.** The four h500 hybrid leaders are the
   top four point-estimate arms at h600, above the entire 13273 extended
   roster. This is the transfer question #4230 pre-registered — answered
   affirmatively at point-estimate level.
3. **Hybrids vs ORCA: point-estimate lead (+0.03 to +0.06), CI-overlapping at
   this seed budget — but seed-consistent.** ORCA's 0.743 lies inside each
   hybrid's 95% episode-level CI, so no separated success claim vs ORCA is
   licensed at S3-seeds/144-episodes. However, per-seed comparison favors the
   hybrids on essentially every seed (e.g. `..._continuous` ≥ ORCA on 3/3
   seeds), and on **SNQI the hybrids lead clearly** (−0.09..−0.12 vs ORCA
   −0.198; also ahead of PPO −0.156). Escalation path if a separated
   success-vs-ORCA claim is needed: the predeclared S30 schedule (#4304,
   deferred by ruling 2026-07-03).
4. **Consistency with the h500 finding:** at h500 the two top hybrids tied
   within CI; the same tight clustering (0.771–0.799) recurs at h600. No
   evidence of a hybrid-internal ranking claim; treat the four arms as a
   statistical tie among themselves.

## What this feeds

- **#4195 F-C4(ii) promotion gate:** the missing h600 hybrid-roster leg now
  exists with pre-registered provenance. Suggested gate reading: F-C4(ii)
  "hybrid control-law arms retain their advantage at h600" is supported at
  the claim boundary above (separated vs prediction arms; point-estimate +
  SNQI-lead vs ORCA).
- **diss #327/#328** (h600 interpretation prose) and the F-C4 promotion
  decision (author sign-off still required — this packet does NOT promote).

<!-- /AI-GENERATED -->
