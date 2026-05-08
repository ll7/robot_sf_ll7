# Issue 1044 H500 Follow-Up Benchmark Plan

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1044>  
Depends on mechanism analysis: <https://github.com/ll7/robot_sf_ll7/issues/1045>

## Decision

Keep the current fixed-horizon camera-ready benchmark as the release surface. Treat h500
scenario horizons as a separate long-horizon sensitivity and follow-up-paper surface with its own
claim boundary, evidence retention policy, and reporting tables.

The h500 question is not "which planner performs best under the current strict time budget?" It is:

> Given enough time to complete the intended scenario interaction, which planners reach the goal,
> and what safety, comfort, exposure, and interaction costs appear while they do it?

That question is useful, but it should not silently replace the fixed-horizon result because longer
runs reduce timeout pressure while increasing exposure to near misses and collisions.

## Evidence Baseline

Use these existing evidence surfaces as the starting point:

- `docs/context/issue_1023_scenario_horizon_benchmark.md` - h500 benchmark config, preflight,
  local full campaign, and fixed-vs-h500 comparison boundary.
- `docs/context/issue_1038_h500_snqi_contract.md` - SNQI analysis and decision not to overwrite
  camera-ready v3 SNQI assets.
- `docs/context/issue_1045_h500_solvability_mechanisms.md` - aggregate timeout-to-success
  mechanism split and trace-required waiting boundary.

Current aggregate signal from issue #1045:

| Mechanism | Cases | Paper-facing meaning |
|---|---:|---|
| `budget_limited_clean_completion` | 38 | Strongest support that some fixed-horizon failures were time-budget artifacts. |
| `late_clean_completion` | 5 | Supports long-horizon realism, but completion uses much of h500. |
| `exposure_enabled_completion` | 40 | Success improves, but near-miss exposure rises; requires exposure-aware safety framing. |
| `partial_timeout_relief` | 18 | H500 helps but does not fully solve the scenario/planner cell. |
| `safety_regressed_completion` | 22 | H500 success gains are paired with collision regressions and should be treated as caveats. |

## Claim Boundary

Acceptable h500 claims:

- h500 separates time-budget artifacts from planner inability on a subset of scenarios.
- h500 reveals additional safety and social-interaction costs because successful runs spend more
  time inside the scenario.
- h500 is suitable as a sensitivity, appendix, benchmark report, or follow-up paper surface once
  raw episodes and trace-backed examples are retained.

Claims to avoid:

- h500 is a drop-in replacement for the current fixed-horizon paper benchmark.
- h500 improves planner safety because success increases.
- h500 near-miss increases are only artifacts of longer runtime.
- h500 successes mostly come from waiting until pedestrians pass, unless step traces or videos show
  that mechanism on representative cases.

## Reporting Plan

Report h500 with multiple tables instead of a single aggregate ranking:

| Table | Contents | Purpose |
|---|---|---|
| Fixed vs h500 outcome table | success, unfinished/timeout, collision, near miss, time-to-goal norm, SNQI deltas | Separates completion gains from safety regressions. |
| Exposure-aware safety table | raw near misses, near misses per episode, near misses per successful episode, and per-step/per-second rates when raw traces exist | Prevents longer runtime from being interpreted as either pure artifact or pure regression. |
| Mechanism table | issue #1045 classes by planner and scenario family | Explains why the horizon changed outcomes. |
| Representative trace table | 3-5 selected planner-scenario-seed traces/videos across clean budget relief, exposure-enabled completion, and safety-regressed completion | Supports causal claims such as waiting, recovery, delayed progress, or risk-taking. |
| SNQI sensitivity table | current v3 SNQI plus separately versioned h500 candidate normalization if calibrated | Keeps h500 SNQI separate from camera-ready v3 assets. |

Do not publish a single h500 "winner" table unless the table also carries the safety/exposure
caveats needed to interpret it.

## Fresh Evidence Requirements

Any publishable h500 follow-up run should retain:

- raw episode JSONL for every planner-scenario-seed cell,
- compact campaign reports copied to `docs/context/evidence/`,
- raw artifact pointer or manifest if the full campaign is too large for git,
- exact config paths and commit hash,
- fixed-horizon reference campaign id,
- h500 campaign id,
- seed schedule,
- planner mode (`native`, `adapter`, `fallback`, or `degraded`),
- videos or step diagnostics for the representative mechanism slice.

Raw episodes are required for per-step or per-second near-miss rates. Without them, report only the
aggregate fixed-vs-h500 deltas and label waiting/yielding mechanisms as hypotheses.

## Pilot Slice

Before a full follow-up campaign, run a small trace-backed slice:

| Mechanism target | Example cells from #1045 | Evidence to collect |
|---|---|---|
| Clean budget relief | ORCA on `classic_bottleneck_low`, `classic_head_on_corridor_low`, `classic_urban_crossing_medium` | fixed and h500 episode traces; confirm whether completion is route-length budget, delayed progress, or waiting. |
| Exposure-enabled completion | ORCA/PPO on `francis2023_parallel_traffic` and PPO on `francis2023_crowd_navigation` | near-miss timing, pedestrian proximity, and whether the planner waits, threads through traffic, or replans. |
| Safety-regressed completion | prediction planner on `francis2023_accompanying_peer`, `francis2023_down_path`, `francis2023_following_human` | collision timing and whether longer horizon merely extends exposure to unsafe behavior. |

Use `scripts/validation/run_policy_search_step_diagnostics.py` when a policy-search candidate can
represent the planner, or generate videos from retained episode JSONL for all-planner campaign
cells. The pilot is complete only when the trace/video evidence can distinguish waiting from
continuous delayed progress and risk-taking.

Issue #1049 now provides the first compact trace-backed pilot:

* `docs/context/issue_1049_h500_mechanism_pilot.md`
* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/README.md`

The pilot supports clean budget relief, exposure/comfort-pressure increase, and safety-regressed
long-horizon exposure examples for ORCA. It does not support a broad wait-then-go claim.

Issue #1055 adds the first exposure-aware representative tables:

* `docs/context/issue_1055_exposure_aware_h500_tables.md`
* `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/report.md`

Issue #1058 adds reusable paper/report wording:

* [Issue #1058 H500 Paper Language](issue_1058_h500_paper_language.md)

## SNQI Contract

Do not overwrite camera-ready v3 SNQI assets. If h500 SNQI is reported, use one of these safer
options:

- report current v3 SNQI as a sensitivity value with an explicit "not calibrated for h500" label,
- calibrate a separately versioned h500 SNQI contract from h500 evidence,
- or avoid a single h500 SNQI headline and instead report the underlying success, collision,
  near-miss, comfort, and exposure terms.

The preferred first paper-facing version is multi-table reporting with optional h500 SNQI
sensitivity, not a new headline aggregate metric.

## Validation Commands

Existing reusable commands:

```bash
python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root <fixed_campaign_root> \
  --candidate-campaign-root <h500_campaign_root> \
  --output-json <comparison_output_dir>/fixed_vs_h500_comparison.json \
  --output-md <comparison_output_dir>/fixed_vs_h500_comparison.md

python scripts/tools/analyze_snqi_contract.py \
  --campaign-root <h500_campaign_root> \
  --output-json <h500_snqi_output_dir>/snqi_diagnostics.json \
  --output-md <h500_snqi_output_dir>/snqi_diagnostics.md \
  --output-csv <h500_snqi_output_dir>/snqi_planner_ordering.csv

python scripts/tools/analyze_h500_solvability_mechanisms.py \
  docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/reports/fixed_vs_scenario_horizon_candidates_comparison.json \
  --output-dir docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07
```

Run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` for any code or report-generation changes
before opening a paper-facing PR.

## Recommendation

Use h500 for a follow-up paper, appendix sensitivity, or benchmark report. The strongest narrative is
not "longer horizons are better"; it is that fixed and h500 horizons answer different scientific
questions. Fixed horizons test strict-time-budget navigation. H500 tests completion under realistic
time allowance and exposes the safety costs that appear when planners are given enough time to keep
interacting with dynamic obstacles.
