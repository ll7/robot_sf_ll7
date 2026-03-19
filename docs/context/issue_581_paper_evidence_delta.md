# Issue 581 Paper Evidence Delta Report

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#581` paper evidence delta report + checklist signoff
- `robot_sf_ll7#580` camera-ready collision summary fix
- `robot_sf_ll7#579` SNQI v3 recalibration + contract update
- `robot_sf_ll7#624` planner quality audit workflow

## Purpose

Consolidate the corrected benchmark evidence that should be carried into the AMV paper workflow.
This note replaces the earlier implicit handoff state with one explicit delta report covering:

1. the broken initial camera-ready rerun,
2. the collision-summary-corrected rerun,
3. the final SNQI-v3 canonical publication bundle.

## Canonical artifact set

Use these artifacts as the paper-facing source of truth.

### Final canonical camera-ready campaign

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407`
- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/campaign_report.md`
- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/campaign_analysis.md`
- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/planner_quality_audit.md`

### Final publication bundle

- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle`

### Intermediate corrected collision-summary campaign

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue580_collision_summary_fix_v2_20260318_121431`

### Superseded broken first rerun

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue580_full_regen_all_extras_20260318_115242`

## Evidence delta summary

| Evidence stage | Campaign ID | Key status | Why it matters |
|---|---|---|---|
| superseded first rerun | `paper_experiment_matrix_v1_issue580_full_regen_all_extras_20260318_115242` | broken collision accounting, SNQI contract `warn` | this run should not be used for paper metrics because collision rates were collapsed to `0.0000` in the summary layer |
| corrected collision-summary rerun | `paper_experiment_matrix_v1_issue580_collision_summary_fix_v2_20260318_121431` | collision accounting fixed, analyzer clean, SNQI contract `pass` | this established the numerically correct planner outcome surface |
| final SNQI-v3 rerun | `paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407` | analyzer clean, SNQI v3 contract `pass`, publication bundle produced | this is the canonical paper-facing artifact set |

## Campaign-level delta

| Metric | Broken first rerun | Collision-summary fix v2 | Final SNQI v3 |
|---|---:|---:|---:|
| SNQI contract | `warn` | `pass` | `pass` |
| Spearman rank alignment | `0.4643` | `0.5357` | `0.5357` |
| Outcome separation | `0.2650` | `0.2565` | `0.1947` |
| Campaign warnings | soft SNQI warning | none | none |
| Collision summary quality | invalid | corrected | corrected |
| Paper publication bundle | no | no | yes |

Interpretation:

- The first rerun is invalid for paper use because planner collision summaries were wrong.
- The collision-summary fix repaired the benchmark surface and removed the campaign-level warning.
- SNQI v3 did not change the underlying planner outcome counts; it changed the ranking contract to a version that passes on the corrected evidence surface and is therefore fit for paper-facing reporting.

## Planner-level final canonical snapshot

These are the planner summary values from the final canonical campaign
`paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407`.

| Planner | Success | Collisions | SNQI | Interpretation |
|---|---:|---:|---:|---|
| `ppo` | `0.2695` | `0.1844` | `-0.3541` | strongest local learned baseline, but still weak in absolute success |
| `orca` | `0.2340` | `0.0426` | `-0.2325` | strongest classical baseline and the cleanest paper-facing comparator |
| `socnav_sampling` | `0.1773` | `0.5106` | `-0.1352` | can finish scenarios, but collision cost is too high for headline use |
| `prediction_planner` | `0.0709` | `0.2128` | `-0.1924` | active but underperforming; should be treated as a weak local implementation |
| `goal` | `0.0142` | `0.2411` | `-0.1608` | useful only as a sanity/control baseline, not a serious comparative planner |
| `sacadrl` | `0.0000` | `0.3901` | `-0.2806` | not credible as a paper-facing family representative |
| `social_force` | `0.0000` | `0.2128` | `-0.8480` | not credible as a headline baseline on the hard matrix |

## Planner quality judgment

Use the merged planner quality audit as the interpretation layer for the final campaign.

Headline current baselines:

- `orca`
- `ppo`

Control only:

- `goal`

Non-headline until improved or reproduced faithfully:

- `prediction_planner`
- `sacadrl`
- `social_force`
- `socnav_sampling`

Required paper-facing claim boundary:

- The current suite is **benchmark-honest but mixed-quality**.
- Final paper text should not imply that all implemented planners are strong or faithful representatives
  of their literature families.
- Weak results for `sacadrl`, `social_force`, and other non-headline planners should be described as
  current implementation-level evidence, not as definitive family-level evidence.

## What changed and why it matters

### 1. Collision accounting was corrected

The superseded first rerun underreported planner collisions in the summary layer. The corrected
collision-summary rerun fixed that by deriving outcome metrics from per-episode termination semantics.

Why it matters:

- planner safety numbers are now numerically trustworthy,
- downstream paper tables no longer inherit the `0.0000 collisions` artifact,
- the corrected campaign is suitable as the basis for final reporting.

### 2. SNQI was recalibrated to v3 on the corrected surface

The SNQI-v3 rerun preserved the corrected episode outcomes while improving the benchmark contract
behavior enough to pass on the final camera-ready surface.

Why it matters:

- paper-facing ranking and summary interpretation now use the corrected benchmark contract,
- the canonical publication bundle is tied to a passing contract rather than a warning state,
- the final artifacts are internally consistent and analyzer-clean.

### 3. Planner quality was classified explicitly

The planner audit removed the ambiguity that every runnable planner is automatically a credible
paper-facing baseline.

Why it matters:

- the AMV paper can separate headline baselines from weak or mismatch-prone implementations,
- follow-up improvement work can target planner quality rather than continuing to treat the suite as
  uniformly strong,
- external reproduction work is now easier to prioritize.

## Remaining limitations

These limitations still matter for paper interpretation.

- AMV coverage status remains `warn`; the final campaign is corrected and publication-ready, but it
  does not eliminate the need for careful manuscript wording.
- Absolute planner success remains low on the hard matrix; the benchmark is honest, but the current
  planner suite is not broadly strong.
- Some planners remain implementation-thin or literature-mismatch candidates, so they should not be
  overclaimed as faithful family representatives.
- `socnav_sampling` trades away too much safety to be used as a headline baseline.
- The paper should avoid presenting the current benchmark as evidence that all reviewed planner
  families are uniformly weak.

## Paper-ingestion checklist

Use this checklist when moving the benchmark evidence into the AMV paper repository.

- [x] Use `paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407` as the final camera-ready campaign root.
- [x] Use `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle` as the canonical publication bundle.
- [x] Treat `paper_experiment_matrix_v1_issue580_full_regen_all_extras_20260318_115242` as superseded and non-canonical.
- [x] Use the planner quality classifications from `reports/planner_quality_audit.md` when writing comparative discussion.
- [x] Preserve the claim boundary between local implementation evidence and literature-family evidence.
- [ ] Mirror this evidence note or its key conclusions into the AMV paper repo.
- [ ] Recheck manuscript wording so zero-success or near-zero planners are not overinterpreted.

## Final recommendation

For AMV paper work, the benchmark stack is now in the correct state to ingest results.

Use:

- the SNQI-v3 canonical campaign,
- the associated publication bundle,
- and the planner quality audit.

Do not use:

- the superseded first issue-580 rerun,
- or any wording that treats the full current planner suite as uniformly strong.
