# Issue 635 SNQI v3 Paper-Facing Contract Note

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#635` SNQI v3 paper-facing contract delta and canonical field mapping
- `robot_sf_ll7#581` paper evidence delta report + checklist signoff
- `robot_sf_ll7#580` camera-ready collision summary fix
- `robot_sf_ll7#579` SNQI v3 recalibration + contract update

## Purpose

Provide one paper-facing contract note for the final canonical publication bundle so downstream AMV
paper work can update Methods wording and figure/table provenance without inferring metric semantics
from artifact names.

This note answers five concrete questions:

1. what exact SNQI v3 contract is used by the canonical publication bundle,
2. what changed relative to the corrected pre-v3 paper-facing rerun,
3. which exported benchmark fields kept the same names and meanings,
4. whether paper tables and plots must be regenerated from the final bundle,
5. what manuscript wording can be tightened and what must remain cautious.

## Canonical source of truth

Use these artifacts as the final paper-facing source of truth.

### Canonical camera-ready campaign

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407`
- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/campaign_report.md`
- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/campaign_summary.json`
- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/snqi_diagnostics.json`

### Canonical publication bundle

- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle`
- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle/publication_manifest.json`
- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle/payload/campaign_manifest.json`

### Pre-v3 comparison point

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue580_collision_summary_fix_v2_20260318_121431`

## Exact SNQI v3 contract used by the canonical bundle

The final canonical campaign manifest pins these SNQI assets:

- weights path: `configs/benchmarks/snqi_weights_camera_ready_v3.json`
- baseline path: `configs/benchmarks/snqi_baseline_camera_ready_v3.json`
- weights version: `snqi_weights_camera_ready_v3`
- baseline version: `snqi_baseline_camera_ready_v3`
- weights SHA-256: `71a67c3c02faff166f8c96bef8bcf898533981ca2b2c4493829988520fb1aeb2`
- baseline SHA-256: `329ca5766491e1587979d0a435c7ba676e148ccdff97040a36546bbb9414035a`
- contract status: `pass`
- contract enforcement: `warn`

### Final term list and weights

| Term | Weight |
|---|---:|
| `success_reward` | `0.19045845847432735` |
| `time_penalty` | `0.09491099136070058` |
| `collisions_penalty` | `0.10483542508043969` |
| `near_penalty` | `0.30825830332144416` |
| `comfort_penalty` | `0.17983060763794978` |
| `force_exceed_penalty` | `0.0692114485473155` |
| `jerk_penalty` | `0.05249476557782281` |

### Normalization anchors used for evaluation

The final SNQI diagnostics use these med/p95 anchors for the evaluated terms:

| Term | med | p95 |
|---|---:|---:|
| `time_to_goal_norm` | `1.0` | `1.05` |
| `collisions` | `0.0` | `1.0` |
| `near_misses` | `0.0` | `21.0` |
| `force_exceed_events` | `1.0` | `50.0` |
| `jerk_mean` | `0.07293007251387372` | `0.8018190086690077` |

The baseline file also records additional non-evaluated anchors such as `comfort_exposure`,
`curvature_mean`, `energy`, `min_distance`, and `path_efficiency`. They are part of the checked-in
baseline asset, but the canonical SNQI diagnostics for this bundle evaluate the five terms above.

### Final diagnostics summary

From `reports/snqi_diagnostics.json`:

- contract status: `pass`
- rank alignment (Spearman): `0.5357142857142857`
- outcome separation: `0.19474729863388224`
- dominant component: `time_penalty`
- dominant component mean absolute contribution: `0.0922742543970636`
- baseline degeneracy adjustments: `0`

## Contract delta versus the corrected pre-v3 rerun

The corrected issue-580 rerun already had fixed collision accounting and a passing SNQI contract, but
it used the v2 SNQI asset set and a smaller diagnostics schema.

### Asset delta

| Contract element | Pre-v3 corrected rerun | Final canonical rerun |
|---|---|---|
| weights path | `configs/benchmarks/snqi_weights_camera_ready_v2.json` | `configs/benchmarks/snqi_weights_camera_ready_v3.json` |
| baseline path | `configs/benchmarks/snqi_baseline_camera_ready_v2.json` | `configs/benchmarks/snqi_baseline_camera_ready_v3.json` |
| weight on `time_penalty` | `0.27972800933897135` | `0.09491099136070058` |
| weight on `near_penalty` | `0.1784948603206522` | `0.30825830332144416` |
| weight on `collisions_penalty` | `0.05020826849915813` | `0.10483542508043969` |
| `time_to_goal_norm` anchors | `med=0.09`, `p95=1.0` | `med=1.0`, `p95=1.05` |
| `near_misses` anchors | `med=0.0`, `p95=1.0` | `med=0.0`, `p95=21.0` |
| `jerk_mean` anchors | `med=0.1309413768279867`, `p95=0.26496668795365447` | `med=0.07293007251387372`, `p95=0.8018190086690077` |
| contract thresholds | rank/separation only | rank/separation + dominance thresholds |

### Diagnostics schema delta

Compared with the pre-v3 corrected rerun, the final v3 diagnostics add explicit provenance and
interpretation fields:

- `weights_path`
- `weights_version`
- `weights_sha256`
- `baseline_path`
- `baseline_version`
- `baseline_sha256`
- `dominant_component`
- `dominant_component_mean_abs`
- dominance thresholds inside `thresholds`

Interpretation:

- the SNQI term family stayed the same,
- but the weights and normalization anchors changed materially,
- and the final diagnostics schema is richer and better suited for paper-facing provenance.

## What changed in exported benchmark fields

### Stable planner-row export contract

The exported planner table header is unchanged between the corrected issue-580 rerun and the final
issue-579 rerun.

Stable planner-row fields include:

- `success_mean`
- `collisions_mean`
- `snqi_mean`
- `runtime_sec`
- `projection_rate`
- `infeasible_rate`
- `execution_mode`
- `socnav_prereq_policy`
- `learned_policy_contract_status`

Interpretation:

- planner-row field names are stable,
- planner-row meanings are stable,
- but their values should still be taken only from the final canonical rerun.

### Campaign-level metadata expansion

The final canonical `campaign_summary.json` adds campaign-level SNQI provenance fields beyond the
corrected issue-580 rerun:

- `snqi_weights_sha256`
- `snqi_baseline_sha256`
- `snqi_contract_dominant_component`
- `snqi_contract_dominant_component_mean_abs`

Interpretation:

- exported planner rows remain paper-table compatible,
- campaign-level contract metadata became more explicit in the final rerun.

## Canonical field mapping for paper use

Use this table when mapping benchmark artifacts into paper tables, figure captions, and Methods
wording.

| Field / source | Meaning | Paper column name can stay unchanged? | Source or semantics changed? |
|---|---|---|---|
| `campaign_table.csv: success_mean` | fraction of successful episodes | yes | semantics stable; regenerate value from final bundle |
| `campaign_table.csv: collisions_mean` | fraction of episodes terminating in collision | yes | semantics stable on corrected stack; regenerate value from final bundle |
| `campaign_table.csv: snqi_mean` | composite SNQI score for the final campaign contract | yes | semantics changed because v3 weights and baselines replaced v2 |
| `campaign_table.csv: runtime_sec` | planner wall-clock runtime across evaluated episodes | yes | semantics stable; regenerate from final bundle |
| `campaign_table.csv: projection_rate` | fraction of commands projected from planner-native command space to benchmark command space | yes if discussed | semantics stable |
| `campaign_table.csv: infeasible_rate` | fraction of raw commands considered infeasible before projection/handling | yes if discussed | semantics stable |
| `campaign_table.csv: execution_mode` | whether the planner ran natively or through an adapter | yes | semantics stable |
| `campaign_table.csv: socnav_prereq_policy` | strict-vs-fallback prereq behavior | yes if discussed | semantics stable |
| `reports/snqi_diagnostics.json: rank_alignment_spearman` | campaign-level SNQI rank alignment check | no direct paper table column unless explicitly discussed | v3 diagnostics schema adds provenance and dominance context |
| `reports/snqi_diagnostics.json: outcome_separation` | campaign-level SNQI separation check | no direct paper table column unless explicitly discussed | semantics stable; use final value only |
| `reports/snqi_diagnostics.json: dominant_component` | name of the SNQI component with the largest mean absolute contribution | likely supporting text only | new v3 paper-facing interpretation field |
| `reports/snqi_diagnostics.json: dominant_component_mean_abs` | value of the largest mean absolute SNQI component contribution | likely supporting text only | new v3 paper-facing interpretation field |
| `publication_manifest.json: provenance.run_dir` | canonical benchmark root for the bundle | not a table column | use in provenance / reproducibility text |
| `payload/campaign_manifest.json: snqi_weights_path` | asset pointer for bundled campaign payload | not a table column | stable pointer, but not a substitute for full diagnostics |

## Important publication-bundle caveat

The final publication bundle contains `reports/snqi_diagnostics.json` and `reports/snqi_diagnostics.md`
as report files, but `payload/campaign_manifest.json` still records:

- `snqi_contract_status: not_evaluated`
- `artifacts.snqi_diagnostics_json: null`
- `artifacts.snqi_diagnostics_md: null`

Interpretation:

- the publication bundle **does** include the diagnostics reports,
- but the payload manifest is not a complete SNQI diagnostics index,
- so paper-side tooling should read the bundled `reports/snqi_diagnostics.*` files directly when it
  needs the final SNQI contract evidence.

## Do paper tables and plots need regeneration?

Yes.

All paper tables and plots that depend on benchmark numbers should be regenerated from the final
canonical publication bundle:
- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle`

Why this is required:

- the final canonical rerun uses different SNQI assets and a different benchmark config hash than the
  corrected pre-v3 rerun,
- planner summary values are not numerically identical between the corrected issue-580 rerun and the
  final issue-579 rerun,
- the final rerun is therefore the only authoritative paper-facing source.

Observed consequence:

- planner-row field names remained stable,
- but planner-row values changed,
- and `snqi_mean` changed both numerically and semantically because the contract changed.

## Did CI / aggregation logic remain identical between issue-580 and issue-579?

Not in a way that should be assumed by downstream paper work.

What can be stated safely from the exported artifacts:

- the scenario matrix hash is the same: `5f31c9866569`
- the resolved eval seed set is the same: `111`, `112`, `113`
- the planner-row export header is unchanged
- the campaign-level SNQI metadata schema expanded in the final rerun
- the final rerun uses a different config hash and different SNQI asset paths
- planner summary values changed between the two reruns

Conservative interpretation:

- planner-row aggregation fields remained structurally compatible,
- but the final rerun should not be treated as a pure metadata-only or ranking-only update,
- so paper-side logic should consume the final issue-579 bundle as a fresh canonical export rather
  than assuming issue-580 and issue-579 are interchangeable.

## Follow-up From Issue 822

On 2026-04-14, [issue 822](./issue_822_snqi_strengthening_analysis.md) replayed the frozen
publication bundle against both the v1 and v3 SNQI assets and found that the v3 contract is
materially better aligned with the planner and component signals on that slice.

Observed follow-up signal:

- planner ordering changed from `goal > orca > social_force` under the v1 contract to
  `orca > goal > social_force` under the v3 contract
- the v3 contract shows materially better alignment with `success`, `time_to_goal_norm`,
  `near_misses`, `comfort_exposure`, and `force_exceed_events`
- `collisions` is constant on the frozen slice, so collision correlation is undefined there

Recommendation carried forward from issue 822:

- keep SNQI in the main narrative as an operational multi-objective aggregation metric
- keep the explicit caveat that it is a benchmark aggregate, not a universal ground-truth utility
- use the strengthened `reports/snqi_diagnostics.{json,md}` and `reports/snqi_sensitivity.csv`
  artifacts as the paper-facing evidence surface for that claim

The scratch analysis artifacts live under:

- `output/ai/autoresearch/snqi_issue822/v1`
- `output/ai/autoresearch/snqi_issue822/v3`

## Follow-up From Issue 838

On 2026-04-16, [issue 838](./issue_838_snqi_calibration_analysis.md) added a reproducible
calibration-robustness workflow for testing v3 against local weight perturbations and alternative
normalization anchors.

The issue-838 run available in this branch used the closest mounted frozen camera-ready campaign,
because the exact issue-635 publication bundle path was not present in the checkout:

- `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_rel_0_0_2_full_rerun_20260414_081244`

Observed signal on that slice:

- v3 component alignment was healthy: `6 / 6` variable metrics aligned with expected direction
- local +-15% weight perturbations stayed highly rank-stable at the planner level
  (`min rho = 0.928571`)
- normalization-anchor variants were materially more sensitive
  (`anchor min planner rho = 0.607143`)
- no tested variant justified a replacement v4 contract

Conservative carry-forward:

- keep the fixed v3 asset contract unchanged for already-frozen paper bundles,
- treat SNQI more cautiously as a supporting synthesis aid unless the canonical publication bundle
  rerun clears the same anchor-robustness check,
- do not retrofit current paper-facing values from the issue-838 analysis alone.

## Conservative manuscript recommendation

Methods wording that can now be tightened:

- the canonical paper-facing benchmark bundle uses SNQI v3,
- the final SNQI weights and baseline assets are fixed and versioned,
- the final publication bundle is the authoritative source for benchmark tables and plots,
- planner-row export fields such as success, collisions, and runtime use a stable exported schema.

Methods wording that should remain cautious:

- do not claim that the issue-579 rerun only changed ranking while leaving all planner outcomes
  identical to the corrected issue-580 rerun,
- do not imply that the bundle payload manifest alone captures the full SNQI diagnostics contract,
- do not state that all implemented planners are faithful literature-family representatives,
- do not overclaim metric-method closure beyond the final v3 asset pinning and diagnostics evidence,
- do not present SNQI v3 as anchor-robust or locally optimal without rerunning the issue-838
  calibration workflow on the canonical publication bundle.

## Final recommendation

For downstream AMV paper work:

- use the final issue-579 publication bundle as the only canonical benchmark source,
- regenerate all paper tables and plots from that bundle,
- cite SNQI v3 using the exact pinned asset paths, versions, and hashes above,
- treat the corrected issue-580 rerun only as the immediate pre-v3 comparison point, not as an
  interchangeable source of benchmark numbers.
