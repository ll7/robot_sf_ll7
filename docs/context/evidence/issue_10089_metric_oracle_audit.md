# Metric oracle audit for 0.1.0

Scope: `tests/benchmark` at base `53f8f2666e98d00b70a8ef5376790be474a743aa`.
This is test-integrity evidence for #10089, under the mutation-proof ruling on
#10176 (2026-10-08), not benchmark-performance or acquired release evidence.

The audit searched Python/JSON fixtures for `approx`, `assert_allclose`, `isclose`,
`golden`, `oracle`, and long decimal literals, then inspected the input source:
a newly simulated episode, a constructed trace, or an immutable archived artifact.
A decimal literal is not by itself a saved physics result.

| Old test or family | Origin and disposition |
| --- | --- |
| `test_campaign_horizon_authority.py::test_historical_runner_cap_matches_main_oracle` (5 cases) | Moving-main native episodes: replace the saved IDs, steps, speeds and progress counts with same-commit paired execution. All five original config/planner/dev-seed combinations remain. |
| `test_metrics_characterization.py::test_mean_distance_and_clearance_average_min_per_step` | Synthetic nearest-distance arithmetic: replace with far-pedestrian invariance and a controlled nearest-pedestrian displacement, independently deriving min/mean deltas. |
| `test_metrics_characterization.py::test_path_motion_metrics_on_straight_line`, `test_path_motion_metrics_on_curved_nonuniform_trajectory` | Retain the hand-derived straight-line definition (path = 3, speed = 1, energy/jerk/curvature = 0, efficiency = 1). Replace bent-trajectory literals with nonzero unit-scaling controls, detour checks and rigid-frame invariants. Four absolute motion values also survive in the release sentinel fixture. |
| `test_metrics_characterization.py::test_force_quantiles_and_mean_on_known_magnitude` | Independent hand-calculated definition: retain the 3-4-5 vector, all quantiles and mean = 5, exceed and comfort = 1. Add scaling and rotation controls alongside it; homogeneous controls alone miss a constant multiplicative error. |
| Remaining `test_metrics_characterization.py` cases | Keep discrete collision/near-miss boundaries, success/timeout semantics, one-sample/empty/NaN guards, per-ped aggregation arithmetic, independently calculated SNQI and stability boundaries. These assert definitions, not physics at a main commit. |
| `test_curvature_v2.py::test_historical_v1_rows_keep_exact_values`; `test_fxm_metric_definitions.py`, `test_fxm2_metrics.py` | Keep historical schema dispatch and source-trace/reset/goal replay checks. These protect old-number recomputation and defect-specific definitions. |
| `test_metric_output_golden.py`, `test_metric_output_golden_surfaces.py` and `tests/fixtures/benchmark/golden/` | Keep end-to-end aggregate, summary, CSV/Markdown/LaTeX/Parquet serialization contracts on four fixed input records. They consume static episodes; simulator changes cannot rebaseline them. Never blessed in this task. |
| `test_scheduled_campaign_compatibility.py`, `test_campaign_horizon_compatibility.py`, `test_campaign_horizon_contracts.py`, `fixtures/scheduled_campaign_main_93ba0d75.json` | Keep structural/config-hash, authored-budget and terminal-label contracts. The long plausibility numbers restore archived YAML metadata for a byte hash; they are not asserted against a newly computed metric. |
| `test_aggregate_characterization.py`, `test_constraints_first_characterization.py`, `test_critical_intervals_characterization.py`, `test_snqi_scalarization_sensitivity_characterization.py` | Keep hand-constructed arithmetic, quantile/CI conventions, TTC thresholds, ranking and missing-value/serialization rules. They are independent definition checks, not saved episode results. |
| `test_heterogeneous_rank_sensitivity_characterization.py`, `test_exemplar_selection_characterization.py`, `test_episode_replay_figure_characterization.py`, `test_map_runner_characterization.py` | Keep synthetic ranking/selection, report structure, finite guards and metadata contracts. |
| `test_cross_benchmark_metrics.py`, `test_rank_metrics.py`, `test_reporting_statistics_regressions.py` | Keep wrapper units/status, hand-derived three-step elapsed time/TTC/separation, tie-handling correlations, and finite-sample statistics. |
| `test_heterogeneous_population_ablation.py::test_issue_3206_three_seed_smoke_audits_as_mean_matched_but_not_per_archetype_ready` | Keep an audit of committed 2026-06-20 receipts (the two 0.9795916847173137 means); it reads archival artifacts and never reruns their simulation. |
| `test_radius_rank_stability.py`, `test_map_runner_observations.py`, `test_dynamics_models.py`, `test_issue_6464_brne_corridor_diagnostic.py`, `test_fallback_policy.py`, `test_h600_hybrid_roster_config.py`, `test_h600_hybrid_vs_orca_s30_config.py` | Long literals are constructed rank inputs, vector/heading calculations, diagnostic records or config/identity pins. Keep; they are not moving-main scored episode oracles. |
| `test_mintorder_contract.py`, release authority/freeze/manifest/seed tests | Keep source-bound 0.0.8 provenance and admission checks. No acquired release metric, release input, freeze commit, or release branch is changed. |

## Paired episode contract

The historical side loads the historical campaign with `horizon_policy=None`;
the comparison side uses production `legacy_runner_cap` admission. Both execute
the same authored simulator limits, runner H600, dt=0.1, native planner and dev
seed at the same source commit. IDs, scenario identity, schema, step counts,
terminal labels, outcomes and effective budgets must agree. All metric keys and
nested payloads agree; numeric leaves use rel/abs 1e-12 with paired NaNs allowed.
The bound row additionally carries the expected legacy accounting provenance.

The comparator deliberately remains the historical runner-cap path, not the
0.0.8 authored-budget mode, whose timeout-label semantics differ intentionally.
The pair can detect one-sided horizon regressions but cannot detect shared
physics or scorer drift. It is not a substitute for absolute definition tests.

## Retained release drift alarm

`tests/benchmark/fixtures/release_0_0_8_metric_sentinels.json` retains four small
synthetic values already present in `test_metrics_characterization.py` at the historical F2 source
`66f402ba` (not the 0.0.8 F3 freeze, which is
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`): straight path length and average speed, bent jerk and v2
curvature. Its source commit, concrete three-width 0.0.8 manifest path and SHA256,
metric schema and tolerance are explicit. The manifest bytes were checked equal
to the freeze-source blob before creating the fixture. Tests read that manifest;
they do not simulate release seeds or execute the freeze branch.

These are synthetic definition sentinels, not new claims about 0.0.8 acquisition
numbers. Existing v1 recomputation and archived release/golden artifacts remain
checkable. Rebaseline only at a release, recording both previous and moving
source commits. Tolerance is rel=1e-6, abs=1e-9 rather than 15-digit equality.

## Proof and limits

New tests pass against unchanged base production code: this is coverage
replacement, not a bug fix. Each new test has a deliberate metric mutation;
additional mutations target individual scaling controls, identity/label fields,
nested metric comparison and the manifest association. The accompanying
mutation receipt records the exact selector, changed behavior, exit code and
assertion failure. All mutations are temporary and restored byte-for-byte;
production metric code has no delivered diff.

The scaling examples stay above curvature displacement/length floors; fixed threshold
boundaries remain covered by `test_curvature_v2.py`. Scaling tests prove homogeneity
in that regime, not invariance across a threshold crossing.

Only targeted tests run. No planner/benchmark behavior changes, so the empty-world
behavior sweep does not apply. No release acquisition, full suite, hosted CI
polling or performance claim is part of this proof.

Recorded results: 77 killed mutations, 94 behavioral assertion failures, all 52
new parameterized tests covered. One incorrectly addressed nested-key attempt
was rejected as a lookup error and replaced. A shared +1 average-speed mutation
leaves all five native pairs green while failing the release-pinned speed alarm.
Final pristine selection: 75 passed; focused integration: 148 passed. Ruff and
diff whitespace checks passed. Exact commands, mutations, failure excerpts and
source/test digests are in [the receipt](issue_10089_metric_oracle_mutations.json).

Review follow-up: #10260 tracks the reported hosted/local native-episode drift and the sealed ledger's stale enforcement reference. Paired comparisons and synthetic definitions cannot detect all native physics drift.

Review-fix validation (2026-10-09): the restored force definition rejects a temporary
`ped_force_mean` multiplier of 2 (10 versus expected 5; one assertion failure).
After byte-for-byte restoration, the three changed test files pass 90 cases;
`tests/unit` passes 854 cases with four workers. The local PR contract passes.
This supplements the original mutation receipt, whose bytes remain unchanged.
