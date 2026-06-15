# Issue #2689 Release Evidence Handoff Snapshot (2026-06-15)

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2689>

## Caveats (Read First)

- This is a **docs-only handoff snapshot**, not new benchmark evidence, a rerun, or a paper claim.
- The snapshot date is **2026-06-15**. All `origin/main` pointers use the HEAD commit at that time.
- Only artifacts from the GitHub release `0.0.2` are treated as **release-backed / claim-grade**.
- Everything merged to `origin/main` after `0.0.2` is **diagnostic-only or infrastructure** unless a later release supersedes `0.0.2`.
- "Future-only" items are explicitly not ready for benchmark or manuscript claims.
- Local `output/`, uncommitted context packs, and unmerged branches are **not** durable evidence.
- SocNavBench remains excluded from the `0.0.2` scope because licensed assets were not staged.

## Snapshot Metadata

| Field | Value |
|---|---|
| Snapshot issue | [#2689](https://github.com/ll7/robot_sf_ll7/issues/2689) |
| Snapshot date | 2026-06-15 |
| `origin/main` HEAD | `ce2f63f34b695b1b0949bbf0195395e761c9600f` |
| Release tag | `0.0.2` |
| Tag target commit | `cbeaca6109654b4053c19542a0a17ed656a387a6` (2026-04-13) |
| Release source commit (per #2686) | `f7ebdcae2375d085e925213197a75a386e26a79c` |
| Release URL | <https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2> |
| Release DOI | <https://doi.org/10.5281/zenodo.19563812> |
| Release archive | `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz` |
| Archive SHA-256 | `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90` |

## 1. Release-Backed Evidence (Claim-Grade)

**Claim boundary before the artifact list:**
The only claim-grade evidence is the frozen `0.0.2` publication bundle. It covers a scoped, seven-planner, 47-scenario, three-seed, differential-drive campaign. It does not include SocNavBench, does not establish new rankings beyond the released tables, and does not generalize to other seeds, scenarios, or planner versions.

| Artifact / surface | Exact location | Manuscript use | Boundary |
|---|---|---|---|
| Release 0.0.2 publication bundle | GitHub release asset (see metadata above) | Core results tables | Frozen release; no rerun. |
| Table bundle handoff note | `docs/context/issue_2686_release_0_0_2_table_bundle.md` | Maps tables to manuscript sections | Repackages existing release tables only. |
| Table artifact spec | `docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_spec.json` | Reviewable manifest | Tracked copy of release bundle metadata. |
| Table artifact manifest | `docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_manifest.json` | Reviewable manifest | Tracked copy of generated manifest. |
| Table checksums | `docs/context/evidence/issue_2686_release_0_0_2_table_bundle/checksums.sha256` | Integrity check | SHA-256 of disposable payload files. |
| Core results table | `tab_results_overview` in release bundle | `tab:results-overview` | 7-planner overview only. |
| Planner results table | `tab_robot_sf_release_planner_results` in release bundle | `tab:robot_sf_release_planner_results` | Existing ranking; no new claim. |
| Failure-count slices | `tab_release_failure_count_slices` in release bundle | `tab:release_failure_count_slices` | Scenario-family breakdown; not a failure taxonomy. |

**Recommended downstream use:** Use the three release tables directly for any manuscript section that matches the `0.0.2` scope. Re-download the release archive and verify SHA-256 before regenerating downstream artifacts.

## 2. Diagnostic-Only `origin/main` Evidence

**Claim boundary before the artifact list:**
All items below are merged to `origin/main` but are explicitly scoped as schema, infrastructure, synthesis, or diagnostic evidence. They do not establish benchmark rankings, planner safety, mechanism effectiveness, or paper-facing claims.

| Issue / PR | Commit | Exact files | Evidence type | Boundary |
|---|---|---|---|---|
| [#2688](https://github.com/ll7/robot_sf_ll7/issues/2688) trace predicate matrix | `e8d45d01` (`#2894`) | `docs/context/issue_2688_trace_predicate_matrix.md`, `configs/benchmarks/issue_2688_trace_predicate_matrix.yaml`, `robot_sf/analysis_workbench/trace_failure_predicates.py`, `scripts/tools/build_trace_failure_predicate_tables.py` | Schema/trace contract | Proposed matrix; rate interpretation is claim-ineligible until promoted. |
| [#2882](https://github.com/ll7/robot_sf_ll7/pull/2882) learned-prediction readiness gate | `3f9504e3` | `docs/context/issue_2768_learned_prediction_readiness.md`, `scripts/validation/validate_learned_prediction_readiness.py`, `tests/validation/test_validate_learned_prediction_readiness.py` | Fail-closed readiness validator | Validator now fails closed; training readiness remains **not satisfied**. |
| [#2659](https://github.com/ll7/robot_sf_ll7/issues/2659) manifest lineage schema | `90754c2d` (`#2702`) | `docs/context/issue_2659_lineage_schema_unification.md`, `robot_sf/benchmark/manifest_lineage.py` | Schema reviewability | Additive contract only; does not make manifests benchmark-valid. |
| [#2662](https://github.com/ll7/robot_sf_ll7/issues/2662) signal-state promotion contract | `044b2c58` (`#2703`) | `docs/context/issue_2662_signal_state_promotion_contract.md`, `robot_sf/benchmark/map_runner.py`, `configs/benchmarks/signalized_pedestrian_crossing_issue_2474.yaml` | Schema/trace contract | Proxy rows are diagnostic; observable rows need future runtime contract. |
| [#2799](https://github.com/ll7/robot_sf_ll7/issues/2799) signalized-crossing runtime smoke | `94eee2c6` | `docs/context/evidence/issue_2799_signalized_runtime/README.md`, `summary.json`, `report.md` | Runtime denominator plumbing | Proves denominator/exclusion plumbing, not traffic-light realism or planner performance. |
| [#2762](https://github.com/ll7/robot_sf_ll7/issues/2762) negative result register | `2a87f526` (`#2773`) | `docs/context/negative_result_register.md`, `docs/context/evidence/issue_2762_negative_result_register/register.json` | Synthesis/planning aid | Tracks `revise`/`diagnostic_only`/`failed`/`inconclusive` findings; not new evidence. |
| [#2864](https://github.com/ll7/robot_sf_ll7/issues/2864) forecast-lane synthesis | `25c2483e` (`#2881`; summarizes PRs `#2849`-`#2862`) | `docs/context/issue_2864_forecast_lane_synthesis.md`, `docs/context/prediction_lane_dependency_graph.json` | Diagnostic-to-infrastructure synthesis | Forecast schema/baseline/scaffolding only; no benchmark safety/progress/transfer claim. |

## 3. Future-Only Evidence

**Claim boundary before the artifact list:**
These surfaces are research-direction or preflight scaffolding. They are explicitly **not** claim-grade in the current snapshot and should not be cited as established results.

| Issue / surface | Exact files | Why future-only |
|---|---|---|
| [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835) forecast research lane epic | `docs/context/issue_2864_forecast_lane_synthesis.md`, `docs/context/prediction_lane_dependency_graph.json` | Open epic; downstream closed-loop benchmark evidence is still blocked. |
| [#2768](https://github.com/ll7/robot_sf_ll7/issues/2768) learned-prediction readiness contract | `docs/context/issue_2768_learned_prediction_readiness.md`, `scripts/validation/validate_learned_prediction_readiness.py` | Issue is closed, but the contract status remains **NOT-TRAINING-READY**; prerequisites are not satisfied. |
| [#2474](https://github.com/ll7/robot_sf_ll7/issues/2474) signalized-crossing benchmark | `docs/context/issue_2474_signalized_crossing_benchmark.md`, `configs/benchmarks/signalized_pedestrian_crossing_issue_2474.yaml` | Observable signal-state contract and traffic-light realism claims are future work. |
| Topology successor / reselection lane | `docs/context/issue_2704_progress_gated_topology_successor.md`, `docs/context/issue_2716_topology_reselection_cross_slice.md`, `docs/context/issue_2742_topology_reselection_successor.md`, `docs/context/issue_2752_topology_reselection_mechanism.md`, `docs/context/issue_2801_topology_successor_recommendation.md`, `docs/context/issue_2804_non_topology_successor.md`, `docs/context/issue_2706_topology_lane_synthesis.md` | Active research lane; current results are `revise` or diagnostic-only (see negative result register NR-001). |

## 4. Open Blockers

| Blocker | Issue / surface | Why it blocks downstream claims |
|---|---|---|
| Learned-prediction training not ready | [#2768](https://github.com/ll7/robot_sf_ll7/issues/2768), `scripts/validation/validate_learned_prediction_readiness.py` | Durable trace dataset registry, leakage-free splits, target horizon, and closed-loop transfer gates are not yet satisfied. |
| Forecast lane benchmark evidence | [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835), `docs/context/prediction_lane_dependency_graph.json` | Closed-loop same-seed campaigns under transfer slices are missing; safety/progress/transfer claims unsupported. |
| SocNavBench assets unavailable | [#1584](https://github.com/ll7/robot_sf_ll7/issues/1584), [#2397](https://github.com/ll7/robot_sf_ll7/issues/2397), [#691](https://github.com/ll7/robot_sf_ll7/issues/691) | Licensed SocNavBench assets were not staged; any benchmark row relying on them must fail closed. |
| Signalized-crossing observable contract | [#2662](https://github.com/ll7/robot_sf_ll7/issues/2662), [#2474](https://github.com/ll7/robot_sf_ll7/issues/2474) | `planner_observable` rows need explicit runtime fields before they can enter benchmark denominators. |
| Topology reselection clearance | [#2716](https://github.com/ll7/robot_sf_ll7/issues/2716), [#2742](https://github.com/ll7/robot_sf_ll7/issues/2742), [#2704](https://github.com/ll7/robot_sf_ll7/issues/2704) | Progress-gated reselection does not clear hard non-canonical slices; needs redesign, not more threshold tuning. |

## 5. Do Not Use

| Source | Reason |
|---|---|
| `output/` and other worktree-local artifacts | Disposable local state; not durable or reproducible across checkouts. |
| Uncommitted context packs under `output/context_packs/` | Generated, temporary, and not source-controlled. |
| Unmerged branches / open PRs | Not reviewed or integrated; claim boundaries are undefined. |
| Predicate table outputs without `#2688` matrix, or with matrix status `proposed` | Rate interpretation is claim-ineligible. |
| Forecast baselines and calibration reports as safety/progress claims | They are diagnostic inputs only. |
| Any SocNavBench row unless the licensed assets are explicitly staged | Must fail closed per [#691](https://github.com/ll7/robot_sf_ll7/issues/691). |

## 6. Recommended Downstream Use

1. **For manuscript Results:** start with the three release-backed tables from `0.0.2` only.
2. **For Discussion / Background:** cite diagnostic notes (#2688, #2659, #2662, #2799, #2864) with their explicit boundaries.
3. **For Future Work:** use the prediction-lane dependency graph and open blockers to sequence the next evidence-producing issues.
4. **Before any learned-prediction claim:** run `scripts/validation/validate_learned_prediction_readiness.py` and confirm every prerequisite in [#2768](https://github.com/ll7/robot_sf_ll7/issues/2768) is satisfied.
5. **Before any signalized-crossing claim:** confirm `planner_observable` runtime fields and denominator policy are populated per [#2662](https://github.com/ll7/robot_sf_ll7/issues/2662).
6. **Before any topology-successor claim:** confirm a successor design that clears hard slices, not just activates, per negative result register NR-001.

## Validation

Run these checks after editing this note or any linked surface:

```bash
# Path existence for tracked evidence
ls docs/context/issue_2686_release_0_0_2_table_bundle.md \
   docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_spec.json \
   docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_manifest.json \
   docs/context/evidence/issue_2686_release_0_0_2_table_bundle/checksums.sha256 \
   docs/context/issue_2688_trace_predicate_matrix.md \
   docs/context/issue_2659_lineage_schema_unification.md \
   docs/context/issue_2662_signal_state_promotion_contract.md \
   docs/context/evidence/issue_2799_signalized_runtime/README.md \
   docs/context/negative_result_register.md \
   docs/context/issue_2864_forecast_lane_synthesis.md \
   docs/context/issue_2768_learned_prediction_readiness.md

# Whitespace / merge-conflict check
git diff --check
```
