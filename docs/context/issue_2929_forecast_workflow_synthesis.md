# Post-#2883-to-#2893 Forecast and Workflow Synthesis

Issue: [#2929](https://github.com/ll7/robot_sf_ll7/issues/2929)

Status: current synthesis.

## Claim Boundary

This note is a compact routing summary for the forecast/analysis wave #2883-#2893 and
workflow-hardening PRs #2891-#2893. It is strictly diagnostic-to-infrastructure evidence.
It does not establish benchmark, dissertation, paper-facing, or safety claims.

See [issue_2864_forecast_lane_synthesis.md](issue_2864_forecast_lane_synthesis.md) for the
prior wave #2849-#2862 synthesis, which remains the authoritative lane-state document for
schema/baseline/metric decisions. This note adds the #2883-#2893 wave on top.

## PR-to-Issue Mapping and Evidence Type

| PR | Title | Issue(s) | Evidence type | Classification |
| --- | --- | --- | --- | --- |
| [#2883](https://github.com/ll7/robot_sf_ll7/pull/2883) | Populate forecast transferability matrix rows | #2866 (closed) | analysis/tool | diagnostic-only, blocked |
| [#2884](https://github.com/ll7/robot_sf_ll7/pull/2884) | Add semantic metadata forecast fixtures | #2868 (closed) | diagnostic evidence | diagnostic-only |
| [#2885](https://github.com/ll7/robot_sf_ll7/pull/2885) | Add forecast calibration report evidence | #2865 (closed) | analysis | diagnostic-only, wait |
| [#2886](https://github.com/ll7/robot_sf_ll7/pull/2886) | Add forecast risk calibration diagnostic | #2869 (closed) | analysis/diagnostic | diagnostic-only, wait |
| [#2887](https://github.com/ll7/robot_sf_ll7/pull/2887) | Add forecast horizon timestep ablation | #2837 (closed) | analysis | analysis-only, not navigation evidence |
| [#2888](https://github.com/ll7/robot_sf_ll7/pull/2888) | Add claim matrix chapter targets | #2761 (closed) | support/tooling | support helper |
| [#2891](https://github.com/ll7/robot_sf_ll7/pull/2891) | Document worktree-safe CI watcher | #2889 (closed) | workflow infra | hardening |
| [#2892](https://github.com/ll7/robot_sf_ll7/pull/2892) | Add bounded CI watcher wall timeout | #2890 (closed) | workflow infra | hardening |
| [#2893](https://github.com/ll7/robot_sf_ll7/pull/2893) | Add manifest lineage graph report | #2722 (closed) | support/tooling | support helper |

PRs #2889 and #2890 were issue-task PRs resolved by #2891 and #2892 respectively; no separate
PR existed for those numbers.

## Established Evidence (Complete Infrastructure)

These items are done and required for the forecast lane:

- **Transferability matrix row coverage** (#2866 / PR #2883): row-level scenario-family,
  actor-class, horizon, semantic-metadata, artifact-input, and unavailable-dimension handling
  added. Deterministic CLI with `--generated-at-utc` for reproducible evidence.
  Evidence: `docs/context/evidence/issue_2866_transferability_matrix_rows/`.
- **Semantic metadata fixtures** (#2868 / PR #2884): four durable
  `simulation_trace_export.v1` fixtures for signalized crossing, goal-directed crossing,
  waiting-with-intent-change, and route-conflict goal cases. Metadata-present vs
  metadata-absent rows are now separable.
  Evidence: `docs/context/evidence/issue_2868_semantic_metadata_fixtures_2026-06-15/`.
- **Calibration/reliability report** (#2865 / PR #2885): `ForecastCalibrationReport.v1` rows
  with actor-class availability, semantic metadata availability, miss rate, failure taxonomy,
  and forecast-risk eligibility. Accepts input from #2868 comparison report.
  Evidence: `docs/context/evidence/issue_2865_forecast_calibration_report_2026-06-15/`.
- **Forecast risk calibration filter** (#2869 / PR #2886): deterministic diagnostic comparing
  five forecast-risk scoring modes (`no_risk`, `raw_risk`, `calibration_filtered`,
  `actor_class_aware`, `observation_tier_aware`).
  Evidence: `docs/context/evidence/issue_2869_forecast_risk_calibration_filter_2026-06-15/`.
- **Horizon/timestep ablation** (#2837 / PR #2887): deterministic horizon x output-`dt_s`
  ablation evaluating 65 of 180 cells across durable fixtures. Short/medium presets
  recommended; long horizon (3.0s) unavailable on this fixture set.
  Evidence: `docs/context/evidence/issue_2837_horizon_timestep_ablation_2026-06-15/`.
- **Claim matrix chapter targets** (#2761 / PR #2888): optional `chapter_target` and
  `chapter_target_justification` fields in dissertation artifact specs and exported manifests.
  Diagnostic-only guardrails for rows targeting limitations/methodology/future-work sections.
- **Worktree-safe CI watcher** (#2889 / PR #2891): documented shared-venv wrapper invocation
  for `check_pr_ci_status.py`; avoids permission fallback and per-worktree `.venv` creation.
- **Bounded CI watcher wall timeout** (#2890 / PR #2892): `--max-wall-seconds` flag for
  non-interactive local stop in long polling CI monitoring.
- **Manifest lineage graph** (#2722 / PR #2893): compact JSON graph and Markdown adjacency
  reports over manifest lineage inputs. Connected, missing, ambiguous, and inconclusive trace
  fixtures with repo-relative paths.

## Diagnostic-Only Outputs (Not Benchmark Evidence)

The following are explicitly diagnostic-only and must not be cited as benchmark, paper-facing,
or safety evidence:

- **Transferability matrix rows**: deployable cells demonstrate metadata preservation only;
  oracle rows remain diagnostic-only; missing deployable metadata cells force `stop`.
  Decision: `stop`. Claim status: `blocked`.
- **Semantic metadata comparison**: authored fixtures with metadata-present/absent rows across
  five baselines. Result: diagnostic-only. No human realism, closed-loop navigation, or safety
  claim supported.
- **Calibration report**: 40 reliability rows, 10 zero-denominator limitation rows. No
  predictor is forecast-risk eligible (actor class unavailable, bottleneck/crossing_proxy rows
  retain zero denominators). Decision: `wait`.
- **Risk calibration filter**: `calibration_filtered`, `actor_class_aware`, and
  `observation_tier_aware` modes are explicitly blocked by #2865 calibration evidence.
  `raw_risk` is diagnostic-only: it penalizes the high-risk fixture and preserves false-positive
  suppression, but this is not benchmark evidence. Decision: `wait`.
- **Horizon ablation**: analysis-only with short/medium/long preset recommendations. Long
  horizon (3.0s) explicitly unavailable. Classification:
  `analysis_only_not_navigation_evidence`.

## Blocked Work

| Work | Blocking condition | Evidence gate |
| --- | --- | --- |
| Learned probabilistic graph predictor (#2844) | Schema/data/metrics/coupling prerequisites not met | #2836, #2843, #2865, #2866 |
| Transformer/diffusion study (#2845) | Lighter prerequisites and bounded data do not yet exist | #2836, #2839, #2843 |
| Predictive planner v2 comparison (#1490) | Do not repeat old four-way expansion until revised gate evidence exists | #2843 |
| Live replay trace-derived comparison (#2790) | Blocked on #2777 live planner observation-noise replay | #2777 |
| Calibration-filtered risk scoring | Actor class unavailable in all calibration rows | #2865 |
| Long-horizon (3.0s) ablation | Durable traces too short | #2837 fixture set |
| Transferability under noise/latency/dropout | Oracle-only or blocked deployable metadata cells | #2866 matrix |

## Next Routing Targets

In recommended execution order:

1. **#2944**: Native CV-only closed-loop smoke (closed). Ran the `none` + `cv`
   forecast-variant replay with the `BaselineProbabilisticPredictor` from #2941. This
   smoked the config key and protocol before full-matrix expansion.
2. **#2941**: Native forecast replay path (closed). Expanded the #2944 smoke toward the
   native replay path. Requires the `forecast_variant` config key and
   `BaselineProbabilisticPredictor` registration from the #2941 implementation. Evidence bundle
   exists at
   `docs/context/evidence/issue_2941_native_forecast_replay_*/`.
3. **#2937**: Denominator-health fixture repair (already closed). Repaired seven #2903 fixture
   gaps; 164/180 evaluated cells (91.1%). Evidence:
   `docs/context/evidence/issue_2937_horizon_denominator_health_2026-06-16/`.
4. **#2902**: Live forecast replay gate v1 (already in place from prior wave). Remains
   diagnostic-only until native planner consumption path is standard.
5. **#2777**: Live planner observation-noise replay. Priority: medium. Follow-up after #2790.
6. **#2844** / **#2845**: Learned predictor expansion. Only after closed-loop coupling gate
   (#2843) reports `continue` and transferability matrix includes oracle + deployable rows.

## Do-Not-Claim Buckets

- No forecast-based improvement in local-navigation safety is benchmark-supported.
- No forecast-based improvement in navigation progress is benchmark-supported.
- No forecast under transfer (noise, latency, dropout, occlusion, map-family, density,
  actor-type shifts) is benchmark-supported.
- No predictor-accuracy or planner-ranking claim from current evidence.
- No human realism or behavioral-realism claim from authored trace fixtures.
- No dissertation or paper-facing claim supported by diagnostic-only ablation, calibration,
  or transferability evidence.
- No safety claim from forecast-risk scoring (weight defaults to 0.0).

## Related Issue Status

| Issue | State (2026-06-16) | Priority | Role | Revival condition |
| --- | --- | --- | --- | --- |
| [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835) | OPEN | high | Epic | Continue infrastructure/scaffold mode |
| [#2777](https://github.com/ll7/robot_sf_ll7/issues/2777) | OPEN | medium | Live planner obs-noise replay | Ready for execution |
| [#2790](https://github.com/ll7/robot_sf_ll7/issues/2790) | OPEN/blocked | medium | Trace-derived vs live comparison | Blocked by #2777 |
| [#2844](https://github.com/ll7/robot_sf_ll7/issues/2844) | OPEN/blocked | medium | Learned graph predictor | Needs #2836, #2843, #2865, #2866 unblocked |
| [#2845](https://github.com/ll7/robot_sf_ll7/issues/2845) | OPEN/blocked | low | Transformer/diffusion study | Needs lighter prerequisites |
| [#1490](https://github.com/ll7/robot_sf_ll7/issues/1490) | OPEN/blocked | low | Predictive planner v2 comparison | Do not repeat until revised gate evidence |

## Validation Notes

- Inspected: Issue #2929 body and comments; PR bodies/metadata for #2883-#2893;
  related Issue #2835, #2777, #2790, #2844, #2845, #1490 titles and labels.
- Inspected local routing surfaces: `docs/ai/prediction_lane.md`,
  `docs/context/prediction_lane_dependency_graph.json`,
  `docs/context/evidence/README.md`, `docs/context/catalog.yaml`,
  `docs/context/INDEX.md`, `docs/context/issue_2864_forecast_lane_synthesis.md`,
  `docs/context/issue_2902_live_forecast_replay_gate.md`,
  `docs/context/issue_2941_native_forecast_replay.md`.
- Verified evidence bundle paths exist in `docs/context/evidence/` for all closed issues.
- Path/link checks passed:
  - `uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml`
  - `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh`
- Skipped expensive experiment re-execution; this is docs-only synthesis work.
