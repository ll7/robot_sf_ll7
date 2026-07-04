# Dissertation Evidence Ledger

Issue: [#2760](https://github.com/ll7/robot_sf_ll7/issues/2760)
Status: synthesis/dissertation-planning aid; not new benchmark, paper, or safety evidence.

## Purpose

This ledger is an integrated snapshot of current Robot SF evidence status for dissertation
planning. It combines the claim matrix, stale-artifact detector, and existing context notes
into a single reviewable surface covering the five thesis areas: topology guidance, signalized
behavior, observation robustness, prediction, and pedestrian-density stress.

This is **not** new benchmark evidence. It is a synthesis tool that classifies existing tracked
evidence by readiness for dissertation reuse. Every row must include allowed wording, caveat,
and claim gaps. Stale, diagnostic-only, fallback, degraded, or missing-payload rows are
explicitly blocked from manuscript promotion.

## Reading Guide

- **Allowed wording**: the maximum safe prose a row supports today.
- **Caveat**: the strongest limitation that must appear alongside any use.
- **Claim gap**: what evidence is still needed before the row can be promoted.
- **Status**: `current` (usable with caveats), `stale` (needs refresh before reuse), or
  `blocked` (must not be used).
- **Evidence tier**: `release-backed`, `diagnostic`, `proposal`, `non-claimable`.
- **Evidence promotion path**: the concrete next step required to promote a diagnostic-only row
  toward benchmark or paper evidence, or `None` when no credible promotion path exists.
  A non-null promotion path does **not** upgrade the row's current classification; the path must
  be completed and the evidence tier reclassified before any promotion.

## Ledger Rows

### 1. Topology Guidance

| Field | Value |
|---|---|
| **Claim** | Topology-guided hybrid rule selection can diversify route selection on the canonical h160 double-bottleneck slice, but repeated selector variants have not improved topology-command influence, route progress, or terminal outcome. |
| **Artifact status** | current |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "Near-parity topology selection can diversify route labels, but current evidence still routes the mechanism to revise without improving non-primary command influence, route progress, or terminal outcome." |
| **Caveat** | Do not claim topology guidance improves success, transfer, or leaderboard performance. The lane is `stop` for same-family selector reruns on the canonical slice. |
| **Source PR/issue** | [#2518](issue_2518_topology_near_parity_gate.md), [#2530](issue_2530_topology_near_parity_corrective_smoke.md), [#2624](issue_2624_topology_reuse_penalty_gate.md), [#2704](issue_2704_progress_gated_topology_successor.md), [#2706](issue_2706_topology_lane_synthesis.md) |
| **Dissertation chapter** | Discussion, Outlook |
| **Claim gap** | A new hypothesis with a different mechanism and metric is needed before further topology work. No current row supports Results wording. |
| **Evidence promotion path** | **live replay**: a fresh topology-guided planner campaign with non-same-family selector variants and an independent seed set is required before any benchmark promotion. |

### 2. Signalized Behavior

| Field | Value |
|---|---|
| **Claim** | The repository has trace-level proxy signal-state plumbing for the waiting-then-crossing fixture and a simulator-backed #2799 smoke proving signal metric denominator/exclusion semantics for explicit `planner_observable` rows. |
| **Artifact status** | current |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "The repository can now produce simulator-backed signalized-crossing rows that separate planner-observable denominator evidence from unavailable/proxy exclusions; this proves denominator plumbing, not traffic-light realism or crossing-legality compliance." |
| **Caveat** | Do not claim forced-waiting reasoning, legality compliance, traffic-signal realism, or benchmark ranking improvement from #2799. The smoke has one baseline planner and synthetic authored signal metadata. |
| **Source PR/issue** | [#2527](issue_2527_waiting_crossing_fixture.md), [#2564](issue_2564_signal_state_proxy_smoke.md), [#2662](issue_2662_signal_state_promotion_contract.md), [#2474](issue_2474_signalized_crossing_benchmark.md), [#2799 evidence](evidence/issue_2799_signalized_runtime/README.md) |
| **Dissertation chapter** | Methods, Limitations |
| **Claim gap** | Requires planner-comparison baselines plus validated zone/legality trace fields before forced-waiting, crossing-legality, traffic-signal-realism, or ranking claims. |
| **Evidence promotion path** | **live replay / benchmark smoke**: expand from the #2799 denominator smoke to comparison-ready signalized scenarios with legality trace fields and predeclared baselines. |

### 3. Observation Robustness

| Field | Value |
|---|---|
| **Claim** | The repository defines a five-level observation vocabulary (oracle full state through occluded partial state) with planner/level compatibility validation and episode propagation, but this is a contract and provenance layer, not calibrated perception. |
| **Artifact status** | current |
| **Evidence tier** | release-backed (for the vocabulary/contract layer only) |
| **Allowed wording** | "The benchmark observation-level vocabulary defines observation-contract boundaries and compatibility gates so results can state which observation assumption a planner used." |
| **Caveat** | The levels are benchmark evidence labels and compatibility gates, not sim-to-real validity claims. No real camera perception, detector training, calibrated tracking, or new environment observation implementation exists yet. |
| **Source PR/issue** | [#1246](issue_1246_observation_levels.md), [#1612](issue_1612_observation_track_architecture.md), [#1613](issue_1613_lidar_observation_track.md), [#1614](issue_1614_lidar_planner_compatibility.md) |
| **Dissertation chapter** | Methods |
| **Claim gap** | Requires actual perception pipeline, calibrated tracking, or simulator-level observation fidelity before any robustness claim. Cross-track comparison is not valid without matched observation contracts. |
| **Evidence promotion path** | None — the observation-level vocabulary is a contract/label layer without a credible standalone promotion path to robustness evidence. |

### 4. Prediction

#### Forecast-lane supported claim boundaries

| Field | Value |
|---|---|
| **Claim** | The forecast lane now has scaffolded support for typed `ForecastBatch.v1` artifacts, observation-tier separation via adapters, probabilistic forecast and calibration metrics, deterministic baseline comparison rows, dataset/split scaffolding, transferability tooling, conformal pilot support, and a closed-loop coupling gate recommendation. |
| **Artifact status** | current |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "The repository now supports forecast-lane infrastructure for typed artifact contracts, observation tiers, probabilistic metric pipelines, deterministic baselines, dataset-level provenance, transfer diagnostics, uncertainty pilots, and a closed-loop coupling gate." |
| **Caveat** | Current evidence is infrastructure and diagnostic in nature: it marks what is implemented and what the next coupling step reports, but it is not yet benchmark- or paper-grade claim support. |
| **Source PR/issue** | [#2761](https://github.com/ll7/robot_sf_ll7/issues/2761), [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835), [#2836](https://github.com/ll7/robot_sf_ll7/issues/2836), [#2838](https://github.com/ll7/robot_sf_ll7/issues/2838), [#2839](https://github.com/ll7/robot_sf_ll7/issues/2839), [#2840](https://github.com/ll7/robot_sf_ll7/issues/2840), [#2841](https://github.com/ll7/robot_sf_ll7/issues/2841), [#2842](https://github.com/ll7/robot_sf_ll7/issues/2842), [#2843](https://github.com/ll7/robot_sf_ll7/issues/2843), [#2846](https://github.com/ll7/robot_sf_ll7/issues/2846), [#2847](https://github.com/ll7/robot_sf_ll7/issues/2847); local routing: [forecast schema](issue_2836_forecast_batch_schema.md), [prediction lane](../ai/prediction_lane.md), [dependency graph](prediction_lane_dependency_graph.json) |
| **Dissertation chapter** | Methods, Outlook |
| **Claim gap** | Requires a transfer-aware, same-seed closed-loop campaign that compares forecast-enabled planners to deterministic and baseline modes using calibrated, fail-closed, and same-denominator metrics. |
| **Evidence promotion path** | **benchmark promotion**: requires executed transfer-aware closed-loop comparison rows with calibrated ADE/FDE/miss-rate and collision/progress metrics. |

#### Forecast-lane unsupported claim boundaries

| Field | Value |
|---|---|
| **Claim** | No durable evidence yet shows that forecast integration improves local-navigation safety, route progress, or transfer robustness; current results classify these outcomes as unsupported or mixed. |
| **Artifact status** | current |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "Forecast artifacts and tooling are in place, but forecast-driven gains in safety, progress, or transfer performance are not established." |
| **Caveat** | Forecast coupling and transfer diagnostics currently show mixed or insufficient evidence (#2843 revise recommendation; mixed baseline-point results in #2781; transferability matrix is not a safety proof). |
| **Source PR/issue** | [#2761](https://github.com/ll7/robot_sf_ll7/issues/2761), [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835), [#2781](https://github.com/ll7/robot_sf_ll7/issues/2781), [#2843](https://github.com/ll7/robot_sf_ll7/issues/2843), [#2847](https://github.com/ll7/robot_sf_ll7/issues/2847); local routing: [dependency graph](prediction_lane_dependency_graph.json) |
| **Dissertation chapter** | Discussion, Outlook |
| **Claim gap** | Requires transfer-aware transferability campaigns with false-positive accounting, non-regression on success/progress, and statistical uncertainty before any safety/progress claim. |
| **Evidence promotion path** | **transfer-aware closed-loop promotion**: run transfer-focused closed-loop planner campaigns with matched deterministic baselines and explicit safety/progress deltas before promoting this claim. |

### 5. Pedestrian-Density Stress

| Field | Value |
|---|---|
| **Claim** | The scenario coverage entropy tool provides config-only diagnostic coverage analysis over authored scenario fields (archetype, density, flow, pedestrian count, etc.), but it does not prove benchmark value or runtime stress effectiveness. |
| **Artifact status** | current |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "A config-only scenario coverage entropy report can identify redundant and novel scenario candidates based on authored metadata features for scenario-set curation." |
| **Caveat** | Coverage entropy is a diagnostic planning aid. It does not prove benchmark value, runtime stress effectiveness, or planner ranking. Dense stress rows should not be promoted without runtime and failure-semantics proof. |
| **Source PR/issue** | [#1240](issue_1240_scenario_coverage_entropy.md), [#1304](issue_1304_pedestrian_config_boundary.md) |
| **Dissertation chapter** | Methods, Limitations |
| **Claim gap** | Requires runtime execution evidence, failure-semantics classification, and planner-comparison rows before any pedestrian-density stress benchmark claim. |
| **Evidence promotion path** | None — config-only entropy analysis lacks a credible standalone promotion path to runtime stress evidence; runtime execution and failure-semantics proof are required. |

### 6. Exported Tables (Dissertation Bundle)

| Field | Value |
|---|---|
| **Claim** | Two scenario-horizon tables were re-exported into a dissertation bundle with payload files, manifest entries, and checksums from a fresh bounded Issue #3203 campaign, but the campaign is invalid as benchmark-success evidence. |
| **Artifact status** | current (payload-complete) |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "Payload-complete scenario-horizon dissertation table exports exist for discussion/provenance only; the fresh row-complete campaign still does not establish benchmark-success, ranking, or Results-chapter evidence." |
| **Caveat** | Issue #3203 2026-07-01 campaign completed 9/9 planner rows, 1296 total episodes, `evidence_status=valid`, PPO `native` execution, and PPO learned-policy contract `pass`; however, Social Navigation Quality Index (SNQI) contract status remains `fail` with rank-alignment Spearman `-0.19999999999999998`, below the `0.3` fail threshold. Treat the packet as diagnostic/provenance only, not benchmark-success evidence. |
| **Source PR/issue** | [#1023](https://github.com/ll7/robot_sf_ll7/issues/1023), [#2542](issue_2542_dissertation_export_bundle.md), [#3203](https://github.com/ll7/robot_sf_ll7/issues/3203) |
| **Dissertation chapter** | Discussion, Limitations |
| **Claim gap** | Payload absence and PPO observation-contract blockers are resolved, but the tables remain diagnostic/provenance artifacts until the SNQI contract caveat is repaired or explicitly scoped out by a new issue contract. |
| **Evidence promotion path** | **SNQI contract repair or explicit narrow-boundary rerun**: repair the SNQI rank-alignment blocker, or predeclare a narrower claim boundary that excludes SNQI validity, then rerun a fresh bounded campaign before any benchmark Results wording. |

### 7. Recovered Reward-Curriculum Seed Runs (Diagnostic)

| Field | Value |
|---|---|
| **Claim** | Three reward-curriculum expert-policy seed runs (seeds 506/508/509) completed training and 70-scenario evaluation, lost their manifests to a cross-worktree serializer bug, and were recovered by backfilling manifests from retained artefacts; they are diagnostic seed evidence, not benchmark success. |
| **Artifact status** | current (manifests backfilled, evidence retrieved local) |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "Recovered reward-curriculum expert-policy seed runs exist with payload-complete, provenance-tracked manifests for seed-variance bookkeeping and pipeline diagnostics only; their marginal SNQI and elevated collision rate mean they do not establish benchmark-success or Results-chapter evidence." |
| **Caveat** | Aggregate success 0.81–0.85 but SNQI is at/below zero (−0.069 / +0.017 / −0.111) and collision is 14–19%, failing the issue_2557 success criterion ("improve success without increased collision"). Each manifest is single-seed (`validation_state: draft`); cross-seed variance is not yet established. A 501–511 variance fill is in flight (jobs 13153/13154/13155) under an explicit override of the campaign do-not-rerun policy. |
| **Source PR/issue** | [#2557](https://github.com/ll7/robot_sf_ll7/issues/2557), [#2919](https://github.com/ll7/robot_sf_ll7/issues/2919), [#3203](https://github.com/ll7/robot_sf_ll7/issues/3203), [#3266](https://github.com/ll7/robot_sf_ll7/issues/3266), [recovered-seeds note](issue_2557_recovered_diagnostic_seeds.md), [#3590](https://github.com/ll7/robot_sf_ll7/pull/3590) |
| **Dissertation chapter** | Methods, Limitations |
| **Claim gap** | Manifest/evidence loss is resolved, but the runs remain diagnostic until SNQI validity and collision-rate regressions are repaired under a new hypothesis and a consolidated multi-seed variance analysis supersedes the recovered-seeds stub. |
| **Evidence promotion path** | **new hypothesis + consolidated variance**: complete the 501–511 seed fill, retrieve/analyse, then author a consolidated seed-variance note; do not promote past diagnostic until the SNQI and collision caveats are repaired or explicitly scoped by a new issue contract. |

## Stale-Artifact Summary

| Artifact | State | Reason |
|---|---|---|
| `tab_issue_1023_campaign_table` | historical-valid | Payload checksum matches; dissertation bundle schema is historical/non-current for the stale-artifact detector. |
| `tab_issue_1023_scenario_family_breakdown` | historical-valid | Payload checksum matches; dissertation bundle schema is historical/non-current for the stale-artifact detector. |

## Dissertation Reuse Recommendations

1. **Topology guidance**: Methods/Outlook only. Use the stop-lane synthesis from #2706 to
   explain why same-family selector reruns were exhausted. Do not cite as benchmark improvement.
2. **Signalized behavior**: Methods/Limitations only. Use the proxy diagnostic surface to
   describe infrastructure readiness. Do not cite as traffic-signal compliance or planner ranking.
3. **Observation robustness**: Methods only. Use the observation-level vocabulary to describe
   benchmark contract discipline. Do not cite as perception or sim-to-real validity.
4. **Prediction supported boundary**: Methods, Outlook only. Use the forecast-lane scaffold
   evidence (artifacts, adapters, metrics, baselines, transferability tooling, conformal pilots,
   and coupling gates) to describe infrastructure readiness. Do not cite as safety/progress results.
5. **Prediction unsupported boundary**: Discussion/Outlook only. Do not claim forecast-driven
   safety, progress, or transfer gains until transfer-aware closed-loop planner evidence is
   published with explicit false-positive and regression accounting.
6. **Pedestrian-density stress**: Methods/Limitations only. Use coverage entropy to describe
   scenario curation discipline. Do not cite as runtime stress or planner-ranking evidence.
7. **Exported tables**: Discussion/Limitations only. Use as payload-complete table provenance
   with the Issue #3203 invalid-campaign caveat; do not cite as benchmark-success or ranking
   evidence.
8. **Recovered reward-curriculum seeds**: Methods/Limitations only. Use as provenance-tracked
   seed-variance bookkeeping and a manifest-recovery case study; do not cite as benchmark-success
   given marginal SNQI and elevated collision rate.

## Claim Boundaries

- This ledger is a **synthesis/dissertation-planning aid**. It does not produce new benchmark
  evidence, paper-facing results, or safety claims.
- Every row includes allowed wording and caveat. Rows with missing or stale evidence use
  `do-not-use` or similarly blocked wording.
- Diagnostic-only, stale, fallback, degraded, unavailable, or launch-packet rows are explicitly
  treated as claim-weakening/blocking caveats.
- Fallback behavior is not acceptable as a successful benchmark outcome unless the task
  explicitly measures fallback mode.
- A non-null `evidence_promotion_path` does not upgrade a diagnostic-only row to benchmark or paper
  evidence. Promotion requires completing the path and reclassifying the evidence tier.

## Validation

```bash
# Claim matrix regeneration
uv run python scripts/tools/benchmark_publication_bundle.py claim-matrix \
  --bundle-dir docs/context/evidence/issue_2542_dissertation_export_bundle \
  --json-output output/issue-2760/claim_matrix.json \
  --markdown-output output/issue-2760/claim_matrix.md

# Stale-artifact detector
uv run python scripts/tools/stale_artifact_detector.py \
  docs/context/evidence/issue_2542_dissertation_export_bundle/artifact_manifest.json \
  --json-out output/issue-2760/stale_artifact_report.json

# Issue #3203 latest fresh bounded campaign command behind the diagnostic packet
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3203_scenario_horizons_h500_reexport_valid.yaml \
  --output-root output/benchmarks/issue_3203 \
  --campaign-id issue3203_scenario_horizons_h500_reexport_valid_2026-07-01 \
  --mode run \
  --skip-publication-bundle \
  --log-level INFO

# JSON schema validation (targeted)
uv run pytest tests/docs/test_dissertation_evidence_ledger.py -q

# Docs proof consistency
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

## Related Surfaces

- Claim matrix: `output/issue-2760/claim_matrix.json`
- Stale artifact report: `output/issue-2760/stale_artifact_report.json`
- Claim export candidate report: [dissertation_claim_export_candidate_report.md](dissertation_claim_export_candidate_report.md)
- Research bridge: [dissertation_research_bridge.md](dissertation_research_bridge.md)
- Artifact manifest: [issue_2542_dissertation_export_bundle/artifact_manifest.json](evidence/issue_2542_dissertation_export_bundle/artifact_manifest.json)
- Machine-readable ledger: [evidence/issue_2760_dissertation_evidence_ledger/ledger.json](evidence/issue_2760_dissertation_evidence_ledger/ledger.json)
