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

| Field | Value |
|---|---|
| **Claim** | The probabilistic prediction interface is merged and the contract smoke emits native/fail-closed rows for reactive, single-trajectory, multimodal-equal-weight, and multimodal-confidence-weighted rows, but no planner-campaign comparison has been executed. |
| **Artifact status** | current |
| **Evidence tier** | diagnostic |
| **Allowed wording** | "The repository has a merged probabilistic prediction interface and a contract-smoke runner that materializes native and fail-closed row shapes for multimodal prediction configurations." |
| **Caveat** | This is contract evidence only. The runner uses deterministic fixtures, does not execute a planner campaign, and does not measure prediction quality or compare planning performance. |
| **Source PR/issue** | [#2475](issue_2475_probabilistic_prediction_interface.md), [#2476](issue_2476_multimodal_prediction_benchmark.md), [#2496](issue_2496_multimodal_prediction_smoke.md) |
| **Dissertation chapter** | Methods, Outlook |
| **Claim gap** | Requires executed planner campaign with reactive, single-trajectory, and multimodal rows; matched metrics (success, collision, min-ped-distance, time-to-goal); and fail-closed row handling before any benchmark claim. |
| **Evidence promotion path** | **denominator repair**: an executed planner campaign with matched metrics and fail-closed row handling is required before any benchmark claim. |

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
| **Claim** | Two historical scenario-horizon tables were exported into a dissertation bundle, but the payload files are missing and the tables remain historical tracked evidence. |
| **Artifact status** | stale (missing payload) |
| **Evidence tier** | non-claimable |
| **Allowed wording** | `do-not-use` |
| **Caveat** | The claim matrix reports missing payload files for both artifacts, and the stale-artifact detector classifies the bundle manifest as stale. The tables remain historical tracked evidence and do not establish new benchmark claims. |
| **Source PR/issue** | [#1023](https://github.com/ll7/robot_sf_ll7/issues/1023), [#2542](issue_2542_dissertation_export_bundle.md) |
| **Dissertation chapter** | N/A (blocked) |
| **Claim gap** | Requires payload file recovery or re-export before any reuse. |
| **Evidence promotion path** | **stale artifact refresh**: payload file recovery or re-export from a fresh campaign is required before any reuse. |

## Stale-Artifact Summary

| Artifact | State | Reason |
|---|---|---|
| `tab_issue_1023_campaign_table` | non-claimable | Missing payload file |
| `tab_issue_1023_scenario_family_breakdown` | non-claimable | Missing payload file |
| Export bundle unnamed artifact | stale-needs-refresh | Output artifact missing output path |

## Dissertation Reuse Recommendations

1. **Topology guidance**: Methods/Outlook only. Use the stop-lane synthesis from #2706 to
   explain why same-family selector reruns were exhausted. Do not cite as benchmark improvement.
2. **Signalized behavior**: Methods/Limitations only. Use the proxy diagnostic surface to
   describe infrastructure readiness. Do not cite as traffic-signal compliance or planner ranking.
3. **Observation robustness**: Methods only. Use the observation-level vocabulary to describe
   benchmark contract discipline. Do not cite as perception or sim-to-real validity.
4. **Prediction**: Methods/Outlook only. Use the merged interface and contract smoke to
   describe infrastructure readiness. Do not cite as prediction-quality or planner-improvement evidence.
5. **Pedestrian-density stress**: Methods/Limitations only. Use coverage entropy to describe
   scenario curation discipline. Do not cite as runtime stress or planner-ranking evidence.
6. **Exported tables**: Blocked. Do not use until payload files are recovered or re-exported.

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
