# Negative Result Register

Issue: [#2762](https://github.com/ll7/robot_sf_ll7/issues/2762)

Status: current.

## Purpose

This register tracks diagnostic-only, failed, inconclusive, or `revise`-classified findings so they
remain visible in future research planning. It is a synthesis and planning aid only, not new
benchmark or paper-facing evidence.

Entries do not promote prior diagnostic results into stronger claims. Fallback, degraded,
diagnostic-only, proxy-only, unavailable, stale, or missing-denominator rows weaken or block
claim wording.

## Entry Schema

Each entry records:

| Field | Description |
|---|---|
| `id` | Stable identifier (`issue-NNNN-short-slug`). |
| `hypothesis` | What was tested or expected. |
| `tested_artifact` | Candidate, planner, or mechanism under test. |
| `scenario` | Scenario, slice, or configuration context. |
| `comparator` | Baseline or control condition, if any. |
| `result_classification` | One of: `revise`, `diagnostic_only`, `failed`, `inconclusive`. |
| `failure_mode` | One of: `mechanism_failed`, `scenario_too_weak`, `evidence_diagnostic_only`, `infrastructure_only`, `stale`, `blocked`. |
| `why_failed_or_inconclusive` | Concise explanation of the negative/diagnostic outcome. |
| `evidence_pointer` | Paths to durable tracked evidence. |
| `recommended_next_action` | What should happen next (stop, revise, retest with stronger scenario, etc.). |
| `linked_issues` | Related issues, PRs, and context notes. |
| `claim_boundary` | Explicit scope restriction (not benchmark, not paper evidence, etc.). |
| `created_at` | Date the entry was added to this register. |

## Classification Definitions

- **`revise`**: The mechanism or hypothesis produced a result that requires design revision,
  not promotion. The finding is durable but does not close a claim boundary.
- **`diagnostic_only`**: The result is informative for infrastructure, plumbing, or scope
  clarification but does not constitute benchmark-strength or paper-facing evidence.
- **`failed`**: The mechanism did not activate, produced an error, or was blocked by a
  dependency.
- **`inconclusive`**: The test ran but the scenario, seed count, or horizon was insufficient
  to draw a conclusion.

## Failure Mode Definitions

- **`mechanism_failed`**: The mechanism itself did not work as intended under the tested
  conditions.
- **`scenario_too_weak`**: The scenario lacked the adversarial conditions, pedestrian density,
  or geometric challenge needed to stress the mechanism.
- **`evidence_diagnostic_only`**: The evidence is real but scoped to diagnostic/plumbing
  validation, not behavioral or benchmark claims.
- **`infrastructure_only`**: Only the infrastructure or tooling was validated, not the
  underlying hypothesis.
- **`stale`**: The finding references artifacts or configs that are no longer current.
- **`blocked`**: The finding is blocked by a dependency that has not been resolved.

## Register Entries

### NR-001: Topology Reselection Cross-Slice Diagnostic

| Field | Value |
|---|---|
| `id` | `issue-2716-topology-reselection-cross-slice` |
| `hypothesis` | Progress-gated topology reselection generalizes beyond the canonical h160 double-bottleneck slice and clears hard non-canonical slices. |
| `tested_artifact` | `topology_guided_hybrid_rule_v0_progress_gated_reselection` |
| `scenario` | Three non-canonical hard slices (`bottleneck_transfer`, `doorway_transfer`, `t_intersection_transfer`) and one negative-control slice (`simple_negative_control`), thresholds 0.05/0.1/0.2m. |
| `comparator` | `topology_guided_hybrid_rule_v0` (baseline), `topology_guided_hybrid_rule_v0_reuse_penalty`. |
| `result_classification` | `revise` |
| `failure_mode` | `mechanism_failed` |
| `why_failed_or_inconclusive` | All 9 hard progress-gated rows ended `horizon_exhausted` (159 deadlock steps, 0 collision). The mechanism activates and generalizes diagnostically but does not clear any hard slice at h160. Negative-control rows succeeded cleanly (3/3 success, 0 topology switches). |
| `evidence_pointer` | `docs/context/evidence/issue_2716_topology_reselection_cross_slice/summary.json`, `docs/context/evidence/issue_2716_topology_reselection_cross_slice/report.md` |
| `recommended_next_action` | Design a successor targeting actual clearance or terminal-outcome movement rather than another threshold chase. Stop same-family selector reruns on the canonical slice. |
| `linked_issues` | [#2716](https://github.com/ll7/robot_sf_ll7/issues/2716), [#2743](https://github.com/ll7/robot_sf_ll7/issues/2743) (merged PR), [#2742](https://github.com/ll7/robot_sf_ll7/issues/2742) (follow-up), [#2704](https://github.com/ll7/robot_sf_ll7/issues/2704) (progress-gated successor) |
| `claim_boundary` | Diagnostic-only, not benchmark or paper evidence. Classification is `revise`, not `promote`. |
| `created_at` | 2026-06-13 |

### NR-002: Observation-Noise Diagnostic on Distant Pedestrian

| Field | Value |
|---|---|
| `id` | `issue-2749-observation-noise-distant-pedestrian` |
| `hypothesis` | Observation noise (Gaussian position noise, missed detection, occlusion, delay) degrades planner behavior on a pedestrian-present scenario. |
| `tested_artifact` | `hybrid_rule_v0_minimal` with `perception_limited` observation config |
| `scenario` | `classic_bottleneck_medium` (stress_slice), seed 111, horizon 40, 1 pedestrian at ~14.9m closest approach. |
| `comparator` | Baseline (no observation noise, `ideal_state` evidence class). |
| `result_classification` | `diagnostic_only` |
| `failure_mode` | `scenario_too_weak` |
| `why_failed_or_inconclusive` | All progress, risk, and planner metrics are identical between baseline and perception-limited. The single pedestrian is too far from the robot to influence planner decisions; the planner is dominated by static obstacle clearance. Perturbation plumbing is confirmed working (occlusion masking active, 1 missed detection at step 15) but no behavioral degradation is measurable. |
| `evidence_pointer` | `docs/context/evidence/issue_2749_observation_noise_diagnostics/summary.json`, `docs/context/evidence/issue_2749_observation_noise_diagnostics/RESULT.md` |
| `recommended_next_action` | Retest with a closer pedestrian, higher pedestrian count, or a scenario where the planner must actively avoid pedestrians. Single-seed distant-pedestrian result is not sufficient for any robustness claim. |
| `linked_issues` | [#2749](https://github.com/ll7/robot_sf_ll7/issues/2749), [#2750](https://github.com/ll7/robot_sf_ll7/issues/2750) (merged PR) |
| `claim_boundary` | Diagnostic-only, not benchmark or paper evidence. Confirms perturbation infrastructure works; does not measure behavioral degradation. |
| `created_at` | 2026-06-13 |

### NR-003: Dissertation Evidence Ledger Rows (Synthesis Cross-Reference)

| Field | Value |
|---|---|
| `id` | `issue-2760-dissertation-evidence-ledger-diagnostic-rows` |
| `hypothesis` | The dissertation evidence ledger synthesizes all current thesis-area evidence into a claim-ready format. |
| `tested_artifact` | Ledger rows for topology_guidance, signalized_behavior, observation_robustness, prediction, pedestrian_density_stress, and exported_tables. |
| `scenario` | Full-repository evidence sweep across six thesis areas. |
| `comparator` | N/A (synthesis, not a paired experiment). |
| `result_classification` | `diagnostic_only` |
| `failure_mode` | `evidence_diagnostic_only` |
| `why_failed_or_inconclusive` | Five of seven ledger rows are diagnostic-tier or blocked evidence (topology, signalized behavior, prediction, pedestrian density, exported tables). One area (observation robustness) is release-backed but is a contract/provenance layer only. The exported-tables payload gap and PPO partial-failure blocker are resolved, but the Issue #3203 rerun remains invalid as benchmark-success evidence because Social Navigation Quality Index (SNQI) contract status failed. No row supports Results-chapter wording without qualification. |
| `evidence_pointer` | `docs/context/dissertation_evidence_ledger.md`, `docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json` |
| `recommended_next_action` | Use the ledger as a planning aid to identify which thesis areas need new experiments before Results wording is possible. Do not cite any current row as benchmark-strength evidence. |
| `linked_issues` | [#2760](https://github.com/ll7/robot_sf_ll7/issues/2760) |
| `claim_boundary` | Synthesis/planning aid only. The ledger does not produce new benchmark evidence, paper-facing results, or safety claims. |
| `created_at` | 2026-06-13 |

### NR-004: Predictive Hard-Case Maneuver Authority (Speed-Cap Only)

| Field | Value |
|---|---|
| `id` | `issue-3213-hardcase-authority-speedcap` |
| `hypothesis` | Richer predictive-planner maneuver authority (turn rate, action lattice, sequence-search depth, near-field geometry) closes the hard-case crossing-conflict success plateau. |
| `tested_artifact` | Predictive planner authority variants (`baseline`, `high_angular`, `dense_lattice`, `deep_sequence`, `nearfield_turn`) plus single-knob ablations (`nf_headings_only`, `nf_speedcap_only`, `nf_horizonboost_only`) across 5 prediction checkpoints. |
| `scenario` | `predictive_hardcase_portfolio_v1` (cross_trap low/medium/high + group_crossing_high), broad robust seeds 200-229 over both halves, ~120 episodes/cell. |
| `comparator` | Baseline camera-ready predictive authority config. |
| `result_classification` | `diagnostic_only` |
| `failure_mode` | `mechanism_failed` |
| `why_failed_or_inconclusive` | Only the near-field speed cap moved hard-success (~0.07 to ~0.10, about +0.03 absolute / +33% relative, consistent across checkpoints, ~120 episodes/cell). `nf_headings_only`, `nf_horizonboost_only`, `high_angular`, `dense_lattice`, `deep_sequence`, and `combined_max_authority` were inert (about baseline). The lift does not close the plateau (hard-success ~0.10) and checkpoint choice barely moved baseline, indicating the binding constraint is model/data-side rather than planner authority. |
| `evidence_pointer` | `docs/context/evidence/issue_3213_authority_sweep/robust_grid.json` |
| `recommended_next_action` | Keep `predictive_near_field_speed_cap` as a minor safety-progress tuning knob, not a success driver. Stop further planner-authority tuning as a plateau fix; prioritize model-side bets ([#3214](https://github.com/ll7/robot_sf_ll7/issues/3214) retraining, richer hard-case data) and proxy-vs-ADE selection ([#3204](https://github.com/ll7/robot_sf_ll7/issues/3204)). |
| `linked_issues` | [#3213](https://github.com/ll7/robot_sf_ll7/issues/3213), [#3215](https://github.com/ll7/robot_sf_ll7/issues/3215), [#3306](https://github.com/ll7/robot_sf_ll7/pull/3306) |
| `claim_boundary` | Diagnostic-only, not benchmark or paper evidence. Small attributable speed-cap effect; classification `diagnostic_only`, not promote. |
| `created_at` | 2026-06-20 |

### NR-005: Near-Field Turn Budget S20 Follow-Up

| Field | Value |
|---|---|
| `id` | `issue-3342-nearfield-turn-budget-s20` |
| `hypothesis` | The near-field turn-budget improvement observed in the Issue #3215 n=7 hard slice is a real small effect that survives a larger hard-seed sample. |
| `tested_artifact` | Predictive planner authority variants `baseline`, `nearfield_turn`, and `nf_speedcap_only` using `predictive_proxy_selected_v2_full`. |
| `scenario` | `predictive_hardcase_portfolio_v1`, S20 hard-seed manifest, clean and `robustness_smoke_v1` observation-noise slices. |
| `comparator` | Baseline predictive planner authority config on the same S20 seed schedule. |
| `result_classification` | `diagnostic_only` |
| `failure_mode` | `evidence_diagnostic_only` |
| `why_failed_or_inconclusive` | The local S20 diagnostic slice did not reproduce the near-field signal: baseline was 0.20 success / 0.80 collision in both clean and noisy slices, while `nearfield_turn` and `nf_speedcap_only` were 0.15 success / 0.85 collision. Intervals are wide and overlapping, so the result is a diagnostic negative rather than a benchmark-strength rejection. |
| `evidence_pointer` | `docs/context/evidence/issue_3342_nearfield_turn_budget_2026-06-21/README.md`, `docs/context/evidence/issue_3342_nearfield_turn_budget_2026-06-21/summary.json` |
| `recommended_next_action` | Do not adopt the near-field turn-budget signal from S20. Run the configured S30 slice only if the remaining uncertainty is worth the local compute; otherwise route forecast-lane effort to model/data-side blockers. |
| `linked_issues` | [#3342](https://github.com/ll7/robot_sf_ll7/issues/3342), [#3215](https://github.com/ll7/robot_sf_ll7/issues/3215), [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835) |
| `claim_boundary` | Diagnostic-local, not benchmark-strength, paper-facing, or release evidence. S30 is configured but not run. |
| `created_at` | 2026-06-21 |

## Research Planning Implications

This register is a planning aid. When scoping new experiments:

1. **Do not re-run failed experiments with the same parameters.** NR-001 and NR-002 both show
   that repeating the same scenario/seed/threshold combination will not produce new information.
2. **Do not promote diagnostic results to benchmark claims.** All three current entries are
   explicitly non-paper-facing. Any new claim must be backed by fresh, benchmark-strength evidence.
3. **Check this register before starting new work in the same area.** New topology, observation,
   or dissertation-synthesis experiments should reference these entries to avoid duplicating
   negative results.
4. **Update this register when new negative/diagnostic results are produced.** Add entries using
   the same schema and classification definitions above.

## Validation

```bash
uv run pytest tests/docs/test_negative_result_register.py -q
uv run ruff check docs/context/negative_result_register.md tests/docs/test_negative_result_register.py
git diff --check
```
