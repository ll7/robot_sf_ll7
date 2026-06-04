# Issue #2236 Trace Mechanism Evidence Rubric

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2236>
Date: 2026-06-04
Status: current guidance for using trace-level artifacts as local-planner mechanism evidence.

## Goal

Define when trace exports, trace viewers, trajectory panels, perturbation trace slices, and
guard/actuation diagnostics can support a local-planner mechanism explanation. This note complements
[issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md): that note
names mechanism labels and confidence classes; this note defines the trace-artifact evidence needed
before those labels can be used.

## Evidence Levels

| Level | Use when | Required fields | Allowed claim |
| --- | --- | --- | --- |
| `qualitative_illustration` | A trace export, viewer, screenshot, panel, or annotation proves a scene can be inspected. | Source trace path or fixture, generation command, commit/checksum, frame/time units, evidence boundary, and validation that the artifact renders or loads. | "This artifact visualizes the episode/scene." Do not claim a mechanism. |
| `mechanism_hypothesis` | One trace or compact diagnostic shows a plausible route, phase, guard, actuation, or proximity pattern, but no controlled pairing or repeat support. | `qualitative_illustration` fields plus event/frame ids, scenario/seed/planner, relevant metrics, and the hypothesized mechanism with caveats. | "This is a plausible mechanism to test." |
| `mechanism_supported_case` | A paired baseline/intervention trace, direct probe, or root-cause repair shows the mechanism in a named scenario/planner/seed case. | Baseline/intervention identity, matching scenario/planner/seed contract, closest-approach or progress metrics, command-source or activation/veto timeline when relevant, and row status/fallback accounting. | "This mechanism is supported for this bounded case." |
| `cross_case_mechanism_evidence` | Multiple completed pairs, seeds, scenarios, or planners show the same mechanism under a stable schema and consistent boundary. | `mechanism_supported_case` fields plus repeated support, completed/excluded row counts, aggregation method, and explicit negative or neutral rows. | "This mechanism is supported across this slice." Still not paper-facing causality unless the benchmark contract says so. |

Fallback, degraded, unavailable, missing, failed, or partial rows cannot be mechanism evidence unless
the claim is specifically about that execution mode and the row is labeled as such.

## Required Fields By Mechanism Family

Use the smallest field set that can distinguish the claimed mechanism from nearby alternatives:

| Mechanism family | Minimum trace fields |
| --- | --- |
| Route or topology | Route hypothesis ids, route/static clearance, dynamic clearance, route-progress reference, selected command source, and blocked/unavailable hypothesis counts. |
| Dynamic phase or ordering | Baseline/intervention pairing, pedestrian id or matching rule, closest-approach time, clearance/center-distance delta, progress delta, and repeated seed support for stronger claims. |
| Proxemic or clearance tradeoff | Clearance/exposure metrics, terminal outcome, horizon or intervention boundary, and whether higher success/speed costs more exposure. |
| Guard, shield, fallback, or handoff | Raw model action if present, adapted action, post-guard action, guard decision counts, veto/intervention/fallback events, observation contract, and row status. |
| Actuation or command saturation | Requested command, projected command, command-space metadata, clip/saturation fractions, profile provenance, and calibrated-vs-synthetic claim boundary. |
| Learned residual contribution | Baseline local command, learned residual, clipped residual, post-guard command, residual checkpoint/dataset pointer, and fallback/degraded status. |

When one of those fields is missing, downgrade the evidence level instead of filling the gap with
aggregate interpretation.

## Existing Example Classification

| Example | Evidence level | Why | Limit |
| --- | --- | --- | --- |
| Real trace viewer smoke in [issue_2038_real_trace_viewer_smoke.md](issue_2038_real_trace_viewer_smoke.md). | `qualitative_illustration` | A generated `simulation_trace_export.v1` trace rendered nonblank in the Three.js viewer and preserved a screenshot with checksum. | Proves tooling visibility only; no planner mechanism or metric correctness. |
| Trace export and annotation schemas in [issue_1689_simulation_trace_export_schema.md](issue_1689_simulation_trace_export_schema.md) and [../debug_visualization.md](../debug_visualization.md). | `qualitative_illustration` | The schema records frame units, robot/pedestrian states, planner action/event blocks, and qualitative annotations with strict analysis-only boundaries. | Valid schema and panels are not mechanism evidence unless tied to a case-level claim. |
| H500 fixed-vs-long trace pilot in [issue_1049_h500_mechanism_pilot.md](issue_1049_h500_mechanism_pilot.md). | `mechanism_supported_case` | The pilot preserves paired h100/h500 per-step traces and compact summaries for clean time-budget relief, exposure-enabled completion, and long-horizon safety regression. | Seed 111 representative cases only; no guard/veto telemetry and no wait-then-go proof. |
| Topology probe on `classic_realworld_double_bottleneck_high` seed 111 in [issue_1692_topology_hypothesis_probe.md](issue_1692_topology_hypothesis_probe.md). | `mechanism_hypothesis` | The diagnostic reports multiple route hypotheses, selected command sources, static/dynamic clearance, and fail-closed unavailable counts. | One scenario/seed slice; mostly `dynamic_window` command source, so it supports investigation rather than proving topology-aware behavior. |
| Corridor pedestrian-route offset trace in [issue_1939_corridor_trace_response.md](issue_1939_corridor_trace_response.md). | `mechanism_supported_case` | It reruns paired no-op and `pedestrian_route_offset` variants for 12 completed planner/seed pairs and records closest-approach clearance, progress, and time deltas. | The large progress effect is seed-local; terminal collisions are not solved. |
| `francis2023_intersection_wait` timing vs speed trace in [issue_1947_intersection_wait_timing_speed_trace.md](issue_1947_intersection_wait_timing_speed_trace.md). | `cross_case_mechanism_evidence` | Two perturbation families produce opposite signed clearance effects across three planners and three seeds each, with completed paired traces. | One scenario family; terminal outcomes stay unchanged, so this supports phase sensitivity rather than robustness. |
| `francis2023_intersection_wait` `speed_h1_p050` grid trace in [issue_1953_intersection_wait_speed_grid_trace.md](issue_1953_intersection_wait_speed_grid_trace.md). | `cross_case_mechanism_evidence` | All 9 targeted no-op-versus-perturbed pairs complete, the same nearest pedestrian index is retained, and the signed clearance shift repeats across planners and seeds. | Pedestrian identity is index-based rather than stable-id based; this is still a 3-seed local diagnostic, not broad robustness evidence. |
| AMMV Social Force pair diagnostic in [issue_2168_ammv_social_force_pair_diagnostic.md](issue_2168_ammv_social_force_pair_diagnostic.md). | `mechanism_supported_case` | The direct mechanism probe activates the AMMV term and records force magnitude, intrusion count, lateral-offset delta, and clearance delta. | Benchmark adapter rows lack AMMV metadata and remain identical, so this is direct-path mechanism evidence only. |
| Guarded-PPO zero-motion repair in [issue_2006_guarded_ppo_zero_motion_repair.md](issue_2006_guarded_ppo_zero_motion_repair.md). | `mechanism_supported_case` | Root cause is tied to observation-mode and padded-pedestrian handoff assumptions; repaired smoke changes guard diagnostics from repeated `goal_reached` to `ppo_clear`. | Single-episode smoke and handoff repair, not nominal/stress evidence. |
| ORCA-residual runtime smoke in [issue_1428_orca_residual_lineage.md](issue_1428_orca_residual_lineage.md). | `mechanism_hypothesis` | The runtime surface and required diagnostic contract are staged, including residual contribution/clipping and guard-veto fields. | No trained residual contribution is measured yet; learned-policy mechanism remains unproven. |

## Future PR Guidance

Before using a trace panel, viewer screenshot, or compact trace slice as mechanism evidence, future
PRs should state:

- evidence level from this note;
- mechanism label or hypothesis, preferably from
  [issue_2220_failure_mechanism_taxonomy.md](issue_2220_failure_mechanism_taxonomy.md);
- scenario, planner, seed, horizon, and intervention/baseline identity;
- artifact path, checksum or tracked fixture, generation command, and commit;
- frame ids or event ids that anchor the observation;
- metrics used to support the mechanism and any missing fields;
- row status counts, including fallback/degraded/unavailable exclusions;
- whether the claim is diagnostic-only, benchmark-strength, or paper-facing.

Trace panels should not be used as mechanism evidence when they only show an appealing trajectory,
omit the baseline/intervention pairing, hide command-source or guard/fallback events needed for the
claim, rely on ignored `output/` artifacts without a durable summary, or select a single seed while
making cross-case language.

## Update Recommendation

Do not update PR templates or trace-report tooling in this PR. The immediate need is a guidance
surface that future PR authors can cite. A later tooling issue is justified only if repeated trace
PRs need the same required-field checklist embedded into `render_trace_report.py`,
`render_trajectory_panels.py`, or the PR template.

`docs/context/artifact_evidence_vocabulary.md` does not need a new artifact class here: trace
exports, panels, screenshots, and compact JSON/Markdown summaries are already artifacts. This note
instead defines how those artifacts may be interpreted.

## Validation

This is docs-only synthesis over tracked context notes and evidence artifacts. Validate with:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
