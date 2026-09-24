# 0.0.8 source audit against 0.0.7

This is a **provisional Stage 0 audit**, not evidence that 0.0.8 reproduces
0.0.7. It records code that could affect old episode values before the new
force metrics and Social Navigation Quality Index (SNQI) v2 branches merge.
Repeat the audit on the final clean release source commit, then require the
20,160-row predecessor equivalence gate before any release claim.

## Checked source and method

- Frozen 0.0.7 source: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`.
- Provisional 0.0.8 execution snapshot inspected:
  `40c18118cdbbc92ca6ab4bc498d80b394adf27df`.
- Compared `git diff --name-only <source> <head> -- robot_sf/benchmark
  robot_sf/nav robot_sf/sensor robot_sf/planner robot_sf/sim robot_sf/ped_npc`,
  then inspected diffs in the changed execution and metric paths.
- The selected campaign template contains no direct reference to the new
  force-residual predictor, scenario-belief adapter, or pedestrian tracker
  names. That search does not prove those modules cannot be called indirectly.

## Existing-value risks in the provisional diff

| Changed path | Observed change | Gate implication |
| --- | --- | --- |
| `robot_sf/benchmark/metric_layers.py` | `failure_to_progress_rate` now rejects nonbinary route-completion values and reports the decisive collision or timeout source. It also treats numeric conversion errors as unavailable. | Could change that old metric for malformed or nonbinary records. Compare all stored old metric fields and stop on any mismatch. |
| `robot_sf/benchmark/runner.py` and `analysis_trace.py` | Map-runner import is delayed; native analysis traces gain episode, seed, and execution identity. | Startup and trace bytes can change. The episode metric and outcome comparison remains mandatory. |
| `robot_sf/sensor/pedestrian_tracking.py` and `socnav_observation.py` | Tracker results gain reset and current-observation identity; observation fusion exposes the latest result as a side channel. | A planner that consumes tracking could change behavior. No direct campaign-config reference was found; compare outcomes and metrics on the actual roster. |
| `robot_sf/nav/map_config.py` | Plotting imports moved into the plotting method. | No numerical path changed in this diff, but execute the exact roster to verify. |
| `robot_sf/nav/predictive_types.py` | Timestamp documentation changed. | No executable change in this file. |
| `robot_sf/nav/force_residual_intent_predictor.py` and `robot_sf/planner/scenario_belief_adapter.py` | New predictor and adapter modules. | No direct reference in the selected campaign template; check the final roster and any indirect hook before launch. |
| `robot_sf/benchmark/diagnostic_report.py` | New diagnostic reporter. | Not an old episode metric calculation path in the inspected diff. |
| `robot_sf/benchmark/artifact_publication.py`, `release_protocol.py`, and `runtime_smoke_admission.py` | Release admission and bundle contracts changed. | May reject a candidate or alter publication metadata; full release acceptance and cold bundle preflight must pass. |

The scenario matrix, evaluation seeds, 14 planner keys, planner-config paths
and hashes, checkpoint-enforcement settings, horizon, timestep,
pedestrian-force settings, and `record_forces` setting were checked against
the frozen 0.0.7 resolved manifest in the #9668 source/config audit. The
0.0.8 campaign template adds
the SNQI-v2 declaration and delays bundle export until post-run gates have
finished. Checkpoint-byte receipts and the exact final source/config comparison
remain open until the upstream branches merge.

**Stop rule:** any mismatch in an old metric or outcome needs attribution to a
named commit and the author's decision, as required by issue #9668. New force
fields and SNQI-v2 fields cannot excuse an old-value mismatch.
