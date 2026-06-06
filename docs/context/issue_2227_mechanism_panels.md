# Issue #2227 Mechanism Panel Readiness

Date: 2026-06-04

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2227>

Status: partially advanced diagnostic lane. Issue #2428 published the first durable AMMV/default
Social Force trace-panel bundle, while the originally requested static-recentering and
topology-guided panels remain blocked on durable mechanism-specific `simulation_trace_export.v1`
trace pairs.

## Goal

Issue #2227 asks whether trace-level visual evidence can explain why a mechanism intervention
succeeds where a baseline fails. The intended panels are diagnostic-only and should compare
baseline/intervention traces for:

- static recentering, using the one-factor component evidence from issues #2170, #2180, and #2182;
- topology-guided recovery, using the topology-hypothesis diagnostics from issues #1674 and #1692.

The desired outputs are one success case and one failure case, baseline/intervention overlays,
key-frame annotations, command-source timelines, captions, selection criteria, and provenance.

## Input Audit

The panel renderer exists:

- `scripts/tools/render_trajectory_panels.py`
- `robot_sf/benchmark/trajectory_panels.py`
- `docs/debug_visualization.md`

The renderer consumes `simulation_trace_export.v1` JSON traces. The current tracked repository
contains only generic analysis-workbench trace fixtures, not durable mechanism-specific traces for
static recentering or topology-guided recovery:

- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/planner_sanity_open_episode_0000.json`

The current mechanism evidence is aggregate or compact-summary evidence:

- `docs/context/evidence/issue_2180_one_factor_h500_2026-06-03/summary.json`
- `docs/context/evidence/issue_2182_component_synthesis_2026-06-03/component_effects.csv`
- `docs/context/evidence/issue_1692_topology_hypothesis_probe_2026-05-30/summary.json`

Older representative traces under
`docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/traces/` are useful historical
ORCA h100/h500 diagnostics, but they are not `simulation_trace_export.v1` traces and they do not
represent the requested static-recentering or topology-guided mechanism pairs.

## Decision

Do not generate decorative or generic panels for #2227 from unrelated fixtures. Without durable
mechanism-specific trace pairs, a panel bundle would risk implying visual evidence that the
repository does not actually preserve.

The correct next step is a targeted trace-generation pass, not a full benchmark matrix:

1. Select one static-recentering baseline/intervention pair from the issue #2170/#2180 one-factor
   manifest where the intervention changes the outcome or mechanism score.
2. Select one topology-guided recovery slice from the issue #1692 diagnostic scenario where route
   hypotheses are available and command-source switching is visible.
3. Generate or export `simulation_trace_export.v1` traces for the selected baseline/intervention
   episodes.
4. Run `scripts/tools/render_trajectory_panels.py` with an explicit selection CSV and promote only
   compact PNG/PDF/caption/manifest artifacts into `docs/context/evidence/`.

Until that targeted regeneration exists, #2227 remains diagnostic-only readiness work, not figure
evidence and not paper-facing mechanism proof.

## 2026-06-06 Partial AMMV Panel Proof

Issue #2428 converted the Issue #2405 AMMV step-export unblocker into a first durable rendered panel
bundle:

- context note: `docs/context/issue_2428_mechanism_trace_panels.md`;
- evidence bundle: `docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/`;
- scope: one selected `classic_head_on_corridor_low` seed `111` row per planner for
  `default_social_force` and `ammv_social_force`;
- result: both promoted traces are loader-valid `simulation_trace_export.v1` files with 20 frames,
  and the renderer produced PNG/PDF trajectory panels with checksums.

This narrows Issue #2227's blocker from "no durable rendered mechanism panel exists" to "AMMV/default has
a diagnostic rendered example; static recentering and topology-guided recovery still need selected
trace pairs." The Issue #2428 bundle remains diagnostic-only and should not be cited as benchmark or
paper-facing evidence.

## Evidence Manifest

The compact audit manifest is tracked at:

- `docs/context/evidence/issue_2227_mechanism_panels_2026-06-04/README.md`
- `docs/context/evidence/issue_2227_mechanism_panels_2026-06-04/artifact_gap_manifest.json`

## Reproducible Command Shape

Discovery commands used for the audit:

```bash
rtk rg -l '"schema_version"\s*:\s*"simulation_trace_export.v1"' docs tests output 2>/dev/null
rtk rg -n "static_recenter|static recenter|recenter|topology_guided|topology-guided|route_guide|selected_source|selected local command|selected_local_command_source" docs/context docs/context/evidence tests/fixtures output 2>/dev/null
rtk find docs/context/evidence tests/fixtures output \( -name "*.jsonl" -o -name "*trace*" \)
```

Future targeted generation should use the existing trace and panel tooling:

```bash
rtk uv run python scripts/tools/build_simulation_trace_export.py \
  --source <episode-recording.jsonl> \
  --output output/issue_2227_mechanism_panels/traces/<case>.json \
  --planner-id <planner-or-candidate-id> \
  --scenario-id <scenario-id>

rtk uv run python scripts/tools/render_trajectory_panels.py \
  --selection-csv output/issue_2227_mechanism_panels/selection.csv \
  --output-dir output/issue_2227_mechanism_panels/panels \
  --command "uv run python scripts/tools/render_trajectory_panels.py ..." \
  --commit "$(git rev-parse --short HEAD)"
```

## Validation

This note was validated with the repository documentation consistency checker and path checks in
the #2227 branch. No benchmark or heavy trace regeneration was run for this audit.
