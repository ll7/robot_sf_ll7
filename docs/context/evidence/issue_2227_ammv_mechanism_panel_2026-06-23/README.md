# Issue #2227 AMMV Contrastive Mechanism Panel (2026-06-23)

Promoted, compact evidence bundle for the **AMMV (AMV actuation-aware) contrastive
mechanism panel** sub-target of issue #2227. This is the only #2227 sub-target built here;
the static-recentering and topology-guided-recovery panels remain as follow-up.

## What this shows

Two same-seed, same-scenario direct `SocialForcePlanner` runs differing ONLY in the
`ammv_aware_enabled` key of `configs/baselines/social_force_ammv_aware.yaml`:

- **control** = `ammv_aware_enabled: false`
- **intervention** = `ammv_aware_enabled: true`

Every other config key is identical, so the contrast isolates the AMMV interaction term.
The scenario is a close static pedestrian directly in the robot's forward path -- where the
AMMV actuation-aware repulsion term is expected to act.

## Observed evidence

| Arm | Max AMMV force | Mean AMMV force | Final robot (x, y) m |
| --- | --- | --- | --- |
| AMMV-off (control) | 0.000000 | 0.000000 | (3.781, 0.057) |
| AMMV-on (intervention) | 2.641802 | 0.534835 | (3.226, 0.235) |

- The AMMV term **activated** in the intervention arm (nonzero force) and stayed at exactly
  zero in the control arm, as expected when the term is disabled.
- The selected command and resulting robot **trajectory diverged**: final-position distance
  between arms `0.582849 m`, final lateral-offset delta `+0.170092 m`.

## Provenance

- Command: `uv run python scripts/analysis/build_ammv_mechanism_panel_issue_2227.py`
- Seed: `42`; scenario: `issue_2227_ammv_close_front_static_ped`; steps `24`; dt `0.1`.
- Commit (git HEAD at run): `fedee58afe8c5834f4e6ccb1179cdc00a8354606`
- claim_boundary: `diagnostic_only`
- evidence_tier: `stress`
- paper_grade: `false`

Traces come from actual `SocialForcePlanner` runs; no trajectory numbers were hand-written.
Both exported traces validate against `simulation_trace_export.v1`. Raw trace JSON is kept
under git-ignored `output/issue_2227_ammv_panel/traces/`; this bundle holds only the panel
PNG/PDF, captions, selection CSV, a compact `trace_summary.json`, and `summary.json`.

## Claim boundary

This is a **planner-level mechanism difference only**. It is NOT a navigation-success,
benchmark-advantage, or sensor/perception-realism claim. The `SocialForcePlanner` is a robot
planner; pedestrians here are static and simulator-owned pedestrian dynamics are not
modelled. The renderer emits one trajectory panel per arm (control + intervention) so the
AMMV-off vs AMMV-on contrast can be read side by side.

## Files

- `trajectory_panel_*_ammv_off_*.png` / `.pdf` -- control arm panel.
- `trajectory_panel_*_ammv_on_*.png` / `.pdf` -- intervention arm panel.
- `captions.md` -- AMMV activation, command/trajectory delta, claim boundary.
- `selection.csv` -- representative-episode selection table.
- `trace_summary.json` -- compact per-arm trace digest.
- `summary.json` -- full run summary (forces, deltas, artifact paths).

## Remaining #2227 sub-targets

- Static-recentering mechanism panel (out of scope here).
- Topology-guided recovery mechanism panel (out of scope here).
