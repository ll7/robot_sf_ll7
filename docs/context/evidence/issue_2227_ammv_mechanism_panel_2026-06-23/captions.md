# Issue #2227 AMMV Contrastive Mechanism Panel (2026-06-23)

Two same-seed, same-scenario direct `SocialForcePlanner` runs. The ONLY config
difference is `ammv_aware_enabled` (false=control, true=intervention) in
`configs/baselines/social_force_ammv_aware.yaml`; every other key is identical so the contrast isolates
the AMMV interaction term.

- Scenario: `issue_2227_ammv_close_front_static_ped`, seed `42`, steps `24`, dt `0.1`.
- Command: `uv run python scripts/analysis/build_ammv_mechanism_panel_issue_2227.py`
- Commit: `fedee58afe8c5834f4e6ccb1179cdc00a8354606`

## Where the mechanism was expected to act

A close static pedestrian sits directly in the robot's forward path. The AMMV
actuation-aware repulsion term is expected to activate there.

## Did it activate?

- AMMV-off max force magnitude: `0.000000` (expected 0).
- AMMV-on  max force magnitude: `2.641802` (nonzero).

## Did command / trajectory behavior change?

- Step-1 linear-velocity delta (on - off): `0.000000` m/s.
- Final-position distance between arms: `0.582849` m.
- Final lateral-offset delta (on - off): `0.170092` m.

## Outcome and claim boundary

Observed evidence: the AMMV term activates (nonzero force) and the selected
command and resulting robot trajectory diverge under an identical seed/scenario.

Claim boundary (diagnostic_only): this is a PLANNER-LEVEL mechanism difference.
It is NOT a navigation-success, benchmark-advantage, or sensor/perception-realism
claim. Pedestrians are static and simulator-owned dynamics are not modelled here.
