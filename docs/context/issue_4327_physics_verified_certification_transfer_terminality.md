# Issue #4327 Physics-Verified Certification-Transfer Terminality

Issue [#4327](https://github.com/ll7/robot_sf_ll7/issues/4327) asked for the empirical
follow-through after the synthetic interacting certification-transfer packet: run the proposed
interacting scenario family through the real CPU probe, record nonzero near-field interaction, and
keep the result diagnostic rather than paper-facing.

## Terminal Status

Issue #4327 is terminal: PR [#4337](https://github.com/ll7/robot_sf_ll7/pull/4337) merged the
requested real-probe slice on 2026-07-03.

The merged evidence packet is
[`docs/context/evidence/issue_4207_interacting_physics_2026-07/`](evidence/issue_4207_interacting_physics_2026-07/).
The packet records a CPU 4-arm probe using the #4315 interacting scenario matrix with
`configs/benchmarks/issue_4207_interacting_physics_probe.yaml`, a 400-step horizon, and seeds
`111`, `112`, and `113`.

## Acceptance Evidence

- The `goal` baseline has a real interacting cell:
  `robot_ped_within_5m_frac = 0.10558655435990806` and
  `min_clearance_m = -0.024278621748445417`.
- `summary.json` records `model_sensitivity_exercised: true`.
- The packet's claim boundary states that the result is diagnostic only, uses provisional gate
  thresholds, and does not promote a deployment, real-world safety, planner-superiority, paper, or
  dissertation claim.
- The issue's only live comment after #4337 merge says #4337 delivered the real CPU 4-arm packet and
  that remaining trained-planner comparison work stays on #4207, not #4327.

## No Distinct Follow-Up In This Issue

No separate implementation follow-up remains under #4327. The learned arms in #4337 ran in fallback
mode because no checkpoints were attached; that caveat is already recorded in the merged PR and is a
#4207 follow-up, not a reason to keep #4327 open.

This note is a terminality record only. It does not add a new certification-transfer run, Hybrid
Social Force Model change, benchmark campaign, Slurm submission, or claim promotion.
