# Issue #6642 — Collision-envelope radius sweep preparation (Gate 2 of #6600)

## Plain-language summary

This note records the **preparation-only** deliverable for issue #6642, the Gate 2
(production sweep) child of the maintainer-approved radius-sensitivity campaign #6600.
The sweep tests whether planner rankings and scenario-family readings are stable
across robot collision-envelope radii **0.5 m, 0.8 m, and the 1.0 m release
baseline**, over the 48-cell `classic_interactions_francis2023` matrix, the complete
14-planner release roster, `paper_eval_s30` seeds 111-140, horizon 600, and
differential-drive kinematics, with all non-radius factors fixed and all arms pinned
to one immutable campaign commit.

This PR is **not benchmark evidence**. It prepares the sweep so it can launch the
moment two hard preconditions are met:

1. The Gate 1 binding-canary child #6641 reports a **passing verdict** proving the
   declared radius propagates consistently to simulator collision geometry, obstacle
   and pedestrian contact logic, feasibility/oracle calculations, metric metadata and
   output rows, and planner inputs that consume the radius.
2. A runtime **radius-binding surface** exists in the camera-ready campaign config so
   each arm's declared radius is actually bound (not silently ignored) at runtime.

Production compute is **not authorized** in this PR (`compute_submit` disabled in the
lease; the manifest records `production_submission_authorized: false`). No SLURM job
is submitted, no episodes are run, and the manifest's
`runtime_binding_status` is `pending_gate1_canary` on every arm.

## Why preparation-only now

- The authoritative release robot radius is 1.0 m
  (`robot_sf/common/robot_defaults.py`), so the **1.0 m arm is the matched release
  comparator**. The frozen 0.0.3.post1 metric semantics must not change (#6600 stop rule).
- A config-level radius-binding surface does **not** exist yet on `origin/main`; it is
  the Gate 1 canary (#6641) deliverable. The radius divergence documented in
  `robot_defaults.py` (differential-drive 1.0 m vs nav-grid 0.3 m vs planner-fallback
  0.4 m) is exactly the binding risk the canary must reconcile before any radius arm
  can be trusted.
- Introducing a half-wired radius field that the loader silently ignores would create
  the exact "silently ignored binding" failure mode the campaign is designed to catch.
  This PR therefore declares the radius treatment as **manifest metadata only** and
  keeps every arm's binding status pending.

## Deliverables

- `configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml` — declarative
  preparation manifest: radii, fixed factors, one-commit policy, gate preconditions,
  and the fail-closed missingness/evidence-exclusion policy.
- `configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml` — the 1.0 m arm campaign
  config (release baseline structure, pinned to the sweep's own issue-scoped release
  tag so all arms share one commit). The preflight dry-run in this PR runs here.
- `robot_sf/benchmark/radius_sweep_manifest.py` — manifest builder and checker that
  enforces the preparation boundary (3 radii; 14-key release roster in order; 48 cells;
  seeds 111-140; horizon 600; one commit; gate block; no degraded-as-evidence).
- `scripts/benchmark/build_radius_sweep_manifest_issue_6642.py` — CLI that resolves
  fixed factors from the arm config + scenario matrix and writes the manifest +
  checker artifacts under `output/`.
- `tests/benchmark/test_radius_sweep_manifest_issue_6642.py` — focused tests covering
  the build and the fail-closed checker contract.

## How to launch (post-Gate-1, by a separate authorized run)

1. Confirm the Gate 1 canary #6641 reports a passing verdict on every binding surface.
2. Add the runtime radius-binding surface (per #6641) so each arm's `radius_m` is
   actually bound to robot collision geometry, contact logic, oracle, metrics, and
   planner inputs.
3. Pin one immutable campaign commit and set the arm `radius_sweep.runtime_binding_status`
   from `pending_gate1_canary` to the canary's binding verdict.
4. Submit the three arms (0.5/0.8/1.0 m) on SLURM/remote only; preserve every declared
   row and classify unavailable/degraded/fallback/failed rows explicitly. Production
   output must carry complete row identities or an explicit fail-closed missingness ledger.

## Evidence tier

`not_benchmark_evidence` (preparation manifest only). The expected row grid is
3 radii x 14 planners x 48 cells x 30 seeds = 60480 rows, but no episodes are run here.
The narrow-doorway family cell (`francis2023_narrow_doorway`) is present in the resolved
48-cell roster for the geometry-sensitive comparison described in #6600.
