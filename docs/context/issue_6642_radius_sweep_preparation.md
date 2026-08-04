# Issue #6642 — Collision-envelope radius sweep preparation (Gate 2 of #6600)

## Plain-language summary

This note records the **pre-submission admission** state for issue #6642, the Gate 2
(production sweep) child of the maintainer-approved radius-sensitivity campaign #6600.
The sweep tests whether planner rankings and scenario-family readings are stable
across robot collision-envelope radii **0.5 m, 0.8 m, and the 1.0 m release
baseline**, over the 48-cell `classic_interactions_francis2023` matrix, the complete
14-planner release roster, `paper_eval_s30` seeds 111-140, horizon 600, and
differential-drive kinematics, with all non-radius factors fixed and all arms pinned
to one immutable campaign commit.

The tracked configs are **not benchmark evidence**. The two hard preconditions are
now satisfied: the Gate 1 binding receipt proves consistent propagation to simulator
collision geometry, obstacle and pedestrian contact logic, feasibility/oracle
calculations, metric metadata and output rows, and planner inputs; and the merged
camera-ready loader binds each arm's declared radius at runtime. The sweep can launch
only after the remaining Gate 2 conditions are met:

1. Pin one immutable campaign commit across all arms and verify the fixed-factor
   configuration and expected row grid.
2. Verify queue/capacity, immutable inputs/checksums, and the private ledger.
3. Run the smallest valid production preflight and obtain explicit campaign admission.

Production compute is **not authorized** in this PR (`compute_submit` remains disabled
in the lease; the manifest records `production_submission_authorized: false`). No
SLURM job or production episode is submitted. Each arm now records
`runtime_binding_status: bound_runtime` with the same Gate 1 receipt and merged
source commit.

## Why production remains blocked

- The authoritative release robot radius is 1.0 m
  (`robot_sf/common/robot_defaults.py`), so the **1.0 m arm is the matched release
  comparator**. The frozen 0.0.3.post1 metric semantics must not change (#6600 stop rule).
- The merged runtime interface is [PR #6752](https://github.com/ll7/robot_sf_ll7/pull/6752),
  and the fresh Gate 1 receipt is recorded in [issue #6641](https://github.com/ll7/robot_sf_ll7/issues/6641).
  The radius divergence documented in `robot_defaults.py` (differential-drive 1.0 m
  vs nav-grid 0.3 m vs planner-fallback 0.4 m) is therefore covered by the
  within-simulator binding canary, but it is not a scientific radius-sensitivity result.
- The remaining blockers are campaign gates: pin one immutable launch commit across
  all arms, verify expected rows and available capacity/queue, record private ledger
  inputs/checksums, run the smallest valid preflight, and obtain explicit production
  admission. No config status change alone authorizes compute or evidence promotion.

## Deliverables

- `configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml` — declarative
  preparation manifest: radii, one arm campaign config per radius, fixed factors,
  one-commit policy, gate preconditions, and the fail-closed
  missingness/evidence-exclusion policy.
- `configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml`,
  `configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml`, and
  `configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml` — one campaign
  config per radius arm. All three are structurally identical (release baseline
  structure, pinned to the sweep's own issue-scoped release tag so all arms
  share one commit); only the `radius_sweep` treatment metadata and the release
  tag differ. The preflight dry-run in this PR runs against each arm config.
- `robot_sf/benchmark/radius_sweep_manifest.py` — manifest builder and checker that
  enforces the preparation boundary (3 radii; one tracked campaign config and
  issue-scoped release tag per arm; 14-key release roster in order; 48 cells;
  seeds 111-140; horizon 600; one commit; gate block; no degraded-as-evidence).
- `scripts/benchmark/build_radius_sweep_manifest_issue_6642.py` — CLI that resolves
  fixed factors and campaign identity from every arm config + scenario matrix,
  fails closed on any non-radius drift across arms or on a divergent
  `radius_sweep` treatment declaration, and writes the manifest + checker
  artifacts under `output/`.
- `tests/benchmark/test_radius_sweep_manifest_issue_6642.py` — focused tests covering
  the build, the per-arm identity and treatment contracts, and the fail-closed
  checker behavior (including isolated-tree drift negative controls).

## How to launch (post-Gate-1, by a separate authorized run)

1. Confirm the fresh Gate 1 canary #6641 receipt and its SHA-256 against every arm.
2. Confirm the merged runtime radius-binding surface and run the three arm preflights;
   each must expose the admitted binding without executing production episodes.
3. Pin one immutable campaign commit across all arms, verify the expected 60,480-row
   grid, queue/capacity, immutable inputs/checksums, and private ledger entry.
4. Obtain explicit campaign admission, then submit the three arms (0.5/0.8/1.0 m) on
   SLURM/remote only; preserve every declared
   row and classify unavailable/degraded/fallback/failed rows explicitly. Production
   output must carry complete row identities or an explicit fail-closed missingness ledger.

## Evidence tier

`not_benchmark_evidence` (pre-submission admission manifest only). The expected row grid is
3 radii x 14 planners x 48 cells x 30 seeds = 60480 rows, but no episodes are run here.
The narrow-doorway family cell (`francis2023_narrow_doorway`) is present in the resolved
48-cell roster for the geometry-sensitive comparison described in #6600.
