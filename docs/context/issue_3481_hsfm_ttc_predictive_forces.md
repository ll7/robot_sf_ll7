# Issue #3481: Headed Social Force Model + Time-To-Collision Predictive Pedestrian Forces

This note records the Phase 2 prototype contract for issue
[#3481](https://github.com/ll7/robot_sf_ll7/issues/3481): an opt-in Headed Social Force Model
(HSFM) selector with Time-To-Collision (TTC) predictive pedestrian-pedestrian repulsion. It is
diagnostic evidence only, not a calibrated realism claim.

## Selector Keys

- `social_force_default`: existing PySocialForce stepping path.
- `hsfm_total_force_v1`: Phase 1 opt-in total-force HSFM stepping path.
- `hsfm_ttc_predictive_v1`: Phase 2 opt-in total-force path plus pedestrian-pedestrian
  time-to-collision predictive repulsion.

The default pedestrian model is unchanged.

## TTC Force Contract

Pure helpers in `robot_sf/sim/pedestrian_model_variants.py` compute pairwise time collision from
relative position, relative velocity, and summed pedestrian radii:

```text
||p_ij + v_ij t|| <= r_i + r_j
```

Pairs with no positive root inside `horizon_s` return `inf`. Active pairs add a bounded repulsion
term weight:

```text
force_scale * exp(-ttc / tau0_s)
```

The per-actor sum is capped by `max_force`.

## Parameters

`SimulationSettings.ttc_predictive_force` stores the opt-in surface:

- `tau0_s > 0`
- `horizon_s > 0`
- `force_scale >= 0`
- `max_force > 0`
- `include_ped_ped=True`
- `include_robot_proxy=False`

Robot-proxy TTC coupling intentionally fails closed when enabled because this slice does not add a
stable robot-state injection point for pedestrian stepping.

Tracked prototype metadata lives in
`configs/research/hsfm_ttc_predictive_forces_issue_3481.yaml`, which now records both the TTC
predictive selector parameters and the opt-in anisotropic FoV selector defaults used in this issue
family.

## Evidence Boundary

Focused simulator tests cover finite TTC values, inactive separating pairs, monotonic force growth
for shorter collision times, force capping, selector validation, and deterministic one-step smoke
for `hsfm_ttc_predictive_v1`.

This slice did not run a full benchmark campaign, submit Slurm/GPU work, perform external
calibration, or edit paper/dissertation claims.

## Pairwise Isolation and Weight-Path Vectorization (issue #3481 follow-up slice)

The Phase 1/2 slices (PR #4144, #4202, #4258) left two maintainer-flagged blockers before
benchmark-scale use of the opt-in HSFM field-of-view (FoV) models. This slice closes both at the
pure-math layer in `robot_sf/sim/pedestrian_model_variants.py`; it remains diagnostic/prototype and
changes no default model.

### Blocker 1 — pairwise isolation of pedestrian-pedestrian repulsion attenuation

`anisotropic_fov_total_force(...)` is the coarse *aggregate* mode: it collapses the per-pair FoV
weight matrix to one `np.min(weights, axis=1)` factor per actor and scales that actor's already
summed force. It cannot distinguish a rear neighbor from an in-cone neighbor, so a pedestrian with
one neighbor ahead and one behind has its whole force scaled by `rear_weight`.

New pure helper `pairwise_fov_attenuated_forces(pairwise_forces, positions, headings, ...)` performs
genuine pairwise isolation. Given per-pair contributions `pairwise_forces[i, j]` (the force neighbor
`j` exerts on actor `i`) it returns:

```text
attenuated[i] = sum_j anisotropic_fov_weights[i, j] * pairwise_forces[i, j]
```

Each neighbor's contribution is attenuated by its own FoV weight before summing, so rear attenuation
no longer bleeds across unrelated in-cone interactions. The layer is simulator-object-independent and
unit-testable without a full environment.

### Blocker 2 — vectorized O(N^2) TTC weight path

`pairwise_time_to_collision(...)` previously used a Python double loop. It is now solved with NumPy
broadcasting over the quadratic collision condition `||p_ij + v_ij t|| <= r_i + r_j`, with masks that
reproduce the earlier scalar branches exactly (overlap → `0.0`; non-closing/miss → `inf`; first root
in `(epsilon, horizon_s]` → TTC). `anisotropic_fov_weights(...)` was already vectorized upstream
(PR #4285), so both O(N^2) FoV/TTC weight paths named on the issue are now loop-free.

### Evidence boundary (this slice)

`tests/sim/test_hsfm_fov_pairwise_isolation.py` covers:

- narrow-passage fixture where pairwise isolation keeps the in-cone push and attenuates only the rear
  push, and differs from the aggregate `np.min` result;
- `attenuated = sum_j weight * pairwise_force` weighted-sum definition on a seeded random cloud;
- full-cone identity and fail-closed validation (bad shape, non-finite, empty population);
- vectorized-vs-scalar TTC equivalence on a seeded random cloud and a deterministic bottleneck
  fixture (two converging columns) with finite, in-horizon, self/separating-`inf` structure.

The pairwise-isolated mode is available as a pure diagnostic helper; wiring per-pair
pedestrian-pedestrian force contributions from the simulator seam (so the runtime step can consume
isolation instead of the aggregate `np.min` path) remains follow-up work, as does narrow-passage
lateral-sliding / bottleneck freeze benchmark evidence and any evidence-tier upgrade. No full
benchmark campaign, Slurm/GPU submission, external calibration, or paper/dissertation claim edit was
performed in this slice.

## Runtime wiring of per-pair pedestrian-pedestrian forces (2026-07-04)

This slice closes the "wire per-pair pp-force contributions from the simulator seam" follow-up named
above and in the maintainer's PR #4297 gate comment. The `hsfm_anisotropic_fov_v1` runtime path now
consumes per-pair pedestrian-pedestrian forces instead of the aggregate `np.min` attenuation.

New capability: **runtime consumption of per-pair pedestrian-pedestrian forces at the simulator
seam.**

### What changed

- New pure helper `pairwise_social_force_contributions(positions, velocities, *, activation_threshold,
  n, n_prime, lambda_importance, gamma, factor)` returns the `(N, N, 2)` per-pair social-force matrix
  where `[i, j]` is the repulsion neighbor `j` exerts on actor `i`. It reuses PySocialForce's own
  pairwise kernel (`social_force_ped_ped`) with the same activation threshold and `factor`, so
  `contributions.sum(axis=1)` reproduces the aggregate `SocialForce()` the physics engine already
  sums into the total force (verified in a unit test against `pysocialforce.forces.social_force`).
- New pure helper `fov_attenuated_total_force(total_forces, pairwise_social_forces, positions,
  headings, *, cone_half_angle_rad, rear_weight)` isolates the aggregate ped-ped social term from the
  full total force and replaces it with the per-pair FoV-attenuated form:

  ```text
  result = total_forces - pairwise_social.sum(axis=1) + pairwise_fov_attenuated_forces(...)
  ```

- The simulator seam (`Simulator._step_pedestrians`, inherited by `PedSimulator`) for
  `hsfm_anisotropic_fov_v1` now: reads the live `SocialForce` component's parameters
  (`_social_force_component`, fail-closed if absent), builds the per-pair matrix from the current
  PySocialForce state (`_pairwise_social_force_contributions`), and calls
  `fov_attenuated_total_force` before `step_hsfm_total_force`.

### Behavior boundary

- Only the `hsfm_anisotropic_fov_v1` runtime path changed. `social_force_default`,
  `hsfm_total_force_v1`, and `hsfm_ttc_predictive_v1` are untouched.
- The previous aggregate path scaled the *entire* per-actor force (goal + social + obstacle + robot
  coupling) by a single `np.min` FoV weight. The new path attenuates *only* the pedestrian-pedestrian
  social contributions, per pair, so a rear neighbor is down-weighted without disturbing an in-cone
  neighbor or the actor's goal/obstacle drive. This is a behavior change for the opt-in FoV model
  only; the runtime FoV smoke test was updated to pin the new per-pair composition.
- The aggregate helper `anisotropic_fov_total_force` is retained (still unit-tested) as the reference
  contrast for the isolation tests.
- Total-force reconstruction subtracts and re-adds the social term, so non-social forces are
  preserved to floating-point tolerance. Evidence tier stays diagnostic/prototype.

### Known limitations / follow-up

- `pairwise_social_force_contributions` is now vectorized (see the section below); the last named
  runtime O(N^2) loop under this issue is closed.
- Narrow-passage lateral-sliding and bottleneck freeze/deadlock benchmark evidence, seed-controlled
  campaigns, and any evidence-tier upgrade remain out of scope. No full benchmark campaign, Slurm/GPU
  submission, external calibration, or paper/dissertation claim edit was performed in this slice.

### Validation

```bash
uv run pytest tests/sim/test_hsfm_fov_pairwise_isolation.py \
  tests/sim/test_hsfm_total_force_model.py \
  tests/sim/test_ttc_predictive_pedestrian_model.py -q
uv run ruff check robot_sf/sim/pedestrian_model_variants.py robot_sf/sim/simulator.py tests/sim
```

Plus an end-to-end smoke: a real `make_robot_env` run with `pedestrian_model=hsfm_anisotropic_fov_v1`
stepped 15 times with finite pedestrian positions, confirming the real force list exposes a
`SocialForce` component for the seam.

## Vectorized pairwise social-force contribution matrix (issue #3481, CPU successor slice)

The runtime FoV seam wired by PR #4352 built its per-pair pedestrian-pedestrian social matrix with
an `O(N^2)` Python double loop that called the PySocialForce njit kernel `social_force_ped_ped` one
pair at a time. That loop was the last named CPU blocker before benchmark-scale pedestrian counts.

- `pairwise_social_force_contributions` now evaluates the whole `(N, N, 2)` matrix in closed NumPy
  form via the pure helper `_pairwise_social_force_kernel`, a direct vectorization of the reference
  kernel's expression (normalize the position diff, build/normalize the interaction vector, angle
  `theta`, `B = gamma * interaction_length + 1e-8`, velocity + angle force terms). Zero-vector
  handling matches `pysocialforce.forces.norm_vec` (zero difference → zero unit direction,
  `arctan2(0, 0) == 0`), so coincident actors and the diagonal give a finite zero force, not NaN.
- The activation-threshold masking and diagonal exclusion are reproduced with a boolean mask so
  out-of-range pairs and self-interactions stay exactly zero, matching the loop's `continue` branches.
- The deferred `numba` import is no longer needed on this path; the module stays numpy-pure at import.

Behavior is preserved to floating-point tolerance (only fastmath-vs-IEEE rounding separates the two
paths):

- per-pair vs. scalar njit loop: max abs error ~5e-15 across N ∈ {2, 5, 20, 50};
- aggregate `contributions.sum(axis=1)` vs. `social_force(...) * factor`: max abs error ~7e-15
  (well under the pre-existing `rtol=1e-9` contract).

Measured speedup (single CPU, diagnostic timing only — not a benchmark campaign):
`N=50` ~20x, `N=100` ~23x, `N=200` ~14x versus the per-pair loop.

Tests (added to `tests/sim/test_hsfm_fov_pairwise_isolation.py`):
`test_vectorized_social_contributions_match_scalar_kernel` (parametrized N, per-pair equivalence),
`test_vectorized_social_contributions_handle_coincident_pairs` (degenerate zero-diff finiteness),
`test_vectorized_social_contributions_scale_to_large_population` (N=256 scale + aggregate parity).

Evidence tier stays diagnostic/prototype: no default-model change, no calibrated-realism, planner
ranking, benchmark-strength, or paper/dissertation claim. Remaining under the issue: narrow-passage
lateral-sliding and bottleneck freeze/deadlock benchmark evidence, seed-controlled campaigns, and any
evidence-tier upgrade.

## CPU-only shared-throat precursor diagnostics (cheap successor slice)

This slice adds a small local harness in
`robot_sf/benchmark/pedestrian_model_fixture_diagnostics.py` so the remaining
narrow-passage / bottleneck work has an in-repo, CPU-only *precursor* surface before any campaign
work. These are synthetic shared-throat interactions, not the final benchmark-grade geometric
fixtures.

- Scenarios:
  - `shared_throat_sliding`: symmetric lane-preserving crossing through a shared throat; reports
    lateral-displacement proxies for passive sliding.
  - `shared_throat_congestion`: symmetric opposing flows through a shared throat; reports
    interaction-zone slowdown proxies for freeze/deadlock-like congestion.
- Supported outputs:
  - `minimum_pairwise_distance_m`
  - `mean_max_lateral_displacement_m`
  - `mean_speed_mps`
  - `entered_interaction_zone`
  - `max_pedestrians_in_interaction_zone`
  - `interaction_zone_slow_steps`
  - `max_consecutive_interaction_zone_slow_steps`
  - `interaction_zone_slow_detected`
  - finite position / velocity checks
- Boundaries:
  - no benchmark pass/fail threshold,
  - no Slurm/GPU work,
  - no seed-sweep or evidence-tier upgrade,
  - no claim that these synthetic shared-throat fixtures replace the remaining true
    narrow-passage / bottleneck benchmark evidence,
  - no claim that any opt-in model is now realistic or benchmark-superior.

The harness writes compact JSON / Markdown artifacts via
`write_pedestrian_model_fixture_report(...)` and is covered by
`tests/benchmark/test_pedestrian_model_fixture_diagnostics.py`.
