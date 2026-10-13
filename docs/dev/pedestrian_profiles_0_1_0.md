# Opt-in pedestrian development profiles

`SimulationSettings(obstacle_force_profile="body_edge_v3")` selects main's
`body_edge_exponential_v3` finite-range wall form at factor 10 (3 m/s² at
zero body-edge clearance before the cutoff subtraction). It does not add
another wall kernel. An explicitly conflicting law is rejected.

Missing profile leaves the legacy law, parameters, configuration serialization
and hash untouched. `legacy_v1` explicitly selects that parameter boundary.
`calibrated_v2` and `gradient_v3` retain rejected historical diagnostic settings;
neither is an accepted fit or release configuration.

Profiles round-trip through settings, copying and scenario overrides. Episode
seeds for development are 1001–1200; the current behaviour receipt uses
1001–1030. No frozen 0.0.8 input, metric anchor or publication artifact changes.

Validation: `pytest -p tests.support.heldout_seed_guard
 tests/sim/test_obstacle_force_profile.py -q` (single command). Force/config
witnesses establish implementation integrity. The #10000 complete sweep,
independent refute review and model qualification remain separate requirements.

The opt-in `pedestrian_radius_m` overrides placement, collision/metric and force
geometry together; missing preserves physical 0.40 / force 0.35 m. The source
measurement runner is `scripts/validation/pedestrian_validation_10074.py` with
`configs/benchmarks/pedestrian_validation_0_1_0_v1.json`. It preserves named
source estimators, censoring and raw trajectory checksums. V7/V8/V9 remain
explicit diagnostics or contextual cases where the source target is unresolved;
a numerical suite score does not turn them into empirical validation.

V6 uses turning above the greater of the five-baseline mean maximum and
0.05 rad/s, sustained for 0.3 s before PoMD in the existing 3 m window.
The floor and duration are an explicit engineering correction, not a Huber
published threshold. Baseline maxima remain diagnostics. Tiny yaw returns
missing onset; a window-boundary result cannot establish an onset outside the
captured window. Synthetic manoeuvre-location controls prevent zero-baseline
Gaussian tails from saturating the estimator.

CALFIT start admission refuses non-finite state, invalid walls, overlapping discs
and walls before integration. V3/V4 holding layouts preserve population size and
explicitly report any engineering density reduction needed for rigid bodies.
`calfit_preflight_10074.py` checks estimator controls without simulator episodes;
its ideal records are structural tests, never accepted model evidence.
`calfit_search_10074.py` preserves separate desired-speed draws and velocity caps.
Acquisition still requires qualified contact/wall physics and a new explicitly
pinned five-day plan; historical rejected grids do not qualify the model.

Select contact with `pedestrian_physics: {contact_rule: projection_v2, passes: 3}`
with an optional shared radius. Without a radius override, contact uses physical
0.40 m discs while the force radius remains 0.35 m. HSFM variants use separate
integrators and are rejected with contact at production construction.
It projects capped normal velocities before
one semi-implicit Euler displacement, with 3–10 correction passes, no friction,
no substeps and no position push-out. The robot is excluded. Tangential movement
is retained. Initial body/wall overlap is refused; it is never hidden by repair.
Controller holds and source-prescribed actors remain fixed to their commanded
motion. Infeasible prescribed motion refuses the step before position changes.

Contact normals sharing a movable actor are resolved jointly. Corrections are
normal impulses, including unilateral static-wall reactions; there is no
component-wide velocity reset. A cached active basis accelerates the same normal
solve and is discarded if its Gram algebra fails. The configured 3–10 budget
bounds normal/cap reconciliation; basis exchanges are linear algebra within a
projection, not physics substeps. A small wall-free cell retains the inexpensive
bounded pair solve.

If normal/cap reconciliation cannot produce admissible motion, the emergency
path projects individual walkers into their incident normal halfspaces and
speed balls. It does not halt a whole reachable component. Its activation is
reported. Infeasible prescribed motion is refused. The unchanged CALFIT physics
gate rejects emergency fallback as qualified evidence. Runtime diagnostics now
report both fallback count and rate.

Post-step velocity/displacement records describe actual constrained motion.
The preview remains the force/cap proposal. Contact-corrected oracle steps remain
ineligible for exact inversion (`force_stage_uninstrumented`); uncorrected steps
retain normal eligibility.

The new rule is versioned separately from the historical `projection_v1` draft.
Original draft commits/evidence remain recoverable. The synthetic cost probe
is not the required release-matrix overhead measurement. Complete #10000
receipts, independent final-head refute review, physical qualification and
external custody remain prerequisites for admission and calibration acquisition.

The round-1 diagnostic at source `3297dc41` measured 14.32% warm-cell overhead
and 2,249 fallback steps in its 3,000-step randomized test. Those observations
and source checksums remain in the immutable
[diagnostic archive](https://github.com/ll7/robot_sf_ll7/blob/8695768397134a36cadd94702f31f020a0f5c772/docs/context/evidence/pedestrian_contact_fix_20261013/diagnostics.json)
and [SHA256 manifest](https://github.com/ll7/robot_sf_ll7/blob/8695768397134a36cadd94702f31f020a0f5c772/docs/context/evidence/pedestrian_contact_fix_20261013/SHA256SUMS).
They describe the superseded component-halting implementation, not this repair.

Round-2 repair diagnostics at source `cecd05824` restore V3 progress to 58/60
crossings and V4 to 350/350 on dev1001 after 160 s, at both 3 and 10 passes.
Contact-off controls cross 60/60 and 350/350. Every contact workload has zero
emergency fallback, cap violations and unresolved constraints. The analytic
swept centre-distance minima are 0.5600000100 m (V3) and 0.560000009999 m (V4)
against 0.56 m physical diameters; endpoint wall penetration is zero.

**The 15% CPU budget is not met.** Paired timing uses the same admissible input
state, goal and desired-speed vector for both arms, alternates arm order each
step, excludes only the first ten warmup steps, and preserves all 1,590 samples
per arm in twenty blocks. V3 overhead is 49.56% / 48.87% (3 / 10 passes); V4 is
176.06% / 173.60%. The separate 30-person warm cell measures 122.87 / 141.32
microseconds off/on, or 15.02%, with zero fallback. Neither a cheaper method that
stalls V3 nor a safe method exceeding budget qualifies the fit.

Reproduce the workload diagnostic with
`uv run python scripts/validation/contact_progress_10104.py --seed 1001 --output <run-directory>`.
It preserves trajectories, input states, raw timing samples, source identity and
SHA256 manifests, and exits 3 if any target fails. It is synthetic implementation
evidence, not empirical or release admission. Calibration remains blocked on a
solver meeting the cost budget, independent review and release integration's
receipt/custody requirements. No calibration job was submitted. Frozen 0.0.8
inputs, global baselines and production defaults remain untouched.
