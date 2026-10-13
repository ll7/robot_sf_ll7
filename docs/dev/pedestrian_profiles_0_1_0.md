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

After the bounded solve, an unresolved reachable contact component is held for
that step; unrelated walkers continue. This graph fallback is reported and the
CALFIT physics gate rejects its use as qualified evidence. Post-step diagnostic
velocity/displacement records describe the actual constrained motion. The
preview remains the force/cap proposal before contact projection. Oracle steps
whose capped velocities are changed by contact are excluded from exact force
inversion with `force_stage_uninstrumented`: the recorded force sum omits the
contact impulse. Uncorrected contact steps retain normal eligibility.

The new rule is versioned separately from the historical `projection_v1` draft.
Original draft commits/evidence remain recoverable. The synthetic cost probe
is not the required release-matrix overhead measurement. Complete #10000
receipts, independent final-head refute review, physical qualification and
external custody remain prerequisites for admission and calibration acquisition.

Fix-round warm CPU diagnostic (30 pedestrians, dev1001, 20 alternating
250-step blocks, admissible states restored outside timing): 129.50 µs/step
contact off and 148.04 µs/step on, **14.32% overhead**, versus 17.18% on the
reviewed source. Clearance rates are precomputed, already resolved constraints
avoid a second scan, and zero-valued counters avoid redundant writes. An earlier
intermediate repeat measured 15.002%; the budget margin is small and a complete
release-matrix timing qualification remains outstanding. This synthetic
diagnostic does not establish the <=15% budget for every production workload.

The 3,000-step dev1001–1030 randomized diagnostic retained zero swept
penetrations (minimum surface clearance 9.693e-9 m), no held-body displacement,
and **2,249 fallback steps**. Aggressive independent force proposals create
competing normal constraints; a capped sequential solve with only 3 or 10 passes
often leaves one unresolved. Its fail-safe then stops the entire reachable
component, including following neighbors, to preserve swept separation. This is
acceptable as an explicitly reported safety diagnostic, not calibrated crowd
progress or qualified fit evidence. The existing physics gate rejects any
fallback. Qualification requires a bounded solver improvement and progress
measurements on the validation workloads; neither increasing the pass budget
beyond 10 nor hiding fallback is permitted. Release integration owns the missing
behaviour receipt and complete workload/custody qualification.

[Raw diagnostic timing samples and source checksums](../context/evidence/pedestrian_contact_fix_20261013/diagnostics.json)
and the [SHA256 manifest](../context/evidence/pedestrian_contact_fix_20261013/SHA256SUMS)
preserve the bounded observations above for independent inspection.
