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
