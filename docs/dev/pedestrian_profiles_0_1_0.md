# Opt-in pedestrian development profiles

`SimulationSettings(obstacle_force_profile="body_edge_v3")` selects main's
`body_edge_exponential_v3` finite-range wall form at factor 10 (3 m/s² at
zero body-edge clearance before the cutoff subtraction). It does not add
another wall kernel. An explicitly conflicting law is rejected.

Missing profile leaves the configured law, parameters, configuration serialization
and hash untouched, including the 0.1.0 physical-margin x10 default.
`legacy_v1` explicitly selects the legacy law and parameter boundary when no law
was supplied. `calibrated_v2` likewise selects its legacy kernel when the law is
missing. Both reject an explicitly conflicting law; a profile never replaces an
explicit law selector. Historical unversioned mappings retain their legacy boundary.
`calibrated_v2` and `gradient_v3` retain rejected historical diagnostic settings;
neither is an accepted fit or release configuration.

Profiles round-trip through settings, copying and scenario overrides. Episode
seeds for development are 1001–1200; the current behaviour receipt uses
1001–1030. No frozen 0.0.8 input, metric anchor or publication artifact changes.

Validation: `pytest -p tests.support.heldout_seed_guard
 tests/sim/test_obstacle_force_profile.py -q` (single command). Force/config
witnesses establish implementation integrity. The #10000 complete sweep,
independent refute review and model qualification remain separate requirements.
