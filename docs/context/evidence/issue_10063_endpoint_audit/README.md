<!-- AI-GENERATED (#10063, 2026-10-01) - NEEDS-REVIEW -->
# Endpoint audit for issue #10063

Diagnostic development geometry, not a release, evaluation or safety claim.
`overlaps.json` contains all 51 scenarios and all 102 full robot endpoint rectangles:
31 intersections before, 27 after. Four unintended intersections are repaired in
three release-only successor maps. Every remaining intersection has an exact
fingerprint and rationale in `configs/scenarios/release_0_0_8_endpoint_dispositions.yaml`.

Resolved scenario pedestrian radius: 0.4 m; substrate radius: 0.35 m. Distance
is from the closed robot-centre sampling rectangle to the nominal pedestrian
centreline or crowd-centre spawn polygon. Distance <=0.4 m includes contact and
round endcaps. Dynamic role motion, later collisions and robot-footprint clearance
are separate questions; repair gaps are 1.5 m (larger than 1.0+0.4 m).

Test-value gate: the full-rectangle sampling fix was tested but no old test
compared its complete support to a scripted pedestrian lane. Tests load real
versioned YAML/SVG bytes through the scenario loader and prove that the original
map is rejected before any simulation. Radius-only crossings, trajectory overrides,
stationary pedestrians, missing maps, changed fingerprints and CLI enforcement are
covered. The two geometry witnesses fail on exact base d3370652cdd90efbbbe3dc484bcc869a2a533e96;
24 importing tests plus the new witnesses pass. The scripted trajectory probe uses
no environment or planner steps.

Current release-template matrix pin is refreshed. The retained PEDFIX historical
input pin in `docs/validation/pedfix_0_0_8/manifest.json` describes its measured
checkouts; `git show bc85705d4f89c5f91a499c77362e30a8f68bdd8f:configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`
reproduces 8b16cb1ba6155567f31855ec453d596e17e4ed859caf60b6e1092f49b4766ac8.
Replacing that past measurement hash with new geometry would falsify provenance.
Regenerate materialized release identities after integrating this branch.

Development Slurm sweeps: before exact base d3370652cdd90efbbbe3dc484bcc869a2a533e96;
after geometry ce4d0f04162f5b1675b864ae71fbdfc3f8828279. Each covers the three changed
scenarios with goal, risk_dwa and social_force on explicit seeds 1001–1030. Results
and source/config/environment identities will be registered when complete. The
first attempt (15950) failed before stepping because a later rehearsal horizon
file was absent on the pinned base; corrected sweeps retain base-authored budgets.
