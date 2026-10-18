# Real flattened observation fixture

`head_on_1001_reset.npz` contains the first observation delivered to the
social-force policy in a real map-runner episode, not a hand-built schema.
Captured at `d41cceb7f9422337bd34b602ae49f3dddbddb2a5`, scenario
`classic_head_on_corridor_medium`, development seed 1001, H600, dt 0.1,
`classic_interactions_francis2023_release_0_0_8_v1.yaml`, and the release-template
social-force config. The archive contains numeric arrays only; load it with
`allow_pickle=False`. The grid, pose, actors, radii, and flattened timestep are
the actual environment bytes.

Reproduce with `probe_issue_10007_fxb.py --algo social_force --snapshot-only
--out <fresh-directory>` at the base. The generated
`last_reset_observation.npz` is the archive captured before the first command.
Tests that replace dt with 0.2 are representation/decoding counterfactuals on
these bytes. The controlled SA-CADRL model isolates action decoding; it does
not establish checkpoint behavior or episode performance.

## Fixed-clock development episode calibration (FXB2)

`dev_seed_calibration.json` records fixed-code runs of the v2 config on dev
seeds 1001–1003, with source/config/lockfile hashes and command digests. The
runner uses H500, dt 0.1 and `tracked_agents_no_noise`; the doorway scenario
overrides the horizon to 400 steps. This is diagnostic regression calibration,
with no held-out or release-performance inference.

Doorway total spin commands were 88/67/94; longest consecutive runs were
22/19/17; large adjacent turn flips were 0/0/0. Spin means linear <0.05 m/s
and |angular| >0.9 rad/s; a large flip requires opposite adjacent signs with
both |angular| >0.5 rad/s. Taking `ceil(1.25 * development maximum)` sets
118 total and 28 consecutive spin commands (2.8 s). Zero flips remains an
exact invariant, with no numerical margin. Mean curvature is ill-conditioned
near zero translation (15.93/24.45/5148.89 in these dev runs), so consecutive
spin duration replaces that historical bound.

All three crossing runs completed without contact and made monotone progress
over the first ten seconds. Both bottleneck configurations completed without
contact on dev seed 1001. The named replacement tests in
`tests/test_issue_9724_social_force_resolution_independent.py` reproduce these
contracts and assert the observed 0.1 s clock at every policy call. The helper
rejects episode seeds outside 1001–1030 before simulation.

Calibration used the already-fixed controller at PR head
`39d9932efb802e4f340502f6fb8d4acfcab20795`, plus the top-level `dt` resolver
repair. No historical episode test or held-out measurement was used to choose
these seeds or bounds. Private raw command/position traces and the calibration
script are preserved under `~/fxb2_probe`; compact summaries are in this fixture.
