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
