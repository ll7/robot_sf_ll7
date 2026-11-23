# Pedestrian group provenance

Map benchmark episode rows include `pedestrian_group_provenance.v1`, captured
after reset and before the episode step loop. It preserves each runtime group ID,
sorted pedestrian slot IDs, and membership count, including singleton and empty
groups. `unassigned_pedestrian_ids` records slots without a group. Social-group
prevalence uses groups of size at least two, reported by
`multi_member_group_count` and `pedestrians_in_multi_member_groups`.

IDs refer to pedestrian array slots **at reset**, not persistent identities across
respawns. This snapshot is independent of the planner, per-step telemetry, authored
map social groups, and group-space metrics. It does not change group generation,
planner actions, simulator dynamics, metrics, or episode resume identity.
Missing runtime grouping produces `status: unavailable`, not a measured empty
population. Historical rows without this optional field remain schema compatible;
their missing memberships cannot be inferred from scenario names or crowd density.

## Development reset audit

An explicit development replay writes a separate derived JSONL file:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/analysis/replay_pedestrian_groups.py \
  --episodes dev_episodes.jsonl --matrix dev_scenarios.yaml \
  --out derived_groups.jsonl --authorize-dev-reset-replay
```

The command accepts only seeds 1001–1200, validates the entire input batch before
constructing an environment, and never steps a planner. Explicit simulator seed
controls must also use this development band. Existing output files are refused;
input episodes and matrices are never modified. Each derived row binds the input
file and row, supplied matrix, resolved scenario and consumed runtime inputs,
and records both the input row's source and the replay source.

Every export is **diagnostic-only** with `historical_equivalence: not_established`.
The supplied matrix and current simulator may differ from historical inputs.
Commit labels alone do not prove replay equivalence. Frozen release artifacts
remain untouched. Published or sealed replay is unsupported by this command and
requires a separate author decision and an approved execution contract.
