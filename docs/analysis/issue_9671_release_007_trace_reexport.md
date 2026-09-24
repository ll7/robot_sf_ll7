# Release 0.0.7 worked-example trace re-export (#9671)

This is a diagnostic acquisition of per-step examples. The benchmark-data release at source
`07f7e8d43084de748915e1b1eb8b2a1603357c6e` and its publication bundle are immutable.
The trace inputs are separately pinned configurations; they do not replace the release campaign
configuration or its 20,160 rows.

## Inputs and comparison boundary

- Frozen archive: `issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz`,
  SHA-256 `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml`,
  SHA-256 `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`.
- `configs/benchmarks/issue_9671_trace_headon_group_v007.yaml`: three release planners
  (`goal`, `social_force`, `orca`), head-on and group-crossing medium, seeds 22–24, H600,
  dt 0.1, differential drive, 18 requested episodes. The social-force arm selects the
  release's `terminal_goal_v1` law.
- `configs/benchmarks/issue_9671_trace_doorway_v007.yaml`: release PPO configuration,
  classic doorway medium, seeds 113–114, H600, dt 0.1, differential drive, two requested episodes.
  The checkpoint must load in native mode without fallback.

Both diagnostic configs turn on force, planner-decision, and simulation-step recording. Their
worker count is one; the release used its campaign execution context. Accordingly, outcome
differences are findings, and per-step values are diagnostic observations from the rerun context.
The configs were derived from the later #9431 trace input, then constrained against the frozen
archive and release's versioned scenario/planner settings. The config bytes are not claimed to
exist at frozen commit `07f7e8d`.

The release used seeds **111–140**. Thus none of the 18 head-on/group tuples at seeds 22–24 has
an outcome row in the release archive. They must be reported as `no_release_row`, never as
matches or mismatches. The archive does contain the two requested doorway PPO rows: both seeds
113 and 114 have `status=collision`. Compare each rerun doorway status with its corresponding
frozen row, even if the diagnostic configuration reproduces the scientific inputs.

## Admission and custody

Run the exact frozen source in a clean cluster checkout. Supply each diagnostic config as a
separately SHA-pinned input and record the config's source PR/commit as well as the execution
commit. Preflight must bind the actual Python import root, scenario matrix, checkpoint and
ORCA dependency. Submit through the private operations queue and retain the packet, scheduler
receipt, producer checksums, result-root path, and cold-readback receipt. Preserve the raw JSONL
and logs in durable artifact storage outside Git.

After collection, run `scripts/validation/check_issue_9671_trace_reexport.py` with the exact
archive, both episode JSONL files, both diagnostic config files, and both produced
`campaign_manifest.json` files. Retain each runner-produced
`episodes.jsonl.provenance.json` beside its JSONL: the comparator checks its whole-file checksum,
every row's line, episode, scenario, seed, source, and scenario-parameter hash, then binds the
producer file to the matching diagnostic campaign directory and scenario-matrix hash. A mixed
or unmanifested JSONL fails admission. The comparator requires the frozen source SHA, the separately
pinned config SHA-256 values and effective hashes, per-step finite robot/pedestrian states and
total pedestrian force vectors, and release-equivalent scientific parameters apart from the
three recording flags. For seeds 22–24, the parameters are compared to the same release
scenario/planner at a release seed with only `route_spawn_seed` substituted; their outcomes remain
`no_release_row`. It writes every row's `match`, `mismatch`, or `no_release_row` classification
with input checksums. A doorway outcome mismatch remains in the report and makes the CLI exit 2;
it is never silently corrected. The frozen simulator has no #9666 robot-attributable force split,
so these traces **do not contain that component**, even if #9666 lands before acquisition. A
separate #9666 diagnostic may establish the component on a later source; it cannot retroactively
change this frozen-source trace record.

## Status

Both configuration preflights passed on the SSH development host on 2026-09-24. No #9671
Slurm job or trace-result checksum is claimed here until its submission and cold readback are
recorded. The 0.0.7 archive SHA above was rechecked locally. The legacy three-context Table 8.3
numbers concern the earlier 0.0.3 source and must be re-measured or relabelled before being
presented as a 0.0.7 result.
