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
The frozen execution worktree stays Git-clean: the two tracked PR config bytes are hydrated at
`output/benchmarks/issue9671/inputs/` from config-origin commit `cb1d650a`, then checked against
their SHA-256 values above before preflight and run. `output/` is transient; the tracked PR and
packet provide the durable source. Staging changes the runner's effective config hashes to
`b195d55f16871ba2` (head-on/group) and `a30c4555ce8a3f0a` (doorway), while the config bytes and
scientific parameters remain identical. The validator binds those staged path identities and hashes.
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
archive, all four episode JSONL files, both diagnostic config files, and both produced
`campaign_manifest.json` files. The validator accepts the absolute staged config paths recorded by
those manifests, so the clean checkout can live under any host path. Retain each runner-produced
`episodes.jsonl.provenance.json` beside its JSONL: the comparator checks its whole-file checksum,
every row's line, episode, scenario, seed, source, and scenario-parameter hash, then binds the
producer file to the matching diagnostic campaign directory. The producer's per-arm scenario
hash differs from the full campaign scenario-matrix hash, so the comparator checks each hash
within its own scope and checks the producer's scenario input SHA-256 against frozen bytes. The
producer invocation must match the campaign invocation, including the named `--config` and
`--campaign-id`; its schema, scenario-matrix, and planner configuration input checksums must
match bytes pinned from the frozen source. A mixed, swapped, or unmanifested JSONL fails
admission. The comparator requires the frozen source SHA, the separately
pinned config SHA-256 values and effective hashes, native or adapter execution without runtime
fallback/degradation, complete contiguous per-step state with count matching the episode record,
finite robot/pedestrian states and total pedestrian force vectors, and release-equivalent scientific parameters apart from the
three recording flags. For seeds 22–24, the parameters are compared to the same release
scenario/planner at a release seed with only `route_spawn_seed` substituted; their outcomes remain
`no_release_row`. It writes every row's `match`, `mismatch`, or `no_release_row` classification
with input checksums. A doorway outcome mismatch remains in the report and makes the CLI exit 2;
it is never silently corrected. The frozen simulator has no #9666 robot-attributable force split,
so these traces **do not contain that component**, even if #9666 lands before acquisition. A
separate #9666 diagnostic may establish the component on a later source; it cannot retroactively
change this frozen-source trace record.

## Status

Both staged-configuration preflights passed on the SSH development host on 2026-09-24. A first
canonical submission attempt stopped before `sbatch`: the clean-source guard rejected untracked
config files inside the frozen worktree. Those exact bytes were then staged under ignored `output/`
from the tracked PR source, verified by SHA-256, and the frozen worktree returned to a clean Git
state. Diagnostic jobs **15758** (head-on/group) and **15760** (doorway) completed with scheduler,
producer, and sync exit code zero. Producer SHA256SUMS and cold retrieval checks passed (57 and
51 files). Artifact-verification receipts passed. Both campaigns have verified W&B v0 and local
snapshot copies with matching preservation manifests; private queue preservation closeout metadata
is still pending.

| Job | Raw producer SHA256SUMS SHA-256 | Verified W&B artifact | Preservation manifest digest |
| --- | --- | --- | --- |
| 15758 | `ec7d633dac430ddb36d78b9dc3a4a318d82e32041b9001080adf033b417082f4` | `wandb://ll7/robot_sf/campaign-issue9671_trace_headon_group_v007_07f7e8d_20260924:v0` | `sha256:b1109c1c085dd491f4f8538bfba68fc88c0e890b7fd75ade675b56bcb6904265` |
| 15760 | `3f8fb73ae92431a46ab0dd5e45d72051d35f06d2ec1a3a06483c71d2bb8c1791` | `wandb://ll7/robot_sf/campaign-issue9671_trace_doorway_v007_07f7e8d_20260924:v0` | `sha256:73b264d8fb8910d0f2ddfffc074005c31eaadd6db9b72d8f23db8d32ffa3c5a9` |

The comparator admitted 20 exact raw rows and wrote `release_007_trace_outcomes.json`, SHA-256
`e1637fa907d87f8a5456ee0f3367524e8e335b480c1d2bd5162215f08a3f7ffd`: **2 matches,
0 mismatches, 18 `no_release_row`**. Doorway PPO seeds 113 and 114 both ended in `collision`,
matching their frozen release rows. The 18 head-on/group seeds 22–24 have no release row and are
diagnostic observations only. These outcomes do not establish trajectory parity because the
trace-only configs and one-worker context differ from the release campaign.

The frozen archive and job 15760 both record doorway PPO seed 114 with 16 exposure steps in 176
retained steps at a 2.0 m radius; seed 113 has zero exposure steps in 49. The comparator report
includes both release and trace numerators. A SHA-pinned direct archive readback receipt,
`release_007_doorway_seed113_114_archive_readback.json` (SHA-256
`7c18e5061f69a471d60e6d990741d80feca0fddf8a6f2b1c25c21c6149897475`), records the exact
episode IDs and metric definition. The dissertation source at ll7/diss commit `b7ce8289`,
`diss/chapters/05_discussion.tex:540`, instead says the current-release seed 114 had four exposure
steps. That statement conflicts with the frozen archive; it must be corrected before citing a
current-release exposure number. The 62/78/37 exposure steps in Table 8.3 are predecessor-release
execution contexts and remain separate. This re-export supplies the current-release campaign and
one-worker cluster contexts for seed 114; it does not yet provide a login-node context or a
fixed-context repeated trace, so Table 8.3 cannot be regenerated from these two jobs alone.

The 0.0.7 archive SHA above was rechecked locally. The legacy three-context Table 8.3
numbers concern the earlier 0.0.3 source and must be re-measured or relabelled before being
presented as a 0.0.7 result.
