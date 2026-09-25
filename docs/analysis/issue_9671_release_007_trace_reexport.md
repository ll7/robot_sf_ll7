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

## Exact acquired episode roster

These are the episode IDs in the cold-retrieved producer JSONL, not IDs from the earlier
dissertation figures. The four JSONL SHA-256 values are `3b10ae893fe0f7b8f152286ba6a4db89b7c5b65c09b625061249e7166e37c712`
(`goal`), `74b9587d799577c3257d7678e78a0a0f0abae5da8778ae4ccae8685ecb384493`
(`orca`), `9fefa08e3fb9421fc370686e561fd3e9cd9e247511056e699a2fbb68ec1f52a2`
(`social_force`), and `5a8656df3d458cf491192d5e542ec395d0cd1396f57881fa0c3220ada69460b1`
(`ppo`). The first three are job 15758; PPO is job 15760. All use differential drive.

| Planner | Scenario | Seed | Episode ID | Status | Release-row comparison |
| --- | --- | ---: | --- | --- | --- |
| goal | group crossing medium | 22 | `classic_group_crossing_medium--22--b5435eeb359163c7` | success | no release row |
| goal | group crossing medium | 23 | `classic_group_crossing_medium--23--16ea4a94ca01f5d5` | success | no release row |
| goal | group crossing medium | 24 | `classic_group_crossing_medium--24--73e53c253f521197` | success | no release row |
| goal | head-on corridor medium | 22 | `classic_head_on_corridor_medium--22--9ef3cac5f81f2edb` | success | no release row |
| goal | head-on corridor medium | 23 | `classic_head_on_corridor_medium--23--78296203ecfacac2` | collision | no release row |
| goal | head-on corridor medium | 24 | `classic_head_on_corridor_medium--24--0f599e16dc6818b3` | collision | no release row |
| ORCA | group crossing medium | 22 | `classic_group_crossing_medium--22--49266c16d5b50efb` | success | no release row |
| ORCA | group crossing medium | 23 | `classic_group_crossing_medium--23--ef6829403c0bd8a6` | success | no release row |
| ORCA | group crossing medium | 24 | `classic_group_crossing_medium--24--45497ec249e5766d` | success | no release row |
| ORCA | head-on corridor medium | 22 | `classic_head_on_corridor_medium--22--5fbe02a3b4747433` | success | no release row |
| ORCA | head-on corridor medium | 23 | `classic_head_on_corridor_medium--23--7f9e6c61b772f737` | success | no release row |
| ORCA | head-on corridor medium | 24 | `classic_head_on_corridor_medium--24--70b33393d582e71d` | success | no release row |
| social force | group crossing medium | 22 | `classic_group_crossing_medium--22--51ad526a5252f1fc` | failure | no release row |
| social force | group crossing medium | 23 | `classic_group_crossing_medium--23--0905ebb6c9ed6a2a` | failure | no release row |
| social force | group crossing medium | 24 | `classic_group_crossing_medium--24--5d52942c47b9e2a5` | failure | no release row |
| social force | head-on corridor medium | 22 | `classic_head_on_corridor_medium--22--cdb662017dde72ab` | failure | no release row |
| social force | head-on corridor medium | 23 | `classic_head_on_corridor_medium--23--411ab8eabb15937e` | failure | no release row |
| social force | head-on corridor medium | 24 | `classic_head_on_corridor_medium--24--96f958a1df409a15` | success | no release row |
| PPO | doorway medium | 113 | `classic_doorway_medium--113--0970d6f82f290390` | collision | match |
| PPO | doorway medium | 114 | `classic_doorway_medium--114--14958fc7b6babde6` | collision | match |

The five dissertation figure receipts from predecessor job 13334 point to source commit
`12d0284f9b316a3c9aa22376088e9690414990c9` in this repository. Their recorded
`trace_series.json` SHA-256 values were recomputed from that commit, and each source's
episode ID and status agree with its receipt. Each key below names the receipt at
`ll7/diss/docs/context/figure_data_receipts/trace-ch7-worked-examples-job13334__<key>.json`;
the receipt records the exact earlier `trace_series.json` source path.

| Figure trace key | Earlier episode ID / status | Earlier trace SHA-256 | Current diagnostic episode ID / status |
| --- | --- | --- | --- |
| `groupcross_seed22_goal` | `classic_group_crossing_medium--22--605d6793ad25c1f5` / success | `428d327d1370d8ccbd7779d4b0d11f27ddd293260ae808a99db80edb39be3443` | `classic_group_crossing_medium--22--b5435eeb359163c7` / success |
| `groupcross_seed22_sf` | `classic_group_crossing_medium--22--6ea3e69c68960055` / failure | `551b7e7be142c254547b9b419f6f5f44fb4c26e998fb8a65b998004ad04bf880` | `classic_group_crossing_medium--22--51ad526a5252f1fc` / failure |
| `headon_seed23_orca` | `classic_head_on_corridor_medium--23--475e0eb34a5e8f23` / collision | `81ec22f92658a299241825640832394c54f1222aa152b6e8a2f51755503ab43a` | `classic_head_on_corridor_medium--23--7f9e6c61b772f737` / success |
| `headon_seed24_orca` | `classic_head_on_corridor_medium--24--9392c5c14a3d9d6f` / success | `7a5494169b7627dcbe094d1b9a733177bbf5a9cf485db77118aeb921dceba3f9` | `classic_head_on_corridor_medium--24--70b33393d582e71d` / success |
| `headon_seed24_sf` | `classic_head_on_corridor_medium--24--1bea887e93462d65` / failure | `351b3906158273c5e6ab2e8ccd7a9886eeba310314548293ffc88da647682642` | `classic_head_on_corridor_medium--24--96f958a1df409a15` / success |

The ORCA seed-23 collision and social-force seed-24 failure used by the existing head-on
figure reading are **not present** in these 0.0.7 diagnostic traces. Figures 7.6 and 7.10–7.11
therefore need a new reading or a different explicitly selected episode before their earlier-run
labels can be removed. This is a cross-run worked-example change, separate from the **zero**
paired release-row outcome mismatches. The source of the cross-run change has not been established.

## Implemented passive robot-force observer — not yet executed

The frozen `PedRobotForce` already retains each component's `last_forces`, but the frozen
benchmark writer records only `last_ped_forces`. A frozen-code JSONL record cannot acquire a new
field without changing code. Post-integration robot/pedestrian positions cannot reproduce the
force-evaluation input reliably, particularly with optional pedestrian response multipliers.
The implemented but unexecuted path is a separately SHA-pinned, opt-in **observer overlay** in
`scripts/validation/issue_9671_force_observer_sitecustomize.py`; it leaves every file
at commit `07f7e8d43084de748915e1b1eb8b2a1603357c6e` and both scientific config bytes
unchanged, but the executed Python process includes observer code. Its sidecar is a new
diagnostic artifact, not an original 0.0.7 release field.

1. Stage the observer from a reviewed commit under ignored `output/`, verify its SHA-256, and
   load it through an explicitly named `sitecustomize` path in the private Slurm packet. Require
   one in-process worker, matching these configs; write one PID-scoped capture stream. Register
   both `sys.settrace` and `threading.settrace` at startup. A later thread that enters
   `run_map_episode` is observed and rejected before its body executes; a fork with the inherited
   observer is also rejected by PID. At installation, verify the three traced source files
   byte-for-byte against pinned commit `07f7e8d`; on each traced call, require the code filename,
   module `__file__` and module name to resolve to those files under that verified checkout.
   Require frame globals to be the live `sys.modules` dictionary, frame code identity to the
   module's actual function/method, and code-object equality to a fresh compile of the pinned
   bytes. A direct import probe in the frozen checkout matched all four target code objects.
   A shadowed import root fails before a sidecar can be written. No sidecar is admitted for
   unobserved or failed episodes.
2. An opt-in `sys.settrace` observer copies the frozen `PedRobotForce.__call__` return value
   and its already evaluated frame locals (`ped_positions`, `robot_pos`, and `multipliers` if
   present). It must never call a force kernel, position provider or multiplier callback again.
   At frozen `Simulator.step_once` line 1699 (or `PedSimulator.step_once` line 2087), after
   `pysf_sim.compute_forces()` and before `_apply_residual_adversary` or pedestrian integration,
   select registered objects with `component_type == "pedestrian_robot"` **and**
   `isinstance(PedRobotForce)`; exclude `adversarial` components. Match exactly one fresh
   return capture per selected instance, copy its `last_forces` and configuration, and sum the
   robot components. An active component left at its scalar initial value, a non-finite vector,
   shape mismatch, missing active component or changed component ID/object roster across steps
   fails closed. The
   observer does not infer a zero vector when no `PedRobotForce` instance executes.
3. On each `step_once` call, copy the pre-behavior pedestrian positions. At the force-return
   event, copy the already evaluated force-input positions. Both must exactly equal reset
   positions on step zero and the prior post-step trace positions thereafter. Require distinct
   positions, so coincident actors cannot be silently assigned by row order. On each step
   return, copy `last_ped_forces` and post-step positions. Observe `run_map_episode` call/return
   at frozen line 5075 to bind the ordered samples to its returned `episode_id`, input scenario,
   seed, planner and step count; the row's canonical SHA-256 binds the observer receipt to
   exact episode content. Require contiguous step indices and the same ordered
   `trace_actor_ids` on reset and every step; compare post-step positions and total vectors to
   the recorded trace. A failed or ambiguous episode is not admitted. This deliberately rejects
   an episode if behavior changes positions before force evaluation, or if two actors occupy
   identical positions; such a row needs a new identity method before force attribution.
4. Keep the runner's raw JSONL and producer manifest unchanged. Emit separate per-episode
   sidecars with observer SHA, frozen source/file SHAs, diagnostic config SHA, campaign and
   Slurm job IDs, PID,
   episode ID, per-step arrays and canonical episode-row SHA-256. The later producer/retrieval
   manifest must bind these row digests to the raw JSONL SHA and include checksums for every
   sidecar and the staged observer input. `check_issue_9671_force_bundle.py` implements a
   deterministic manifest writer and cold validator, requiring the independently pinned
   manifest SHA-256. It binds the startup receipt, separately SHA-pinned reviewed private
   launch-packet YAML, immutable submission-intent receipt, campaign manifest, producer JSONL/manifest,
   observer bytes, every sidecar byte stream and the 20 exact episode IDs. It checks complete
   step/actor/force series and the component sum, then compares every new outcome and recorded
   robot/pedestrian state/total-force vector against the SHA-pinned job 15758/15760 baseline.
   A changed outcome or state is retained as `diagnostic_mismatch`; it is not silently promoted.
   The existing baseline remains **2 match, 0 mismatch, 18 `no_release_row`**. The bundle gate
   is implemented but **not yet executed on observer-run artifacts**; no force-enriched bundle
   is admitted.
5. Before any force-enriched bundle is admitted, rerun the same 20 tuples on the same frozen
   source/config under a **new** Slurm campaign and compare status, step count and every
   recorded robot/pedestrian state and total-force vector with jobs 15758/15760. Require exact
   numeric equality; any difference is an observer/context finding, not parity. Recheck the two
   doorway outcomes against the frozen archive; retain `no_release_row` for the other 18.
   Preserve new packet/config/observer SHAs, source SHA, job ID, producer checksums, cold
   retrieval receipt and a separately versioned durable artifact. Do not overwrite either
   existing campaign or the 0.0.7 release bundle.

For the later reviewed launch, the bundle gate takes a JSON `--spec` with `archive`,
`baseline_report`, all four `baseline_traces`, `observer_path`, `observer_sha256`, and
`campaigns.headon_group`/`campaigns.doorway`. Each campaign entry supplies its unchanged
`config`, producer `campaign_manifest`, canonical `startup_receipt`, reviewed `launch_packet`
and `launch_packet_sha256`, runtime `packet_sha256`, `submission_intent_receipt` and its
`submission_intent_sha256`, reviewed `queue_id` and `submission_id`, new `campaign_id` and
`job_id`, raw `traces`, and `sidecar_dir`. The intent receipt must derive its submission ID
from queue ID, runtime packet digest, attempt, and nonce exactly as canonical private ops does;
all three identities must match the producer startup receipt. The runtime packet digest is
computed from script/config/route/submit arguments; it is **not** the YAML file SHA or proof
that `startup.packet` names the YAML. Missing packet identity fails closed before candidate
admission. The launch YAML must declare `identity.observer_sha256` and match the staged observer.
The YAML's reviewed SHA must be supplied **outside the bundle spec** with
`--approved-headon-launch-packet-sha256` and `--approved-doorway-launch-packet-sha256`
for both writing and cold validation. Obtain those pins from the independent packet review;
recomputing them from the spec or a modified local YAML defeats the approval boundary.
The baseline report defaults to pinned SHA-256
`e1637fa907d87f8a5456ee0f3367524e8e335b480c1d2bd5162215f08a3f7ffd`.
Manifest artifact keys are campaign-relative, so a complete cold-retrieved tree can move to a
new host path without changing its manifest bytes; files outside the campaign tree fail closed.
After producer checksum and cold-retrieval checks, run
`uv run python scripts/validation/check_issue_9671_force_bundle.py write --spec <spec.json> --manifest <new-manifest.json> --approved-headon-launch-packet-sha256 <reviewed-sha> --approved-doorway-launch-packet-sha256 <reviewed-sha>`;
record the printed manifest SHA separately, then use `validate --spec ... --manifest ...
--manifest-sha256 <recorded-sha>` with both reviewed packet SHA arguments on cold artifacts.
Exit 2 means a recorded outcome/state
finding and is not candidate evidence. The writer refuses to overwrite an existing manifest.

This plan needs review and #9667 to merge before a new packet or Slurm submission. The observer
may change timing or planner behavior despite leaving force results untouched; the parity gate
detects an observed change but cannot prove absence of every timing effect. If the requirement
means an uninstrumented 0.0.7 Python process, recording the missing component is impossible;
the existing 20 source-faithful traces remain the honest deliverable. Table 8.3 still lacks a
login-node context and a fixed-context repeat; no value is inferred from the predecessor's
62/78/37 counts.
