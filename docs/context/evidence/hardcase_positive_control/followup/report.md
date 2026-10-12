# Seed-1001 window repair and matched benchmark audit

**Window replay PASS twice; corpus admission BLOCKED_EXTERNAL.** The one-step
failure in #10330 is repaired on PR #10331. The existing persistence CLI was
executed and rejected the window; no case was admitted. The PR stays draft,
open and unmerged. The parent folder preserves the historical pre-fix result.
This report supersedes that report for the authorized follow-up.

Base/source main: `53721be3248600e08641a9816d094b605e3c93e4`.
Production repair: `1baaa3eb08cfbc1866f18702eb939c1f60b400fd`.
Branch: `lane/hardcase-positive-control`.
This is one-case development simulator evidence, not a release, paper,
real-world safety result, adversarial search campaign or generic checkpoint claim.
All real episodes used **1001**. No sealed-seed reset/replay, frozen artifact edit,
merge, readiness promotion or attribution line was performed.

## Per-stage result and exact command

The following invocation executed source capture, extraction, materialization,
two consecutive independent fresh-environment replays, comparison, and the
existing persistence CLI. Observed exit **0** means the replay recipe completed;
it does **not** mean its downstream gate admitted the case.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  docs/context/evidence/hardcase_positive_control/followup/reproduce.py \
  --output docs/context/evidence/hardcase_positive_control/followup/packet
```

The retained output is curated after execution; use an empty output directory
(e.g. `../hardcase-followup-reproduced`) for a new run. Runtime inputs must match
the repair commit, and original source input hashes must match the historical
provenance. The recipe checks those conditions before simulation.

| Stage | Result | Exact operation inside that command / receipt |
| --- | --- | --- |
| Scenario/model/config provenance | PASS | `load_scenarios(MATRIX)` selects `classic_cross_trap_high`, seed `[1001]`; verifies every original input SHA; `packet/provenance.json` binds source and repair commits, checkpoint compatibility and input/artifact digests. |
| Reproducible critical source and event trace | PASS | `run_map_episode(..., 1001, horizon=600, dt=0.1, record_forces=True, record_simulation_step_trace=True, before_step_hook=capture, policy_builder=build_map_policy)`; all 600 source trace frames equal the independent pre-fix retained record. |
| Critical-window extraction | PASS | `extract_critical_segment(..., pre_margin_s=1, post_margin_s=1)`; entire entry equals the original catalog entry: 21 samples at 37.0–39.0 s. |
| Replay materialization | PASS, `loads_only` | `materialize_generated_scenario(entry, max_episode_steps=20, source_scenario=scenario, continuation_checkpoint=checkpoint)`; original world/goals/population/budget preserved; v2 descriptor binds the actual checkpoint. |
| Fresh window replay 1 | PASS | `run_map_batch(window.yaml, window_replay_1.jsonl, episode.schema.v1.json, horizon=600, dt=0.1, record_forces=True, algo="social_force", algo_config_path=CONFIG, benchmark_profile="experimental", record_simulation_step_trace=True, resume=False, workers=1)`; executes absolute steps 370–389 after restore. |
| Fresh window replay 2 | PASS | Same call with independent output `window_replay_2.jsonl`; new environment and restore, immediately after replay 1. |
| Exact parity comparison | PASS twice | `actual == retained_source_steps[370:390]`; canonical step digest, event count, first-event step, coarse outcome, and robot/pedestrian trajectories all checked; `packet/comparison.json`. |
| Existing persistence gate | **FAIL: REJECT** | CLI below executes successfully (exit 0) but emits zero promoted / one rejected; `packet/persistence/*.json` and `gate_command.json`. First failure is whole-episode identity mismatch. |
| Versioned corpus admission | **FAIL: unavailable / not executed** | `git ls-tree -r --name-only origin/main` contains no `counterexample_corpus.py` or planned admission implementation; bounded broker discovery plus verified tracked tree and fresh #9652 state. No authoritative admission command exists on this main. |

The exact existing gate command invoked by the recipe was:

```bash
.venv/bin/python scripts/tools/evaluate_pipeline_persistence_gate.py \
  --candidate docs/context/evidence/hardcase_positive_control/followup/packet/entry.json \
  --episodes docs/context/evidence/hardcase_positive_control/followup/packet/gate_source.jsonl \
  --config configs/analysis/issue_5600_persistence_gate.yaml \
  --output docs/context/evidence/hardcase_positive_control/followup/packet/persistence \
  --replay-results docs/context/evidence/hardcase_positive_control/followup/packet/gate_replay_results.json \
  --commit-hash 1baaa3eb08cfbc1866f18702eb939c1f60b400fd
```

At execution the recipe passed absolute output paths; `gate_command.json`
preserves the normalized argv. The duplicate source gate input was omitted
and replay input compressed during curation; the recipe regenerates both.
No persistence configuration was modified.

## Root cause, repair and parity scope

The directional adapter substituted the extracted endpoint for the source goal.
That endpoint is **0.8365615368996463 m** away, within the default waypoint
acceptance radius of **2 m**. The recorded done condition was route completion
on step zero, not budget exhaustion. A 20-step budget cannot prevent this done.
Fresh reset also discarded source velocities, navigators, RNGs, sensor heading,
reward history and the analytic planner's filter state.

Directional materialization now requests explicit goal-zone entry into a finite
endpoint polygon. It remains a load/execution smoke, without a parity claim.
The opt-in continuation path uses the existing typed simulator snapshot plus a
strict companion for clock/distance history, cached observation, last action,
SNQI history, sensor heading and mutable analytic adapter state. It restores
into the original source world/goals, keeps its **600-step / 60 s** budget and
runs only the **20** window transitions, with absolute step/time offsets.
Initial trace bookkeeping is restored so the first pedestrian velocity is not
zeroed. Seed, map, config, code, platform, boundary and companion hashes are
checked before use. Unsupported controller, noise, tracking/foresight, wrapper,
latency and recording histories fail closed. This is the supported single
RobotEnv / SocNav / SocialForcePlannerAdapter case, not a general restart format.

Both retained replays match **every recorded channel** and actor identity in
source steps 370–389. Robot and pedestrian maximum position error is **0 m**
against tolerance **1e-9 m**; both canonical trace digests are
`37488c586d0a7fd203b9385377434516f108edb8bef011a6003a3236e135fed3`.
Full source digest remains
`7a73ee858ba8248969015c4f31273c014d15fd328bd97671876f9520c2bf2b34`.
Both have **11 near-miss samples**, minimum surface clearance
**0.47727323760374385 m**, first near-miss **global step 374 / local step 4 /
37.5 s**, and no contact. Minimum is at step 379 / 38.0 s.
The extractor's 1.8772732376037438 m center distance subtracts 1.4 m of actor
radii for the safety metric; it is not used as a surface-clearance threshold.

Coarse outcome flags match the source: collision false, route complete false,
timeout true. **Window stopping is censoring at 39 s**, while the source's
original timeout occurs at 60 s. All twenty resumed RL `terminated` and
`truncated` flags are false and match source frames. This proves exact event
and trace continuation in the extracted window, not reproduction of a later
terminal transition or equality of whole-episode metrics.

Source config SHA-256:
`9a7015a5f66b5d9ddb5d96c706d33ac8fe027bc0aee256e9ab87d2e89804f773`;
effective planner hash `a4913be8bee6f3f3`; original run hash `f330dd29077077ac`.
No learned model is applicable. Analytic implementation digest is
`fe4e5b99c0a9f51ada80cfccbf1361bb4f1872c0c523e2b35ce89e94e7b32958`.
Historical provenance pins all included source YAML/map/lock bytes; none changed.
Registry execution/readiness are `adapter`, availability `available`, with no
fallback/degraded execution. This does not promote optimizer-facing native
eligibility or the materializer's review/evidence flags.

## Admission blocker and revival

The first failing gate expects the whole source episode's identity and complete
trace. The real generated window has its own config-derived ID; the repair
keeps that ID rather than renaming it to pass. The gate also reads a truthy
integer native `steps` field as though it were a frame list, and defaults planner
to `goal` rather than reading native `algo`. Its event failure is an envelope
mismatch, not evidence that the independently verified near-miss disappeared.
Nine preregistered timing/speed cells explicitly remain **unknown**. No independent
perturbation execution or persistence claim is made; interpolation was not used.

Versioned corpus admission remains owned by
[#9652](https://github.com/ll7/robot_sf_ll7/issues/9652#issuecomment-6125063538),
whose implementation is absent on this main. The native-envelope diagnostic is
recorded on existing
[#5600](https://github.com/ll7/robot_sf_ll7/issues/5600#issuecomment-6125063906).
No unmerged owner branch was imported. Revival requires a landed fail-closed
admission implementation accepting an explicit source-bound window receipt,
canonical native envelope handling, and whatever independent perturbation proof
its contract requires. Admission/feasibility/duplicates must remain honest;
unknown feasibility is not fabricated into a proof. Corpus size is unchanged.

## Changed-module checks and benchmark baseline

Bug regression was first run before repair:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/benchmark/test_generated_window_continuation.py::test_seed1001_window_does_not_complete_at_its_boundary \
  -q --tb=short
```

**RED: `assert 1 == 20`**, one failure in 10.34 s, actual premature completion;
not an import/fixture failure. The same test passed after the goal repair.
Final full real regression module command:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/benchmark/test_generated_window_continuation.py -q --tb=short
```

**4 passed in 16.58 s**: actual window duration, two independent exact checkpoint
replays, tampered companion rejection and parent-path rejection using a real
captured packet. The new continuation feature has no pre-fix API; its positive
and negative tests are marked `new_feature_no_base`, not claimed as old-API red
proof. The retained historical trace is an independent oracle.

The existing tests touching both changed modules and the downstream wiring used
this exact command (with the broad-audit guard environment below):

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/benchmark/test_generated_scenario_replay_adapter.py \
  tests/benchmark/test_map_runner_episode_typed.py \
  tests/benchmark/test_map_runner_decomposition_helpers.py \
  tests/benchmark/test_data_driven_scenario_generation.py \
  tests/benchmark/test_pipeline_persistence_gate_wiring.py -q --tb=short
```

**109 passed, 7 skipped in 2.27 s**. The seven simulator-entry skips are not passes.
Ruff lint and format checks passed for all four changed code/test files and both
new evidence scripts. `verify.py` independently checks retained snapshot hashes,
complete frame equality, event steps and rejection without any simulator reset.

The required **full `tests/benchmark` check** was named before PR delivery and
completed on main and on the repaired branch with identical interpreter,
dependencies, pytest arguments, and task-only interlock bytes:

```bash
bash ../run_main_benchmark_check.sh    # from separate main-baseline checkout
bash ../run_benchmark_check.sh         # repaired branch
```

Both wrappers set `HARDCASE_NO_TEST_SIMULATION=1`,
`ROBOT_SF_PYTEST_SEED_GUARD=1`, a task seed-audit path, and
`PYTHONPATH=<lane>/test_guard:$PWD/tests/support:$PWD`. Main uses:

```bash
scripts/dev/run_worktree_shared_venv.sh \
  --venv /home/luttkule/lanes/hardcase/repo/.venv -- uv run pytest \
  tests/benchmark -q --tb=short --continue-on-collection-errors \
  --junitxml=../main_benchmark.xml
```

Branch uses the same command without the explicit shared-venv argument and
`--junitxml=../benchmark.xml`. Import paths were checked to point at main's
`robot_sf` and `fast-pysf`, not branch sources. Main was unchanged at the pinned
base. The interrupted initial baseline had no result and was preserved; only
the completed matched run is counted. Production/test bytes in the final branch
audit equal the committed repair; later edits are evidence/report only.

**Baseline: 12,116 passed, 119 failed, 336 skipped, 13 collection errors**
in 24m53s, exit 1. **Repaired branch: 12,116 passed, 119 failed, 340 skipped,
13 collection errors** in 24m47s, exit 1. All **132** failing/error IDs match,
with identical kinds: **zero introduced**, zero baseline failures removed.
The four additional skips are the new real-case tests, passed separately above.

The complete grouped ID list, per-test kind and baseline exception are in
`benchmark_audit.json`. Root causes are grouped without unrelated fixes:

| Root cause | Failures | Collection errors | Tracking |
| --- | ---: | ---: | --- |
| Missing optional packages/profile expectations, including installed learned-policy version metadata | 82 | 11 | Existing [#7640](https://github.com/ll7/robot_sf_ll7/issues/7640#issuecomment-6125050881); updated, no dependency/profile changes. |
| Collection-time partial torch module leaked through `sys.modules` | 11 | 2 | New [#10332](https://github.com/ll7/robot_sf_ll7/issues/10332); no existing specific issue found; left unchanged. |
| Historical Git generation/source objects not in fetched refs | 23 | 0 | Existing [#10207](https://github.com/ll7/robot_sf_ll7/issues/10207#issuecomment-6125051694), companion #10251; grouped once, no fixture changes/hydration. |
| Task simulator interlock: two multiprocessing skip-pickling errors and one child-canary assertion | 3 | 0 | Intentional task coverage restriction, not a production defect or public infrastructure issue. |

No separate caller-thread failure is demonstrated by these tracebacks; the SNQI
failures shown here originate from absent distribution-version metadata.
Optional coverage and guarded simulator tests remain unavailable in this broad
check. This is not a green full-suite result.

Test-value gate: the duration test protects against accepting a close window
goal at its start; reverting goal semantics makes it fail; prior adapter coverage
only loaded a short synthetic trace; no test-only seam is used. The continuation
test protects source events/channels across fresh restores; losing clock/filter/
sensor/trace history breaks it; simulator-only snapshot tests omitted these
histories; the source observer is also used by the evidence producer. The two
negative tests protect real packet custody against modified companions and parent
paths; weakening bindings makes them fail; old adapter tests had no checkpoint
packet; they use the production serializer/reader and need no test-only seam.

## Custody and delivery

The follow-up adds about 0.25 MiB: digest-bound JSON/NPZ checkpoint companions,
materialized input, two complete compressed native window records, comparison,
provenance, actual rejection, and audit summary. The duplicate 600-step source
is omitted because it equals the retained original at every trace frame. Raw,
normalized and compressed record digests are preserved. Only literal local path
prefixes were normalized; snapshot bytes, floats, events and actor IDs are intact.
Uncompressed local logs/JUnit and duplicate probe records remain outside Git.
The packet checksum manifest covers retained bytes.

PR [#10331](https://github.com/ll7/robot_sf_ll7/pull/10331) remains draft. Fix
[#10330](https://github.com/ll7/robot_sf_ll7/issues/10330) is pending review/merge
in that PR, not silently closed. Corpus/persistence integration is the external
blocker above; no author permission question or merger action is pending.

TERMINAL=BLOCKED_EXTERNAL
