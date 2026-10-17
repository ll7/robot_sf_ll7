# One-case positive control: blocked at exact window replay

Development-only simulator smoke evidence. No corpus admission, release result,
nominal benchmark comparison, planner-improvement or real-world safety claim.
The source case is a near-miss with a subsequent timeout, rather than a
timeout-only positive control. No frozen artifacts were changed.

Repository: `ll7/robot_sf_ll7`. Source main at execution:
`53721be3248600e08641a9816d094b605e3c93e4` (2026-10-12).
Owner: generated-replay/simulator maintainers. Blocker:
[#10330](https://github.com/ll7/robot_sf_ll7/issues/10330).

## Stage results and exact commands

The single recipe runs stages in order and stops before persistence/admission.
It was executed from the pinned source checkout with source-owned tracked files
unchanged. The recipe verifies both conditions before simulation:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  docs/context/evidence/hardcase_positive_control/reproduce.py \
  --output ../recipe_final
```

Observed exit **2**, the expected explicit parity failure. `comparison.json`
contains the measurements. Commands below describe stage operations inside that
exact invocation; no alternate replay implementation was substituted.

| Stage | Acceptance | Exact operation in recipe | Evidence |
| --- | --- | --- | --- |
| Source scenario/planner/config provenance | PASS | `load_scenarios(MATRIX)`; select `classic_cross_trap_high`, replace `seeds` with `[1001]`; `run_map_batch(..., algo="social_force", algo_config_path=CONFIG, horizon=600, dt=0.1, record_forces=True, record_simulation_step_trace=True, benchmark_profile="experimental", workers=1, resume=False)` | Full commit, all included YAML/map/lock SHA-256 values, effective scenario and config identities in `provenance.json`. |
| Known reproducible critical source | PASS | Two consecutive source `run_map_batch` calls; `find_first_trace_difference(source_steps, repeat_steps)` | All 600 steps match exactly; same exact events and near-miss flag; trace digest below. |
| Detailed safety/event trace | PASS | Same source calls with simulation-step recording | Complete source record in `source.jsonl.gz`; typed event ledger, velocities, actor IDs, commands and step collision flags. |
| Critical-window extraction | PASS | `extract_critical_segment(..., pre_margin_s=1, post_margin_s=1)` | `entry.json`: 21 sampled boundary/transition frames, 37.0–39.0 s, minimum at 38.0 s. |
| Replay materialization/load/execution | PASS, `loads_only` | `materialize_generated_scenario(entry, max_episode_steps=20)`; `dump_generated_scenario_yaml`; two `run_map_batch(window_yaml, ..., horizon=20, ...)` calls | `window.yaml`; both replay jobs available with declared adapter readiness; each writes one step. Loading is the only supported adapter verdict. |
| Exact replay parity, twice | **FAIL** | Compare subsequent 20 source steps against each replay, using 1e-9 m trajectory tolerance and exact first-event/outcome comparisons | Both replays miss the near-miss, terminate early, and diverge on the first position. |
| Persistence gate | FAIL — not reached | **NOT RUN**: first failing stage stops the chain | No perturbation-grid evidence or promotion receipt was fabricated. |
| Corpus admission through existing gates | FAIL — not reached | **NOT RUN**: first failing stage stops the chain | No case was admitted; no manual-review flags were promoted. |

For the two unrun stages, FAIL means requested acceptance is unmet; it does not
claim that a downstream gate was executed and returned a rejection.

## Source case and provenance

The release planner uses `configs/algos/social_force_release_v0_0_8.yaml` on
`configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`.
Only episode seed **1001** was executed. The effective source scenario keeps
the original 0.0.8 geometry, population and simulation overrides, with a
600-step budget at 0.1 s. The selected map digest is
`d3bc9ad6205108a480469338282611730410787cc897abd4d71f3f0fbd47d6a8`.

Planner-config file SHA-256:
`9a7015a5f66b5d9ddb5d96c706d33ac8fe027bc0aee256e9ab87d2e89804f773`.
Effective planner hash: `a4913be8bee6f3f3`; run config hash:
`f330dd29077077ac`. There is no learned checkpoint for this analytic planner:
model digest is explicitly not applicable. Its three-file analytic implementation
digest is `fe4e5b99c0a9f51ada80cfccbf1361bb4f1872c0c523e2b35ce89e94e7b32958`;
the full source commit pins its dependencies. `provenance.json` also records
the Python/package identities and file-level input hashes.

Execution is registry-declared **adapter**, readiness **adapter**, availability
**available**, algorithm status **ok**, with no degraded execution in the
integrity receipt. This is the released SocialForcePlannerAdapter contract,
not native command execution and not optimizer-facing native eligibility.

The source has 11 near-miss **samples**, zero contacts, and minimum surface
clearance **0.47727323760374385 m**. The predicate is
`0 <= clearance < 0.5 m`, with robot radius 1.0 m and pedestrian metric/contact
radius 0.4 m. First near-miss: zero-based step **374** (one-based transition
375), time **37.5 s**. Minimum: step **379**, time **38.0 s**. Source exact outcome:
collision false, route complete false, timeout true; 600 recorded steps.

The extractor's field named `min_clearance_m` is a **center distance**:
1.8772732376037438 m. The benchmark's surface clearance subtracts 1.4 m.
The extraction still selects the same minimum frame because radii are constant.
No center-distance value is used as a surface-clearance safety threshold.

Source and independent rerun complete step digest:
`7a73ee858ba8248969015c4f31273c014d15fd328bd97671876f9520c2bf2b34`.
Source-template reproducibility does not establish generated-window parity.

## First failing stage and diagnosis

The boundary state is the recorded post-step state at 37.0 s (source step 369).
Expected replay transitions are source steps 370–389; the first near-miss should
therefore be local step **4**. Both consecutive generated-window replays instead
record only local step 0, with route complete true, timeout false, collision false,
zero near-misses and minimum surface clearance **0.7095036862680422 m**.
Both replay step digests are
`8806cbcc2b66d30773ef24b62d64deaaf5d4c43f5ddc8684fb99889da6965d65`.
They reproduce the same wrong continuation.

The first paired robot-position error is **0.03689150220919005 m**, versus
the declared **1e-9 m** tolerance. Pedestrian slot-position error on that one
paired frame is **0.15037364817541385 m**. These are errors on the sole comparable
step, not maxima over a completed 20-step replay. Actor bindings are also
unverified after the adapter reindexes the extracted position lists.

The code and reset evidence identify the cause:

- `segment_extraction._state` discards heading, velocity and actor identity.
- `replay_adapter.materialize_generated_scenario` substitutes the last window
  position for the original robot goal, makes new pedestrian routes, and emits
  budget/density overrides without preserving the source simulation contract.
- The substituted endpoint is only **0.8365615368996463 m** from the boundary
  position. The fresh default waypoint-radius completion uses robot radius plus
  goal radius (2.0 m); the source used `goal_zone_entry_v1`. The observed replay
  completes after one step.
- The source boundary robot velocity is `[0.2101007567407187,
  0.2801191720361018]`; replay reset velocity is `[0,0]`. Pedestrian 0 changes
  from `[0.42380256188153,0.25418457019458884]` to `[0.5,0]`.
- `apply_generated_replay_runtime` constructs placements/routes, rather than
  restoring robot/controller, pedestrian navigator, desired-speed and RNG state.
  Existing typed snapshots explicitly leave controller, sensor history, metrics
  and recurrent planner state unsupported and are conditional continuations.

This is a missing continuation contract, not a one-line numerical tolerance
problem. Changing the goal alone cannot restore dynamic state. Production code
was therefore left unchanged and the blocker was filed instead of broadening
this bounded demonstration into a simulator checkpoint redesign.

Revival condition: a lossless supported continuation adapter retains original
goal/termination semantics and required runtime state, or fails closed on absent
state. Re-run this same dev-seed case with two exact window replays, then execute
the existing persistence and corpus gates with real receipts. Owner and required
regression-test behavior are recorded in #10330. Keep that issue open.

## Artifact custody and reproduction

This curated one-case packet is approximately 1 MiB. The complete source record
and two replay records are compressed because they are the concrete regression
fixture needed to inspect the failure. Absolute checkout/lane path prefixes were
replaced with `<repository_root>` / `<lane_root>` **only**; numeric values, events
and trace payloads were not rewritten. Raw, normalized and compressed digests
are recorded in `provenance.json`. The duplicate source rerun trace is omitted;
its independent digest and event comparison receipt are preserved. Raw companion
logs and uncompressed records remain local and are not required to inspect the
committed source/replay records.

After obtaining this PR's evidence, run the recipe against a separate source
checkout at the pinned commit (the evidence recipe itself is outside that source
commit). From an evidence checkout:

```bash
cp docs/context/evidence/hardcase_positive_control/reproduce.py ../hardcase_reproduce.py
git worktree add --detach ../hardcase-source 53721be3248600e08641a9816d094b605e3c93e4
cd ../hardcase-source
scripts/dev/bootstrap_worktree.sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python ../hardcase_reproduce.py \
  --output ../hardcase-reproduced
```

Expected exit 2 and `comparison.json` parity FAIL. This executes only four bounded
episodes at dev seed 1001: two sources and two generated-window replays. It
performs no corpus admission or search campaign.

Acquisition and recipe verification together executed 12 small local episodes,
all at seed 1001, totaling 3,606 transitions and 67.27 s of summed recorded
episode wall time. These are repeated checks of this single case, with no new
scenario/seed search. The full benchmark audit below excludes simulator entry.

## Validation and delivery

Production modules, tests, frozen configs/maps/models and release artifacts are
unchanged. This PR adds the evidence recipe/packet and a context-index link.
There are no changed production modules requiring an additional module-test list.
The required full directory command was named before the PR:

```bash
bash ../run_benchmark_check.sh
```

It expands to `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest
tests/benchmark -q --tb=short --continue-on-collection-errors
--junitxml=../benchmark.xml`, with the task simulator interlock and repository
seed guard propagated to children. The interlock excludes actual simulator entry
from this audit; static/mocked tests still execute. No skipped test is counted as
passing simulation evidence. The guard, exact command and final counts are
preserved in `validation.json` and its companion files.

The one full-directory attempt completed in 25m09s with exit **1**:
**12,116 passed, 119 failed, 336 skipped, 13 collection errors**. Of the skips,
326 were the task simulator interlock. Missing optional packages also affected
collection/execution. This is not a clean full-suite pass; failure IDs and first
exception lines are preserved in `validation.json`. No unrestricted baseline
comparison was attempted, so failures are not classified as new regressions.

Final test results, lint/format, artifact verification and the unmerged PR identity
are recorded in `validation.json` and the lane handoff report. The canonical broad
PR readiness script was not run: this is a diagnostic-only evidence packet with
no semantic change, and the full-directory check is reported without a green
readiness claim.

TERMINAL=BLOCKED_EXTERNAL
