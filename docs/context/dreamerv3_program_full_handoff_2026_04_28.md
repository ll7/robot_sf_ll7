# DreamerV3 program full handoff — issues #578, #789, #782, #609, #608

Date: 2026-04-28
Author: handoff prepared from `578-br-08-dreamerv3-retraining-v2` HEAD (merge `8c694bee` of main).
Predecessor note: [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
Predecessor SLURM doc: [../training/dreamerv3_br08_slurm_handoff.md](../training/dreamerv3_br08_slurm_handoff.md)

This note is the consolidated execution plan for **completing** the five DreamerV3 issues
in one coherent pass. It is intentionally explicit about what is already done, what work
remains, and what SLURM jobs to submit. Treat it as the single entry point until each
issue closes.

> Update 2026-04-28: this repo now includes the checked-in BR-08 SLURM launchers,
> a targeted eval-parity regression test slice, the #782 pretraining design note,
> and a fail-closed #789 stop note. The multimodal Dreamer path did not proceed on
> this branch because Ray 2.53.0 DreamerV3 still lacks mixed-observation support.

## TL;DR sequencing

| Order | Issue | Type | Blocker | Action |
| --- | --- | --- | --- | --- |
| 1 | #609 | impl | - | Open PR from `578-br-08-dreamerv3-retraining-v2` to `main`; landing the parity surface. |
| 2 | #608 | impl | #609 merged | Verify + harden eval-side parity already wired in the same branch; PR if extra glue is needed. |
| 3 | #782 | design-only | #609 merged | Author the world-model pretraining design note in `docs/context/`. No SLURM. |
| 4a | #789 | impl + research | independent of #608/#578 | Wire RLlib DreamerV3 multi-modal encoder; gate run only after local unit-test green. |
| 4b | #578 | umbrella + run | #609 + #608 merged; #789 either landed or explicitly deferred | Run BR-08 gate, then BR-08 full, write verdict. |

#789 (4a) and #578 (4b) can run in parallel once #609/#608 merge: 4a is a code change
(local + 1× short SLURM gate); 4b is an evaluation campaign (1× long SLURM full job). If
#789 lands during the #578 gate, switch the #578 full run to the multi-modal config; if
#789 fails closed (RLlib too rigid), document it and run #578 on the flat-vector path one
last time as a baseline before deciding whether to spend more compute on Dreamer.

## Current state per issue

### #609 — Scenario-matrix parity (training side)
**Status:** functionally landed on `578-br-08-dreamerv3-retraining-v2` (not yet on `main`).
Existing parity note `docs/context/issue_578_608_609_dreamerv3_parity.md` records:
- scenario-matrix sampling with deterministic worker seeds
  (`robot_sf/training/scenario_sampling.py`)
- recursive Dict observation flattening
  (`robot_sf/training/rllib_env_wrappers.py`)
- gate / full BR-08 configs
  ([benchmark_socnav_grid_br08_gate.yaml](../../configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml),
  [benchmark_socnav_grid_br08_full.yaml](../../configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml))
- 39 passing tests across `test_rllib_env_wrappers.py`, `test_scenario_sampling.py`,
  `test_train_dreamerv3_rllib_config.py`, `test_train_dreamerv3_rllib_runtime.py`
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` already green on the branch as of
  the parity note.

**Remaining work (small):**
1. Sync the branch with `origin/main` (already done at commit `8c694bee`); rerun
   `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` to confirm freshness.
2. Open PR `578-br-08-dreamerv3-retraining-v2` → `main` titled
   `feat: DreamerV3 BR-08 scenario-matrix parity (#609, #608, #578-infra)`.
3. **Do not** cherry-pick from the stale `codex/609-dreamer-scenario-matrix-parity`
   branch; that branch is ~96k lines behind main and its useful Dreamer-specific work
   (balanced full config, GPU smoke diagnostics, pro6000 SLURM wrapper) should be
   reviewed and re-applied as small additive PRs after the main parity PR lands, not
   merged wholesale.

**SLURM jobs:** none for the merge itself. CI gate is `scripts/dev/pr_ready_check.sh`.

### #608 — Scenario-matrix parity (evaluation side)
**Status:** infrastructure mostly in place via the same branch's "periodic cycle-order
evaluation" surface (gate config: eval block defined but disabled by default; full
config: `evaluation.enabled: true` every 100 training iterations). A targeted runtime
parity test slice now exists in `tests/training/test_train_dreamerv3_rllib_runtime.py`,
but still needs fresh execution on CI/compute.

**Remaining work:**
1. Confirm the eval surface uses the same scenario-matrix sampler as PPO's benchmark
   eval (check `_make_env_creator` honours the `evaluation.scenario_config` block and
   that the resolved seeds match the PPO `evaluation.randomize_seeds: false` contract
   from `configs/training/ppo/expert_ppo_*` configs).
2. Add a targeted parity test if missing: load the BR-08 full config, instantiate the
   eval env, assert the observation/action spaces match the PPO benchmark eval env on
   the classic-interactions matrix. Add to `tests/training/test_train_dreamerv3_rllib_runtime.py`.
3. Update `docs/training/dreamerv3_rllib_drive_state_rays.md` to drop the "not yet
   full scenario-matrix parity" caveat once the parity test passes.

**SLURM jobs:** none. Eval-parity changes are validated by unit tests + a `--dry-run` of
the full config.

### #782 — World-model pretraining design note
**Status:** authored in `docs/context/issue_782_dreamerv3_pretraining_design.md`.
Issue remains design-only (Effort: 4h, no code).

**Remaining work:**
1. Author `docs/context/issue_782_dreamerv3_pretraining_design.md` with:
   - Inventory of reusable rollout sources in this repo: PPO checkpoints under
     `output/model_cache/ppo_*`, ORCA rollouts (if any are checked in), scripted
     planners under `robot_sf/nav/`. For each, list observation/action/reward
     fields and storage format.
   - Compare Options A (RLlib offline-rollout ingestion), B (world-model
     weight export/import adapter), C (external representation model) against
     the proof-first/fail-closed constraints in `AGENTS.md`.
   - Recommend Option B with the caveat from the issue: stop and convert to a
     follow-up if RLlib does not expose a clean checkpoint/import boundary.
   - Define one minimal gate experiment (e.g. import a Dreamer checkpoint from a
     successful run, fine-tune for 100k steps on the BR-08 gate scenario, compare
     against from-scratch gate) and one explicit stop condition (no measurable
     improvement in `eval/success_rate` after the fine-tune budget → no-action
     decision).
2. End the note with one of: a follow-up implementation issue or a clear no-action
   recommendation. Link from the parity note and from #782.

**SLURM jobs:** none.

### #789 — Multi-modal encoder for DreamerV3
**Status:** investigation completed; stop condition met on the current dependency stack.
See `docs/context/issue_789_dreamer_multimodal_encoder.md`.

**Remaining work:**
1. If maintainers still want multimodal DreamerV3, open a separate follow-up issue for
   either an upstream RLlib contribution or a repo-local Dreamer catalog/module fork.
2. Do not add multimodal wrapper/config work to #578/#608/#609 while the current Ray
   catalog still expects a single `Box` observation space.

**SLURM jobs:** none on this branch for the multimodal path; the stop condition was met
before a dedicated gate run was justified.

### #578 — BR-08 retraining v2 (umbrella)
**Status:** infra ready on the 578 branch. The previous probe (Slurm 11433) ran
end-to-end but produced `eval/success_rate = 0`, `eval/collision_rate = 0.5`. Per the
2026-04-10 comment, "BR-08 Dreamer is now infra-complete but not yet a competitive
checkpoint path."

**Remaining work:**
1. After #609/#608 merge **and** #789 either lands or explicitly fails closed, run the
   BR-08 gate (1× short SLURM job).
2. If gate is healthy (improving reward, decreasing collision/timeout trends), submit
   the BR-08 full (1× long SLURM job).
3. Externally evaluate the resulting best checkpoint with the PPO-style policy-analysis
   gate before any paper-facing claim. Re-use the eval-aligned scenarios from
   `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml` for direct comparability.
4. Write the verdict note in `docs/context/issue_578_br08_dreamerv3_v2_verdict_<date>.md`
   with the W&B run URL, run directory, eval numbers, and decision: promote, retain as
   negative result, or escalate to #782's pretraining path.
5. **Stop condition:** do not relaunch the same config a third time without a new
   hypothesis. If #789 didn't land and the gate also stays poor, close #578 with a
   negative-result verdict pointing at #782.

**SLURM jobs:** 1× gate (BR-08, ~6h CPU); 1× full (BR-08, ~24h GPU). See below.

## SLURM jobs to submit

All jobs follow the repo's Auxme convention. Submit via the canonical wrapper so the
log directory is created and max-time is honored:

```bash
scripts/dev/sbatch_use_max_time.sh <SLURM/Auxme/dreamer_*.sl>
```

Pre-submit pressure check (recommended for a30/l40s/pro6000):

```bash
scripts/dev/auxme_partition_status.sh --recommend
```

Reference: [`SLURM/Auxme/README.md`](../../SLURM/Auxme/README.md),
[`docs/training/dreamerv3_br08_slurm_handoff.md`](../training/dreamerv3_br08_slurm_handoff.md).

The job templates now exist as `SLURM/Auxme/dreamer_br08_gate.sl` and
`SLURM/Auxme/dreamer_br08_full.sl`. Pattern reference remains
`SLURM/Auxme/issue_791_attention_head.sl` for future Dreamer-adjacent launchers.

### Job 1 — `dreamer_br08_gate.sl` (#578 / #789 gate)

Purpose: catch construction, Ray, observation, and reward-contract failures before any
GPU spend. CPU-only, offline W&B, ~6h cap. Use this for the flat-vector BR-08 path.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-dreamer-br08-gate
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=output/slurm/%j-dreamer-br08-gate.out

set -euo pipefail
mkdir -p output/slurm

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONWARNINGS=ignore

# Override DREAMER_BR08_CONFIG only when intentionally probing a different committed
# Dreamer config sibling.
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml \
  --log-level WARNING
```

Submit:

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/dreamer_br08_gate.sl
```

Gate decision:
- continue if no env/Ray failures **and** training metrics are not obviously stalled or
  degenerate;
- stop and inspect otherwise.

### Job 2 — `dreamer_br08_full.sl` (#578 full retrain)

Purpose: BR-08 full-budget retrain. GPU, online W&B, ~24h cap. Submit only after Job 1
is healthy.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-dreamer-br08-full
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --output=output/slurm/%j-dreamer-br08-full.out

set -euo pipefail
mkdir -p output/slurm

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONWARNINGS=ignore

# Switch the --config path to the multi-modal sibling iff #789 landed and Job 1
# (multi-modal gate) was healthy.
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml \
  --log-level WARNING
```

Submit:

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/dreamer_br08_full.sl
```

Full-run stop conditions:
- `result.jsonl` stops updating for >30min while job is allocated → check Ray status,
  inspect non-finite metric counters, decide kill vs. wait;
- periodic eval stays near-zero success with high collision/timeout after multiple
  evaluation intervals → kill and pivot to #782.

### No SLURM job for #608, #609, #782

- #609 / #608 land via PR + `pr_ready_check.sh` only.
- #782 is design-only.

## Validation gates per issue

| Issue | Local validation | SLURM validation | Documentation gate |
| --- | --- | --- | --- |
| #609 | `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` green; 39+ DreamerV3 unit/runtime tests pass | none | parity note + caveat dropped from `dreamerv3_rllib_drive_state_rays.md` |
| #608 | new parity test green (eval env shape matches PPO benchmark eval); full-config `--dry-run` exit 0 | none | same as #609 |
| #782 | none (design note only) | none | new note `issue_782_dreamerv3_pretraining_design.md` checked in, linked from #782 + parity note |
| #789 | encoder forward-pass unit test green; multi-modal gate `--dry-run` exit 0 | Job 1 (multi-modal variant) green | new contract note `issue_789_dreamer_multimodal_encoder.md` |
| #578 | `--dry-run` of gate + full configs exit 0 | Job 1 (flat-vector) green; Job 2 completes with parseable `result.jsonl` and `evaluation/*.jsonl` | verdict note `issue_578_br08_dreamerv3_v2_verdict_<date>.md` |

## Risks and decision points

1. **#789 RLlib rigidity (Option A failure).** If RLlib DreamerV3 does not natively
   accept a Dict-obs CNN+MLP encoder, fail closed *inside* #789 and convert to a
   follow-up design issue. Do not slide into a fork in this PR.
2. **#578 third-time-blind risk.** Slurm 11433 already showed the flat-vector path
   does not converge on the current reward. A second blind retrain on the same config
   is hard to justify. Either land #789 first **or** be explicit in the verdict note
   that this is a baseline negative-result run, not a serious challenger attempt.
3. **#609 stale branch trap.** `codex/609-dreamer-scenario-matrix-parity` is ~96k
   lines behind main; do not merge it directly. The parity surface that issue asks for
   is already on `578-br-08-dreamerv3-retraining-v2`.
4. **Run-history immutability** (per [feedback](../../memory/feedback_run_history_configs.md)):
   if PR review flags an issue with a config that has already been launched, do not
   edit the config in place — open a follow-up issue with a new sibling config + new
   SLURM job.
5. **Camera-ready scope.** None of these five issues should make paper-facing claims
   without external policy-analysis evaluation. The benchmark contract in
   `docs/code_review.md` and the proof-first policy in `AGENTS.md` apply.

## Pointers

- Predecessor parity note: [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
- Predecessor SLURM doc: [../training/dreamerv3_br08_slurm_handoff.md](../training/dreamerv3_br08_slurm_handoff.md)
- Auxme infra: [../../SLURM/Auxme/README.md](../../SLURM/Auxme/README.md)
- Issue list: #578 (umbrella), #609 (training parity), #608 (eval parity),
  #789 (multi-modal encoder), #782 (pretraining design)
- Related infra issue from the parallel PR triage: #852 (single-factor ablations,
  seed-fixed replicas, portable baseline — does not block any of the five Dreamer
  issues but uses the same Auxme conventions)
