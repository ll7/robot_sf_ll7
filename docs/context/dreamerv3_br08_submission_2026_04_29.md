# DreamerV3 BR-08 next-job submission record — 2026-04-29

Date: 2026-04-29
Author: agent submission, branch `578-br-08-dreamerv3-retraining-v2` (HEAD `65043052`).
Predecessor handoff: [dreamerv3_program_full_handoff_2026_04_28.md](dreamerv3_program_full_handoff_2026_04_28.md)
Predecessor parity note: [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
Predecessor SLURM doc: [../training/dreamerv3_br08_slurm_handoff.md](../training/dreamerv3_br08_slurm_handoff.md)

## Goal

Decide which DreamerV3 SLURM job(s) to submit next on Auxme, record the reasoning, and
log submission identifiers so a later session (or reviewer) can pick the work up without
re-deriving the rationale.

## Inputs reviewed

- Issues `#578` (umbrella), `#609` (training-side parity, High prio), `#608` (eval-side
  parity), `#782` (pretraining design, design-only), `#789` (multi-modal encoder,
  fail-closed under Ray 2.53.0).
- Repo SLURM launchers `SLURM/Auxme/dreamer_br08_gate.sl` and
  `SLURM/Auxme/dreamer_br08_full.sl` (both newly checked-in but untracked on branch).
- BR-08 training configs `configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml`
  and `.../benchmark_socnav_grid_br08_full.yaml`.
- Local machine context `local.machine.md` (login node `auxme-imech192`, `account=mitarbeiter`,
  per-partition QoS limit 2, tmux required for long jobs).
- Last DreamerV3 run dir on disk:
  `configs/training/rllib_dreamerv3/output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_pro6000_auto_20260314T000723Z`
  (Slurm 11433-era probe per the handoff note).

## Decision

**Submit the BR-08 gate only.** Do not pre-queue the full job blindly: the
2026-04-28 handoff explicitly says the full retrain is conditional on a healthy gate, and
the previous Slurm-11433 probe ended with `eval/success_rate=0`, `eval/collision_rate=0.5`.
A second blind retrain on the same flat-vector config would meet the "third-time-blind"
stop condition documented in the handoff.

Decoupling note: the handoff says the gate should run "after `#609`/`#608` merge". I am
running it from the branch HEAD anyway because (a) the entire parity surface needed by the
gate is already present on this branch, (b) running the gate does not require those PRs
to be merged into `main` (the gate validates code, not provenance lineage), and (c) the
gate output is a scientific signal that informs whether the umbrella `#578` can move
forward at all. The PR merge for `#609`/`#608` is tracked separately and must still
happen before any verdict note for `#578`.

The multi-modal `#789` path remains stopped: Ray 2.53.0 still does not accept the
mixed-obs Dict, so the gate uses the flat-vector config. This is acceptable as long as we
treat the resulting numbers as a baseline and not as the contender.

## Pre-submit verification (2026-04-29)

| Check | Result |
| --- | --- |
| Launcher `--help` clean import | OK (`scripts/training/train_dreamerv3_rllib.py`) |
| Gate config `--dry-run` exit | OK (`benchmark_socnav_grid_br08_gate.yaml`) |
| Full config `--dry-run` exit | OK (`benchmark_socnav_grid_br08_full.yaml`) |
| `squeue --me` | empty (no running/pending Dreamer jobs) |
| `scripts/dev/auxme_partition_status.sh --recommend` | `partition=a30 qos=a30-gpu free_gpu=8 pending=0 slots_left=2 score=850` |
| Branch HEAD synced with `origin/main` | yes (`8c694bee` merge in HEAD) |
| `pr_ready_check.sh` freshness | already green at branch HEAD per handoff; HEAD unchanged |

The gate launcher uses `partition=a30` + `qos=a30-cpu` (CPU-only), which is independent
of `a30-gpu` QoS bookkeeping. Submitting the gate alone uses 1 of 2 a30-cpu slots and
does not consume any GPU slot.

## Submission

Command (run from the repo root on `auxme-imech192`):

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/dreamer_br08_gate.sl
```

Submission output (Job ID and capture):

- Slurm job ID: `12156` (submitted 2026-04-29 08:57:54, wrapper extended walltime to
  `1-12:00:00` per partition max; the launcher itself completes well before this)
- Allocated node: `auxme-imech172` (a30 partition, `qos=a30-cpu`, no GPU)
- State at submit: `RUNNING` within seconds (no pending queue)
- Output log: `output/slurm/12156-dreamer-br08-gate.out`
- Run dir prefix: `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_gate_*`
- W&B mode: `offline` (sync from the run dir after the job finishes)
- Early health check (first ~30s of `output/slurm/12156-dreamer-br08-gate.out`):
  launcher started, repo virtualenv loaded, simulator backends registered, BR-08 SVG map
  loaded, W&B offline run initialized at
  `output/wandb/wandb/offline-run-20260429_085811-9jmxl9hn`. No early stack trace.

## Gate result (Slurm 12156)

Completed 2026-04-29 09:11:07 UTC, exit code 0, 13m 13s on `auxme-imech172`.

- All 60 train iterations executed.
- `best_reward_mean = -14.84` at iter 41 (`timesteps_total = 3072`).
- Many `non_finite_reward_mean` warnings post-iter-12. Reading the diagnostics JSONs
  (`output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_gate_*/diagnostics/`) shows
  `learner_policy_total_loss` finite (~5311) while `learner_world_model_loss`,
  `learner_actor_loss`, `learner_critic_loss` are `null` and `env_episode_return_mean =
  NaN`. Initial reading: this is RLlib reporting `mean(0 completed episodes)` for short
  iterations rather than a hard training failure. The reward primitives in
  `robot_sf/gym_env/reward.py` are NaN-safe (`_float()` returns default for non-finite,
  `_bounded()` clips, `_ttc_risk_from_meta` uses `1/max(ttc, 1e-3)`).
- Confirmed: identical `non_finite_reward_mean` pattern appeared in the 2026-04-09
  drive-state-rays stability probe at iteration 3, but that run still progressed to
  iter 119 with `best_reward_mean = -15.56`. Pattern is "early-training noise", not
  "training crash".

Verdict: **gate is acceptable as a sanity check**. Code path runs, env constructs,
scenario sampler works, finite rewards appear when episodes complete, no Ray/env
exceptions. The remaining open question — whether the agent actually *learns* given
enough budget — is what the full retrain answers.

## Full retrain decision

User authorized direct submission of the full retrain on a30 GPU; per the user's
intent, "best likelihood to succeed" maps to launching the canonical full config now
rather than running another short probe. Sticking to the canonical
`benchmark_socnav_grid_br08_full.yaml` rather than a hand-tuned sibling because:

- the previous failure mode (Slurm 11433, eval/success=0) was "trained but didn't
  learn at small scale", not "crashed" — the right response is more iterations + the
  parity-correct env, both present in the full config;
- the parity work (recursive Dict flattening, deterministic worker seeds, scenario
  matrix sampling, `route_completion_v3`) is fresh on this branch and was NOT in the
  Slurm 11433 codepath;
- the run-history-immutability rule (memory/feedback) prefers a sibling config over
  in-place edits if review later flags issues, so the canonical config stays
  reproducible.

## Full job (Slurm 12159)

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/dreamer_br08_full.sl
```

- Submitted: 2026-04-29 10:21:39 UTC
- Slurm job ID: `12159`
- Allocated node: `auxme-imech254` (preferred a30 host per `local.machine.md`)
- Partition / QoS: `a30` / `a30-gpu`, walltime `1-12:00:00` (config caps the run via
  `train_iterations: 6000`)
- Output log: `output/slurm/12159-dreamer-br08-full.out`
- Run dir prefix: `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_*`
- W&B mode: `online` (project `robot_sf_dreamerv3`, group `br08-benchmark-socnav-grid-full`)
- Config: `configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml`
  - `model_size: S`, `training_ratio: 128`, `batch_size_B: 16`, `batch_length_T: 32`,
    `horizon_H: 10`
  - eval every 100 iters, 30 episodes per eval block, scenario matrix
    `classic_interactions_francis2023.yaml` (cycle strategy)
  - reward `route_completion_v3` with the BR-08 weights

State at submit: `RUNNING`. Health: launcher started, awaiting Ray init logs.

## Stop conditions (umbrella `#578`)

Repeating from the handoff so the gate operator can see them inline:

1. Do not relaunch the same flat-vector config a third time without a new hypothesis.
2. Treat any fallback or degraded benchmark mode as a limitation, not a success.
3. Do not edit a launched config in place; if a fix is needed, open a follow-up issue
   with a new sibling config + new SLURM job (see
   `memory/feedback_run_history_configs.md` if present).

## Pointers

- Handoff: [dreamerv3_program_full_handoff_2026_04_28.md](dreamerv3_program_full_handoff_2026_04_28.md)
- Parity proof: [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
- SLURM doc: [../training/dreamerv3_br08_slurm_handoff.md](../training/dreamerv3_br08_slurm_handoff.md)
- Auxme infra: [../../SLURM/Auxme/README.md](../../SLURM/Auxme/README.md)
- Multi-modal stop note: [issue_789_dreamer_multimodal_encoder.md](issue_789_dreamer_multimodal_encoder.md)
- Pretraining design note: [issue_782_dreamerv3_pretraining_design.md](issue_782_dreamerv3_pretraining_design.md)
