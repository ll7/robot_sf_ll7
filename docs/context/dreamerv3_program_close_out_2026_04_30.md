# DreamerV3 program close-out verdict, 2026-04-30

Date: 2026-04-30
Author: program-level kill decision after the third blind BR-08 retrain
(Slurm `12159`) failed.
Status: **CLOSED — DreamerV3 track deprioritized; flat-vector BR-08 retired**.

This note is the single program-level record explaining why the DreamerV3
umbrella issue and its open dependents are being closed without a
paper-facing checkpoint.

## Issues closed by this verdict

| Issue | Title | Reason for closure |
| --- | --- | --- |
| `#578` | BR-08: DreamerV3 retraining v2 (umbrella) | Third-time-blind stop triggered; flat-vector BR-08 retired. |
| `#789` | DreamerV3 multi-modal encoder — occupancy grid as 2D image | Architectural pivot dropped: program no longer worth the additional engineering investment vs. opportunity cost. |
| `#782` | Design DreamerV3 world-model pretraining path | Pretraining only earns its keep if downstream RL is the bottleneck the program is still trying to clear. It isn't. |
| `#609` | Add scenario-matrix parity to DreamerV3 training | Parity work *was* delivered and verified; the closing run still failed for unrelated reasons (NaN + RAM). The follow-up retrain it enabled is the artifact being cancelled, not the parity work. |
| `#608` | Add scenario-matrix parity to DreamerV3 training/evaluation pipeline | Same as `#609`; superseded by the program-level decision to stop. |

## Evidence summary

- 2026-04-09 stability probe (drive-state-rays, model_size=XS): NaN-from-iter-3,
  no eval signal, `best_reward_mean = -15.56`.
- 2026-04-29 gate (Slurm `12156`, 13m, model_size=XS): NaN-from-iter-12,
  `best_reward_mean = -14.84`, no eval block fired.
- 2026-04-29 full (Slurm `12159`, model_size=S, parity-corrected env):
  - `sacct State = OUT_OF_MEMORY`, exit `0:125`, elapsed 23h20m,
    `MaxRSS = 106 GB` against `ReqMem = 64 GB` (host RAM leak).
  - 585 train iters / 177,088 env steps; **first NaN at iter 207**, 164
    `iteration_*_nonfinite.json` diagnostics, peak `reward_mean_raw = +23.96`
    at iter 453.
  - **No eval block fired.** Eval cadence was `every_iterations: 100`,
    but every eval iter on or after 207 was inside a NaN window.
  - W&B run: <https://wandb.ai/ll7/robot_sf_dreamerv3/runs/1t6gadx5>.

Per-run detail: [dreamerv3_br08_full_progress_2026_04_29.md](dreamerv3_br08_full_progress_2026_04_29.md).

## Verdict — why we are stopping the program

Two independent failure modes hit the same run, and the cost of disambiguating
them exceeds the expected value of an additional flat-vector BR-08 attempt.

### 1. The cost / signal ratio is bad and getting worse

- Three attempts, **zero eval blocks**. No out-of-sample number has ever
  been produced for a DreamerV3 BR-08 policy on this branch. ~24 GPU-hours
  on the closing run alone with no paper-facing artifact.
- Every flat-vector BR-08 run shows the NaN pattern (XS at iter 3, XS at
  iter 12, S parity-corrected at iter 207). The parity work delayed the
  onset but did not eliminate it. That is structural, not a config bug.
- PPO already wins this benchmark. Distribution alignment dominates the
  PPO lift to ~0.929 success on the broad scenario matrix
  (memory: `experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`).
  DreamerV3 is not displacing PPO in the camera-ready paper even on a
  best-case run; it would be a comparison data-point at best.

### 2. The two failure modes compound badly

- **NaN cascade alone** is debuggable but not cheap: the 164
  `iteration_*_nonfinite.json` files would need to be walked to find which
  reward component or env tensor goes inf/NaN first. Likely week-of-grinding
  cost.
- **Host RAM leak alone** is genuinely hard: 64 → 106 GB over 23h on
  Ray + RLlib new-API-stack + DreamerV3 + custom envs is a known-fragile
  upstream combination. Pinning the leak requires heap profiling across
  Ray actor boundaries with uncertain payoff.
- **Both at once on the same run** prevents clean bisection: did the NaN
  retain dead tensors in the replay buffer (NaN → leak), or did memory
  pressure starve workers and corrupt batches (leak → NaN)? Disambiguating
  costs another full run.

### 3. The umbrella's third-time-blind stop has triggered

The `#578` umbrella encoded a "third blind retrain → stop" rule precisely so
that compute would not keep being spent on attempts that produce no eval
signal. That rule has now triggered cleanly (probe → gate → full all blind).
Defer to it.

## What is and is not being abandoned

**Abandoned (this verdict):**

- Further compute on flat-vector BR-08 DreamerV3 retrains.
- The multi-modal encoder pivot in `#789`. The architectural argument is
  sound, but the program around it is no longer worth the engineering and
  compute it would consume.
- The world-model pretraining design in `#782`. Pretraining only helps if
  downstream RL is the binding constraint. With PPO already strong and
  DreamerV3 deprioritized, it is not.
- The follow-up retrains that `#608` / `#609` were intended to enable.

**Kept on disk for reference:**

- Run dir
  `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_20260429T082210Z`
  (result.jsonl, diagnostics, `checkpoints/best_reward` at iter 453).
  Diagnostic only — never passed an eval block, lives downstream of an
  OOM-killed run, **not paper-facing**.
- W&B run `1t6gadx5` is the cheapest artifact for any future post-mortem.
- Parity work delivered for `#608` / `#609` (recursive Dict flattening,
  deterministic worker seeds, scenario-matrix parity hooks) stays merged on
  `main`. Closing those issues does **not** revert the parity code; it
  only retires the BR-08 retrain that the parity work was meant to enable.

## What we are doing instead

In rough yield-per-hour order:

1. Saved compute is redirected to the PPO scenario matrix that is already
   producing paper-facing numbers (issue `#791` and dependents).
2. DreamerV3 is removed from the camera-ready scope; if it is mentioned at
   all in the manuscript, it is described as an attempted baseline that
   did not converge under our resource budget, with the W&B run linked as
   evidence rather than a success claim.
3. If the DreamerV3 track is ever revisited, it must start from a
   structurally different setup — at minimum, a small repro
   (1 env, model_size=XS, training_ratio=32) used to bound the host RAM
   leak inside two days *before* any compute is spent on a full retrain.
   Without that bound, do not resubmit.

## Pointers

- Final run record: [dreamerv3_br08_full_progress_2026_04_29.md](dreamerv3_br08_full_progress_2026_04_29.md)
- Submission record: [dreamerv3_br08_submission_2026_04_29.md](dreamerv3_br08_submission_2026_04_29.md)
- Program handoff that this note supersedes: [dreamerv3_program_full_handoff_2026_04_28.md](dreamerv3_program_full_handoff_2026_04_28.md)
- Parity proof (kept): [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
- Multi-modal encoder note (closed): [issue_789_dreamer_multimodal_encoder.md](issue_789_dreamer_multimodal_encoder.md)
- Pretraining design note (closed): [issue_782_dreamerv3_pretraining_design.md](issue_782_dreamerv3_pretraining_design.md)
- Consolidated plan that this note supersedes: [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
