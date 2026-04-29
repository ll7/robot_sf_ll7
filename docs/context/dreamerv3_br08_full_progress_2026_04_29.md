# DreamerV3 BR-08 full retrain — progress check, 2026-04-29 ~4h elapsed

Date: 2026-04-29 (snapshot at 12:33 UTC, run elapsed ~4h 11m)
Author: live progress check during user session.
Predecessor submission record: [dreamerv3_br08_submission_2026_04_29.md](dreamerv3_br08_submission_2026_04_29.md)
Predecessor handoff: [dreamerv3_program_full_handoff_2026_04_28.md](dreamerv3_program_full_handoff_2026_04_28.md)

## Subject

- Slurm job: `12159` (`d3-br08-full`)
- Node: `auxme-imech254` (a30 partition, `qos=a30-gpu`, 1× A30 24GB)
- Walltime cap: `1-12:00:00` (36h) — wrapper-extended; config nominally requests 6000 iters
- Config: [configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml](../../configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml)
- W&B: https://wandb.ai/ll7/robot_sf_dreamerv3/runs/1t6gadx5
  (project `robot_sf_dreamerv3`, group `br08-benchmark-socnav-grid-full`)
- Run dir: `output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_20260429T082210Z`
- Predecessor gate (Slurm 12156): completed cleanly 13m, exit 0, `best_reward_mean=-14.84`,
  inflight `non_finite_reward_mean` warnings later proven to be RLlib reporting empty-episode
  means — see submission record.

## Health: training is clean, NaN-free, and learning

Compared to the gate, the full run shows **no NaN drift**:

- `output/dreamerv3/.../diagnostics/` is empty (zero `iteration_*_nonfinite.json` files).
- All 52 rows in `result.jsonl` carry `reward_mean_status: finite`.
- Launcher log has no Ray exception, env construction error, or eval-block fail-closed
  trip; just standard DreamerV3-on-new-API-stack deprecation/info noise.
- W&B online sync is live (run id `1t6gadx5`).

A `best_reward` checkpoint has already been saved with full algorithm state
(`algorithm_state.pkl`, `env_runner/`, `learner_group/`, `rllib_checkpoint.json`).

## Reward trajectory — meaningful improvement

Reward statistics over the 52 reported iterations (raw `reward_mean_raw` from `result.jsonl`):

| Window | `reward_mean_raw` |
| --- | --- |
| First 10 iters | **−21.47** (random-policy floor) |
| Iters 21–30 | mostly negative, occasional positive spikes |
| Iter 31 | first positive episode mean (+5.79) |
| Iter 45 | **peak +22.90** (almost certainly route-completion `terminal_bonus`) |
| Last 10 iters | **+7.98** mean |
| All-iters min / max | **−55.08 / +22.90** |
| All-iters mean | −12.17 |

The +22.9 peak strongly implies at least some episodes are completing the route and
hitting `terminal_bonus: +20.0` from `route_completion_v3`. The shift from −21 → +8 over
~4h is consistent with healthy early-stage DreamerV3 progress — the world model is
finding signal, the actor is exploiting it.

Key contrast with prior runs:

- Gate (Slurm 12156, 13m, model_size=XS): `best_reward_mean=-14.84` at iter 41 of 60.
- 2026-04-09 stability probe (drive-state-rays, model_size=XS): `best_reward_mean=-15.56`
  at iter 119, NaN cascade started at iter 3 and never fully recovered.
- This run (Slurm 12159, model_size=S, parity-corrected env): peak +22.90 at iter 45,
  no NaN cascades, mean improving.

## Throughput — slow per-iter, walltime-bound rather than iter-bound

- 52 iters in 244 min wall ⇒ **~281 sec/iter** (~4.7 min/iter)
- Projected at this rate: **~460 iters** in the 36h walltime cap
- Configured `train_iterations: 6000` is unreachable; the run will terminate on time
- Estimated total env steps at termination: 460 iters × ~640 env steps/iter ≈
  **~294K env steps**

The throughput limit is set by `model_size: S` × `training_ratio: 128` (lots of
world-model gradient updates per env step) on a single A30. Not a problem per se;
DreamerV3 is sample-efficient by design, and 294K env steps is in the right ballpark
to learn a SocNav-class task if the reward provides enough signal.

## Eval cadence — first verdict block expected at iter 100

`evaluation: every_iterations: 100, evaluation_episodes: 30`. We are at iter 52 with
~5h until iter 100. **No eval block has fired yet.** Until iter-100 eval lands, the
training-side `reward_mean` is the only learning indicator.

## What to watch for next

In rough order of priority:

1. **Iter-100 eval block (~9h elapsed, ~14:21 UTC).** First out-of-sample signal.
   Look at `eval/success_rate`, `eval/collision_rate`, `eval/timeout_rate` in the run
   dir's `evaluation/` subdir or in W&B. **Decision rule:**
   - `eval/success_rate > 0` and improving across two eval blocks → keep running, plan
     a verdict note when walltime ends.
   - `eval/success_rate == 0` after iter 200 eval despite finite reward gains → kill
     and route to `#782` (pretraining design) instead of letting the budget burn.
2. **Reward stability past the first eval block.** A reasonable concern: large
   evaluation episode count (30) might cause the per-iter wallclock to jump. If
   sec/iter spikes by 5×+ on iter 100, projected iters drops to ~250 and the run is
   even more budget-bound.
3. **Best-checkpoint promotion.** The `best_reward` checkpoint at peak +22.9 is the
   current candidate for downstream policy-analysis. If reward keeps climbing the
   checkpoint advances on its own; if it regresses, we still have the +22.9 snapshot.

## Risks and stop conditions still in force

- Do not edit `benchmark_socnav_grid_br08_full.yaml` while 12159 is running. If the
  run reveals a config defect, open a follow-up issue with a new sibling config and a
  new SLURM job, per `memory/feedback_run_history_configs.md`.
- The `#578` umbrella's "third-time-blind" stop still stands for the *flat-vector
  architecture*. This run is the first BR-08 trial against the parity-corrected env,
  so it doesn't trigger that stop, but a second blind retrain after this one would.
- No paper-facing or benchmark-facing claim until the resulting checkpoint passes the
  PPO-style policy-analysis gate against
  `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml` (per
  `dreamerv3_program_full_handoff_2026_04_28.md` step 3 of #578).

## Convenience commands

```bash
# Live status
squeue -j 12159 --format='%i %j %T %P %M %l %S %R'
tail -f output/slurm/12159-dreamer-br08-full.out

# Reward trend snapshot
uv run --extra dev python - <<'PY'
import json, pathlib, statistics
p = pathlib.Path("output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_20260429T082210Z/result.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
raws = [r['reward_mean_raw'] for r in rows if r.get('reward_mean_raw') is not None]
print(f"iters={len(raws)} mean={statistics.mean(raws):.2f} "
      f"first10={statistics.mean(raws[:10]):.2f} last10={statistics.mean(raws[-10:]):.2f} "
      f"min={min(raws):.2f} max={max(raws):.2f}")
PY

# Eval block presence (after iter 100)
ls output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_20260429T082210Z/evaluation 2>/dev/null
```

## Pointers

- Submission record: [dreamerv3_br08_submission_2026_04_29.md](dreamerv3_br08_submission_2026_04_29.md)
- Handoff: [dreamerv3_program_full_handoff_2026_04_28.md](dreamerv3_program_full_handoff_2026_04_28.md)
- Parity proof: [issue_578_608_609_dreamerv3_parity.md](issue_578_608_609_dreamerv3_parity.md)
- SLURM doc: [../training/dreamerv3_br08_slurm_handoff.md](../training/dreamerv3_br08_slurm_handoff.md)
- Multi-modal stop: [issue_789_dreamer_multimodal_encoder.md](issue_789_dreamer_multimodal_encoder.md)
- Pretraining design: [issue_782_dreamerv3_pretraining_design.md](issue_782_dreamerv3_pretraining_design.md)
