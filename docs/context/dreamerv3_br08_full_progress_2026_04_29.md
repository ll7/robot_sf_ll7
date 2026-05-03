# DreamerV3 BR-08 full retrain — outcome record, 2026-04-29 / 2026-04-30

Date: 2026-04-29 (run start) → 2026-04-30 (slurm OOM kill, retrospective updated 2026-04-30)
Author: live progress check during user session, then post-mortem update once the run terminated.
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

## Final outcome — FAILED (OUT_OF_MEMORY + NaN reward collapse)

`sacct` verdict for `12159`:

| Field | Value |
| --- | --- |
| State | `OUT_OF_MEMORY` |
| Exit code | `0:125` (slurm OOM kill, batch step) |
| Elapsed | 23:20:44 |
| `MaxRSS` | **106.27 GB** |
| `ReqMem` | 64 GB |

Two independent failure modes hit the same run:

1. **Host RAM leak.** `MaxRSS = 106 GB` against a 64 GB cgroup. Slurm killed the
   batch step; immediately before, 20 Ray worker `oom_kill` events are logged in
   [output/slurm/12159-dreamer-br08-full.out](../../output/slurm/12159-dreamer-br08-full.out)
   and the env runners died en masse. The terminal Python traceback
   (`ZeroDivisionError: division by zero` in
   `ray/rllib/algorithms/dreamerv3/dreamerv3.py:591` —
   `replayed_steps_this_iter / env_steps_last_regular_sample`) is a downstream
   symptom: with all env runners OOM-killed, `env_steps_last_regular_sample = 0`.
2. **Reward NaN cascade re-emerged.** Despite the parity-corrected env, the same
   non-finite reward pattern that doomed the 2026-04-09 stability probe and the
   2026-04-29 gate reappeared in the full run.

Final-state numbers from
`output/dreamerv3/dreamerv3_br08_benchmark_socnav_grid_full_20260429T082210Z`:

- `result.jsonl`: 585 rows (iters 1–585), final `timesteps_total = 177,088`
- `diagnostics/`: **164** `iteration_*_nonfinite.json` files
- First nonfinite iter: **207**; last nonfinite iter: **585**
- Peak `reward_mean_raw`: **+23.96 at iter 453** (saved to `checkpoints/best_reward`)
- Final iter 585 `reward_mean_raw = NaN`, `reward_mean_status = nonfinite`
- ~470/585 iters are finite-reward; the run was learning, then drifted into
  intermittent NaN spikes from iter 207 onward, never recovering to a clean regime.

The run did *not* hit walltime; it was killed at ~23h20m by the OOM cgroup,
well short of the projected ~36h budget.

## Trajectory contrast (full vs gate vs prior probe)

| Run | Model size | Best `reward_mean` | NaN onset | Terminated by |
| --- | --- | --- | --- | --- |
| 2026-04-09 stability probe (drive-state-rays) | XS | −15.56 @ iter 119 | iter 3 | clean exit, no learning |
| 2026-04-29 gate (12156, 13m) | XS | −14.84 @ iter 41 | iter 12 | clean exit at 60 iters |
| **2026-04-29 full (12159)** | **S** | **+23.96 @ iter 453** | **iter 207** | **OOM kill at iter 585** |

The full run *did* clear the gate's reward floor (positive-mean episodes,
+24 peak strongly suggests `terminal_bonus` route completions) but never
escaped the NaN regime once it started, and bled host memory until slurm
killed the cgroup. **No eval block fired** — eval cadence was every 100
iters, but every eval iter on or after iter 207 was inside a NaN window or
host RAM was already past 64 GB. The `evaluation/` subdir contains no
verdict block.

## "Third-time-blind" stop is now triggered

Per the umbrella `#578` stop rule, this is the third blind BR-08 retrain
that produced no usable eval signal:

1. 2026-04-09 stability probe — flat-vector, NaN-from-iter-3, no eval signal.
2. 2026-04-29 gate (12156) — flat-vector, NaN-from-iter-12, gate reward only.
3. 2026-04-29 full (12159) — flat-vector parity-corrected, NaN-from-iter-207
   *plus* OOM, no eval block, OOM-killed before walltime.

The flat-vector BR-08 architecture should not get another compute slot
without (a) a host-RAM leak diagnosis, and (b) a structural change to the
reward / observation pipeline that explains why the NaN regime keeps
re-emerging. Pivot options remain
[issue_789_dreamer_multimodal_encoder.md](issue_789_dreamer_multimodal_encoder.md)
and [issue_782_dreamerv3_pretraining_design.md](issue_782_dreamerv3_pretraining_design.md).

## Cost spent on this attempt

- Gate (12156): 13m on a30 (cheap, scientific value: confirmed NaN re-emerges).
- Full (12159): 23h20m on a30 (single A30 24GB), no eval block, OOM kill.
- W&B run id `1t6gadx5` exists with reward curves and is the cheapest
  artifact for any future post-mortem; do not re-run from this branch
  without resolving the two failure modes first.

## What can still be salvaged

- `checkpoints/best_reward` (peak iter 453, reward +23.96) is on disk.
  It is **not** a paper-facing checkpoint — never passed an eval block,
  was followed by NaN, and lives downstream of an OOM-killed run. At most
  it could be loaded for a one-off policy-analysis sanity check to see if
  any deterministic episode can complete a route, but treat the result as
  diagnostic only.
- `result.jsonl` and the 164 `diagnostics/iteration_*_nonfinite.json`
  files are the right starting point for a NaN-pattern investigation
  (which iter, which env runner, which reward component). The first
  nonfinite at iter 207 (177k env steps in) is the wedge.

---

## Mid-run snapshot (historical, 2026-04-29 12:33 UTC, ~4h elapsed)

The section below is the original live progress check written when the run
still looked healthy (52 iters in, no NaN yet). It is preserved as a
historical record of what the run looked like *before* the failures
emerged. **Do not treat its conclusions as the final outcome** — the
current outcome is the OOM + NaN-collapse summary above.

### Health: training was clean, NaN-free, and learning

Compared to the gate, the full run shows **no NaN drift**:

- `output/dreamerv3/.../diagnostics/` is empty (zero `iteration_*_nonfinite.json` files).
- All 52 rows in `result.jsonl` carry `reward_mean_status: finite`.
- Launcher log has no Ray exception, env construction error, or eval-block fail-closed
  trip; just standard DreamerV3-on-new-API-stack deprecation/info noise.
- W&B online sync is live (run id `1t6gadx5`).

A `best_reward` checkpoint has already been saved with full algorithm state
(`algorithm_state.pkl`, `env_runner/`, `learner_group/`, `rllib_checkpoint.json`).

### Reward trajectory — meaningful improvement (mid-run, since reverted)

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

### Throughput — slow per-iter, walltime-bound rather than iter-bound (mid-run)

- 52 iters in 244 min wall ⇒ **~281 sec/iter** (~4.7 min/iter)
- Projected at this rate: **~460 iters** in the 36h walltime cap
- Configured `train_iterations: 6000` is unreachable; the run will terminate on time
- Estimated total env steps at termination: 460 iters × ~640 env steps/iter ≈
  **~294K env steps**

The throughput limit is set by `model_size: S` × `training_ratio: 128` (lots of
world-model gradient updates per env step) on a single A30. Not a problem per se;
DreamerV3 is sample-efficient by design, and 294K env steps is in the right ballpark
to learn a SocNav-class task if the reward provides enough signal.

### Eval cadence — first verdict block expected at iter 100 (never fired)

`evaluation: every_iterations: 100, evaluation_episodes: 30`. We are at iter 52 with
~5h until iter 100. **No eval block has fired yet.** Until iter-100 eval lands, the
training-side `reward_mean` is the only learning indicator.

### What to watch for next (obsolete — captured for completeness)

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

### Risks and stop conditions still in force (mid-run wording)

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

### Convenience commands (still valid for inspecting the dead run)

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
