# Narrow-doorway PPO crash-vs-wait diagnostic (issue #9545)

Claim boundary: diagnostic-only evidence for the bound checkpoint in
`francis2023_narrow_doorway`. Not benchmark evidence; no benchmark number
changes; no dissertation edit.

Classification: **`reward_refuted`**

## 1. Binding (measured, not assumed)

- Checkpoint: `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
  (`output/model_cache/...-model.zip`, sha256
  `2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d`).
- Training recipe:
  `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml`
  — `reward_name: route_completion_v3`, final-stage weights
  `{progress 1.1, living -0.015, collision -15.0, near_miss -1.5, ttc_risk -1.2,
  comfort -0.5, smoothness -0.18, timeout -6.5, stagnation -0.8, terminal_bonus 20.0}`,
  2-stage curriculum (stage 0 = first 100 episodes, progress/collision/timeout/terminal
  only). The 10M-step checkpoint is past stage 0; the stage active at checkpoint time is
  inferred, not logged.
- Gamma: the base training config declares **no** `ppo_hyperparams.gamma`. The value
  `gamma = 0.99` is read from the checkpoint SB3 data blob (`"gamma": 0.99`,
  `gae_lambda 0.95`, `n_steps 2048`) and is used here for eval-time discounted replay
  only — not as proof of the training-time objective.
- Reward implementation: `robot_sf/gym_env/reward.py:route_completion_v3_reward`
  (git blob `73d9315ad48a0f9da55389088275c760f140b081` at diagnostic head).
- Scenario cap: `configs/scenarios/single/francis2023_narrow_doorway.yaml` declares
  `max_episode_steps: 400`; diagnostic env resolves `sim_time 40.0 s / dt 0.1 s = 400 steps`.
  The historical 600-step horizon assumption is not used anywhere.
- Wrappers/adapters: benchmark `ppo_action_to_unicycle` (mixed) + feasibility projection;
  `safety_wrapper`/`cbf_safety_filter` disabled; `fallback_to_goal: false` (fail-closed);
  predictive foresight disabled in diagnostic replay while the checkpoint declares
  predictive keys (6 `predictive_*` keys backfilled with space defaults — visible,
  degraded-input caveat).
- Termination: `terminated = route_complete OR timeout(timestep >= max_sim_steps) OR
  ped/robot/obstacle collision`; `RobotEnv` returns `truncated=False` always. The v3
  `timeout` reward term fires only on non-collision, non-success timeout.
- Full machine-readable binding: `docs/context/evidence/issue_9545_narrow_doorway_crash_vs_wait/binding.json`.

## 2. Measured replay (newly generated, seeds 225/226/227)

Reproducible command (CPU, deterministic policy):

```bash
uv run python scripts/analysis/narrow_doorway_crash_vs_wait_issue_9545.py \
  --output-dir <dir> --seeds 225,226,227 --hold-steps 60
```

Episode summary (`episode_summary.csv`):

| seed | steps | contact_step | undiscounted return | discounted return (γ=0.99) | outcome |
| --- | --- | --- | --- | --- | --- |
| 225 | 101 | 100 | -9.07 | -1.84 | collision |
| 226 | 100 | 99 | -8.83 | -1.72 | collision |
| 227 | 93 | 92 | -9.55 | -2.53 | collision |

All three canonical seeds end in obstacle collision at ~25% of the 400-step cap.
Contact location is systematic: x ≈ 13.2–13.9, y ≈ 1.9 (≈3.5–3.8 m from the doorway
gap center (15.5, 5.0)), i.e. the face of the lower doorway wall rect (x 15–16, y 1–4),
approached from the south — the robot never reaches the gap. Term sums per episode:
progress ≈ +9–9.6, living ≈ −1.4–(−1.5), collision −15.0; all other terms ≈ 0
(near_miss/ttc_risk/comfort/stagnation/timeout/terminal_bonus contribute nothing).
At the contact step the decomposition is
`progress +0.21, living −0.015, collision −15.0` (reward ≈ −14.8 to −15.0).

Mechanism trace (all seeds): the policy commands near-constant forward drive
(cmd_v ≈ 1.0–1.1 m/s, |cmd_w| ≤ 0.2) for the whole episode, drifting south from
y ≈ 5.2 at spawn to y ≈ 1.9 at contact while the pedestrian walks the far corridor
(min ped distance 7.5–8.5 m at contact, never below ~7 m). No waiting, stopping, or
avoidance steering is ever commanded; there is no deadlock/escape sequence, no time
pressure (contact at step ~100 of 400), and no pedestrian interaction. The pedestrian
state is `behavior: none` per the scenario file and far away — it cannot explain contact.

Per-step traces: `trace_seed{225,226,227}.jsonl` (slimmed to measured fields;
`ped_positions` dropped from counterfactual files to keep them compact).
Figures: `figure_trajectory.png` (paths vs doorway walls),
`figure_reward_timeline.png` (reward + wall clearance to contact),
`figure_mechanism_timeline.png` (progress term, cmd_v, ped distance).

Historical 0.0.6 evidence (`docs/analysis/issue_9429_goal_adjacent_timeouts_0_0_6.md`)
is retained separately: it is a descriptive bundle analysis with no step traces and
does not identify any mechanism. Nothing in this note reuses it as mechanism evidence.

## 3. Matched crash-vs-wait counterfactual (`return_table.csv`)

Fork state: 10 steps before measured contact (t=90/89/82), prefix replayed open-loop
through the canonical command path. Hold branch commands (v=0, w=0) velocity, which
decelerates the robot to rest in ~10 steps and holds it collision-free for the full
60-step horizon. Policy-continuation branch re-runs the bound policy live and contacts
the wall within 11 steps on all seeds.

| seed | fork | crash disc. (γ=0.99) | wait disc. (γ=0.99) | crash undisc. | wait undisc. | wait contacts? |
| --- | --- | --- | --- | --- | --- | --- |
| 225 | t90 | −13.15 | −0.96 | −14.57 | −1.37 | no |
| 226 | t89 | −13.13 | −0.90 | −14.54 | −1.30 | no |
| 227 | t82 | −13.09 | −0.98 | −14.50 | −1.40 | no |

Waiting strictly dominates crashing under the bound evaluation objective, by ~12
discounted points on every seed. The counterfactual is controlled but not bit-identical:
pedestrian reactions after the fork may differ (documented per row); here the pedestrian
is >8 m away and its motion is unaffected in practice.

## 4. Bounded sensitivity (`sensitivity.csv`, eval-time replay only)

Sweep over γ ∈ {0.95, 0.99, 0.995} × collision weight ∈ {−5, −15, −25} at the same fork
states: `prefers_crash` is False in all 27 cells. Even a −5 collision penalty with
γ=0.95 leaves waiting ahead by ~2.4 points; the bound (−15, 0.99) margin is ~12 points.
No modest parameter change reverses the preference. This is diagnostic replay arithmetic,
not retraining evidence.

## 5. Why this refutes the reward hypothesis

The tested explanation — that the training objective makes wall contact preferable to
waiting — predicts crash return ≥ wait return under the bound objective. The matched
replay shows the opposite, robustly: waiting avoids the −15 collision, keeps the small
per-step progress/living terms, and never triggers the −6.5 timeout term inside the
60-step window (and the full-episode arithmetic shows even 300 further wait steps at
−0.015 living plus zero progress cannot approach −15). The policy nevertheless drives
forward into the wall on all three seeds without any pedestrian, time, or deadlock
trigger. The reward did not incentivize this contact; the mechanism is a systematic
geometry/obstacle-handling failure (persistent forward drive into the lower doorway
wall face from a southerly approach line), with the reward gradient, if followed, pointing
toward waiting instead.

## 6. What remains unidentified

- Why the policy holds forward drive: value/gradient attribution, observation ablation
  (occupancy grid vs SocNav state vs backfilled predictive keys), and training-coverage
  analysis are not performed here.
- Whether the 6 backfilled `predictive_*` keys (eval-time degradation from disabling
  foresight) shift the contact step vs the canonical `ppo_15m_grid_socnav.yaml`
  configuration (foresight enabled, `predictive_proxy_selected_v2_full`). The contact
  mechanism is systematic in this replay; its exact step may differ with foresight on.
- Training-time incentives: eval-time replay cannot prove what the optimizer saw
  (curriculum stage, vec-env wrappers, training gamma). The refutation is scoped to the
  bound evaluation objective, which is the artifact the dissertation actually cites.
- Pedestrian-reactive divergence in counterfactuals is documented but unmaterialized
  here (pedestrian >8 m away, `behavior: none`).

## 7. Artifacts

- Script: `scripts/analysis/narrow_doorway_crash_vs_wait_issue_9545.py`
- Test: `tests/analysis/test_narrow_doorway_crash_vs_wait_issue_9545.py`
- Evidence: `docs/context/evidence/issue_9545_narrow_doorway_crash_vs_wait/`
  (`binding.json`, `trace_seed*.jsonl`, `counterfactual_*.jsonl`, `episode_summary.csv`,
  `return_table.csv`, `sensitivity.csv`, `figure_trajectory.png`,
  `figure_reward_timeline.png`, `figure_mechanism_timeline.png`)
- Full per-step simulator dumps (with `ped_positions`) are in the trace files above;
  no additional worktree-local `output/` is relied on.
