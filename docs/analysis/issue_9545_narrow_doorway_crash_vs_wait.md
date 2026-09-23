# Narrow-doorway PPO crash-vs-wait diagnostic (issue #9545)

Claim boundary: diagnostic-only evidence for the bound checkpoint in
`francis2023_narrow_doorway`. Not benchmark evidence; no benchmark number
changes; no dissertation edit.

Classification: **`reward_refuted` at the bound evaluation boundary only**

## 1. Binding (measured, not assumed)

- Checkpoint: `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
  (`output/model_cache/...-model.zip`, sha256
  `2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d`).
- Predictive foresight checkpoint: `predictive_proxy_selected_v2_full` (CPU inference,
  sha256 `a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1`).
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
  predictive foresight is enabled from the registry-pinned checkpoint and the replay
  fails closed on predictive-model degradation. The raw RobotEnv dict (including the
  occupancy grid and metadata) is passed through `normalize_map_observation` to the
  native PPO dict adapter; no predictive keys are backfilled.
- Counterfactual fork: 20 steps before measured contact, then 60 steps. This is early
  enough for the zero-command branch to decelerate; both branches replay the same
  measured prefix.
- Termination: `terminated = route_complete OR timeout(timestep >= max_sim_steps) OR
  ped/robot/obstacle collision`; `RobotEnv` returns `truncated=False` always. The v3
  `timeout` reward term fires only on non-collision, non-success timeout.
- Full machine-readable binding: `docs/context/evidence/issue_9545_narrow_doorway_crash_vs_wait/binding.json`.

## 2. Measured replay (newly generated, seeds 225/226/227)

Reproducible command (CPU, deterministic policy):

```bash
uv run python scripts/analysis/narrow_doorway_crash_vs_wait_issue_9545.py \
  --output-dir <dir> --seeds 225,226,227 --hold-start-offset 20 --hold-steps 60
```

Episode summary (`episode_summary.csv`):

| seed | steps | contact_step | undiscounted return | discounted return (γ=0.99) | outcome |
| --- | --- | --- | --- | --- | --- |
| 225 | 63 | 62 | -6.98 | -2.37 | collision |
| 226 | 60 | 59 | -7.41 | -2.83 | collision |
| 227 | 58 | 57 | -7.47 | -3.01 | collision |

All three canonical seeds end in obstacle collision at ~15% of the 400-step cap.
Contact location is systematic: x ≈ 14.21–14.95, y ≈ 4.60–5.12, at the doorway
opening. With the corrected `min_obstacle_clearance_m` calculation, the nearest
geometry is the doorway corner/edge (lower doorway edge for seeds 225/227 and upper
doorway edge for seed 226), not the bottom boundary. Start clearance is positive
(2.23–2.63 m) and first non-positive clearance equals the collision step (57/59/62).
Term sums per episode: progress ≈ +10.4–11.6, living ≈ −0.87–−0.95, smoothness
≈ −2.0–−2.6, collision −15.0; near-miss/TTC/comfort/stagnation/timeout/terminal
bonus contribute nothing.
At the contact step the decomposition is
`progress +0.19–0.21, living −0.015, smoothness −0.05–−0.07, collision −15.0`
(reward ≈ −14.8 to −14.9).

Mechanism trace (all seeds): the policy maintains forward drive (cmd_v ≈ 1.30–2.00 m/s)
and issues substantial steering (cmd_w reaches the configured ±1.0 bounds), but still
contacts doorway geometry without a stopping/escape sequence. The robot starts near
y ≈ 4.23–5.37 and contacts at y ≈ 4.60–5.12; the pedestrian remains far away
(minimum distance 7.96–9.03 m). There is no time pressure (contact at step 57–62 of
400) and no pedestrian interaction. The pedestrian state is `behavior: none` per the
scenario file and cannot explain contact.

Per-step traces: `trace_seed{225,226,227}.jsonl` (slimmed to measured fields;
`ped_positions` dropped from counterfactual files to keep them compact).
Figures: `figure_trajectory.png` (paths vs doorway walls),
`figure_reward_timeline.png` (reward + wall clearance to contact),
`figure_mechanism_timeline.png` (progress term, cmd_v, ped distance).

Historical 0.0.6 evidence (`docs/analysis/issue_9429_goal_adjacent_timeouts_0_0_6.md`)
is retained separately: it is a descriptive bundle analysis with no step traces and
does not identify any mechanism. Nothing in this note reuses it as mechanism evidence.

## 3. Matched crash-vs-wait counterfactual (`return_table.csv`)

Fork state: 20 steps before measured contact (t=42/39/37), prefix replayed open-loop
through the canonical command path. Hold branch commands (v=0, w=0) velocity, which
decelerates the robot to rest in ~10 steps and holds it collision-free for the full
60-step measured horizon. Policy-continuation branches re-run the bound policy live; each
contains 21 rows indexed `branch_step=0..20`, with the wall collision recorded at
`branch_step=20` (for forks t=42/39/37, respectively).

| seed | fork | crash disc. (γ=0.99) | wait disc. (γ=0.99) | crash undisc. | wait undisc. | wait contacts? |
| --- | --- | --- | --- | --- | --- | --- |
| 225 | t42 | −9.43 | −0.55 | −11.87 | −1.18 | no |
| 226 | t39 | −9.46 | −0.31 | −11.92 | −0.87 | no |
| 227 | t37 | −9.32 | −0.32 | −11.75 | −0.88 | no |

Over the measured 60-step hold window, waiting strictly dominates the policy continuation
under the bound evaluation objective by ~9 discounted points on every seed. The
counterfactual is controlled but not bit-identical:
pedestrian reactions after the fork may differ (documented per row); here the pedestrian
is >8 m away and its motion is unaffected in practice.

## 4. Bounded sensitivity (`sensitivity.csv`, eval-time replay only)

Sweep over γ ∈ {0.95, 0.99, 0.995} × collision weight ∈ {−5, −15, −25} at the same fork
states: `prefers_crash` is False in all 27 cells. This is diagnostic replay arithmetic,
not retraining evidence; it does not establish a released-policy causal mechanism or a
population-level claim.

## 5. Why this refutes the reward hypothesis

The tested explanation — that the training objective makes wall contact preferable to
waiting — predicts crash return ≥ wait return under the bound objective. The matched
replay shows the opposite at the bound objective, robustly across these three seeds
within the measured 60-step counterfactual window: waiting avoids the −15 collision,
retains the observed per-step reward terms, and does not trigger the −6.5 timeout term
within that window. No full-horizon wait-return extrapolation is made: later smoothness
and terminal/timeout rewards are not evaluated by these 60-step traces. The policy
nevertheless drives forward into the wall on all three seeds without any pedestrian,
time, or deadlock trigger. The reward did not incentivize this contact at the bound
evaluation objective over the measured window;
the mechanism is a systematic geometry/obstacle-handling failure (persistent forward
drive and steering into doorway geometry), with the reward gradient, if followed,
pointing toward waiting instead.

## 6. What remains unidentified

- Why the policy holds forward drive and steering: value/gradient attribution,
  observation ablation (occupancy grid vs SocNav state), and training-coverage analysis
  are not performed here.
- The replay is canonical for the bound checkpoint/configuration and verified predictive
  model cache, but it remains one scenario and three seeds. It does not establish a
  released-policy mechanism outside this evaluation boundary.
- Training-time incentives: eval-time replay cannot prove what the optimizer saw
  (curriculum stage, vec-env wrappers, training gamma). The refutation is scoped to the
  bound evaluation objective; no broader paper or dissertation claim is admitted here.
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
