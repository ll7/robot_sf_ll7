# Existing-artifact diagnosis of the seed-policy comparison

The final-budget development comparison on 1101–1130 across 48 scenarios is a
measurement of these checkpoints, not an isolated causal seed-policy result.

| Policy | Timeout | Collision | Collision-free route completion | Other |
| --- | ---: | ---: | ---: | ---: |
| train010 fixed | 796 | 499 | 145 | 0 |
| seedpol fixed | 845 | 490 | 105 | 0 |
| train010 safety | 693 | 556 | 191 | 0 |
| seedpol safety | 1043 | 393 | 4 | 0 |

These are all 1,440 raw episodes per policy. One spawn-invalid key per arm is
excluded from the common valid paired rates. `outcome.timeout_event` explains
the runner's generic `terminated` label; it is not an unknown termination.

The new safety run never converged to a successful policy. Its selection success
is 0, 1, 0, 0, 1 of 1,440 at the five million-step boundaries. At the final actual
5,017,600 steps its reported training success is 0.0811%, compared with 6.5162%
for train010 safety. Reported episode reward is -43.38 versus -37.89 and episode
length 358.61 versus 304.86. These reward/length points are rolling episode-buffer
means; they do not demonstrate a previously converged run collapsing late.

Saved scenario coverage includes all 48 scenario IDs with broadly uniform counts:
243–315 activations per scenario for new safety, 338–432 for train010. Reward
kwargs, physics, optimiser hyperparameters, horizon, extractor and architecture
match. The master permutation is partitioned into disjoint worker strides;
scenario sampling uses a separate Generator and never derives a scenario from an
episode seed. Factory tests verify construction/reset identities, including the
initial SB3 reset. The archived execution source binds this wiring. Per-episode
training identities were not individually recorded: the launch-level monkeypatch
only follows the parent, while training workers use spawn. Future telemetry must
be owned by the workers; parent selection seed logs are not a training receipt.

All evaluated checkpoint hashes match the intended final 5M exports. New safety
selection-best is a distinct 2,007,040-step file. Both safety exports load on CPU.
Training uses VecMonitor over SubprocVecEnv, with no VecNormalize or missing
external normalizer. SB3 preprocessing settings match and prediction uses
inference mode. No model-load fallback was admitted.

The most likely mechanism is an adverse optimisation trajectory into turning and
stalling under the safety reward. Mean speed falls from 0.857 to 0.330 m/s and
angular-cap occupancy rises from 10.71% to 54.02%. No causal runtime defect was
demonstrated. PPO's saved initialisation seed also changes from 1001 to 20261012,
changing actor initialisation and optimiser/rollout randomness along with layout
and scenario streams. One master per arm and differing training GPUs prevent
isolating a genuine episode-seed-policy effect from these artifacts.

An attribution study would hold optimiser initialisation/RNG controls fixed,
vary the layout-stream policy, and replicate across multiple independent training
runs. New training layouts must stay in the permitted training-only range.
Selection and comparison remain separate; do not choose another winner on the
comparison pool. No new training or evaluation was started for this diagnosis.

Existing sources: [new safety training](https://wandb.ai/ll7/robot_sf/runs/759709fc),
[train010 safety training](https://wandb.ai/ll7/robot_sf/runs/08302a2f), and
[paired evaluation](https://wandb.ai/ll7/robot_sf/runs/dapfnnlv).

[Preserved diagnosis, curves, scripts and manifest](https://wandb.ai/ll7/robot_sf/runs/m380cu9k)
binds the existing inputs and passed W&B and external-hub hash readback.
