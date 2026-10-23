# Bounded development policy training

Two 5M-step PPO arms use the current main release-geometry scenario input (including its declared narrow-doorway probe),
velocity-delta actions, a 2 m/s differential-drive robot with no reverse, and the
existing grid/SocNav observation. `dev_fixed_objective_5m.yaml` applies full fixed
weights from step zero. `dev_safety_objective_5m.yaml` changes only collision,
near-miss and living weights. The unmeasured TTC surrogate has zero weight.
Neither arm implements the outstanding static-clearance speed penalty or the
full reward correction programme. Neither promises an improvement.

Training episode seeds are restricted by `scenario_sampling.episode_seed_pool`
to 1001–1020. Model selection uses 1021–1030, 48 scenarios per seed, each million
steps. Select the highest validation success subject to collision rate no worse
than the paired existing retrain; break ties by collision rate. Report final and
best checkpoints, and train/selection overlap in scenario families. Development
results are tuning evidence and never sealed release evidence. The group fix is
not part of this training branch; it needs separate behaviour admission.

Each arm runs its full 5M-step budget. Convergence early stopping is disabled by
unreachable success and plateau thresholds. Numerical exceptions abort immediately
(SB3 distribution/observation checks at each update/step); preserve all diagnostics.
No throughput, reward, or success plateau justifies cancellation. A 12-hour wall
limit is the hard bound; its signal must preserve the latest saved checkpoints.
Checkpoints at evaluation boundaries and the final model are written outside the
checkout. The launcher must require authenticated online W&B and native CUDA,
record the source SHA, resolved config, lockfile digest, package versions and GPU
identity, and upload checkpoints/provenance as a versioned W&B artifact even on
interruption. Missing CUDA or authentication is an infrastructure failure, never
a degraded training result. Maximum two GPU jobs, one GPU and 20 CPUs each.

Launch using `python scripts/training/train_ppo.py --config <arm.yaml> --run-id <unique-id>`
with `ROBOT_SF_ARTIFACT_ROOT` pointing at durable run storage and all BLAS thread
limits set to one. Do not run `uv sync` while a shared immutable job environment
is in use. The smoke budget changes only steps, vectorization and evaluation
cadence; both arm objectives must be exercised by real PPO updates within two
minutes before submitting the full budget.
