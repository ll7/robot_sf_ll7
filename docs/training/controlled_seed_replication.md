# Controlled episode-seed replication

The orchestration decision recorded on issue #9628 requests safety first, with
three paired training masters and a common PPO initialisation seed within each
policy pair. The 2026-10-12 training-seed rule remains in force while this study
is prepared. PR #10324 is a dependency for the range stream; this change does not
modify that PR or its domain-approval declaration.

`ppo_initialization_seed` optionally separates PPO actor initialisation and
initial optimiser RNG state from the layout master. SB3 normally schedules
physical resets using its policy seed. After constructing PPO, the trainer
restores the layout seed on the VecEnv before its first reset. Unseeded layout
modes cannot use the separate control. Existing configurations without the new
field retain their previous behaviour.

`training_seed_telemetry: true` makes each worker persist one JSONL receipt after
an actual successful reset. The worker owns the wrapper and sink, so spawn does
not rely on a monkeypatch inherited from the parent. Receipts use the canonical
artifact root, under `training_seed_receipts/`; files distinguish worker index
and process lifetime. Fields are schema, worker index, episode index, effective
seed and scenario ID. Each line is flushed and fsynced. Selection environments
are not training factories and do not write these receipts. Telemetry requires a
fixed reset pool or a range stream with training-only episode identities.

The planned layout masters are 100000, 100001, 100002; paired PPO initialisation
seeds are 1001, 1002, 1003. The latter are RNG controls, not simulation layout
identities. Policy A uses train010's repeated finite-pool approach, with a
20-seed pool starting at each layout master and entirely inside 100000–199999.
Policy B uses the full inclusive training range and the corresponding master.
Scenario samplers use the same master-plus-worker seed within each pair. This is
an allowed-range replication of the historical schedule, not reuse of its
historical development layouts.

All six safety runs use A30 hardware, the same reward, physics, network,
optimiser hyperparameters, 20 environments and checkpoint schedule. The inherited
5,000,000 planned-step configuration executes 5,017,600 actual steps through
rollout rounding; checkpoint metadata must verify that exact common budget.
Selection remains 1031–1060 across 48 scenarios. The predeclared comparator is
the final export, evaluated on 1101–1130 across 48 scenarios with common CPU
inference. Selection-best files remain preserved and cannot replace the final
export after seeing comparison results.

Before evaluating, audit all worker receipts: every worker present, episode
indices contiguous, identities allowed, scenario coverage unrestricted, policy A
pool membership correct, and policy B identities unique across workers. Bind the
receipt files, source, config, hardware and checkpoints to artifact hashes with
W&B and external-hub readback. Per-worker seeds cannot be inferred from the
previous parent-only selection logs.

Report each master pair and its paired mean difference. Uncertainty needs both
paired training-master and held-out seed-cluster variation; a crossed bootstrap
must resample whole paired master and evaluation-seed clusters rather than treat
all episodes as independent. With three masters, training-variation intervals
remain exploratory. This is development tuning, with no release or paper claim.

The earlier random-policy safety run consumed about 4.0 hours of training and
5.2 hours of selection inside its GPU allocation. Three such replicas alone
exceed one allocated GPU-day. A six-run projection is roughly 39–48 A30
GPU-hours; reopen the budget decision before submission. Do not start the fixed
arm unless safety is complete and residual budget is authorised.
