<!-- AI-GENERATED (robot_sf#10064) - NEEDS-REVIEW -->
# Issue #10064: clearance recovery

Shared ranking fixes forced stasis when no ordinary command is admissible. This is best effort, with no safety certificate, sealed evaluation or release claim.

`validation.json` records the full default suite and fail-on-base proof; `empty_world.json` preserves the 30 successful paired empty episodes and IEEE command parity. `provenance.json` pins immutable source, inputs, model bytes and raw-log digests. `reproduce.txt`, the exact authored horizon schedule and empty map support diagnostic reproduction; use only explicit dev seeds and prescribed Slurm resources.

`paired_analysis.json` declares whether all three arms are complete. `changed_outcomes.json` classifies every changed outcome in completed pairs. `trace_samples.json` and `manual_inspections.json` preserve directly inspected windows. `inner_checkpoints.json` verifies all 29 guarded-PPO first divergences with safe outer labels: identical observations, inner RiskDWA infeasibility in both versions, and 4,416 identical preceding paired commands.

Producing commit: `de8f8a105ab91ad665773bbed204d7d25c9c6c30`. Comparator commit: `93ba0d75fbecc69ddeb62bbf77a435de385caa3b`. Producing config: `configs/algos/risk_dwa_release_v0_0_8.yaml`; config sha256 hash `27d0b269e5a65572bdf2f5af069a5e0037eb0728d501ea12438597404e7e1527`. MPPI and guard producing configs/hashes are listed in `provenance.json`; the campaign matrix and horizon bytes are identical in both conditions. Episode seeds: 1001–1005; empty-world seeds: 1001–1010.

Current completed arms: risk_dwa, predictive_mppi, guarded_ppo. Full matrix complete: True.

risk_dwa: base S/C/T 203/1/36; PR 204/0/36. Changed outcomes 1.

predictive_mppi: base S/C/T 105/1/134; PR 149/0/91. Changed outcomes 46.

guarded_ppo: base S/C/T 146/1/93; PR 152/0/88. Changed outcomes 16.

risk_dwa mean no-admissible 0.028177 -> 0.024200; physical frozen 0.069424 -> 0.061942. Worse aggregate success: False; worse collision count: False. Paired outcome transitions: {'C->S': 1, 'S->S': 203, 'T->T': 36}.

predictive_mppi mean no-admissible 0.140298 -> 0.105727; physical frozen 0.362849 -> 0.230646. Worse aggregate success: False; worse collision count: False. Paired outcome transitions: {'C->S': 1, 'S->S': 104, 'S->T': 1, 'T->S': 44, 'T->T': 90}.

guarded_ppo mean no-admissible 0.206074 -> 0.167115; physical frozen 0.233926 -> 0.184234. Worse aggregate success: False; worse collision count: False. Paired outcome transitions: {'C->S': 1, 'S->S': 141, 'S->T': 5, 'T->S': 10, 'T->T': 83}.

Every arm improves aggregate success and collision counts. Five guard successes and one MPPI success regress to timeout. The MPPI regression enters a later finite-score stall after three changed infeasible commands; its final 116-step stall is outside the no-admissible branch. All 63 changed outcomes are classified; 17 paired traces, including every regression, are directly inspected. The ranking balances rollout pedestrian and obstacle margins; first-step/TTC vetoes still define admissibility, but are not extra terms in recovery ranking. Small steering changes can produce stalls or miss goal-zone entry. Recorded controller-target distance is not terminal goal remaining.

Hosted CI remains skipped while draft: ready collects legacy environment tests on forbidden 111/123, and the wheel smoke independently steps seed 123 outside pytest. Raw archives have explicit local/remote locations but no durable shared raw-data archive. The command implementation is unchanged except for a typed two-float tuple spelling, independently verified on 168,869 recorded command conversions (`validation.json`). Tracked compact windows and source/input identities are the preserved review evidence.
