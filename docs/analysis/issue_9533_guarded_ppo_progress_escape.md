# Issue #9533: RiskDWA progress-escape diagnostic

## Reproducibility status

**Blocked for independent numerical reuse.** The raw rows, traces and analysis inputs
named below are not available from a fresh clone or a durable external bundle.
Until the evidence owner supplies a retrievable checksum-bound bundle with the
analysis script and source/config identities, every numerical statement below
remains an unrecomputed historical diagnostic. It may not justify a treatment,
release gate, ranking or downstream scientific claim. This repair does not
regenerate the historical evaluation runs. Fresh development sweeps are separate
source-bound evidence and cannot repair historical custody retrospectively.

## Claim boundary

This is a bounded local diagnostic, not benchmark-success, safety, release, paper, or dissertation evidence. Both guarded arms completed the paired rows and traces, but the canonical runner failed their admission because `guard_stats.fallback_safe` was positive. The raw PPO control was available. The frozen progress-escape candidate did not change any paired episode outcome and never selected an escape command. The issue remains open for a different minimal guard variant.

## Frozen experiment identity

- Base commit: `50553743acd0b3ae5b5d03fe585821ff0c283ae8`; tracked working-tree diff SHA-256: `b4c829196926b09a6b5b8f7070b1af62d1acb475552ef2db097be9da57c5c290`.
- Checkpoint: `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`, SHA-256 `8367af109a27e8879ced0c8913f6eff26df7ec59c31ea88f9a297bb2c141eb09`.
- Treatment: `configs/algos/guarded_ppo_issue_9533_progress_escape.yaml`, derived from the CPU guarded baseline with only `fallback_risk_dwa.progress_escape_enabled` enabled.
- Raw control: `configs/algos/ppo_v3_issue_9533_cpu.yaml`, a copy of `ppo_v3_camera_ready.yaml` changing only `device` from `auto` to `cpu`. The initial `auto` run failed in forked workers with CUDA reinitialization; the CPU copy kept the same checkpoint and policy and allowed all arms to execute on CPU.
- Dev campaign: `issue9533_dev_cpu_20260927t1043z`, 16 scenarios × seeds 101–103 × three arms = 144 rows. The guard-control dev rows alone selected the constrained eight by timeout rate descending, route-complete rate ascending, obstacle-contact count descending, then scenario name.
- Eval campaign: `issue9533_eval_cpu_20260927t1049z`, same 16 scenarios × disjoint seeds 111–113 × three arms = 144 rows. The selected-eight slice contains 24 rows per arm.
- Frozen eval packet SHA-256: `67c3d729d54f9c913fac6a7862db0226df1d2e6bed6f0ac5f730a4cc3853d9d8`; updated issue body SHA-256: `f69f50b887430649d08b39bdf91dfa36277ce462bf6594322dee0943c2bf11ad`.

The dev and eval campaigns both exited 2. The raw PPO arm was available in the corrected runs. Guarded baseline and treatment were rejected as `availability_status=failed`, `readiness_status=fallback`, with the eval reason `planner runtime reported forbidden marker guard_stats.fallback_safe=3536`. The complete guarded rows remain diagnostic only and cannot pass the repository benchmark-success policy. Planner and simulation traces were present for all 48 episodes of each guarded arm, and simulation traces were present for all three arms.

## Results

# Issue #9533 progress-escape diagnostic

**Claim boundary:** diagnostic-only. The guarded arms are not benchmark-success evidence because the canonical runner rejects their positive `guard_stats.fallback_safe` markers. Three eval seeds support descriptive intervals only.

- Eval campaign: `issue9533_eval_cpu_20260927T1049Z`
- Eval seeds: [111, 112, 113]
- Full constrained cohort: 16 scenarios; frozen subset: classic_bottleneck_high, classic_realworld_double_bottleneck_high, classic_bottleneck_medium, classic_cross_trap_low, classic_doorway_low, classic_doorway_medium, classic_overtaking_low, classic_overtaking_medium
- Complete episode rows: `True`; complete traces: `True`
- Acceptance adjudication: `not_benchmark_evidence_due_to_fail_closed_run_status`

## Availability and completeness

| Arm | Run status | Availability | Benchmark success | Rows (full 16) | Guard planner traces | Simulation traces |
|---|---|---|---:|---:|---:|---:|
| ppo_br06_matched_guard_off | ok | available | True | 48 | 0 | 48 |
| guarded_ppo_baseline | failed | failed | False | 48 | 48 | 48 |
| guarded_ppo_progress_escape | failed | failed | False | 48 | 48 | 48 |

## selected8 outcome means

| Arm | Success | Timeout | Pedestrian contacts/ep | Obstacle contacts/ep | Time-to-goal norm |
|---|---:|---:|---:|---:|---:|
| ppo_br06_matched_guard_off | 0.1250 | 0.0000 | 0.2083 | 0.6667 | 0.9301 |
| guarded_ppo_baseline | 0.0417 | 0.4167 | 0.0000 | 0.5417 | 0.9743 |
| guarded_ppo_progress_escape | 0.0417 | 0.4167 | 0.0000 | 0.5417 | 0.9743 |

### Paired candidate minus guarded control

| Metric | Mean difference | Descriptive 95% t interval | Per-seed differences |
|---|---:|---:|---|
| success | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| timeout | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| pedestrian_contacts | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| obstacle_contacts | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| time_to_goal_norm | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |

## full16 outcome means

| Arm | Success | Timeout | Pedestrian contacts/ep | Obstacle contacts/ep | Time-to-goal norm |
|---|---:|---:|---:|---:|---:|
| ppo_br06_matched_guard_off | 0.1250 | 0.0000 | 0.2083 | 0.6667 | 0.9162 |
| guarded_ppo_baseline | 0.1042 | 0.2708 | 0.0000 | 0.6250 | 0.9290 |
| guarded_ppo_progress_escape | 0.1042 | 0.2708 | 0.0000 | 0.6250 | 0.9290 |

### Paired candidate minus guarded control

| Metric | Mean difference | Descriptive 95% t interval | Per-seed differences |
|---|---:|---:|---|
| success | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| timeout | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| pedestrian_contacts | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| obstacle_contacts | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |
| time_to_goal_norm | 0.0000 | [0.0000, 0.0000] | 111: 0.0000, 112: 0.0000, 113: 0.0000 |

## Guard and progress-escape trace

| Arm | Guarded steps | Starts | Ends | Mean first intervention (s) | Mean intervention duration/episode (s) | Guard-filtered commands | Progress-escape selected commands | Status counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ppo_br06_matched_guard_off | 0 | 0 | 0 | n/a | 0.00 | 0 | 0 | `{}` |
| guarded_ppo_baseline | 11402 | 378 | 343 | 12.89 | 23.75 | 11402 | 0 | `{"disabled": 11402, "missing": 3020}` |
| guarded_ppo_progress_escape | 11402 | 378 | 343 | 12.89 | 23.75 | 11402 | 0 | `{"evaluated_but_not_selected": 60, "missing": 3020, "not_evaluated": 11342}` |

## Frozen numerical screen

| Criterion | Numeric result |
|---|---|
| selected8_success_gain_met | not met |
| selected8_timeout_reduction_met | not met |
| selected8_pedestrian_contact_margin_met | met |
| selected8_obstacle_contact_margin_met | met |
| full16_pedestrian_contact_margin_met | met |
| full16_obstacle_contact_margin_met | met |

The numerical screen is not benchmark-success evidence: availability remains fail-closed. See the accompanying JSON for config/source packet identity and exact run roots.


## Interpretation and next action

On the frozen selected eight, the candidate-minus-guarded-control mean differences were 0.0000 for success, timeout, pedestrian contacts, obstacle contacts, and normalized time-to-goal; each paired eval-seed difference (111, 112, 113) was also 0.0000. The pre-registered success-gain (+0.10) and timeout-reduction (-0.10) screens were not met. The contact margins were numerically met, with 0.0000 paired pedestrian- and obstacle-contact differences on both the selected eight and full 16, but fail-closed availability prevents any acceptance claim.

The trace recorded 60 evaluated progress-escape candidates in eval and zero selections (the candidate was evaluated 52 times in dev, also with zero selections). Guard intervention timing was identical across the guarded arms: 378 starts, 343 ends, 12.89 s mean first intervention, and 23.75 s mean intervened time per episode across full eval. This supports the narrow conclusion that the existing score-checked progress-escape candidate did not activate in this cohort. It does not identify a causal explanation for the broader guarded-PPO success loss.

The next action under #9533 is to select another minimal option already listed in the issue (recovery/hysteresis after intervention, obstacle-aware fallback selection, or adaptive intervention release), preregister its settings and thresholds on dev, then use the disjoint eval seeds. Do not retune this progress-escape setting from eval or promote these outputs beyond diagnostic use.

## Artifact custody

Compact results are tracked in this report. Episode JSONL files, per-step traces, runner summaries, the frozen packets, ranking CSV, and the analysis script remain in the local common Git-dir artifact area under `.git/codex-agent-runs/issue-9533/`; they are checksummed there and were not committed. Their classification is local diagnostic output, not durable benchmark evidence. The checkpoint remains an ignored local model cache.
