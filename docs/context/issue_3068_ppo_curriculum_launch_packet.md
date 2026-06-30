# Issue #3068 PPO Curriculum Launch Packet

Date: 2026-06-23

## Scope

This note records the pre-launch packet for a PPO **density/complexity curriculum**
social-navigation experiment. It is a spec/config deliverable authored BEFORE any long
training run. It does **not** train PPO, submit SLURM, run benchmark evaluation, or promote
a learned policy. Running the long SLURM training is an explicitly separate downstream issue.

## Launch Packet

- Config: `configs/training/ppo_curriculum_issue_3068_launch_packet.yaml`
- Schema: `ppo-curriculum-launch-packet.v1`
- Campaign: `issue_3068_ppo_curriculum_v1`
- Test: `tests/training/test_ppo_curriculum_launch_packet_issue_3068.py`

## Hypothesis

A density/complexity curriculum that gradually raises pedestrian density and scene
complexity reaches a **higher final benchmark** success rate (and/or lower collision rate)
than a **fixed-difficulty PPO baseline** trained on the full-difficulty distribution with a
**matched total budget**. This is a proposal to be tested, not a result.

## Competing Explanations and Discriminating Checks

The packet encodes three competing explanations as discriminating checks:

1. **Gains from extra budget, not curriculum** → `matched_budget`: the curriculum and
   baseline arms use the same documented `total_timesteps` (15M) and the same five seeds.
   Any budget difference is recorded as a confound and cannot be reported as a curriculum
   gain.
2. **Curriculum improves the training curve but not the final benchmark** →
   `train_curve_and_final_benchmark`: both training-curve metrics and final held-out
   benchmark metrics must be reported; a curriculum claim requires final-benchmark
   improvement, not a curve-only readout. No early-stopping on the curve may stand in for a
   final-benchmark claim.
3. **Insufficient provenance** → `config_checksums_and_artifact_pointers`: every referenced
   config carries a sha256 checksum in the packet, and any future run must record concrete
   checkpoint URIs and W&B run ids. Pending aliases are placeholders, not evidence.

## Curriculum Schedule

Expressed over real env knobs: `ped_density` (scenario `simulation_config.ped_density`),
`max_peds_per_group` (`env_overrides.sim_config.max_peds_per_group`), and the sampled
archetype mix (scene complexity). Four stages partition the 15M budget; each non-final stage
advances on a within-stage success-rate threshold:

| Stage | ped_density | max_peds_per_group | timesteps | advance success |
| --- | --- | --- | --- | --- |
| stage_0_sparse | 0.0 | 1 | 3M | 0.85 |
| stage_1_low | 0.01 | 2 | 4M | 0.80 |
| stage_2_medium | 0.02 | 3 | 4M | 0.75 |
| stage_3_full | 0.04 | 3 | 4M | terminal |

## Baseline Comparator

`fixed_difficulty_ppo_v1`: fixed-difficulty PPO trained directly on the full-difficulty
distribution (`ped_density 0.04`, `max_peds_per_group 3`,
`classic_interactions_francis2023_full`) for the same 15M timesteps, seeds, architecture
(`grid_socnav`) and reward (`route_completion_v3`) as the curriculum arm.

## Seeds, Budget, Stop Rule, Metrics, Artifacts

- Seeds: training `[123, 231, 777, 992, 1337]` (matches base config), evaluation dev
  `[101, 102, 103]` / eval `[111, 112, 113]` from `configs/benchmarks/seed_sets_v1.yaml`.
- Budget: two arms × 5 seeds × 15M timesteps = 10 training runs, budget matched between arms.
- Stop rule: per run stop at 15M timesteps or the base-config convergence gate
  (success_rate >= 0.9, collision_rate <= 0.05 over plateau_window 2000), whichever first;
  curriculum stage advance gated on per-stage `advance_success_rate`.
- Metrics: train-curve (`eval/success_rate_at_step`, `eval/collision_rate_at_step`,
  `rollout/ep_rew_mean`) AND final-benchmark (`benchmark/success_rate`,
  `benchmark/collision_rate`, `benchmark/timeout_rate`, `benchmark/route_completion`);
  primary `benchmark/success_rate`.
- Expected artifacts: per-arm/per-seed checkpoints, training-curve CSVs, final-benchmark
  summary JSON, curriculum stage-advance log, comparison table.

## Validation

```bash
uv run python -c "import yaml; d=yaml.safe_load(open('configs/training/ppo_curriculum_issue_3068_launch_packet.yaml')); assert d['no_training_result_claim'] is True; print('valid', d['campaign_id'])"
uv run pytest tests/training/test_ppo_curriculum_launch_packet_issue_3068.py -q
```

## No-Submit Route And Preflight Fields

Issue #3807 completes the #3068 launch-packet decision surface without submitting a
SLURM job. The packet now sets `execution_boundary.submit_slurm_from_this_issue:
false`, `queue_hint.submit_ready: false`, and `queue_hint.state:
no_submit_preflight_ready`.

Route and resource fields are explicit under `queue_hint`:

- `route_id`: `imech192:a30`
- `cluster`: `imech192`
- `partition`: `a30`
- `qos`: `a30-gpu`
- `script`: `/home/luttkule/git/robot_sf_ll7-private-ops/auxme/SLURM/issue_791_reward_curriculum.sl`
- `resources`: 20 CPUs, 1 GPU, 120 GB memory, 43,200 second estimate

The packet also records local-only preflight proof commands for the private script
path, public training config, and packet contract test. Expected outputs are
grouped under `queue_hint.expected_outputs`, and exact no-submit validation
commands are listed under `queue_hint.validation_commands`.

## Durable Artifact Policy

Checkpoints and raw logs stay in the durable W&B backend, not in git. Before training, the
`:pending` artifact aliases must be replaced with concrete URIs, the training commit and base
checkpoint recorded, and the matched budget/seeds documented. After training, small
final-benchmark summaries may be promoted to `docs/context/evidence/` with checksums.

## Prerequisite Note (honest scope boundary)

The packet expresses the curriculum over real, existing env knobs (`ped_density`,
`max_peds_per_group`, scenario archetype mix). The repository does **not** currently ship a
single packaged "PPO curriculum scheduler" that consumes this packet and advances stages
automatically end to end. Wiring the per-stage density/complexity scheduler into
`train_ppo.py` (stage advance on `advance_success_rate`, budget partition across stages) is a
**prerequisite** for the downstream training issue and is intentionally out of scope here.
This packet defines the schedule, baseline, seeds, budget, stop rule, metrics, and artifact
policy so that the scheduler implementation and the long run can be built against a frozen,
checksummed spec.

## No-Claim Boundary

No training-result, performance, or benchmark claim exists for this campaign. Nothing here
has been trained or evaluated. Any future claim requires the long training run, the
predeclared train-curve AND final-benchmark metrics, matched budget, and durable artifacts.
Evidence status: `proposal`.
