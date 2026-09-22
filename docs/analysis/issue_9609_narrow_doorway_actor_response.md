# PPO narrow-doorway actor/critic response diagnostic (issue #9609)

Claim boundary: **diagnostic-only** evidence for the checkpoint, scenario, seeds, and
twelve matched pre-contact states below. This is not benchmark evidence, does not
identify a training-time optimization cause, and is not admitted to a paper or
dissertation.

Classification: **`actor_value_response_mismatch_supported` at the tested-state boundary**

## Question and frozen inputs

Issue #9545 / PR #9546 refuted a reward-preference explanation for the PPO wall
contacts, but did not identify why the policy keeps moving forward. This follow-up
distinguishes three narrower explanations:

1. the model-ready observation omits or misregisters the doorway wall;
2. the adapter or robot dynamics converts a safe model action into unsafe motion;
3. the policy sees a bad geometry state but its actor still selects forward motion.

The diagnostic is stacked on PR #9546 exact head
`59949b0cac2fd35988e2d8947c55ab1dac2c70d3` and reuses its two checksum-bound
registry artifacts, canonical `francis2023_narrow_doorway` configuration, deterministic
policy, and seeds 225/226/227. Predictive foresight loaded natively on CPU and no
fallback or degraded row was accepted.

## Matched-state result

Four states were sampled per seed: 20, 10, 5, and 1 step before measured contact.
The table condenses the earliest and latest sampled state; the tracked CSV preserves
all twelve rows.

| Seed | Contact step | Nearest forward static cell, early -> late (m) | Critic, early -> late | Minimum model-predict v (m/s) | Minimum adapter v (m/s) | Minimum executed v after step (m/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 225 | 62 | 4.30 -> 1.03 | -11.31 -> -15.35 | 1.85 | 1.85 | 1.85 |
| 226 | 59 | 4.53 -> 1.14 | -11.43 -> -14.48 | 1.30 | 1.30 | 1.81 |
| 227 | 57 | 2.90 -> 1.27 | -11.49 -> -14.14 | 2.50 | 2.00 | 2.00 |

The static obstacle channel contains forward doorway geometry in every sampled state.
Its nearest occupied cell approaches the robot as the independent simulator clearance
shrinks. The difference between a roughly 1 m cell-center distance and near-zero
surface clearance at contact is expected because the robot collision radius is about
1 m; it is not evidence that the grid missed the wall.

The critic becomes more negative toward contact for every seed. At the same time, the
Stable-Baselines deterministic prediction remains strongly forward (at least
1.30 m/s across all rows). The PPO adapter caps some predictions at its configured
2.0 m/s limit, but never reverses or creates forward intent. Executed speed also stays
forward. Thus the command is already unsafe at the model-prediction boundary; the
velocity-to-acceleration adapter preserves rather than originates it.

## Static-geometry intervention

At each identical state, the obstacle channel was zeroed and the combined channel was
recomputed from the preserved pedestrian channel. Every goal, robot, pedestrian,
metadata, and predictive-foresight field was held fixed. This is an out-of-distribution
intervention used only for attribution.

- Removing static geometry raised the critic by +18.75 to +27.97 across all twelve
  states. The critic therefore strongly responds to the wall representation.
- Removing static geometry reduced actor-mean forward speed by 0.04 to 1.19 m/s in
  all twelve states. The actor is not invariant to the geometry, but its canonical
  response to the present wall remains forward rather than stopping.
- Because the intervention is out of distribution, these deltas do not establish a
  safe counterfactual policy or a training-time cause. They do rule against the simple
  claim that the model receives no usable static-geometry signal at these states.

## Evidence synthesis

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Missing/misregistered doorway geometry | #9609 | diagnostic-only | PR #9546 bound configuration | 225-227 | matched-state CSV | occupied forward cells, simulator clearance | contradicted at the 12 tested states | Does not prove every grid feature is optimal or learned robustly |
| Adapter/projection creates forward motion | #9609 | diagnostic-only | PPO `unicycle`, v cap 2.0 m/s | 225-227 | matched-state CSV | model action, adapter command, native action, executed speed | contradicted at the 12 tested states | Adapter changes magnitude and dynamics remain acceleration-limited |
| Actor continues despite critic-recognized bad geometry | #9609 | diagnostic-only | bound PPO + predictive foresight | 225-227 | summary JSON + matched-state CSV | actor mean, deterministic prediction, critic value, geometry ablation | supported at the tested-state boundary | Evaluation-time mismatch does not identify why training produced it |

## Trace mechanism summary

```yaml
mechanism_activation:
  activated: true
  activation_count: 12
  changed_command_source: false
  changed_outcome: unknown
  likely_failure_reason: >-
    At the tested states, static doorway geometry is present and the critic assigns
    increasingly poor value, while the actor and adapter preserve forward motion.
```

This narrows the residual mechanism to a learned actor-response failure (with an
actor/critic mismatch visible at evaluation time), not a reward preference, absent wall
input, or adapter-created command. It does not yet distinguish representation learning,
actor-head optimization, training coverage, or action-distribution saturation as the
training-time root cause. No retraining is justified by this diagnostic alone.

## Reproduction and artifacts

```bash
uv run python scripts/analysis/narrow_doorway_actor_response_issue_9609.py \
  --output-dir docs/context/evidence/issue_9609_narrow_doorway_actor_response
```

Compact tracked evidence:

- `binding.json`: source PR/head, artifact/config binding, seeds, offsets, and fail-closed status;
- `matched_state_rows.csv`: all twelve observation/action/value rows;
- `mechanism_summary.json`: conservative classification and per-seed summary;
- adjacent `.review.json` files: exact-byte checksums and independent-review status.

The PPO and predictive model files under `output/model_cache/` are hydrated caches only.
Their durable sources and checksums remain the registry/release bindings inherited from
PR #9546. No worktree-local `output/` path is cited as evidence.

## Limitations and next boundary

- Three deterministic seeds and twelve states do not support a population-level PPO claim.
- Evaluation actor/critic outputs do not reveal the optimizer's historical gradients or
  training-data coverage.
- The static-geometry ablation is deliberately diagnostic and out of distribution.
- Outcome change is unknown because this probe compares decisions at identical states; it
  does not run a new closed-loop intervention.
- The next empirical step, only if worth pursuing after review, is a separately scoped
  representation/actor-head attribution over these same frozen states—not retraining or a
  broader campaign.
