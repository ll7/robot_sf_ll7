# Issue #4016 Distributional RL PR 1 Design

This note records the first implementation slice for issue
[#4016](https://github.com/ll7/robot_sf_ll7/issues/4016): primitive support for
QR-DQN-style return quantiles over a fixed discrete unicycle command lattice.

## Claim Boundary

This slice is diagnostic infrastructure only. It proves deterministic lattice,
risk-objective, quantile-loss, target-construction, and metadata contracts with
unit tests. It does not train a policy, integrate a benchmark adapter, submit a
GPU or Slurm job, compare against a mean-value baseline, or make a benchmark,
paper, or dissertation claim.

## Contract

- `DiscreteUnicycleActionLattice` stores a deterministic Cartesian product of
  absolute unicycle velocity commands in `unicycle_vw` command space.
- `risk_objectives` scores ordered return quantile estimates with `mean`,
  `var_lower`, `cvar_lower`, and `cvar_blend`.
- `QuantileQNetwork` emits tensors shaped `[batch, action_count, num_quantiles]`.
- `quantile_huber_loss` implements the QR-DQN fixed-quantile regression loss.
- QR-DQN target construction selects the online action index separately from the
  target-network action distribution, leaving full training orchestration to a
  later slice.

## Next Empirical Action

The next slice should add a dry-run-capable QR-DQN training entry point and smoke
configuration that consume these primitives and write non-evidence manifests.
Benchmark comparison, mean-value baseline evaluation, and map-runner integration
remain out of scope until a smoke-trained checkpoint exists.
