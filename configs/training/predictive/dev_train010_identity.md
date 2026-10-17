# train010 development restart

The paired PPO configs `../ppo/dev_train010_fixed_5m.yaml` and
`../ppo/dev_train010_safety_5m.yaml` retain PR #10247's 5M budgets, fixed objective
and stronger safety objective. Both use current-main runtime, the development
geometry, construction/reset pool 1001–1020 and selection pool 1021–1030.
They establish bounded reward comparisons, not a static-clearance reward fix or
completion of #9628's curriculum correction programme.

`dev_train010_identity.yaml` recollects base and hardcase data, using the reviewed
identity/velocity collectors from #10282/#10011. The explicit scenario seed
manifests use collection seeds 1001–1015; evaluation seeds are 1016–1030. Both
predictive consumers receive new model IDs. Forecast length is eight steps;
evaluation must never request a longer horizon. Supervision metadata must match
between fresh base, hardcase and mixed NPZ files; legacy rows are refused.
Main now contains the native swept-footprint static-wall veto (commit c8dbd4268c).
Its development evaluation with these new checkpoints remains outstanding.

PPO runs retain 20 environments and 256-step rollouts; no convergence early stop.
Each predictive model runs 220 epochs with validation-loss checkpoint selection;
ADE/FDE gates are dataset diagnostics and do not establish navigation improvement.
No test-band or release admission is authorized.

Every execution archives its source commit/archive, resolved configuration,
scenario/seed manifests, package versions, CUDA identity, driver script digest,
W&B URLs, checkpoints and hashes in the external data hub. Online W&B logging
is required. Preservation snapshots include regular evidence files and exclude
W&B runtime directories and symlinks. The strict preservation validator is
unchanged. This prevents the SDK symlinks that aborted the previous post-training
preservation from entering the upload payload. The lane driver and scheduler
scripts are archived as operational artifacts rather than public host policy.

Training remains an experiment from an unmerged configuration branch; its PR
requires independent review. No model registry or released artifact is replaced.

The open pedestrian-model, grid-rasterization and planner-hygiene dependencies
remain outside this restart. Training on current main produces development
checkpoints only; it does not resolve those issue-level admission prerequisites.
