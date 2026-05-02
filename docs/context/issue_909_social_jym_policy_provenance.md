# Issue #909 Social-Jym Trained-Policy Provenance

Date: 2026-05-02
Related issues:
- <https://github.com/ll7/robot_sf_ll7/issues/909>
- <https://github.com/ll7/robot_sf_ll7/issues/907>
- <https://github.com/ll7/robot_sf_ll7/issues/905>
- <https://github.com/ll7/robot_sf_ll7/issues/792>

## Goal

Check whether the pinned upstream `social-jym` checkout contains, references, or can directly load
a trained SARL/SARL-PPO policy artifact that would justify moving from wrapper/parity probes toward
Robot SF benchmark smoke coverage.

This is trained-policy provenance evidence only. It does not add benchmark support.

## Source Revision

The investigation uses the same ignored source checkout proven in issue #792:

- source checkout: `output/repos/social-jym`
- root repository: `https://github.com/TommasoVandermeer/social-jym.git`
- root revision: `212ea7759f614ff646f39462c4f51dca67d01ed0`

The checkout is worktree-local under `output/` and is not a durable dependency.

## Artifact Search

Concrete upstream SARL/SARL-PPO paths checked:

- `output/repos/social-jym/tests/sarl_params.pkl`
- `output/repos/social-jym/tests/sarl_ppo_params.pkl`
- `output/repos/social-jym/tests/best_sarl.pkl`
- `output/repos/social-jym/tests/metrics_sarl_vs_sarl_ppo.pkl`
- `output/repos/social-jym/scripts/metrics_tests_with_static_humans.pkl`
- `output/repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_cc/rl_model.pth`
- `output/repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_ccso/rl_model.pth`

All of those paths were missing in the pinned checkout on this machine.

The broader file search for likely policy artifacts under `output/repos/social-jym` found only
source files, tests, notebooks, and non-policy custom configuration pickles. No `.pth`, `.pt`,
`.ckpt`, `.safetensors`, or SARL/SARL-PPO parameter pickle was present.

## Loading Contracts Found

`socialjym.utils.aux_functions.save_policy_params` writes `social-jym` policy artifacts as pickle
files containing:

- `policy_name`
- `policy_params`
- `train_env_params`
- `reward_params`
- `hyperparameters`

`socialjym.utils.aux_functions.load_socialjym_policy(path)` then reads `policy_params` from that
pickle structure.

The source also has `load_crowdnav_policy(policy_name, path)`, which imports PyTorch and converts
CrowdNav `.pth` value-network weights into JAX/Haiku parameter dictionaries. The upstream script
`scripts/acm_thri_static_obstacles_trials.py` references local home-directory paths:

- `~/Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_cc/rl_model.pth`
- `~/Repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_ccso/rl_model.pth`

Those referenced artifacts are not in the pinned checkout.

The upstream comparison test `tests/test_sarl_against_sarl_ppo.py` expects local fixture files:

- `tests/sarl_params.pkl`
- `tests/sarl_ppo_params.pkl`

Those fixture files are also absent.

## Interpretation

No benchmark-facing trained SARL/SARL-PPO policy artifact is currently reproducible from the pinned
source checkout alone.

The repository can safely claim:

- source-harness reproduction for a minimal `SocialNav` reset and random SARL policy step;
- one-step Robot SF wrapper smoke with randomly initialized upstream SARL parameters;
- controlled one-human SARL input parity for wrapper-built source observations and VNet inputs.

The repository cannot safely claim:

- trained upstream SARL or SARL-PPO policy availability;
- source-policy quality parity;
- source benchmark reproduction with trained policy weights;
- Robot SF benchmark support for `social-jym`.

## Kinematics Boundary

Issue #907 showed that controlled SARL input parity can be exact for the tested state, but
holonomic source actions lose semantics when projected into Robot SF `unicycle_vw` commands.
Lateral, reverse, and diagonal holonomic actions are not equivalent after projection.

Because trained-policy provenance is absent, there is no evidence to decide whether a future
benchmark comparison should accept the current unicycle projection, add a holonomic compatibility
surface, or exclude trained `social-jym` policies until source-faithful action semantics are
available.

## Validation Commands

Targeted missing-artifact check:

```bash
for p in \
  output/repos/social-jym/tests/sarl_params.pkl \
  output/repos/social-jym/tests/sarl_ppo_params.pkl \
  output/repos/social-jym/tests/best_sarl.pkl \
  output/repos/social-jym/tests/metrics_sarl_vs_sarl_ppo.pkl \
  output/repos/social-jym/scripts/metrics_tests_with_static_humans.pkl \
  output/repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_cc/rl_model.pth \
  output/repos/social-jym/trained_policies/crowdnav_policies/sarl_5_hsfm_ccso/rl_model.pth
do
  if [ -e "$p" ]; then echo "found:$p"; else echo "missing:$p"; fi
done
```

Broader local artifact search:

```bash
find output/repos/social-jym -type f \
  \( -iname '*sarl*' -o -iname '*rl_model*' -o -iname '*policy*' \
     -o -iname '*params*' -o -iname '*trained*' \) -print
```

Source loading-contract inspection:

```bash
sed -n '1480,1565p' output/repos/social-jym/socialjym/utils/aux_functions.py
sed -n '1,125p' output/repos/social-jym/tests/test_sarl_against_sarl_ppo.py
sed -n '1,105p' output/repos/social-jym/scripts/acm_thri_static_obstacles_trials.py
```

## Recommendation

Recommendation: `trained-policy provenance absent; benchmark smoke not justified`

Keep the `social-jym` row as conceptually adjacent only. A future issue may reopen benchmark smoke
work only after a durable trained artifact source is identified, licensed, version-pinned, and
loaded through a reproduced source command. Until then, `social-jym` benchmark outcomes should fail
closed rather than falling back to random, untrained, missing, or projection-only execution.
