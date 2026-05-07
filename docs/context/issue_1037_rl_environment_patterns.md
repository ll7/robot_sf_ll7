# Issue #1037 RL Environment Patterns For Robot SF

## Scope

Issue: [#1037](https://github.com/ll7/robot_sf_ll7/issues/1037)

Primary source: Hugging Face Space
[RL Environments Guide](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide),
last updated May 2026. The rendered Space is dynamic, so this note used the Space repository
chapter files under `app/src/content/chapters/` as the review source.

This is a design note, not an implementation change. It maps LLM-era RL environment patterns onto
the existing Robot SF training and benchmark stack and records which ideas should be adopted,
deferred, or rejected.

## HF Guide Summary

The guide's useful framing is that modern RL systems share the same broad spine:
task source, interaction harness, reward signal, rollout collector, and trainer. The differences
between frameworks are mostly about where that boundary is drawn and which pieces are bundled.

The parts that transfer to Robot SF are the design axes, not the LLM-specific frameworks:

- Environment boundary: in-process objects versus HTTP/session services.
- Reward ownership and timing: external reward, environment-owned reward, or post-episode verifier.
- Rollout ownership: trainer-driven loops versus harness-controlled rollout execution.
- Task coupling: loose scenario rows versus versioned task bundles with metadata and artifacts.
- Scaling model: Python objects inside the trainer versus independently scaled environment servers.
- Provenance: task, reward, harness, config, and trainer choices must be recorded together to make a
  run replayable.

The guide also highlights a non-transferable trend for Robot SF: LLM task environments often expose
tools, sandboxes, file trees, and post-hoc verifiers. Robot SF instead has physics state,
continuous actions, Gymnasium step semantics, and benchmark schemas. That makes framework adoption
less useful than tightening the local contract.

## Current Robot SF Boundary Map

Robot SF already uses a classical, in-process Gymnasium-style environment boundary.

Task source:
`configs/`, scenario config files, scenario loaders, and benchmark matrices. A task is a
scenario/config tuple plus seed and map assets, not an LLM prompt row.

Environment harness:
`robot_sf/gym_env/environment_factory.py`. Public creation goes through `make_robot_env`,
`make_image_robot_env`, and `make_pedestrian_env`.

Trainer-driven rollout:
`scripts/training/train_ppo.py` and `scripts/training/train_dreamerv3_rllib.py`. SB3/RLlib own the
training loop and drive vectorized envs.

Benchmark-driven rollout:
`robot_sf/benchmark/runner.py` and `robot_sf/benchmark/cli.py`. The benchmark harness owns
evaluation episodes and writes schema-checked JSONL.

Reward signal:
`robot_sf/gym_env/reward.py` and training YAML `env_factory_kwargs`. Dense training rewards live
inside the Gym env through named reward profiles and curricula.

Verifier/outcome:
`robot_sf/benchmark/schemas/episode.schema.v1.json`, metrics/SNQI, and termination policy.
Benchmark success is an evaluation contract, separate from training reward.

Provenance:
training manifests, benchmark `algorithm_metadata`, `config_hash`, and issue notes. Provenance
exists, but the reward/harness/task boundary is not documented in one compact run-record checklist.

Scaling:
SB3 vector envs plus benchmark process workers and resume. Current scaling is in-process and
process-based, not a remote environment service.

The existing docs support this map:

- `docs/dev_guide.md` requires factory functions for environment creation and identifies the
  training and benchmark data flows.
- `docs/context/issue_691_benchmark_fallback_policy.md` keeps benchmark outcomes fail-closed and
  prevents fallback/degraded execution from counting as benchmark success.
- `robot_sf/benchmark/schemas/episode.schema.v1.json` already records episode identity,
  scenario identity, metrics, outcome, integrity, config hashes, and adapter/execution metadata.
- `scripts/training/train_ppo.py` already logs the scenario config and resolved reward profile, uses
  vectorized envs, and writes training run manifests.

## Recommendations

Adopt: Document a compact Robot SF environment contract.
The repo has the contract in code and scattered docs, but future training and benchmark work would
benefit from one canonical map of factory inputs, reset/step semantics, `info` metadata, reward
ownership, termination, and rollout ownership.

Adopt: Add a training run-record checklist for reward and environment provenance.
Training manifests exist, but reviewers need a quick checklist that names the scenario source,
env factory kwargs, reward profile/curriculum, seed policy, vectorization mode, git/config hashes,
and evaluation scenario.

Adopt: Keep training reward separate from benchmark verifier/outcome semantics.
Dense reward is useful for learning, while benchmark outcomes must remain schema-checked,
fail-closed, and comparable across planners. Blending these concepts would weaken benchmark claims.

Defer: Treat task bundles as a future scenario-dataset packaging idea.
Robot SF scenarios are already config-first. A stricter bundle format may help if scenario
families, maps, model artifacts, and evaluation splits become harder to replay across machines.
It is not needed for issue #1037.

Defer: Introduce HTTP or MCP-style environment services.
The current bottlenecks are not proven to be environment-service orchestration. In-process
Gymnasium envs match SB3/RLlib and local benchmark runners. Revisit only if remote scaling,
sandbox isolation, or cross-language environment reuse becomes an actual blocker.

Reject: Adopt an LLM-era RL environment framework wholesale.
Verifiers, OpenEnv, ORS, NeMo Gym, SkyRL Gym, and GEM solve tool-calling or sandboxed LLM tasks.
Robot SF needs physics simulation, continuous control, deterministic benchmark identity, and
planner/provenance contracts.

Reject: Use LLM-as-judge rewards for benchmark outcomes.
Robot SF benchmark outcomes should stay deterministic and schema-backed. Subjective judge rewards
would add an avoidable source of drift and reward hacking risk.

Reject: Treat fallback or degraded execution as successful benchmark evidence.
This is already ruled out by the fail-closed benchmark fallback policy.

Reject for now: Move rollout ownership into an environment-owned generation loop.
SB3/RLlib training and benchmark harnesses both rely on external rollout ownership. Flipping that
model would be a major architecture change without a demonstrated need.

## Follow-Up Split

Implementation work should stay out of #1037 and land as smaller follow-up issues:

1. [#1039](https://github.com/ll7/robot_sf_ll7/issues/1039): document the Robot SF environment
   contract and rollout ownership map.
2. [#1040](https://github.com/ll7/robot_sf_ll7/issues/1040): add a training reward/provenance
   run-record checklist.
3. [#1041](https://github.com/ll7/robot_sf_ll7/issues/1041): clarify the
   reward-versus-benchmark-verifier boundary in the docs.

These follow-ups are documentation and reviewability work first. Code changes should only follow if
the docs expose a concrete missing manifest field or ambiguous runtime contract.

## Validation

This note is validated by checking the referenced local paths and the Hugging Face chapter source
files used for the review. No benchmark, training run, or model artifact is produced by this issue.
