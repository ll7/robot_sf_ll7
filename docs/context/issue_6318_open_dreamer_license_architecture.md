# Issue #6318 — Open Dreamer license determination and architecture study (Gate 0)

Date: 2026-07-26
Status: **Gate 0 complete — clean-room route required; no upstream code copied, vendored, or adapted.**
Evidence tier: `idea` (docs-only license/architecture determination; no execution, no policy, no benchmark evidence).
Parent issue: [#6318 — research: test Open Dreamer-style latent imagination for stronger Robot SF policies](https://github.com/ll7/robot_sf_ll7/issues/6318).
Sequencing: this is **Step 1 (Gate 0, compute-free)** of the maintainer-authorized plan. It is a hard predecessor for the adapter (Step 2), the model-quality gate (Step 3), and the three matched SAC arms (Step 4). Those steps are blocked until Gate 0 closes and remain out of scope for this note.

## Plain-language summary

Open Dreamer is an appealing source of *ideas* for action-conditioned latent world models, but at
the pinned commit its license reserves all rights and grants no permission to copy, vendor, or
adapt the code. Gate 0 therefore concludes that the only permissible route for Robot SF right now is
a **clean-room reimplementation** derived from the public paper and architecture description, not
from the upstream source. If a formal license or explicit permission is later obtained and recorded,
the route widens. This note records that determination and the supporting architecture study so
Steps 2–4 have an auditable provenance boundary. It makes no policy, benchmark, or paper claim.

## Pinned upstream commit consulted

- Repository: [`next-state/open-dreamer`](https://github.com/next-state/open-dreamer) (default branch
  `main`).
- Pinned commit: [`5a4127fcbbd37b3cd9237904d645fb11348bab28`](https://github.com/next-state/open-dreamer/tree/5a4127fcbbd37b3cd9237904d645fb11348bab28)
  (commit date 2026-07-25T15:34:15Z; commit subject "Fix blog byline layout, citations, and author
  credits"; repository `pushed_at` 2026-07-25T15:34:18Z).
- Public surfaces consulted for this study: the repository `README.md`, the `LICENSE` file, the
  published `## Roadmap` section, and the documented repository layout. The Dreamer 4 algorithmic
  reference is [arXiv:2509.24527](https://arxiv.org/abs/2509.24527),
  "Training Agents Inside of Scalable World Models" (Hafner, Yan, Lillicrap — the Dreamer 4 paper).

**Provenance boundary (hard).** This study is derived from those public documentation surfaces and
the Dreamer 4 paper. It is **not** derived from line-by-line reading of upstream source. No upstream
Open Dreamer source code is copied, vendored, adapted line-by-line, paraphrased as code, or
committed anywhere in this change or in Robot SF. The capabilities described below are the public
README/roadmap claims and the paper's described mechanisms, restated in Robot SF's own words for the
purpose of deciding a license route.

## License determination (the Gate 0 result)

The upstream [`LICENSE`](https://github.com/next-state/open-dreamer/blob/5a4127fcbbd37b3cd9237904d645fb11348bab28/LICENSE)
at the pinned commit reads, in its operative parts:

> Copyright (c) 2026 The Open Dreamer Authors. All rights reserved.
>
> All rights in and to this software and its associated files are reserved by the copyright holders.
> No license or permission is granted, whether express or implied, to use, copy, modify, merge,
> publish, distribute, sublicense, or sell copies of this software, in whole or in part, without the
> prior written consent of the copyright holders.
>
> This notice is provisional: it is expected to be replaced by a formal license in a future release.

Determination:

- **Status at pinned commit: all rights reserved.** The license grants no permission to copy, modify,
  merge, distribute, sublicense, or adapt the software. GitHub's repository metadata reports the
  license as `NOASSERTION` / `Other`, consistent with a non-OSI all-rights-reserved notice.
- The notice is explicitly *provisional* and is "expected to be replaced by a formal license in a
  future release." That means the license may widen later, but it has not widened at this commit.
  Robot SF cannot rely on a future license that does not yet exist.
- The repository's own self-description ("open-source Dreamer world-model implementation in JAX") is
  an aspiration, not a present license grant. The operative document is the `LICENSE` file, which
  reserves all rights.

This confirms issue #6318's stated preflight expectation ("The current upstream `LICENSE` reserves
all rights") and the maintainer's 2026-07-25 Gate 0 checklist item.

## Route decision

**Route taken: clean-room reimplementation, effective immediately for all Robot SF work under
issue #6318.**

- Until and unless a formal license is published at a pinned upstream commit, **or** explicit
  written permission is obtained from the Open Dreamer copyright holders and recorded in this
  repository, the only permissible route is clean-room: Robot SF contributors and agents may read the
  Dreamer 4 paper and the public upstream *documentation* (README, roadmap, blog/website) for
  architectural ideas, and then write Robot SF's own implementation from those ideas.
- Robot SF contributors and agents must **not** read upstream source files line-by-line and reproduce
  them, must not vendor upstream files, and must not commit adapted upstream code. Reading the
  public `LICENSE`, `README.md`, and roadmap (as this note does) is permissible because those are
  documentation surfaces, not source.
- If permission or a formal license is later obtained, record it here (commit hash, license text or
  permission statement, date, grantor) and the route widens to direct reuse under the recorded
  terms. Until then, treat any "just import upstream" shortcut as blocked by this note.

This route is auditable: any future Step 2–4 PR can point back to this note to show which route was
taken and why.

## Upstream current capabilities (architecture study from public docs)

These are the capabilities the upstream `README.md` and documented repository layout claim at the
pinned commit, restated in Robot SF's own words. They describe *what the public project says it
does*, not a line-by-line audit of its source.

- **Causal video tokenizer.** A tokenizer that learns a discrete latent code from raw video clips
  and can tokenize full episodes into latent ArrayRecord shards. Upstream trains and evaluates this
  on Minecraft/VPT-style video.
- **Action-conditioned latent sequence dynamics.** A dynamics model that operates over the tokenized
  latent sequence and is conditioned on aligned actions, i.e. it predicts future latent states as a
  function of past latent state and action. The Dreamer 4 paper (arXiv:2509.24527) attributes the
  real-time quality to a *shortcut forcing* objective and an efficient transformer architecture; the
  upstream README describes the dynamics model as part of the world-model training pipeline.
- **Autoregressive latent rollouts with cached temporal state.** The generation path produces
  rollouts autoregressively from a starting latent context plus an action sequence, reusing cached
  temporal state across steps so that multi-step imagination is cheaper than re-encoding from
  scratch. This cached-state rollout is the upstream mechanism that makes short-horizon imagination
  attractive for RL.
- **Checkpointed tokenizer/dynamics bundles.** Upstream uses Orbax checkpoint bundles so a trained
  tokenizer and a trained dynamics model can be saved, reloaded, and paired. This is relevant to
  Robot SF because it implies a two-stage artifact boundary (tokenizer checkpoint + dynamics
  checkpoint) that a clean-room reimplementation would need to reproduce in Robot SF's own
  serialization.
- **JAX/Flax (NNX) data-parallel training.** The implementation targets a CUDA-12 JAX environment
  and is structured for data-parallel training across accelerators.
- **Simple offline episode format.** Upstream consumes offline episodes that pair observations/video
  latents with aligned actions (ArrayRecord shards). There is no online interaction loop in the
  released training pipeline.

In short, at the pinned commit Open Dreamer is a **world-model training and rollout pipeline**, not
a reinforcement-learning agent.

## RL-agent pieces Open Dreamer lacks at the pinned commit

This is the decisive architectural gap for issue #6318. The upstream `README.md` `## Roadmap`
contains exactly one unchecked item:

- [ ] Full Dreamer 4 Behaviour-Cloning / RL agent training loop

Concretely, the released pipeline does **not** provide, at the pinned commit:

- **No reward head.** Nothing predicts scalar reward from latent state, so latent rollouts do not
  produce usable return estimates for a Robot SF policy without Robot SF adding this head itself.
- **No continuation/termination head.** Nothing predicts episode continuation or termination from
  latent state, so imagined rollouts cannot model `terminated`/`truncated` semantics without an
  added head.
- **No value/critic or actor interface.** There is no value function, no actor/critic training loop,
  and no policy-improvement step in the released training path.
- **No BC/RL agent loop.** The single roadmap item above is unchecked, confirming the
  behaviour-cloning and reinforcement-learning agent training loop is unfinished and not released.
- **No environment/agent interaction loop.** Training is offline over recorded episodes; there is no
  online interaction harness that a Robot SAC policy could plug into.

Implication for Robot SF: even on a permission-cleared route, Open Dreamer would only be a
*dynamics-model ingredient*. The reward, continuation, value, and actor pieces required to turn
imagined rollouts into stronger Robot SF policies would have to be built in Robot SF regardless
(issue #6318 Steps 2–4). This is consistent with the issue's own framing.

## Relationship to the retired DreamerV3 campaign (boundary)

This Gate 0 is deliberately shaped by the prior world-model decision so it does not repeat it.

- Issues [#1623](https://github.com/ll7/robot_sf_ll7/issues/1623) and
  [#782](https://github.com/ll7/robot_sf_ll7/issues/782) retired a flat-vector DreamerV3 campaign
  after repeated NaNs, no evaluation signal, and a run consuming roughly 106 GB host RAM against a
  64 GB request. The close-out record is
  [dreamerv3_program_close_out_2026_04_30.md](dreamerv3_program_close_out_2026_04_30.md).
- The maintainer authorization for #6318 is explicitly for a *different* design — small, repo-owned,
  structured-observation-first, gate-driven — and is **void** if the work drifts back toward RLlib,
  flat-vector observations, pixel tokenizers, or any run whose memory envelope is not predeclared and
  enforced.
- Gate 0 adds a further guardrail the old campaign did not have: a license/provenance gate that
  fires *before* any adapter or model code is written. This note is that gate.

## Mapping to Robot SF surfaces (descriptive only; Gate 0 makes no code changes)

This Gate 0 note makes **no edits** to code, configs, maps, model, or tests. The mapping below is
descriptive, to give Steps 2–4 a clear target surface; it is not an implementation and creates no
claim.

- `RLTrajectoryDataset.v1` (`robot_sf/benchmark/rl_trajectory_dataset.py`) stores episode sequences
  with observations, actions, rewards, return-to-go, terminal/truncated flags, pedestrian and robot
  state, split assignment, and provenance. A future Step 2 episode-major adapter would consume this
  contract, preserving reward, terminated/truncated, pedestrian/robot state, action semantics, and
  provenance with leakage-safe scenario/seed splits.
- The offline-to-online SAC workflow and `HybridReplayBuffer`
  (`robot_sf/training/hybrid_replay_buffer.py`) already support a paired offline-to-online versus
  from-scratch diagnostic, which is the comparison surface a future Step 4 imagined-replay bridge
  would plug into (with an explicit source marker, uncertainty/error gate, and configurable
  imagined-to-real ratio).
- The action contract differs from upstream: Robot SF uses a bounded two-dimensional continuous
  action (linear velocity, angular velocity) after the existing `[-1, 1]` adapter, not the upstream
  VPT-style binary/categorical/continuous camera action container. A clean-room dynamics model would
  be built around Robot SF's action contract from the start.

These surfaces are referenced for orientation only. They are out of scope for this PR and are listed
under the immutable contract's forbidden paths precisely to keep Gate 0 docs-only.

## Sequenced plan and where Gate 0 sits

Per the maintainer's 2026-07-25 authorization:

1. **Step 1 — Gate 0 (compute-free).** License determination + this architecture study note. Blocks
   everything below. ← **this note is Step 1.**
2. **Step 2 — adapter contract smoke (compute-light).** Episode-major adapter from
   `RLTrajectoryDataset.v1` with leakage-safe splits; structured observations only (group-aware
   encoder over `drive_state` + `rays`); do not flatten an occupancy grid into fake video channels.
3. **Step 3 — model quality gate (compute-light).** Holdout one-step and short multi-step prediction
   for observation, reward, and continuation versus a persistence/MLP predictor. The model must beat
   the simple predictor here or the direction is rejected.
4. **Step 4 — three matched SAC arms (compute-bearing).** Scratch SAC vs offline-to-online SAC vs
   model-imagination SAC, identical reward profile (`route_completion_v3`), observation contract,
   environment steps, update budget, scenario matrix, and seeds.

Gate 0's stop rule: if clean-room had been judged intractable within the bounded scope, this note
would have recorded an architecture-only no-go and stopped. Clean-room is judged **tractable** for
the bounded scope: the Dreamer 4 paper and the public upstream documentation describe the
mechanisms at enough altitude to write a small structured-observation dynamics model in Robot SF's
own code, and Robot SF already owns the trajectory and SAC surfaces the model would attach to.
Tractable does **not** mean low-effort, and it is not a policy result.

## Caveats and claim boundary

- **Evidence tier is `idea`.** This note is a license/architecture determination only. It is not
  benchmark evidence, not policy evidence, not paper-grade, and not a claim that a clean-room Open
  Dreamer-style model will improve any Robot SF policy.
- **Capability descriptions are upstream's public claims, restated.** They reflect the pinned
  commit's README/roadmap and the Dreamer 4 paper. They are not a line-by-line verification of
  upstream source, and they must not be treated as a verified contract for a Robot SF
  reimplementation; Step 2–4 work must define and test its own contract.
- **License may change.** The all-rights-reserved notice is explicitly provisional. If a future
  pinned commit ships a permissive license, re-run Gate 0 against that commit and record the new
  route here. Do not assume the route from this note persists across upstream releases.
- **Uncertainty.** The license determination itself is high-confidence (the LICENSE text is
  unambiguous). The "clean-room is tractable" judgment is a medium-confidence scoping estimate
  (roughly 70–80%): the mechanisms are well-described publicly, but Step 2–4 effort and the
  model-quality-gate outcome remain unknown and could still force a no-go later.

## Sources consulted (all public, no upstream source read line-by-line)

- Upstream pinned commit tree: <https://github.com/next-state/open-dreamer/tree/5a4127fcbbd37b3cd9237904d645fb11348bab28>
- Upstream `LICENSE` (all rights reserved, provisional): <https://github.com/next-state/open-dreamer/blob/5a4127fcbbd37b3cd9237904d645fb11348bab28/LICENSE>
- Upstream `README.md` (capabilities, roadmap, repository layout): <https://github.com/next-state/open-dreamer/blob/5a4127fcbbd37b3cd9237904d645fb11348bab28/README.md>
- Dreamer 4 paper (algorithmic reference): Danijar Hafner, Wilson Yan, Timothy Lillicrap, "Training
  Agents Inside of Scalable World Models," [arXiv:2509.24527](https://arxiv.org/abs/2509.24527).
- Robot SF prior world-model decision: [dreamerv3_program_close_out_2026_04_30.md](dreamerv3_program_close_out_2026_04_30.md)
  (issues #1623, #782).

## Follow-up

- Steps 2–4 may now reference this note for provenance. Each of those PRs must state its own route
  (clean-room vs, if applicable by then, permission-cleared) and must not copy upstream code.
- If a formal license or explicit permission is obtained, update this note's "Route decision"
  section with the grantor, date, and recorded instrument, then widen the route.
- If Step 2–4 work finds the public descriptions insufficient to build a clean-room model, record
  that as a Step 2 blocker on the issue rather than silently falling back to reading upstream source.
