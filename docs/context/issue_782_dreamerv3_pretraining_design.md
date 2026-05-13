# Issue 782: DreamerV3 world-model pretraining design

Date: 2026-04-28
Related notes:
- `docs/context/dreamerv3_program_full_handoff_2026_04_28.md`
- `docs/context/issue_578_608_609_dreamerv3_parity.md`
- `docs/context/issue_789_dreamer_multimodal_encoder.md`

> Update 2026-05-14: the follow-up import-boundary probe is recorded in
> [issue_1190_dreamerv3_checkpoint_import_boundary.md](issue_1190_dreamerv3_checkpoint_import_boundary.md).
> It fails closed on Ray 2.53.0 because Robot SF has no clean world-model import contract beyond
> RLlib's full Algorithm/RLModule checkpoint restore surface.

## Goal

Decide whether this repository should pursue DreamerV3 world-model pretraining for the
BR-08 challenger path, and if so, choose the smallest proof-first route that can fail
closed without forking RLlib prematurely.

## Reusable rollout sources in this repo

| Source | What exists today | Observation fields | Action fields | Reward fields | Storage format | Reuse assessment |
| --- | --- | --- | --- | --- | --- | --- |
| PPO checkpoints under `output/model_cache/ppo_*` | Approved SB3 PPO checkpoints such as `output/model_cache/ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200/model.zip` and issue-791 promotion candidates, plus summary metadata like `best_summary.json` and `model/registry.yaml` entries | None stored in the checkpoint itself; replayable through the repo's PPO rollout collector, which records raw env observations exactly as emitted by the env contract used for the source checkpoint | None stored in the checkpoint itself; collector exports per-step env actions | Only aggregate eval metrics in summary files; no step-level reward stream is persisted in the checkpoint artifact | SB3 `model.zip` checkpoint, optional `best_summary.json`, registry metadata | Best current teacher source, but only after replay. The checkpoint artifact alone is not an offline Dreamer dataset. |
| PPO trajectory export path | `scripts/training/collect_expert_trajectories.py` can replay a PPO policy and write a governed dataset plus manifest | `observations` object array in NPZ; each element is the raw env observation for one step | `actions` object array in NPZ; one env action per step | Not currently stored in the NPZ dataset; rewards would need an explicit exporter extension | `.npz` with `positions`, `actions`, `observations`, `episode_count`, and `metadata`, plus JSON manifest from `robot_sf.benchmark.imitation_manifest` | Repo-native and reproducible, but reward-free. Good for imitation-style warm starts; incomplete for world-model supervision unless extended. |
| ORCA rollouts checked into this repo | No checked-in ORCA offline trajectory dataset was found under `output/`, `docs/`, or tracked artifact roots during this handoff pass | None available as committed data | None available as committed data | None available as committed data | No governed dataset present | Not usable today. A new exporter would be required before ORCA can contribute offline Dreamer data. |
| Scripted planners under `robot_sf/nav/` | Route, occupancy, and map utilities such as `global_route.py`, `navigation.py`, and `occupancy_grid.py` | Planner inputs are map/route/occupancy primitives, not a standardized stepwise Dreamer observation tensor | No repo-native env-step action log is stored; outputs are route/planning intermediates | No reward contract outside the environment loop | Python modules only; no committed rollout artifact | Not a drop-in pretraining dataset. These utilities can seed future teacher policies, but they are not themselves offline Dreamer training data. |

Supplementary artifact surfaces already exist but are still secondary to the table above:

- `robot_sf/render/jsonl_recording.py` provides per-step JSONL state recording with sidecar metadata.
  It is useful for replay and debugging, but its default schema captures state snapshots rather than
  a full Dreamer-ready `(obs, action, reward, continuation)` dataset.
- `robot_sf/telemetry/pane.py` records telemetry JSONL streams, again useful for diagnostics but not
  a complete world-model training corpus.

## Option comparison

### Option A: RLlib offline-rollout ingestion

Description:
Use RLlib's offline episode ingestion path and feed DreamerV3 with replayed PPO episodes exported
from this repo.

Pros:
- Keeps training inside RLlib's existing DreamerV3 stack.
- Reuses the repo's trajectory-export path rather than inventing a new artifact format.

Cons:
- The current repo-native trajectory export does not store reward or continuation terms, so the
  collector would need to grow before it can supervise a Dreamer world model faithfully.
- This still does not solve the current multimodal limitation documented in
  `docs/context/issue_789_dreamer_multimodal_encoder.md`; Ray 2.53.0 DreamerV3 still expects a
  plain `Box` observation space.
- Proof-first burden is high because the first milestone would already require a new dataset schema,
  export path, and RLlib offline plumbing before any BR-08 benefit is measurable.

Assessment:
Not recommended as the first follow-up. It is reproducible in principle, but it expands the data
surface before the checkpoint/import boundary is understood.

### Option B: World-model weight export/import adapter

Description:
Treat RLlib DreamerV3 checkpoints as the canonical artifact, probe whether the world-model weights
can be exported/imported cleanly, and only then attempt a short BR-08 fine-tune from a pretrained
state.

Pros:
- Smallest scope that can answer the real question: does pretraining help BR-08 enough to justify
  more compute?
- Aligns with the repo's fail-closed rule: if the checkpoint boundary is not clean, the effort can
  stop without forking RLlib or changing benchmark claims.
- Reuses existing checkpoint and gate/full config surfaces.

Cons:
- RLlib does not document a DreamerV3-specific world-model import boundary.
- The probe must inspect RLModule or checkpoint state contents carefully; if the only path is a
  custom Dreamer catalog/module fork, this option should stop and convert to a follow-up issue.

Assessment:
Recommended conditional path. Pursue this only if the checkpoint boundary is demonstrably clean.

### Option C: External representation model

Description:
Train or import a separate encoder/world-model outside RLlib and bridge its features back into the
 Dreamer or policy stack.

Pros:
- Most flexible path for multimodal or custom representation work.

Cons:
- Highest integration burden and the weakest fit for the current proof-first scope.
- Changes the model-data path below the current launcher contract and would need new provenance,
  validation, and likely new artifacts.
- This is exactly the sort of scope creep the handoff note warns against while #578 remains an
  unproven challenger.

Assessment:
Defer. This belongs in a separate research/design track, not the immediate BR-08 unblock path.

## Recommendation

Recommend **Option B** with a hard fail-closed caveat:

- proceed only if RLlib DreamerV3 exposes a clean checkpoint import/export boundary for the world
  model or its containing RLModule,
- stop immediately and convert the effort into a follow-up issue if the first working path requires
  a Dreamer-specific RLlib fork, custom catalog rewrite, or checkpoint surgery that cannot be
  validated on a small gate run.

## Minimal gate experiment

Use one bounded probe before any large retraining budget:

1. Start from a successful Dreamer gate checkpoint produced by the existing BR-08 gate config.
2. Implement a standalone adapter probe that restores checkpoint state and attempts to import the
   world-model weights into a fresh BR-08 gate run.
3. Fine-tune for **100k env steps** on the gate profile with the same seed family used for the
   from-scratch comparison.
4. Compare against a from-scratch 100k-step gate run on:
   - `eval/success_rate`
   - `eval/collision_rate`
   - `episode_return_mean`
5. Treat the result as positive only if success improves without a compensating safety regression.

## Stop conditions

- **Checkpoint boundary stop**: if Ray/RLlib does not expose a clean world-model import boundary,
  do not fork DreamerV3 under #782.
- **Performance stop**: if the 100k-step fine-tune shows no measurable improvement in
  `eval/success_rate` and no safety improvement, record a no-action decision instead of scaling the
  experiment.
- **Data-contract stop**: if the team decides rewards/continuation labels must come from offline
  rollouts first, open a dedicated data-export issue rather than expanding #782 silently.

## Decision

Recommended follow-up implementation issue:

`DreamerV3 checkpoint import boundary probe for BR-08 gate`

Acceptance criteria for that follow-up should be:

- list the exact checkpoint tensors/modules needed,
- demonstrate import into one fresh gate run without a local RLlib fork,
- compare imported-vs-scratch 100k-step gate metrics,
- close with either a measured improvement or a no-action recommendation.

If that issue is not opened, the correct decision for #782 is **no action for now**.
