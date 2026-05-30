# Issue 1623 World-Model Navigation Feasibility

Related issue: [#1623](https://github.com/ll7/robot_sf_ll7/issues/1623)
Prior Dreamer close-out: [dreamerv3_program_close_out_2026_04_30.md](dreamerv3_program_close_out_2026_04_30.md)
Checkpoint boundary probe: [issue_1190_dreamerv3_checkpoint_import_boundary.md](issue_1190_dreamerv3_checkpoint_import_boundary.md)
Pretraining design: [issue_782_dreamerv3_pretraining_design.md](issue_782_dreamerv3_pretraining_design.md)
LiDAR learned-policy launch plan: [issue_1615_lidar_learned_policy_plan.md](issue_1615_lidar_learned_policy_plan.md)
Pedestrian env consolidation: [issue_1291_pedestrian_env_consolidation.md](issue_1291_pedestrian_env_consolidation.md)
Policy-search registry:
[reject_monitor_registry.md](policy_search/reject_monitor_registry.md)

## Decision

Recommendation: `monitor_external_world_models_reject_local_retrain`.

Do not open another Robot SF flat-vector DreamerV3, PlaNet, or TD-MPC2 training campaign now. The
near-term useful work is source and artifact monitoring, plus small adapter/data-contract
preflights only when a specific candidate has public implementation, license, checkpoint or
training recipe, and a Robot SF observation/action reduction that can fail closed.

This is not a claim that world models are poor for navigation. It is a repository-fit decision: the
Robot SF DreamerV3 track already failed its signal/resource gates, the current RLlib boundary does
not expose a clean Robot-SF-owned world-model import contract, and the durable local offline data
surface does not yet contain the full transition tuple needed for robust world-model pretraining.

## Candidate Comparison

| Candidate | Fit to Robot SF local navigation | Current blocker | Recommendation |
| --- | --- | --- | --- |
| DreamerV3 through the existing RLlib launcher | Strongest local surface: config-first launcher, Ray/RLlib tests, scenario-matrix parity, and BR-08 configs already exist. | The program close-out records repeated NaN/no-eval evidence plus a host-RAM leak/OOM failure, and #1190 shows no clean world-model import boundary. | Treat flat-vector BR-08 as retired; monitor only for a cleaner RLlib or repo-owned import boundary. |
| PlaNet / latent dynamics planning | Conceptually adjacent: learns latent dynamics and plans in latent space. | No local launcher, Robot SF dataset, or adapter surface; older than the repo's current DreamerV3 attempt and would restart the same data/provenance burden from less local infrastructure. | Monitor as background literature, not a Robot SF implementation issue now. |
| TD-MPC2 / latent MPC | Best conceptual match among external candidates for AMV-style continuous control and future actuation-aware planning, because it combines learned latent dynamics with short-horizon MPC. | Upstream source, license, commands, and checkpoints exist, but Robot SF has no integrated/provenanced checkpoint, source-side reproduction record, or typed observation/action reduction for local navigation. | Monitor; consider a source-reproduction/design issue only after source, license, checkpoint, and reduction proof are concrete for Robot SF. |
| DreamerNav-style multimodal world models | Interesting for richer perception stacks and system-level navigation. | Assumes multimodal/depth/occupancy-style perception and curriculum surfaces outside the current Robot SF local-planner contract. | Monitor only; do not treat it as a restart of retired BR-08 DreamerV3. |
| Existing Robot SF predictive/local planners | Already fit the local planner contract and benchmark provenance rules better than a new world-model track. | They do not provide a general learned latent dynamics model, but they are lower-cost and closer to current benchmark evidence. | Prefer these near-term when improving local navigation or AMV actuation support. |

External anchors used for candidate framing:

- PlaNet: <https://proceedings.mlr.press/v97/hafner19a.html>;
  source/license: <https://github.com/google-research/planet>
- DreamerV3: <https://arxiv.org/abs/2301.04104>
- TD-MPC2: <https://arxiv.org/abs/2310.16828>;
  source/license/checkpoints: <https://github.com/nicklashansen/tdmpc2> and
  <https://huggingface.co/nicklashansen/tdmpc2>

## Local Evidence

The strongest local signal is the existing DreamerV3 history:

- `docs/context/dreamerv3_program_close_out_2026_04_30.md` closes the BR-08 DreamerV3 program
  after probe/gate/full attempts produced no eval signal, NaNs, and an OOM-killed full run.
  The full run reached 106 GB RSS against 64 GB requested memory, with NaNs appearing before the
  OOM, so the stop decision is also about Ray/RLlib/custom-env infrastructure fragility rather than
  only model quality.
- `docs/context/issue_1190_dreamerv3_checkpoint_import_boundary.md` fails closed on warm-starting:
  Ray/RLlib 2.53.0 exposes full Algorithm/RLModule restore, not a stable repo-owned world-model
  import endpoint.
- `docs/context/issue_789_dreamer_multimodal_encoder.md` records that mixed/Dict observation
  support is not available through the current Ray 2.53.0 DreamerV3 catalog path.
- `docs/context/issue_782_dreamerv3_pretraining_design.md` shows that existing trajectory exports
  are useful for imitation-style warm starts but do not currently persist reward and continuation
  labels as a complete world-model dataset contract.
- `docs/context/policy_search/reject_monitor_registry.md` already marks DreamerV3/world-model
  navigation as `reject for now`, with DreamerNav-style multimodal navigation as `monitor only`.
- `docs/context/issue_1615_lidar_learned_policy_plan.md` still lists
  `dreamerv3_lidar_world_model_gate_v1` as a research-only future candidate. This #1623 decision
  narrows that entry: it should not become a training campaign unless it first passes the same
  source/provenance, data-contract, and small preflight gates described here.
- `docs/context/issue_1291_pedestrian_env_consolidation.md` records the current canonical
  `PedestrianEnv` boundary. Future external world-model reductions should use the factory/config
  contracts that survived that consolidation rather than binding to older env-module shapes.

The durable artifact picture matches that recommendation. `model/registry.yaml` and the PPO
trajectory collector support replaying teacher policies, but there is no promoted DreamerV3
checkpoint lineage or world-model-ready offline dataset. Files under `output/` are worktree-local
unless explicitly promoted, so they cannot be treated as future campaign dependencies.

## AMV Actuation Implication

AMV actuation constraints make latent-MPC-style methods more interesting in principle because
planning over a learned dynamics model can account for delayed or constrained control response.
That does not change the immediate recommendation. Before a TD-MPC2-like branch would be credible,
Robot SF would need:

- a typed AMV observation/action interface for the candidate,
- a source-side reproduction or checkpoint provenance record,
- a transition dataset or online training command that records reward, terminal/truncated, and
  actuation-mode metadata,
- fail-closed status reporting for missing source assets or incompatible action semantics.

The lower-risk path is to continue the existing AMV preflight, latency, and learned-policy adapter
work before launching a new world-model family. This note also updates the #1615 LiDAR launch-plan
boundary so the older DreamerV3 LiDAR smoke idea is explicitly preflight-gated rather than a
standing training recommendation.

## Follow-Up Boundary

No new child issue is recommended from #1623 right now. A future child issue is warranted only when
one candidate can be named with concrete source/provenance inputs. The child should be framed as a
preflight, not as a training campaign, for example:

`preflight: reduce TD-MPC2 source contract to Robot SF local navigation`

Acceptance criteria for such a future issue should include:

- exact source URL, license, commit, and checkpoint or training recipe;
- observation/action mapping from Robot SF env state to the candidate input contract;
- missing-data and missing-checkpoint behavior reported as `not_available`, not fallback success;
- one tiny executable smoke path or a documented no-action result.

## Validation

This is a documentation and feasibility decision. Validation for the branch:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
rg -n "monitor_external_world_models_reject_local_retrain|106 GB|1t6gadx5|dreamerv3_lidar_world_model_gate_v1|TD-MPC2|PlaNet|DreamerNav|not_available" docs/context/issue_1623_world_model_feasibility.md docs/context/dreamerv3_program_close_out_2026_04_30.md
```

The evidence is inspection-level and source-fit analysis. It should not be cited as benchmark
evidence for or against any world-model method.
