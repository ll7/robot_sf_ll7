# PPO retrained on the 0.0.8 release robot

<!-- AI-GENERATED: NEEDS-REVIEW -->

PPO retrained on the release robot; collision-prone learned reference.

Model-only prerelease for author decision D-062 (2026-10-01). This asset replaces the plain `ppo` checkpoint in the forthcoming 0.0.8 benchmark contract. `guarded_ppo` remains unchanged. This is not the 0.0.8 release and does not claim competitive baseline quality or sealed-seed performance.

Asset: `ppo_release_robot_b1002_last_20261001-model.zip`

SHA256: `764a7d88f5b608237641d973634899e05a67b25e65f8b1607cfca025459824bc`

Training source: `07bd2b8037053f8e979bd97db32d096fe58e5257`; leaf config: `configs/training/ppo/expert_ppo_release_contract_b_seed1002.yaml`; seed 1002; 15M steps; last checkpoint. Plant: 2.0 m/s, no reverse, +/-1 m/s^2 linear acceleration/deceleration, +/-1 rad/s^2 angular acceleration, dt 0.1 s. Actions: `velocity_delta`. Predictive foresight: disabled, matching variant B. Observation contract inherits the BR-06 v3 parent.

Selection used the pre-registered 48-scenario x dev-seeds 1001-1005 matrix, with the most successes and fewer collisions for candidates within five successes of the maximum. B1002 last achieved 151 successes / 89 collisions / 0 timeouts. Job 15835 completed; its A1002 best and last checkpoints were evaluated on the same matrix and runner with foresight enabled, yielding 132/105/3 and 132/106/2. The published PPO achieved 35/204/1 on this dev matrix.

Durable training evidence: W&B `ll7/robot_sf/ppotrain-b-seed1002-20260930-complete:v0`, verified COMMITTED; custody run https://wandb.ai/ll7/robot_sf/runs/m6m6glg9. Selected file in that artifact: `benchmarks/expert_policies/checkpoints/ppo_release_contract_br06_v3_delta_seed1002_20260930/ppo_release_contract_br06_v3_delta_seed1002_20260930_step15000000.zip`.

The existing seed-1001 trace slice found about 85% of retrained-policy collisions were with static geometry. Variant-B policies held 2.0 m/s on approximately 97-100% of steps (B1002 last: 99.9%). These are diagnostic dev results; the training simulator predates release scenario/physics changes, seed 1003 was used for in-training checkpoint selection, and learned-arm counts can vary across CPU platforms.

## Full selection table

All rows use 48 scenarios x dev seeds 1001-1005 (240 episodes each). Existing PPOEVAL results were reused; only A1002 best/last were newly evaluated.

| Checkpoint | Success | Collision | Timeout |
| --- | ---: | ---: | ---: |
| **B1002 last (selected)** | **151** | **89** | **0** |
| B1001 best | 148 | 92 | 0 |
| B1002 best | 146 | 94 | 0 |
| A1001 last | 142 | 90 | 8 |
| A1002 best | 132 | 105 | 3 |
| A1002 last | 132 | 106 | 2 |
| B1001 last | 128 | 112 | 0 |
| A1001 best | 125 | 103 | 12 |
| Published PPO (comparison) | 35 | 204 | 1 |

B1001 best and B1002 best are within five successes of the maximum, but have more collisions. Neither A1002 checkpoint enters that tie band.

Reference arms in PPOEVAL: ORCA 213/22/5, social force 162/2/76, guarded PPO 153/5/82, and goal 95/141/4 (success/collision/timeout). Both ORCA and social force have higher success and fewer collisions than the selected policy.

## Comparability and reproduction

The arm keeps `ppo`, as required by the active `alyassi_comparability_map_v2.yaml` mapping `ppo: ppo`. Its implementation and model identity change; 0.0.7 PPO rows remain historical observations of a different checkpoint and plant contract. The template, calibration configuration and v0_5 runtime-smoke companion resolve the same new profile. Historical campaign profiles and frozen configs retain their existing bindings. These preparation changes are not a release acceptance result.

The evaluation used runner source `e8132439b42a71e4347bff58dfc91d73eecc0c1d`, the 0.0.8 rehearsal matrix `classic_interactions_francis2023_release_0_0_8_v1.yaml`, authored scenario horizons, dt 0.1, and deterministic CPU inference. A arms used `predictive_proxy_selected_v2_full` foresight with its checksum verified and fallback excluded; B arms used no foresight.

Hydrate the selected model through `robot_sf.models.resolve_model_path("ppo_release_robot_b1002_last_20261001")`; the registry checks the public release-asset SHA256. The training recipe and development evaluation-seed manifest are checked in under `configs/training/ppo/`.
