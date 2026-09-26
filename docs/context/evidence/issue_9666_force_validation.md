# Robot-attributable force diagnostic validation — issue #9666

> AI-GENERATED · NEEDS-REVIEW. Diagnostic evidence on development seeds, not release evaluation.

## Result and boundary

Job 15754 produced 384 unique episodes: four planners × 48 scenarios × seeds 101 and 102, horizon 600, timestep 0.1 s. All 384 algorithm records are `ok`, with no degraded integrity flags, contradictions, or nonzero runtime fallback counters. Goal, ORCA, SocialForce and the configured hybrid use their intended repository implementations; ORCA and SocialForce are exposed through repository adapters. Outcomes were 210 success, 117 collision and 57 failure; these are simulation outcomes, not execution failures.

The force is a simulated acceleration before speed capping, not measured human discomfort. The peak closely tracks sign-flipped minimum distance (Spearman ρ = 0.976526, n = 376), whereas integrated force additionally depends on interaction duration and pedestrian count. The actual kernel reference is approximately 3.7030154332523164 m/s², not the issue’s approximate 2.6: the SocialForce lateral component contributes at zero bearing.

Real force impulse versus near misses: ρ = 0.8169379191861208 (n = 384). Pedestrian-equivalent impulse versus near misses: ρ = 0.7374736892291968 (n = 380). The #9667 switch uses `N = clip((near_misses / executed steps) / 0.25, 0, 1)` on a separate 1,344-row calibration split. Here, diagnostic impulse versus the **raw near-miss fraction** (before division by 0.25 and clipping) gives ρ = 0.7553665091251708 (n = 384), and pedestrian-equivalent impulse versus the raw fraction gives ρ = 0.7239784500448517 (n = 380). This 384-row diagnostic does not select the calibration variant; none of these coefficients is a causal finding. No uncertainty interval is estimated; scenarios and seeds are correlated, so the descriptive coefficients should not be generalized beyond this slice.

## Reproducibility and custody

- Producer source: `71464357ca89e6578ee593a15167bd1eda612d99`.
- Analysis revision: `97a50d1612d04375572433186998a09e21856155`; simulator, episode runner and diagnostic config remain byte-identical to producer source. A later serialization-only refactor replaces a loop with an equivalent dictionary transformation to satisfy the complexity gate. Both versions produce byte-identical compact JSON on all 384 stored metric payloads and a synthetic nested-null case (concatenated digest `6984422bf4b8df631ee41da75c16fbccc3f2e264be8645e78bb6d02342be33ae`). This is post-processing parity, not final-source re-execution. Fresh main was merged before final readiness.
- Config: `configs/benchmarks/issue_9666_force_validation.yaml`, SHA-256 `a5e629e326da5a26ddf06af20d79ea30ec1e10ff7f9d3d75af53515d58b71c93`; effective hash `f96ecf5d754082e0`.
- Scenario matrix SHA-256: `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`.
- Scheduler: COMPLETED, exit 0:0, derived exit 0:0, elapsed 316 s. Producer campaign/sync exits: 0/0. Canonical retrieval verified all 60 producer entries; 384 row identities were checked independently.
- Producer manifest SHA-256: `77afd03ab71414d456b5e23f72de83606f0a3db22fb49890da38c8e60b5e2df4`.
- Preserved artifact: `wandb://ll7/robot_sf/campaign-issue9666_force_probe_71464357c_custody_r2_20260924:v0`; [upload run](https://wandb.ai/ll7/robot_sf/runs/scrf2191).
- Preservation manifest digest: `sha256:1f86505249d38d53688460629b8bbf1ef7af9abf334db4550870a525dfcb626b`. Cold readback re-fetched the remote entry table, compared paths/sizes/MD5, and downloaded/re-hashed the embedded manifest. This is not a full remote payload download.
- [Machine custody inventory](issue_9666_force_custody.json) and [complete correlation matrix and trace summaries](issue_9666_force_validation.json) contain exact values and row-file checksums.

Startup custody limitation: the legacy `packet` field names the config file, while `packet_sha256` is a digest of the submission command inputs. They are not a file/hash pair (private-ops issue #371). YAML identity is established separately by the canonical guard receipt at private commit `c262207ebfd1f30f71373cc25426cd419c23e718`, matching canonical/submitted `artifact_manifest` to YAML SHA-256 `cd748b9c0b966e0686d37c043106f2b7a86f5f274583807985ab4e6806ec1c57`. Admission binds source/config/script/queue/campaign and submission identity; startup binds that same submission identity and job. Original receipts were not rewritten. This chain supports diagnostic interpretation subject to independent review, not a claim that startup directly hashed the YAML packet.

Earlier job 15752 completed 384 rows but lacked startup and producer checksum receipts. Its retrieval failed; it remains diagnostic and unaccepted, preserved separately with original bytes unchanged. The fresh job uses the reviewed launcher correction and distinct campaign/result identity. No 0.0.7 artifact, tag, or publication was changed.

Reproduce the analysis after retrieving the preserved `runs/*/episodes.jsonl` files:

```bash
python scripts/analysis/issue_9666_robot_force_validation.py \
  --episodes "$ROOT/runs/goal__differential_drive/episodes.jsonl" \
  --episodes "$ROOT/runs/hybrid_rule_v3_fast_progress_static_escape__differential_drive/episodes.jsonl" \
  --episodes "$ROOT/runs/orca__differential_drive/episodes.jsonl" \
  --episodes "$ROOT/runs/social_force__differential_drive/episodes.jsonl" \
  --artifact-location wandb://ll7/robot_sf/campaign-issue9666_force_probe_71464357c_custody_r2_20260924:v0 \
  --output force_validation.json
```

## Missing and post-hoc quantities

There are eight zero-pedestrian rows and 91 zero-exposure rows (83 contain pedestrians). Exactly 91 per-exposed impulse values and 91 active-mean values are documented JSON nulls. Minimum distance is nonfinite on the eight empty scenes; those rows are excluded pairwise only from distance correlations.

Four one-step collision episodes—every planner on `classic_cross_trap_high`, seed 102—omit the experimental pedestrian-equivalent metrics. Each has real impulse 0.4775616836435781 m/s and two exposed pedestrians. Effective pair distances span 0.6694686227196637–33.34749556402961 m against a 20 m activation range; velocity is unavailable and zero cannot be imputed. A downstream index using this experimental term must declare handling of such rows.

The campaign did not enable the existing human-interaction proxy. Its discomfort comparator is explicitly post-hoc: the analyzer calls `experimental_human_interaction_proxy_metrics` on recorded **post-integration** trajectories, proxemic radius 1.2 m, yield speed 0.15 m/s. The geometric footprint is recovered from recorded surface clearance and checked across every pedestrian sample within 1e-9 m. It uses a 0.4 m pedestrian footprint, distinct from the force kernel’s 0.35 m radius. Original episode files are unchanged. An initial analyzer lookup yielded no discomfort observations; the final report fixes this with canonical recomputation, not substituted comfort metrics.

## Overall Spearman correlations

Distance is sign-flipped so larger values mean more exposure. Pairwise finite counts accompany every coefficient.

| Force reduction | Comparator | n | ρ |
|---|---|---:|---:|
| `robot_force_impulse_total` | `min_distance` | 376 | 0.610913 |
| `robot_force_impulse_total` | `near_misses` | 384 | 0.816938 |
| `robot_force_impulse_total` | `human_discomfort_exposure_m_s` | 384 | 0.966260 |
| `robot_force_peak` | `min_distance` | 376 | 0.976526 |
| `robot_force_peak` | `near_misses` | 384 | 0.733816 |
| `robot_force_peak` | `human_discomfort_exposure_m_s` | 384 | 0.714903 |
| `robot_force_time_above_ref_s` | `min_distance` | 376 | 0.135660 |
| `robot_force_time_above_ref_s` | `near_misses` | 384 | -0.100483 |
| `robot_force_time_above_ref_s` | `human_discomfort_exposure_m_s` | 384 | -0.062817 |
| `robot_force_pp_equiv_impulse_total` | `min_distance` | 372 | 0.776806 |
| `robot_force_pp_equiv_impulse_total` | `near_misses` | 380 | 0.737474 |
| `robot_force_pp_equiv_impulse_total` | `human_discomfort_exposure_m_s` | 380 | 0.820294 |
| `robot_force_pp_equiv_peak` | `min_distance` | 372 | 0.867289 |
| `robot_force_pp_equiv_peak` | `near_misses` | 380 | 0.621108 |
| `robot_force_pp_equiv_peak` | `human_discomfort_exposure_m_s` | 380 | 0.667779 |
| `robot_force_pp_equiv_time_above_ref_s` | `min_distance` | 372 | 0.124611 |
| `robot_force_pp_equiv_time_above_ref_s` | `near_misses` | 380 | -0.006627 |
| `robot_force_pp_equiv_time_above_ref_s` | `human_discomfort_exposure_m_s` | 380 | -0.018822 |
| `robot_force_impulse_total` | `near_misses_per_step` | 384 | 0.755367 |
| `robot_force_peak` | `near_misses_per_step` | 384 | 0.770230 |
| `robot_force_time_above_ref_s` | `near_misses_per_step` | 384 | -0.100466 |
| `robot_force_pp_equiv_impulse_total` | `near_misses_per_step` | 380 | 0.723978 |
| `robot_force_pp_equiv_peak` | `near_misses_per_step` | 380 | 0.646769 |
| `robot_force_pp_equiv_time_above_ref_s` | `near_misses_per_step` | 380 | 0.021615 |

## Planner and scenario-family redundancy

The full JSON contains all six force reductions × four comparators (including the raw near-miss fraction) for every cohort. This table shows diagnostic correlations with raw near-miss counts. Family is the recorded `scenario_params.metadata.archetype` (35 distinct families), not an inferred name grouping. The hybrid’s serialized algorithm name is `hybrid_rule_local_planner`; its configured arm is `hybrid_rule_v3_fast_progress_static_escape`.

| Cohort | Real impulse ρ (n) | Pedestrian-equivalent impulse ρ (n) |
|---|---:|---:|
| `planner:goal` | 0.676487 (96) | 0.760892 (95) |
| `family:bottleneck` | 0.827774 (32) | 0.521574 (32) |
| `family:station_platform` | 0.857143 (8) | 0.857143 (8) |
| `family:cross_trap` | 0.890004 (24) | 0.908852 (20) |
| `family:doorway` | 0.525690 (24) | 0.619659 (24) |
| `family:group_crossing` | 0.565330 (24) | 0.495523 (24) |
| `family:head_on_corridor` | 0.910094 (16) | -0.089225 (16) |
| `family:merging` | 0.958875 (16) | 0.865829 (16) |
| `family:overtaking` | 0.968226 (16) | 0.772189 (16) |
| `family:t_intersection` | 0.874671 (16) | 0.761191 (16) |
| `family:crossing` | 0.957217 (8) | 0.707528 (8) |
| `family:frontal_approach` | 0.981981 (8) | -0.230341 (8) |
| `family:pedestrian_obstruction` | 0.747042 (8) | 0.000000 (8) |
| `family:pedestrian_overtaking` | 0.577350 (8) | 0.577350 (8) |
| `family:robot_overtaking` | 0.891631 (8) | 0.144589 (8) |
| `family:down_path` | 0.763763 (8) | 0.763763 (8) |
| `family:intersection_no_gesture` | 1.000000 (8) | 0.577350 (8) |
| `family:blind_corner` | 0.993958 (8) | 0.412129 (8) |
| `family:narrow_hallway` | 0.993958 (8) | 0.795238 (8) |
| `family:narrow_doorway` | undefined (8) | undefined (8) |
| `family:entering_room` | undefined (8) | undefined (8) |
| `family:exiting_room` | undefined (8) | undefined (8) |
| `family:entering_elevator` | undefined (8) | undefined (8) |
| `family:exiting_elevator` | 0.596561 (8) | 0.024708 (8) |
| `family:intersection_wait` | 1.000000 (8) | 0.577350 (8) |
| `family:intersection_proceed` | 1.000000 (8) | 0.577350 (8) |
| `family:following_human` | -0.050735 (8) | 0.659550 (8) |
| `family:leading_human` | -0.125988 (8) | 0.755929 (8) |
| `family:accompanying_peer` | -0.503953 (8) | 0.755929 (8) |
| `family:join_group` | 0.969782 (8) | -0.454201 (8) |
| `family:leave_group` | 0.902708 (8) | 0.341565 (8) |
| `family:crowd_navigation` | 0.143715 (8) | 0.215573 (8) |
| `family:parallel_traffic` | 0.219578 (8) | 0.463553 (8) |
| `family:perpendicular_traffic` | 0.951503 (8) | 0.951503 (8) |
| `family:circular_crossing` | undefined (8) | undefined (8) |
| `family:robot_crowding` | 0.994030 (8) | 0.670671 (8) |
| `planner:hybrid_rule_local_planner` | 0.919190 (96) | 0.864255 (95) |
| `planner:orca` | 0.923619 (96) | 0.745003 (95) |
| `planner:social_force` | 0.660566 (96) | 0.708927 (95) |

## Twenty largest absolute rank disagreements

Ranks are average tie ranks over 376 finite-distance episodes. The twenty largest absolute gaps all have relatively close distance but lower accumulated force; no opposite-direction episode is hidden from this absolute-gap selection. Every listed episode ends in collision. The trace evidence identifies short **force-active exposure**, even when total elapsed time is longer: 0.1–3.4 s of active force and one or two simultaneously exposed pedestrians. Four terminate after one step. This explains how a low integrated burden can coexist with a close minimum; it is a duration/multiplicity observation, not causal proof or a claim of safe behavior.

| Planner | Scenario / seed | Force rank / distance rank | Impulse (m/s) | Minimum distance (m) | Active time (s) / pedestrian-seconds / maximum concurrent |
|---|---|---:|---:|---:|---|
| `social_force` | `francis2023_accompanying_peer` / 101 | 114 / 373 | 1.114969 | 1.285898 | 1.0 / 1.0 / 1 |
| `social_force` | `francis2023_accompanying_peer` / 102 | 119 / 372 | 1.155177 | 1.290631 | 1.0 / 1.0 / 1 |
| `social_force` | `francis2023_leading_human` / 101 | 113 / 365 | 1.100450 | 1.319113 | 1.0 / 1.0 / 1 |
| `social_force` | `francis2023_leading_human` / 102 | 118 / 368 | 1.154599 | 1.307261 | 1.0 / 1.0 / 1 |
| `goal` | `classic_cross_trap_high` / 102 | 87.5 / 331.5 | 0.477562 | 1.375369 | 0.1 / 0.2 / 2 |
| `social_force` | `classic_cross_trap_high` / 102 | 87.5 / 331.5 | 0.477562 | 1.375369 | 0.1 / 0.2 / 2 |
| `orca` | `classic_cross_trap_high` / 102 | 87.5 / 330 | 0.477562 | 1.375393 | 0.1 / 0.2 / 2 |
| `hybrid_rule_local_planner` | `classic_cross_trap_high` / 102 | 87.5 / 329 | 0.477562 | 1.376840 | 0.1 / 0.2 / 2 |
| `social_force` | `francis2023_frontal_approach` / 101 | 108 / 335 | 0.988721 | 1.370125 | 1.0 / 1.0 / 1 |
| `goal` | `classic_bottleneck_high` / 102 | 138 / 363 | 1.749982 | 1.331616 | 1.3 / 1.7 / 2 |
| `social_force` | `francis2023_intersection_wait` / 101 | 153.5 / 374.5 | 2.175899 | 1.284743 | 2.9 / 2.9 / 1 |
| `social_force` | `francis2023_intersection_proceed` / 101 | 153.5 / 374.5 | 2.175899 | 1.284743 | 2.9 / 2.9 / 1 |
| `goal` | `classic_bottleneck_high` / 101 | 135 / 350 | 1.640555 | 1.358548 | 1.3 / 1.7 / 2 |
| `goal` | `francis2023_narrow_hallway` / 101 | 146 / 355 | 2.004659 | 1.349739 | 1.5 / 1.5 / 1 |
| `social_force` | `francis2023_robot_overtaking` / 101 | 168 / 376 | 2.552526 | 1.271043 | 3.4 / 3.4 / 1 |
| `goal` | `classic_bottleneck_medium` / 102 | 134 / 338 | 1.598551 | 1.368225 | 1.3 / 1.3 / 1 |
| `goal` | `classic_head_on_corridor_medium` / 101 | 148 / 352 | 2.026826 | 1.357257 | 1.5 / 1.5 / 1 |
| `goal` | `francis2023_frontal_approach` / 102 | 144 / 348 | 1.938485 | 1.361339 | 1.5 / 1.5 / 1 |
| `social_force` | `francis2023_following_human` / 102 | 140 / 344 | 1.837980 | 1.365512 | 1.3 / 1.3 / 1 |
| `social_force` | `francis2023_frontal_approach` / 102 | 125 / 327 | 1.425848 | 1.377782 | 1.0 / 1.0 / 1 |

## Implementation proof

A fixed-seed native five-step probe compared the actual baseline metrics module at `120c870d80daba4a06389f9df3c469a2f467da94` against source714: all 55 legacy metric JSON values were byte-identical (SHA-256 `8c5edbf1421fff9634ba2f76e016557ad8aa7c299352a8582d27445c5ae1f8d3`). A nonzero step recorded and recomputed `[1.25, 0]` m/s² with zero maximum error; inactive capture returned `[0, 0]`. Focused tests cover component subtraction, inverse-cubic law, reference kernel, null/empty cases, schema serialization and both compact/trace persistence modes. Final PR validation reports the complete gate and retry history.
