# Receipt identity audit

Audit updated after the sixth PR #10366 review at `1805b43a6e163340387b47e64e9fac11a671f23f`. Source anchors below refer to this revision; update the audit when these boundaries change. The native slot refactor is supplied by prerequisite [#10375](https://github.com/ll7/robot_sf_ll7/pull/10375); #10366 is stacked on that branch and does not change the sweep path relative to its base. The prerequisite retains its independent full-receipt gate.

The authoritative distinct slot is the runtime sweep's `EpisodeSlot`. Its current fields, obtained from `EpisodeSlot._fields`, are `(arm, kinematics, scenario, seed)`. `reconcile_execution` uses the shared `episode_slot(actual_row)` and `expected_episode_slots(config, scenarios)` functions for observed and expected identities. Receipt tooling imports those functions and the native type; it has no separate slot-axis inventory or Cartesian tuple definition.

The actual sealed row is passed intact with loader-style observed source context only when `_source_file` is absent. `_sweep_arm`, `_sweep_kinematics`, existing `_source_file`, `scenario_params.robot_config.type`, `algo` and future extractor inputs remain available. Directory, every present raw declaration, summary and producer links must agree. No source bytes are stripped, rewritten or resealed by validation.

`joined_sweep_episodes` keys raw and summary rows by the full native slot and checks uniqueness and exact coverage before exposing rows. `reviewed_slots` loads the same pinned release campaign templates as native acquisition, retaining their complete configuration, substitutes independently reviewed arm/map declarations and the required development slice, and calls `expected_episode_slots`. Source kinematics declarations never choose the expected matrix. Collection, full comparison, committed admission and both regular/LFS queue transports require observed native slot sets to equal this reviewed set: zero missing and zero unexpected slots.

New compact rows record `native_slot = slot._asdict()`, so every runtime field survives transport automatically. Existing v1 presentation labels can still be decoded by the native extractor, but they are never retained as reduced identity keys. Legacy direct-call dictionaries are immediately rekeyed using actual row declarations before deduplication, counting or comparison; contradictory or duplicate declarations refuse. Compact native fields and presentation labels must agree, and admission checks compact coverage against the reviewed native set as well as the original sealed sources.

This inventory includes direct identity extraction/key/count sites and the source/admission wrappers that consume them. Job IDs, source commit IDs, head/baseline roles, metric leaf names and algorithm names are not logical pair axes. `receipt_outcomes.py` and `receipt_producer_capture.py` validate outcome/physical-file seals; neither derives or keys logical pairs. Execution algorithm/profile checks remain separate from native roster identity.

## scripts/ci/receipt_provenance.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/receipt_provenance.py:21](../../scripts/ci/receipt_provenance.py#L21) `_native_id_values` | Imported native projections for standalone declared labels, including reserved roster delimiters; not raw-row extraction. |
| [scripts/ci/receipt_provenance.py:50](../../scripts/ci/receipt_provenance.py#L50) `canonical_id` | Require unchanged, nonempty native text IDs. |
| [scripts/ci/receipt_provenance.py:61](../../scripts/ci/receipt_provenance.py#L61) `canonical_seed` | Native seed coercer fixed point plus exact integer type. |
| [scripts/ci/receipt_provenance.py:74](../../scripts/ci/receipt_provenance.py#L74) `validate_seed_ids` | Canonical seeds and no normalized slice collisions. |
| [scripts/ci/receipt_provenance.py:81](../../scripts/ci/receipt_provenance.py#L81) `canonical_scenario_seed` | Validate declared scenario/seed companion identities. |
| [scripts/ci/receipt_provenance.py:96](../../scripts/ci/receipt_provenance.py#L96) `canonical_pair_key` | Decode legacy presentation labels with the native extractor into a complete slot. |
| [scripts/ci/receipt_provenance.py:138](../../scripts/ci/receipt_provenance.py#L138) `validate_pair_keys` | Validate caller-supplied comparison/transport key axes. |
| [scripts/ci/receipt_provenance.py:200](../../scripts/ci/receipt_provenance.py#L200) `validate_scope_ids` | Validate scope arm/map labels and every native projection collision. |
| [scripts/ci/receipt_provenance.py:220](../../scripts/ci/receipt_provenance.py#L220) `directory_arm` | Independent observed-directory roster projection, using native _arm_key. |
| [scripts/ci/receipt_provenance.py:227](../../scripts/ci/receipt_provenance.py#L227) `_row_source_matches_observed` | Require the retained raw source location to identify the observed file. |
| [scripts/ci/receipt_provenance.py:241](../../scripts/ci/receipt_provenance.py#L241) `native_row_key` | Entire actual row to native episode_slot and native selector agreement; preserve precedence inputs; require directory/source and all present scenario declarations to agree. |
| [scripts/ci/receipt_provenance.py:320](../../scripts/ci/receipt_provenance.py#L320) `_sweep_summaries` | Validate independent summary IDs, contained source paths and nested robot type against the extracted slot; summaries do not determine accepted native keys. |
| [scripts/ci/receipt_provenance.py:362](../../scripts/ci/receipt_provenance.py#L362) `joined_sweep_episodes` | Native raw-key uniqueness, unique agreeing summaries, exact raw/summary/path coverage; return native keys only after all checks. |
| [scripts/ci/receipt_provenance.py:515](../../scripts/ci/receipt_provenance.py#L515) `_producer_identity` | Actual native key must agree with original producer scenario/seed links before execution proof is minted. |
| [scripts/ci/receipt_provenance.py:591](../../scripts/ci/receipt_provenance.py#L591) `verify_producer_seals` | Directory selects strict proof decoding only; logical row identity is checked by _producer_identity. |
| [scripts/ci/receipt_provenance.py:671](../../scripts/ci/receipt_provenance.py#L671) `write_binding` | Fresh allocation binding requires verified producer/native identity first. |
| [scripts/ci/receipt_provenance.py:694](../../scripts/ci/receipt_provenance.py#L694) `verify_binding` | Verify frozen bindings and producer/native identity before exposing sealed rows. |
| [scripts/ci/receipt_provenance.py:734](../../scripts/ci/receipt_provenance.py#L734) `verify_optional_binding` | Collection custody entry to the same producer verifier. |
| [scripts/ci/receipt_provenance.py:743](../../scripts/ci/receipt_provenance.py#L743) `verified_raw_rows` | Committed/collection raw-source frontier: shared native join on original sealed objects; preserve proof attributes and compare outcomes/status. |
| [scripts/ci/receipt_provenance.py:86](../../scripts/ci/receipt_provenance.py#L86) `canonical_slot` | Enumerate the runtime type fields to validate every axis before key use. |
| [scripts/ci/receipt_provenance.py:111](../../scripts/ci/receipt_provenance.py#L111) `compact_slot` | Decode the whole native_slot mapping; require presentation labels and native extraction to agree. |
| [scripts/ci/receipt_provenance.py:149](../../scripts/ci/receipt_provenance.py#L149) `native_keyed_rows` | Immediately replace legacy direct-call declaration keys with actual-row native slots; no reduced key survives. |
| [scripts/ci/receipt_provenance.py:178](../../scripts/ci/receipt_provenance.py#L178) `reviewed_slots` | Use native campaign loader and expected_episode_slots for the independently reviewed expected set, including all configuration axes. |
| [scripts/ci/receipt_provenance.py:288](../../scripts/ci/receipt_provenance.py#L288) `_validate_slot_declarations` | Compare directory, every runtime row encoding field and explicit robot configuration with the extracted slot. |
| [scripts/ci/receipt_provenance.py:303](../../scripts/ci/receipt_provenance.py#L303) `_validate_robot_kinematics` | Shared raw/summary check for scenario_params.robot_config.type against the verified native slot. |

## scripts/ci/receipt_arm_execution.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/receipt_arm_execution.py:175](../../scripts/ci/receipt_arm_execution.py#L175) `_from_sealed_producer` | Mint frozen arm/scenario/source proof using native_row_key on the actual sealed row; no directory split. |
| [scripts/ci/receipt_arm_execution.py:224](../../scripts/ci/receipt_arm_execution.py#L224) `expected_algorithm` | Use previously verified arm/scenario proof to bind frozen scenario-specific algorithm overrides. |
| [scripts/ci/receipt_arm_execution.py:59](../../scripts/ci/receipt_arm_execution.py#L59) `carry_execution_proof` | Carry only verifier-minted arm/scenario/source attributes across in-memory projections. |
| [scripts/ci/receipt_arm_execution.py:67](../../scripts/ci/receipt_arm_execution.py#L67) `_profile_algorithm` | Read the frozen, source-bound scenario override; does not independently assign episode IDs. |

## scripts/ci/make_behaviour_receipt.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/make_behaviour_receipt.py:896](../../scripts/ci/make_behaviour_receipt.py#L896) `_check_seed_range` | Validate native seed identities and unique declared dev slice before acquisition/metadata use. |
| [scripts/ci/make_behaviour_receipt.py:924](../../scripts/ci/make_behaviour_receipt.py#L924) `_sweep_command` | Pass declared arm/scenario selectors and checked seeds to the native acquisition CLI; no equality or episode-key admission. |
| [scripts/ci/make_behaviour_receipt.py:972](../../scripts/ci/make_behaviour_receipt.py#L972) `_submit_sbatch_sweep` | Forward checked acquisition declarations; admission later uses original rows. |
| [scripts/ci/make_behaviour_receipt.py:1081](../../scripts/ci/make_behaviour_receipt.py#L1081) `_existing_sweep` | Existing-source metadata gate before aggregate artifact handling. |
| [scripts/ci/make_behaviour_receipt.py:1108](../../scripts/ci/make_behaviour_receipt.py#L1108) `_verify_sweep_metadata` | Counts and seed sets come only from the complete shared native join, not independent summary keys. |
| [scripts/ci/make_behaviour_receipt.py:1173](../../scripts/ci/make_behaviour_receipt.py#L1173) `_load_sweep_rows` | Shared native join before row projection/outcome checks; raw file decoded once per invocation. |
| [scripts/ci/make_behaviour_receipt.py:1198](../../scripts/ci/make_behaviour_receipt.py#L1198) `_load_run_rows` | Verified native rows across shards; reject cross-shard duplicate keys; reread original bytes without losing execution proof. |
| [scripts/ci/make_behaviour_receipt.py:1904](../../scripts/ci/make_behaviour_receipt.py#L1904) `build_rows_and_classifications` | Canonical scope/key inputs; emit declared full-grid compact rows/classifications from native source dictionaries. |
| [scripts/ci/make_behaviour_receipt.py:2005](../../scripts/ci/make_behaviour_receipt.py#L2005) `derive_valid_comparison` | Require exact expected scope/grid coverage for both native source dictionaries before algorithm dispatch and full-route counts. |
| [scripts/ci/make_behaviour_receipt.py:2090](../../scripts/ci/make_behaviour_receipt.py#L2090) `write_real_row_audit` | Serialize already derived compact rows and execution audit, without choosing new IDs. |
| [scripts/ci/make_behaviour_receipt.py:2178](../../scripts/ci/make_behaviour_receipt.py#L2178) `write_receipt_and_header` | Serialize derived full-route scope/row/classification counts; cannot bypass the source/admission frontier. |
| [scripts/ci/make_behaviour_receipt.py:2305](../../scripts/ci/make_behaviour_receipt.py#L2305) `_load_scope` | Canonical scope labels before Cartesian grids. |
| [scripts/ci/make_behaviour_receipt.py:2380](../../scripts/ci/make_behaviour_receipt.py#L2380) `_prepare_sweeps` | Checked seed slice and native selector inputs; collection reads return through the shared join. |
| [scripts/ci/make_behaviour_receipt.py:2440](../../scripts/ci/make_behaviour_receipt.py#L2440) `_sweep_episode_paths` | Transport paths indexed only by shared native keys after agreement and coverage. |
| [scripts/ci/make_behaviour_receipt.py:2448](../../scripts/ci/make_behaviour_receipt.py#L2448) `_sweep_execution_statuses` | Summary statuses indexed only by shared native keys after agreement and coverage. |
| [scripts/ci/make_behaviour_receipt.py:2456](../../scripts/ci/make_behaviour_receipt.py#L2456) `_run_episode_transport` | Shard path/status aggregation after native joining; reject cross-shard duplicates. |
| [scripts/ci/make_behaviour_receipt.py:2468](../../scripts/ci/make_behaviour_receipt.py#L2468) `_verify_equivalence_tooling` | Compare archived, checked seed declarations to the expected dev slice. |
| [scripts/ci/make_behaviour_receipt.py:2517](../../scripts/ci/make_behaviour_receipt.py#L2517) `_validate_equivalence_sources` | Private producer executes the same verified raw-source and committed projection rules. |
| [scripts/ci/make_behaviour_receipt.py:2535](../../scripts/ci/make_behaviour_receipt.py#L2535) `_make_equivalence_receipt` | Canonical scope/seeds, native joined sources, exact 14 by 51 by 3 grid and expected 2142 episodes before equality/serialization. |
| [scripts/ci/make_behaviour_receipt.py:2693](../../scripts/ci/make_behaviour_receipt.py#L2693) `main` | Full/equivalence producer dispatch; no alternate source identity extraction. |

## scripts/ci/default_path_equivalence.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/default_path_equivalence.py:748](../../scripts/ci/default_path_equivalence.py#L748) `episode_bytes` | Compare preserved raw evidence including identity bytes; does not assign pair identities. |
| [scripts/ci/default_path_equivalence.py:820](../../scripts/ci/default_path_equivalence.py#L820) `validate_scope_entries` | Canonical scope before counts or Cartesian products. |
| [scripts/ci/default_path_equivalence.py:834](../../scripts/ci/default_path_equivalence.py#L834) `comparison_rows` | Validate scope and incoming native dictionary keys, require exact paired grid before execution/hash comparisons. |
| [scripts/ci/default_path_equivalence.py:955](../../scripts/ci/default_path_equivalence.py#L955) `compare_rows` | Comparison wrapper around the same validated native key path. |
| [scripts/ci/default_path_equivalence.py:966](../../scripts/ci/default_path_equivalence.py#L966) `validate_archive_paths` | Bind row transports to verified allocation archives; no independent roster extraction. |
| [scripts/ci/default_path_equivalence.py:983](../../scripts/ci/default_path_equivalence.py#L983) `validate_receipt` | Canonical committed keys, unique full scope/seed grid, archive and execution/review bindings; actual raw identity is re-derived by source admission. |

## scripts/ci/behaviour_receipt.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/behaviour_receipt.py:475](../../scripts/ci/behaviour_receipt.py#L475) `_key` | Canonical compact arm/map/seed declarations; compact keys are independently matched against native source keys. |
| [scripts/ci/behaviour_receipt.py:503](../../scripts/ci/behaviour_receipt.py#L503) `validate_receipt` | Full-route scope/key/classification and expected-grid counts, using source-derived execution proof. |
| [scripts/ci/behaviour_receipt.py:719](../../scripts/ci/behaviour_receipt.py#L719) `read_committed_payload_bytes` | Regular-file and LFS are only transport alternatives; both decode and admit identical source semantics. |
| [scripts/ci/behaviour_receipt.py:795](../../scripts/ci/behaviour_receipt.py#L795) `decode_committed_payload` | Committed source-artifact revalidation before compact evidence is accepted. |
| [scripts/ci/behaviour_receipt.py:842](../../scripts/ci/behaviour_receipt.py#L842) `load_committed_payload` | Regular/LFS committed admission wrapper. |
| [scripts/ci/behaviour_receipt.py:847](../../scripts/ci/behaviour_receipt.py#L847) `load_receipt` | Committed full-route wrapper plus restored native source execution proofs. |
| [scripts/ci/behaviour_receipt.py:872](../../scripts/ci/behaviour_receipt.py#L872) `admit_equivalence_payload` | Common semantic admission for PR and both queue transport paths. |
| [scripts/ci/behaviour_receipt.py:991](../../scripts/ci/behaviour_receipt.py#L991) `_equivalence_source_projection` | Shared metadata, row, path/status joins; compare canonical native source keys and original raw observations. |
| [scripts/ci/behaviour_receipt.py:1084](../../scripts/ci/behaviour_receipt.py#L1084) `_verify_equivalence_projection` | Index committed claims by canonical keys and require exact source-derived projection/execution agreement. |
| [scripts/ci/behaviour_receipt.py:1116](../../scripts/ci/behaviour_receipt.py#L1116) `verify_source_artifacts` | Call verified_raw_rows for each sealed archive; reject cross-shard native collisions, require compact/source coverage and exact rederivation. |
| [scripts/ci/behaviour_receipt.py:1245](../../scripts/ci/behaviour_receipt.py#L1245) `_restore_execution_proofs` | Match canonical compact keys to already verified native/source-bound proof objects. |
| [scripts/ci/behaviour_receipt.py:1265](../../scripts/ci/behaviour_receipt.py#L1265) `check_receipt` | PR admission enters shared committed decode/source and semantic validators. |

## scripts/dev/merge_queue_gate.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/dev/merge_queue_gate.py:694](../../scripts/dev/merge_queue_gate.py#L694) `_fetch_equivalence_freshness` | Read current regular/LFS committed payload, call admit_equivalence_payload; refuse source identity errors before recording admission. |
| [scripts/dev/merge_queue_gate.py:677](../../scripts/dev/merge_queue_gate.py#L677) `_record_equivalence_admission` | Record acceptance only after the common source validator and final current-main read. |
| [scripts/dev/merge_queue_gate.py:798](../../scripts/dev/merge_queue_gate.py#L798) `evaluate_merge_gate` | Require fully admitted, exact-head/current-main proof; does not infer logical pair identities from queue metadata. |
| [scripts/dev/merge_queue_gate.py:3198](../../scripts/dev/merge_queue_gate.py#L3198) `main` | Both queue transport invocations use the same committed/source frontier; no PR source checkout or independent roster counting. |

## Native owners (imported, not copied)

| Site | Runtime authority |
| --- | --- |
| [scripts/validation/run_empty_world_sweep.py:310](../../scripts/validation/run_empty_world_sweep.py#L310) | `load_rows`; runtime source unchanged |
| [scripts/validation/run_empty_world_sweep.py:335](../../scripts/validation/run_empty_world_sweep.py#L335) | `_arm_key`; runtime source unchanged |
| [scripts/validation/run_empty_world_sweep.py:457](../../scripts/validation/run_empty_world_sweep.py#L457) | `_kinematics`; runtime source unchanged |
| [scripts/validation/run_empty_world_sweep.py:380](../../scripts/validation/run_empty_world_sweep.py#L380) | `flatten`; runtime source unchanged |
| [scripts/validation/run_empty_world_sweep.py:570](../../scripts/validation/run_empty_world_sweep.py#L570) | `reconcile_execution`; shared native slot refactor is supplied by the separate prerequisite PR; simulated execution unchanged |
| [robot_sf/benchmark/camera_ready/_config.py:386](../../robot_sf/benchmark/camera_ready/_config.py#L386) | `_sanitize_name`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_config.py:732](../../robot_sf/benchmark/camera_ready/_config.py#L732) | `_scenario_name_key`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_run_state.py:81](../../robot_sf/benchmark/camera_ready/_run_state.py#L81) | `_coerce_identifier`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_run_state.py:95](../../robot_sf/benchmark/camera_ready/_run_state.py#L95) | `_episode_identity`; runtime source unchanged |
| [robot_sf/training/scenario_loader.py:1804](../../robot_sf/training/scenario_loader.py#L1804) | `_scenario_runtime_identity`; runtime source unchanged |
| [robot_sf/training/scenario_loader.py:1573](../../robot_sf/training/scenario_loader.py#L1573) | `select_scenario`; runtime source unchanged |

The native reconciler is also an independent test oracle: the original 2142-row fixture is complete; setting every actual `_sweep_arm` to `ppo` leaves 153 unique pairs with 1989 missing and duplicate slots. The receipt producer, committed admission and regular/LFS queue must all refuse the coherently resealed evidence.

Validation includes individual directory/row/summary forgeries through every reader, a changed native-function priority input that proves the full actual dictionary is passed, canonical-ID and strict trace/action regressions, and the original sealed #10342 pilot. These checks launch no simulation or receipt batch and claim historical compatibility only.

## Native slot definition and regression evidence

- [scripts/validation/run_empty_world_sweep.py:505](../../scripts/validation/run_empty_world_sweep.py#L505) `directory_episode_identity` owns independent native directory declarations, used both by runtime reconciliation and receipt readers.
- [scripts/validation/run_empty_world_sweep.py:473](../../scripts/validation/run_empty_world_sweep.py#L473) `EpisodeSlot` owns the slot schema, extraction, encoding or runtime reconciliation.
- [scripts/validation/run_empty_world_sweep.py:482](../../scripts/validation/run_empty_world_sweep.py#L482) `episode_slot` owns the slot schema, extraction, encoding or runtime reconciliation.
- [scripts/validation/run_empty_world_sweep.py:491](../../scripts/validation/run_empty_world_sweep.py#L491) `episode_slot_row` owns the slot schema, extraction, encoding or runtime reconciliation.
- [scripts/validation/run_empty_world_sweep.py:515](../../scripts/validation/run_empty_world_sweep.py#L515) `expected_episode_slots` owns the slot schema, extraction, encoding or runtime reconciliation.
- [scripts/validation/run_empty_world_sweep.py:570](../../scripts/validation/run_empty_world_sweep.py#L570) `reconcile_execution` owns the slot schema, extraction, encoding or runtime reconciliation.

`test_receipt_slot_identity.py` contains the full-matrix and PPO-only holonomic attacks at collection, committed admission and both queue transports. All eight controls failed at the reviewed head because equality/admission was incorrectly accepted. Its generic coherent-forgery test enumerates `EpisodeSlot._fields` and uses the runtime row encoder, so an added runtime field automatically receives a probe. The generic test was also run with the factored native definition and the reviewed receipt binding: the kinematics case still admitted the attack. Canonical pilot axes and the actual unchanged sealed #10342 corpus remain positive controls. The additional row/directory/summary kinematics and future native-priority controls failed before receipt changes (16 failures), then passed after the shared native binding (16 passes). Repeating the complete, coherently relabelled 28-case file against the factored runtime schema and reviewed receipt implementation produced 25 failures for admitted attacks and 3 refusals for the already guarded axes; the fixed implementation passes all 28. All attack probes use seeds 1001–1006; no receipt batch was run.

The hosted main baseline at `c26838b140d4e4073f7414c33496d781f4eea4a0` passed the macOS/Python 3.11 native HTTP audit-workbench test in [job 115284937815](https://github.com/ll7/robot_sf_ll7/actions/runs/38394055489/job/115284937815) (3168 passed, 2 skipped). The sixth review confirms that PR head `1805b43a6e163340387b47e64e9fac11a671f23f` also passed that same test in [job 115534683187](https://github.com/ll7/robot_sf_ll7/actions/runs/38466456062/job/115534683187), with the same totals. The audit implementation, test, dependency lock and CI command remain unchanged. The earlier intermittent failure was not reproduced on either head; its cause remains unidentified. No audit-workbench code change is included.
