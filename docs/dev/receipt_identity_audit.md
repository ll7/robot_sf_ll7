# Receipt identity audit

Audit for the correction after PR #10366 review at `157283f3f3f164800ad818bf0576db11d5c8cac6`. Source anchors below refer to this revision; update the audit when these boundaries change.

The receipt pair is `(native arm, native scenario/map, native seed)`. The actual sealed row is passed intact to the native extractor with loader-style observed source context only when `_source_file` is absent. `_sweep_arm`, existing `_source_file`, `algo`, and future priority inputs remain available. Directory, raw declarations, summary and producer links must agree. No source bytes are stripped, rewritten or resealed by validation.

One shared `joined_sweep_episodes` frontier validates native raw uniqueness, summary uniqueness and exact raw/summary/path coverage before exposing keyed rows. Declared full/equivalence scope coverage is then checked before outcome counts or equality. Metadata, row readers, path/status transport, committed admission and both queue transports all traverse it. Standalone scalar validation remains a declaration check, not evidence of native row identity.

This inventory includes direct identity extraction/key/count sites and the source/admission wrappers that consume them. Job IDs, source commit IDs, head/baseline roles, metric leaf names and algorithm names are not logical pair axes. `receipt_outcomes.py` and `receipt_producer_capture.py` validate outcome/physical-file seals; neither derives or keys logical pairs. Execution algorithm/profile checks remain separate from native roster identity.

## scripts/ci/receipt_provenance.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/receipt_provenance.py:18](../../scripts/ci/receipt_provenance.py#L18) `_native_id_values` | Imported native projections for standalone declared labels, including reserved roster delimiters; not raw-row extraction. |
| [scripts/ci/receipt_provenance.py:45](../../scripts/ci/receipt_provenance.py#L45) `canonical_id` | Require unchanged, nonempty native text IDs. |
| [scripts/ci/receipt_provenance.py:56](../../scripts/ci/receipt_provenance.py#L56) `canonical_seed` | Native seed coercer fixed point plus exact integer type. |
| [scripts/ci/receipt_provenance.py:69](../../scripts/ci/receipt_provenance.py#L69) `validate_seed_ids` | Canonical seeds and no normalized slice collisions. |
| [scripts/ci/receipt_provenance.py:76](../../scripts/ci/receipt_provenance.py#L76) `canonical_scenario_seed` | Validate declared scenario/seed companion identities. |
| [scripts/ci/receipt_provenance.py:81](../../scripts/ci/receipt_provenance.py#L81) `canonical_pair_key` | Validate declared arm/scenario/seed keys. |
| [scripts/ci/receipt_provenance.py:86](../../scripts/ci/receipt_provenance.py#L86) `validate_pair_keys` | Validate caller-supplied comparison/transport key axes. |
| [scripts/ci/receipt_provenance.py:94](../../scripts/ci/receipt_provenance.py#L94) `validate_scope_ids` | Validate scope arm/map labels and every native projection collision. |
| [scripts/ci/receipt_provenance.py:114](../../scripts/ci/receipt_provenance.py#L114) `directory_arm` | Independent observed-directory roster projection, using native _arm_key. |
| [scripts/ci/receipt_provenance.py:121](../../scripts/ci/receipt_provenance.py#L121) `_row_source_matches_observed` | Require the retained raw source location to identify the observed file. |
| [scripts/ci/receipt_provenance.py:135](../../scripts/ci/receipt_provenance.py#L135) `native_row_key` | Entire actual row to native _arm_key and _episode_identity; preserve precedence inputs; require directory/source and all present scenario declarations to agree. |
| [scripts/ci/receipt_provenance.py:190](../../scripts/ci/receipt_provenance.py#L190) `_sweep_summaries` | Validate independent summary IDs and contained source paths; summaries do not determine accepted native keys. |
| [scripts/ci/receipt_provenance.py:215](../../scripts/ci/receipt_provenance.py#L215) `joined_sweep_episodes` | Native raw-key uniqueness, unique agreeing summaries, exact raw/summary/path coverage; return native keys only after all checks. |
| [scripts/ci/receipt_provenance.py:368](../../scripts/ci/receipt_provenance.py#L368) `_producer_identity` | Actual native key must agree with original producer scenario/seed links before execution proof is minted. |
| [scripts/ci/receipt_provenance.py:443](../../scripts/ci/receipt_provenance.py#L443) `verify_producer_seals` | Directory selects strict proof decoding only; logical row identity is checked by _producer_identity. |
| [scripts/ci/receipt_provenance.py:523](../../scripts/ci/receipt_provenance.py#L523) `write_binding` | Fresh allocation binding requires verified producer/native identity first. |
| [scripts/ci/receipt_provenance.py:546](../../scripts/ci/receipt_provenance.py#L546) `verify_binding` | Verify frozen bindings and producer/native identity before exposing sealed rows. |
| [scripts/ci/receipt_provenance.py:586](../../scripts/ci/receipt_provenance.py#L586) `verify_optional_binding` | Collection custody entry to the same producer verifier. |
| [scripts/ci/receipt_provenance.py:595](../../scripts/ci/receipt_provenance.py#L595) `verified_raw_rows` | Committed/collection raw-source frontier: shared native join on original sealed objects; preserve proof attributes and compare outcomes/status. |

## scripts/ci/receipt_arm_execution.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/receipt_arm_execution.py:175](../../scripts/ci/receipt_arm_execution.py#L175) `_from_sealed_producer` | Mint frozen arm/scenario/source proof using native_row_key on the actual sealed row; no directory split. |
| [scripts/ci/receipt_arm_execution.py:223](../../scripts/ci/receipt_arm_execution.py#L223) `expected_algorithm` | Use previously verified arm/scenario proof to bind frozen scenario-specific algorithm overrides. |
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
| [scripts/ci/make_behaviour_receipt.py:1992](../../scripts/ci/make_behaviour_receipt.py#L1992) `derive_valid_comparison` | Require exact expected scope/grid coverage for both native source dictionaries before algorithm dispatch and full-route counts. |
| [scripts/ci/make_behaviour_receipt.py:2080](../../scripts/ci/make_behaviour_receipt.py#L2080) `write_real_row_audit` | Serialize already derived compact rows and execution audit, without choosing new IDs. |
| [scripts/ci/make_behaviour_receipt.py:2168](../../scripts/ci/make_behaviour_receipt.py#L2168) `write_receipt_and_header` | Serialize derived full-route scope/row/classification counts; cannot bypass the source/admission frontier. |
| [scripts/ci/make_behaviour_receipt.py:2295](../../scripts/ci/make_behaviour_receipt.py#L2295) `_load_scope` | Canonical scope labels before Cartesian grids. |
| [scripts/ci/make_behaviour_receipt.py:2370](../../scripts/ci/make_behaviour_receipt.py#L2370) `_prepare_sweeps` | Checked seed slice and native selector inputs; collection reads return through the shared join. |
| [scripts/ci/make_behaviour_receipt.py:2430](../../scripts/ci/make_behaviour_receipt.py#L2430) `_sweep_episode_paths` | Transport paths indexed only by shared native keys after agreement and coverage. |
| [scripts/ci/make_behaviour_receipt.py:2438](../../scripts/ci/make_behaviour_receipt.py#L2438) `_sweep_execution_statuses` | Summary statuses indexed only by shared native keys after agreement and coverage. |
| [scripts/ci/make_behaviour_receipt.py:2446](../../scripts/ci/make_behaviour_receipt.py#L2446) `_run_episode_transport` | Shard path/status aggregation after native joining; reject cross-shard duplicates. |
| [scripts/ci/make_behaviour_receipt.py:2458](../../scripts/ci/make_behaviour_receipt.py#L2458) `_verify_equivalence_tooling` | Compare archived, checked seed declarations to the expected dev slice. |
| [scripts/ci/make_behaviour_receipt.py:2507](../../scripts/ci/make_behaviour_receipt.py#L2507) `_validate_equivalence_sources` | Private producer executes the same verified raw-source and committed projection rules. |
| [scripts/ci/make_behaviour_receipt.py:2525](../../scripts/ci/make_behaviour_receipt.py#L2525) `_make_equivalence_receipt` | Canonical scope/seeds, native joined sources, exact 14 by 51 by 3 grid and expected 2142 episodes before equality/serialization. |
| [scripts/ci/make_behaviour_receipt.py:2683](../../scripts/ci/make_behaviour_receipt.py#L2683) `main` | Full/equivalence producer dispatch; no alternate source identity extraction. |

## scripts/ci/default_path_equivalence.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/default_path_equivalence.py:749](../../scripts/ci/default_path_equivalence.py#L749) `episode_bytes` | Compare preserved raw evidence including identity bytes; does not assign pair identities. |
| [scripts/ci/default_path_equivalence.py:821](../../scripts/ci/default_path_equivalence.py#L821) `validate_scope_entries` | Canonical scope before counts or Cartesian products. |
| [scripts/ci/default_path_equivalence.py:835](../../scripts/ci/default_path_equivalence.py#L835) `comparison_rows` | Validate scope and incoming native dictionary keys, require exact paired grid before execution/hash comparisons. |
| [scripts/ci/default_path_equivalence.py:940](../../scripts/ci/default_path_equivalence.py#L940) `compare_rows` | Comparison wrapper around the same validated native key path. |
| [scripts/ci/default_path_equivalence.py:951](../../scripts/ci/default_path_equivalence.py#L951) `validate_archive_paths` | Bind row transports to verified allocation archives; no independent roster extraction. |
| [scripts/ci/default_path_equivalence.py:968](../../scripts/ci/default_path_equivalence.py#L968) `validate_receipt` | Canonical committed keys, unique full scope/seed grid, archive and execution/review bindings; actual raw identity is re-derived by source admission. |

## scripts/ci/behaviour_receipt.py

| Site | Identity use and guard |
| --- | --- |
| [scripts/ci/behaviour_receipt.py:474](../../scripts/ci/behaviour_receipt.py#L474) `_key` | Canonical compact arm/map/seed declarations; compact keys are independently matched against native source keys. |
| [scripts/ci/behaviour_receipt.py:502](../../scripts/ci/behaviour_receipt.py#L502) `validate_receipt` | Full-route scope/key/classification and expected-grid counts, using source-derived execution proof. |
| [scripts/ci/behaviour_receipt.py:718](../../scripts/ci/behaviour_receipt.py#L718) `read_committed_payload_bytes` | Regular-file and LFS are only transport alternatives; both decode and admit identical source semantics. |
| [scripts/ci/behaviour_receipt.py:794](../../scripts/ci/behaviour_receipt.py#L794) `decode_committed_payload` | Committed source-artifact revalidation before compact evidence is accepted. |
| [scripts/ci/behaviour_receipt.py:841](../../scripts/ci/behaviour_receipt.py#L841) `load_committed_payload` | Regular/LFS committed admission wrapper. |
| [scripts/ci/behaviour_receipt.py:846](../../scripts/ci/behaviour_receipt.py#L846) `load_receipt` | Committed full-route wrapper plus restored native source execution proofs. |
| [scripts/ci/behaviour_receipt.py:871](../../scripts/ci/behaviour_receipt.py#L871) `admit_equivalence_payload` | Common semantic admission for PR and both queue transport paths. |
| [scripts/ci/behaviour_receipt.py:990](../../scripts/ci/behaviour_receipt.py#L990) `_equivalence_source_projection` | Shared metadata, row, path/status joins; compare canonical native source keys and original raw observations. |
| [scripts/ci/behaviour_receipt.py:1083](../../scripts/ci/behaviour_receipt.py#L1083) `_verify_equivalence_projection` | Index committed claims by canonical keys and require exact source-derived projection/execution agreement. |
| [scripts/ci/behaviour_receipt.py:1115](../../scripts/ci/behaviour_receipt.py#L1115) `verify_source_artifacts` | Call verified_raw_rows for each sealed archive; reject cross-shard native collisions, require compact/source coverage and exact rederivation. |
| [scripts/ci/behaviour_receipt.py:1243](../../scripts/ci/behaviour_receipt.py#L1243) `_restore_execution_proofs` | Match canonical compact keys to already verified native/source-bound proof objects. |
| [scripts/ci/behaviour_receipt.py:1263](../../scripts/ci/behaviour_receipt.py#L1263) `check_receipt` | PR admission enters shared committed decode/source and semantic validators. |

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
| [scripts/validation/run_empty_world_sweep.py:511](../../scripts/validation/run_empty_world_sweep.py#L511) | `reconcile_execution`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_config.py:386](../../robot_sf/benchmark/camera_ready/_config.py#L386) | `_sanitize_name`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_config.py:732](../../robot_sf/benchmark/camera_ready/_config.py#L732) | `_scenario_name_key`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_run_state.py:81](../../robot_sf/benchmark/camera_ready/_run_state.py#L81) | `_coerce_identifier`; runtime source unchanged |
| [robot_sf/benchmark/camera_ready/_run_state.py:95](../../robot_sf/benchmark/camera_ready/_run_state.py#L95) | `_episode_identity`; runtime source unchanged |
| [robot_sf/training/scenario_loader.py:1804](../../robot_sf/training/scenario_loader.py#L1804) | `_scenario_runtime_identity`; runtime source unchanged |
| [robot_sf/training/scenario_loader.py:1573](../../robot_sf/training/scenario_loader.py#L1573) | `select_scenario`; runtime source unchanged |

The native reconciler is also an independent test oracle: the original 2142-row fixture is complete; setting every actual `_sweep_arm` to `ppo` leaves 153 unique pairs with 1989 missing and duplicate slots. The receipt producer, committed admission and regular/LFS queue must all refuse the coherently resealed evidence.

Validation includes individual directory/row/summary forgeries through every reader, a changed native-function priority input that proves the full actual dictionary is passed, canonical-ID and strict trace/action regressions, and the original sealed #10342 pilot. These checks launch no simulation or receipt batch and claim historical compatibility only.
