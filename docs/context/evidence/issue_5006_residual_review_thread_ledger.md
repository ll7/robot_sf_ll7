# Issue #5006 Residual Review-Thread Ledger (Resolved)

This ledger audits the 33 unresolved threads classified in merged
[merged pull request (PR) #5046](https://github.com/ll7/robot_sf_ll7/pull/5046) against `origin/main` at
`7fb2339590246b70d96d8d48e2ffed2a7f0871d8` on 2026-07-11. Issue #5006 now
finalizes the remaining 11 dispositions: the 7 `decision_required` entries are classified with
code/test evidence, and the 4 `follow_up_required` entries are resolved by the
now-closed owning issues.

## Claim Boundary

This is workflow bookkeeping, not benchmark evidence. `covered` means the requested code or
documentation state is visible in the linked merged change. `superseded` means a later merged
change intentionally replaced the reviewed contract. `reject` means the current code or docs
are the correct intentional design, supported by explicit code/test evidence documented in
the row. `follow_up_required` means the suggestion remains mechanically applicable but
implementing it is outside this documentation-only audit. No unresolved thread is represented
as fixed without evidence.

## Audit Summary

| Disposition | Threads | Meaning |
| --- | ---: | --- |
| `covered` | 23 | The requested state is present in a linked merged PR or commit (19 original + 4 follow-up items now resolved by closing PRs). |
| `superseded` | 3 | A later merged contract made the original suggestion inapplicable. |
| `reject` | 7 | Current design is correct and intentional, supported by explicit code/test evidence. |
| `follow_up_required` | 0 | All 4 items resolved by now-closed owning issues. |
| **Total** | **33** | Matches the residual count reported by PR #5046. |

## Residual Entries

| Source | Review thread | Disposition | Mechanical evidence or decision needed |
| --- | --- | --- | --- |
| Issue #4890 | [cache `restore-keys`](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843510) | `superseded` | [PR #5117](https://github.com/ll7/robot_sf_ll7/pull/5117) removed cross-lockfile cache restoration to prevent oversized restores. |
| Issue #4890 | [validate retry backoff inputs](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843515) | `covered` | [PR #4890](https://github.com/ll7/robot_sf_ll7/pull/4890) added non-negative integer validation for both backoff variables. |
| Issue #4890 | [diagnostic test for `restore-keys`](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843527) | `superseded` | [PR #5117](https://github.com/ll7/robot_sf_ll7/pull/5117) intentionally removed the behavior the assertion would require. |
| Issue #4890 | [retry test for `restore-keys`](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843534) | `superseded` | [PR #5117](https://github.com/ll7/robot_sf_ll7/pull/5117) intentionally removed the behavior the assertion would require. |
| Issue #4928 | [`kinematics_feasibility` required-key consistency](https://github.com/ll7/robot_sf_ll7/pull/4928#discussion_r3549512251) | `reject` | The key IS initialized at policy-build time by `init_feasibility_metadata()` (`planner_command_contract.py:43`), called from every builder path. The soft check in `test_map_runner_characterization.py:676` correctly documents the build-time vs. episode-finalized lifecycle: initialized form has raw accumulators (`commands_evaluated`, `infeasible_native_count`, `projected_count`, `_sum_abs_delta_*`), while `finalize_feasibility_metadata()` (`map_runner_policy_metadata.py:72`) pops internal accumulators and adds computed rates. The `REQUIRED_META_KEYS` inclusion with explicit subtraction is intentional design — the key is always present in initialized form, but its finalized structure differs. No change needed. |
| Issue #4931 | [reproducibility test docstring and timestamp input](https://github.com/ll7/robot_sf_ll7/pull/4931#discussion_r3550014113) | `covered` | [Issue #5227](https://github.com/ll7/robot_sf_ll7/issues/5227) was resolved by [PR #5231](https://github.com/ll7/robot_sf_ll7/pull/5231), aligning the ignored `timestamp` argument, class docstring, and pinned-output tests. |
| Issue #4936 | [pin exact timestamp in record-builder test](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338668) | `reject` | The pinned-timestamp mechanism works correctly: `pin_generated_at` is used verbatim when provided, `None` triggers proper `datetime.now(UTC)` fallback (`export_issue_4848_group_crossing_exemplars.py:299`, `export_issue_4891_head_on_corridor_exemplars.py:352`). Tests cover both paths: `test_pin_generated_at_overrides_wall_clock` asserts exact match, `test_pin_none_uses_wall_clock` asserts fallback, `test_pin_generated_at_byte_identical` asserts determinism from pinned input. An injected-clock contract would add speculative infrastructure — the pinned-timestamp mechanism already serves the reproducibility need. |
| Issue #4936 | [`None` sentinel in group-crossing bundle](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338690) | `reject` | `None` is the established sentinel for "omit the date." `extract_marker_date()` (`evidence/writers.py:77`) normalizes both `generated_at_utc=None` and `generated_at_utc=""` to `None` via the falsy check `if generated_at`. `review_marker()` (`evidence/writers.py:61`) handles `None` by omitting the date from the HTML marker. There is no semantic distinction worth preserving between "explicitly no date" (JSON null) and "empty date" — both mean the provenance field is absent or unset. The `None` sentinel is consistently used in both exporters and the shared library. |
| Issue #4936 | [`None` sentinel in head-on bundle](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338698) | `reject` | Same ruling as group-crossing thread above. Both exporters use the same shared library functions (`extract_marker_date`, `review_marker`) from `robot_sf.evidence.writers`, so the `None`-sentinel behavior is consistent by construction. The head-on exporter also adds a safety `None` fallback at line 629 (`if bundle_metadata else None`) for the case where `_process_planner` returns `None` — this is correct defensive behavior that does not create a new sentinel contract. |
| Issue #4936 | [deterministic group-crossing selection report](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338705) | `covered` | [PR #4938](https://github.com/ll7/robot_sf_ll7/pull/4938) threads the pinned generation time into the report and adds byte-identity tests. |
| Issue #4936 | [deterministic head-on selection report](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338710) | `covered` | [PR #4938](https://github.com/ll7/robot_sf_ll7/pull/4938) threads the pinned generation time into the report and adds byte-identity tests. |
| Issue #4940 | [duplicate planner-failure computation](https://github.com/ll7/robot_sf_ll7/pull/4940#discussion_r3551519011) | `covered` | The merged [PR #4940](https://github.com/ll7/robot_sf_ll7/pull/4940) calls `_compute_planner_failures` once; the duplicate inline block is absent. |
| Issue #4940 | [`_planner_run` fallback guard](https://github.com/ll7/robot_sf_ll7/pull/4940#discussion_r3551519018) | `covered` | The merged [PR #4940](https://github.com/ll7/robot_sf_ll7/pull/4940) uses an explicit `None`-aware planner-key helper. |
| Issue #4940 | [`_derive_mechanism_label` return type](https://github.com/ll7/robot_sf_ll7/pull/4940#discussion_r3551519031) | `covered` | The merged [PR #4940](https://github.com/ll7/robot_sf_ll7/pull/4940) declares `dict[str, Any] \| None`. |
| Issue #4945 | [import finite-value support](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851277) | `covered` | The merged [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) imports `math` and centralizes finite parsing in `_finite_float`. |
| Issue #4945 | [filter non-finite comparison values](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851291) | `covered` | [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) routes parsed comparison values through `_finite_float`. |
| Issue #4945 | [filter non-finite confidence values](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851301) | `covered` | [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) routes confidence values through `_finite_float`. |
| Issue #4945 | [filter non-finite variability values](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851308) | `covered` | [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) routes variability values through `_finite_float`. |
| Issue #4948 | [isolate the macOS worker cap](https://github.com/ll7/robot_sf_ll7/pull/4948#discussion_r3551957882) | `covered` | The merged [PR #4948](https://github.com/ll7/robot_sf_ll7/pull/4948) returns immediately after applying the macOS-specific bounds. |
| Issue #4949 | [filter non-finite minimum separation](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552390432) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) added `_min_finite_or_inf` and uses it for the summary. |
| Issue #4949 | [normalize explicit `obs_mode=None`](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552390442) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) normalizes `None` before string coercion. |
| Issue #4949 | [broaden `_build_policy` return payload type](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552390446) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) types the callable payload as `Any` and documents the holonomic dictionary form. |
| Issue #4949 | [include the `sf` alias in provenance text](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552437601) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) handles `{"social_force", "sf"}` together. |
| Issue #4951 | [preserve falsy `obs_mode` values](https://github.com/ll7/robot_sf_ll7/pull/4951#discussion_r3552609180) | `reject` | Both `None` and empty/whitespace strings correctly normalize to the default. This was the explicit fix in commit `99e377721` of PR #4951: `_resolve_planner_obs_mode()` (`map_runner.py:596`) adds `if raw_obs_mode is None or str(raw_obs_mode).strip() == "": raw_obs_mode = default` to address the real regression where `dict.get()` supplies the default only for *missing* keys, not for explicit `None`. Regression tests `test_resolve_obs_mode_normalizes_explicit_none_to_default` and `test_resolve_obs_mode_normalizes_empty_string_to_default` (`test_map_runner_decomposition_helpers.py:160-177`) cover both cases. There is no valid use case for an empty-string `obs_mode` as a distinct value — empty is not a valid observation mode, and preserving it would silently break downstream string processing. |
| Issue #4952 | [exception-safe pytest-tree cleanup](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743567) | `covered` | The merged [PR #4952](https://github.com/ll7/robot_sf_ll7/pull/4952) calls `_terminate_process_tree` from `finally`. |
| Issue #4952 | [remove redundant `is_running`](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743586) | `covered` | [Issue #5228](https://github.com/ll7/robot_sf_ll7/issues/5228) was resolved by [PR #5234](https://github.com/ll7/robot_sf_ll7/pull/5234), which closed the diagnostic-only cleanup and focused test. |
| Issue #4952 | [handle `AccessDenied` when attaching](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743589) | `covered` | The merged [PR #4952](https://github.com/ll7/robot_sf_ll7/pull/4952) catches both `NoSuchProcess` and `AccessDenied`. |
| Issue #4952 | [validate projection arguments before the sweep](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743594) | `covered` | [Issue #5228](https://github.com/ll7/robot_sf_ll7/issues/5228) was resolved by [PR #5234](https://github.com/ll7/robot_sf_ll7/pull/5234), which added early ceiling, auto-worker, and projection-point validation. |
| Issue #4952 | [terminate the test child in `finally`](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743601) | `covered` | [Issue #5228](https://github.com/ll7/robot_sf_ll7/issues/5228) was resolved by [PR #5234](https://github.com/ll7/robot_sf_ll7/pull/5234), which added exception-safe test-child termination. |
| Issue #4953 | [deep-copy finalized `algo_meta`](https://github.com/ll7/robot_sf_ll7/pull/4953#discussion_r3552867587) | `covered` | [PR #4955](https://github.com/ll7/robot_sf_ll7/pull/4955) deep-copies the metadata before finalization mutations. |
| Issue #5022 | [validate `evaluation_replacement` as a file](https://github.com/ll7/robot_sf_ll7/pull/5022#discussion_r3558181347) | `covered` | [PR #5038](https://github.com/ll7/robot_sf_ll7/pull/5038) adds an `is_file()` contract and invalid-path tests. |
| Issue #5024 | [format canonical LiDAR literals](https://github.com/ll7/robot_sf_ll7/pull/5024#discussion_r3558259838) | `reject` | The current mixed format is appropriate documentation practice. Exact Python literals (backtick-quoted) are used for config-code values (`True`, `[0.005, 0.002]`, `1.0`) where developers need the precise config syntax. Reader-facing units provide physical context for scalar quantities (`10 m`, `about 1.32 degrees per ray`) where the numeric value alone lacks intuitive meaning. The "Meaning" column adds further context. Standardizing to all-exact-literals would lose readability (e.g., `10.0` vs `10 m`), while all-reader-facing-units would lose config-precise values for code-facing settings. The mix is intentional and correct. |
| Issue #5024 | [format legacy LiDAR literals](https://github.com/ll7/robot_sf_ll7/pull/5024#discussion_r3558259846) | `reject` | Same ruling as the canonical table thread above. The ego-pedestrian variant table follows the same mixed-format convention: numeric units for physical quantities (`30 m`, `120 degrees`), parenthetical exact-literals where they aid config comprehension (`visual_angle_portion = 1 / 3`). Consistent with the canonical table approach. |

## Resolution Summary

All 33 residual threads now have finalized dispositions:
- 23 `covered`: the requested state is present in a linked merged PR or commit (including 4
  follow-up items whose owning issues #5227 and #5228 are now closed by PRs #5231 and #5234).
- 3 `superseded`: a later merged contract made the original suggestion inapplicable.
- 7 `reject`: the current design is correct and intentional, supported by explicit code/test
  evidence documented in the row above.

No pending maintainer decisions remain for this ledger. If a source thread can be resolved or
replied to on GitHub when lane permissions allow, link the disposition evidence from this row.
