# Issue #5006 Residual Review-Thread Ledger

This ledger makes the remaining post-merge review work visible: each advisory thread from the
first sweep is linked to proof that it was already covered, or is explicitly left for a maintainer
decision or a bounded follow-up. It audits the 33 unresolved threads classified in merged
[PR #5046](https://github.com/ll7/robot_sf_ll7/pull/5046) against `origin/main` at
`7fb2339590246b70d96d8d48e2ffed2a7f0871d8` on 2026-07-11.

## Claim Boundary

This is workflow bookkeeping, not benchmark evidence. `covered` means the requested code or
documentation state is visible in the linked merged change. `superseded` means a later merged
change intentionally replaced the reviewed contract. `decision_required` means the suggestion
still depends on a maintainer choice; it is not a silent rejection. `follow_up_required` means the
suggestion remains mechanically applicable but implementing it is outside this documentation-only
audit. No unresolved thread is represented as fixed.

## Audit Summary

| Disposition | Threads | Meaning |
| --- | ---: | --- |
| `covered` | 19 | The requested state is present in a linked merged PR or commit. |
| `superseded` | 3 | A later merged contract made the original suggestion inapplicable. |
| `decision_required` | 7 | A maintainer must choose between plausible contracts or editorial outcomes. |
| `follow_up_required` | 4 | The suggestion is still actionable but needs a separate implementation slice. |
| **Total** | **33** | Matches the residual count reported by PR #5046. |

## Residual Entries

| Source | Review thread | Disposition | Mechanical evidence or decision needed |
| --- | --- | --- | --- |
| Issue #4890 | [cache `restore-keys`](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843510) | `superseded` | [PR #5117](https://github.com/ll7/robot_sf_ll7/pull/5117) removed cross-lockfile cache restoration to prevent oversized restores. |
| Issue #4890 | [validate retry backoff inputs](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843515) | `covered` | [PR #4890](https://github.com/ll7/robot_sf_ll7/pull/4890) added non-negative integer validation for both backoff variables. |
| Issue #4890 | [diagnostic test for `restore-keys`](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843527) | `superseded` | [PR #5117](https://github.com/ll7/robot_sf_ll7/pull/5117) intentionally removed the behavior the assertion would require. |
| Issue #4890 | [retry test for `restore-keys`](https://github.com/ll7/robot_sf_ll7/pull/4890#discussion_r3545843534) | `superseded` | [PR #5117](https://github.com/ll7/robot_sf_ll7/pull/5117) intentionally removed the behavior the assertion would require. |
| Issue #4928 | [`kinematics_feasibility` required-key consistency](https://github.com/ll7/robot_sf_ll7/pull/4928#discussion_r3549512251) | `decision_required` | Decide whether policy-build metadata must contain the unfinished episode-level block, or whether the characterization test should document it as episode-finalized. |
| Issue #4931 | [reproducibility test docstring and timestamp input](https://github.com/ll7/robot_sf_ll7/pull/4931#discussion_r3550014113) | `follow_up_required` | [Issue #5227](https://github.com/ll7/robot_sf_ll7/issues/5227) owns aligning the ignored `timestamp` argument, class docstring, and later pinned-output tests. |
| Issue #4936 | [pin exact timestamp in record-builder test](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338668) | `decision_required` | Decide whether the wall-clock fallback test should remain intentionally dynamic or be replaced by an injected-clock contract. |
| Issue #4936 | [`None` sentinel in group-crossing bundle](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338690) | `decision_required` | Decide whether an explicit empty timestamp is invalid and should fall back, or must be preserved distinctly from `None`. |
| Issue #4936 | [`None` sentinel in head-on bundle](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338698) | `decision_required` | Apply the same timestamp-empty-value ruling to both exemplar exporters. |
| Issue #4936 | [deterministic group-crossing selection report](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338705) | `covered` | [PR #4938](https://github.com/ll7/robot_sf_ll7/pull/4938) threads the pinned generation time into the report and adds byte-identity tests. |
| Issue #4936 | [deterministic head-on selection report](https://github.com/ll7/robot_sf_ll7/pull/4936#discussion_r3550338710) | `covered` | [PR #4938](https://github.com/ll7/robot_sf_ll7/pull/4938) threads the pinned generation time into the report and adds byte-identity tests. |
| Issue #4940 | [duplicate planner-failure computation](https://github.com/ll7/robot_sf_ll7/pull/4940#discussion_r3551519011) | `covered` | The merged [PR #4940](https://github.com/ll7/robot_sf_ll7/pull/4940) calls `_compute_planner_failures` once; the duplicate inline block is absent. |
| Issue #4940 | [`_planner_run` fallback guard](https://github.com/ll7/robot_sf_ll7/pull/4940#discussion_r3551519018) | `covered` | The merged [PR #4940](https://github.com/ll7/robot_sf_ll7/pull/4940) uses an explicit `None`-aware planner-key helper. |
| Issue #4940 | [`_derive_mechanism_label` return type](https://github.com/ll7/robot_sf_ll7/pull/4940#discussion_r3551519031) | `covered` | The merged [PR #4940](https://github.com/ll7/robot_sf_ll7/pull/4940) declares `dict[str, Any] | None`. |
| Issue #4945 | [import finite-value support](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851277) | `covered` | The merged [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) imports `math` and centralizes finite parsing in `_finite_float`. |
| Issue #4945 | [filter non-finite comparison values](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851291) | `covered` | [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) routes parsed comparison values through `_finite_float`. |
| Issue #4945 | [filter non-finite confidence values](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851301) | `covered` | [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) routes confidence values through `_finite_float`. |
| Issue #4945 | [filter non-finite variability values](https://github.com/ll7/robot_sf_ll7/pull/4945#discussion_r3551851308) | `covered` | [PR #4945](https://github.com/ll7/robot_sf_ll7/pull/4945) routes variability values through `_finite_float`. |
| Issue #4948 | [isolate the macOS worker cap](https://github.com/ll7/robot_sf_ll7/pull/4948#discussion_r3551957882) | `covered` | The merged [PR #4948](https://github.com/ll7/robot_sf_ll7/pull/4948) returns immediately after applying the macOS-specific bounds. |
| Issue #4949 | [filter non-finite minimum separation](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552390432) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) added `_min_finite_or_inf` and uses it for the summary. |
| Issue #4949 | [normalize explicit `obs_mode=None`](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552390442) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) normalizes `None` before string coercion. |
| Issue #4949 | [broaden `_build_policy` return payload type](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552390446) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) types the callable payload as `Any` and documents the holonomic dictionary form. |
| Issue #4949 | [include the `sf` alias in provenance text](https://github.com/ll7/robot_sf_ll7/pull/4949#discussion_r3552437601) | `covered` | [PR #4951](https://github.com/ll7/robot_sf_ll7/pull/4951) handles `{"social_force", "sf"}` together. |
| Issue #4951 | [preserve falsy `obs_mode` values](https://github.com/ll7/robot_sf_ll7/pull/4951#discussion_r3552609180) | `decision_required` | Current code intentionally maps both `None` and an empty string to the default. Decide whether empty-string preservation is a supported configuration contract. |
| Issue #4952 | [exception-safe pytest-tree cleanup](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743567) | `covered` | The merged [PR #4952](https://github.com/ll7/robot_sf_ll7/pull/4952) calls `_terminate_process_tree` from `finally`. |
| Issue #4952 | [remove redundant `is_running`](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743586) | `follow_up_required` | [Issue #5228](https://github.com/ll7/robot_sf_ll7/issues/5228) owns the diagnostic-only cleanup and focused test. |
| Issue #4952 | [handle `AccessDenied` when attaching](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743589) | `covered` | The merged [PR #4952](https://github.com/ll7/robot_sf_ll7/pull/4952) catches both `NoSuchProcess` and `AccessDenied`. |
| Issue #4952 | [validate projection arguments before the sweep](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743594) | `follow_up_required` | [Issue #5228](https://github.com/ll7/robot_sf_ll7/issues/5228) owns early ceiling, auto-worker, and projection-point validation. |
| Issue #4952 | [terminate the test child in `finally`](https://github.com/ll7/robot_sf_ll7/pull/4952#discussion_r3552743601) | `follow_up_required` | [Issue #5228](https://github.com/ll7/robot_sf_ll7/issues/5228) owns exception-safe test-child termination. |
| Issue #4953 | [deep-copy finalized `algo_meta`](https://github.com/ll7/robot_sf_ll7/pull/4953#discussion_r3552867587) | `covered` | [PR #4955](https://github.com/ll7/robot_sf_ll7/pull/4955) deep-copies the metadata before finalization mutations. |
| Issue #5022 | [validate `evaluation_replacement` as a file](https://github.com/ll7/robot_sf_ll7/pull/5022#discussion_r3558181347) | `covered` | [PR #5038](https://github.com/ll7/robot_sf_ll7/pull/5038) adds an `is_file()` contract and invalid-path tests. |
| Issue #5024 | [format canonical LiDAR literals](https://github.com/ll7/robot_sf_ll7/pull/5024#discussion_r3558259838) | `decision_required` | Decide whether the table should show exact Python literals or retain reader-facing units in the value column. |
| Issue #5024 | [format legacy LiDAR literals](https://github.com/ll7/robot_sf_ll7/pull/5024#discussion_r3558259846) | `decision_required` | Apply the same exact-literal-versus-reader-units ruling to the legacy preset table. |

## Next Actions

- Maintainers can adjudicate the seven `decision_required` rows without reconstructing the original
  review context; each row states the smallest concrete choice.
- The four `follow_up_required` rows should be implemented only in a bounded code/test slice. This
  ledger does not invent acceptance of those suggestions or broaden this documentation-only PR.
- After a successor merges, replace the row's disposition with `covered`, add its PR or commit link,
  and resolve or reply to the source thread when GitHub permissions allow.
