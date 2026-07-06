# Issue #4291 Closure Audit

Created: 2026-07-04
Updated: 2026-07-06 (consolidation — folds in PRs #4519 and #4572; the code lane is now complete
and the residual is a single maintainer-owned data-publication step)

Issue: [#4291](https://github.com/ll7/robot_sf_ll7/issues/4291)

Merged implementation PRs (all landed on `main`):

| PR | Role |
| --- | --- |
| [#4296](https://github.com/ll7/robot_sf_ll7/pull/4296) | Generator `scripts/tools/generate_socnavbench_traversible.py` + skip-if-absent / `--dry-run` tests + runbook docs. Delivered all three issue-body scope items. |
| [#4519](https://github.com/ll7/robot_sf_ll7/pull/4519) | Additive `output_tree_sha256` / `output_tree_file_count` / `output_tree_total_size_bytes` registry-style checksum metadata in generator reports + tests + this evidence note. Fail-closed behavior unchanged. |
| [#4572](https://github.com/ll7/robot_sf_ll7/pull/4572) | `--pin-report-json` maintainer-review sidecar (`expected_tree_sha256`, registry owner `socnavbench-s3dis-eth`, no-raw-data guardrails, fail-closed when `data.pkl` is absent) + tests. |

Plain-language summary: issue #4291 asked for a reproducible way to generate the missing
SocNavBench ETH traversible file (`traversibles/ETH/data.pkl`) from staged local mesh data, so that
#1134 (ETH map conversion) can unblock. All requested code, tests, and docs are merged. The issue
is **not** fully closed by repository code alone because the live maintainer comments keep it open
for the data-side publication step: run the real mesh build in the trusted SocNavBench environment,
re-seed the internal store, and pin the external-data registry checksum. That step needs the staged
ETH mesh and the trusted environment, which are maintainer-owned and out of scope for this lane.

SocNavBench means Social Navigation Benchmark. ETH is the ETH pedestrian map from that dataset
family.

## Acceptance Criteria Evidence

| Criterion from #4291 | Evidence | Closure status |
| --- | --- | --- |
| Scope 1 — Add `scripts/tools/generate_socnavbench_traversible.py` (or extend the wrapper) so staged ETH mesh data can produce `traversibles/<MAP>/data.pkl` in the data root, never in git. | PR #4296 added `scripts/tools/generate_socnavbench_traversible.py`; the wrapper builds `traversibles/<MAP>/data.pkl` from staged per-map mesh into the data root. Contract recorded in `docs/context/issue_4291_socnavbench_traversible_generation.md`. | Met (#4296). |
| Scope 1 — Print a tree hash so the registry pin can be updated once the trusted artifact exists. | PR #4519 adds `output_tree_sha256` (+ file count and total size) to reports for existing and newly generated `data.pkl`. PR #4572 adds the `--pin-report-json` sidecar that surfaces `expected_tree_sha256` for maintainer review before the `socnavbench-s3dis-eth` registry pin. Covered by `tests/tools/test_generate_socnavbench_traversible.py` (`test_preflight_already_present_reports_hash`, `test_registry_pin_report_surfaces_expected_tree_sha256`). | Met (#4519 + #4572). |
| Scope 2 — Skip-if-absent smoke behavior: fail clearly when the mesh is not staged, and support `--dry-run` without building. | PR #4296 added the tests; a missing mesh exits `2` with status `blocked_missing_mesh` (`STATUS_BLOCKED_MISSING_MESH`, `scripts/tools/generate_socnavbench_traversible.py:76`), and `--dry-run` validates inputs without building. Verified by `test_dry_run_absent_mesh_exits_blocked`, `test_dry_run_staged_mesh_exits_ok_without_building`, `test_main_blocked_without_dry_run`. | Met (#4296). |
| Scope 3 — Document the generation command, expected output path, and derived-data policy. | PR #4296 updated `docs/socnav_assets_setup.md` (custom-map traversible generation section, wraps `generate_socnavbench_traversible.py`) and added `docs/context/issue_4291_socnavbench_traversible_generation.md` stating the output is derived data that stays in the data root and is never committed. | Met (#4296). |
| Out of scope — Do not run generation in CI and do not commit generated artifacts. | Tracked changes across all three PRs are tool, tests, docs, and evidence only; no `data.pkl` is committed and no CI generation job was added. The mesh pipeline (`dotmap`, `mp_env`) is imported lazily only on an explicit build. | Honored. |
| Out of scope — Keep actual internal run, hub re-seed, and registry hash update as maintainer post-merge work. | Live issue comments (2026-07-03, 2026-07-04, 2026-07-05) keep #4291 open for the trusted real mesh build, internal-store re-seed, and external-data registry pin. `docs/context/issue_1498_state.yaml` records local generated-file evidence; the `socnavbench-s3dis-eth` registry entry in `scripts/tools/manage_external_data.py` still carries no pinned `expected_tree_sha256`. | Intentionally deferred to maintainer. |

## Integration Report

- **Blockers remaining (maintainer-owned):** trusted mesh-based build in the SocNavBench environment
  → internal-store re-seed → external-data registry checksum pin for `socnavbench-s3dis-eth`. These
  require the staged ETH mesh and the trusted environment; no repository code path can substitute.
- **New blockers:** none. The three merged PRs are additive and preserve fail-closed behavior; no
  regression or new dependency was introduced.
- **Intentional (by issue design):** the run + re-seed + pin steps are declared out of scope in the
  issue body ("The actual internal run + hub re-seed + registry hash update is a maintainer step
  after merge"). The issue is intentionally kept open as a tracker for that step.
- **Next empirical action (maintainer):**
  1. Stage the ETH mesh, then run
     `uv run python scripts/tools/generate_socnavbench_traversible.py --map ETH` in the trusted
     SocNavBench environment; capture the printed `output_tree_sha256`.
  2. Write the pin-review sidecar with `--pin-report-json <path>` and review `expected_tree_sha256`.
  3. Re-seed the internal store and pin `expected_tree_sha256` into the `socnavbench-s3dis-eth`
     registry entry in `scripts/tools/manage_external_data.py`.
  4. Re-run `validate_socnav_map_batch.py --batch-id eth_first --preflight` (expect `ready`), then
     proceed with #1134.

## Closure Decision

Code, tests, and docs acceptance criteria for #4291 are **fully met** by merged PRs #4296, #4519,
and #4572. Do **not** create another generator or checker micro-slice — the remaining work is the
maintainer-owned data-publication step above, not a repository code gap. Close #4291 after that step
is done and the `socnavbench-s3dis-eth` registry pin is recorded.

This audit did not run a full benchmark campaign, submit Slurm or GPU work, edit paper or
dissertation claims, or commit generated SocNavBench data.
</content>
