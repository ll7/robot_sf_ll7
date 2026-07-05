# Issue #4520 GPU Memory Closure Audit

Plain-language summary: issue
[#4520](https://github.com/ll7/robot_sf_ll7/issues/4520) has an implemented CPU-testable fix for
serial benchmark arms leaking graphics processing unit (GPU) memory across arms. Merged pull
request [#4528](https://github.com/ll7/robot_sf_ll7/pull/4528) added teardown in the serial
map-runner path and a regression test that verifies the teardown hook runs between consecutive
serial arms. This audit does not rerun the failed Slurm (Simple Linux Utility for Resource
Management) GPU campaign and does not promote a benchmark, paper, or dissertation claim.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4520>
- Audit date: 2026-07-05.
- Live issue thread reviewed through owner comment
  <https://github.com/ll7/robot_sf_ll7/issues/4520#issuecomment-4884145347>.
- Merged pull request reviewed:
  [#4528](https://github.com/ll7/robot_sf_ll7/pull/4528), merge commit `d615dc682`,
  "Issue #4520: Release GPU memory between serial arms (cheap-lane worker)".

## Claim Boundary

This is a closure-audit evidence note. It records the criterion-to-evidence mapping for the merged
code/test slice. It does not run the original 30-seed (S30) hybrid-vs-Optimal Reciprocal Collision
Avoidance (ORCA) GPU campaign, submit compute,
rerun Slurm job 13296, close the issue, or assert that the full campaign now passes.

## Acceptance Mapping

| Acceptance criterion from issue #4520 | Evidence | Status |
| --- | --- | --- |
| Serial multi-arm benchmark execution releases GPU memory between arms or planners so a late arm cannot inherit all earlier PyTorch allocations. | PR #4528 adds a `finally` block in `robot_sf/benchmark/map_runner_batch_runner.py::_serial_execute_map_jobs` that runs `gc.collect()` after every serial job and calls `torch.cuda.empty_cache()` when Torch is already loaded and CUDA is available. It also sets `PYTORCH_ALLOC_CONF=expandable_segments:True` as defense in depth. | Met for repository code path. |
| Add a regression test that proves allocated GPU memory is released or the teardown hook is invoked between consecutive serial arms; CPU-only simulation is acceptable. | PR #4528 adds `tests/benchmark/test_issue_4520_gpu_memory.py`, which monkeypatches `torch.cuda.is_available`, `torch.cuda.empty_cache`, and `gc.collect`, then runs two serial jobs through `_serial_execute_map_jobs` and asserts both teardown hooks are invoked twice. | Met. |
| Keep continuous integration CPU-only; no GPU campaign is required for the fix PR. | The regression test uses monkeypatching and does not require a GPU. PR #4528 validation reported `tests/benchmark/test_issue_4520_gpu_memory.py` plus map-runner utility tests passing locally, and the owner gate comment verified the CPU-only boundary. | Met. |
| Do not treat the fix as proof that the original S30 campaign now succeeds until a separate rerun exists. | PR #4528 and the owner gate comment explicitly deferred the actual S30 `ppo` rerun to the private-ops queue and did not claim campaign success. | Met as claim boundary; empirical campaign rerun remains out of scope for this audit. |

## Closure Decision

From the public issue and PR evidence available on 2026-07-05, the issue's code and test acceptance
criteria are met by PR #4528. No additional repository code slice is justified by this audit. The
remaining non-code action is issue-state propagation: close issue #4520 or record a status update
that closure waits only for maintainer-chosen post-fix campaign rerun evidence.

## Local Verification

Audit-time validation is docs-only and code-path inspection:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/benchmark/test_issue_4520_gpu_memory.py
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_4520_gpu_memory_closure_audit_2026-07-05.md \
  --path docs/context/INDEX.md \
  --path docs/context/README.md
git diff --check
```
