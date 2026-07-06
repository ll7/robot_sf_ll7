# Issue #4232 Closure Audit

This audit maps the live issue #4232 acceptance criteria to merged PR evidence as of
2026-07-06. It is a closure-audit artifact only: it does not run a full benchmark
campaign, submit Slurm/GPU work, promote a paper or dissertation claim, or close the
issue by itself.

## Verdict

Issue #4232 is not fully closed by the merged CPU-only slices. The implementation and
diagnostic evidence path is complete through observed nonzero-alpha mechanism
activation, but benchmark-strength campaign evidence and later claim gating remain
outside the authorized local slice.

## Acceptance Evidence

| Criterion | Status | Evidence | Boundary |
| --- | --- | --- | --- |
| Preserve #4229 as runtime integrity only; do not reinterpret the fixed-seed smoke as benchmark evidence. | Met | PR #4256 records the claim boundary in `configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml`; PR #4559 and PR #4626 retain all rows as `diagnostic_only`. | No benchmark, safety, performance, conformal, paper, or dissertation claim is made. |
| Add a pre-registration packet defining evidence tiers, claim modes, forbidden claims, and execution boundary. | Met | PR #4256 added `uncertainty-envelope-claim-packet.v1` with `run_benchmark=false`, `compute_submit_authorized=false`, allowed diagnostic modes, forbidden claim modes, and evidence tiers. | Packet is planning evidence, not empirical benchmark evidence. |
| Decide paired alpha/control benchmark scope before any campaign. | Met | PR #4256 predeclares `classic_interactions_francis2023.yaml`, fixed seeds, `differential_drive`, MPC-family planners, `envelope_off_alpha_0`, `envelope_on_alpha_0`, and nonzero alpha arms. | Scenario/planner roster is predeclared for later campaign use. |
| Add visible campaign-arm configuration or deterministic runner inputs. | Met | PR #4256 records alpha arms in the packet; PR #4521 added `scripts/benchmark/run_issue_4232_uncertainty_envelope_alpha_sweep.py` to resolve packet arms deterministically. | No ad hoc claim-promoting CLI override is required for the diagnostic slice. |
| Add static validation before campaign execution and focused tests. | Met | PR #4256 added `scripts/validation/check_issue_4232_uncertainty_envelope_claim_packet.py` and tests covering missing regression arm, negative alpha, mismatched surfaces, non-MPC claims, forbidden claims, and row-status exclusions. | Validator fails closed before any campaign submission. |
| Define metrics and stop/revise rules before running. | Met | PR #4256 packet declares primary safety/performance tradeoff metrics, secondary diagnostics, alpha-zero equivalence, fallback-only, safety/runtime revision, and claim-review entry stop rules. | Metrics and stop rules remain pre-claim gates. |
| Run a CPU diagnostic alpha slice first. | Met | PR #4521 added the runner; PR #4559 tracked a diagnostic-smoke evidence bundle; PR #4576 emitted measured runtime telemetry; PR #4626 tracked an activation probe on `francis2023_crowd_navigation` seed 111. | Diagnostic-only; all compact rows remain excluded from benchmark-strength evidence. |
| Resolve the post-#4576 mechanism-activation blocker. | Met | PR #4626 records `envelope_on_alpha_0p10` as `activated`, with `effective_radius_used_by_planner=true` and `envelope_activation_count=1254285` in `docs/context/evidence/issue_4232_uncertainty_envelope_activation_probe_2026-07/envelope_activation_diagnostics.json`. | Mechanism activation is observed on a suitable diagnostic row; no safety/performance improvement is inferred. |
| Submit a private Slurm benchmark campaign only if justified by diagnostics. | Not met in authorized scope | The live issue comment after PR #4626 states the remaining work is the actual benchmark-strength campaign under a later authorized claim-promotion task. | This run has `compute_submit=false`; no Slurm/GPU/campaign submission is authorized. |
| Build compact benchmark evidence with provenance and checksums if a benchmark is run. | Partially met | PR #4281 added the evidence builder; PR #4559 and PR #4626 tracked compact diagnostic bundles with `metadata.json`, tables, diagnostics, and `SHA256SUMS`. | Compact diagnostic evidence exists. Benchmark-strength evidence does not, because the benchmark campaign has not run. |
| Exclude fallback, degraded, not-available, failed, blocked, and diagnostic-only rows from benchmark-strength evidence. | Met | PR #4256 row-status policy and PR #4281 evidence builder enforce exclusions; PR #4559 and PR #4626 row-status audits show zero benchmark-strength rows. | Exclusions are working as intended. |
| Gate claim-map, report, paper, or dissertation edits on bounded supported evidence and maintainer sign-off. | Met for current slices | PR #4559 and PR #4626 claim-readiness artifacts keep the result diagnostic-only and not ready for benchmark-strength, conformal, safety, paper, or dissertation claims. | No claim-map, benchmark-report, paper, or dissertation edit is made. |
| Avoid conformal-calibration, deployment-safety, real-world safety, and generalized planner-superiority claims. | Met | Packet forbidden-claim modes and claim-readiness files explicitly block those claims. | This audit does not weaken the claim boundary. |

## Residual Work

The only remaining issue-level blocker is empirical: run the authorized benchmark-strength
campaign or explicitly record a maintainer decision to close #4232 at diagnostic-only
status. Until then, the correct closure call is `Refs #4232`, not `Closes #4232`.

## Source Threads Reviewed

- Issue #4232 full thread through owner comment created at 2026-07-06T11:25:53Z.
- PR #4229: runtime completion boundary for issue #4141.
- PR #4256: pre-registration packet and static validator.
- PR #4281: fixture-driven compact evidence builder.
- PR #4521: CPU-local diagnostic alpha-sweep runner.
- PR #4559: tracked diagnostic-smoke evidence bundle.
- PR #4576: measured runtime activation telemetry.
- PR #4626: activation diagnostic probe evidence.

