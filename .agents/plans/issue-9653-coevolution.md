# Goal

Implement one config-driven command for a provenance-bound tiny two-round planner↔falsifier loop. Reuse the existing optimizer, held-out evaluator, bounded falsification, replay, feasibility/admissibility, and corpus owners.

# Scope

- In scope: immutable round inputs/outputs; explicit finite optimizer and search budgets; separate optimization and held-out/regression sets; resume validation; a round-N case evaluated by round N+1; safe stop diagnostics; one small real run after upstream contracts are usable.
- Out of scope: replacing optimizer/search/replay/admission/corpus internals, any scaled #9648 campaign, fabricating or reconstructing evidence, mathematical feasibility or safety claims, and global-optimum claims.
- Acceptance: one command runs at least two bounded rounds, persists exact provenance and every evaluation, and emits a report whose statuses preserve invalid, degraded/fallback, infeasible, and unknown cases.

# Evidence sources

- Live #9653 issue body and dependency/autonomy comment; #9645 pilot synthesis (explicit NO-GO for #9648); #9650 optimizer; #9651 admissibility; #9652 corpus; #9645/#9646 falsification; #9647 replay; #9657 showcase.
- Existing owners: `scripts/validation/planner_optimizer.py`, `robot_sf/adversarial/search.py`, `robot_sf/adversarial/search_harness.py`, and candidate APIs in the #9651, #9652, and #9647 worktrees. Treat candidate APIs as provisional until their exact reviews and repairs pass.
- `AGENTS.md`, `docs/dev/worktree_lifecycle.md`, `docs/code_review.md`, `.agents/PLANS.md`, and `docs/benchmark_governance.md`.

# Steps

1. Inspect and record the stable public APIs and schemas from merged/current code and reviewed upstream branches; do not integrate known-broken candidate APIs.
2. Define the smallest round manifest/config that binds source revision, planner/config, map/scenario inputs, sampler, seeds, budgets, case IDs, artifacts, statuses, and stop decision.
3. Implement the orchestrator as a thin coordinator over existing entry points. Deep-freeze and digest each round before execution; journal paired manifest transitions and verify persisted inputs before resume.
4. Add fixture tests for two rounds, round-N-to-N+1 regression flow, no-discovery, invalid/mismatch/unknown preservation, stop rules, budget exhaustion, infrastructure failure, sampler identity, case-payload immutability, manifest crash recovery, and resume identity.
5. Once the upstream contracts are reviewed and available, run one finite tiny two-round end-to-end demonstration. Require two rounds unless an infrastructure gate fails; preserve a no-discovery result as budget-qualified.
6. Generate a machine-readable receipt/report, perform independent exact-head review, repair findings, and run delivery readiness after the #9651 readiness lane clears.

# Decisions and risks

- Observed: the #9645 pilot found no objective-changing critical case and explicitly says NO-GO for scaled #9648. No scaled campaign is authorized by this plan.
- Decision: the tiny integration run is a plumbing/finite-budget demonstration; no signal means only that no admissible case was found within the recorded budget.
- Decision: feasibility `unknown`, source identity gaps, replay mismatches, and degraded/fallback rows remain explicit and cannot be counted as admitted or solved.
- Risk: #9651/#9652/#9647 candidate contracts are still under repair/review; integration against them before exact review could bake in unsound identity or corpus behavior.

# Validation route

- Focused fixtures for the orchestrator and owner-specific regression tests for optimizer/search/admission/replay/corpus.
- Ruff, format, changed-doc evidence/link checks, `git diff --check`, and exact-head implementation review.
- One explicitly budgeted tiny integration run after upstream review; no repeated run unless a material repair changes the execution path.
- PR-wide readiness only after the explicit #9651 readiness gate clears.

# Recovery / handoff

- Worktree: `/home/luttkule/git/robot_sf_ll7.worktrees/issue-9653-coevolution-20260924`; branch `autopilot/issue9653-coevolution-20260924`.
- Current source base: `origin/main=07750662c1eabb3c650c07236cd7c1c41d7baa4a`; worktree was fast-forwarded and bootstrapped with its local `.venv` and `local.machine.md` link.
- Store reproducible round manifests and the tiny run packet in a versioned, checksummed `docs/context/evidence/issue_9653/` bundle for downstream #9654 consumption; keep private review/control receipts under the common Git-dir task-artifact area. Never rely on worktree-local ignored output as durable evidence. Resume from verified manifests and hashes rather than rerunning a completed experiment.
