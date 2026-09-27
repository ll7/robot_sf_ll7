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

# Current context map (2026-09-26)

- **Primary:** `robot_sf/adversarial/coevolution.py` and `tests/adversarial/test_coevolution.py`; merged #9650 optimizer owner `scripts/validation/planner_optimizer.py`; bounded search owner `robot_sf/adversarial/search.py` and `robot_sf/adversarial/config.py`.
- **Verified interface detail:** the merged optimizer emits a policy-search candidate wrapper as `best_candidate.yaml`, `candidate_registry.yaml`, and `run_manifest.json`; falsification `SearchConfig` forwards `algo_config_path` to its benchmark runner and `run_adversarial_search` records each candidate evaluation. `prepare_falsification_search` now resolves the optimizer wrapper through the canonical policy-search loader, materializes the merged runtime config per round, and binds the source/derived digests and previous planner identity. `ProductionFalsificationAdapter` calls the existing bounded search and preserves its generated files as phase artifacts. Its no-simulator test uses injected evaluator/certifier callables. The planner family stays experimental.
- **Coordinator carry-forward repair:** Round N+1 now freezes Round N's selected planner identity/config digest in its input and requires the optimizer adapter to bind that exact identity as its baseline. The run/round artifact schema is v2. This closes the fixture-state-machine reset gap; it does not yet provide or validate the production optimizer adapter's actual config derivation.
- **Adjacent:** #9651 verdict candidate exposes `classify_scenario_admissibility` and `validate_scenario_admissibility`, but its PR is still gated by final readiness. #9652's second adversarial persisted downgrade/rekey finding is being repaired and needs another independent exact review before a production corpus adapter. #9647's reviewed renderer/materializer is usable for replay presentation, but not a feasibility oracle. #9645 remains NO-GO for #9648.
- **Validation:** fixture suite `tests/adversarial/test_coevolution.py`; owner tests for optimizer, search, admissibility, corpus, replay; then exact-head review and PR readiness. One finite tiny real run only after the real adapters and all owner contracts are reviewed/merged; retain every failed and invalid evaluation.
- **Risks:** current coordinator is fixture-only. It accepts case IDs without a durable source resolver, so those IDs cannot safely stand in for scenario payloads or corpus provenance. Candidate upstream API shapes may change during review. Real search may validly return no new admissible case under its finite budget.
- **Open questions:** bind challenge IDs to exact immutable scenarios from a read-only corpus snapshot; expose/consume one reviewed case scenario representation for held-out and regression evaluation, replay, admissibility, and corpus admission; implement an optimizer adapter that derives its candidate baseline from the frozen prior planner identity and digest; define the finite tiny run's explicit runtime/evaluation budget while satisfying #9653's two-round minimum.

# Steps

1. Inspect and record the stable public APIs and schemas from merged/current code and reviewed upstream branches; do not integrate known-broken candidate APIs.
2. Define the smallest round manifest/config that binds source revision, planner/config, map/scenario inputs, sampler, seeds, budgets, case IDs, artifacts, statuses, and stop decision.
3. Implement the orchestrator as a thin coordinator over existing entry points. Deep-freeze and digest each round before execution; journal paired manifest transitions and verify persisted inputs before resume.
4. Add fixture tests for two rounds, round-N-to-N+1 regression flow, no-discovery, invalid/mismatch/unknown preservation, stop rules, budget exhaustion, infrastructure failure, sampler identity, case-payload immutability, manifest crash recovery, and resume identity.
5. Once the upstream contracts are reviewed and available, run one finite tiny two-round end-to-end demonstration. Require two rounds unless an infrastructure gate fails; preserve a no-discovery result as budget-qualified.
6. Generate a machine-readable receipt/report, perform independent exact-head review, repair findings, and run delivery readiness after the #9651 readiness lane clears.

# Decisions and risks

- Observed: the #9645 pilot found no objective-changing critical case and explicitly says NO-GO for scaled #9648. No scaled campaign is authorized by this plan.
- Observed: current `origin/main` is `03b699c607677a6b73652c5211f3400a66ffca54`; this worktree is refreshed on branch `autopilot/issue9653-coevolution-current-main-20260926` at `27922f2f7c6393cef3a104bee49fef5f683eb63c`. The old branch name and base below are stale and superseded.
- Observed: #9651 has an exact-review PASS on its current-main head but final PR readiness is running. #9652 has an active second repair for a persisted downgrade/rekey finding. #9656's repaired stack is rebased onto current main and awaiting a fresh exact review. #9654's three report integrity/determinism findings are repaired and awaiting independent review.
- Decision: the tiny integration run is a plumbing/finite-budget demonstration; no signal means only that no admissible case was found within the recorded budget.
- Decision: feasibility `unknown`, source identity gaps, replay mismatches, and degraded/fallback rows remain explicit and cannot be counted as admitted or solved.
- Decision: adapters must use and echo the previous selected planner's stable ID and raw-config digest as the next round's optimizer baseline; a static-baseline reset is an infrastructure diagnostic, not a valid optimization round.
- Risk: #9651/#9652/#9647 candidate contracts are still under repair/review; integration against them before exact review could bake in unsound identity or corpus behavior.

# Validation route

- Focused fixtures for the orchestrator and owner-specific regression tests for optimizer/search/admission/replay/corpus.
- Ruff, format, changed-doc evidence/link checks, `git diff --check`, and exact-head implementation review.
- One explicitly budgeted tiny integration run after upstream review; no repeated run unless a material repair changes the execution path.
- PR-wide readiness only after the explicit #9651 readiness gate clears.

# Recovery / handoff

- Worktree: `/home/luttkule/git/robot_sf_ll7.worktrees/issue-9653-coevolution-20260924`; branch `autopilot/issue9653-coevolution-current-main-20260926`.
- Current source base: `origin/main=03b699c607677a6b73652c5211f3400a66ffca54`; worktree is bootstrapped with its local `.venv` and `local.machine.md` link. The pre-refresh state is preserved by Git history and the superseded remote branch.
- Store reproducible round manifests and the tiny run packet in a versioned, checksummed `docs/context/evidence/issue_9653/` bundle for downstream #9654 consumption; keep private review/control receipts under the common Git-dir task-artifact area. Never rely on worktree-local ignored output as durable evidence. Resume from verified manifests and hashes rather than rerunning a completed experiment.
