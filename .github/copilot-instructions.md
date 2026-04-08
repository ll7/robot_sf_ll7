# Copilot Instructions

ALWAYS use the official [dev_guide](../docs/dev_guide.md) as the primary reference for development-related tasks.
It is everyones guide on how to use this repository effectively.

## Additional Instructions

- Use scriptable interfaces instead of cli interfaces when possible.
- Make everything reproducible.
- In benchmark work, use the canonical fail-closed fallback policy in
  `docs/context/issue_691_benchmark_fallback_policy.md`.
- For GitHub issue batches and Project #5 writes, follow the batch-first workflow in
  `docs/context/issue_713_batch_first_issue_workflow.md`: clean up issues first, route project
  metadata second, run derived score sync last, and cache IDs once per shell session.
- For measurement-driven improvement loops, use `.agents/skills/autoresearch/SKILL.md`.
  For shorter refinement passes, use `.agents/skills/auto-improvement/SKILL.md`.
  See `docs/ai/awesome_copilot_adaptation.md` for the selection guide.
- For context discovery, use `.agents/skills/context-map/SKILL.md` before multi-file changes and
  `.agents/skills/what-context-needed/SKILL.md` when the task is underspecified.
- For proof-first quality work, use `.agents/skills/quality-playbook/SKILL.md` for non-trivial
  changes, `.agents/skills/agentic-eval/SKILL.md` for AI-workflow artifacts, and
  `.agents/skills/review-and-refactor/SKILL.md` for narrow review-then-refactor passes.
- For doc synchronization, use `.agents/skills/update-docs-on-code-change/SKILL.md` when code
  changes would make docs stale.
- Prefer GitHub MCP / GitHub app tools for interactive issue, PR, and project work.
- Keep `gh` for deterministic batch automation, score sync, and auth/debugging fallback.
- Before opening a PR, follow the latest-main sync rule in
  `docs/dev_guide.md`: fetch `origin/main`, merge or rebase it into the feature branch, then run
  `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`.
- Central point to link new documentation pages is `docs/README.md`.
  - Link new documentation (sub-)pages in the appropriate section.
- For any changes that affect users, update the `CHANGELOG.md` file.
- Source the environment before using python or uv `source .venv/bin/activate`.

## Recent Changes
- 343-telemetry-viz: Added Python 3.11 + Pygame/SDL, matplotlib (agg), NumPy, robot_sf.telemetry (manifest writer, sampler), Loguru

- 270-imitation-report: Added Python 3.11 (existing uv-managed environment)
- 001-map-verification: Added Python 3.11 (uv-managed virtual environment) + `robot_sf.gym_env` factories + unified configs, Loguru logging, SVG parsing utilities already present in `robot_sf.maps`, optional geometry helpers (Shapely)

## Active Technologies
- Python 3.11 + Pygame/SDL, matplotlib (agg), NumPy, robot_sf.telemetry (manifest writer, sampler), Loguru (343-telemetry-viz)
- Append-only JSONL telemetry + PNG/SVG summary artifacts under `output/` (respecting artifact policy) (343-telemetry-viz)


## Test Failure Evaluation

**When encountering test failures, always evaluate test significance before fixing:**

1. **Verify test value first**: Not all tests are equally important. Ask:

   - Does this test verify a core public contract (factories, schemas, metrics)?
   - Would this failure impact users in production?
   - Is this testing a known regression or critical edge case?
   - Is the test brittle/flaky (timing, display, environmental issues)?
   - Is the same logic already covered by other tests?

2. **Priority classification**:

   - **High** (fix immediately): Core features, user-facing behavior, schema compliance
   - **Medium** (fix within sprint): Known regressions, important edge cases
   - **Low** (consider archiving): Rare scenarios, redundant coverage, no real-world incidents
   - **Flaky** (refactor or remove): Frequently fails without indicating real bugs

3. **Actions based on priority**:

   - High priority: Fix immediately to protect critical invariants
   - Medium priority: Fix or document deferral with tracking issue
   - Low priority: Consider archiving with documented rationale
   - Flaky: Stabilize with retries/mocks if valuable, else remove

4. **Documentation requirements**:
   - Removing tests requires commit message explaining why
   - New tests need docstring: (1) what is verified, (2) why it matters
   - Deferred failures need "test-debt" label and value assessment

**Reference**: See Constitution Principle XIII and dev_guide.md Testing Strategy for complete criteria.
