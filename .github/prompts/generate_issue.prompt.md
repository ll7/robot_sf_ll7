---
mode: agent
---
You are drafting a high‑quality GitHub Issue for this repository (social navigation RL framework). Produce a concise, actionable issue description that enables fast implementation while capturing necessary context and scope boundaries.

Follow the structure below. Omit any section that would be truly empty, but keep ordering. Use clear bullet points, not prose walls. Prefer present tense and imperative voice ("Add", "Refactor", "Clarify"). Avoid speculative language unless explicitly marking open questions.

==================================================
TEMPLATE SECTIONS
==================================================
1. Summary
	- One sentence problem framing (WHAT and WHY in <= 140 chars if possible).
	- If it’s a regression, say: Regression since <commit/tag>.

2. Context / Motivation
	- Brief background (what subsystem: env factory, benchmark CLI, recording/playback, fast-pysf wrapper, training script, etc.).
	- Why this matters (accuracy, performance, DX, maintainability, API clarity, test reliability, etc.).
	- Link to: related issues / PRs / docs (e.g. `docs/refactoring/`, `copilot-instructions.md`, design notes) if relevant.

3. Goals (In Scope)
	- Bullet list of concrete objectives. Each should be testable / reviewable.
	- Use measurable or observable phrasing ("MapDefinition serialized into metadata", "Switch timestamp semantics to simulation time", etc.).

4. Non‑Goals (Out of Scope)
	- Explicitly state what will NOT be addressed to prevent scope creep (e.g., "No hyperparameter tuning", "No new RL algorithms", "No GPU optimization here").

5. Proposed Approach (Initial Draft)
	- Optional if solution unclear. Otherwise outline likely implementation strategy.
	- Reference architecture patterns (e.g., factory functions only, avoid direct env constructors, reuse existing config dataclasses, streaming JSONL schema stability, etc.).
	- Mention expected new/changed files and any data model adjustments.

6. Data / API / Schema Impact
	- Does it change JSONL recording schema? (If yes, specify version bump + migration path.)
	- Does it add/modify config objects in `robot_sf.gym_env.unified_config`?
	- Backward compatibility expectations.

7. Acceptance Criteria
	- Structured bullet list; reviewer must be able to tick each.
	- Include quality gates (lint, tests, type-check, perf threshold if relevant, docs updated, validation scripts pass).
	- Example:
		- [ ] New `map_definition` persisted in episode metadata when available.
		- [ ] Playback reconstructs map without fallback inference when metadata present.
		- [ ] All tests (176) pass locally; no increase in runtime > +5%.
		- [ ] Added targeted unit tests covering edge cases X, Y.
		- [ ] Docs updated (`docs/<feature>/README.md`).

8. Risks / Trade‑offs
	- Performance regressions, increased memory, schema churn, coupling, test fragility.
	- Mitigations.

9. Open Questions
	- Convert ambiguities into multiple‑choice where possible (e.g., "Store map as full object vs. minimal bounds? (A) full JSON, (B) compressed, (C) just bounds").

10. Estimation
	- Provide a size label suggestion: XS (<1h), S (1–2h), M (0.5–1d), L (1–2d), XL (>2d) ignoring review time.

11. Suggested Labels
	- Choose from typical set: feature, bug, tech-debt, docs, performance, refactor, test, schema, playback, training, env-factory.

12. Validation Plan
	- Commands / scripts to run (e.g., validation scripts in `scripts/validation/`, perf benchmark, demo scripts).
	- Include exact acceptance threshold if performance-sensitive.

13. Definition of Done Reminder (auto‑trim if redundant)
	- All acceptance criteria satisfied.
	- Quality gates: Ruff clean, pylint errors-only clean, type check (no errors), tests green.
	- Docs & examples updated where relevant.
	- No stray large artifacts committed.

==================================================
STYLE & QUALITY GUIDELINES
==================================================
Clarity
	- Prefer structure over narrative. Keep each bullet single concept.
	- Avoid future tense; write as actionable requirements.

Specificity
	- Avoid vague words: "improve", "optimize" without a metric; provide target (e.g., "Reduce step time p95 from 55ms → <50ms").

Traceability
	- Link any affected config, class, or function using backticks (e.g., `make_robot_env`, `JsonlRecorder`, `MapDefinition`).

Backward Compatibility
	- Always note if existing JSONL logs remain loadable.

Testing
	- Indicate which test suites are touched: core (`tests/`), GUI (`test_pygame/`), submodule (`fast-pysf/tests`).
	- If adding new tests, specify rough count and focus.

Performance
	- If risk, define measurement method (e.g., run `scripts/benchmark02.py`, compare steps/sec baseline ~22). Provide acceptable regression budget (default <5%).

Documentation
	- For non-trivial changes: create `docs/<issue-number>-<slug>/README.md` following repo doc standards.

Security / Safety
	- Note if network, file system, threading changes occur.

Labels & Automation
	- Suggest labels; maintainers finalize.

==================================================
EXAMPLE (ABBREVIATED)
==================================================
Summary
Add serialization of full `MapDefinition` into JSONL metadata to eliminate playback inference ambiguity.

Context / Motivation
Playback reconstruction currently infers bounds → potential scaling mismatch for complex obstacle layouts.

Goals
- Persist `map_definition` JSON in episode metadata.
- Prefer metadata map over inferred map in playback.
- Add unit test ensuring fidelity (width/height, obstacles count).

Non-Goals
- No change to route planning logic.
- No compression of map assets.

Data / API / Schema Impact
- Adds optional `map_definition` key to episode `.meta.json`.
- Backward compatible (fallback to inference as today).

Acceptance Criteria
- [ ] Metadata includes `map_definition` when available.
- [ ] Playback chooses metadata map if present.
- [ ] New test passes and entire suite (176 tests) green.
- [ ] Docs added under `docs/123-map-definition-metadata/`.

Risks / Trade-offs
- Slight metadata size increase (<10KB typical). Acceptable.

Open Questions
1. Include full obstacle geometry? (A) yes, (B) bounds only.

Estimation
S (2–3 focused hours incl. tests & docs).

Validation Plan
Run: lint → tests → demo playback script; inspect map visually.

==================================================
OUTPUT FORMAT REQUIREMENTS FOR THIS PROMPT
==================================================
When using this prompt to generate an issue automatically, the assistant should:
1. Produce only the filled-out issue body (no frontmatter, no extra commentary).
2. Preserve section headings exactly as listed (omit ones intentionally dropped because empty).
3. Use GitHub task list syntax for Acceptance Criteria.
4. Wrap code identifiers in backticks.
5. Keep line width reasonable (<110 chars) but do not hard-wrap awkwardly.

If information is missing, convert to explicit Open Questions rather than guessing.

End of instructions.