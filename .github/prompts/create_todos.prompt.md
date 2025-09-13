---
mode: agent
---
You are tasked with generating a structured TODO list for this repository by:
1. Mining the current codebase (comments, docstrings, filenames, docs) for pending work signals.
2. Leveraging recent conversation / issue context (if provided to the model) for implicit follow-ups.

Deliver a concise, prioritized, categorized TODO set suitable for maintainers to convert directly into issues or sprint tasks.

==================================================
OBJECTIVE
==================================================
Produce a high-signal list of actionable tasks (not vague intentions) with clear titles, short rationale, and suggested sizing.

==================================================
INPUT SIGNAL SOURCES (SCAN IN THIS ORDER)
==================================================
1. Code Markers
	- Inline comments containing: TODO, FIXME, BUG, HACK, NOTE (only if implies action), OPTIMIZE, PERFORMANCE.
	- Pattern: case-insensitive; also detect variations like `# todo:` or `#TODO(`.
2. Docstrings & Markdown
	- Sections titled: Limitations, Future Work, Next Steps, Known Issues.
3. Repository Structure
	- Empty placeholder files or obvious stubs (e.g., modules with only pass, or doc folders missing README).
4. Conversation / History
	- Unimplemented suggestions, deferred enhancements, optional improvements previously listed (e.g., metadata map serialization, schema versioning, retrofitting old logs).
5. Configuration & CI
	- Lints disabled with comments (`noqa`, `pylint: disable`) if they mask tech debt actionable items.

==================================================
CLASSIFICATION CATEGORIES
==================================================
Use one primary category per task (choose best-fit):
  feature, bug, test, docs, performance, refactor, tech-debt, build, ci, schema, tooling, playback, env-factory, training, benchmark, data, infra

==================================================
PRIORITIZATION HEURISTICS
==================================================
Priority (P0–P3):
  P0 Critical breakage / data corruption / security risk.
  P1 High-impact correctness, developer friction, schema drift risk.
  P2 Valuable but not urgent improvements, performance tuning, clarity.
  P3 Nice-to-have polish or exploratory items.

Assign the lowest priority number that justifies action. Default to P2 if uncertain.

==================================================
ESTIMATION SCALE
==================================================
XS < 30m
S  30m–2h
M  Half day – 1 day
L  1–2 days
XL > 2 days / needs design doc

Flag XL items with: requires design doc.

==================================================
TASK QUALITY RULES
==================================================
Each TODO must include:
  - Title (imperative, 6–10 words, unique)
  - Category
  - Priority (P0–P3)
  - Size (XS–XL)
  - Rationale (1 concise sentence)
  - Acceptance Criteria (1–4 bullets, verifiable)
  - (Optional) Blocking Dependencies

Avoid:
  - Vague verbs (improve, optimize) without metric.
  - Combining unrelated concerns in one task.
  - Restating code comments verbatim without synthesis.

==================================================
DERIVATION LOG (OPTIONAL SECTION)
==================================================
If ambiguity high, include a short derivation log listing raw signals → interpreted task (max 6 lines) after the TODO list.

==================================================
OUTPUT FORMAT
==================================================
Produce Markdown using this structure:

### TODOs
1. Title — category | P? | Size ?
	- Rationale: ...
	- Acceptance:
	  - [ ] ...
	  - [ ] ...
	- Dependencies: (omit if none)

Group tasks by Priority (P0 → P3) with a heading per priority if >1 task.

If fewer than 3 tasks found, explicitly state: "Low signal: consider deeper scan" and propose at least 2 inferred improvements based on architecture.

==================================================
SOURCE-SPECIFIC SIGNAL INTERPRETATION
==================================================
1. Map / Playback Enhancements
	- If metadata map serialization missing → task candidate.
2. Schema Evolution
	- Simulation timestamp migration suggests adding schema version constant + migration doc.
3. Testing Gaps
	- Add explicit test for timestamp semantics (episode_start=0.0 invariant).
4. Analytics / Tooling
	- Potential script to retro-convert old logs to new schema.

==================================================
FAILSAFES
==================================================
If conflicting tasks found (duplicate intent), merge into one with clearer scope.
If a task seems multi-part (e.g., record + playback + docs), either:
  - Split into coherent subtasks (≤ 3) OR
  - Keep as single if atomic deliverable is small.

==================================================
EXAMPLE (ABBREVIATED)
==================================================
### TODOs
P1
1. Persist map definition in metadata — schema | P1 | S
	- Rationale: Eliminates scaling ambiguity in playback.
	- Acceptance:
	  - [ ] `map_definition` saved in `.meta.json` when available
	  - [ ] Playback prefers metadata map over inference
	  - [ ] Added unit test validating obstacle count fidelity

P2
2. Add schema version constant — schema | P2 | XS
	- Rationale: Make future JSONL changes traceable.
	- Acceptance:
	  - [ ] Version constant defined in single module
	  - [ ] Recorder embeds version in metadata
	  - [ ] Docs mention migration path

==================================================
EXECUTION NOTES FOR ASSISTANT USING THIS PROMPT
==================================================
1. Do NOT fabricate nonexistent files; only infer from signals.
2. If conversation history mentions optional ideas, treat them as P2 unless urgency implied.
3. Keep total tasks ≤ 15 unless explicitly asked for exhaustive list.
4. Prefer fewer, higher-quality tasks over noisy enumeration.
5. If performance tasks suggested, include metric & baseline reference.

End of instructions.