# Feature Specification: Great Starting Set for Ruff Linting Rules Expansion

**Feature Branch**: `137-great-starting-set`  
**Created**: September 26, 2025  
**Status**: Draft  
**Input**: User description: "I'd layer in a few high-signal rule families that catch real bugs, modernize syntax, and keep imports/exception handling tidy—without turning your CI into a nag.

Here's a curated, battle-tested expansion:

[tool.ruff.lint]
select = [
  # you already have:
  "E4","E7","E9","F","W","C901","I001",

  # **Bug catchers & safety**
  "B",      # flake8-bugbear: common footguns (B006/B008/B904 are gold)
  "BLE",    # no bare/overbroad except
  "TRY",    # try/except anti-patterns (tryceratops)
  "A",      # no shadowing of builtins
  "ARG",    # unused args in public APIs
  "S",      # Bandit security checks (safe subset, see per-file ignores)

  # **Modernization & simplification**
  "UP",     # pyupgrade: keep code current
  "SIM",    # flake8-simplify: prune needless branches/loops
  "C4",     # flake8-comprehensions: cleaner comps/gens
  "PTH",    # prefer pathlib over os.path
  "ICN",    # canonical import aliases (np, pd, plt, etc.)

  # **Performance & correctness nits**
  "PERF",   # perflint: avoid tiny perf traps (informational)
  "PL",     # selected pylint-derived rules (see ignores below)

  # **Time handling**
  "DTZ",    # timezone-aware datetime usage

  # **Logging / prints / commented code**
  "G",      # logging format string correctness
  "T20",    # discourage print() outside scripts/tests
  "ERA",    # eradicate commented-out code

  # **Style that doesn't fight your formatter**
  "COM",    # trailing commas where helpful
  "ISC",    # implicit string concat gotchas

  # **Housekeeping**
  "RUF",    # Ruff's own rules (e.g., unused `noqa`)
  "PGH",    # blanket noqa / useless noqa hygiene
  "TCH",    # move typing-only imports under TYPE_CHECKING
  "TID",    # tidy imports (relative vs absolute)
  "N",      # pep8-naming
]

# Tame the noisy ones and carve out test & script latitude
ignore = [
  # Pylint-derived "opinionated" refactors that often fire in scientific code:
  "PLR0911","PLR0912","PLR0913","PLR0915", # many returns/branches/args
  "PLR2004", # magic values in comparisons (okay in tests/configs)
  "S", # Security checks not critical in research code
]

[tool.ruff.lint.per-file-ignores]
# Tests: allow asserts, prints, some magic values & security test scaffolding
"tests/**/*" = ["S101","T201","PLR2004"]
# One-off scripts / notebooks exports: allow prints
"scripts/**/*" = ["T201"]
# Examples and docs: allow prints and other leniencies
"examples/**/*" = ["T201"]
"docs/**/*" = ["T201"]

Why these:
	•	B / BLE / TRY: catch real bugs like mutable defaults, bare excepts, and sketchy try/except flows (e.g., TRY300/TRY203).  
	•	UP: nudges you to modern Python (e.g., f-strings, pathlib PurePath bits) with safe auto-fixes.  
	•	SIM / C4: replaces re-implemented builtins and collapses nested conditionals; clearer, smaller diffs.  
	•	DTZ: prevents subtle time bugs (naive datetimes), which bite in sims/logging.  
	•	S (Bandit): cheap security wins (unsafe subprocess, yaml.load, etc.). Keep a few ignores in tests.  
	•	PERF: gentle micro-optimizations; good signal, low cost.  
	•	PTH / TCH / TID / ICN: cleaner imports and faster cold-start by deferring heavy typing deps.  
	•	RUF / PGH: trims stale noqa and blanket ignores so your config stays honest.  
	•	G / T20 / ERA / COM / ISC: avoids logging/print/string pitfalls and keeps diffs neat.  

Tips:
	•	Let Ruff auto-fix what it can (ruff check --fix). You can restrict fixability if desired (e.g., only E,F,UP,SIM,C4).  
	•	Keep docstrings & annotations optional at first. If you want stricter APIs later, consider ANN (annotations) and a minimal D1xx docstring baseline in public modules.  
	•	If you prefer the "strict by default" style, one viable approach is select = ["ALL"] and then ignore what you truly dislike—useful to benefit from new rules over time. (Communities do this; just be deliberate with ignores.)"

## Clarifications

### Session 2025-09-26
- Q: What specific performance targets should the linting configuration maintain to avoid "turning CI into a nag"? → A: CI linting must complete in under 60 seconds
- Q: How should conflicts between new rules and existing code be resolved? → A: When possible, fix with ruff check --fix. When not, exclude conflicting rules from select list
- Q: Are there any specific rules or rule families that should be excluded due to the research/scientific nature of the codebase? → A: Exclude S for security checks in non-production research
- Q: What constitutes "excessive noise" in CI - any specific thresholds or examples? → A: Any increase in CI failures
- Q: Should the configuration include any additional per-file-ignores for other directories (e.g., examples/, docs/)? → A: A or even more folders

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors (developers), actions (update pyproject.toml), data (linting rules), constraints (research-heavy codebase, avoid nag)
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies  
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
I want to expand the Ruff linting rules in pyproject.toml to catch more bugs, modernize syntax, and improve code quality without making the CI overly noisy or naggy.

### Acceptance Scenarios
1. **Given** the current pyproject.toml has basic Ruff rules, **When** I apply the expanded rule set, **Then** the linter catches additional bugs like mutable defaults and bare excepts while allowing necessary flexibility in tests and scripts.
2. **Given** code with outdated patterns, **When** running ruff check --fix, **Then** it automatically modernizes to f-strings and pathlib where safe.
3. **Given** code with potential security issues, **When** linting, **Then** it flags unsafe subprocess usage but allows test scaffolding.

### Edge Cases
- What happens when new rules conflict with existing code patterns? When possible, fix with ruff check --fix. When not, exclude conflicting rules from select list.
- How does the system handle rules that might be too strict for scientific computing code? Exclude security checks (S) as not critical in research code.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST update the [tool.ruff.lint] select list to include the specified rule families (B, BLE, TRY, A, ARG, S, UP, SIM, C4, PTH, ICN, PERF, PL, DTZ, G, T20, ERA, COM, ISC, RUF, PGH, TCH, TID, N)
- **FR-002**: System MUST set ignore rules to exclude noisy pylint-derived rules (PLR0911, PLR0912, PLR0913, PLR0915, PLR2004) and security checks (S)
- **FR-003**: System MUST configure per-file-ignores to allow specific rules in tests (S101, T201, PLR2004), scripts (T201), examples (T201), and docs (T201)
- **FR-004**: The linting configuration MUST ensure CI linting completes in under 60 seconds without excessive noise (defined as any increase in CI failures)
- **FR-005**: The rules MUST be suitable for research-heavy codebases with scientific computing patterns

### Key Entities *(include if feature involves data)*
- **pyproject.toml**: Configuration file containing Ruff linting settings, including select, ignore, and per-file-ignores sections.

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
