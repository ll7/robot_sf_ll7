
# Implementation Plan: Full Classic Interaction Benchmark

**Branch**: `122-full-classic-interaction` | **Date**: 2025-09-19 | **Spec**: `specs/122-full-classic-interaction/spec.md`
**Input**: Feature specification from `/specs/122-full-classic-interaction/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
End‑to‑end benchmark for the classic interaction scenario matrix producing: (a) JSONL episodes with stable schema, (b) aggregated metrics (grouped by archetype & density) with bootstrap CIs and effect sizes, (c) statistical sufficiency report validating CI half‑width thresholds, (d) standardized plots (distributions, KDE spatial density, trajectory overlays, Pareto/SNQI, force interaction heatmaps), (e) annotated representative videos, all orchestrated by a single reproducible script `classic_benchmark_full.py` supporting resume, parallel workers, smoke mode (<2 min), and deterministic seeding. Artifacts stored under a timestamped results root with structured subfolders.

High‑level approach: Reuse existing `run_batch` episode execution core; add orchestration layer that (1) expands scenario matrix + seed plan, (2) streams/validates episodes, (3) triggers aggregation + bootstrap, (4) computes effect sizes & precision evaluation with early‑stop option, (5) generates plots/videos via modular producers. All configuration (workers, samples/precision, bootstrap params, SNQI weights) exposed via CLI args and a small BenchmarkConfig dataclass.

## Technical Context
**Language/Version**: Python >=3.10 (repo targets 3.12 in Ruff config; ensure compatibility)  
**Primary Dependencies**: gymnasium, numpy, pandas, scipy, matplotlib, seaborn (optional analysis extra), moviepy (video), pysocialforce (submodule physics), stable-baselines3 (policy episodes), jsonschema, loguru, rich.  
**Storage**: Local filesystem only (JSONL episode logs, JSON summaries, PDF/PNG plots, MP4 videos).  
**Testing**: pytest (main + GUI + fast-pysf suites); add new unit tests for aggregation/effect size + smoke integration test for full script.  
**Target Platform**: Headless macOS/Linux (CI) with optional display disabled (`SDL_VIDEODRIVER=dummy`).  
**Project Type**: Single Python package (Option 1).  
**Performance Goals**: Full run <4h on reference hardware; smoke mode <2m (p95); episode throughput baseline ~22 steps/sec; parallel scaling efficiency ≥80% up to 8 workers.  
**Constraints**: Deterministic seeds; memory footprint <2GB for full benchmark; no network calls; reproducible figures (vector PDF); resume idempotency.  
**Scale/Scope**: ~8 archetype×density combinations (exact from scenario matrix) × target episodes per scenario (initial estimate 200) = O(1600) episodes; each ≤500 steps horizon; episodic JSONL size manageable (<200MB).  

UNKNOWNS TO RESOLVE IN PHASE 0 (convert to decisions): reference hardware spec, minimum episodes formula & early stop criteria, effect size definitions per metric, video annotation detail set, scaling acceptance metric.

## Constitution Check
Initial alignment (pre‑research):
1. Reproducibility: Will record git hash, scenario matrix hash, config, SNQI weight file path → OK.
2. Factory Abstraction: Benchmark invokes environments only via factory functions → OK.
3. Benchmark & Metrics First: All outputs are JSONL + structured plots; no opaque intermediate state → OK.
4. Unified Config: Introduce `BenchmarkConfig` dataclass (no ad‑hoc globals) → OK.
5. Minimal Baselines: Uses existing planners (no new baseline) → OK.
6. Metrics Transparency: Provide per-metric CIs + effect sizes; no single hidden scalar → OK.
7. Backward Compatibility: Episode schema unchanged; additions only in separate summary/manifest files → OK.
8. Documentation as API: Will add quickstart + link from central docs index → PENDING (Phase 1 deliverable).
9. Test Coverage: Add smoke test + unit tests for aggregation/effect size → PENDING.
10. Scope Discipline: No out‑of‑scope features (no new RL algos) → OK.

No violations requiring Complexity Tracking at this stage.

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 1 (existing single Python package). New script under `scripts/` + utilities under `robot_sf/benchmark/` (submodule: e.g., `full_classic/`). Plots/videos modules placed in `robot_sf/benchmark/visualization/` (if directory does not exist, create) to keep domain separation.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh copilot` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*Description only; tasks.md generated by /tasks.*

Refinements for this feature:
* Derive tasks per Functional Requirement FR-001..FR-020 ensuring 1:1 coverage mapping in tasks.md.
* For precision goals introduce adaptive sampling tasks (loop: evaluate CI → decide continue/stop).
* Parallelizable ([P]) tasks: plot generators, video annotation module, effect size calculator, statistical report generator (independent once schema defined).
* Sequential core: schema/unit tests → episode orchestrator → aggregation pipeline → effect size + sufficiency evaluation → plots → videos → documentation linking.

Task Generation Strategy:
1. Enumerate entities (from data-model.md) → creation tasks.
2. Enumerate contracts (benchmark_full_contract.md) → interface test stubs.
3. Map acceptance scenarios → integration test tasks (smoke + full run dry-run).
4. Add quality gate tasks (lint/type/test, docs index update, changelog if user-facing script).
5. Add reproducibility tasks (manifest hash verification, resume idempotency test).

Ordering Strategy:
1. Contract & data model tests first (fail initially).
2. Orchestrator skeleton returning empty manifest (get tests executing early).
3. Episode job expansion + resume logic.
4. Aggregation + bootstrap + effect sizes.
5. Statistical sufficiency & early stop.
6. Plot producers (can be parallel).
7. Video generator (can be parallel after episodes exist).
8. Documentation + quickstart linking + final validation script updates.

Estimated Output: ~32–38 tasks (slightly higher due to statistical sufficiency + adaptive sampling + plots/videos separation).

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
Currently empty – no constitutional deviations identified.


## Progress Tracking
Phase 0 (research) and Phase 1 (design & contracts) completed; all prior NEEDS CLARIFICATION items resolved.

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
