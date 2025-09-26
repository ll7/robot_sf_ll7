# Feature Specification: Consolidate Episode Schema Definitions

**Feature Branch**: `136-consolidate-episode-schema`
**Created**: 2025-09-26
**Status**: Draft
**Input**: User description: "There seems to be significant duplication of this episode schema definition across the repository. I've found identical or very similar definitions in the following files: robot_sf/benchmark/schemas/episode.schema.v1.json specs/120-social-navigation-benchmark-plan/contracts/episode.schema.v1.json Maintaining multiple copies of the same schema is a significant maintainability risk, as any changes must be manually synchronized across all files, which is error-prone. To avoid inconsistencies, I strongly recommend refactoring to a single source of truth for this schema. You could have a canonical schema file that is then copied or referenced by other parts of the system during a build or setup step."

## Clarifications

### Session 2025-09-26
- Q: What enforcement mechanism should prevent creation of new duplicate schema files? → A: Git hooks that block commits containing duplicate schemas
- Q: How should schema evolution and versioning be handled when the canonical schema changes? → A: Semantic versioning with major.minor.patch format and breaking change detection
- Q: How should schema synchronization work during the build process? → A: Runtime resolution where code loads schema from canonical location
- Q: How should conflicts be resolved when multiple schema versions exist simultaneously? → A: Manual conflict resolution with developer intervention required

## Execution Flow (main)
```
1. Parse user description from Input
   → Identified: episode schema duplication across multiple files
2. Extract key concepts from description
   → Actors: developers maintaining schema definitions
   → Actions: consolidate duplicate schemas, establish single source of truth
   → Data: JSON schema files for episode metrics
   → Constraints: maintain backward compatibility, ensure build-time consistency
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → Clear user flow identified for schema consolidation
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
As a developer maintaining the episode schema definitions, I want a single source of truth for episode schemas so that I don't have to manually synchronize changes across multiple duplicate files, reducing the risk of inconsistencies and maintenance overhead.

### Acceptance Scenarios
1. **Given** multiple duplicate episode schema files exist in the repository, **When** I make a change to the schema, **Then** the change should automatically propagate to all locations where the schema is used
2. **Given** a new episode schema version is created, **When** the build process runs, **Then** all dependent systems should use the updated schema without manual intervention
3. **Given** the schema consolidation is complete, **When** I search for episode schema files, **Then** I should find only the canonical schema file and references/copies

### Edge Cases
- What happens when existing code expects the schema in a specific location?
- How does the system handle schema validation during development vs production?
- What happens if the canonical schema becomes corrupted or unavailable?
- How are conflicts resolved when multiple schema versions exist simultaneously?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST maintain a single canonical episode schema file as the source of truth
- **FR-002**: System MUST enable runtime resolution where code loads schema from the canonical location
- **FR-003**: System MUST preserve backward compatibility for existing schema consumers
- **FR-004**: System MUST provide clear documentation on how to reference the canonical schema
- **FR-005**: System MUST validate schema integrity and consistency across all usage locations
- **FR-006**: System MUST prevent creation of new duplicate schema files using git hooks that block commits containing duplicate schemas
- **FR-007**: System MUST use semantic versioning (major.minor.patch) for schema evolution with automatic breaking change detection
- **FR-008**: System MUST require manual conflict resolution with developer intervention when multiple schema versions exist simultaneously

### Key Entities *(include if feature involves data)*
- **EpisodeSchema**: JSON schema definition for episode metrics data structure
- **SchemaReference**: Runtime resolution mechanism where code loads schema directly from the canonical location
- **SchemaVersion**: Version identifier using semantic versioning (major.minor.patch) for schema evolution and compatibility tracking, with automatic breaking change detection

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
- [ ] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed

---
