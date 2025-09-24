---
description: Update the project constitution following governance requirements.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

Given the constitution update description, do this:

1. Read the current constitution from `.specify/memory/constitution.md` to understand existing principles and governance.
2. If the update requires a written proposal, create `docs/dev/issues/<topic>/design.md` with:
   - Problem statement and justification
   - Proposed changes to principles/contracts
   - Impact analysis on existing functionality
   - Migration plan if contracts change
3. Update the constitution file with the approved changes, following the governance requirements:
   - Explicit enumeration of affected contracts (env, config, metrics, benchmark schema)
   - Migration guidance or deprecation plan
   - Version/date update in the footer
4. Ensure the update aligns with Core Principles I-X and doesn't introduce out-of-scope functionality.
5. Report completion with updated version number and summary of changes.

Note: Constitution amendments require careful consideration and must maintain backward compatibility unless explicitly versioned.