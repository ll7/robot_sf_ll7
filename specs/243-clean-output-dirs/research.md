# Research Findings: Clean Root Output Directories

**Feature Branch**: `243-clean-output-dirs`
**Date**: November 13, 2025

## Summary

The repository currently allows benchmark, coverage, and recording artifacts to accumulate in multiple top-level paths (e.g., `results/`, `recordings/`, `tmp/`, `wandb/`, `htmlcov/`). Consolidating these into a single, configurable artifact root improves discoverability, new-contributor experience, and guardrail enforcement. Research focused on best practices for artifact segregation in Python research repositories and strategies for introducing breaking path changes with minimal disruption.

## Decisions & Rationale

### Decision: Adopt a single `output/` artifact root
- **Rationale**: A single entry point keeps the repository root clean and simplifies guard checks. Keeping artifacts under one directory reduces the number of paths scripts must manage and avoids git-ignore drift. Comparable research repos (e.g., RL baselines, OpenAI gym forks) centralize results in `output/` or `runs/` directories for the same reason.
- **Alternatives considered**:
  - Maintain separate `.cache/` and `results/` roots. Rejected because two roots still clutter the repository root and complicate guard enforcement.
  - Leave existing layout unchanged. Rejected because it fails the issue goal and keeps contributor onboarding friction high.

### Decision: Enforce fail-fast errors on legacy paths after remediation
- **Rationale**: Guarding against regressions requires a strict enforcement mechanism. Once high-traffic producers (tests, validation scripts, benchmark harness) are updated to use the new structure, retaining legacy writes leads to silent backsliding. Failing fast with actionable errors preserves the clean state and signals missing migrations.
- **Alternatives considered**:
  - Transitional symlinks. Rejected because symlinks clutter version control on macOS/Windows and make it harder to detect stragglers.
  - Permanent backward compatibility. Rejected because it undermines the goal of cleaning the root and makes guard checks toothless.

### Decision: Provide migration tooling and documentation
- **Rationale**: Contributors often have existing artifacts; a helper script that relocates known directories protects data while guiding manual cleanup. Documentation updates ensure the new layout is discoverable and reduces support burden.
- **Alternatives considered**:
  - Manual instructions only. Rejected because manual steps are easy to miss and risk accidental deletion.

## Best Practices Applied

- Use environment variable (`ROBOT_SF_ARTIFACT_ROOT`) as primary override, mirroring existing repo pattern.
- Treat guard checks as part of CI and pre-commit workflows for consistent enforcement.
- Document cleanup and overrides in `docs/dev_guide.md` and README so new contributors discover the policy quickly.

## Outstanding Questions

None. All clarifications resolved prior to planning.
