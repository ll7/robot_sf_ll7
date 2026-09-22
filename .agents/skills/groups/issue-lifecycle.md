# Issue Lifecycle Skills

Use `gh-issue-sequencer` for Project #5 queue order,
`issue-audit-autonomous` for deterministic issue cleanup,
`issue-audit` for one-at-a-time maintainer decision envelopes,
`issue-contract-maintainer` for routing and `goal-issue-implementation` for both selected-issue and
multi-issue loops. The legacy `gh-issue-autopilot`, `issue-to-pr`, and `gh-issue-to-pr` names are
compatibility aliases for the selected-issue mode.

## Relationship contract

Every issue-producing or issue-repair skill reads `docs/context/issue_relationships.md` and keeps a
canonical `## Relationships` block with explicit `none` values when no edge is established. Native
Parent, Blocked by, and Blocking links are set and read back only from an owning writable worktree;
`Relates to` remains a manual UI association. Legacy headings and incidental mentions are review
evidence, never write authorization. When work flows into a PR, the PR mirrors the reviewed native
state in `## Issue Relationship Mirror` while retaining `Closes`/`Refs` for coverage semantics.
