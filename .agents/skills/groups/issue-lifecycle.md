# Issue Lifecycle Skills

Use `gh-issue-sequencer` for Project #5 queue order,
`issue-audit-autonomous` for deterministic issue cleanup,
`issue-audit` for one-at-a-time maintainer decision envelopes,
`issue-contract-maintainer` for routing and `goal-issue-implementation` for both selected-issue and
multi-issue loops. The legacy `gh-issue-autopilot`, `issue-to-pr`, and `gh-issue-to-pr` names are
compatibility aliases for the selected-issue mode.

## Relationship contract

Every issue-producing or issue-repair skill reads `docs/context/issue_relationships.md`. Native
Parent, Blocked by, and Blocking links are the only relationship record and are set/read back from
an owning writable worktree. Issue bodies and comments provide evidence for reviewing a candidate;
a mention alone does not establish a dependency. `Relates to` remains outside the CLI workflow.
PRs retain `Closes`/`Refs` for coverage semantics and do not mirror the native graph.
