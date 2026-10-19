## Shared Hazards

- SH-01 — Work only in the assigned checkout, branch, and scope; preserve other lanes' changes.
- SH-02 — Bind multi-repository Git commands to the intended checkout with `git -C <repo-root>`.
- SH-03 — Stage explicit task-owned paths and inspect the staged diff; never use `git add -A`.
- SH-04 — Never use `git stash` during parallel work; stashes are shared across linked worktrees.
- SH-05 — Stop processes only by a freshly verified PID you own; never kill by name pattern.
- SH-06 — Rewrite only your authorized feature branch with `--force-with-lease=<branch>:<expected-sha>`; verify the result.
- SH-07 — Preserve Git mutation exit status; never pipe mutating Git commands into output filters.
- SH-08 — Keep secret values out of prompts, logs, and commits, and private material out of public artifacts.
