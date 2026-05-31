# SLURM Job Discovery Snapshot (2026-05-31)

Status: Current as of 2026-05-31 10:30 Europe/Berlin

Related:

- <https://github.com/ll7/robot_sf_ll7/issues/1470>
- <https://github.com/ll7/robot_sf_ll7/issues/1472>
- <https://github.com/ll7/robot_sf_ll7/issues/1474>
- <https://github.com/ll7/robot_sf_ll7/issues/1475>
- <https://github.com/ll7/robot_sf_ll7/issues/1502>
- <https://github.com/ll7/robot_sf_ll7/issues/1503>
- <https://github.com/ll7/robot_sf_ll7/issues/1784>

## Discovery Inputs

Commands used:

```bash
squeue --me --format='%i %j %T %P %Q %y %b %M %l %S %R'
sacct -u "$USER" --starttime now-7days --format=JobID,JobName%28,State,ExitCode,Partition,Elapsed,Start,End -P
gh issue list --repo ll7/robot_sf_ll7 --state open --label slurm --limit 100 --json number,title,labels,updatedAt,url
gh issue list --repo ll7/robot_sf_ll7 --state open --search 'label:"resource:slurm" -label:"state:blocked"' --limit 100 --json number,title,labels,updatedAt,url
gh issue view <issue> --repo ll7/robot_sf_ll7 --comments --json number,title,body,comments,labels,state,url
```

Live state:

- `squeue --me` returned no running or pending jobs.
- Recent `sacct` showed completed development-stress and camera-ready smoke jobs, including
  adversarial job `12664` for #1502.
- The only open `resource:slurm` issue without `state:blocked` was #1502, and #1502 already had
  completed result evidence plus `state:needs-interpretation`.

## Candidate Classification

| Issue | Discovery state | Basis | Route |
|---|---|---|---|
| #1470 | `blocked_dependency` | comments require concrete durable dataset artifact aliases, exact collection commit, and collection wrapper/output paths | Do not submit; unblock artifact storage and launch command first. |
| #1472 | `blocked_dependency` | comments require durable trace/baseline artifact pointers and a concrete learned-risk training entrypoint | Do not submit; keep launch-packet-ready only. |
| #1474 | `blocked_dependency` | comments require exact training commit, base checkpoint, artifact destinations, and concrete guarded PPO launcher | Do not submit; preserve one-delta repair scope. |
| #1475 | `blocked_dependency` | comments require replacing pending ORCA-residual output artifact aliases before smoke and nominal execution | Do not submit; assign durable dataset/checkpoint/report pointers first. |
| #1502 | `completed_needs_analysis` | Slurm job `12664` completed and compact evidence landed in PR #1777 | Do not rerun; analyze via #1503 from tracked evidence. |
| #1503 | `analysis_only` | issue body says no new search execution; PR #1777 merged #1502 evidence | Produce a synthesis note only. |

## Workflow Insight

The useful automation boundary is not "find every issue with a `slurm` label and submit it." The
safe boundary is:

1. refresh live queue and recent job outcomes;
2. inspect open `resource:slurm` issues plus comments for blocker language;
3. classify dependency state before suitability;
4. submit only `ready_to_submit` jobs from their owning worktrees;
5. route `completed_needs_analysis` issues to result-analysis skills instead of spending another
   allocation.

This matters because several launch-packet issues are valid enough to look tempting but still
intentionally blocked on durable artifact pointers or concrete command surfaces. Submitting them
would create local-only outputs that cannot satisfy their own acceptance criteria.

## Follow-Up

- Keep `slurm-campaign-submit` responsible for the discovery gate as well as the actual `sbatch`
  suitability gate.
- When a future discovery pass finds multiple `ready_to_submit` jobs, create one worktree per
  branch/config and respect the per-partition QoS cap from `local.machine.md`.
- If the only unblocked item has completed results, create a result-analysis PR rather than a new
  SLURM job.
