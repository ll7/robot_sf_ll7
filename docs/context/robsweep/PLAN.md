# Examiner robustness study, 2026-10-12

Evidence tier: diagnostic-only development evidence. No dissertation admission.

Test whether success/collision rankings and scenario difficulty depend on robot
radius, preferred pedestrian speed, and planner command caps. Comparator: a new
matched release-setting control on source main `33a4735900153e742828227cc2171436ea09cd90`,
not historical release outcomes. Frozen inputs are read only.

Use dev seeds 1061–1090, differential drive, CPU inference, release algorithm
bindings and authored episode budgets. Select one medium cell per multi-cell
Figure 7.3 family; retain its five singleton Francis cells. Narrow doorway is
outside the figure and gets a separate feasibility diagnostic. Sweep radii
0.3/0.5/1.0 m and deterministic preferred speeds 0.65/1.0/1.3 m/s with ORCA,
social force, static hybrid v4, goal, and release PPO. Keep a spawn-coupled
legacy control, since a uniform 0.65 treatment is not necessarily identical.
Speed-cap comparison covers the complete release roster with the requested
cap swap. Record conditional near-field limits separately from global caps.

1. Read current related issue/PR threads and verify physical/planner defaults.
2. Generate new expanded configs after replacing their entire seed lists.
3. Run manipulation canaries before committing to the full sweep: physical
   radius, observation/planner geometry, preferred-speed binding, checkpoint
   identity, native execution. A failed binding stops the affected axis.
4. Execute eligible matched cells with explicit failed/missing/fallback counts.
5. Measure lane segregation and valid 1.2 m doorway crossings on the canonical
   released-substrate harness; preserve trajectories and wall violations.
6. Aggregate success and episode collision incidence per cell; use Kendall
   tau-b (ties retained), paired seed bootstrap CIs, and family rankings.
7. Preserve configs, compressed episode records, trajectories, summaries,
   source/environment identity, and a relative-path SHA256 manifest in Git.
8. Run all requested local CI equivalents, open a ready PR, inspect hosted
   checks, and classify any failure. No merge or self-approval.

Resume only completed identities with matching config/source hashes. Never
retry an unchanged failure automatically. Keep partial outputs and an explicit
missingness ledger. Full or partial data remain diagnostic; blocked cells
cannot be converted into successes. Findings reuse existing issues where possible.
