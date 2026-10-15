# Goal

Make the independently reviewed guard/escape diagnostic resolvable from a Git
commit, with portable inputs that reproduce the reported estimates and intervals.

# Scope and evidence

Own `docs/context/guard_escape_ablation/`, its context-index entry and catalog
entries. Preserve the reviewed execution at
`0035fba05c9b679e39f58ec286f5d57b96bb9a0a`, the 18-cell factorial design,
development seeds 1001–1030, and the original paired seed-block analysis.
This is diagnostic-only evidence custody, with no new simulation, planner or
active-config changes, release promotion, manuscript admission, or merge.

Sources: the original study contract, four producer episode files, acquisition
configs, original analyzer/CSV outputs, independently CONFIRMED review and
two-copy preservation receipts. Follow the compact tracked evidence pattern in
`../robsweep/` and the large-output rule in
`../../benchmark_artifact_publication.md`.

# Steps

1. Start an isolated worktree of a fresh origin/main clone; freeze source digests.
2. Project the four producer files into endpoint-complete per-episode inputs;
   retain source-file and row digests, configs, jobs, study spec and review.
3. Package the original numerical analysis with portable paths and validation.
4. Recompute all 48 rates and 36 contrasts; compare with reviewed outputs and
   assert exact one-decimal headline reproduction. Check SHA-256 manifest.
5. Inspect the complete diff, run proportional repository checks, commit, push
   and open an unmerged PR. Read back the remote head and pin artifact.
6. Write the exact downstream YAML entry and checks to the task report.

# Scientific contract and risks

Guard and escape on-minus-off main effects average over the other factor;
cells have equal weight and all four arms remain paired within seed blocks.
Use 10,000 percentile-bootstrap draws, NumPy default RNG seed 1001, and pointwise
95% intervals conditional on this fixed roster. Reject missing/duplicate slots,
fallback/degraded execution, invalid endpoints and non-development seeds.

No outcome-dependent exclusions or new stop rule. The documented 5 pp decision
threshold was absent from the original contract; numerical verification does
not authenticate its pre-result registration. The guard bundle is the scoped
intervention, not a mediation test or explanation of historical planner ranks.
Raw logs and the large full bundle stay in the existing preserved archive;
the compact packet independently suffices for numerical reproduction.

# Validation and recovery

The packet analyzer checks complete pairing and compares generated CSVs and
summary against frozen expected files. SHA256SUMS covers every packet payload
except itself. Keep original sources read-only and all scratch output outside
the packet. Resume from this plan and the exact Git/input hashes; do not reset
or clean the experiment lane. Final handoff records checks not run and the PR
head. No merge or automatic merge-readiness label is authorized.
