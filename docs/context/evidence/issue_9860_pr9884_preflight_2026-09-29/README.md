# PR #9884 setup preflight evidence

This packet checks static spawn and footprint feasibility for 48 scenario identities
and seeds 111–140. The preflights perform no planner actions and report no planner
outcomes. The corrected matrix is a physical diagnostic v0.1 input assembled from
reviewed open input PR heads. It is not the final DOI-free 0.0.8 candidate, a
nominal evaluation, or release admission.

## Reproduce from a fresh PR branch clone

Run these commands from a directory outside the repository:

```sh
gh repo clone ll7/robot_sf_ll7 pr9884-repro -- --branch fix/issue-9860-grid-precision-r2 --single-branch
cd pr9884-repro
uv sync --all-extras
uv run python docs/context/evidence/issue_9860_pr9884_preflight_2026-09-29/reproduce.py --output-dir "$(mktemp -d)"
```

The script verifies all 79 input hashes in `input_closure.json`, reconstructs
the baseline checker from PR ancestor `4665cd13fc205761a4edb256a04d23d37ffbc235`,
applies the committed `goal_clearance_runtime.patch.gz` dependency to isolated
baseline and fixed worktrees, and runs the three preflights with four workers and default
0.10 m clearance, 20-step stationary respawn window, and 0.10 m grid. Expected
exit codes are 2, 0, and 2. It compares all JSON fields other than `runtime_s`
and requires exact Markdown report bytes. Its output includes
`reproduction_receipt.json` and the three fresh reports.

The fixed checker is commit `b626384207ca88422d3f0014d53785976d10fc0b`.
The fixed diagnostic source is pinned to PR commit
`03957faa332577008169d957b4fbe83ce66de0a9`; later merges into this PR do
not alter this historical comparison. All three source commits are reachable
in the PR history. The script verifies that the pinned checker bytes equal the
fixed checker commit. Its baseline worktree receives
the same checksummed input files from the PR branch. Both temporary worktrees
are removed after the run. The goal-clearance patch captures the opt-in runtime
dependency used for this diagnostic and is applied only in those temporary
worktrees; it does not activate that runtime on this PR branch or replace its
separate review under #9859.

## Packet contents

- `before.json.gz`, `after.json.gz`, and `unsafe_probe.json.gz` contain the full
  JSON reports. Their uncompressed SHA-256 values and timing-independent SHA-256
  values are in `paired.json` and `paired.md`.
- `before.md`, `after.md`, and `unsafe_probe.md` are the full readable reports.
- `pair_reports.py` checks every paired identity, invariant field, and guard cell;
  `reproduce.py` runs and compares the three preflights.
- `goal_clearance_runtime.patch.gz` contains the exact six-module diagnostic source
  overlay needed to reconstruct the original runs from this PR branch alone.
  Inspect it with `gzip -dc goal_clearance_runtime.patch.gz` from this directory.
- `input_closure.json` lists every input file and SHA-256.
- `seed_sets_v1.yaml` is an adjacent byte-for-byte copy of the existing tracked
  `configs/benchmarks/seed_sets_v1.yaml` used by both manifests.
- `checksums.sha256` covers all packet files except itself.

The executable corrected manifest and matrix are
`configs/benchmarks/releases/issue_9860_corrected_diagnostic_v1.yaml` and
`configs/scenarios/issue_9860_corrected_diagnostic_v1.yaml`. The guard manifest
is `configs/benchmarks/releases/issue_9860_unsafe_probe_diagnostic_v1.yaml`;
its v4 matrix and five required successor maps are also tracked on this branch.

The paired result is 75 blocked cells before and zero after, with no newly
blocked cells. The guard run keeps 71 unsafe nominal goals and all 30 historical
2 m doorway cells blocked. These are setup diagnostics only.
