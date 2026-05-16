# Issue #1255 Open-Issue Dependency Graph

Date: 2026-05-16

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1255>

## Goal

Normalize the currently open backlog so maintainers and agents can filter ready work without
re-reading every issue body. This pass uses the repository's existing metadata surfaces instead of
inventing new labels or priority semantics:

- existing GitHub labels, especially `blocked` and `slurm`,
- Project #5 `Status` values,
- short issue comments for blocker rationale where the blocker was not already visible,
- this context note as the durable handoff surface.

## Current Lanes

### Covered By Open PR

These issues already have open implementation or workflow PRs and remain `In progress` in Project
#5:

| Issue | PR | Notes |
| --- | --- | --- |
| #1257 | #1260 | Artifact evidence vocabulary. |
| #1256 | #1261 | Backlog issue forms. |
| #1244 | #1253 | Optional Rerun trajectory debug export. |
| #1243 | #1251 | Question-first experiment registry. |
| #1242 | #1252 | Planner observation/action contracts. |
| #1238 | #1258 | Parquet benchmark analytics export. |
| #1235 | #1259 | `scenario_contract.v1` governance schema. |
| #1234 | #1254 | Legacy environment factory kwargs retirement. |
| #1233 | #1249 | Immutable workflow action pins. |
| #1232 | #1248 | CI job timeouts. |
| #1231 | #1250 | Durable paper handoff fixture. |
| #1222 | #1225 | Seed-determinism reset contract. |
| #1221 | #1224 | Required `jsonschema` dependency for visual schema tests. |
| #1219 | #1228 | Manual-control Pygame MVP runner. |
| #1218 | #1229 | Predictive map obstacle features. |
| #1217 | #1220 | Render encode placeholder skip retirement. |
| #1213 | #1223 | BC pretraining device policy. |
| #1191 | #1227 / #1226 | Duplicate ml-intern workflow extraction PRs exist; prefer one final PR. |
| #1119 | #1230 | Docker benchmark reproduction smoke. |

### Ready

These issues have no current hard blocker recorded in this audit and were moved from `Tracked` to
`Ready` in Project #5:

- #1236: optimizer-backed adversarial search sampler pilot.
- #1237: adversarial failure archives with minimization and clustering.
- #1240: scenario coverage entropy for benchmark diversity.
- #1241: proxemic social-acceptability metric pilot.

### Hold

These issues are intentionally not the next local-agent work queue. They have either a visible
`blocked` label already or received one in this pass, and their Project #5 status is `Hold`:

| Issue | Hold reason |
| --- | --- |
| #1245 | Wait for #1260 / #1257 so BenchmarkClaim categories align with the shared artifact-evidence vocabulary. |
| #1246 | Wait for #1252 / #1242 so graded observation levels build on merged planner contracts. |
| #1247 | Wait for #1252 / #1242 and #1246 before defining prediction-aware shield provenance and metrics. |
| #1239 | Existing human-model transfer robustness blocker; keep blocked until transfer prerequisites are concrete. |
| #1179 | CARLA runtime blocker; needs pinned Docker/NVIDIA CARLA 0.9.16 harness on a capable host. |
| #1169 | CARLA live T1 replay depends on #1179 and #1111. |
| #1167 | Predictive planner same-seed comparison remains blocked on prerequisite predictive-feature work. |
| #1134 | External SocNavBench ETH asset staging blocker. |
| #1126 | External SDD-derived dataset curation blocker. |
| #1111 | CARLA setup-only T1 smoke depends on #1179. |

The CARLA parent epic #872 and manual-control parent epic #1151 remain `Tracked`, not `Hold`, because
their bodies already enumerate children and they should stay visible as umbrella work rather than
single executable tasks.

#1108 remains `In progress` because it is an active SLURM/training execution issue, not a local
implementation ticket for this machine.

## Metadata Changes Applied

- Added `blocked` label and unblock-condition comments to #1245, #1246, and #1247.
- Set Project #5 status for #1255 to `In progress`.
- Set Project #5 status for #1236, #1237, #1240, and #1241 to `Ready`.
- Set Project #5 status for #1245, #1246, #1247, #1239, #1179, #1169, #1167, #1134, #1126, and
  #1111 to `Hold`.
- Did not change Project #5 priority fields or derived score inputs.

## Validation

Metadata proof:

```bash
gh issue list --repo ll7/robot_sf_ll7 --state open --limit 140 --json number,title,labels,url
gh project item-list 5 --owner ll7 --limit 700 --format json
```

Observed status check after the writes:

- #1255: `In progress`.
- #1236, #1237, #1240, #1241: `Ready`.
- #1245, #1246, #1247, #1239, #1179, #1169, #1167, #1134, #1126, #1111: `Hold`.
- #1245, #1246, and #1247 include `blocked` after the label pass.

Tracked-file validation:

```bash
uv run --active python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_1255_open_issue_dependency_graph.md \
  --path docs/context/README.md
git diff --check
```

Both commands passed on 2026-05-16 before the branch was committed.

## Follow-Up Boundary

After the open PRs above merge, re-run this audit before selecting the next batch. In particular:

- #1246 should become ready after #1242 lands.
- #1245 should become ready after #1257 lands.
- #1247 should remain on hold until both #1242 and #1246 have landed.
- CARLA and external-data issues should stay on hold on this host unless their runtime or artifact
  prerequisites become available.
