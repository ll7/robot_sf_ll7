# Issue #7340 real-manifest feasibility-first diagnostic

Issue #7340 extends the fixture-only feasibility-first scenario-search protocol from #7315 to one
real, versioned Robot SF manifest. The implementation is a diagnostic harness, not a benchmark
campaign or a source-method replication.

## Claim boundary

The exact report claim boundary is:

> diagnostic-only real-manifest comparison: native execution and observed safety events; no
> simulator, planner, safety, benchmark, paper, or source-method claim

The wording intentionally says “no simulator claim” even though the runner records native episode
availability: local execution proves only that this runtime accepted and executed these inputs. It
does not establish simulator validity beyond the recorded rows, planner quality, safety, discovery
superiority, or transfer from the source method.

## Runtime contract

The canonical CLI is:

```bash
uv run python scripts/validation/run_feasibility_first_real_manifest.py \
  --config configs/benchmarks/issue_7340_feasibility_first_real_manifest_v1.yaml \
  --output output/issue_7340_real_manifest/report.json \
  --output-dir output/issue_7340_real_manifest
```

The manifest fixes a four-candidate pool budget, a two-candidate selection budget, sampling seed
`7340`, certification, horizon `60`, timestep `0.1`, and one worker. Each candidate retains its
control hash, source-order identity, scenario seed, config/input digests, materialized bundle,
episode record, four feasibility checks, rejection reasons, and runtime availability. Rejected,
unavailable, fallback, and degraded rows remain visible and cannot enter safety denominators.

The search selection features come from `scenario_cert.v1` route evidence plus stable candidate
identity diversity. Native episode outcomes are not used to select candidates. The seeded uniform
and hierarchical selections therefore share one fixed pool; the baseline is marked
`claim_eligible: false` until a separate, budget-matched campaign is approved.

## Observed bounded run

The 2026-08-17 local run used manifest digest
`74024ae340837ad874c952a43f8276f319bd4357b027a324376d4e1f9a9ae718`, station-map digest
`5f03421ac3d16d6008e92e00634fa9f3d0c896304ffd204691d79a3a115be156`, and native execution for all
four candidates. All candidates passed kinematic reachability, loader-backed behavioral
consistency, geometry/traffic, and simulator-validity availability checks. The observed episode
termination counts were 3 collisions and 1 timeout. The denominator was 4 feasible candidates;
these counts are not a safety rate, guarantee, or planner comparison.

The two seeded selections each had two candidates and each observed two collisions. The pool
contained one scenario family, so no diversity-across-family conclusion is possible. The result is
diagnostic-only and remains pending domain-aware approval on #7340.

## Route-binding finding

The existing #5303 `classic_group_crossing_medium_v2` probe is not a valid real-manifest starting
point: its `single_pedestrians` override targets a map with no single-pedestrian markers, so strict
candidate certification rejects it before simulator execution with
`scenario_loader_error: single_pedestrians overrides provided but map has no single pedestrians`.
This stale input contract is tracked in follow-up issue #7400.
The issue-scoped station template uses the map-backed `p2` marker and `route_mode: template` to
preserve the authored pedestrian path while varying candidate speed and start timing. In this mode
`pedestrian_delay_s` is recorded as provenance-only because the loader only accepts waits on
explicit trajectories.

## Next research direction

Before any evidence admission, resolve the stale #5303 template/map binding, add a strict
preflight for marker-backed pedestrian identities, and repeat the protocol across multiple
approved scenario families and independent fixed seed manifests. The next comparison should match
candidate and execution budgets between feasibility-first risk feedback, the existing adversarial
random sampler, and a declared uniform baseline. Report discovery yield, valid-scenario rate,
within- and across-family diversity, observed event severity, and unavailable/degraded rows with
the same fail-closed denominators.

The generated report and episode bundles stay in worktree-local `output/`; this note preserves only
the command, provenance, observed bounded result, and limitations.
