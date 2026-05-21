# Thursday Development Review - 2026-05-21

Date: 2026-05-21

Current baseline:

- Branch checked: `main`
- Latest commit checked: `9b2192d4 feat: expose parser map capability metadata (#1419)`
- Review window: primarily 2026-05-14 through 2026-05-21, with a broader 2026-05-07 through
  2026-05-21 scan for context.

This note is a course-correction review after a high-autonomy merge period. It summarizes the
current development direction, the main new features, and the recommended next direction. It is a
planning artifact, not a benchmark result or paper-facing claim.

## Evidence Sources

Local evidence was generated from the current checkout on 2026-05-21:

```bash
git fetch origin main
git merge --ff-only origin/main
git log --since="7 days ago" --date=short --pretty=format:"%h%x09%ad%x09%s"
git log --since="14 days ago" --date=short --pretty=format:"%h%x09%ad%x09%s"
gh pr list --state merged --limit 300 --json number,title,mergedAt,author,labels,url
gh issue list --state all --limit 300 --json number,title,state,createdAt,updatedAt,closedAt,labels,url
gh pr list --state open --limit 100 --json number,title,isDraft,createdAt,updatedAt,labels,url
gh issue list --state open --limit 200 --json number,title,labels,createdAt,updatedAt,url
```

Two read-only Spark sidecars were used through the `ai-delegation-routing` workflow:

- one feature/theme taxonomy pass,
- one adversarial course-correction critique.

Their outputs were treated as unverified leads and checked against local commits, PR lists, and
context notes before being incorporated here.

## Activity Snapshot

The repository moved very quickly.

| Window | Merged PRs | Touched issues | Current open PRs | Current open issues |
| --- | ---: | ---: | ---: | ---: |
| 2026-05-14 to 2026-05-21 | 116 | n/a | 6 | 18 |
| 2026-05-07 to 2026-05-21 | 187 | 195 | 6 | 18 |

Merged PR title prefixes in the last seven days:

| Prefix | Count |
| --- | ---: |
| `docs` | 37 |
| `feat` | 30 |
| `fix` | 8 |
| `test` | 7 |
| `perf` | 6 |
| `refactor` | 6 |
| `ci` | 5 |
| `workflow` | 5 |
| other / uncategorized | 12 |

Current open PRs at review time:

- PR #1425 `Refactor skills documentation for clarity and conciseness`
- PR #1424 `docs: add Thursday development course review`
- PR #1420 `feat: add capability-aware map catalog`
- PR #1418 `benchmark: add broader AMV baseline preflight`
- PR #1412 `feat: wire predictive obstacle pipeline`
- PR #1401 `fix: reconcile campaign SNQI rollups`

## Current Direction

The current repository direction is coherent but overloaded. The dominant theme is not a single new
planner; it is building a stronger benchmark operating system around planners:

1. stronger benchmark execution and reproducibility infrastructure,
2. stricter map/scenario capability contracts,
3. more explicit learned-policy admission and rejection gates,
4. new local-planner surfaces and observation adapters,
5. manual-control, replay, CARLA, and diagnostics tooling for failure reproduction,
6. benchmark semantics hardening around collision, timeout, SNQI, and fallback behavior.

That direction is mostly correct. The course risk is that contract surfaces, candidate assessments,
and planner entry points are now arriving faster than full campaign evidence and semantic closure.
The next cycle should therefore consolidate and prove rather than expand.

## New Capabilities

### Benchmark and Campaign Infrastructure

Recent work made benchmark execution more reproducible and less ad hoc:

- Generic camera-ready SLURM launcher: `SLURM/Auxme/camera_ready_benchmark.sl`.
- Feature-extractor SLURM output canonicalization.
- Docker repro smoke workflow and pinned/timeout-hardened CI paths.
- Runtime-requirement documentation and preflight checks.
- Faster JSONL append throughput and cached occupancy-grid layers.
- Scenario task bundles and question-first experiment registry.

This is valuable because many next steps are campaign-sized. The repo is getting closer to a state
where benchmark evidence can be launched, repeated, and audited from stable scripts rather than
session memory.

### Benchmark Semantics and Fail-Closed Behavior

Several merged changes improved benchmark correctness:

- malformed benchmark JSONL now fails closed,
- policy step timeout boundaries are enforced,
- mismatched simulator action counts are rejected,
- exact collision flags are preserved in map-runner metrics,
- TEB corridor-deadlock runtime/stall evidence was added,
- advisory typecheck policy and docs-proof consistency checks were clarified.

This is directionally strong. It also means current aggregate claims should be treated carefully
until the remaining episode-integrity/SNQI reconciliation work closes.

### Learned-Policy Screening and Eligibility

The policy-search lane became much more disciplined:

- learned local-navigation candidate screening was added,
- NavDP/NoMaD, Tentabot, SAGE, GenSafeNav, CrowdNav-family, NeuPAN, DRL-VO, and TEB-style candidates
  now have bounded assessment notes or registry entries,
- a learned-policy eligibility checklist was added,
- `scripts/validation/check_learned_policy_eligibility.py` validates candidate metadata,
- a reject/monitor registry records why unsuitable candidates should not be repeatedly revisited.

This is a good shift. It reduces novelty chasing and forces candidates through observation/action,
provenance, leakage, action-interface, logging, and fail-closed gates.

### Planner and Observation Surfaces

There were real control-stack additions:

- `robot_sf/sensor/social_graph_observation.py` adds deterministic graph-style pedestrian/static
  obstacle observations for future CrowdNav/SoNIC/GenSafeNav-style adapters.
- `robot_sf/planner/guarded_ppo.py` now has an ORCA-prior residual mode surface.
- Safety-shield and graded-observation-level contracts were added.
- SocNavBench personal-space cost now accounts for velocity.
- TEB, command-lattice, and motion-primitive candidate paths were assessed against corridor cases.

The important distinction is that most of these are enabling surfaces, not final promoted planners.
The ORCA-residual direction is the most promising learned-policy path, but it still needs training
lineage and campaign evidence before it should become a benchmark comparison row.

### Map and Scenario Capability Work

The map system is moving toward capability-aware cataloging:

- parser-derived capability metadata is exposed through
  `robot_sf/maps/verification/svg_inspection.py`,
- capability-aware map catalog design exists,
- follow-up issues now separate schema/sync checking, runtime enforcement, and generated cache
  evaluation,
- map-definition invariant work was excluded after PR #1362 closed unmerged,
- capability-aware catalog work is still open in PR #1420.

This is the right direction. Map validity should be profile-specific, not one universal binary
valid/invalid decision. The risk is that the design/metadata/enforcement pieces are still split
across open PRs and issues.

### CARLA, Manual Control, Replay, and Diagnostics

Recent additions also improved debugging and external-world integration:

- CARLA Docker runtime preflight,
- CARLA live oracle replay,
- CARLA static geometry proxy replay,
- manual-control pygame runner and rewind semantics,
- render frame contact sheets,
- per-GPU telemetry samples,
- trajectory debug export.

These are not just convenience features. They improve failure reproduction and make it easier to
connect benchmark failures with visual or external-runtime evidence.

## Current Open Backlog Shape

The current open issue set is small enough to reason about. The most important groups are:

- Benchmark semantics / rollups:
  - Issue #1398 reconcile episode-integrity flags with collision and SNQI rollups.
- Map capability contract:
  - Issue #1413 capability schema and sync checker,
  - Issue #1415 capability-aware scenario map resolution,
  - Issue #1416 generated converted-map cache evaluation.
- Campaign execution:
  - Issue #1395 learned risk model v1,
  - Issue #1396 shielded PPO repair,
  - Issue #1397 oracle imitation dataset,
  - Issue #1358 bounded ORCA-residual learned local policy,
  - Issue #1344, Issue #1353, Issue #1354 AMV protocol/campaign issues.
- Planner research spikes:
  - Issue #1387 Tentabot-style value scorer,
  - Issue #1318 TEB corridor-deadlock evaluation.
- External data blockers:
  - Issue #1126 SDD scenario set,
  - Issue #1134 SocNavBench ETH map.

This is a useful backlog, but it should be sequenced. Several open PRs and issues touch the same
contracts, so the next week should reduce concurrency before adding new lanes.

## Course Assessment

I would adjust course in one way:

**Pause expansion of new planner families and focus on consolidation/proof for one week.**

The repository is not drifting randomly. It is moving toward a credible benchmark platform with
better contracts, stronger evidence hygiene, and more candidate coverage. The risk is not wrong
direction; it is too many partially connected surfaces being added by autonomous PRs before the
core evidence pipeline catches up.

The main correction should be:

1. close semantic correctness first,
2. close map capability contract second,
3. run one or two high-value benchmark/training campaigns third,
4. only then resume candidate intake.

## Recommendation To Issue Mapping

The course recommendation maps to mostly existing work. Only the sequencing/governance decision
needed a new issue.

| Recommendation | Existing or new owner | Status / role |
| --- | --- | --- |
| Close benchmark integrity before relying on new aggregate claims | Issue #1398 and PR #1401 | Existing owner. Treat as the first consolidation lane because collision, timeout, and SNQI rollup semantics affect every benchmark comparison. |
| Finish the map capability contract before expanding benchmark surfaces | Issues #1413, #1415, #1416 and PR #1420 | Existing owner set. These cover schema/sync checking, runtime resolution, cache policy, and the active map-catalog implementation PR. |
| Run one evidence campaign to completion instead of several in parallel | Primary recommendation: Issue #1358. Alternatives: Issues #1395, #1396, #1397, #1353, #1354, merged PR #1399, and PR #1418 | Existing owner set. Pick exactly one first campaign lane after benchmark integrity and map capability blockers are settled. |
| Keep broad candidate intake frozen except already-open source/spike work | Issue #1387 and merged PR #1404 for Tentabot-style value scoring | Existing owner. GenSafeNav and CrowdNav HEIGHT source-harness checks are already recorded in closed Issues #1393 and #1394; reopen only if missing upstream assets become available. |
| Make the consolidation-week sequencing decision explicit | Issue #1423 | New owner created from this review. This is a coordination issue, not a code implementation issue. |

No new issue is needed for PR #1362. It is closed and should not be treated as part of the active
map-catalog merge chain; the active chain is Issues #1413, #1415, #1416 and PR #1420.

### Priority 1: Close Benchmark Integrity

Finish Issue #1398 and PR #1401 before relying on new aggregate claims.

Reason: many recent changes touch collision metrics, timeout handling, SNQI, exact collision flags,
and campaign rollups. If these semantics are inconsistent, planner comparisons become noisy or
misleading.

Expected proof:

- targeted tests for collision/SNQI rollup reconciliation,
- at least one small fixture or JSONL sample proving the intended aggregate behavior,
- clear context note explaining whether exact collision flags, sampled metrics, and SNQI rollups
  are authoritative in conflict cases.

### Priority 2: Finish the Map Capability Contract

Finish the map-catalog chain before more benchmark expansion:

- Issue #1413 schema and sync checker,
- Issue #1415 runtime/scenario resolver enforcement,
- Issue #1416 cache policy decision,
- PR #1420 review/merge order.

Reason: planner evaluation quality depends heavily on whether a scenario map is actually valid for
the required capability. Capability-aware maps are a strong idea, but incomplete enforcement could
create hidden skips or accidental mismatches.

Expected proof:

- catalog sync checker,
- capability metadata round-trip,
- fail-closed scenario resolution for unsupported map/profile combinations,
- one smoke path showing a benchmark entrypoint exercises the resolver.

### Priority 3: Run One Evidence Campaign, Not Three

Pick one campaign lane and drive it to a complete artifact:

- strongest choice: Issue #1358 bounded ORCA-residual learned local policy,
- alternative if training infrastructure is the bottleneck: Issue #1395 learned risk model v1,
- alternative if benchmark breadth is the priority: Issue #1399/#1418 paired AMV baseline/preflight.

Reason: surface work for learned policies is now good enough to start falsifying. Another candidate
assessment is less valuable than one completed evidence loop.

Expected proof:

- fixed config,
- seed manifest,
- JSONL/summary artifacts,
- context note with pass/fail decision,
- explicit comparison against ORCA/PPO/current hybrid-rule where applicable.

### Priority 4: Keep Candidate Intake Frozen Except for Already-Open Source Harnesses

Do not start another broad external-policy search yet. If anything continues, keep it to source-side
harness closure for already-open candidates:

- GenSafeNav source harness,
- CrowdNav HEIGHT source harness,
- Tentabot-style value scorer spike.

Reason: the candidate registry is now rich enough. More intake will not help until the benchmark can
run and classify the first few candidates cleanly.

## What I Would Not Do Next

- Do not add another diffusion/world-model/visual-navigation candidate issue unless a clean
  `observation_t -> action_t` local-planner contract is proven.
- Do not promote ORCA-residual guarded PPO from smoke evidence alone.
- Do not interpret source-harness blockers as benchmark comparison results.
- Do not expand the AMV paper-facing benchmark matrix until Issue #1398 and the map capability contract
  are stable.
- Do not rely on the stale open-issue audit stash from 2026-05-20 without refreshing it; current
  open issue counts and priorities have already changed.

## Proposed Direction Statement

For the next development cycle, Robot SF should position itself as:

> A reproducible local-navigation benchmark platform that admits planners through explicit
> observation/action/provenance contracts, validates maps by capability, and prioritizes
> fail-closed evidence over broad planner coverage.

That direction is stronger than trying to become a broad robotics integration zoo. The most valuable
near-term work is not more candidate breadth; it is making the current candidate funnel produce one
or two decisive, repeatable results.

## Delegated Perspective Incorporated

The feature-taxonomy sidecar independently identified five dominant themes:

- benchmark/research infrastructure hardening,
- map-capability-aware scenario selection,
- learned-policy admission controls,
- planner/observation capability expansion,
- manual-control, diagnostics, and CARLA integration.

The adversarial sidecar independently recommended a consolidation sprint ordered around:

- Issue #1398 benchmark integrity,
- Issue #1413/#1415/#1416 map capability contract,
- one evidence campaign from the training/benchmark backlog.

I agree with that ordering after local evidence checks.

## Follow-Up Decision Needed

The key maintainer decision is whether to declare a short "consolidation week" policy:

- no new broad planner-family intake,
- no benchmark-matrix expansion until Issue #1398 and the map capability contract close,
- one selected evidence campaign gets priority compute and review bandwidth.

My recommendation is yes.
