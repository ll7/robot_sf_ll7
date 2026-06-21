# Issue 3366 External MPC Staging Audit

Date: 2026-06-22

Related:

- Issue #3366: migrate `third_party/external_mpc_repos` clones to pinned external repo staging.
- Issue #3347 / PR #3367: first production external repo staging assistant.
- `docs/templates/external_repo_audit.md`
- `docs/context/issue_771_drmpscnav_assessment.md`

## Inventory

The current checkout and main checkout do not contain a local
`third_party/external_mpc_repos/` directory. The remaining migration work is therefore tracked
path and provenance cleanup, not preservation of a machine-local clone.

Tracked legacy references found on 2026-06-22:

- `robot_sf/baselines/sicnav.py`
- `configs/algos/sicnav_camera_ready.yaml`
- `robot_sf/benchmark/algorithm_metadata.py`
- `robot_sf/baselines/dr_mpc.py`
- `configs/algos/dr_mpc_exploratory.yaml`
- DR-MPC wrapper and map-runner tests that pass explicit legacy paths.

## SICNav Audit

- Repository name: `sicnav`
- Upstream URL: `https://github.com/sepsamavi/safe-interactive-crowdnav`
- Optional fork or private mirror URL: none
- Access date: 2026-06-21 for the pinned registry entry; GitHub metadata rechecked 2026-06-22.
- Pinned commit SHA: `c702fb8ac9ba6439ca61da7dde68b8524bbc6a1f`
- Related issue or PR: Issue #3366, Issue #3347, PR #3367
- Intended Robot SF use: reference checkout for SICNav wrapper and provenance work.
- Inclusion model: gitignored pinned clone

License and redistribution:

- Observed license: MIT License in GitHub repository metadata.
- License URL or file: upstream repository license metadata.
- Citation or notice requirement: preserve upstream license/notice in any future redistributed fork.
- Commercial, research-only, or non-redistribution limits: none identified from GitHub metadata.
- Public `ll7` fork decision: allowed by license after maintainer review; not required for this
  staging slice.
- License compatibility decision: compatible for local research/reference staging.
- Redistribution decision: public fork allowed by MIT after maintainer review; this registry stages
  from upstream until an `ll7` fork is created.
- Decision rationale: upstream license metadata is permissive and the clone remains local-only
  under `third_party/external_repos/sicnav`.

Source contract:

- Canonical upstream files that define the contract: `sicnav_diffusion/` package and checkpoint
  paths referenced by the Robot SF wrapper.
- Expected entrypoint, package, or executable: `sicnav_diffusion` or `sicnav` import surface with
  a supported policy constructor.
- Required runtime or environment: upstream solver/dependency stack; not installed by the staging
  helper.
- Expected inputs: Robot SF wrapper maps structured robot/human state into the upstream policy
  contract.
- Expected outputs: velocity or unicycle command projected into Robot SF `unicycle_vw` semantics.
- Known API, kinematics, checkpoint, or data assumptions: upstream checkpoint availability remains
  separate from clone staging.
- Verdict: integrate next as a pinned local reference only; benchmark eligibility still requires a
  dependency-backed wrapper smoke.

Staging and manifest:

- Staging path: `third_party/external_repos/sicnav/`
- Manifest path: `output/external_repos/manifests/sicnav.provenance.json`
- Checksum algorithm: aggregate SHA-256 over tracked file path, size, and file hash.
- Verification command:
  `uv run python scripts/tools/manage_external_repos.py --json stage sicnav --manifest-out output/external_repos/manifests/sicnav.provenance.json`

Smoke test convention:

- Staged-check command:
  `uv run pytest tests/baselines/test_external_mpc_wrappers.py -k sicnav_skip_without_external_repo -q`
- Robot SF wrapper or adapter smoke command:
  same skip-gated test for source checkout presence; dependency-backed wrapper execution remains a
  future proof gate.
- Skip-if-not-staged behavior: skip when `third_party/external_repos/sicnav` is absent.
- Fail-closed condition for missing dependencies: missing upstream package or unsupported policy
  surface is not benchmark evidence.
- Difference between source-harness proof and Robot SF wrapper proof: staging proves a pinned source
  checkout and import surface only; it does not prove the full Robot SF adapter or benchmark path.

## DR-MPC Audit

- Repository name: `dr_mpc`
- Upstream URL: `https://github.com/James-R-Han/DR-MPC`
- Optional fork or private mirror URL: none
- Access date: GitHub metadata rechecked 2026-06-22.
- Pinned commit SHA: none registered.
- Related issue or PR: Issue #3366, Issue #771
- Intended Robot SF use: source-side assessment anchor for residual MPC wrapper work.
- Inclusion model: reject / reference-only

License and redistribution:

- Observed license: no license metadata reported by GitHub.
- License URL or file: none verified.
- Citation or notice requirement: unknown.
- Commercial, research-only, or non-redistribution limits: unknown.
- Public `ll7` fork decision: blocked until an explicit license or maintainer approval exists.
- License compatibility decision: blocked for registered staging and redistribution.
- Redistribution decision: no redistribution.
- Decision rationale: absent license metadata is enough to reject public fork/staging in the
  license-gated pipeline.

Source contract:

- Canonical upstream files that define the contract: `scripts/online_continuous_task.py`,
  `scripts/configs`, and environment-specific dependency files from the prior #771 assessment.
- Expected entrypoint, package, or executable: source-side reproduction first; no registered Robot
  SF staging command.
- Required runtime or environment: Python/torch stack plus RVO2/pysteam-style dependencies.
- Expected inputs: robot pose/velocity/goal and human trajectory state stacks.
- Expected outputs: residual MPC command projected into Robot SF unicycle semantics.
- Known API, kinematics, checkpoint, or data assumptions: residual policy and dependency/runtime
  requirements remain unproven in this repository.
- Verdict: reject from pinned external staging for now; keep DR-MPC as reference-only until license
  and source-side reproduction are resolved.

Staging and manifest:

- Staging path: none.
- Manifest path: none.
- Verification command: none; no staging entry should be added while license metadata is absent.

Smoke test convention:

- Staged-check command: none.
- Robot SF wrapper or adapter smoke command: existing wrapper tests with fake modules remain
  source-independent contract tests only.
- Skip-if-not-staged behavior: not applicable until a staging entry is accepted.
- Fail-closed condition for missing dependencies: missing DR-MPC runtime remains an unavailable
  dependency, not fallback success.
- Difference between source-harness proof and Robot SF wrapper proof: DR-MPC still lacks a
  source-harness proof, so Robot SF wrapper proof is out of scope for this staging slice.

## Follow-Up

- Required follow-up issue: create one only if maintainers decide to pursue DR-MPC license
  clarification or source-side reproduction.
- Remaining provenance gaps: DR-MPC license and pinned commit.
- Remaining runtime or adapter gaps: SICNav dependency-backed wrapper smoke; DR-MPC source-side
  reproduction.
- Context note, registry, or benchmark report to update: this note and `docs/external_repo_setup.md`.

## Validation Checklist

- [x] Upstream URL, access date, and pinned commit SHA are recorded for SICNav.
- [x] License, fork, compatibility, and redistribution decisions are recorded for SICNav.
- [x] Inclusion model is explicit and matches the license decision.
- [x] Staging path is under `third_party/external_repos/sicnav/` and gitignored.
- [x] Manifest path is local-only under `output/external_repos/manifests/`.
- [x] Pinned SHA is reachable from the declared source when `stage sicnav` succeeds.
- [x] Validation command or skip-gated smoke test is recorded.
- [x] No restricted source code is committed.
- [x] DR-MPC is explicitly rejected from staging until license and reproduction gaps are resolved.
