# Issue #3481 Closure Audit

This audit maps the [issue #3481](https://github.com/ll7/robot_sf_ll7/issues/3481) Definition of
Done to merged pull-request evidence and records the closure boundary after PR #4634 added the
opt-in HSFM body-orientation alignment-torque model — the maintainer-named last remaining code
prerequisite (2026-07-05 gate comment).

Headed Social Force Model (HSFM), Field-of-View (FoV) attenuation, and Time-to-Collision (TTC)
predictive repulsion are the three force-law extensions the issue proposed. All three now exist as
opt-in pedestrian-model selectors that leave the default `social_force_default` path unchanged.

## Claim Boundary

- Evidence status: CPU-local diagnostic / prototype evidence only.
- Scope: the issue's implementable force-law prototype, config surface, fixtures, and design notes.
- Closure call: the Definition-of-Done items that do not require a benchmark campaign are met by the
  merged PR set below. The remaining seed-controlled benchmark campaign and any evidence-tier
  upgrade are explicitly out of scope for this CPU-only closure audit and are not promoted here.
- Out of scope: full benchmark campaign run, Slurm/GPU submission, calibrated behavioral realism,
  default-model replacement, planner-ranking or paper/dissertation claim edits.

The issue is labeled `evidence:proposal`; its own Definition of Done requires that
`evidence_tier` stay honest ("idea"/diagnostic) with no calibrated-realism or paper-facing claim
before durable benchmark evidence. The merged slices satisfy that constraint, so honoring the
claim boundary is part of closure, not a gap.

## Acceptance Mapping

| Definition-of-Done criterion | Status | Evidence |
| --- | --- | --- |
| Anisotropic FoV weight prototyped on the vectorized force evaluation (in-cone full strength, ~0.1 behind heading). | Met | `hsfm_anisotropic_fov_v1` selector added by PR #4258; pairwise-isolated and vectorized by PR #4297; consumed per-pair at the simulator seam by PR #4352; the O(N^2) contribution loop vectorized against the PySocialForce scalar kernel by PR #4354. Tests in `tests/sim/test_hsfm_fov_pairwise_isolation.py`. |
| HSFM heading state + alignment-torque term added, decoupling heading `phi_i` from instantaneous velocity `v_i`. | Met | Heading state introduced by the Phase 1 `hsfm_total_force_v1` selector (PR #4144). Damped second-order alignment torque added by PR #4634 as `hsfm_alignment_torque_v1`, replacing the instantaneous heading snap; a regression test pins that `hsfm_total_force_v1` still snaps. Note `docs/context/issue_3481_hsfm_alignment_torque.md`; tests `tests/sim/test_hsfm_alignment_torque_model.py`. |
| Opt-in TTC-scaled predictive repulsion term available as an alternative to Euclidean-distance repulsion. | Met | `hsfm_ttc_predictive_v1` selector, bounded pedestrian-pedestrian TTC repulsion, and validated `ttc_predictive_force` config added by PR #4202. Pure helpers and tests in `tests/sim/test_ttc_predictive_pedestrian_model.py`. |
| Scoping/design note describing the force-law changes and their modeling assumptions. | Met | `docs/context/issue_3481_hsfm_ttc_predictive_forces.md` (TTC/FoV force law) and `docs/context/issue_3481_hsfm_alignment_torque.md` (alignment torque) record selector keys, equations, parameters, default-preservation, and diagnostic evidence tier. |
| Fixtures show (i) narrow-passage lateral-sliding proxy and (ii) bottleneck freeze/deadlock proxy across selector variants. | Met (diagnostic tier) | `narrow_passage_lateral_sliding` and `bottleneck_freeze_deadlock` geometric fixtures in `robot_sf/benchmark/pedestrian_model_fixture_diagnostics.py` emit mean-max lateral displacement and consecutive interaction-zone slow-step proxies with local diagnostic threshold checks (`thresholds_applied=True` = local assertions, not benchmark gates). Added by PR #4482 (shared-throat harness) and PR #4593 (geometric fixtures). Tests `tests/benchmark/test_pedestrian_model_fixture_diagnostics.py`. |
| HSFM/TTC parameters versioned; `evidence_tier` kept honest — no calibrated-realism or paper-facing claim before durable evidence. | Met | Every selector ships a frozen, fail-closed config dataclass (`ttc_predictive_force`, anisotropic FoV, `AlignmentTorqueConfig`) wired through `SimulationSettings`. All merged notes and reports carry a diagnostic/prototype boundary; no planner-ranking, benchmark-strength, or paper claim was promoted. |

## Evidence Trail

| PR | Merged | Contribution | Closure relevance |
| --- | --- | --- | --- |
| PR 4144 | 2026-07-02 | Phase 1 runtime selector: `SimulationSettings.pedestrian_model` with `social_force_default` and `hsfm_total_force_v1`, HSFM total-force step path. | Establishes the opt-in selector seam and HSFM heading state all later slices extend. |
| PR 4202 | 2026-07-03 | Opt-in `hsfm_ttc_predictive_v1` selector + bounded pedestrian-pedestrian TTC repulsion + validated config. | Satisfies the TTC-predictive force-law criterion. |
| PR 4258 | 2026-07-03 | `hsfm_anisotropic_fov_v1` FoV-attenuation selector/config/scenario surface + deterministic fixtures. | Introduces the anisotropic FoV force-law prototype. |
| PR 4297 | 2026-07-03 | Pairwise-isolated FoV attenuation helper + behavior-preserving vectorized TTC weight path. | Closes both maintainer-flagged FoV blockers at the pure-math layer. |
| PR 4352 | 2026-07-04 | Runtime consumes per-pair pedestrian-pedestrian FoV forces at the simulator seam (replaces coarse `np.min` whole-force scaling). | Wires FoV isolation into runtime with a fail-closed social-force guard. |
| PR 4354 | 2026-07-04 | Vectorizes the O(N^2) `pairwise_social_force_contributions` loop, pinned pair-by-pair to the PySocialForce scalar kernel (rtol=1e-9). | Removes the last named runtime O(N^2) loop before benchmark-scale use. |
| PR 4482 | 2026-07-04 | Shared-throat diagnostic fixture harness over selector variants (descriptive-only, fail-closed config). | Precursor harness for the geometric fixtures. |
| PR 4593 | 2026-07-05 | `narrow_passage_lateral_sliding` and `bottleneck_freeze_deadlock` geometric fixtures with local threshold emission. | Satisfies the narrow-passage / bottleneck fixture criterion at diagnostic tier. |
| PR 4634 | 2026-07-06 | Opt-in `hsfm_alignment_torque_v1` body-orientation alignment torque decoupling heading from velocity. | Closes the maintainer-named last remaining code prerequisite (heading/velocity decoupling). |

## Closure Decision

Issue #3481 can close on the merged PR set above. Every Definition-of-Done item that is
implementable without a benchmark campaign — anisotropic FoV weight, HSFM heading state plus
alignment torque, opt-in TTC-predictive repulsion, design notes, geometric fixtures, and versioned
fail-closed parameters at an honest diagnostic tier — is satisfied by merged code, tests, configs,
and notes. The only remainder is a seed-controlled benchmark campaign that would upgrade the
evidence tier and support calibrated-realism / planner-ranking claims; the issue puts that work out
of scope ("Replacing the default pedestrian model without a benchmarked, seed-controlled
comparison" and "Paper-facing realism conclusions before durable in-repo evidence"). Under the
maintainer COMPLETE-FIRST directive an issue whose only remainder is a campaign run counts as
complete for CPU-only work. Any future campaign should be tracked separately as a claim-promotion
task with predeclared scope, artifact provenance, and benchmark claim boundaries.

## Validation

```bash
uv run pytest tests/sim/test_hsfm_total_force_model.py \
  tests/sim/test_ttc_predictive_pedestrian_model.py \
  tests/sim/test_hsfm_fov_pairwise_isolation.py \
  tests/sim/test_hsfm_alignment_torque_model.py \
  tests/benchmark/test_pedestrian_model_fixture_diagnostics.py -q
```

Result on `origin/main` at audit time: `71 passed`. This confirms the merged acceptance code for
all four opt-in selectors and the geometric fixtures executes cleanly on CPU.

## Artifacts Consulted

- `robot_sf/sim/pedestrian_model_variants.py`
- `robot_sf/benchmark/pedestrian_model_fixture_diagnostics.py`
- `docs/context/issue_3481_hsfm_ttc_predictive_forces.md`
- `docs/context/issue_3481_hsfm_alignment_torque.md`
- `configs/research/hsfm_ttc_predictive_forces_issue_3481.yaml`
- `tests/sim/test_hsfm_total_force_model.py`
- `tests/sim/test_ttc_predictive_pedestrian_model.py`
- `tests/sim/test_hsfm_fov_pairwise_isolation.py`
- `tests/sim/test_hsfm_alignment_torque_model.py`
- `tests/benchmark/test_pedestrian_model_fixture_diagnostics.py`
