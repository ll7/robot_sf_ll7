# FREEZEPREP freeze-candidate audit (2026-10-03)

AI-GENERATED / NEEDS-REVIEW. Candidate source: `c979e0337da4ad053d59a76225fbb9154140ee73`.
This prepares an orchestrator decision; it does not name a freeze or admit release evidence.

## Scope and snapshot

`gh pr list -R ll7/robot_sf_ll7 --state open --limit 200 --json number,title,headRefName,baseRefName,body,url,files`
returned 37 PRs. Every head was fetched; contribution diffs were inspected as
`git diff origin/main...origin/audit-<number>`. GitHub's cumulative files list
on old/reanchored branches includes already-merged ancestry, so it alone is not
evidence of a new runtime change. Recheck new heads/open PRs when naming the
freeze; these classifications are bound to this snapshot, not future PR revisions.

The D-083 authored roster is prediction_planner, goal, social_force, orca, ppo,
socnav_sampling, sacadrl, scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4,
scenario_adaptive_hybrid_orca_v2_collision_guard_v4,
hybrid_rule_v4_fast_progress_static_escape,
hybrid_rule_v4_fast_progress_static_escape_continuous, guarded_ppo, predictive_mppi, risk_dwa. It resolves 14×48×30=20,160
cells; the fixed doorway companion is 14×3×30=1,260 cells/H400.

**Freeze blocked — orchestrator ruling of 2026-10-03:**
[#10112](https://github.com/ll7/robot_sf_ll7/issues/10112) **blocks the freeze**.
The D-083 template lacks an SNQI-v2 spec; the smoke contract and mint ordering
also require fixes to the release source that the resolved identity binds.
SNQI v2 is a reported 0.0.8 number, so these are pre-freeze corrections.
[#10110](https://github.com/ll7/robot_sf_ll7/issues/10110) **must land before the
freeze** too: its mint/preflight release-notes gate belongs to that bound source.
This ruling supersedes this audit's earlier mint/publication-only disposition.
The candidate's historical test/dry-identity proof below remains evidence for
that source, not admission of a final freeze lacking these fixes.

Other runtime opt-ins and diagnostic changes remain classified below; the ruling
does not grant blanket permission to merge them into the final freeze.

## Every open PR

| PR | Title | Touches selected runtime/release surface? | Freeze disposition and evidence | Code/input witness |
|---|---|---|---|---|
| [#10104](https://github.com/ll7/robot_sf_ll7/pull/10104) | PEDCONTACT: opt-in pedestrian contact and bounded wall force | Yes, shared physics; opt-in | No: contact/wall law is disabled on the release path; next-release validation/adoption remains rejected. | `fast-pysf/pysocialforce/{contact,forces,simulator}.py; configuration selector` |
| [#10100](https://github.com/ll7/robot_sf_ll7/pull/10100) | fix(planner): repair bicycle steering and add opt-in T60 proxies | Yes, shared episode/action plumbing | No: changes apply to bicycle_drive; all 14 release arms run differential_drive. T60 configs are new opt-ins. | `map_runner_episode.py bicycle_drive branch; configs/robots/t60_*` |
| [#10099](https://github.com/ll7/robot_sf_ll7/pull/10099) | fix(planner): opt in to swept exclusion and admissible speed candidates (#10092) | Yes, frozen-hybrid implementation/observations | No measured P1: physical_static_exclusion_enabled, platform_speed_candidates_enabled and goal_next_validity_enabled default false and are absent from frozen release configs. | `hybrid_rule_local_planner.py; unified_config.py opt-in flags` |
| [#10094](https://github.com/ll7/robot_sf_ll7/pull/10094) | fix(validation): repair pedestrian starts and report bounded CALFIT rerun | No selected runtime | No: pedestrian validation initial states/acceptance only; no release scenario change. | `robot_sf/research/pedestrian_{initial_state,acceptance}.py` |
| [#10075](https://github.com/ll7/robot_sf_ll7/pull/10075) | feat: finish source-protocol pedestrian validation and radius plumbing | Yes, shared simulation/runner plumbing | No: opt-in body radius/profile/validation, released defaults retained; no adopted 0.0.8 calibration. | `sim_config.py; scenario_loader.py; pedestrian_validation.py` |
| [#10073](https://github.com/ll7/robot_sf_ll7/pull/10073) | feat: compare obstacle-force profiles for 0.0.9 (#10061) | Yes, shared simulation/config plumbing | No: absent selector returns legacy_v1; calibrated_v2/gradient_v3 are opt-in and neither is adopted for 0.0.8. | `obstacle_force_profile.py resolve/apply; D-055/D-057` |
| [#10003](https://github.com/ll7/robot_sf_ll7/pull/10003) | feat(training): align four PPO retrains with the release action contract | Yes, PPO and native-action plumbing | No demonstrated P1: shared velocity-delta/target-to-acceleration arithmetic is refactored with the same clipping; training environment opt-in defaults acceleration. New checkpoints/profiles are not selected D-053 model. | `baselines/ppo.py velocity_delta; action_adapters.py; registry D-053` |
| [#9993](https://github.com/ll7/robot_sf_ll7/pull/9993) | fix(dev): classify state:ready-to-submit and state:reviewing (#9938) | No | No: developer issue-state taxonomy, no runtime/release config change. | `scripts/dev/lifecycle state consumers` |
| [#9984](https://github.com/ll7/robot_sf_ll7/pull/9984) | Fix #9982: add opt-in bounded goal input for GA3C-CADRL | Yes, SA-CADRL | No: optional distance cap; default release network goal input unchanged. | `socnav_sacadrl.py/socnav_base.py optional selector` |
| [#9939](https://github.com/ll7/robot_sf_ll7/pull/9939) | docs(context): bind #9764 evidence note to final paired diagnostic (#9910) | No | No: documentation binds final existing kernel diagnostic; measured numbers unchanged. | `docs/context/issue_9764_social_force_kernel_versioning.md` |
| [#9937](https://github.com/ll7/robot_sf_ll7/pull/9937) | deps: bump the actions group with 3 updates | No numeric runtime | No: CodeQL actions update. | `workflow action pins` |
| [#9934](https://github.com/ll7/robot_sf_ll7/pull/9934) | deps: bump mutmut from 3.7.0 to 3.8.0 | No numeric runtime | No: mutmut developer dependency. | `pyproject/lock developer tool` |
| [#9933](https://github.com/ll7/robot_sf_ll7/pull/9933) | deps: bump the developer-tooling group with 2 updates | No numeric runtime | No: Ruff/Pylint developer dependency updates. | `pyproject/lock developer tools` |
| [#9907](https://github.com/ll7/robot_sf_ll7/pull/9907) | feat(benchmark): add #8872 pedestrian-speed campaign executor | No selected runtime | No: issue-owned #8872 speed experiment executor, not D-083 campaign. Never execute its retired seeds here. | `scripts/validation run_issue_8872 family` |
| [#9903](https://github.com/ll7/robot_sf_ll7/pull/9903) | docs(validation): preserve ten-fault VV-5 diagnostic matrix (#9735) | No | No: preserved VV-5 report and diagnostic scripts; effective runtime diff empty after excluding merged ancestry. | `main...PR runtime diff is empty` |
| [#9897](https://github.com/ll7/robot_sf_ll7/pull/9897) | analysis: land narrow-doorway crash-vs-wait diagnostic replay on current main | No | No: narrow-doorway diagnostic producer/report only, no released policy, map or metric edits. | `PR-only narrow_doorway_crash_vs_wait_issue_9545.py` |
| [#9896](https://github.com/ll7/robot_sf_ll7/pull/9896) | feat(planner): add evaluate-once arbitration API (#9868) | No selected arm; optional module only | No: additive evaluate-once/reorder APIs; no registration or release caller. | `multimodal_trajectory_arbitrator.py; rg callers` |
| [#9895](https://github.com/ll7/robot_sf_ll7/pull/9895) | feat(planner): add maneuver commitment manager (#8064) | No selected arm | No: new maneuver_commitment manager; additive until #8067 integration, outside roster. | `robot_sf/planner/maneuver_commitment.py` |
| [#9894](https://github.com/ll7/robot_sf_ll7/pull/9894) | feat(benchmark): add Stage-3 0.0.7-to-0.0.8 comparison gate (#9668) | Yes, release tooling/config/report surfaces | No current numeric defect demonstrated: alternative Stage-3/paired-era path is not D-083/D-062 authority; its v2 binding proposal needs reconciliation before mint. | `PR-only compare_release_007_008.py; release template proposal; #10112` |
| [#9866](https://github.com/ll7/robot_sf_ll7/pull/9866) | fix: consume full discrete CVaR tail mass (#9853) | No selected arm; optional CVaR code | No: only multimodal discrete_tail_metrics is changed. None of the 14 arms calls it; no conditional development A/B campaign is required. | `detailed CVaR reachability proof below` |
| [#9857](https://github.com/ll7/robot_sf_ll7/pull/9857) | fix(socnav): complete frame crosswalk and Risk-DWA count handling (#9845) | No runtime | No: crosswalk/frame tests only; zero-count fix is already merged #9869. | `net diff excludes risk_dwa.py` |
| [#9842](https://github.com/ll7/robot_sf_ll7/pull/9842) | docs: use path sources for focused coverage (#9516) | No | No: focused coverage documentation. | `docs coverage source references` |
| [#9836](https://github.com/ll7/robot_sf_ll7/pull/9836) | docs: classify unavailable evidence artifacts (#9711) | No | No: unavailable evidence classification documentation. | `documentation only` |
| [#9832](https://github.com/ll7/robot_sf_ll7/pull/9832) | feat(adversarial): add fixture-first feasibility-frontier report (#9654) | No selected runtime | No: fixture-first adversarial frontier report/API and figure export; no D-083 execution change. | `adversarial/feasibility_frontier_report.py; figures/export.py` |
| [#9831](https://github.com/ll7/robot_sf_ll7/pull/9831) | fix: anchor PR budget diff to event base SHA (#9409) | No | No: PR budget event-base check; effective runtime diff empty. | `scripts/ci/pr_contract_check.py` |
| [#9830](https://github.com/ll7/robot_sf_ll7/pull/9830) | test: cap Torch startup probe threads (#9593) | No | No: cap threads in Torch startup test child. | `tests/test_seed_utils.py` |
| [#9827](https://github.com/ll7/robot_sf_ll7/pull/9827) | fix(ci): read closing metadata via REST (#9715) | No | No: CI closing metadata REST route. | `scripts/ci/pr_contract_check.py` |
| [#9825](https://github.com/ll7/robot_sf_ll7/pull/9825) | fix(analysis-workbench): enable measured start-delay activation (#9357) | No selected runtime | No: workbench start-delay diagnostic activation. | `analysis_workbench/review_execute.py` |
| [#9821](https://github.com/ll7/robot_sf_ll7/pull/9821) | feat(planner): expose RiskDWA progress-escape status (#9533) | Yes, risk_dwa/guarded_ppo telemetry | No: call-scoped progress-escape status/trace, action selection unchanged; reported experiment negative. | `risk_dwa.py and guarded_ppo.py telemetry-only contribution` |
| [#9804](https://github.com/ll7/robot_sf_ll7/pull/9804) | fix: make DWA clearance grid-resolution invariant (#9740) | No selected arm | No: plain DWA grid clearance changes; roster uses risk_dwa, not dwa, and RiskDWA does not subclass/call DWA clearance. | `planner/dwa.py; roster; risk_dwa.py imports` |
| [#9796](https://github.com/ll7/robot_sf_ll7/pull/9796) | deps: bump pylint from 4.0.7 to 4.0.9 in /fast-pysf | No numeric runtime | No: fast-pysf Pylint lock dependency only. | `effective fast-pysf diff is developer uv.lock` |
| [#9793](https://github.com/ll7/robot_sf_ll7/pull/9793) | fix(benchmark): serialize camera-ready planner identity (#6643) | Yes, row/release identity plumbing | No measured numeric fix: additive root planner_key preserves arm identity. Main already binds runtime identities through #10023; no fresh selected-roster aggregate delta supplied. | `episode_planner_key schema; map_runner/camera_ready threading` |
| [#9792](https://github.com/ll7/robot_sf_ll7/pull/9792) | fix: align standalone PR contract v2 validation (#9517) | No | No: standalone PR contract parser. | `scripts/ci/pr_contract_check.py` |
| [#9789](https://github.com/ll7/robot_sf_ll7/pull/9789) | fix(benchmark): separate resumed rows from current throughput (#9105) | Yes, operational reporting | No selected scientific-number P1: reused rows excluded from current invocation throughput; fresh campaign numeric outcomes unchanged. Resume throughput defect is operational reporting. | `camera_ready/_reporting.py/_run_state.py current-work numerator` |
| [#9745](https://github.com/ll7/robot_sf_ll7/pull/9745) | feat(benchmark): add VV-2 reference planner oracles (#9732) | Yes, map-runner/reference validation surface | No: adds stand-still/reference oracles outside release roster; no selected policy or observed release-number correction. | `planner_command_contract.py; stand_still.py; reference oracle gate` |
| [#9719](https://github.com/ll7/robot_sf_ll7/pull/9719) | feat(release): bootstrap DOI-bound Zenodo metadata repair (#9431) | Yes, release/DOI tooling | No: guarded metadata repair/bootstrap changes publication admission, not episodes/numbers; no DOI mutation performed. | `zenodo_publisher.py; release_cli.py` |
| [#9244](https://github.com/ll7/robot_sf_ll7/pull/9244) | fix(benchmark): preserve obstacle-force diagnostic provenance (#8277) | Yes, shared metadata/fallback/occupancy | No observed numeric P1: valid dynamics unchanged, malformed occupancy/fallback provenance fail closed; opt-in obstacle-law diagnostic receipt. | `socnav_occupancy.py; fallback_policy.py; diagnostic receipt` |

## #9866 CVaR reachability

Fetched head `5d67f0180` (full SHA in the preserved PR-head inventory). The net
contribution changes only `robot_sf/planner/multimodal_trajectory_arbitrator.py`
and `tests/test_multimodal_tail_metrics.py`: it consumes the entire discrete
upper tail and avoids near-one-alpha/finite-tiny-loss underflow. At candidate
main, `rg -n 'multimodal_trajectory_arbitrator|discrete_tail_metrics|discrete_upper_tail_cvar' robot_sf --glob '*.py'`
finds **only the arbitrator itself** (definitions, internal summary calls and
exports). No registered release policy imports it. Its intended consumer is
future multimodal planner integration #8067, not one of the 14 arms.

`predictive_mppi` has a separately implemented risk objective in
`socnav_prediction.py`; chance-constrained MPC/provider have separate empirical
CVaR functions. Neither uses this changed discrete-tail function, and
chance-constrained MPC is not a release arm. Therefore no affected release arm
exists and no development-seed before/after simulation was run or required by
the conditional instruction. #9895/#9896 are additive optional future APIs;
#9897/#9939/#9993 are diagnostic/docs/developer state; #10073 retains the
unselected legacy default and is next-release calibration work.

## Open P1/high issues mentioning 0.0.8

Read issue bodies/labels via `gh issue list --state open --search '0.0.8 in:title,body label:"priority: high"' --limit 1000`
and the equivalent `label:p1` query. Empty/missing implementation evidence is
not treated as closed because an issue's label remains high.

| Issue | Classification | Evidence / required completion |
|---|---|---|
| #10110 | **blocks-freeze (orchestrator, 2026-10-03)** | Missing disclosure checks D-039/D-053/D-055/D-057 and related disclosures. The mint/preflight gate is release source the identity binds, so implementation must land before freeze, then pass at mint and again before publication. Runtime tests do not inspect notes. |
| #10112 | **blocks-freeze (orchestrator, 2026-10-03)** | SNQI v2 is a reported 0.0.8 number; absent D-083 v2 spec/binding, smoke contract and mint ordering require changes to the identity-bound release source. Pending anchors, public 103/private 1003, v0_2/v3 validator and empty trust pins remain concrete admission gaps. |
| #10055 | not blocking | Direct campaign production guard/fallback migration is next-release work. Existing pytest and sealed release source/seed guards protect this route; never launch old eval/seedless configs here. |
| #10052 | blocks-mint-or-publish only | 8/20 cross-host reruns differed; 76/76 same-node repeats equal. Record one node/CPU and enforce same-node resume, disclose hardware limitation; no new planner arithmetic fix is selected. |
| #10051 | not blocking under recorded disposition | ADVV: #9926 fixes contact TTC/exact static geometry; sampling native rollouts now #10008. Residual guard probe success 298→298, collisions 3→2 over 480 paired dev episodes, McNemar p=1; adopted next-release residuals, not an admitted release correction. NaN occurred 0/15,412; parser unused. Reopen if a new admitted measurable P1 is shown. |
| #10050 | not blocking | Present-force NaN guard gap remains, but invalid samples0/2,016 rehearsal rows; no established reported-number defect. Do not claim the proposed presence-mask fix is implemented. |
| #10049 | blocks-mint-or-publish only | Definitions digest not implemented. Procedural calibration-source→campaign-source zero metric/runtime-drift checklist must pass; anchor pinning/final source binding is unresolved. #10045 schema/calibration custody is not this digest check. |
| #10044 | blocks-mint-or-publish only | #10043 fixes new jerk_mean table key; 0.0.7 has 672/672 and 490/490 empty cells and stays immutable. Erratum disclosure and non-empty new output check remain. |
| #10021 | not blocking (numeric fixes integrated) | #10023 binds expected identity/config/checksum/slot sets; selected schedule/seeds D-083 and D-062 distribution comparator replace paired-era comparison. Remaining chain custody is #10112/G12. |
| #10017 | blocks-mint-or-publish only for disclosure | Wall law/blocked walkers retained by D-038/D-055; desired×1.3 retained by D-039, next-release physics work. Do not override the adopted disclosure disposition because the issue label is high. |
| #10015 | not blocking (fixed) | #10019 implements eligibility cohort, Spearman, duplicate-pair refusal, CI estimand, bootstrap seed0 and numeric grouping; source/register witnesses on main. |
| #10013 | blocks-mint-or-publish only | Release tracker: intake, calibration/scientific pins, current-source smoke/stress, campaign/export/comparator/publication still due. It is not itself a numeric defect. |
| #10007 | not blocking (identified numeric fixes integrated) | #10008/#10009/#10011/#10014/#10019 and both trains are ancestors; residuals have adopted disclosure/next-release dispositions. |
| #10006 | blocks-mint-or-publish only for disclosure | Inert keys/scenario overrides retained by D-010; no current freeze correction chosen. Notes must disclose selected overrides/tuning limitation. |
| #9850 | not blocking (calibration config fixed) | #10045 parity test loads actual calibration and D-083 configs with same social_force binding and dev 1001/1002. Smoke/public admission remains separately blocked by #10112. |
| #9668 | blocks-mint-or-publish only | Release epic; body has historical paired-seed assumptions superseded by D-049/D-062/D-083. Needs actual source-bound custody/intake/comparator, not old numeric freeze. |
| #9489 | not blocking | BA-06 workbench integration explicitly declined by author until a fresh ruling; not an authorized release prerequisite. |
| #9488 | not blocking | BA-05 broad autonomous service gates explicitly declined; supervised existing auditor use remains. |
| #9483 | not blocking | Auditor epic; supervised BA01–04/use does not require declined BA05/06 integration to freeze numbers. |

The 2026-10-03 ruling requires #10112 and #10110 to land before the freeze.
No additional blocker is established by the remaining snapshot classifications.
Scientific, operational and publication admission remains separately required.

## Integrated fixes and ancestry

`git merge-base --is-ancestor <merge-commit> c979e0337da4ad053d59a76225fbb9154140ee73`
was checked for both trains (#10046/#10080/#10095), #9999, #10081, #10045,
#10103 and #10108. #10067 is GitHub CLOSED, not separately MERGED: train #10095
contains its reviewed implementation via `913694c2d` and packet-pin repair
`76d40ed1d`; its source/config overlap witness is on main. HZN-1/HZN-2 and
PLAUS-1 are integrated fixes. PPO #10077, scenario #10026, physics #10024,
row reader #10047, comparator #10058 and curvature #10054 are also integrated.
Stale record qualifiers were reconciled; historical author rulings were retained.

## Candidate proof

Proof was run at clean, unmodified candidate main before documentation edits.
No runtime/planner/frozen-config or 0.0.7 artifact changes are made.

```bash
uv sync --all-extras
uv run python scripts/dev/ensure_result_interpretation_provenance.py
SEEDGUARD_WORKER_PROOF="$LANE/evidence/worker-proof" uv run pytest \
  -m "slow or not slow" -n 8 --dist=each tests/test_heldout_seed_guard.py \
  --junitxml=../evidence/guard-proof.xml -q
ROBOT_SF_PYTEST_SEED_AUDIT="$LANE/evidence/suite-seed-audit-{worker}.jsonl" \
  uv run pytest -m "slow or not slow" -n 8 --tb=short \
  --continue-on-collection-errors --junitxml=../evidence/full-suite.xml
```

`LANE=/home/luttkule/lanes/freezeprep`; commands use lane-local UV/HF/XDG/TMP
caches and OMP/OPENBLAS/MKL threads=1. Initial load 4.77 on imech039 admitted
8 workers. Sync/provenance/guard/full suite all exit 0. Guard proof: 304 passed,
8/8 worker receipts active. Full suite: **42,493 passed, 69 skipped, 7 existing
xfails, 48 warnings, 0 failures/errors**, 29m27s. All tests including slow were
selected; nothing was deselected to pass. XML includes one collection-skip
receipt in addition to the 42,568 collected nodes.

Pin/identity witnesses in that suite all passed: sealed-source pins 67;
sealed-runtime sources 7; D-083 authority 2; horizon authority 18/compatibility 11/
contracts 47; doorway acceptance 18; hybrid-v4 slots 19; resolved identity 32;
tag identity 27; development identity 28; PPO binding 5 + asset 1; SNQI-v2 witnesses 433;
seed-diff witnesses 127; seed boundary tests 38. The full suite includes every
other collected pin witness as well.

**Seed admission:** root guard installed on workers/children, both forbidden
bands refused before simulation boundaries; 9 worker/main audit files have
0 unexpected attempts. Expected guard probes deliberately suppress that audit
sink; their passed XML nodes/worker receipts are recorded refusal witnesses,
not permission to reset/step. No actual forbidden reset/step occurred.

**Failure/limitation classification:** no test failure or xdist-only failure.
The 69 existing skips cover optional credentials/external planners/datasets,
real archived bundles/harvests, browser/OMPL/PDF/network capabilities, distinct
validator checkout and opt-in matrix preflight; they are not new release
admission evidence. The 7 inherited xfails concern plain DWA(#9740), the unused
legacy social-force grid law(#9724), hybrid-v3 mirror/unit contracts(#9733/#9726)
and adopted radius/envelope limitations(#9750). They were not added/changed
here; selected v4/config witnesses pass.48 warnings are dependency/API
deprecations, negative-test numeric/archive inputs and diagnostic rendering/
temporary evidence-registration warnings. Per-node reasons are preserved in
`suite-dispositions.json` and XML, not suppressed.

Candidate non-pytest gates passed: broad-exception inventory, archetype geometry,
release endpoint dispositions, archetype parameters and seed-holdout diff.
The first Sphinx gate exited1 because `--all-extras` omits the docs dependency
group. After `UV_PROJECT_ENVIRONMENT="$LANE/docs-env" uv sync --group docs
--no-default-groups`, `scripts/dev/sphinx_strict_build.sh --output-dir
"$LANE/evidence/sphinx-candidate" --json` exited0/status=pass, no blocking warnings.
Its internal Sphinx exit 1 has 1,659 repository-allowed cross-reference warnings;
that is not a clean raw Sphinx build. This was an environment correction, not
a candidate code failure. Exact PR-head documentation gates and hosted CI are
recorded in the lane report separately from this candidate test proof.

The dry resolver generated and verified the main sealed resolved identity, SHA-256
`103d3cacd02c5fd3dd007ae11b67d11cee122adbf8929faa5244af7919be43a0`, using
tag `paper-matrix-v2-h600-s30-freezeprep-dry-c979e0337da4ad053d59a76225fbb9154140ee73`
and diagnostic coordinates `10.5281/zenodo.99000001` / `10.5281/zenodo.99000002`.
These are static preparation inputs, not authentic reservation/seed admission.
Resolved identity contains exactly the sealed tuple and authored 20,160 cells;
no reset, step, worker or campaign was invoked. Initial short tag was refused
(exit2) for lacking a canonical full-SHA suffix; corrected invocation and verify
both exit 0. This is an input-contract correction, not a source defect.

The fixed H400 doorway companion was also generated and verified dry (both
exit 0), SHA-256 `21eb784eff3d7c7fcd9491fa6843ba9fe1cc5feb379071b6f4eed8826f125318`,
with the same source/tag/diagnostic DOI coordinates and exactly 1,260 cells.

Raw proof logs, GitHub inventories, per-PR runtime diffs, worker receipts,
identity bytes and test XML are kept under the task lane's `evidence/` directory.
The final lane report supplies their absolute location, SHA inventory and hosted
CI outcome. These local artifacts are review inputs, not published benchmark data.

PR-head gate corrections: the docs-integrity checker initially rejected two
PR-only script paths as missing on main. Their audit citations now explicitly
identify PR-only basenames; no nonexistent main tool is offered. The changed-
coverage gate initially lacked its required coverage JSON; the existing static
ledger suite was rerun with real coverage capture, then the exact-SHA gate
correctly reported documentation-only coverage as not required. These are
preparation/gate-input corrections, not candidate runtime failures.
