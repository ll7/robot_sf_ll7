# Changelog

All notable changes to the Robot SF project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* **issue #5444 action-conditioned online collision-risk API and baselines.** New
  `robot_sf/research/collision_risk/` package exposes a planner-agnostic, versioned
  (`action_conditioned_collision_risk.v1`) estimate of `P(contact in (t, t+H] | action u)`.
  A constant-velocity Monte Carlo baseline (exact disc-footprint segment geometry, declared
  velocity-noise and cross-actor-correlation assumptions) emits the joint contact probability,
  per-actor marginals (explicitly not summed as independent), first-passage/hazard decomposition,
  and a union-bound vs intentionally-invalid independence comparison. Deterministic TTC /
  velocity-obstacle / reachability warnings are labelled non-probabilistic, and an
  uncertainty/OOD/abstention block plus estimator/forecast/geometry/horizon/action/config
  provenance and p50/p95/p99 latency accompany every estimate.
  `scripts/analysis/collision_risk_report.py` runs a frozen reference workload
  (`configs/research/collision_risk_baseline.yaml`): p95 latency is ~10 ms (well under the 100 ms
  deadline, classified `online`) and risk differs in the expected direction between two candidate
  actions. Hard guards remain authoritative; no `safe` label is emitted and low probability is
  never treated as safety. Evidence is API + baseline fixture, not a calibrated benchmark risk
  claim; no benchmark campaign or Slurm/GPU run is included.
* **issue #5442 frozen-state counterfactual replay: locate the last avoidable control action.**
  New simulator-agnostic engine (`robot_sf/benchmark/last_avoidable_replay.py`) restores a
  decision-point snapshot (including RNG state), verifies deterministic baseline replay, and branches
  over the admissible robot action lattice at each step in `[t_danger, t_contact)` to report `t_uca`
  (earliest avoidable unsafe control action) and `t_inevitable` (point of no return). Fail-closed: a
  nondeterministic baseline or a missing feasible action set returns `unknown`, never `unavoidable`.
  Ships a deterministic controlled kinematic fixture
  (`robot_sf/benchmark/last_avoidable_fixtures.py`), the `last_avoidable_replay.v1` output schema, an
  offline report CLI (`scripts/analysis/run_last_avoidable_replay_issue_5442.py`), and a context note
  (`docs/context/issue_5442_last_avoidable_replay.md`). Controlled-fixture diagnostic evidence only;
  `normative_fault` is always `not_assessed`. No production-simulator snapshot seam (that would be a
  broad change — see the note), no benchmark/Slurm run, no metric/paper claim.
* **issue #5441 `collision_causal_report.v1` fail-closed cause-report contract.** Adds
  `robot_sf/benchmark/schemas/collision_causal_report.v1.json` and validator
  `robot_sf/benchmark/collision_causal_report.py` that separate observed reconstruction, proximate
  mechanism, and intervention-supported causal contribution for a single collision, always setting
  `normative_fault: not_assessed`. The contract reuses `MECHANISM_LABELS`/`MECHANISM_CONFIDENCES`
  (no competing taxonomy), marks unsupported fields (`t_uca`, `t_inevitable`, planner internals for
  black-box planners) unavailable rather than inferred, forbids asserting a planner cause once
  contact is inevitable, and ships one complete and one abstaining fixture. Producer inventory and
  decision record: `docs/context/collision_causal_report_field_map_2026-07-13.md`. Schema/fixture
  evidence only — not proof that real collisions can be causally attributed.

* **issue #5419 authorization-gated DPCBF executor.** Bounded local `run_batch` arms require the
  exact public authorization ID. Atomic checkpoints bind inputs and completed JSONL artifacts;
  dirty, orphaned, malformed, duplicate, or mismatched state fails closed. No Slurm/GPU run or
  safety-performance claim is included.

### Fixed

* **issue #5464 PR Contract Check no longer flags modified evidence files as new.** The
  `pr-contract-check.yml` workflow used `actions/checkout` on `pull_request` without fetching the
  base branch, so `origin/main` was absent in the runner. `pr_contract_check.py`'s `is_file_new`
  then treated the failed `git show origin/main:path` as "file is new" for *every* changed evidence
  file, raising false-positive `AI-GENERATED`/`NEEDS-REVIEW` marker blockers on `docs/context/evidence/**`
  files that already exist marker-less on `main` (observed on PR #5463). Fixed on two fronts: the
  workflow now fetches the base ref (so `origin/<base>` resolves) and passes an authoritative
  `--added-files-file` derived from the GitHub `pulls/{n}/files` API (`status == "added"`); and the
  script now treats an unresolvable base ref as "unknown, not new" and prefers the authoritative
  added-files signal over the git heuristic when available. CI-tooling correctness fix
  (`diagnostic-only`); no benchmark, metric, or evidence-content change.
* **issue #5429 `load_scenario_matrix` no longer misroutes single-document abstract scenario
  files.** A single-document YAML file whose top-level content is a *list* of abstract benchmark
  scenarios (the `density`/`flow`/`obstacle` form, e.g. `yaml.safe_dump([s1, s2])`) is now returned
  directly — symmetric with the existing multi-document stream behavior — instead of being sent
  through the map-oriented `robot_sf/training/scenario_loader.py::load_scenarios`. Previously such
  files were validated as map manifests and emitted misleading
  `Scenario entry N is missing a name or scenario_id` / `has no map_file or map_id` warnings even
  though abstract scenarios legitimately carry neither. Single-document *mappings* still route to the
  include-aware manifest loader (unchanged), and an empty single-document list now fails closed with
  a clear `ValueError` rather than a silent zero-job run.

* **issue #5340 main CI regression from unconditional `spaces.Sequence` leaf.** PRs #5335
  and #5337 unconditionally added `spaces.Sequence` leaves to the SocNav structured observation
  space, breaking `asymmetric_critic` (`Box` leaves) and `FlattenDictObservationWrapper`. Reverted
  in PR #5339 to restore `Box`-only observation spaces for consumers. Reverted PRs may be relanded with a
  guarded registration path later.

* **issue #5217 restore typical pedestrian desired-speed propagation.** `SimulationSettings`
  with `ped_speed_tier="typical"` (or explicit `desired_speed_mean`) now correctly sets
  `pysf_sim.peds.max_speeds` to the decoupled desired-speed distribution (~1.3 m/s for the
  `typical` tier) on all pysocialforce install versions. The previous implementation relied solely
  on `pysf_config.scene_config` propagation, which was silently ignored by pysf installs that
  predated the fast-pysf #5042 update. The fix adds `_enforce_ped_desired_speeds` in
  `simulator.py`, which applies the sampled speeds directly to `peds.max_speeds` and encodes
  them into `peds.initial_speeds` so the legacy `max_speed_multiplier * initial_speed`
  recomputation also yields the correct values on every state update. A companion
  `sample_desired_pedestrian_speeds` helper in `pedestrian_speed_tiers.py` provides the
  standalone sampling without a pysf-version dependency.

### Changed

* **Single-source package version from the git tag (kills version drift).** `pyproject.toml` no longer hardcodes a version; the package version is derived automatically from the git tag/release line via `hatch-vcs` (release tags `X.Y.Z`, release-candidate tags `rcX.Y.Z`, PEP 440 dev fallback `0.0.3.dev0` for untagged builds). `robot_sf.__version__` now resolves from the build-time `_version.py` (or installed metadata). `CITATION.cff` is aligned to the latest full release tag (`0.0.2`). A new `scripts/dev/check_version_alignment.py` guards the three version axes; it runs gating on tag pushes (`release-functional-badge` workflow) and advisory on every CI run (`ci_driver.sh` lint phase). Previously the three axes had drifted: tag `rc0.0.3`, `pyproject` `2.0.0`, `CITATION` `benchmark-protocol-0.1.0`.

* **issue #5228 xdist memory-diagnostic robustness.** Three residual gaps from the
  PR #4952 review are closed: (1) `_tree_rss_gb` no longer calls `is_running()` before
  `memory_info()` — the latter already supplies the handled `NoSuchProcess` signal,
  removing a superfluous kernel round-trip per process; (2) `main()` validates
  `--ceiling-gb`, `--auto-workers`, and `--project-at` arguments early (exit 2 before
  any sweep starts) instead of failing late during report construction; (3)
  `test_sample_process_tree_peak_captures_allocation` now calls `_terminate_process_tree`
  before `popen.wait` in its `finally` block so a still-alive child cannot be left
  running after a sampling or assertion failure. Regression coverage added for all three
  changes in `tests/dev/test_measure_xdist_worker_memory.py`.

* **issue #5071 absolute continuous-integration coverage floor.** The unsharded full-suite
  coverage report on `main` and manual CI runs must cover at least 85% of the `robot_sf` package.
  This blocking floor reuses the existing report while the baseline-relative comparison remains
  advisory; pull-request test sharding and fast-feedback latency are unchanged.

* **issue #4993 complete RobotSfError migration.** Re-parents the remaining 76 ad-hoc
  exception classes across `analysis_workbench`, `benchmark` (individual files), `common`,
  `data_ingestion`, `examples`, `nav`, `planner`, `research`, and `training` subpackages onto
  `RobotSfError` while preserving all existing `ValueError` / `RuntimeError` / `Exception`
  ancestry. `except ValueError` and `except SpecificError` clauses are unaffected; new code
  can now also target `except RobotSfError` for broad-package catches. Covered by
  `tests/test_robot_sf_error_migration_4993.py` (59 compatibility tests).

### Added

* **issue #5372 fidelity smoke + arm-config identity checks for the #5355 factorial.** Adds two CPU-only, tiny-horizon test modules under `tests/planner/`: a toggle-effect fidelity smoke (`test_prediction_mpc_factorial_fidelity_smoke.py`) that runs all four canonical arm configs (`configs/algos/prediction_mpc_factorial_A{0,1}_B{0,1}.yaml`) on two named scenarios and asserts each factor flip changes the decision trace, the toggles are orthogonal, and the B-OFF arms complete episodes as functional soft-cost planners (prereg §6); and a static preflight identity test (`test_prediction_mpc_factorial_arm_identity.py`) asserting the four arms share the observation contract, kinematics, and runtime budget and differ only in the two factor flags plus the implied soft pedestrian weight (prereg §2, §7). Test-only; no campaign, GPU, or metric change.

* **issue #4978 scenario flakiness audit applied to a real multi-planner campaign.** Ships a
  tracked, compact subset of a real benchmark campaign (4 planners × 5 scenarios × 20 seeds = 400
  episodes, with provenance) under
  `tests/fixtures/benchmark/scenario_flakiness_issue_4978/real_campaign_episodes.jsonl`, plus the
  committed `scenario_flakiness.v1` report the audit produces from it
  (`real_campaign_flakiness_report.json`). This delivers the *real-campaign application* that PRs
  #5069 and #5115 deferred: the audit now has regression-protected evidence that it surfaces real
  per-cell outcome instability — 7 of 20 assessable cells flagged knife-edge (35%), including a
  perfect coin-flip cell (`classic_doorway_medium`/`ppo`, stability 0.50 across 20 seeds). Exact-repeat
  determinism is reported as `null` (unknown) because each seed ran once; asserting it needs a
  dedicated exact-repeat campaign. Claim boundary: diagnostic-grade only — no planner-quality or
  ranking claim, no benchmark campaign was run. Covered by
  `tests/benchmark/test_scenario_flakiness_real_campaign.py`.
* **issue #5137 planner-free feasibility oracle per scenario cell + envelope-sensitivity axis.**
  `robot_sf/scenario_certification/feasibility_oracle.py` is a planner-free oracle that discharges
  route-infeasibility for zero-completion cells (bottleneck, cross-trap, head-on-corridor families).
  It combines the existing `scenario_cert.v1` route certifier (inflated A* shortest path under the
  same robot envelope and kinematics) with a scripted actor-free waypoint traversal (the canonical
  `goal`-algo map episode, pedestrians removed) and reports per-cell feasible yes/no **plus margins**:
  minimum corridor width vs envelope diameter, and minimum completion steps vs horizon (observed and
  a kinematic lower bound). It adds an **envelope-sensitivity axis** that re-runs the oracle at a
  reduced envelope radius (default `1.0 m` nominal vs `0.5 m` reduced) to separate "hard for planners"
  from "infeasible by construction". It also emits the oracle verdict into campaign metadata via
  `annotate_zero_completion_cells`, so zero-completion cells are automatically annotated as
  route-infeasible-by-construction, envelope-sensitive-hard, or still-planner-limited. This is new
  capability: issue #3484 `feasibility_diagnostics.py` has no envelope axis and no margin assembly,
  and the static MAPF oracle explicitly excludes kinematics. Focused tests in
  `tests/scenario_certification/test_feasibility_oracle.py` cover margin math, envelope-sensitivity
  classification, campaign annotation, fail-closed behavior, and one real end-to-end rollout. Claim
  boundary: `diagnostic_only_not_benchmark_evidence` — no benchmark metric or paper-facing claim.

* **issue #5118 CPU vectorized environment (VecEnv) worker-mode throughput comparator.**
  `scripts/validation/run_vecenv_worker_mode_throughput.py` accepts a standard training config YAML,
  constructs `dummy`, `subproc`, and `threaded` VecEnv modes, runs configurable warmup and step
  loops, and writes a machine-readable JSON artifact (`vecenv_throughput_comparator.v1`) with
  transitions-per-second, speedup vs. the dummy baseline, config SHA-256, git commit, and host
  provenance. Focused tests in `tests/validation/test_run_vecenv_worker_mode_throughput.py` cover
  helper contracts and CLI output schema. The `threaded_lidar_batch` mode awaits PR #5123.

### Fixed

* **issue #5090 make `robot_sf.benchmark` package imports lazy.**
  The broad eager import surface in `robot_sf/benchmark/__init__.py` previously triggered
  TensorFlow/oneDNN initialisation and simulator-registry messages (~8 s startup delay) whenever
  any `robot_sf.benchmark.<sub-module>` was imported. Replaced with a `__getattr__`-based lazy
  loader: each public name in `__all__` is resolved on first access and then cached. Lightweight
  schema and error modules now start in milliseconds. Public API surface and `__all__` are
  unchanged. A subprocess regression test (`tests/benchmark/test_benchmark_package_lazy_imports.py`)
  asserts no TF/simulator-registry noise and startup within a 5 s budget for canonically
  lightweight sub-modules. No benchmark metric semantics were changed.

* **issue #5091 fresh worktree virtualenv bootstrap now fails closed.** Added
  `scripts/dev/bootstrap_worktree.sh`, a helper that runs `uv venv .venv` before
  `uv sync --all-extras` and verifies `.venv/bin/python` exists afterward. Without the
  explicit `uv venv .venv` step, `uv sync` in a fresh linked worktree may silently reuse the
  main checkout's `.venv` without creating one locally, leaving `.venv/bin/activate` missing.
  Updated `AGENTS.md` Fresh Worktree Bootstrap instructions to document the required order and
  reference the new helper. Nine targeted contract tests added to
  `tests/test_ci_script_contract.py`. No benchmark metric or schema changes.

* **issue #4988 benchmark CLI surfaces typed errors (not raw tracebacks) for malformed input.**
  `robot_sf/benchmark/parquet_export.py` now raises the canonical `EpisodeRecordInputError`
  (a `ValueError` subclass, so backward-compatible) instead of a bare `ValueError` when a JSONL line
  is unparseable. This lands inside the `export-parquet` CLI boundary's typed `_CLI_INPUT_ERRORS`
  handler, so a corrupt input file now exits `2` with a logged message rather than escaping
  `cli_main` as a raw traceback. A new parametrized CLI-boundary contract test
  (`tests/benchmark/test_cli_typed_error_contract.py`) ratchets the missing- and malformed-input
  fail-closed behavior across 20 input-consuming subcommands. Claim boundary: error-surface/exit-code
  behavior only — no success-path output or benchmark metric value changes.
* **issue #5031 docs-only PR bodies can now select "domain approval not required".** The PR
  follow-up checker (`scripts/dev/check_pr_followups.py::analyze_domain_approval`) previously forced
  `domain_approval_required` on any body that merely *mentioned* an evidence concept in prose (e.g. a
  docs page discussing "benchmark interpretation" or "diagnostic-only"), so a genuinely docs-only PR
  could not use the template's documented `Required for this PR: no - reason` / `Status: not required`
  opt-out. `analyze_domain_approval` now accepts that opt-out when the only triggers are weak
  free-form prose mentions. A filled Research Result Guidance declaration (a concrete `Evidence tier`
  / `Result classification`) remains a strong self-declaration and keeps the strict approval path, so
  this cannot wave through an evidence-sensitive PR. Found while implementing #4967. No benchmark
  metric semantics change.

* **issue #5000 goal-planner late-evasive latency instrumentation (fail-closed).** The
  `late_evasive_predicate` (`robot_sf/benchmark/safety_predicates.py`, schema bumped
  `safety_predicate.late_evasive.v1` → `.v2`) now emits a `latency_unavailable_reason` alongside
  `response_latency_s`, so a `late_evasive=true` event is never a silent-empty latency. Root cause of
  the reported 109/110 goal-planner anomaly: the goal planner almost never produces a
  clearance-restoring deceleration after the hazard becomes visible, so `late_evasive` fires via the
  no-action branch (`first_clearance_restoring_action_step is None`) while `response_latency_s` is
  genuinely undefined — not a false positive and not a missing timestamp. The reason value is
  `no_clearance_restoring_action` (dominant goal case) or `hazard_never_visible`, and `null` exactly
  when latency is finite. The trace-surface `_late_evasive_reaction`
  (`robot_sf/analysis_workbench/trace_failure_predicates.py`) gains a seconds-valued
  `response_latency_s` companion to its `reaction_delay_steps`. The downstream
  `scripts/analysis/issue_4904_latency_decel_profile.py` derivation now surfaces the reason tally
  (`late_evasive_no_latency`, `dominant_latency_unavailable_reason`) instead of a silent gap. Claim
  boundary: telemetry/analysis-surface correctness only — no benchmark metric semantics change, no
  hybrid-/goal-planner effectiveness claim on the 0-latency data.

### Added

* **issue #4978 scenario-flakiness scores can be embedded in aggregate summaries.** The
  `robot_sf_bench aggregate` command now accepts `--include-flakiness-audit` to add the existing
  advisory `scenario_flakiness.v1` report under `_meta.scenario_flakiness`. The opt-in report gives
  campaign-summary consumers per-scenario/planner stability scores and explicit knife-edge flags
  without changing rankings or default aggregate output. It fails closed when the input has no
  usable binary outcome evidence. Claim boundary: summary integration only; no campaign was run and
  no benchmark metric or ranking semantics changed.
* **issue #5039 compat-matrix promotion readiness gate.** New
  `scripts/ci/check_compat_matrix_promotion_readiness.py` turns "is the advisory
  `compat-matrix` job proven enough to promote to a required CI gate?" into a machine-checkable,
  fail-closed decision. It reads
  `docs/context/issue_5039_compat_matrix_promotion_manifest.yaml` (recorded hosted-run evidence
  plus the objective gate: all four `ubuntu`/`macos` × Python 3.11/3.13 cells green ≥3 times each
  within the 30-minute budget) and reports `ready`/`blocked`; `--require-ready` exits non-zero
  until the evidence exists. Current state is `blocked`: the advisory matrix (PR #5037) is now on
  `main`, but no eligible hosted evidence has been recorded in the manifest — so promotion is
  deferred to an evidence-carrying follow-up. The absolute coverage floor is tracked separately in
  issue #5071. Covered by `tests/test_compat_matrix_promotion_readiness.py`.
  Claim boundary: readiness bookkeeping only; no CI gate is changed and no benchmark/paper claim
  is asserted.
* **issue #5034 control-action-latency sweep fail-closed preflight + blocker packet.** New
  `robot_sf/benchmark/control_action_latency_preflight.py` (`check_control_action_latency_axis`) and
  CLI `scripts/benchmark/preflight_control_action_latency_sweep.py` guard the issue-#5034 sweep: they
  fail closed unless the fidelity-sensitivity study config carries a `control_action_latency` axis
  whose variants cover the required action-latency steps `[0, 1, 3]` (the 0/100/300 ms-equivalent
  delays). Its durable packet records the historical blocked result before PR #5026 landed the axis;
  the same preflight now returns `ready` on `main`. Durable fail-closed decision packet at
  `docs/context/evidence/issue_5034_control_action_latency_sweep_blocked_2026-07-10/`. Claim boundary:
  launch/readiness preflight only — not benchmark evidence, not paper-facing; no campaign run and no
  Slurm/GPU submission were performed.
* **issue #5034 control-action-latency sweep metric-evidence promoter (successor slice).** New
  `robot_sf/benchmark/control_action_latency_evidence.py` (`build_latency_evidence`,
  `promote_latency_evidence`) and CLI `scripts/benchmark/promote_control_action_latency_evidence.py`
  promote raw fidelity-campaign episode rows into a durable compact control-action-latency evidence
  summary: they isolate the `control_action_latency` axis, report the action-latency metadata plus
  `success_rate`, `collision_rate`, and `min_clearance` per native latency cell (0/100/300 ms-equivalent
  steps `[0, 1, 3]`), and classify every fallback / degraded / non-native row (per the issue #691
  benchmark fallback policy) as an exclusion that never contributes to the result metrics. The promoter
  fails closed when the latency preflight is not ready or when the native result rows do not cover all
  required steps, so a partial or non-latency run cannot be promoted as the latency sweep. Successor
  slice to PR #5061 (which added the fail-closed preflight + historical blocker packet); it adds NEW
  metric-evidence promotion capability and does not duplicate #5061. It runs no episode and makes no
  benchmark / simulator-realism / sim-to-real / paper-facing claim; focused coverage lives in
  `tests/benchmark/test_control_action_latency_evidence.py`. The native campaign run remains out of
  scope for this slice.
* **issue #5048 gh list truncation guard extended to the remaining bounded callers.** The shared
  `scripts/dev/_gh_pagination.py` guard (from #4991 / PR #5040) is now applied to the six remaining
  bounded `gh ... list --limit N` call sites so a result at the cap is never silently mistaken for a
  full page. `snapshot_issue_batch.snapshot_claimable_issues`, `closed_state_label_hygiene`, and
  `open_issue_closure_audit` add structured `truncated` / `truncation_note` markers (per-label and
  per-issue where applicable) to their JSON reports; `compact_ci_snapshot` adds a `truncated` field
  to its `DriftSample`; `watch_pr_ci_status.fetch_recent_successful_ci_durations` logs a structured
  truncation warning on a capped drift sample; and `project_priority_score.GhProjectClient.item_list`
  fails closed with `GhListTruncated` because it drives Priority Score write-backs. Focused
  regression coverage lives in `tests/dev/test_gh_list_truncation_remaining.py` and
  `tests/tools/test_project_priority_score_truncation.py`. Tooling/evidence-integrity only — no
  benchmark, metric, or paper-facing claim.
* **issue #5059 disjoint modulo scopes for concurrent PR gates.** New
  `scripts/tools/pr_gate_scopes.py` gives the autonomous PR-gate orchestrator a canonical,
  immutable dispatch contract: an active gate owns a *residue class* (`pr % modulus == residue`)
  rather than an ad-hoc PR range, so a gate told to "process newly appearing PRs" can never drift
  into another gate's PRs on a re-list pass. `validate_active_gates()` is the fail-closed regression
  check — it rejects any non-residue (range) scope as non-immutable and reports any pair of live
  gates that could own the same PR (with the smallest crossing PR as a witness); residue
  disjointness is decided exactly by the CRT criterion `(r_a - r_b) % gcd(N_a, N_b) != 0`. Includes a
  CLI (`python -m scripts.tools.pr_gate_scopes --gates gates.json`) for checking a live gate manifest
  before admitting a second gate, and a docs note at
  `docs/dev/agents/pr_gate_scope_contract.md`. Pure arithmetic over gate descriptors — no GitHub
  calls, no benchmark/metric semantics. Tests in `tests/tools/test_pr_gate_scopes.py`.
* **issue #4978 scenario flakiness audit — exact-repeat determinism + per-cell outcome-stability.**
  New `robot_sf/benchmark/scenario_flakiness.py` (`compute_flakiness_audit`, schema
  `scenario_flakiness.v1`) and a `robot_sf_bench flakiness-audit` CLI subcommand measure two forms of
  outcome instability from existing episode JSONL: (1) exact-repeat determinism — when a
  `(scenario, planner, seed)` cell is executed more than once, whether repeats share the same binary
  outcome (a `False` verdict is a reproducibility bug the audit doubles as a detector for); and
  (2) per-cell outcome stability — the fraction of seeds in a `(scenario, planner)` cell that agree
  with the majority outcome, flagging `knife_edge` cells below a configurable threshold (default
  0.8). The audit is advisory and read-only: it emits a schema-versioned report and does not change
  rankings, campaign summaries, or metric semantics. Fail-closed by design — empty input raises,
  cells below `min_seeds` are reported as not assessable rather than counted as stable, and
  determinism is reported as `null` (unknown) when no exact-repeat data exists. Claim boundary:
  new diagnostic capability validated on synthetic and CPU fixtures; no benchmark campaign run.
  *(Issue #5072 adds fail-closed observation-track handling to this audit; see the #5072 entry
  below.)*
* **issue #5072 flakiness audit observation-track handling is now fail-closed.** The
  `robot_sf_bench flakiness-audit` audit (`robot_sf/benchmark/scenario_flakiness.py` from #4978)
  now enforces the benchmark observation-track boundary it previously deferred. By default
  (`--observation-track-mode strict`) it refuses to pool records that declare different
  `benchmark_track` values into one stability cell, raising `AggregationMetadataError` (CLI exit
  code 2) so rows with incompatible observation contracts are never silently compared. An opt-in
  `--observation-track-mode diagnostic-cross-track` mode partitions cells per track
  (`track :: scenario :: planner`) and emits an explicit `cross_track_caveat` for an explicitly
  caveated cross-track diagnostic. This reuses the canonical `observation-track` policy helpers
  shared by the aggregate/report CLIs (`robot_sf/benchmark/aggregate.py`) so the behavior is
  consistent across benchmark subcommands. Reports gain an `observation_track_mode` field and an
  `observation_tracks` metadata block, and each cell gains a `benchmark_track` field, making the
  track policy reproducible. Single-track and undeclared-track fixtures remain compatible in
  strict mode. Read-only diagnostic claim boundary is preserved; no ranking, summary, or metric
  semantics change. Deferred from PR #5069 as a distinct benchmark-input contract; validated on
  CPU fixtures only.
* **issue #3574 realized-distribution audit for heterogeneous-population traces.**
  `robot_sf/benchmark/heterogeneous_population_metrics.py` gains `realized_distribution_audit` and
  `summarize_distribution` (plus a `RealizedDistributionSpec`), covering DoD item 5: configured
  target vs. realized per-step distributions (overall and per-archetype) from one control trace, so
  the interaction/truncation shift the #3206 smoke could not compute is made explicit. A numeric
  configured→realized mean shift is reported only when the caller declares the two sides comparable,
  and missing or non-finite trace fields fail closed as blockers. Pure analysis only — no ablation
  campaign, Slurm submission, or heterogeneity/realism claim. Tests in
  `tests/benchmark/test_heterogeneous_population_distribution_audit.py`.
* **issue #4871 CrowdNav_Prediction_AttnGraph external learned-baseline feasibility smoke.** New
  `robot_sf/planner/crowdnav_pred_attng.py` is the thinnest model-only adapter proving the shipped
  ICRA 2023 attention-graph SRNN checkpoint (`41200.pt`, MIT, pinned at upstream `3907731`) loads and
  acts on synthetic Robot SF observations, plus the per-step wall-clock measurement. The repo is
  registered in `scripts/tools/manage_external_repos.py` (`stage crowdnav_pred_attng`) and the
  PyTorch-only inference path is exercised via a `tests/planner/test_crowdnav_pred_attng.py` smoke
  that skips cleanly when the repo is not staged. The adapter reconstructs the upstream dict
  observation (5-step constant-velocity future edges, holonomic `ActionXY`) and reports ~1–2 ms/step
  on CPU, flat in neighbor count. Verdict recorded in `docs/benchmark_experimental_planners.md` with
  the full contract + transfer-caveat analysis in
  `docs/context/issue_4871_crowdnav_pred_attng_smoke.md`. Claim boundary: `smoke evidence` only —
  no roster addition (not in `algorithm_metadata`/`algorithm_readiness`), no benchmark campaign, no
  retraining; zero-shot transfer into Robot SF is documented as not defensible without an
  out-of-training-distribution caveat (holonomic vs unicycle, ORCA vs social-force crowd, GST
  prediction variant needs TensorFlow).
* **issue #3501 safety-wrapper deadlock-recovery stage wired into benchmark runtime.** The stateful
  fourth wrapper stage (`DeadlockRecoveryMonitor`) is now bound into the benchmark episode step loop
  via `robot_sf/benchmark/safety_wrapper_runtime.py`. `SafetyWrapperRuntimeConfig` gains an opt-in
  `deadlock_recovery_enabled` flag plus its fixed, predeclared thresholds; new
  `deadlock_recovery_config()` / `make_deadlock_recovery_monitor()` helpers construct one monitor per
  episode, and `apply_runtime_safety_wrapper()` accepts it and records a `deadlock_recovery`
  sub-block per step. `summarize_safety_wrapper_trace()` aggregates a per-episode
  `deadlock_recovery` summary (frozen/detected/recovery-active step counts). The stage is
  **off by default** and only valid on the `wrapper_on` arm; when enabled its thresholds are locked
  to the predeclared defaults (fail-closed, no per-planner tuning), and the monitor only ever
  overrides angular velocity — forward speed is preserved so the hard-stop veto still holds. Claim
  boundary: `diagnostic_proxy` evidence only; this closes the last CPU-only implementable criterion
  of #3501. Remaining work is compute-gated (run the paired `planner × {wrapper_off, wrapper_on}`
  ablation, then build the effect-size report). No benchmark campaign run, no SLURM/GPU submission,
  no paper/dissertation claim edits. See `tests/benchmark/test_safety_wrapper_runtime.py`.
* **issue #4617 learned-risk model v1 trainer entrypoint + config slice (parent #1472).** New
  `scripts/training/train_learned_risk_model.py` (core logic in
  `robot_sf/training/learned_risk_trainer.py`) and `configs/training/learned_risk_model_v1.yaml`
  materialize the two paths draft PR #4552's launch-packet `slurm_execution` block cites as
  must-exist. The entrypoint validates the #1472 launch packet, then either runs a dependency-light
  CPU smoke (`--smoke`: fits one numpy logistic head per risk label on a tiny seeded synthetic
  fixture, writes a `smoke_completed` status artifact with `auroc`/`auprc`/`brier`/
  `false_negative_rate` diagnostics) or, in real mode, fails closed against the
  `check_learned_risk_campaign_readiness` gate and writes a `blocked_trace_manifest` status while
  the durable trace manifest (issue #2312 / #4586) stays unresolved. The training config mirrors the
  launch-packet `label_targets`/`feature_inputs` and keeps declared artifact paths off worktree-local
  `output/`. Claim boundary: launch entrypoint + CPU smoke only — no Slurm submission, no full
  training campaign, no checkpoint publication, no learned-risk claim promotion; hard guards remain
  authoritative and the learned output is auxiliary-cost-only. `smoke evidence` only. See
  `tests/training/test_train_learned_risk_model.py`.
* **issue #4013 paired diagnostic model-based planning comparison RUN (diagnostic).** New
  `scripts/benchmark/run_issue_4013_model_based_comparison.py` runs the end-to-end comparison that
  was deferred across the #4013 scaffolding PRs: it trains the short-horizon predictor checkpoint
  (CPU, git-ignored `output/`) when missing, runs three arms through `map_runner.run_map_batch`
  (`learned_prediction_mpc` on the trained checkpoint, `cv_prediction_mpc`, and a model-free `goal`
  baseline via new `configs/benchmarks/issue_4013_model_based_checkpoint_smoke.yaml` and
  `issue_4013_model_free_baseline_smoke.yaml`), then builds the diagnostic comparison report.
  Observed local run (scenario `francis2023_blind_corner`, seed 4013, horizon 30): all three arms
  produced one non-fallback evidence episode (`algorithm_metadata.status=ok`), the report reached
  `status=diagnostic_ready` with `paired_seed_count=1`, zero blockers, and all five closure criteria
  met. Promoted `comparison_report.v1.{json,md}` to
  `docs/context/evidence/issue_4013_learned_model_based_planning/`. Claim boundary: single
  scenario/single seed diagnostic smoke — not benchmark, navigation-quality, or paper-facing
  evidence, and not a large generative world model. Covered by
  `tests/benchmark/test_run_issue_4013_model_based_comparison.py`.
* **issue #4627 behavior-token motion-prior diagnostics (experimental, offline).** New
  `experiments/behavior_tokens/` namespace with an offline, read-only prototype that turns saved
  benchmark interaction traces into discrete "behavior tokens" for diagnostics. `extract_windows.py`
  slides fixed-size windows over `algorithm_metadata.simulation_step_trace.steps` and converts each
  into a documented, interpretable feature vector (clearance, time-to-contact proxy, robot-speed
  statistics, stop/yield and oscillation proxies, near-conflict recovery); `quantize_trace_windows.py`
  standardizes the finite feature columns and assigns each valid window a deterministic discrete
  token id via k-means (scikit-learn with a NumPy-only fallback, token ids canonicalized by cluster
  center so runs and libraries agree); `inspect_token_motifs.py` summarizes token distributions by
  scenario/planner/outcome and exports heuristic *candidate* motif labels plus bounded example
  windows. `schemas.py` holds the shared feature vocabulary and schema-version constants. Rows without
  a usable trace are skipped with an explicit reason; non-derivable features are recorded as `null`
  (never fabricated zeros). Covered by `tests/experiments/test_behavior_tokens.py`. Claim boundary:
  very low priority, exploratory, diagnostic-only tooling — not validated metrics, benchmark evidence,
  release-gate input, or paper/dissertation claim support, and no safety decision may depend on the
  tokens. No new controller training, no transformer dependency, no benchmark pipeline integration,
  no benchmark campaign run, and no SLURM/GPU submission.
* **issue #4013 checkpoint-backed model-based action selection (diagnostic).** New
  `configs/algos/learned_prediction_mpc_issue_4013_checkpoint.yaml` wires the trained short-horizon
  predictor checkpoint into the `learned_prediction_mpc` adapter fail-closed (`allow_untrained_smoke`
  and `fallback_to_constant_velocity` both false), and `tests/planner/test_learned_prediction_mpc_checkpoint.py`
  proves the adapter's `plan()` emits a finite, bounded, goal-directed unicycle command from the
  loaded checkpoint — in open space and with a pedestrian in the path — with the predictor reporting
  `evidence_tier=checkpoint_loaded` (no fallback), plus a fail-closed check for a missing checkpoint.
  This exercises the "model-based action selection runs on a smoke scenario" acceptance criterion of
  Issue #4013 end to end (previously only the predictor `predict()` path and metadata registration were
  tested). Claim boundary: diagnostic-only path execution; not benchmark, navigation-quality, or
  paper/dissertation evidence. The paired 3-arm smoke comparison (learned vs `cv_prediction_mpc` vs a
  model-free baseline) and Phase 3 real-trajectory training remain open on #4013. No benchmark
  campaign, no SLURM/GPU submission, no paper/dissertation claim edits.
* **issue #4013 trained short-horizon pedestrian predictor (diagnostic).** New
  `robot_sf/planner/learned_short_horizon_trainer.py` plus
  `scripts/training/train_learned_short_horizon_predictor_issue_4013.py`
  (`configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml`) train the small
  state-based predictor from `robot_sf/planner/learned_short_horizon_predictor.py` on a seeded
  synthetic robot-repulsion residual task and publish a checkpoint, training manifest, and metrics.
  The trainer reuses the predictor's own architecture and feature encoding (newly exposed as
  `build_predictor_module`, `predictor_io_dims`, `encode_predictor_features`, `pedestrian_world_state`)
  so the checkpoint loads without shape drift. Loading the checkpoint yields
  `evidence_tier=checkpoint_loaded` (not `diagnostic_untrained_smoke`), unblocking the "predictor
  trained; model-based action selection runs" acceptance criterion of #4013. Local CPU run (seed
  4013, 512 samples, 400 epochs): training loss `0.0987 -> 0.00044`. Claim boundary: the synthetic
  task is a reproducible learnability probe, not real pedestrian data; `smoke evidence` only, not
  benchmark, navigation-quality, or paper/dissertation evidence. No benchmark campaign, no SLURM/GPU
  submission, no paper/dissertation claim edits. See `tests/planner/test_learned_short_horizon_trainer.py`.
* **issue #1489 hybrid-learning synthesis recommendation builder.**
  `robot_sf/benchmark/hybrid_evidence_matrix.py` now exposes
  `build_hybrid_synthesis_report()` / `build_hybrid_synthesis_report_file()`, the synthesis-
  deliverable half of the #1489 contract. Given the existing prerequisite/status matrix, it emits an
  explicit per-mechanism recommendation (`continue`/`revise`/`stop`/`gather_more_evidence`) while
  staying fail-closed: a `continue`/`revise` verdict is *promoted* (marked authoritative via
  `synthesis_verdict_promoted`) only when the synthesis gate is open (≥2 durable `complete` lanes and
  no invalid rows), so no single pre-result lane, launch packet, smoke run, fallback/degraded row, or
  invalid row can promote a synthesis verdict. Terminal `stop` decisions on an executed
  stress/full-matrix slice are surfaced but never promoted. Focused tests cover the blocked, single-
  complete, two-complete (gate opens), terminal-stop, invalid-row, and missing-lane paths
  (`tests/benchmark/test_hybrid_synthesis_report.py`, 8 pass). This advances the #1489 acceptance
  criterion *"synthesis recommends continue/revise/stop/gather-more-evidence for each mechanism"* by
  providing the machinery that emits it; the issue stays open (`Refs #1489`), blocked on component
  campaigns producing ≥2 durable comparable `complete` lanes (#1470/#1472/#1474/#1475/#1358). No
  benchmark campaign run, no SLURM/GPU submission, no paper/dissertation claim edits.
* **issue #1489 synthesis-report CLI + first durable artifact.**
  `scripts/validation/validate_hybrid_evidence_matrix.py` now accepts `--synthesis-report`, exposing
  the #4628 `build_hybrid_synthesis_report()` machinery through a reproducible command (previously it
  was Python/tests-only). The synthesis report now also echoes `rows_valid` so a consumer can
  distinguish a fail-closed "blocked but valid" gate from a matrix whose rows failed validation. The
  flag was run once on the committed component matrix
  (`docs/context/evidence/issue_2274_hybrid_component_matrix_2026-06-05/matrix.yaml`) to emit the
  first durable synthesis-report artifact,
  `docs/context/evidence/issue_1489_synthesis_report_2026-07-06/synthesis_report.json`: `status:
  blocked`, all five mechanisms `gather_more_evidence`, `promoted_verdict_count: 0` — the correct
  conservative fail-closed result while component campaigns remain incomplete. Focused CLI + echo
  tests added to `tests/benchmark/test_hybrid_synthesis_report.py`. Issue stays open (`Refs #1489`),
  still blocked on ≥2 durable comparable `complete` component lanes. No benchmark campaign run, no
  SLURM/GPU submission, no paper/dissertation claim edits.

### Fixed

* **issue #5027 fresh-worktree `.venv` symlink was untracked.** The `.gitignore` virtual-environment
  rules (`.venv/`, `venv/`) used a trailing slash, which matches directories but not symlinks, so a
  linked worktree that points `.venv` at the main checkout's virtualenv via a symlink (a shared-venv
  runner setup) showed up as untracked (`?? .venv`). Dropped the trailing slash (`.venv`, `venv`) so
  the ignore rule covers directory, symlink, and file forms. Guarded by
  `tests/dev/test_gitignore_venv_symlink.py`. No behavior change for the common real-directory `.venv`.
* **issue #4919 SNQI aggregate diagnostic-mode logging regression from exception narrowing.** The
  `robot_sf/benchmark/aggregate.py::_ensure_snqi` exception handler, narrowed from a bare
  `except Exception:` to `except (ValueError, TypeError):` by #4887's broad-except ratchet, dropped
  the diagnostic-mode logging contract: in `observation_track_mode="diagnostic-cross-track"`,
  `KeyError` (missing metric keys) and `AttributeError` (the "unexpected SNQI failure" class) from
  `snqi_fn` now propagated uncaught instead of being logged and swallowed, failing
  `test_aggregate_snqi_recompute.py::test_compute_aggregates_logs_key_error_snqi_failure_in_diagnostic_mode`
  and `..._logs_unexpected_snqi_failure_in_diagnostic_mode`. The handler now catches the explicit,
  finite tuple `(ValueError, TypeError, KeyError, AttributeError)` — diagnostic mode logs and
  continues, strict mode logs and re-raises — without re-widening to a broad `except Exception:`.
  Acceptance met: both named tests pass; the broad-exception ratchet baseline stays at 217 and
  `ruff check --select BLE001` on `robot_sf/benchmark/` remains clean (a named tuple is not a
  broad-except site). Refs #4887, #4900.
* **issue #4908 diagnostic CLI `--help` omits the canonical `uv run` invocation.** The seven
  developer-facing diagnostic CLIs under `scripts/tools/` (`validate_scenario`,
  `validate_socnav_map_batch`, `validate_experiment_registry`, `validate_report`,
  `check_artifact_root`, `preflight_scenario_perturbations`, `preflight_adversarial_package_b`) now
  end `--help` with an `Example:` block showing the copy-pasteable
  `uv run python scripts/tools/<name>.py <required-arg>` command, so a new contributor no longer has
  to guess the invocation (the bare `python scripts/tools/<name>.py ...` path still dies with
  `ModuleNotFoundError: No module named 'scripts'` because these tools import `scripts.tools.*`).
  Help-text only: each parser gained an `epilog` plus `RawDescriptionHelpFormatter`; no runtime
  behavior, exit codes, or non-`--help` output changed. New
  `tests/tools/test_diagnostic_cli_help.py` asserts each `--help` contains `Example:` and the exact
  canonical line.
* **issue #4896 three pre-existing failures in tests/dev/test_pr_ready_preflight.py on main.**
  PR #4865 ("allowlist triplication") made `scripts/dev/pr_ready_check.sh` hard-require
  `tests/support/optional_test_allowlist.txt` — `is_optional_readiness_path` reads it to classify
  each changed test path (`tests/planner/` → optional lane, `tests/unit/` → core lane) and exits `1`
  when the file is absent — but the preflight test's fake-repo fixture never provided that file, so
  three lane/base-ref tests tripped the hard error. Root cause for all three is the same stale
  fixture, not broken preflight code: the hard-error is correct production behavior (a missing
  allowlist on a real repo would silently misclassify optional tests as core, defeating the lane
  split). Fixed by having `_make_fake_scripts` copy the real allowlist into the fake repo so the
  lane-detection assertions stay faithful to production. No production code changed; no test deleted
  or skipped. Acceptance bar met: full `tests/dev/` green (644 pass).
* **issue #4895 SNQI weights `git_sha` provenance always recorded as `"unknown"`.** The
  `recompute_snqi_weights` SHA lookup passed BOTH `capture_output=True` and `stderr=DEVNULL` to
  `subprocess.run` — an invalid combination that raises `ValueError` at call time, which the
  provenance handler swallowed, so every weight config produced via the recompute path silently
  recorded `git_sha: "unknown"` even inside a real git checkout. Removed the incompatible
  `stderr=DEVNULL` (`capture_output=True` already captures stderr); the output contract is preserved
  (real short SHA when git is available, `"unknown"` only on genuine failure such as no repo/timeout).
  Regression tests pin both the in-repo resolution and the out-of-repo fallback, plus a static guard
  against reintroducing the invalid argument combination.
* **issue #4873 user-facing docs drift audit — corrected stale claims against current code (docs-only).**
  Verified `README.md` and `docs/*.md` top-level docs against the current codebase and fixed incorrect
  statements in place across 13 files: renamed module path `robot_sf/sim/FastPysfWrapper.py` →
  `fast_pysf_wrapper.py` (SUBTREE_MIGRATION, dev_guide); fixed the `02_trained_model.py` JSONL output path
  `output/benchmarks/` → `output/results/` and the `tests/test_gym_env.py` → `tests/test_gymnasium_env_contracts.py`
  example (dev_guide); corrected the `robot_sf_bench` CLI subcommand count 15 → 30 top-level commands (docs/README); corrected the
  benchmark scenario-schema location (`schema/` singular holds `scenarios.schema.json`, `schemas/` holds the
  episode schema); fixed OSM API examples to real kwargs (`osm_to_map_definition`/`render_osm_background`),
  `MapDefinition.bounds` (not `map_bounds`), and removed the non-existent `zones_config` `RobotSimulationConfig`
  kwarg (osm_map_workflow); corrected coverage config (`[tool.pytest.ini_options] addopts` has no `--cov*`;
  `[tool.coverage.run] source` includes `fast-pysf/pysocialforce`; real gym-env test path) (coverage_guide);
  corrected `research_reporting` CLI flags to the real `generate_report.py`/`compare_ablations.py` options;
  updated `security_triage` (Ruff `S` is a per-code baseline under issue #3477, not a global `S` ignore); fixed
  `imitation_learning_pipeline` (`best_checkpoint_metric` default is `success_rate`; spec link is `spec.md`);
  fixed `single_pedestrians` example link to `examples/advanced/07_single_pedestrian.py`; fixed
  `snqi_weight_cli_updates` validation rule (`>= 0`, not `> 0` — a weight of 0 disables a term); fixed
  `trajectory_visualization` test invocation; and removed `ffmpeg` from the documented headless-CI apt list
  (CI installs only `libglib2.0-0 libgl1 fonts-dejavu-core jq`). Illustrative sample-output filenames, tutorial
  placeholders, and historical migration/fix logs were intentionally left unchanged. No code changes — docs only.
* **issue #4183 hybrid_global_rl diagnostic runner — executable baseline arm.**
  The paired diagnostic runner `scripts/benchmark/run_hybrid_global_rl_diagnostic_issue_4183.py` now
  (a) validates episodes against the canonical `robot_sf/benchmark/schemas/episode.schema.v1.json`
  instead of a stale strict schema that rejected native PPO episode records with
  `additionalProperties` errors, and (b) hydrates the benchmark-promoted learned checkpoint from its
  public GitHub release before preflight (`--no-hydrate` to skip). The #4183 preflight
  (`robot_sf/benchmark/hybrid_global_rl_diagnostic.py`) also recognizes a hydrated release asset
  whose cached file name (`<model_id>-model.zip`) differs from the registry `local_path`
  (`model.zip`), so a present-and-loadable checkpoint no longer reports
  `blocked_missing_learned_checkpoint`. With these fixes the unconditioned (baseline) arm produces
  native episode rows (3/3 seeds) and the packet advances from `blocked_no_valid_episode_rows` to
  `completed_with_fail_closed_exclusions`. Diagnostic-tier only; the route-conditioned arm remains
  fail-closed because the benchmark observation carries no `occupancy_grid` channel for the
  grid-route waypoint provider, so route-conditioned effect evidence stays blocked (see #4183).
  New regression tests: `tests/benchmark/test_hybrid_global_rl_diagnostic_issue_4183.py` and
  `tests/benchmark/test_issue_4183_paired_runner.py` (30 pass). No benchmark campaign, no SLURM/GPU
  submission, no paper/dissertation claim edits.
* **issue #1126 SDD curation decision packet — runnable import command.**
  `scripts/tools/sdd_curation_preflight.py::build_decision_packet` now emits an `import` handoff
  command that actually parses against the canonical importer
  `scripts/tools/import_sdd_scenarios.py`: it uses the importer's real flags `--annotations` and
  `--out-dir` (previously `--annotation`/`--output-dir`, which the importer rejects with exit `2`)
  and includes the *required* `--meters-per-pixel` scale assumption (previously omitted). A new
  `--decision-meters-per-pixel` CLI flag records the scene scale in `curation_parameters`
  (`meters_per_pixel`); when unset the command carries an explicit `<meters-per-pixel>` placeholder
  the curator must fill from the selected scene's calibration. Regression tests parse the generated
  command with the importer's own parser so the handoff cannot silently drift again
  (`tests/tools/test_sdd_curation_preflight.py`, 15 pass; 75 pass across the SDD suite). This closes
  a gap in an acceptance criterion of #1126 (record import command + scale assumptions); the issue
  itself stays open, blocked on BYO real SDD annotation staging (raw-data + checksum/license
  provenance). No real SDD data touched, no benchmark campaign, no SLURM/GPU submission, no
  paper/dissertation claim edits.

### Changed

* Added the issue #4018 density-curriculum closure audit and matched 96-timestep CPU diagnostic
  smoke evidence bundle. The note maps acceptance criteria to merged PRs #4169, #4478, and #4580,
  records readiness status `ready_diagnostic_smoke`, and keeps the claim boundary diagnostic-only:
  not benchmark evidence, not a training-result quality claim, and not a paper or dissertation
  claim.
* Recorded the **issue #2918 closure audit** at
  `docs/context/evidence/issue_2918_closure_audit.md` (linked from the #2918 preflight context
  note). It maps each acceptance criterion to its merged PR (#3754 staging/preflight contract,
  Issue #4566 fixture extraction pipeline + CLI) and to a reproduced validation run (23 focused tests
  pass; fixture smoke emits bounded proxy-placeholder priors; the manifest checker fails closed
  with `contract_status: blocked` and `manage_external_data.py list` shows all external datasets
  `missing`/`incomplete`). The agent-executable slice is complete; the only residual — a
  dataset-backed prior smoke from real staged trajectories — is gated on a license-compatible
  external dataset the project does not hold and stays tracked by #3065/#2657/#1498. Docs-only; no
  external-data ingest, no calibrated/representative prior claim, no benchmark/SLURM run.
* Recorded the **issue #4437 closure audit** at
  `docs/context/evidence/issue_4437_closure_audit_2026-07-06.md` (indexed in
  `docs/context/catalog.yaml`). It maps each acceptance criterion of the closure-audit hygiene lane
  to merged-PR evidence: the read-only candidate/classification tool (#4440
  `open_issue_closure_audit.py`), the comment templates + dry-run mechanics (#4503
  `closure_mechanics.py`), and the human-gated close path requiring both `--close-issues` and
  `--apply` (#4571). The **enablement tooling is complete and validated** (24 focused tests pass; a
  live read-only run fails closed with a schema-valid packet on the GitHub search rate-limit), but
  two acceptance criteria remain open — the audit **execution** write pass and the final #4437
  **summary comment** — both requiring GitHub issue comment/close authority. The audit therefore
  keeps #4437 open with a dispatchable residual checklist (`Refs #4437`). Docs-only; no queue edits,
  new issues, benchmark execution, Slurm/GPU submission, or research claim.
* Recorded the **issue #1126 SDD-curation closure audit** at
  `docs/context/evidence/issue_1126_closure_audit_2026-07-06.md` (linked from the #1126 context
  note). It maps each acceptance criterion to merged-PR evidence (#1091 importer, #3765 fail-closed
  preflight, #4564 decision packet, plus this PR's import-command fix) and to freshly reproduced
  fail-closed validation, then records the closure decision: **keep open**, blocked on BYO licensed
  SDD annotation staging (the only remaining criteria require real staged data, which does not exist
  locally). Docs + support-tooling only.
* Recorded the **issue #1456 closure audit** at
  `docs/context/evidence/issue_1456_closure_audit.md` (registered in `docs/context/catalog.yaml`).
  It maps every acceptance criterion — original issue body plus the appended `agent-exec-spec:v1`
  slice — to merged-PR evidence (#1924 external-data assistant, #2400 status note, #1596 row policy,
  Issue #3755 fail-closed readiness vs placeholder shells, #4526 tightened `socnavbench-control` contract)
  and to a reproduced fail-closed validation (`manage_external_data.py --json check
  socnavbench-control` / `socnavbench-s3dis-eth` exit `2`, `prepare_socnav_assets.py` exit `2`
  `MISSING_REQUIRED_ASSETS`, 43 focused socnav map/asset tests pass). Decision: **keep #1456 open,
  `state:blocked-external-input`** — all agent-executable tooling/inventory/row-policy criteria are
  met; the three core asset criteria remain blocked on maintainer-staged licensed external data
  (not compute-gated, so `COMPLETE-FIRST` does not apply). Docs-only; no asset staging, benchmark
  run, SLURM submission, or research claim.
* Recorded the **issue #1470 closure audit** at
  `docs/context/evidence/issue_1470_closure_audit_2026-07-06.md` (linked from the #1397 launch
  note). It maps each acceptance criterion for the oracle-imitation dataset-collection lane to its
  merged/closed evidence (#1469 launch packet + validator, #2441 completed Slurm collection, #2989
  durable git-tracked trace bundle) and to a reproduced validation run
  (`validate_oracle_imitation_launch_packet.py` → `status=valid`, 6 scenarios, 12 episodes; 54
  focused imitation tests pass). The lane's trace-collection scope is complete (closeout state
  `dataset_ready`); the residual durable trace-URI registry (`training_ready=true`) is owned by
  Issue #2655 and the imitation-training benchmark by #1496, both out of #1470's scope. Docs-only; no
  trace/NPZ materialization, training run, Slurm/GPU submission, or research claim.
* Recorded the **issue #2312 closure audit** at
  `docs/context/evidence/issue_2312_closure_audit.md` (linked from the #2312 context note). It maps
  each acceptance criterion to its merged PR (#3762 manifest+validator, #3772 #1472 campaign gate,
  Issue #4549 status packet) and to a reproduced fail-closed validation run
  (`validate_learned_risk_trace_manifest.py --status-json` exits `3` /
  `artifact_retrieval_blocked`; 24 focused tests pass). The agent-executable slice is complete; the
  only residual — a resolvable baseline/trace URI with `retrieval_status: available` — is
  compute-gated on the #1472 / #2441 SLURM run and stays tracked by #1472. Docs-only; no trace
  materialization, training run, SLURM submission, or research claim.
* Added `--pin-report-json` to `scripts/tools/generate_socnavbench_traversible.py` for issue #4291.
  The generator can now write a maintainer-review sidecar with `expected_tree_sha256`,
  registry-owner, and no-raw-data guardrails after `traversibles/ETH/data.pkl` exists, keeping the
  trusted registry pin step explicit without committing generated SocNavBench data.
* Added a compact **learned-risk trace status packet** for issue #2312. The existing
  `scripts/validation/validate_learned_risk_trace_manifest.py` validator now supports
  `--status-json`, and `robot_sf/training/learned_risk_trace_manifest.py` exposes
  `build_trace_manifest_status_packet()` so #1472 handoff/state propagation can reuse the same
  fail-closed manifest decision without re-reading or reinterpreting the manifest. This is
  data-preflight status only: no trace materialization, external-data copy, training run, SLURM
  (Simple Linux Utility for Resource Management) submission, or research claim.
* Added **typed collision-event export records** to new episode rows (issue #4454). The canonical
  `event_ledger` block is now `EpisodeEventLedger.v2` and can carry per-collision records with
  partner type/id, collision time, relative contact speed, clearance-series provenance, and exact
  event provenance. `run_map_episode` now emits the ledger natively before JSONL validation, so the
  camera-ready retention path keeps the typed events automatically. Behavioral note: the scalar
  `collision` / `outcome.collision_event` flag remains a termination indicator; safety claims should
  cite the typed `event_ledger.collision_events` records instead of inferring collisions from the
  scalar flag alone.
* **Vectorized the O(N²) pairwise pedestrian-pedestrian social-force contribution path** used by the
  opt-in `hsfm_anisotropic_fov_v1` runtime seam (issue #3481). `pairwise_social_force_contributions`
  (`robot_sf/sim/pedestrian_model_variants.py`) previously built its `(N, N, 2)` per-pair matrix with
  a Python double loop calling the PySocialForce njit kernel one pair at a time; it now evaluates the
  whole matrix in closed NumPy form via the new pure helper `_pairwise_social_force_kernel`, matching
  the scalar kernel pair-by-pair to ~5e-15 (aggregate sum unchanged to ~7e-15, well under the existing
  `rtol=1e-9` contract) and running ~14–23× faster for N=50–200 in single-CPU diagnostic timing. The
  activation-threshold mask, zero-vector/`norm_vec` handling, and diagonal exclusion are preserved
  exactly, and the module stays numpy-pure at import (the deferred numba import is no longer needed on
  this path). Behavior-preserving; default pedestrian models are untouched. Evidence tier remains
  diagnostic/prototype — no benchmark campaign, Slurm/GPU submission, calibrated-realism, planner
  ranking, or paper/dissertation claim. New equivalence, degenerate-pair, and N=256 scale tests in
  `tests/sim/test_hsfm_fov_pairwise_isolation.py`.

### Added

* Added a **campaign arm checkpoint preflight** for issue #4613 so a missing or corrupt policy
  checkpoint fails in seconds on the submit node instead of ~14h into compute (the S30 campaign
  jobs 13296 and 13301 both failed identically on a missing PPO `output/model_cache` checkpoint).
  `robot_sf/benchmark/campaign_checkpoint_preflight.py` inspects every enabled arm's `algo_config`
  for `model_id` / `model_path` checkpoint references (recursively, covering nested prior policies)
  and fails closed naming the arm. It runs in two modes: a cheap network-free resolvability guard
  (present locally OR a durable remote source to stage from) now wired into
  `prepare_campaign_preflight` before scenarios load, and an enforced pre-sbatch staging mode that
  downloads and checksum-verifies each registry checkpoint into the durable cache. Ops runs the new
  `scripts/benchmark/preflight_campaign_checkpoints.py --config <campaign> [--stage]` before `sbatch`
  (exit `0` resolvable/staged, `2` config error, `3` unresolvable → do not submit). Provisioning
  preflight only: it runs no benchmark, submits no Slurm job, and is not benchmark evidence.
* Added the **enforced-staged submit-time checkpoint gate** for issue #4613 (follow-up #4663):
  `prepare_campaign_preflight()` now accepts `checkpoint_preflight_mode` (`metadata_only` default
  vs `enforced_staged`), `checkpoint_cache_dir`, and `checkpoint_registry_path`. The enforced
  mode runs the existing cheap guard with `stage=True` so a remote-backed but locally-absent
  checkpoint is caught before `sbatch`, and a `submit_safe` boolean now reports whether the
  resolvability is sufficient for compute (false when any arm is only `stageable_remote`). The
  per-arm summary is persisted as `preflight/checkpoint_staging.json` (enforced) or
  `preflight/checkpoint_resolvability.json` (metadata-only) and embedded in the campaign manifest
  under `artifacts.preflight_checkpoint_provisioning`. New public pre-sbatch gate
  `scripts/benchmark/submit_camera_ready_checkpoint_gate.sh` runs the staging CLI with
  `--report-path` so the requeue packet records per-arm staging status.
  `scripts/tools/run_camera_ready_benchmark.py` gains `--checkpoint-preflight-mode`,
  `--checkpoint-cache-dir`, and `--checkpoint-registry-path` flags for the preflight-only path.
  The `preflight_campaign_checkpoints.py --json` output now includes `submit_safe` and supports
  `--report-path`. Provisioning only: no benchmark, Slurm, or paper-facing claim.
* **issue #3481 — opt-in HSFM body-orientation alignment torque (`hsfm_alignment_torque_v1`).**
  A new pedestrian-model selector that decouples pedestrian body orientation `phi` from the
  instantaneous total-force direction: instead of snapping the heading each step (as
  `hsfm_total_force_v1` does), the orientation relaxes toward the desired direction via a damped
  second-order alignment torque (`k_theta` stiffness, `k_omega` damping, bounded turn rate). Adds the
  pure helpers `wrap_to_pi` and `step_alignment_torque_heading` in
  `robot_sf/sim/pedestrian_model_variants.py`, an `AlignmentTorqueConfig` opt-in surface on
  `SimulationSettings` (validated fail-closed), scenario-config selection, and simulator-seam
  angular-velocity state. Default pedestrian model and all prior selectors are unchanged. This
  delivers the maintainer-named remaining code prerequisite for #3481 (the HSFM
  heading-state/alignment-torque Definition-of-Done bullet); evidence tier stays
  diagnostic/prototype — no calibrated-realism, benchmark, or paper claim. See
  `docs/context/issue_3481_hsfm_alignment_torque.md`.
* Extended the **Package C readiness helper** so issue #4547 can commit the real rerun artifact from
  the retained #2916 coupling report. `scripts/tools/prediction_package_c_readiness.py` now accepts
  `--output-markdown` alongside `--output-json`, which lets the cheap local lane write the durable
  JSON + Markdown readiness bundle directly under `docs/context/evidence/` instead of relying on
  shell redirection around stdout. No campaign, Slurm, or paper-facing claim was added; this is
  coordination-proof only.
* Added **metadata-only public-source discovery ledger checker** for real micromobility trace
  collection issue #3278. New schema `real_trace_source_discovery.v1`, module
  `robot_sf.analysis_workbench.real_trace_source_discovery`, CLI
  `scripts/tools/check_real_trace_source_discovery.py`, and example ledger
  `configs/benchmarks/issue_3278_real_trace_source_discovery_example.yaml` record candidate public
  sources searched for `late_evasive_reaction` and `oscillatory_local_control` validation traces.
  The checker fails closed until every required target has an available, accepted/permissive,
  directly covering source. It stores no raw external data, downloads nothing, and makes no
  real-world validation, calibration, benchmark, or paper-facing claim.

* **Wired the CrowdBot and SCAND external datasets into the external-data setup nav and completed
  their registry traceability** (issue #4357, #4224 program). `docs/external_data_setup.md` now lists
  and tabulates `crowdbot` and `scand-demos` alongside their program siblings (SDD, ETH/UCY, ATC,
  inD) with links to the existing `docs/datasets/crowdbot.md` / `docs/datasets/scand.md` acquisition
  pages, and the two `scripts/tools/manage_external_data.py` registry entries plus the
  `docs/context/issue_4224_external_dataset_registry.md` rows now cite the delivering issue #4357 and
  program parent #4224. Registry/docs discoverability only — no dataset bytes, no automated
  acquisition, and no new loader/benchmark/paper claim (the shape-contract loaders landed earlier in
  Issue #4346). Both assets still fail closed with `status: missing` when unstaged.
* Extended the **issue #3207 fidelity-sensitivity campaign runner to consume the fixed-scope preflight
  plan** (`scripts/benchmark/run_fidelity_sensitivity_campaign.py`). A new `--fixed-scope-plan-only`
  mode builds the preflight packet and enumerates the concrete full run plan — every
  planner_group × axis-variant × seed run cell (the shipped config materializes 108 cells/scenario) —
  with each cell carrying its resolved `algorithm_readiness` catalog algorithm and opt-in/tier state.
  It runs **no** episode and promotes no claim; actual launch stays **fail-closed** behind the
  preflight's unmet launch prerequisites (ORCA/rvo2 runtime dependency, hybrid-rule explicit opt-in,
  runner-not-yet-wired for the full planner set, and the post-run rank-identifiability recheck). A
  `--require-launchable` flag exits non-zero while any gate remains, and `ensure_fixed_scope_launchable`
  raises `FixedScopeNotLaunchableError` so no full campaign can launch against unresolved prerequisites.
  The enumerated plan JSON is written to gitignored `output/`; the bounded two-planner slice runner
  behavior is unchanged. Not benchmark, simulator-realism, sim-to-real, or paper-facing evidence.
* Added a **full fixed-scope fidelity-sensitivity preflight** for issue #3207 that pre-registers the
  simulator-fidelity sensitivity campaign before launch (`robot_sf/benchmark/fidelity_fixed_scope_preflight.py`,
  CLI `scripts/benchmark/preflight_fidelity_fixed_scope.py`). It materializes the explicit run plan
  (planner groups × axis variants × seeds over the fixed scenario set), resolves every planner group
  against the canonical `algorithm_readiness` catalog and **fails closed** on unavailable/placeholder
  planners, and validates the primary (ranking) metric is identifiable by contract (failing closed on
  metrics pre-declared non-identifiable/zero-variance per #3299). It emits a launch/readiness packet
  only — `preflight_ready` is a pre-registration gate, not execution-ready, and not benchmark,
  simulator-realism, sim-to-real, or paper-facing evidence. The shipped
  `configs/research/fidelity_sensitivity_v1.yaml` now carries an explicit
  `fixed_scope.planner_algorithms` binding so its planner labels resolve. No benchmark campaign was
  run.
* Added a **trace-capable h600 re-run pre-registration contract and fail-closed validator** for the
  issue #4206 failure-mechanism cross-cut (#4206). PR #4341 proved the retained h600 runs (jobs
  13268 confirm, 13273 extended roster) predate the trace-capable episode exporter (#4301), so their
  failure-mechanism labels are all `not_derivable` and a sidecar backfill cannot recover them — the
  only remaining CPU-dispatchable work is a *contract* for the eventual re-run, not another blocked
  analysis packet. `configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml`
  declares the re-run's required `failure_mechanism_taxonomy.v1` and `interaction_exposure.v1`
  outputs, trace-capture switches, planner roster (grouped by structural class), frozen seed
  schedule, predecessor-run provenance, downstream consumer, and fail-closed exclusions.
  `scripts/validation/check_issue_4206_trace_capable_h600_rerun_preregistration.py` validates the
  contract fail-closed and, critically, cross-checks the declared required-field lists against the
  canonical schema owners (`robot_sf/benchmark/failure_mechanism_taxonomy.py`,
  `robot_sf/benchmark/interaction_exposure.py`) so the contract cannot silently drop a claim-bearing
  field or drift from the schema; it also rejects geometry-bucket substitution, an all-`not_derivable`
  re-run counting as success, an empty roster/seed schedule, duplicate planner keys, and in-PR
  submission. Pre-registration only: runs no episodes, submits no campaign, and derives no mechanism
  label. Validated with `tests/validation/test_issue_4206_trace_capable_h600_rerun_preregistration.py`
  (16 fixture cases).
* Wired **per-pair pedestrian-pedestrian forces into the opt-in `hsfm_anisotropic_fov_v1` simulator
  runtime** (#3481). The field-of-view (FoV) pedestrian model now consumes per-pair social-force
  contributions instead of the coarse aggregate `np.min` attenuation. Two pure helpers were added to
  `robot_sf/sim/pedestrian_model_variants.py`: `pairwise_social_force_contributions(...)` builds the
  `(N, N, 2)` per-pair social-force matrix by reusing PySocialForce's own `social_force_ped_ped`
  kernel (so summing over neighbors reproduces the aggregate `SocialForce()` the engine already
  computes), and `fov_attenuated_total_force(...)` isolates the ped-ped social term from the total
  force and replaces it with its per-pair FoV-attenuated form. `Simulator._step_pedestrians`
  (inherited by `PedSimulator`) now reads the live `SocialForce` component parameters (fail-closed if
  absent), builds the per-pair matrix from the current PySocialForce state, and attenuates each
  neighbor's push by its own FoV weight before HSFM stepping. Only the `hsfm_anisotropic_fov_v1` path
  changed: a rear neighbor is now down-weighted without disturbing an in-cone neighbor or the actor's
  goal/obstacle drive, whereas the previous path scaled the *entire* per-actor force by a single
  weight. `social_force_default`, `hsfm_total_force_v1`, and `hsfm_ttc_predictive_v1` are unchanged;
  the aggregate helper `anisotropic_fov_total_force` is retained as the reference contrast. Evidence
  tier stays diagnostic/prototype — **no** calibrated-realism, benchmark-strength, or
  paper/dissertation claim. Vectorizing the `O(N^2)` per-pair matrix and narrow-passage / bottleneck
  benchmark evidence remain follow-ups. Context note:
  `docs/context/issue_3481_hsfm_ttc_predictive_forces.md`.

* Extended the **AMMV feasibility batch-summary artifact block** with the claim-boundary markers it
  was missing, so a consumer reading only the summary sees the same non-hardware boundary as the
  per-episode payload (#3466). `robot_sf/benchmark/map_runner_batch_summary.py` now emits
  `algorithm_metadata_contract.ammv_feasibility` via the new pure helper
  `build_ammv_feasibility_summary`, which adds `evidence_kind: "diagnostic_proxy"` and a
  `status` field (`"available"` / `"no_ammv_episodes"`) alongside the existing
  `proxy_kind: "internal_non_hardware"` and the four folded fields (`min_stability_margin`,
  `tip_over_violation`, `n_curvature_violations`, `feasible`). Folds stay worst-case (minimum
  margin, OR tip-over, all-episodes-feasible), so a single tip-over-prone episode is never averaged
  away. Additive and backward-compatible; internal proxy only — **no** hardware-calibrated AMMV
  safety claim. Refreshed `docs/context/issue_3466_ammv_command_feasibility.md` (the note previously
  described the artifact wiring as deferred; it landed in #3845).

* Added a **cross-module pipeline contract test and consolidation note for the issue #4142 dense
  DPCBF comparison** (#4142). The dense-comparison pipeline landed as separate slices — readiness
  (#4299), the packet-consuming run planner (#4318), and the plan-consuming summarizer (#4345) —
  each with its own module, schema constant, and focused test, but no single test drove all three
  top-level entry points as one unit or pinned the invariants that must hold *across* the slices.
  `tests/benchmark/test_issue_4142_dpcbf_dense_pipeline_contract.py` closes that gap: it chains
  `evaluate_readiness` → `build_run_plan` → `summarize_dense_comparison` on the tracked packet and
  asserts the packet → plan → summary schema lineage
  (`robot_sf.issue_4142_dpcbf_dense_comparison{,_plan,_summary}.v1`), that the runner and summarizer
  reuse readiness's required-arms and fail-closed excluded-row-status *objects* (guarding against a
  re-hardcoded copy drifting), and that execution stays authorization-gated across the whole
  pipeline. Consolidation/state note `docs/context/issue_4142_dpcbf_dense_pipeline.md` records the
  now schema-complete pipeline, remaining/intentional blockers, and the one remaining empirical
  action (the authorized campaign). Diagnostic contract/regression guard only: runs no episodes,
  authorizes no campaign, submits no Slurm/GPU job, and makes no safety-performance,
  collision-reduction, or paper/dissertation claim.
* Added license-safe **CrowdBot and SCAND shape-contract loaders** plus skip-if-absent tests for the
  Issue #4224 external-data program (public slice b). `robot_sf/data/external/crowdbot.py` and
  `robot_sf/data/external/scand.py` only inspect locally staged files — they never download, vendor,
  or redistribute dataset bytes — and validate a cheap structural contract (documented
  recording + license/readme layout present; every staged CSV export non-empty and rectangular;
  every JSON export parseable; no empty ROS bag). Both share the reusable engine
  `robot_sf/data/external/recording_shape_contract.py`, which reuses the canonical registry
  availability contract in `scripts/tools/manage_external_data.py`. Acquisition/layout docs added at
  `docs/datasets/crowdbot.md` and `docs/datasets/scand.md`. Registry/contract only: no dataset
  bytes, no automated acquisition, no benchmark or paper/dissertation claim.

### Fixed

* Removed absolute, username-bearing local paths from the committed issue #4207
  certification-transfer evidence packet and hardened the guard against the class (#4324). The
  `config.path` / `gate_spec.path` provenance fields in
  `docs/context/evidence/issue_4207_interacting_smoke_2026-07/{metadata,summary}.json` had baked in
  the author's worktree path (`/home/<user>/git/robot_sf_ll7.worktrees/...`, same non-reproducibility
  defect as #4302/#4303) and are now repo-relative (`configs/benchmarks/...`); `SHA256SUMS` is
  regenerated. `robot_sf/benchmark/certification_transfer.py` now normalizes those fields at
  generation via `_repo_relative_path(...)`, so future packets stay portable. The
  `Check Configs For Absolute Home-Dir Paths` pre-commit hook
  (`hooks/check_config_abs_paths.py`) now also scans `docs/context/evidence/**`, so this class fails
  closed at commit time. Diagnostic-only evidence hygiene: no benchmark, metric, or paper/dissertation
  claim changes.
* Fixed the **LiCCA post-guard full-suite teardown hang** (#4216): the suite could reach 100% and
  then never exit because `tests/perf_utils/test_enforce_mode.py` spawned a nested `pytest` child
  with an unbounded `subprocess.run(...)` (its `@pytest.mark.timeout` marker is a no-op because
  `pytest-timeout` is not installed). On a shared GPU/HPC node that child can deadlock during
  CUDA/interpreter teardown or leave a descendant holding device handles, blocking the parent
  forever. Added `tests/support/process_teardown.py::run_bounded_subprocess`, which runs the child
  in its own process group and, on timeout, reaps the whole group (SIGTERM→SIGKILL) so descendants
  are terminated too — mirroring the termination pattern already used by
  `scripts/dev/run_compact_validation.py`. The nested-pytest call site is now bounded
  (`ROBOT_SF_NESTED_PYTEST_TIMEOUT`, default 180s) and a CPU-only regression test proves a
  long-lived descendant is reaped. No GitHub Actions behavior changes.

### Added

* Added a **plan-consuming result summarizer for the issue #4142 dense DPCBF comparison** (#4142):
  `robot_sf/benchmark/issue_4142_dpcbf_dense_summary.py` consumes the resolved three-arm run plan
  (from PR #4318's `build_run_plan`, the single source of truth for arms, output paths, and the
  fail-closed row-status exclusion) and reads each arm's per-episode JSONL output into a comparison
  summary under schema `robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1`. It closes the run
  planner's declared next gate ("a dense-comparison summarizer that consumes the per-arm outputs").
  It stays fail-closed: an invalid/blocked plan yields `plan_blocked` with no artifacts consumed; a
  missing, empty, or unparseable arm artifact keeps the comparison out of `complete`
  (`results_incomplete`), which is the expected state while execution stays authorization-gated (no
  arm output exists yet); and each row's status is classified against the plan's declared
  `excluded_row_statuses` (`fallback`, `degraded`, `failed`, `ineligible`) so excluded and
  unrecognized rows are counted as caveats, broken out by status, and never added to
  success-evidence counts. Adds the thin CLI
  `scripts/tools/summarize_issue_4142_dpcbf_dense_comparison.py`
  (`--format markdown|json`, `--fail-on-incomplete`), tests
  `tests/benchmark/test_issue_4142_dpcbf_dense_summary.py`, and context note
  `docs/context/issue_4142_dpcbf_dense_summary.md`. Runs no episodes, submits no Slurm/GPU job, and
  makes no safety-performance or collision-reduction claim.
* Taught the **issue #4206 mechanism-level policy-structure cross-cut builder** to consume the
  declared external failure-mechanism sidecar (the #4305 declared-sidecar path) and to fail closed
  with a precise new status `blocked_trace_labels_not_derivable_predates_trace_capture`
  (`scripts/validation/build_issue_4206_policy_structure_mechanism_crosscut.py`, config
  `configs/analysis/issue_4206_policy_structure_mechanism_crosscut.yaml`). The retained h600 confirm
  (13268) and extended-roster (13273) episode rows predate trace capture (#4301), so every label in
  the declared #4242 backfill sidecar
  (`docs/context/evidence/issue_3810_h600_interpretation_2026-07/h600_mechanism_labels_sidecar.csv`)
  is `not_derivable_missing_trace`. The builder now distinguishes "taxonomy applied but no trace was
  derivable → needs a trace-instrumented re-run" from the generic "add a sidecar" block, so the
  F-C4(ii) blocker names the correct next empirical action instead of pointing at a sidecar backfill
  that already produced only unknowns. Geometry-bucket substitution stays rejected; a declared
  sidecar carrying real trace labels unblocks the F-C4(ii) rank/probe tables. Diagnostic analysis
  support only: no benchmark campaign, Slurm/GPU submission, or paper/dissertation claim change.
* Added the **physics-verified interacting certification-transfer probe evidence** (#4327,
  following #4207 / #4315): a real CPU run of the interacting scenario family through the 4-arm
  certification-transfer probe, replacing the synthetic smoke fixture as the *empirical* diagnostic
  reference. New config `configs/benchmarks/issue_4207_interacting_physics_probe.yaml` raises the
  probe horizon to the scenario's own `max_episode_steps` (400); at the smoke config's horizon 60
  the robot cannot traverse the blind-corner L-route, so the real run stayed non_interacting (robot
  ~24 m from the pedestrian, `robot_ped_within_5m_frac = 0`). At horizon 400 the route-following
  `goal` baseline reaches the corner and makes near-field contact
  (`robot_ped_within_5m_frac = 0.106`, `min_clearance_m = -0.024`, a collision), so
  `model_sensitivity_exercised = true` is backed by a real interacting cell for the first time.
  Evidence packet `docs/context/evidence/issue_4207_interacting_physics_2026-07/` (diagnostic tier,
  CPU-only). **Caveats:** the learned arms run without checkpoints in goal/sampling fallback and
  never reach the pedestrian (their gate statuses are vacuous w.r.t. certification); and the
  `social_force_default` / `hsfm_total_force_v1` cells are byte-identical, so `flip_cases = 0` —
  the synthetic fixture's fabricated `ppo` flip does not reproduce under physics. The #4315 synthetic
  packet is unchanged and remains tooling-validation only. The runner
  `scripts/benchmark/run_certification_transfer_issue_4207.py` now records repo-relative (portable)
  config/gate-spec provenance paths.
* Documented the **runtime uncertainty-triggered fallback** for guarded PPO (#3974), which merged in
  Issue #4193 without a CHANGELOG entry. This is the successor slice the earlier #4138 entry below called
  out as "not included"; that statement is now stale — the fallback ships in
  `robot_sf/planner/guarded_ppo.py`. When `uncertainty_fallback_enabled` is set (default off), the
  `GuardedPPOAdapter` guard can override a PPO command that is nominally clearance-safe but intrudes
  into conformal uncertainty buffers (reusing #4138's `compute_intrusion_metrics`), crosses a
  low-time-to-collision (TTC, the time until the robot and a pedestrian would collide at current
  relative velocity) threshold, or crosses a diagnostic predicted-collision-probability proxy. The
  override emits one of three decision labels — `uncertainty_fallback_stop`,
  `uncertainty_fallback_slow_down`, or `uncertainty_fallback_configured` (delegates to the configured
  fallback adapter, e.g. ORCA) —
  and `robot_sf/benchmark/map_runner.py` counts each in `guard_stats`. The probability value is an
  explicitly labeled **diagnostic proxy**, not a calibrated probability: shield metadata carries
  `claim_boundary: diagnostic_proxy_not_safety_guarantee`. Diagnostic-only, default-off; no safety
  guarantee, no benchmark claim, no paper-facing claim. Validation:
  `uv run pytest tests/ -k "conformal or intrusion or fallback_trigger" -q`.
* Added a **named-candidate seed-sufficiency closure evaluator for the retained h600 roots** (#4328,
  toward #3556): `scripts/validation/evaluate_issue_4328_h600_seed_sufficiency_candidates.py` plus pure
  logic in the canonical screening owner (`robot_sf/benchmark/scenario_belief_screening.py`). It
  evaluates the three named h600 report roots proposed in issue #4328 against the #3556 ScenarioBelief
  seed-sufficiency closure contract — existence on the current host, the two analyzer-required report
  files, **and** a ScenarioBelief provenance/lineage gate — then runs the analyzer on the best fully
  compatible candidate or fails closed with an explicit per-candidate blocker. The provenance gate is
  the new capability over PR #4310's durable-root resolver: it prevents a foreign h600 roster
  campaign's seed-sufficiency (a different question) from being promoted as #3556 closure evidence.
  On a host without the roots the committed packet records `blocked_no_compatible_candidate` (all three
  candidates absent + provenance-incompatible) with a deferred queue-row request for a #3556-specific
  campaign. Diagnostic evidence-closure tooling only; no benchmark/paper-grade claim, no campaign run.
* Added a **skip-if-absent inD shape-contract loader** for issue #4224:
  `robot_sf/data/external/ind.py` inspects a locally staged inD copy (naturalistic road-user
  trajectories at German intersections, Bock et al. 2020) and validates the documented per-recording
  file group (`*_tracks.csv`, `*_tracksMeta.csv`, `*_recordingMeta.csv`, and a background image)
  without ever downloading or redistributing the request-gated dataset bytes. `is_available`,
  `require_available`, and `load_shape_contract` provide a cheap structural contract (non-empty,
  rectangular CSVs carrying inD's published header columns, with finite coordinate/id parsing) and
  return per-recording row/column shape metadata; no scene content, benchmark, or paper-facing claim
  is asserted. `tests/data/external/test_ind_shape.py` covers the absent (skip), synthetic-layout,
  multi-recording, background-fallback, and fail-closed cases. This is the public slice (b) of the
  maintainer external-data split for the `ind-crossings` asset (registry entry #4238, acquisition
  docs #4290); private-ops staging remains a follow-up. `docs/datasets/ind.md` documents the loader.
* Added a **license-safe ATC pedestrian-tracking loader + skip-if-absent shape-contract tests**
  for issue #4289 (follow-up to the #4224 external-data program): `robot_sf/data/external/atc.py`
  exposes `is_available`, `require_available`, and `load_shape_contract` over the canonical
  `atc-pedestrian` external-data registry entry. The loader only inspects locally staged files — it
  never downloads, vendors, or redistributes the license-gated ATC bytes. It reuses
  `manage_external_data.check_asset` for presence (one daily CSV plus a local terms/README note) and
  validates each staged daily CSV as headerless, comma-delimited, exactly eight numeric columns wide;
  scanning is bounded by default (`max_rows=10000`, reported via `scan_truncated`) since a single ATC
  day can hold millions of samples. Malformed staged CSVs fail closed with `AtcDataError` pointing to
  `docs/datasets/atc.md`. Tests in `tests/data/external/test_atc_shape.py` skip cleanly when no local
  ATC copy is staged and otherwise run on synthetic fixtures. No dataset bytes, download code,
  benchmark consumer, campaign run, or paper-facing claim is introduced; follows the #4279
  (`socnavbench-s3dis-eth`) exemplar pattern.
* Retained **`min_clearance_m` and `proxemic_intrusion_rate`** in the camera-ready campaign
  summary/retention schema (issue #4326, from #4313). `robot_sf/benchmark/camera_ready/_reporting.py`
  now emits both fields per planner row, aggregated from per-episode values that already exist in
  episode rows: `min_clearance_m` is the campaign-wide **worst-case (minimum)** clearance (distinct
  from the mean-of-per-episode-minimums kept as `min_clearance_mean`), and `proxemic_intrusion_rate`
  is the mean per-episode personal-space intrusion fraction (`social_proxemic_intrusion_frac`). This
  makes the clearance/proxemic release-gate coverage gaps from #4313 evaluable for **future**
  campaigns with no evaluator code change (the gate spec already targets these exact names). Past and
  degraded campaigns are **not** backfilled — with no source values both fields fail closed to `nan`
  so their gates stay `not_evaluable`. No benchmark campaign was run; this is a schema/aggregation
  change only.
* Added **diff-scoped context-note freshness gating** for issue #3190. The docs-proof consistency
  checker (`scripts/validation/check_docs_proof_consistency.py`) gains a `--freshness-scope
  {repo,diff}` option: `repo` (default) preserves the existing repo-wide `--check-context-note-freshness`
  behavior, while `diff` restricts freshness findings to context notes changed against `--base` (plus
  catalog-driven `superseded_replacement`/`stale_current_dated` rules when `docs/context/catalog.yaml`
  itself changed). This makes the freshness check safe to run as a per-PR gate without failing on the
  pre-existing repo-wide backlog of stale/orphan notes. The diff wrapper
  (`scripts/dev/check_docs_proof_consistency_diff.sh`) now runs the freshness checker in diff scope
  when `DOCS_PROOF_CHECK_FRESHNESS=1` (opt-in, off by default; `DOCS_PROOF_FRESHNESS_STRICT=1` promotes
  stale/orphan warnings to failures). Superseded-without-replacement errors fail closed. No content was
  moved or archived; no benchmark/paper claims changed.

* Added a **packet-consuming run planner for the issue #4142 dense DPCBF comparison** (#4142):
  `robot_sf/benchmark/issue_4142_dpcbf_dense_runner.py` consumes the predeclared packet schema
  `robot_sf.issue_4142_dpcbf_dense_comparison.v1` and resolves it into an ordered, per-arm run plan
  (`cbf_off`, `cbf_collision_cone_on`, `cbf_dynamic_parabolic_v1_on`), closing the readiness surface's
  first downstream gate ("no packet-consuming runner is wired to this schema"). Each arm resolves to
  one benchmark job pinned to the shared `prediction_mpc_cv` algorithm, that arm's validated adapter
  config, the shared scenario manifest, and a per-arm output path, reusing the canonical readiness
  validator as the single source of truth (no parallel validation). It stays fail-closed: an invalid
  packet yields `prerequisites_incomplete` with no executable arm jobs, the fail-closed row-status
  exclusion (`fallback`, `degraded`, `failed`, `ineligible`) is carried verbatim into the resolved
  plan (`robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1`), and `execute_run_plan()` always fails
  closed because running the comparison requires explicit human/Slurm authorization. Adds the thin CLI
  `scripts/tools/run_issue_4142_dpcbf_dense_comparison.py` (dry-run default), tests
  `tests/benchmark/test_issue_4142_dpcbf_dense_runner.py`, and context note
  `docs/context/issue_4142_dpcbf_dense_runner.md`. Runs no episodes, submits no Slurm/GPU job, and
  makes no safety-performance or collision-reduction claim.
* Added a **consolidated failure-archive rerun closure packet** for issue #3275:
  `robot_sf/benchmark/failure_archive_rerun_closure.py` (`build_rerun_closure_packet`, schema
  `failure_archive_rerun_closure_packet.v1`) folds the accumulated rerun readiness/leakage guards into
  one durable verdict — a single `disposition` (`ready_for_rerun`, `fail_closed_blocked`, or
  `diagnostic_only`), the consolidated blocker list, and a deterministic `next_empirical_action`.
  `scripts/adversarial/produce_rerun_closure_packet.py` exposes it as a fail-closed CLI (exit codes
  `0`/`2`/`3`), and a missing or malformed archive fails closed instead of substituting a synthetic
  fixture. The closure packet adds no new gate; it composes the canonical pair gate
  `classify_failure_archive_rerun_readiness`. Running it on the two real smoke archives (`issue_1502`
  source, `issue_1501` rerun) produces a `fail_closed_blocked` evidence packet under
  `docs/context/evidence/issue_3275_rerun_closure_2026-07-03/`. No benchmark campaign run, no
  proposal-model inference, no held-out yield claim, and no paper/dissertation claim edit; the real
  disjoint certified rerun with independent planner-execution outcomes remains the open issue #3275
  contract.
* Generated the **current-roster release-gate evidence report** for issue #4166, the real-campaign
  application of the reporting layer merged in PR #4184. The merged evaluator now consumes the
  canonical retained camera-ready `campaign_summary.json` directly (`release_gates._rows_from_mapping`
  recognizes the `planner_rows` container), and a new provisional gate spec
  (`configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml`) targets the metric field
  names the retained camera-ready campaign actually records (`collisions_mean`, `near_misses_mean`,
  `jerk_mean`, `comfort_exposure_mean`). Unrecorded metrics (`min_clearance_m`,
  `proxemic_intrusion_rate`) are shipped as `required: false` fail-closed coverage-gap gates that
  render `not_evaluable` without swamping their category. The resulting packet under
  `docs/context/evidence/issue_4166_release_gates/current_roster/` reports 2 pass / 5 fail / 1
  not_evaluable over the 8-planner roster (degraded `socnav_bench` fails closed to `not_evaluable`).
  Thresholds are provisional configuration, not certification, threshold approval, or a planner
  ranking; no new benchmark run was performed.
* Added an **evidence-closure regression guard for the issue #4011 RL trajectory smoke bundle**
  (#4011): `tests/benchmark/test_rl_trajectory_dataset_smoke_evidence.py` pins the committed
  `RLTrajectoryDataset.v1` smoke evidence bundle under
  `docs/context/evidence/issue_4011_rl_trajectory_dataset_smoke_2026-07-02/` to the canonical
  recorder/loader contract in `robot_sf/benchmark/rl_trajectory_dataset.py`. The guard fails
  closed if the committed preview stops loading, the manifest `dataset_sha256` drifts from the
  preview, the manifest stops validating against its JSON Schema and split-leakage semantics, or
  the manifest is no longer exactly reproducible from the committed preview via
  `build_rl_trajectory_manifest`. This closes the provenance/checksum loop so downstream offline-RL
  work can depend on the smoke artifact. No pipeline behavior changes and no dataset is regenerated
  from a live runner.
* Added a **ScenarioBelief seed-sufficiency closure resolver** for issue #3556:
  `scripts/validation/close_issue_3556_seed_sufficiency.py` searches an ordered set of durable roots
  (`docs/context/evidence` first, then the ephemeral runner output root) for a retained ScenarioBelief
  campaign root that exposes the analyzer-required report files
  (`reports/seed_variability_by_scenario.json`, `reports/seed_episode_rows.csv`). When one is found it
  runs `scripts/tools/analyze_seed_sufficiency.py` and promotes a compact closure summary; when none is
  found it fails closed with an explicit, reproducible per-root missing-artifact blocker. The pure
  packet builder lives in `robot_sf/benchmark/scenario_belief_screening.py`
  (`build_seed_sufficiency_closure_packet`). This supersedes the single-path handoff probe from PR #4273
  by searching durable locations. Current state on `main`: **blocked** — no retained #3556 campaign root
  exists yet, so the committed packet under
  `docs/context/evidence/issue_3556_seed_sufficiency_closure_2026-07-03/` records the fail-closed blocker.
  No campaign was run; no Slurm/GPU submission; no benchmark or paper-grade claim.
* Added an **interaction-validity guard** to the issue #4207 certification-transfer probe (#4207):
  `robot_sf/benchmark/certification_transfer.py` gains `classify_interaction_status(...)` and now
  tags every gate cell with `interaction_status` (`interacting` / `non_interacting` / `unknown`) and
  every transfer-matrix row with `interaction_status` plus an `interaction_exercised` flag. Because
  `social_force_default` (social-force model, SFM) and `hsfm_total_force_v1` (headed social-force
  model, HSFM) only diverge when the robot enters the 5 m
  pedestrian near field, a `stable_pass`/`stable_fail` built from cells that never do is *vacuous* —
  it does not demonstrate certification robustness. The report/metadata now carry
  `interaction_status_counts` and a `model_sensitivity_exercised` boolean, and the README/claim
  boundary spell out the vacuity caveat. New
  `scripts/benchmark/annotate_certification_transfer_interaction_issue_4207.py` applies the guard
  post-hoc to a recorded `summary.json` (no new simulation) and emits `interaction_validity.{md,csv}`.
  Running it on the committed 2026-07 packet shows all 8 cells are `non_interacting`
  (`robot_ped_within_5m_frac == 0`, `min_clearance_m ≈ 20 m`), so its "0 flips" result is not yet
  evidence of model-robust certification; the residual empirical action is an interacting scenario
  family. Diagnostic-only; no deployment, benchmark-strength, or paper/dissertation claim.
* Added an **interacting certification-transfer smoke scenario family** for issue #4207 (#4207) — the
  residual empirical action named by the interaction-validity guard above. New probe config
  `configs/benchmarks/issue_4207_interacting_smoke_probe.yaml`, gate spec
  `configs/benchmarks/release_gates/issue_4207_interacting_smoke_gates.yaml`, scenario descriptor
  `configs/scenarios/single/issue_4207_interacting_smoke.yaml`, and a deterministic CPU-scale smoke
  episode fixture `tests/benchmark/fixtures/issue_4207_interacting_smoke/interacting_smoke_episodes.jsonl`.
  Run through the existing runner in report-only mode, the family drives every transfer cell into the
  5 m near field so the guard reports `model_sensitivity_exercised = true` (16/16 `interacting`) with a
  genuine interacting flip on the `ppo` arm (`fragile_pass_to_fail` under `hsfm_total_force_v1`). The
  committed packet `docs/context/evidence/issue_4207_interacting_smoke_2026-07/` is generated from the
  **synthetic** fixture — it is not a physics run (see its `SYNTHETIC_SMOKE_NOTICE.md`); its purpose is
  to exercise the guard's positive path end-to-end and template a future physics-verified geometry/spawn
  CPU re-run, which remains the tracked next step. No simulation, Slurm/GPU submission, retraining,
  deployment, benchmark-strength, or paper/dissertation claim.
* Added a **safety-wrapper false-stop diagnostic** for `wrapper_on` benchmark rows (#3501):
  `robot_sf/benchmark/safety_wrapper_runtime.py` gains `analyze_false_stop_diagnostic(...)`, and the
  episode summary now embeds a schema-tagged `false_stop_diagnostic` block plus a
  `false_stop_proxy_supported` flag. The diagnostic classifies each hard-stop veto over a forward
  `false_stop_lookahead_s` window into `hazard_confirmed` (the trigger step or a step inside the
  window shows non-positive predicted clearance — a clearly valid intervention),
  `analysis_unsupported` (clearance stayed positive across a complete window, so a false stop cannot
  be told apart from a wrapper-prevented collision), or `window_truncated` (the episode ended before
  a full window elapsed). This is deliberately non-causal `diagnostic_proxy` evidence over the
  executed (wrapped) trajectory: the summary keeps `false_stop_analysis_supported: false` because a
  causal false-stop *rate* still needs the paired `wrapper_off` counterfactual. It changes no wrapper
  defaults, thresholds, or runtime behavior; the block is only populated on opt-in `wrapper_on` runs.
* Added a **fail-closed readiness preflight for the issue #4142 dense DPCBF comparison**
  (#4142): new `robot_sf/benchmark/issue_4142_dpcbf_dense_readiness.py` and the CLI
  `scripts/tools/check_issue_4142_dpcbf_dense_readiness.py` validate the predeclared
  comparison packet `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml` before any
  campaign can be authorized. It reuses the canonical CBF runtime validator to confirm the
  three arms (`cbf_off`, `cbf_collision_cone_on`, `cbf_dynamic_parabolic_v1_on`) stay
  predeclared, distinct, and fail-closed, cross-checks each arm's adapter config against
  its runtime variant, confirms the scenario manifest exists, and enforces the
  fallback/degraded exclusion. The packet gains a structured `canonical_command` field and
  an explicit `summary_contract.excluded_row_statuses` list. The surface is read-only:
  status is `prerequisites_incomplete` (any structural gap) or `inputs_ready_campaign_gated`
  (inputs valid, campaign still gated behind a not-yet-wired packet runner and human/Slurm
  authorization). It runs no episodes, submits no Slurm/GPU job, and makes no
  safety-performance or collision-reduction claim. See
  `docs/context/issue_4142_dpcbf_dense_readiness.md`.
* Added **write-time episode-row mechanism and exposure instrumentation** to the map runner
  (#4242): `robot_sf/benchmark/map_runner_episode.run_map_episode` now attaches native
  `failure_mechanism` (`failure_mechanism_taxonomy.v1`) and `interaction_exposure`
  (`interaction_exposure.v1`) schema blocks to every episode record. The failure-mechanism block is
  fail-closed `unknown` (a single episode is not a paired-trace analysis; geometry/scenario names are
  never substituted), and interaction exposure is computed from the episode's own trajectory or
  emitted as an explicit `not_derivable` block rather than fabricated zeros. Diagnostic-only; no
  benchmark-ranking claim promotion.
* Added **pairwise-isolated HSFM field-of-view (FoV) repulsion attenuation** and vectorized the
  `O(N^2)` time-to-collision (TTC) weight path (#3481). New pure helper
  `pairwise_fov_attenuated_forces(...)` in `robot_sf/sim/pedestrian_model_variants.py` attenuates each
  pedestrian-pedestrian force contribution by its own FoV weight before summing
  (`attenuated[i] = sum_j weights[i, j] * pairwise_forces[i, j]`), so a rear neighbor is down-weighted
  without disturbing an in-cone neighbor's push — unlike the coarse `anisotropic_fov_total_force(...)`
  aggregate mode that collapses to one `np.min` factor per actor. `pairwise_time_to_collision(...)` is
  now solved with NumPy broadcasting instead of a Python double loop, with masks that reproduce the
  earlier scalar branches exactly. Adds `tests/sim/test_hsfm_fov_pairwise_isolation.py` (narrow-passage
  isolation, weighted-sum definition, fail-closed validation, and vectorized-vs-scalar TTC equivalence
  on a deterministic bottleneck fixture). Diagnostic/prototype only: no default model change, no
  calibrated-realism claim; simulator per-pair force wiring and benchmark evidence remain follow-up.
* Added a **reproducible SocNavBench custom-map traversible generator** (#4291):
  `scripts/tools/generate_socnavbench_traversible.py` builds `traversibles/<MAP>/data.pkl` from a
  staged per-map mesh using SocNavBench's own renderer, writing the derived artifact **into the data
  root** (never git) and printing its SHA-256 for the external-data registry pin. Input validation
  is fail-closed and CI-safe: `--dry-run` and the skip-if-absent path use only the standard library
  plus the shared external-data path resolver (reusing the `socnavbench-s3dis-eth` registry id), so
  they run without SocNavBench's heavy mesh dependencies; a missing mesh exits `2` with an actionable
  message naming the expected path. The build is idempotent (`--force` to rebuild). This closes the
  last generation gap from #1498 and produces the `eth_traversible_pickle` input that
  `validate_socnav_map_batch.py --preflight` reports missing, unblocking ETH map conversion (#1134).
  Docs: new section 7 in `docs/socnav_assets_setup.md`. No generated artifact is committed and no
  benchmark claim is made.
* Added a **pedestrian uncertainty-envelope abstraction** for conservative obstacle clearance
  (#4141): new `robot_sf/nav/uncertainty_envelope.py` defines `PedestrianUncertaintyEnvelope`, a
  `linear_inflation_policy(alpha, dt)` factory, an `effective_pedestrian_radius(...)` planner
  substitution helper, an `envelope_diagnostics(...)` provenance builder, and a
  `ConformalInflationPolicy` stub interface documenting the future conformal upgrade seam (#4138).
  The envelope inflates the effective pedestrian radius by `alpha * horizon_step * dt`, so a planner
  keeps more room at longer prediction horizons; `alpha == 0.0` (or a disabled config) reproduces the
  deterministic baseline exactly. The prediction-aware MPC planner gains opt-in
  `pedestrian_uncertainty_envelope_enabled` / `pedestrian_uncertainty_alpha_mps` config fields
  (default off) wired into its hard per-horizon-step clearance constraints, records the envelope
  settings and claim boundary in `diagnostics()`, and ships an example
  `configs/algos/prediction_mpc_cv_uncertainty_envelope.yaml`. This is a structured-conservatism
  abstraction only; it makes no calibration or benchmark-improvement claim, changes no pedestrian
  dynamics or simulator collision semantics, leaves existing clearance metrics unchanged, and adds no
  new dependencies.
* Added a **proximity-released pedestrian hold** for scripted single pedestrians (#3977):
  `SinglePedestrianDefinition` gains `hold_until_robot_within_m`, `hold_ref_point` (defaults to the
  scenario event-contract `conflict_point`), and `hold_timeout_s` (fail-open, ~6s default). A
  pedestrian holds at its curb waypoint until a robot enters the hold radius of the reference point
  (or the timeout elapses), then steps across — making the `pedestrian_steps_in_front` event a
  runtime-real, robot-proximity-triggered crossing instead of a fixed open-loop timer.
  `SinglePedestrianBehavior` reuses the existing `robot_pose_provider` and exposes
  `hold_release_reasons()` (`robot_proximity`/`timeout`). The issue #3977 `safe_braking` scenario
  now uses this hold (hold at `(14, 17.5)` until a robot is within 5.5 m of `(14, 14)`); its
  `start_delay_s` was removed because a zeroed spawn velocity leaves the pysocialforce desired speed
  at zero, which otherwise froze the pedestrian in place at run time.

* Added `RLTrajectoryDataset.v1` infrastructure (#4011): episode-major JSONL loader/writer,
  return-to-go computation, split/provenance manifest schema with leakage checks, map-runner
  simulation-trace reward/terminal capture, recorder CLI, validation CLI support, focused tests, and
  a tiny preview evidence bundle. This is an infrastructure contract only; it does not train offline
  reinforcement learning policies, submit jobs, or promote benchmark claims.
* Added **uncertainty-aware safety primitives** (#3974): conformal prediction buffers and
  cumulative-intrusion metrics in new `robot_sf/benchmark/uncertainty_safety.py`. Split-conformal
  (`split_conformal_radius`) and online Adaptive Conformal Inference (`adaptive_conformal_buffers`)
  turn pedestrian-prediction residuals into per-step spatial buffers calibrated to a target
  coverage, and `compute_intrusion_metrics` reports current-position / predicted-trajectory /
  uncertainty-buffer intrusion time ratios plus cumulative and max intrusion depth over a run.
  This is a bounded first slice: pure, versioned, fixture-tested computation labeled
  **diagnostic** (no safety guarantee, no benchmark claim). The runtime uncertainty-triggered
  fallback trigger is a deliberate successor slice and is not included.
* Tightened the certified adversarial failure-archive readiness checker (#3275) so optional archive
  summary counts fail closed when they disagree with the actual `entries` or `clusters` payload.
* Tightened the proxy checkpoint-selection readiness preflight (#3204) so registry entries with
  declared-but-incomplete public release metadata fail closed before they can be treated as ready
  checkpoint inputs.
* Added a machine-readable known-blocker map to the proxy checkpoint-selection readiness preflight
  (#3204). The checker now reports configured blocker IDs and revival conditions as a fail-closed
  prerequisite without running training, selecting a checkpoint, or promoting benchmark evidence.


* Added a **diagnostic missing-export blocker report** for frozen-trace `EpisodeEventLedger.v1`
  before/after reconciliation (#3482). New `build_missing_frozen_trace_export_report`
  (`robot_sf/benchmark/frozen_trace_reconciliation.py`) emits a
  `frozen_trace_event_export_blocker.v1` report with null comparison counts and per-artifact
  `not_evaluable_missing_event_ledger_export` statuses, and the comparator CLI
  (`scripts/benchmark/compare_frozen_trace_event_ledgers.py`) gains `--diagnose-missing-exports`
  to write that report when one or both durable exports are absent. The reconciliation guard was
  also tightened so a metric-semantics export that only names `EpisodeEventLedger.v1` (without the
  durable exact/surrogate event payload) fails closed instead of being treated as comparable. This
  is **diagnostic-only**: it promotes no benchmark claim and invents no old/new event counts; the
  durable frozen 0.0.2 before/after rerun/backfill remains the open blocker under #3482.
* Added a read-only **readiness preflight for proxy-based predictive-planner checkpoint selection**
  (#3204). New config `configs/research/predictive_checkpoint_proxy_v1.yaml` declares the inputs the
  merged proxy-vs-ADE analyzer (`scripts/research/analyze_predictive_checkpoint_proxy.py`, #3307)
  needs, and new tool `scripts/research/check_predictive_checkpoint_proxy_readiness.py` fails closed
  (`status: blocked`, exit 2) when those inputs are absent or degenerate. It maps each `predictive`
  registry checkpoint to its `local_path` presence (reusing the canonical
  `robot_sf.models.registry.load_registry`, no download), gates on the claim contract's "≥ 6
  resolvable checkpoints", and — when a training summary is supplied — reuses the analyzer's verdict
  so an `inconclusive` (no hard-success spread) summary is rejected. This operationalizes the manual
  "predictive checkpoints not available" diagnostic recorded on #3204: against the live registry all
  8 predictive checkpoints resolve to absent `output/tmp/...` paths, so the tool reports `blocked`.
  It is **diagnostic/preflight only** — it selects no checkpoint, runs no training, submits no jobs,
  promotes no evidence, and asserts no benchmark result. Fixture tests cover missing config, missing
  checkpoint mapping, the all-zero degenerate-spread state, `proxy.enabled=false`, the ready case,
  and CLI exit codes, plus a pin on the current blocked live-registry state (the intended revival
  signal once checkpoints hydrate).
* Added a read-only research-package registry and preflight for the continuous
  social-navigation research-engine epic (#3057). New module
  `robot_sf/research/package_registry.py` exposes `load_registry` and
  `evaluate_registry_preflight`, which read the declarative registry
  `configs/research/research_package_registry_issue_3057.yaml` and report, for each
  research-engine package (scenario suite v0, planner-readiness matrix, campaign-manifest
  flow, canonical result store, and sprint packages A/B/C plus the July-2026 release),
  which required tracked artifacts are present vs. missing and whether declared
  prerequisite packages are satisfied. A package is reported `ready` only when every
  required artifact exists and every prerequisite is `ready`; any missing artifact or
  unsatisfied prerequisite **fails closed to `blocked`** with explicit gaps. The helper
  **composes existing config/contract/issue metadata only — it makes no benchmark, metric,
  or research claim, schedules nothing, and runs no campaign** (`claim_boundary:
  registry/preflight metadata only`). CLI:
  `scripts/tools/research_package_preflight.py` (Markdown or JSON, with an optional
  `--fail-on-blocked` gate).
* Added a **design-stage evidence-stream integration contract inventory** for issue #3293 — the
  local, no-data slice of "design evidence integration between simulation and real-world AMV data".
  `robot_sf/research/evidence_integration_inventory.py` enumerates the evidence streams Robot SF
  could integrate (`simulation_trace`, `amv_command_response`, `external_pedestrian_trajectory`,
  `pilot_fleet_operational`), separates them into `calibration` / `benchmark` / `operational`
  categories (which use different denominators and must not be mixed), and declares the required
  provenance + uncertainty fields per stream. A mandatory `calibration_status` field prevents any
  synthetic/proxy envelope from silently passing as calibrated. `check_stream_metadata` (and the
  `scripts/tools/check_evidence_integration_inventory.py` CLI: `--list` / `--check`) is a
  **presence-only** structural check on synthetic metadata — it ingests no real data, validates no
  field values, weights no evidence, and makes no safety/benchmark/paper-facing claim. Externally
  blocked streams (notably AMV command-response, per the #3293 maintainer decision: <5% feasibility,
  implementation hard-blocked) declare an explicit `blocked_until` unblock condition. Design note:
  `docs/context/issue_3293_evidence_integration_contract_inventory.md`.
* Added a **fail-closed readiness check for false-positive actor-injection replay inputs** (#3300),
  the acceptance dimension PR #3271 closed out of #2927 as *unavailable*. New pure module
  `robot_sf/benchmark/false_positive_injection_readiness.py` exposes
  `check_false_positive_injection_readiness(spec)`, which validates a replay-condition spec's injected
  actor inputs (reusing the canonical `ObservationPerturbationSpec` for shape rules rather than
  re-deriving them) and its provenance fields (scenario, seed, planner mode, perturbation family,
  execution mode). It returns an explicit `ready` / `not_available` / `blocked` verdict so a malformed
  or unavailable false-positive condition fails closed with an actionable blocker list instead of
  silently passing. A thin CLI `scripts/benchmark/check_false_positive_injection_readiness.py` runs the
  check on a YAML/JSON spec and exits non-zero (3) when blocked. This is a bounded readiness/contract
  slice only — it does not run a replay campaign, change sensor semantics, or make any benchmark or
  safety claim.
* Added a **fail-closed release-readiness / claim-audit preflight checklist** for research-package
  releases (#3081). New module `robot_sf/benchmark/release_preflight.py` evaluates a declarative
  checklist (`load_release_preflight_checklist` + `evaluate_release_preflight`) that maps issue
  Issue #3081's four acceptance criteria to concrete, mechanically checkable prerequisites: a reproduction
  record (`artifact_present`), regenerated tables/figures bound to canonical-source digests
  (`checksum_manifest`), promoted claim cards that exclude fallback/degraded/unavailable execution
  modes (`claim_audit`), and a sprint-issue classification ledger (`issue_classification_ledger`).
  Every check **fails closed** — a missing artifact, a checksum mismatch, a symlinked or
  worktree-local `output/` path, a promoted claim resting on an excluded mode, or an unclassified
  sprint issue all resolve to `blocked` with explicit gaps rather than silently passing. The
  companion CLI `scripts/tools/release_preflight_check.py` renders a Markdown/JSON report with an
  optional `--fail-on-blocked` gate, and the shipped checklist
  `configs/benchmarks/releases/release_july_2026_preflight_issue_3081.yaml` honestly reports
  `blocked` against the current checkout (the durable July-2026 artifacts do not exist yet). This is
  a **preflight, not a release step**: it never publishes, tags, uploads, regenerates artifacts,
  closes issues, edits claims, or *declares* readiness — a passing run only means no blocking gaps
  were found among declared prerequisites, and a maintainer still owns the readiness decision. It
  composes existing contracts (`release_protocol`, `benchmark_row_claim`) rather than duplicating
  them, and is complementary to the package-level `research/package_registry` preflight (#3057).
  Synthetic tests cover each fail-closed path plus a smoke check of the shipped checklist.
* Added a **real-trajectory ingestion and artifact-staging contract** (#3065): a dataset-agnostic,
  bring-your-own-dataset (BYO) manifest schema plus a fail-closed preflight checker. New package
  `robot_sf/data_ingestion/` defines the JSON Schema
  (`schemas/real_trajectory_ingestion_manifest.v1.json`) and `real_trajectory_contract.py`
  (`load_manifest`, `validate_manifest_structure`, `run_preflight`). The manifest tracks dataset
  metadata, license posture + supplier acknowledgment (the repo never redistributes raw data),
  retrieval instructions, the canonical conversion shape, SHA-256 checksums, split naming, a
  git-ignored staging dir, and an explicit durable-storage pointer. `run_preflight` enforces the
  semantic gates the schema cannot: BYO license acknowledgment, git-ignored staging, a durable
  boundary that is not the disposable `output/` root, and `benchmark_eligibility` that stays below
  claim grade until availability is checksum-`validated`. Ships a copy-me template
  (`configs/data/real_trajectory_manifest.example.yaml`) and a CLI
  (`scripts/tools/check_real_trajectory_manifest.py`). Contract-only: no external dataset is
  downloaded, copied, committed, or claimed as real-world validation. See
  `docs/context/issue_3065_real_trajectory_ingestion_contract.md`.
* Added a **read-only capability inventory / preflight** for the learned probabilistic graph
  predictor v1 lane (#2844). New module
  `robot_sf/benchmark/learned_predictor_capability_inventory.py` enumerates the *code-level*
  prerequisites a v1 learned predictor would extend — the `ProbabilisticPredictor` protocol and
  `ProbabilisticPrediction` container, the `BaselineProbabilisticPredictor` surface, the
  `ForecastBatch.v1` contract, the forecast dataset recorder + split-manifest builder, the durable
  model-artifact registry classifier, and the readiness evidence gate + contract doc — and reports
  whether each hook is present in the checkout. CLI
  `scripts/validation/inventory_learned_predictor_capability.py [--json]` prints the report and
  exits non-zero only on a *missing wiring* hook. This is strictly a wiring/preflight surface: it
  does **not** implement, train, or run a predictor, change planner behavior, or run any campaign,
  and `unblocks_training` is always `False` — the lane unblock decision remains owned by the
  evidence gate `scripts/validation/validate_learned_prediction_readiness.py`. A complete inventory
  matches the 2026-06-23 readiness audit: the lane is blocked on *evidence*, not on missing hooks.
* Added a presence-only **cross-benchmark comparison readiness** checker
  (`scripts/tools/cross_benchmark_comparison_readiness.py`, #3287) for the downstream cross-suite
  policy-comparison campaign. It inventories the four prerequisite families named by the issue —
  scenario converter (#3285), metric wrappers (#3286), campaign policy metadata/manifest, and
  external social-nav benchmark assets (#1456 / #1498 / #2414 / #3161 / #2918) — and reduces each to
  a `ready` / `blocked` / `waived` state: `ready` when every expected local artifact is present,
  `blocked` when an artifact is missing (the default for external assets, which are never staged
  in-repo), and `waived` when a maintainer explicitly waives a family with a recorded reason
  (mirroring the issue's "satisfied or explicitly waived" acceptance criterion). The report is
  fail-closed: `campaign_authorized` is always `False` and `run_gates` lists the standing blockers,
  so a "prerequisites ready" report can never be mistaken for authorization to run the campaign or
  claim cross-suite equivalence. The tool does not access external assets, run a campaign, or assert
  equivalence.
* Added a **bring-your-own (BYO) staging preflight** for licensed Stanford Drone Dataset (SDD)
  annotations (#1497). Under the BYO-dataset reframe (#3065) the repository never licenses, hosts,
  or redistributes SDD; a contributor stages a copy they already have rights to. The canonical SDD
  manifest (`configs/data/sdd_staging_manifest.yaml`) now carries an ordered `retrieval_recipe`
  (concrete acquisition steps, no auto-download) and a `license_acknowledgment` opt-in
  (`{required, acknowledged, statement}`, shipped `acknowledged: false` so the committed default
  never implies redistribution rights). A new `sdd-preflight` command in
  `scripts/tools/manage_external_data.py` (`build_sdd_preflight`) reports the two staging
  prerequisites and the blocked-external-input state, and **fails closed** (CLI exit 2) until the
  license acknowledgment is affirmed *and* the annotation files are present locally. Manifest
  parsing also fails closed on a non-boolean acknowledgment, a malformed recipe, or a non-string
  statement, and rejects `license_acknowledgment.required: false`, so the mandatory gate cannot be
  bypassed by a typo or disabled by a locally edited manifest. This is staging-gate/provenance work
  only: it does **not** download,
  ingest, or transform any SDD data, run benchmarks, or edit any benchmark/paper claim. Scenario
  curation against real annotations remains #1126.
* Added a **fail-closed readiness/preflight checker for the predictive planner v2 same-seed
  comparison** (#1490 umbrella, ego-conditioning child #1504). New module
  `robot_sf/benchmark/predictive_v2_comparison_readiness.py` exposes
  `validate_predictive_v2_comparison_readiness`, which validates the committed feature contract
  `configs/training/predictive/predictive_ego_features_contract_v1.yaml` across four metadata stage
  gates: variant completeness (baseline / obstacle-only / ego-only / combined with the expected
  schema and input-dim), provenance (every referenced config and seed/scenario/grid manifest exists),
  ego-obstacle conditioning metadata (ego variants declare a defined `ego_motion_channel_producer` and
  share a single comparability key), and same-seed schedule (seed manifests, fixed seed, and forecast
  vs. navigation metric separation). A fifth gate, `blocked_slurm_gate`, fails **closed**: the
  four-way expansion (#1505/#1506/#1507) stays `blocked` behind the maintainer-selected revised
  hypothesis and the same-seed coupling gate #2916, clearing only when an explicit `continue`
  coupling-gate artifact and the maintainer-hypothesis acknowledgement are both supplied. A new CLI
  `scripts/validation/validate_predictive_v2_comparison_readiness.py` exposes the check with
  decision-coded exit status (`0` ready, `2` blocked/incomplete, `1` contract load error). Against the
  committed contract the preflight reports metadata-complete-but-`blocked`, mirroring the recorded
  Issue #1490 decision. This is **coordination/preflight readiness only**: it does not train, evaluate, tune
  planners, run benchmarks, or submit SLURM.
* Added a **fail-closed campaign-readiness gate for the learned-risk model v1 Slurm campaign**
  (#1472). New module `robot_sf/training/learned_risk_campaign_readiness.py` exposes
  `evaluate_campaign_readiness`, which aggregates the two existing canonical owners — the
  launch-packet validator (`validate_launch_packet`) and the durable trace-manifest validator
  (`validate_trace_manifest`) — into a single campaign launch decision. The decision is
  `campaign_launch_ready` only when **both** gates pass; an invalid launch packet, a structurally
  invalid manifest, or unresolved durable artifact pointers all fold into a fail-closed
  `campaign_blocked` result with the underlying per-gate blockers surfaced. A new CLI
  `scripts/validation/check_learned_risk_campaign_readiness.py` defaults to the checked-in #1472
  inputs and reports decision-coded exit status (`0` ready, `2` input file missing, `3` blocked).
  This is **readiness/preflight only**: it submits no SLURM job, trains nothing, fetches nothing,
  and promotes no artifacts — a ready decision means the checked-in contract is locally complete.
  Against current `main` the campaign correctly reports `campaign_blocked` (the launch packet is
  valid; the durable trace/baseline artifacts are still `:pending`).
* Added a **durable trace-URI registry contract and validator** for oracle-imitation artifacts so
  the downstream `training_ready` state is mechanically checkable (#2655). The new canonical module
  `robot_sf/training/oracle_trace_uri_registry.py` (schema `oracle-trace-uri-registry.v1`) records,
  per split, the durable trace URI, its SHA-256 checksum, the split/trace identity, the schema
  version, and a `retrieval_status` (`resolvable`/`pending`/`blocked`); large traces stay out of git.
  A fail-closed CLI (`scripts/validation/validate_oracle_trace_uri_registry.py --require-training-ready`)
  and an example registry (`configs/training/ppo_imitation/oracle_trace_uri_registry_example.yaml`)
  let the oracle-imitation lane leave `artifact_retrieval_blocked` only when every required split has
  a concrete, durable, resolvable pointer with a valid checksum. This is the local registry/validator
  slice only: it collects no traces, publishes no artifacts, submits no jobs, and makes no training
  readiness claim (the checked-in example is intentionally not training-ready). See
  `docs/context/issue_2655_oracle_trace_uri_registry.md`.

* Added a **fail-closed curation readiness preflight** for SDD-derived benchmark scenarios (#1126).
  Issue #1126 curates the first real Stanford Drone Dataset (SDD) benchmark scenario set, but stays
  blocked on licensed external data (#1497/#2413). The new `scripts/tools/sdd_curation_preflight.py`
  is a thin *curation-step* gate that composes the canonical owners — `resolve_sdd_scenario_prior_mode`
  from `scripts/tools/manage_external_data.py` (staging gate) and `load_sdd_points` from
  `scripts/tools/import_sdd_scenarios.py` (the importer's own parser) — to classify whether a
  curation run may be promoted as benchmark evidence. `benchmark_promotion_allowed` is True **only**
  when SDD is staged and checksum-validated (`dataset_backed_prior`) *and* the candidate annotation
  satisfies the deterministic selection rule; a parseable fixture or unvalidated copy stays
  `proxy_schema_smoke` and is never promoted. `--require-benchmark-ready` exits non-zero so callers
  fail closed. This is readiness/schema preflight only: it does **not** download, ingest, or curate
  real SDD data, write any scenario/map artifact, run a benchmark campaign, or assert any benchmark
  result. Fixture-backed tests cover the blocked, proxy, and dataset-backed-candidate states.
  Documented in `docs/context/issue_1126_sdd_curation_preflight.md`.

* Added a read-only **ORCA-residual learned-policy lane readiness/preflight surface** (#1358).
  New module `robot_sf/benchmark/orca_residual_lane_readiness.py` exposes `assess_lane_readiness`,
  which inventories the lane's local prerequisites (behavior-cloning lineage packet, smoke/pretrain
  config, the `orca_residual_guarded_ppo_v0` / `orca_residual_guarded_ppo_progress_v1` candidate
  configs and their `training_required: true` registry entries, the policy-search runner, and the
  grounding 2026-05-05 evidence report), the canonical command shapes (routes), and the declared
  external blockers (child #1475 continue/revise/stop classification, `resource:slurm` training, and
  pending durable dataset/checkpoint artifacts). The diagnostics contract and lineage-packet schema
  are reused from `robot_sf/training/orca_residual_lineage_packet.py` (no fork). Status fails
  **closed** to `prerequisites_incomplete` when any local surface is missing or the packet is
  invalid, otherwise `blocked_on_followup` (scaffolding handoff-complete, lane still gated). A new
  CLI `scripts/tools/orca_residual_lane_readiness.py` exposes the report (`--json`) with exit `0`
  handoff-complete / `2` incomplete. This is **inventory/preflight only**: it does **not** submit
  SLURM, train policies, alter planner behavior, run benchmarks, or assert any benchmark/paper
  result. #1358 remains a parent/umbrella coordination issue gated by child #1475.
* Added a read-only **oracle-imitation warm-start readiness preflight** for the downstream
  warm-start training and benchmark comparison (#1496). Issue #1496 is the training step that
  *consumes* the durable oracle-imitation dataset from #1470; safe shared-PC work is verifying
  prerequisites, not launching training or data collection. The new
  `check_warm_start_readiness` function in
  `robot_sf/training/oracle_imitation_warm_start_readiness.py` and the
  `scripts/validation/check_oracle_imitation_warm_start_readiness.py` CLI take a readiness
  manifest, validate the referenced dataset launch packet via the canonical
  `validate_launch_packet(..., require_training_ready=True)` checker, and confirm the
  behaviour-cloning warm-start config, RL-only baseline config, optional PPO fine-tuning config,
  and split/leakage contract all exist. They emit a compact readiness report (`ready`/`blocked`)
  with an explicit blocker list and **fail closed** under `--require-ready`. This is
  preflight/provenance hygiene only: it collects **no** data, trains nothing, submits no Slurm,
  and asserts no benchmark result. The shipped manifest
  (`configs/training/ppo_imitation/oracle_warm_start_readiness_issue_1496.yaml`) currently reports
  `blocked` because the #1397 dataset packet is intentionally not training-ready until #1470 lands.
* Added a **fail-closed SocNavBench map-conversion readiness preflight** for the ETH import batch
  (#1134, parent #334). The batch validator
  (`scripts/tools/validate_socnav_map_batch.py`) gains a `conversion_readiness()` function and a
  `--preflight` CLI mode that layer a conversion *decision* on top of the existing raw
  asset-existence report. Conversion is reported `ready` only when every `required_for_conversion`
  source asset is staged; while the ETH traversible `data.pkl` (or mesh dir) is missing the verdict
  is `blocked_pending_source_assets` with `conversion_ready: false`, a `next_action` naming the
  missing source paths, and a non-zero exit code. When blocked, any pre-existing planned map
  *artifact* (`map_svg`, `scenario_config`) is surfaced as `placeholder_risk` so a hand-authored or
  inferred ETH-like SVG cannot be silently mistaken for an official conversion; the provenance note
  is intentionally excluded because it is expected to exist during the blocked phase. This is
  **preflight/readiness only**: it does not download assets, stage data, convert maps, run
  simulations, or assert any benchmark result. Against the current repo state the preflight reports
  `blocked_pending_source_assets`, matching the issue's blocked status.
* Added a **read-only readiness preflight for the compact CARLA native↔aligned parity bundle**
  (#1510). New module `robot_sf_carla_bridge/parity_bundle_preflight.py` exposes
  `check_parity_bundle_readiness` (and the pure `evaluate_payload_metadata`), which checks — per
  candidate scenario, fail-closed — that each T0 export manifest reads/validates, every referenced
  export payload exists and is schema-valid, the scenario fixture certificate is present with an
  eligible status, provenance carries a `robot_sf_commit`, and the recorded `trajectory_fields` are
  non-empty, plus that the intended output directory is path-safe (never created). It composes the
  canonical export readers (`resolve_export_manifest_payload_paths`, `read_export_payload`) and the
  canonical parity metric list (`parity.DEFAULT_PARITY_METRICS`). CLI
  `robot-sf-carla-parity-bundle-preflight` prints a JSON report and exits nonzero when not ready.
  This is the **agent-executable local slice only**: it does **not** import CARLA, run a simulator,
  execute a replay, or assert metric parity — a `ready` verdict means the static prerequisites are
  staged, not that native↔aligned parity holds (runtime eligibility must be confirmed on a capable
  CARLA host; see `docs/context/issue_1511_carla_native_aligned_parity_claim_boundary.md`).
* Added a **coherence regression guard** for the real AMV command-response trace *acquisition*
  issue (#2000). Issue #2000 is hard-blocked on external data (no realistic real-trace source,
  <5% feasibility) and its only agent-executable action is to keep the acquisition path documented
  and the downstream consumer (#2415 staging manifest) and proxy fallback (#1585) coherent. The
  consumer manifest mechanism is already owned by #2415
  (`robot_sf/research/amv_command_response_trace_manifest.py`); this change adds the missing test
  (`tests/research/test_issue_2000_amv_acquisition_coherence.py`) that locks in #2000's acceptance
  criteria so future edits cannot silently drop the #2000/#1585 cross-reference from the shipped
  manifest, flip the still-blocked acquisition into a "ready"/"staged" state without a real source,
  or let the preflight imply a calibrated/hardware realism claim. It also guards that the trace
  declares the command/response/timing field classes #2000 names. This is a test-only guard: it
  does **not** collect traces, calibrate any envelope value, copy private data, run a campaign, or
  edit any paper/dissertation claim.

* Added a **diagnostic readiness/preflight checker for AMV actuation-envelope calibration inputs**
  (#1559). New module `robot_sf/benchmark/amv_calibration_readiness.py` exposes
  `assess_amv_calibration_readiness` / `assess_amv_calibration_readiness_from_config`, which inspect a
  candidate calibrated-actuation profile (e.g. the
  `configs/benchmarks/issue_1586_calibrated_actuation_profile_skeleton_v0.yaml` skeleton) and report
  `ready` vs `blocked`, **fail-closed**. It closes a real gap: the existing
  `synthetic_actuation.validate_actuation_profile_claim_boundary` checks only that provenance fields
  are *present and non-empty*, so the placeholder skeleton (`source_id: "pending-#1585"`,
  `measurement_date: "pending"`, …) passes structural validation while remaining unfit for use. The
  readiness checker additionally flags placeholder/pending provenance, missing fields, tracking-issue
  `source_uri`s, synthetic-vs-calibrated conflation, and the proxy-vs-hardware source distinction. CLI
  `scripts/benchmark/check_amv_calibration_readiness.py` prints a JSON report and exits non-zero when
  blocked. **Claim boundary:** paper-facing AMV actuation use stays **blocked** — `paper_facing_allowed`
  is True only for a hardware/official-spec source class (a real trace #2000 or official spec), never
  for the accepted #1585 proxy. This change does **not** calibrate from data, tune envelope values, or
  run any campaign.
* Added a **durable learned-risk training trace manifest contract and fail-closed validator**
  (#2312, parent #1472). New module `robot_sf/training/learned_risk_trace_manifest.py` exposes
  `validate_trace_manifest`, which checks a `learned-risk-trace-manifest.v1` YAML (durable baseline
  and per-slice trace artifact URIs, recorded SHA-256 checksums, scenario split ids, required
  episode fields, a per-label availability table, and `retrieval_status`) and returns a
  `training_readiness_decision`. The decision is `ready_for_training_handoff` only when every input
  is locally contract-complete; any placeholder alias (`:pending`), missing checksum, absent label,
  or missing slice fails **closed** to `artifact_retrieval_blocked` — never an implied training-ready
  state. A new CLI `scripts/validation/validate_learned_risk_trace_manifest.py` exposes the check
  with decision-coded exit status (`0` ready, `2` structurally invalid, `3` blocked). The tracked
  manifest `configs/training/learned_risk_trace_manifest_issue_2312.yaml` records the honest current
  `blocked` state. This is **manifest/preflight readiness only**: it does **not** materialize traces,
  copy external data, run training, or submit SLURM (the trace bytes come from the #1472 / #2441
  runs). Shared checksum logic was promoted to `sha256_file()` in
  `robot_sf/training/learned_risk_launch_packet.py`. See
  `docs/context/issue_2312_learned_risk_trace_manifest.md`.
* Added a read-only **blocklist coverage audit** for local-only baseline model artifacts (#1764).
  The local-artifact preflight blocklist
  (`configs/baselines/local_model_artifact_blocklist.yaml`) names exact `(path, field, value)`
  triples, but nothing previously detected entries left stale when a baseline config is retired,
  removed, or migrated to a durable `model_id`. The new `audit_blocklist_coverage` function in
  `robot_sf/benchmark/local_model_artifacts.py` and the `--audit-blocklist` mode of
  `scripts/validation/check_local_model_artifacts.py` classify each blocklist entry as `active`,
  `orphaned_config_missing` (config no longer exists), or `orphaned_reference_gone` (config
  migrated/rewritten away from the blocked path), and **fail closed** (non-zero exit) when any
  orphaned entry remains so the allowlist shrinks as configs are recovered or retired. This is
  inventory/provenance hygiene only: it does **not** publish or recover any checkpoint artifact,
  rewrite benchmark configs, change registry entries, or assert any benchmark result. On the
  current tree all seven shipped entries report `active`.
* Made the **SocNavBench control-pipeline asset readiness checker fail-closed against empty
  placeholder directories** (#1456). `scripts/tools/prepare_socnav_assets.py` now classifies each
  required asset as `available` (directory backed by real files), `placeholder` (directory exists
  but is empty), `missing`, or `excluded` (not required for the selected `render_mode`). Both
  `placeholder` and `missing` required assets are reported under `missing_required`, so an empty
  `wayptnav_data/` shell can no longer pass as a restored asset — matching the non-empty directory
  contract already used by `scripts/tools/manage_external_data.py`. This is asset-readiness
  reporting only: it does not download external assets, change benchmark results, or count
  fallback/degraded rows as evidence. Focused fixture tests cover the available, placeholder, and
  excluded states. Docs updated in `docs/socnav_assets_setup.md`.
* Added a **metadata-only staging-manifest preflight** for real AMV command-response actuation
  traces routed through the `amv-calibration` external-data path (#2415). New schema
  `robot_sf/research/schemas/amv_command_response_trace_manifest.v1.json` and checker
  `robot_sf/research/amv_command_response_trace_manifest.py` validate, per candidate trace bundle,
  the provenance/license, the command/response/timing channels the bundle would expose, and the
  declared calibration targets — fail-closed against the canonical synthetic-actuation envelope
  vocabulary (`robot_sf.benchmark.synthetic_actuation.actuation_variability_fields()`) so the
  manifest cannot drift from what calibration can consume. CLI
  `scripts/validation/check_amv_command_response_trace_manifest_issue_2415.py` prints a JSON report
  and (with `--probe-live-staging`) reconciles each trace's declared staging status against a live
  `manage_external_data.check_asset` presence probe. The shipped example manifest
  (`configs/research/amv_command_response_trace_manifest_issue_2415.yaml`) is
  `blocked-external-input` today, matching the maintainer decision on #2415 (2026-06-22) that no
  realistic real-data source is currently available. This is **manifest-contract only**: it does
  not ingest traces, run a calibration, or make a hardware-calibrated realism claim
  (`evidence_boundary = manifest_contract_only_no_trace_ingest_no_calibration_run_no_calibrated_claim`).
* Added an **opt-in, diagnostic-only closing-speed / time-to-collision (TTC) aware near-miss**
  surface (#3700). New module `robot_sf/benchmark/near_miss_ttc.py` exposes
  `near_miss_ttc_input_readiness` (a fail-closed validator of the timing/velocity inputs a TTC-aware
  near-miss requires: a finite positive `dt`, a `(T,2)` `robot_pos`/`robot_vel` with at least two
  frames, and a `(T,K,2)` `peds_pos`) and `compute_ttc_near_miss_diagnostic`, which counts steps
  whose minimum projected TTC falls below a threshold and reports the worst closing speed under
  distinct `near_miss_ttc__*` keys. It reuses the existing TTC convention from
  `time_to_collision_min` and the canonical pedestrian-velocity primitive. This is **additive and
  diagnostic-only**: it does **not** modify the canonical distance-based `near_misses` metric, wire
  anything into SNQI or any scoring path, calibrate the threshold (the default
  `DIAGNOSTIC_TTC_THRESHOLD_S` is an explicit uncalibrated placeholder, `decision-required` per
  Issue #3700), or assert any safety result. The diagnostic fails closed — raising `NearMissTtcInputError`
  rather than returning zeros — when the timing/velocity inputs are missing or invalid. Tests cover
  fast-closing vs. opening synthetic trajectories, the fail-closed contract for missing timing
  fields, and that the canonical `near_misses` output is unchanged.
* Added a read-only diagnostic inventory / fail-closed preflight for **conflicting "canonical" SNQI
  weight sets** (#3723). New module `robot_sf/benchmark/snqi/weights_inventory.py` discovers every
  known SNQI weight source — the code default `recompute_snqi_weights("canonical")` plus the shipped
  JSON files under `model/` and `configs/benchmarks/` — records each set's dominant term and numeric
  scale (raw vs normalized), and reports provenance conflicts: two sources both claiming the
  "canonical" designation but yielding different weight *directions* (e.g. the collision-dominant
  code default vs the jerk-dominant `model/snqi_canonical_weights_v1.json`), raw-vs-normalized scale
  splits, and duplicate weights shipped under distinct labels. A new `inventory` subcommand on the
  SNQI CLI (`python -m robot_sf.benchmark.snqi.cli inventory [--json] [--no-fail-on-conflict]`) and
  the `preflight_snqi_weight_sets(strict=True)` API expose the report and **fail closed** (non-zero
  exit / `SNQIWeightProvenanceError`) when a blocking conflict is detected. This is provenance
  disambiguation only: it does **not** choose a canonical set, re-tune weights, change normalization
  (#3699), or alter SNQI scoring — picking the source of truth remains a maintainer decision. See
  `docs/snqi-weight-tools/weights_provenance.md`.
* Added a **diagnostic inventory** for the two incompatible collision/near-miss definitions
  (#3724). The benchmark metric (`robot_sf/benchmark/metrics.py`) classifies collision/near-miss
  with a radius-aware *clearance* rule, while the SNQI proxy (`robot_sf/gym_env/snqi_proxy.py`)
  and policy-search validation (`scripts/validation/policy_search_common.py`) use the *raw center
  distance* against `COLLISION_DIST=0.25` — so the same geometry is labeled differently (the
  clearance collision boundary sits at a center distance of ~1.4 m with default radii vs 0.25 m, a
  ~5× gap). New module `robot_sf/benchmark/collision_definition_inventory.py` classifies center
  distances under both regimes and reports where they diverge, and CLI
  `scripts/benchmark/collision_definition_inventory_report.py` prints/saves a preflight report with
  an optional `--fail-on-divergence` (fail-closed) exit. This is **diagnostic only**: it does not
  change any threshold, metric, proxy, or validation behavior, and does not choose a canonical
  definition (that remains `decision-required` on #3724).
* Added a **read-only heavy forecast-model family inventory / preflight** for the offline
  prediction study (#2845). New pure module `robot_sf/research/forecast_heavy_model_inventory.py`
  documents the candidate heavy predictor families (transformer, AgentFormer-like, CVAE,
  diffusion) with qualitative literature-derived planning estimates of compute cost, inference
  latency, uncertainty quality, and repository integration burden; probes that the
  offline-evaluation surfaces an experiment would touch are importable (the forecast
  metrics/calibration/conformal/dataset/batch/baseline surfaces; fail-closed on the required
  ones); and reports the minimum-offline-experiment prerequisites (a staged held-out dataset, a
  heavy-model→`ForecastBatch` adapter, a CPU runtime budget, the study report, plus external
  dependency/checkpoint decisions) as explicit blockers, rolled up into a `ready`/`blocked`
  minimum-experiment status. Thin CLI
  `scripts/research/check_forecast_heavy_model_inventory.py` (`--json`/`--list`) and study report
  `docs/context/forecast_heavy_model_study_2026-06-20.md`. Inventory slice only: trains no model,
  runs no inference, adds no dependency, runs no benchmark, and makes no model-quality claim
  (`evidence_tier` stays blocked → analysis_only).
* Added a metadata-only staging/preflight checker for external pedestrian-prior extraction (#2918).
  New module `robot_sf/benchmark/pedestrian_prior_extraction_manifest.py` exposes
  `check_pedestrian_prior_extraction_manifest`, which validates a
  `pedestrian_prior_extraction_manifest.v1` manifest (allowed external source type, the bounded
  prior parameters the run will emit — walking speed, crossing angle, density, interaction
  distance, stop/yield timing — provenance fields, and the authored-vs-dataset-backed separation)
  and reports missing parameters, provenance/separation blockers, and whether a dataset-backed
  prior claim is yet allowed. The checker **ingests no external data, stores no raw trajectories,
  and makes no calibrated- or representative-prior claim** (`evidence_boundary:
  prior_extraction_plan_only_no_calibrated_prior_claim`); a dataset-backed claim is gated behind
  accepted provenance and a source family with a registered external-data staging contract, while
  `blocked-external-input` (the default, matching issue #2918's external-data block) and
  `proxy-only` manifests cannot assert a prior, and a `proxy-only` manifest declaring a
  dataset-backed source is rejected as boundary conflation. The allowed source-type → external-data
  asset-id map is cross-checked against the canonical `scripts/tools/manage_external_data.py`
  registry. CLI: `scripts/tools/check_pedestrian_prior_extraction_manifest.py`; example manifest:
  `configs/research/pedestrian_prior_extraction_manifest_issue_2918_example.yaml`; context note:
  `docs/context/issue_2918_pedestrian_prior_extraction_preflight.md`.
* Added a metadata-only staging-contract checker for dataset-backed scenario priors (#3161). New
  module `robot_sf/research/scenario_prior_staging_contract.py` exposes
  `check_scenario_prior_staging_contract`, which validates a `scenario_prior_staging_contract.v1`
  contract (per-dataset provenance/license, the canonical scenario-prior distribution fields a
  dataset-backed prior would expose, and the explicit external-data blocker) for the Stanford Drone
  Dataset, SocNavBench ETH, and AMV candidates. Declared distribution fields are checked against the
  live `PARAMETER_GROUPS` vocabulary of the #2919 comparison harness so the contract cannot drift
  from what the comparison can compute, and a dataset declared `staged` is reconciled against a live
  `manage_external_data.check_asset` presence probe (fail-closed). The checker **ingests no dataset,
  stores no raw trajectories, runs no comparison, and makes no real-world realism claim**; with no
  dataset staged it reports `blocked-external-input`. Example contract
  `configs/research/scenario_prior_staging_contract_issue_3161.yaml`, CLI
  `scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py`, context note
  `docs/context/issue_3161_scenario_prior_staging_contract.md`.
* Added a metadata-only measurement/intake-manifest checker for autonomous micromobility vehicle
  (AMV) actuation latency and rider-coupling response (#3283). New module
  `robot_sf/benchmark/actuation_latency_measurement_manifest.py` exposes
  `check_amv_actuation_latency_measurement_manifest`, which validates an
  `amv_actuation_latency_measurement_manifest.v1` manifest (declared sensor channels, per-channel
  sampling rate, time synchronization, provenance, and the synthetic-vs-measured separation)
  against the canonical command-response latency and rider-coupling quantity contract and reports
  missing channels, synchronization/provenance/separation blockers, and whether a measured-value
  claim is yet allowed. The checker **collects no data, fabricates nothing, and makes no
  measured-value or calibrated-actuation claim** (`evidence_boundary:
  measurement_intake_plan_only_no_measured_value_claim`); a `measured` claim is gated behind
  accepted provenance, while `blocked-external-input` (the default, matching issue #3283's
  external-data block) and `synthetic-only` manifests cannot assert measured values, and a
  `synthetic-only` manifest declaring a measured source is rejected as boundary conflation. It also
  proposes the latency and rider-coupling fields a future measured AMV actuation profile would
  expose, extending the synthetic actuation-envelope schema in
  `robot_sf/benchmark/synthetic_actuation.py` without promoting placeholders into calibration
  evidence. CLI: `scripts/tools/check_amv_actuation_latency_measurement_manifest.py`; example
  manifest: `configs/benchmarks/issue_3283_amv_actuation_latency_measurement_manifest_example.yaml`;
  protocol note: `docs/context/issue_3283_amv_actuation_latency_measurement_protocol.md`.
* Added a **diagnostic-only** inventory of SNQI per-term normalization status (#3699). New module
  `robot_sf/benchmark/snqi/normalization_inventory.py` and CLI
  `scripts/benchmark/snqi_normalization_inventory_report.py` enumerate each SNQI term's scaling
  regime and surface that `compute_snqi` mixes *raw, unbounded* penalty terms (`time`, `comfort`)
  with *baseline-normalized* `[0, 1]` terms (collisions, near-misses, force-exceed, jerk), which
  makes the weight coefficients non-comparable as relative priorities. The report flags the
  mixed-scale condition and any baseline-normalized term lacking median/p95 coverage, and can fail
  closed (`--fail-on-mixed-scale`, `--fail-on-missing-baseline`). It does **not** change the SNQI
  formula, weights, `normalize_metric`, or any emitted score, and does **not** choose between the
  normalize vs. clip-and-document remedies (that remains `decision-required` on #3699). An anti-drift
  test reconstructs the SNQI score from the inventory's term table and asserts it equals
  `compute_snqi` exactly.
* Added a fail-closed Package A readiness checker so a rank-stability / held-out-family transfer
  campaign can verify its input prerequisites before execution (#3078). New manifest
  `configs/benchmarks/issue_3078_package_a_readiness.yaml` declares the held-out-family scenario
  inputs, seed-plan metadata, and frozen-protocol entry points; new checker
  `scripts/validation/check_package_a_readiness.py` verifies every declared input exists, that the
  seed-plan metadata is explicit, and that the output location is disposable under `output/`,
  exiting non-zero with `status: not_ready` when any prerequisite is missing. This is a
  provenance/readiness gate only — it does not execute the benchmark, submit Slurm, or interpret
  ranks.
* Added a **static launch preflight** for crossing-conflict predictive retraining configs (#3214).
  New module `robot_sf/training/predictive_retrain_preflight.py` and CLI
  `scripts/validation/validate_predictive_retrain_preflight.py` validate a predictive training
  pipeline config (the kind consumed by `scripts/training/run_predictive_training_pipeline.py`)
  *before* any SLURM/GPU launch: config structure, data prerequisites (scenario matrix, hard-seed
  manifest, planner grid, and weighting spec) exist, base/hard-case feature-width compatibility (the
  checkpoint-lineage contract), the navigation gate kept separate from the trajectory gate
  (`max_val_ade`/`max_val_fde`), a present evaluation block, and a declared output root. The check is
  cheap and CPU-only — it fills the gap left by the pipeline's own guards, which only run mid-pipeline
  after expensive dataset collection. It does **not** collect data, train, submit Slurm, change
  augmentation semantics, or make any model-improvement claim; the actual weighted retraining run is
  owned by #3254. Text and `--json` reports; exit code 0 when valid, 2 when invalid.
* Added a metadata-only validation-contract checker for candidate real-world micromobility traces
  (#3278). New module `robot_sf/analysis_workbench/real_trace_validation_contract.py` exposes
  `check_real_trace_validation_contract`, which maps a candidate dataset descriptor
  (`real_trace_validation_contract.v1` schema) onto the existing trace-failure predicate input
  contract and reports, per predicate, whether the declared channels make it *validatable* or
  *blocked*, which required channels are missing, whether a directly observed ground-truth label
  exists, and the resulting limitation (e.g. `late_evasive_reaction` / `oscillatory_local_control`
  are computable from kinematics but usually have no directly observed label to cross-check). It
  also surfaces metadata, provenance/access, and missing-data blockers. The checker **does not
  ingest, copy, or read any external/private data and makes no real-world validation claim**
  (`evidence_boundary: contract_check_only_no_real_world_validation`); the committed example
  descriptor (`configs/benchmarks/issue_3278_real_trace_validation_contract_example.yaml`) is a
  placeholder with `access_status: blocked` because external data access is not yet accepted. CLI:
  `scripts/tools/check_real_trace_validation_contract.py`. Decision note:
  `docs/context/issue_3278_real_trace_validation_contract.md`. Tests cover complete, incompatible,
  and missing/placeholder-metadata descriptors.
* Added a **read-only pedestrian-model assumption inventory / preflight** for the proposed
  headed social-force (HSFM) + time-to-collision (TTC) predictive-force experiments (#3481).
  New pure module `robot_sf/research/ped_model_assumption_inventory.py` documents the current
  force-model assumptions the upgrade would change (no field-of-view attenuation on ped-ped
  repulsion, heading coupled to instantaneous velocity, Euclidean-distance repulsion with no
  TTC term), probes that the entry-point surfaces an HSFM/TTC experiment would touch are
  importable (the vendored Social Force core plus the `robot_sf/ped_npc` behavior/population
  surfaces; fail-closed if any required surface is missing), and reports the still-missing
  prerequisites (HSFM heading state, FoV weight, TTC term, narrow-passage and bottleneck
  fixtures, versioned parameters, external calibration data) as explicit blockers. A documented
  CLI (`scripts/research/check_ped_model_assumption_inventory.py`, `--json`/`--list`) renders the
  inventory. This is the **assumption-and-artifact inventory** slice only: it implements **no**
  force law, changes **no** scenario behavior, runs **no** benchmark, and makes **no** realism
  claim (the force-model upgrade itself stays `evidence_tier: idea`).

* Added a **dry-run Robot SF -> external-benchmark scenario converter** that emits a deterministic,
  schema-validated **intermediate representation (IR)** for Robot SF scenario-matrix entries (#3285).
  New pure module `robot_sf/benchmark/scenario_interop.py` exposes `convert_scenario_to_ir(scenario)`,
  which maps geometry, agent paths, start/goal states, timing, and environment semantics into a
  target-neutral IR (`robot_sf/benchmark/schemas/scenario_interop_ir.v1.json`), preserves source
  provenance (scenario id, source file, source fields, metadata), and reports every unmapped source
  field explicitly in `unsupported_fields` instead of silently dropping it. A documented dry-run CLI
  (`scripts/tools/convert_scenario_interop.py --matrix <file>`) prints the IR per scenario and a
  validity summary. This is the local, asset-free slice of the cross-benchmark interop work: it emits
  **no** SocNavBench/HuNavSim asset and makes **no** cross-benchmark validity or score-parity claim;
  producing real external assets remains blocked on #1456/#1498/#2414/#1134. See
  `docs/context/issue_3285_scenario_interop_converter.md`.
* Added a **readiness/preflight helper for closed-loop prediction Package C** (#3080).
  `scripts/tools/prediction_package_c_readiness.py` inventories the local prerequisites for the four
  Package C forecast arms — `no_forecast`, `cv`, `semantic_cv`, and `interaction_aware` — across the
  three coordination stages (open-loop forecast analysis #2915, observation-perturbation replay #2777,
  closed-loop forecast-risk coupling #2916). For each arm it reports the required configs and code
  entry points, the declared same-seed plan (`[111, 2868]` read from the #2915/#2916 configs), the
  declared output roots, and the named blockers, then classifies the arm as fail-closed
  `ready` / `blocked` / `missing`. Per the issue-audit on 2026-06-22, Package C is gated solely on
  Issue #2916 producing a durable campaign result store, so the default status is `blocked` until a
  `--coupling-result-store` with a canonical `summary.json` is supplied; `blocked`/`missing` are never
  treated as success evidence. The helper inspects the repository only — it does **not** execute any
  benchmark campaign, alter predictor semantics, or claim forecast performance. Text and `--markdown`
  reports plus optional `--output-json`; exit code 0 only when every arm is `ready`, 1 otherwise. New
  fixture tests `tests/tools/test_prediction_package_c_readiness.py` cover the ready/blocked/missing
  states (synthetic trees) and assert the real repo is wired (fails closed to `blocked`, not
  `missing`).
* Added a **presence-only tournament-readiness helper** for the predictive hard-case breakthrough
  portfolio (#3215). `scripts/tools/predictive_tournament_readiness.py` inventories the local
  prerequisites (expected configs, harness scripts, and output path) for the three concurrent
  tournament arms — Selection (#3204), Authority (#3213), and Model (#3214) — plus the shared
  `predictive_hard_seeds_v1` benchmark protocol, and classifies each as `ready` or `blocked` with the
  missing paths named as blockers. The check is deliberately fail-closed and **never** asserts run
  authorization: `run_authorized` is always `False` and the standing `run_gates` (open child bets,
  maintainer-set overnight SLURM/GPU budget, Autonomous Usage Stop Guard) are reported so a
  "prerequisites ready" result is not mistaken for "authorized to launch". It does not submit Slurm
  jobs, run the tournament, rank arms, or edit any claim. Text and `--json` reports; exit code 0 when
  local prerequisites are ready, 1 when blocked.
* Tightened the **oracle-imitation dataset launch-packet preflight** to require a `collection_roots`
  block before a collection job runs (#1470). The validator
  (`robot_sf/training/oracle_imitation_launch_packet.py`,
  `scripts/validation/validate_oracle_imitation_launch_packet.py`) now fails closed unless the packet
  declares durable destinations for collection logs (`log_root`), raw trace output
  (`dataset_output_root`), and the dataset manifest (`manifest_destination`). Each root must be a
  durable artifact URI (a `:pending` alias is allowed because collection has not run yet) and may
  never point at the gitignored worktree-local `output/` directory, which is not a safe shared
  destination on a multi-agent host. The checked-in `#1397` packet now carries these `:pending`
  destinations. This is a preflight-contract change only: it submits no jobs and collects no data.
* Added a read-only **re-export readiness preflight** for stale dissertation table bundles
  (`scripts/tools/reexport_readiness_preflight.py`, #3203). It composes the existing
  `scripts/tools/stale_artifact_detector.py` freshness classifier with a required-input availability
  check and reduces both signals to a single `fresh` / `stale` / `blocked` state: `fresh` when the
  payload is present and checksums match (no re-export needed), `stale` when a re-export is needed and
  all required inputs (campaign config, generation script, source-commit provenance) are present
  (re-export unblocked), and `blocked` when a re-export is needed but a required input is missing or
  the provenance cannot be reconstructed. This prevents a stale bundle from being silently cited as
  current evidence without first checking that the bounded campaign can be reproduced here. The tool
  never runs the campaign, regenerates a table, or edits dissertation claims.
* Added a machine-readable classic archetype density / tier index
  (`configs/scenarios/archetypes/classic_density_tier_index.yaml`) that documents the *current*
  per-archetype density-tier coverage, density bands, and pedestrian spawn semantics of the classic
  interaction archetypes (#3725). It clarifies the benchmark scenario denominator (the in-matrix
  graded total is **23 rows across 11 configs**, not "12 archetypes × 3 densities = 36") and the
  overloaded meaning of `simulation_config.ped_density`: for `spawn_mode: markers` configs
  (`classic_bottleneck`, `classic_realworld_bottleneck`) `ped_density: 0.0` is a placement-mode
  placeholder (pedestrians come from fixed markers, with `density_advisory:
  zero_baseline_route_spawn`), **not** an empty scene. The index is a *derived, documentation-only*
  artifact — it does not change scenario generation behavior, rename any field, or make tier coverage
  uniform. New test `tests/test_classic_archetype_density_index.py` re-derives the same facts from the
  `classic_*.yaml` configs and `classic_interactions.yaml` and fails closed on drift or missing
  config/tier coverage. The scenario zoo index links the new file for discoverability.
* Added a **presence-only publication-prerequisites preflight** for the v0.1 validation/falsification
  benchmark package (epic #2910). A declarative checklist
  (`configs/benchmarks/releases/benchmark_v0_1_publication_prerequisites.yaml`) enumerates the
  canonical prerequisite owners a v0.1 package must be built on — ODD/scenario contracts, scenario
  certification, benchmark/row claim metadata, ODD/hazard coverage matrix, the release protocol and
  pinned v0.1 release manifest, the seed schedule, the release checklist doc, and citation metadata.
  The companion checker `scripts/validation/check_benchmark_v0_1_publication_prerequisites.py`
  verifies every referenced path exists and fails closed (exit `1`) when a required prerequisite is
  missing. It is deliberately presence-only: it does **not** release, tag, upload artifacts, run a
  benchmark/falsification campaign, judge scenario certification, or declare the benchmark "ready" —
  every report carries an explicit `claim_boundary` to that effect. Run with
  `uv run python scripts/validation/check_benchmark_v0_1_publication_prerequisites.py`.
* Recorded metric-affecting run configuration in benchmark result provenance so result artifacts are
  self-describing (#3701). New pure module `robot_sf/benchmark/run_config_provenance.py` exposes
  `metric_affecting_run_config(config)`, which serializes the two run-config toggles that change *what*
  the reported safety metrics mean — LiDAR `scan_noise` (noisy default `[0.005, 0.002]` vs deterministic
  `[0.0, 0.0]`; see `robot_sf/sensor/range_sensor.py`) and the collision-handling regime
  (`terminate_on_contact` vs `bounce_back`; the robot benchmark env terminates on collision per
  `RobotState.is_terminal`). The map-runner now embeds this block under
  `provenance.config_identity.metric_affecting_config` in every batch summary
  (`robot_sf/benchmark/map_runner.py`), derived once per batch via the fail-soft
  `representative_metric_affecting_config` helper, so two result sets can be checked for comparability
  without out-of-band knowledge of each run's config. The helper is descriptive provenance only — it
  does not redefine metrics, rerun campaigns, or promote benchmark claims, and degrades to a
  `status: not_available` block rather than ever breaking a run.
* Added an **opt-in factorial-ablation manifest + checker** for the merged planner-agnostic safety
  wrapper (#3501). `robot_sf/benchmark/safety_wrapper_ablation_manifest.py` enumerates the
  `planner × {wrapper off, wrapper on}` ablation cells from a tracked config
  (`configs/research/safety_wrapper_ablation_v1.yaml`) and **checks** the design so later runs can
  compare the wrapper on/off consistently: exactly the two `{wrapper_off, wrapper_on}` arms with one
  baseline (off), the off arm disabled and the on arm enabled, the on-arm thresholds validated as a
  real `SafetyWrapperConfig` (predeclared, no per-planner tuning), every planner present in both
  arms, and paired seeds shared identically across arms. The manifest records provenance fields
  (`schema_version`, `safety_wrapper_schema`, `config_path`, `git_head`, `claim_boundary`,
  `evidence_status`) and a `factorial_check` summary; a thin runner
  (`scripts/benchmark/build_safety_wrapper_ablation_manifest.py --dry-run`) writes it. This is a
  **dry-run design contract only** — it binds no runtime objects, runs no benchmark episodes, tunes
  no thresholds, and makes no mitigation-effectiveness claim (the factorial campaign run is the
  deferred downstream follow-up tracked by #3501).
* Added a fail-closed readiness check for scenario-horizon Results evidence (#3266). New module
  `robot_sf/benchmark/scenario_horizon_readiness.py` and CLI
  `scripts/validation/check_scenario_horizon_results_readiness.py` read a re-exported scenario-horizon
  campaign artifact (the dissertation `campaign_table.md` from PR #3263 / issue #3203, or an equivalent
  campaign-summary JSON) and classify it as `valid`, `diagnostic_only`, or `blocked`. The check is
  diagnostic-only — it never reruns a campaign or promotes evidence — and fails closed: a missing or
  unparseable artifact is `blocked`, any non-benchmark-success planner row (e.g. the PPO
  `partial-failure` recorded for #3266) caps the verdict at `diagnostic_only`, and an unasserted SNQI
  contract status keeps the evidence diagnostic-only rather than assuming it passed. Per-row status
  normalization reuses the canonical `fallback_policy.classify_planner_row_status` owner. Run against
  the real #3203 bundle, the check reports `diagnostic_only` (exit 2), confirming those tables remain
  diagnostic/provenance evidence only.

### Fixed

* Fixed a **docs-proof readiness false-blocker for docs/context note PRs** (#4178, successor to
  Issue #4191). The `scripts/dev/check_docs_proof_consistency_diff.sh` wrapper unconditionally injected
  `docs/context/catalog.yaml` into the `--path` selection for *any* docs/context-only diff, which
  forced full catalog schema/provenance validation and surfaced pre-existing, unrelated catalog
  debt (evidence rows that point at ignored `output/` artifacts). A PR that only edited a context
  note (never touching `catalog.yaml`) was blocked by that baseline debt. The wrapper now keeps only
  `README.md`/`INDEX.md` as always-selected link anchors; full catalog validation runs when
  `catalog.yaml` is itself in the diff (selected naturally) or under the explicit
  `--check-context-catalog` flag. #4191 already made the direct Python checker diff-scoped for
  code-only diffs; this closes the same gap for docs/context note diffs. Strict catalog proof on
  actual `catalog.yaml` changes and the explicit `--check-evidence-catalog` mode are unchanged, and
  no catalog rows were repaired. Added a focused regression test.

* Fixed **absolute machine paths leaking into the #4239 h600 SNQI evidence packet** (#4302). The
  builder `scripts/validation/build_issue_4239_h600_snqi_weight_set_ranking.py` hardened `_rel()` so a
  worktree `output/` symlink (which made `path.resolve()` escape the worktree root) no longer falls
  back to an absolute, username-bearing path, and relativizes the config/source-manifest/baseline
  provenance blocks. Regenerated the committed packet
  (`docs/context/evidence/issue_3810_h600_interpretation_2026-07/`): the SNQI ranks, pairwise
  agreement, dedup audit, and diss#331 snippet are **byte-identical**; only provenance path strings,
  the `default_uniform_1p0` no-hash sentinel (now `null` everywhere, rendered empty in CSV/Markdown),
  and checksums changed. Also raises an attributable `ValueError` naming the file/line on a malformed
  `episodes.jsonl` instead of a raw `JSONDecodeError` traceback. Diagnostic-only evidence hygiene: no
  SNQI weight decision, benchmark, or paper/dissertation claim change.
* Fixed **`scripts/dev/pr_ready_check.sh` mishandling a missing `BASE_REF`** on fresh checkouts
  (#3702). When the default `origin/main` (or any configured `BASE_REF`) was not present locally, the
  `git diff "$BASE_REF...HEAD"` comparison emitted a raw `fatal: ambiguous argument` error; because the
  command runs inside a process substitution, `set -e` silently swallowed the failure and the script
  proceeded with an empty changed-file set while still passing the unresolved ref to the downstream
  coverage, perf-evidence, and freshness checks. The script now resolves `BASE_REF` once up front via a
  new `resolve_base_ref` helper: a resolvable ref is used unchanged, a remote-tracking ref (e.g.
  `origin/main`) triggers a best-effort `git fetch`, and an otherwise-unresolvable ref falls back to
  `HEAD` with a clear message instead of crashing. Regression coverage lives in
  `tests/dev/test_pr_ready_preflight.py`. This is a developer-tooling fix only — it changes no test
  lane coverage semantics, coverage thresholds, or CI policy.
* Fixed **OSM SVG `viewBox` parsing failing on comma-separated values** in
  `fast-pysf/pysocialforce/map_osm_converter.py` (#3708). The converter parsed the `viewBox`
  attribute with bare `str.split()`, which only handles whitespace separators, so SVG exports using
  the equally valid comma-delimited form (e.g. `"0,0,488.48,458.33"`) raised
  `ValueError: could not convert string to float`. A new `parse_viewbox` helper splits on any run of
  commas and/or whitespace per the SVG spec, and both call sites (`extract_buildings_as_obstacle` and
  `add_scale_bar_to_root`) now use it. Whitespace-separated viewBoxes are unaffected.
* Fixed the PPO baseline adapter **crashing on partial dict observations** instead of backfilling
  missing keys (#3704). In `robot_sf/baselines/ppo.py`, `_align_model_obs_dict` previously raised
  `ValueError: Missing required dict observation keys` whenever a runner emitted a subset of the keys
  a checkpoint's `Dict` observation space declares (after flattening, prefix-expansion, and alias
  resolution); this failed all 144/144 serial-worker jobs in the #3203 campaign. Missing keys are now
  backfilled with an in-bounds default derived from the target subspace via the new
  `_default_for_space` helper: nested `Dict` subspaces recurse so each declared leaf is materialized,
  and leaf `Box` subspaces yield zeros **clipped to the declared low/high bounds** so a non-zero
  lower bound (e.g. counts/radii) stays in range. The backfill is logged at debug level so the
  substitution stays visible. Genuine shape mismatches and non-Dict payloads for nested keys still
  raise. This keeps PPO evaluation running on partial observations; heavily backfilled runs should be
  treated as degraded rather than faithful evidence. Gym wrapper observation spaces are unchanged.
* Fixed the **SAC baseline velocity action space ignoring `action_semantics`** when converting model
  output to benchmark commands (#3705). In `robot_sf/baselines/sac.py`, `_action_vec_to_dict` always
  scaled the `velocity` (`vx`, `vy`) output vector to `v_max`, even when `action_semantics="delta"`.
  Deltas are increments the env later accumulates (`new_v = old_v + delta`) and are already bounded by
  the SAC action space, so scaling them toward `v_max` distorted the accumulated trajectory. The
  velocity branch now mirrors the existing unicycle contract: the speed cap is applied only for the
  absolute-velocity contract, while delta increments pass through unchanged. Behavior for the
  `unicycle` action space (the current benchmark config default) is unchanged; the default
  `action_semantics="delta"` is intentionally preserved per the canonical training setting.
* Fixed `_spaces_compatible` **rejecting wrapper- or subclass-derived `Tuple` spaces** under strict
  top-level type checking (#3709). In `robot_sf/training/scenario_sampling.py`, the helper opened with
  `type(base) is not type(other)`, so a `gymnasium.spaces.Tuple` and a Tuple subclass (as produced by
  some vectorized-env wrappers/adapters) were reported incompatible even when their element spaces
  matched. Two `Tuple`-compatible spaces are now compared **structurally on their child spaces** while
  every other space family keeps the strict type check, so `Box`/`Dict`/`Discrete` comparison
  semantics are unchanged. The recursive child checks (length, per-element compatibility, Tuple
  vs. non-Tuple) still reject genuine mismatches.
* Fixed `SNQIWeights.load` / `from_dict` **raising unhandled `KeyError`s on malformed config**
  (#3710). In `robot_sf/benchmark/snqi/types.py`, reconstructing an `SNQIWeights` from JSON used bare
  `data["key"]` access (so a missing provenance field surfaced as an opaque `KeyError`) and never
  validated the weights mapping values. Loading now **fails closed with structured diagnostics**:
  non-mapping input, missing or non-string required metadata fields, a non-mapping `bootstrap_params`,
  a non-list `components`, and weight values that are non-numeric, non-finite, or negative each raise a
  descriptive `ValueError` naming the offending field and the source path. Invalid JSON is also
  wrapped in a `ValueError` that includes the file path. This is bounded to input validation — SNQI
  metric definitions, weights, normalization, and benchmark results are unchanged.
* Fixed differential-drive velocity updates **ignoring timestep scaling** (#3711). In
  `DifferentialDriveMotion._robot_velocity` the commanded acceleration (clipped to
  `max_linear_accel` / `max_angular_accel`, the follow-up to #3666/#3689) was applied directly as a
  per-step velocity delta without multiplying by the simulation timestep `d_t`. At the default
  `d_t = 0.1` this permitted 10× the labeled acceleration per step and made the dynamics
  timestep-dependent (smaller timesteps inflated the effective acceleration — the unstable velocity
  spikes reported in the issue). The action is now integrated over `d_t`
  (`max_delta = max_accel * d_t`), making the velocity update timestep-invariant and consistent with
  the `bicycle_drive` model. All pre-existing differential-drive tests use `d_t = 1.0`, where
  `accel * d_t == accel`, so their behavior is unchanged; new regression tests cover `d_t != 1.0`.
  **Behavioral note:** at the simulator's default `d_t = 0.1` this reduces the realized per-step
  velocity change for a given action by 10×; differential-drive controllers tuned against the
  previous (timestep-dependent) integration may need re-tuning. Controller re-tuning and benchmark
  re-validation are out of scope for this fix.
* Fixed the HEIGHT planner adapter's lidar raycasting **ignoring dynamic pedestrians** (#3629).
  `CrowdNavHeightAdapter._raycast_obstacles` intersected each ray only against cached static obstacle
  segments, so the HEIGHT policy's lidar channel was blind to moving pedestrians (they were used for the
  human spatial-edge tensor but never fed into the raycast). Rays are now also intersected against
  pedestrian discs (reusing `circle_line_intersection_distance` from `robot_sf/sensor/range_sensor.py`,
  the same primitive the live env's range sensor uses); the nearest of {static, pedestrian, sensor
  range} wins per ray, and the disc radius is read from the observation when present (default 0.3 m).
  Backward-compatible: an empty pedestrian set reproduces the previous static-only behavior exactly.
* Fixed collision **undercounting** in `summarize_collision_metrics` (#3627): the aggregator read the
  sampled `metrics.collisions` value even when it was a finite `0.0`, only falling back to the exact
  `outcome.collision_event` flag when the metric key was entirely absent — so an episode whose exact
  environment collision fired but whose sampled metric missed the contact was silently aggregated as
  **zero** collisions (a validation-integrity hole independent of the per-episode write-time and
  `EpisodeEventLedger.v1` guards, since this aggregator also reads resumed/loaded JSONL records).
  `_episode_collision_value` now fails closed: when the exact collision flag fired and the sampled
  value is `<= 0`, the count is floored to `1` (sourced as `episode.outcome.collision_event`). No
  inflation — no-collision episodes stay `0` and a larger sampled count is never reduced.
* Fixed a `scripts/dev/check_skills.py` false-positive path validation (#3623): backticked prose
  placeholders such as `SLURM/data-gated` (in a SKILL.md) were validated as repo paths and failed the
  preflight, because `SLURM/` is a real top-level dir. The broken-path check now only flags a
  non-resolving token when it *looks like* a path (has a file extension or nests ≥2 segments below its
  prefix), so single-segment extension-less placeholders are treated as prose — while genuinely broken
  path references (e.g. `docs/does_not_exist.md`, `SLURM/missing/template.sl`) are still caught. The
  offending SKILL.md was left untouched; the fix is in the validator.
* Added a **fail-closed route-clearance guard** to the camera-ready benchmark (#3628). The
  originally-reported `0.0 m` centerline clearances in `classic_merging_*` / `classic_station_platform_*`
  were already re-routed to positive margins on `main` (now +0.56 m and +2.5 m), but the preflight
  only emitted *informational* warnings, so a route with a **negative** margin (centerline-to-obstacle
  distance < robot radius = guaranteed collision) would still run silently. `prepare_campaign_preflight`
  / `run_campaign` now raise `RouteClearanceError` on any `min_clearance_margin_m < 0` (certification
  cannot excuse hard geometric infeasibility; tangent/positive-narrow geometry stays a warning), so
  both `--mode preflight` and `--mode run` fail closed. Also corrected the stale
  `route_clearance_certifications_v1.yaml` entries for the 3 repaired scenarios (`negative clearance` →
  `repaired_geometry`). User-facing: the benchmark can no longer silently run a geometrically-impossible
  route.
* **`stream_gap` was blind in the real benchmark runner** (found while promoting the #3556 belief-mode safety contrast to `map_runner`). `StreamGapPlannerAdapter._extract_state` read the nested SOCNAV observation (`obs["robot"]`, `obs["pedestrians"]`) but `map_runner` feeds a flat observation (`robot_position`, `pedestrians_positions`, `goal_current`), so the planner extracted `robot=[0,0]`, `n_peds=0` and drove blind every episode. `_extract_state` now accepts both the nested and flat observation formats (backward-compatible; the ScenarioBelief harness and 24 existing tests still pass), and `robot_sf/benchmark/scenario_belief_policy_hook.py` reads the flat observation and writes the uncertainty sidecar to `pedestrians_uncertainty` where the fixed extractor reads it. After the fix the planner engages pedestrians and the belief modes differentiate in the real runner.

### Added

* Added a **fail-closed archive-readiness checker** for the adversarial proposal-vs-random study
  (#3275). Before the proposal runner consumes a *real* certified failure archive, the archive must
  carry the fields the downstream disjoint split, overlap provenance, candidate certification, and
  null tests depend on — otherwise those steps cannot be computed. The new pure checker
  (`assess_archive_readiness` / `assess_archive_file_readiness` in
  [`robot_sf/adversarial/disjoint_evaluation.py`](robot_sf/adversarial/disjoint_evaluation.py),
  composing the existing `scenario_family_key` / `disjoint_family_split`) reports a precise
  `ready` / `not_ready` verdict over schema, per-entry `archive_id` / `candidate.scenario_seed` /
  `failure_attribution` presence, derivable scenario families, and whether a disjoint scenario-family
  split with non-empty fit/eval sides is even possible. Unlike the runner's loader, it **never falls
  back to a synthetic fixture**: a missing, empty, unreadable, malformed, or under-populated archive
  is reported `ready=False` with the blocking reasons. A thin CLI
  [`scripts/tools/check_adversarial_archive_readiness.py`](scripts/tools/check_adversarial_archive_readiness.py)
  prints the JSON report and exits non-zero when not ready, so it is safe to use as an input gate.
  This is a readiness/input-hygiene slice only — it runs no proposal-model evaluation, fabricates no
  archive inputs, and makes no held-out-yield or benchmark claim.
* Added a **plan-level preflight for the paper-grade reactivity-vs-replay rank study** (#3637,
  split from #3573). The new pure checker
  [`robot_sf/benchmark/reactivity_replay_preflight.py`](robot_sf/benchmark/reactivity_replay_preflight.py)
  (`reactivity_replay_rank_study_preflight.v1`) verifies a proposed run plan's **preconditions**
  before any compute is spent: ≥3 planners, exactly the reactive/replay arms with **paired** seeds
  (common random numbers), a seed budget at/above the S20 rank-stability floor and above the #3573
  diagnostic matrix, a contrast-registering horizon, and that the **replay limitation** is stated
  (`'replay' = robot→pedestrian force-off in a live sim, NOT trajectory playback`). The limitation
  constants now live in the canonical owner `robot_sf/benchmark/reactivity_ablation.py`
  (`REPLAY_LIMITATION`, `REPLAY_IS_TRAJECTORY_PLAYBACK`, `REACTIVITY_ARMS`) and are imported, not
  duplicated. Thin CPU-only CLI
  [`scripts/benchmark/preflight_reactivity_replay_rank_study_issue_3637.py`](scripts/benchmark/preflight_reactivity_replay_rank_study_issue_3637.py)
  emits a `ready`/`blocked` manifest from the launch packet
  [`configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml`](configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml).
  This is an evidence-control layer only: it does **not** run the benchmark, interpret rank
  stability, or make any paper-facing claim (sufficiency stays with
  `scripts/tools/seed_sufficiency_gate.py` post-run).
* Added a **context-note archival-sweep planner** (#3190):
  [`scripts/tools/plan_context_note_archival.py`](scripts/tools/plan_context_note_archival.py)
  consumes the existing freshness checker plus `docs/context/catalog.yaml` and proposes which notes
  should move to `docs/context/archive/`. It classifies **high-confidence** candidates (catalog rows
  marked `status: superseded` that already name a valid, existing replacement) and **review**
  candidates (notes flagged stale by `check_context_note_freshness`), computes per-note archive
  targets, and detects blocking conflicts (basename collisions or a pre-existing target). The planner
  is **plan-only** — it never moves, writes, or deletes a note (`--json-output` emits a
  maintainer-reviewable plan) — so the actual archival sweep stays a separate, review-gated PR per
  issue #3190. It reuses (does not duplicate) the checker's catalog parsing and stale-note rules.
* Added an **opt-in three-wheeled rollover proxy** runtime hook in `RobotEnv.step()` (#3479). When
  `rollover_proxy_enabled` is set, the env feeds the executed robot `current_speed`
  `(linear_velocity, yaw_rate)` into the existing `rollover_proxy.v1` closed-form stability margin,
  surfaces rollover telemetry in step info / telemetry-analyzer metrics, emits a `ROLLOVER_CRITICAL`
  termination reason, terminates the step, and applies a configured non-positive
  `rollover_proxy_penalty` when the internal proxy trips. **Disabled by default**, so existing
  benchmark/training runs keep identical episode semantics; config validation rejects a non-typed
  `rollover_proxy_params` and a non-finite or positive penalty. This remains an explicit
  **internal non-hardware proxy** (governance gate #2416/#2417), not a hardware-calibrated AMV
  stability claim. The `(linear_velocity, yaw_rate)` assumption holds for differential/holonomic
  drives but not bicycle drive (tracked follow-up #3683).
* Added a regression guard for **pygame-free headless execution** (#3631). A subprocess-probe test
  (`tests/test_pygame_headless.py::test_headless_step_does_not_import_pygame`) builds and **steps** the
  env under forced-headless drivers and asserts `pygame` is never imported and no `sim_ui` is created —
  extending the no-pygame contract from env *construction* (already covered) through a full step, which
  the issue's reproduction specifically targets. The headless path was verified already pygame-free, so
  this is a test-only guard (a negative probe with an injected `import pygame` flips it red, confirming
  it is not vacuous).
* Added a **performance-PR evidence contract** that fails PR readiness when a `perf`-typed
  conventional-commit change (the #3611 → #3613 failure mode: a `perf(planner): cache ...` whose
  claimed speed-up targeted the wrong layer and had to be reverted) lacks a `## Performance
  Evidence` PR-body section with concrete before/after runtime, a representative command, a
  rollback criterion, and — when caching is claimed — a cache-hit/reuse counter. The trigger is the
  local `perf(...)` commit subject (no GitHub label needed), so it runs identically in
  `pr_ready_check.sh` (fail-closed in final mode, advisory in interim) and CI. Check:
  [`scripts/dev/check_perf_evidence.py`](scripts/dev/check_perf_evidence.py); template section in
  [`.github/PULL_REQUEST_TEMPLATE/pr_default.md`](.github/PULL_REQUEST_TEMPLATE/pr_default.md);
  tests: [`tests/dev/test_check_perf_evidence.py`](tests/dev/test_check_perf_evidence.py).
* Added a **fail-closed planner observation-view integrity guard** in the benchmark runner (#3634,
  the runtime guard deferred from #3568). New `robot_sf/benchmark/map_runner_view_integrity.py`
  (`evaluate_effective_view_integrity` + `DegeneratePlannerViewError`, reason `degenerate_planner_view`)
  is checked on the first step in `map_runner_episode.py`: when the observation handed to the planner
  contains pedestrians (`observation_ped_count > 0`) but the planner's own extractor returns zero — and
  the planner is not pedestrian-blind-by-design (per its `observation_spec.inputs`) — the runner raises
  **before any metrics are recorded**, instead of silently emitting collision "results" from a blind
  planner (the `stream_gap` failure class fixed in #3567). Ground truth is the *observation's own*
  pedestrian count, so visibility-masked/occluded or genuinely-empty scenarios do not false-trip. The
  passing diagnostic is stored at `record["integrity"]["effective_view"]`. 20 guard tests + 145
  regression + 13 end-to-end episode tests pass.
* Began the #3463 corrective engineering for the topology near-parity selector lane (diagnostic-tier,
  no benchmark claim). `TopologyGuidedLocalPolicyConfig` gains two current-behavior-preserving knobs:
  `primary_route_progress_gate_use_monotone_accounting` (default `False`) hardens primary-route
  progress accounting against premature stalls — a single A\* re-plan that transiently raises
  `route_remaining` no longer masks real progress (uses `max_sample − newest` instead of
  `oldest − newest`); and `primary_route_progress_gate_min_samples` (default `2`, the historical
  hardcoded value) makes the progress-gate sample threshold explicit. Progress-gate config is now
  fail-closed (non-finite threshold or `< 2` / non-integer min-samples raises `ValueError`), and the
  accounting mode + sample threshold are surfaced in the `topology_reuse_penalty` diagnostics. Deferred
  to follow-ups: command-arbitration-strength tuning, a registered candidate config + paired benchmark
  diagnostic for the monotone variant, and any benchmark promotion (#3463).
* Added the next #3463 successor slice for the topology near-parity selector lane: a registered
  diagnostic-only monotone progress-gated reselection candidate
  (`topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone`) plus a bounded CPU-only
  cross-slice sensitivity packet at
  [`configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml`](configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml).
  This does not duplicate the runtime/code slices in PRs #4176, #4388, #4411, or #4426; it packages
  the monotone corrective behavior for reproducible future smoke runs without making a benchmark claim.
* Added a pre-commit lint that fails when a file under `configs/**` contains an absolute
  home-dir path (`/home/`, `/Users/`, `/root/`), which is non-portable for other contributors
  and automated runners. Intentional absolute paths (e.g. private-ops SLURM routing targets
  that live outside this repo) are exempted by annotating the line with `# allow-abs-path: <reason>`.
  Hook: [`hooks/check_config_abs_paths.py`](hooks/check_config_abs_paths.py); tests:
  [`tests/integration/test_check_config_abs_paths.py`](tests/integration/test_check_config_abs_paths.py) (#3605).
* Ran the deferred #3558 **`stream_gap` gate-threshold calibration sweep**:
  [`scripts/validation/run_stream_gap_gate_threshold_sweep_issue_3558.py`](scripts/validation/run_stream_gap_gate_threshold_sweep_issue_3558.py)
  reuses the #3471 episode harness (now accepting optional per-run gate-threshold overrides;
  backward-compatible, existing #3471 tests still pass) to roll the `uncertain_dropped` mode across a
  grid of `uncertainty_min_existence_probability` thresholds and feeds the per-setting safety
  aggregates + conservative-retention baseline into the merged `calibrate_stream_gap_gate` decision
  layer. Diagnostic-tier result on the controlled crossing scenario: a **safe region exists** only at
  thresholds permissive enough (≤ the degraded existence 0.2) to *retain* the corridor agent — every
  setting that actually drops it (≥0.3, including the **0.5 production default**) is `less_safe`,
  confirming the #3471 finding that the default gate is unsafe. The sweep exercises only the existence
  axis (the scenario degrades existence alone) and does not change the production default (#3558).
* Added a pre-commit lint that catches issue-provenance mismatches in cloned training configs:
  when a config under `configs/training/**` has identity fields (`policy_id` / tracking `group` /
  `tags` / `campaign_id`) that agree on a single issue number, its leading header comment must not
  name a *different* issue (the #3570 pattern, where a header read "Issue-1024" while every identity
  field targeted issue-3068). Headers that also cite a source/leader issue alongside the canonical
  one pass; an intentional mismatch can be annotated with `# allow-provenance-mismatch: <reason>`.
  Hook: [`hooks/check_config_provenance.py`](hooks/check_config_provenance.py); tests:
  [`tests/integration/test_check_config_provenance.py`](tests/integration/test_check_config_provenance.py) (#3606).
* Ran the deferred #3573 **reactive-vs-replay pedestrian-reactivity ablation** through the real
  runner. Added the open-loop replay (non-reactive) pedestrian mode as a scenario-driven toggle:
  `build_env_config` reads `peds_have_robot_repulsion` and sets `sim_config.prf_config.is_active`
  (backward-compatible; default reactive). New campaign
  [`scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py`](scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py)
  runs each planner under reactive vs replay over identical scenarios + seeds (common random numbers)
  via `map_runner.run_map_batch` and feeds the per-planner contrast into the merged
  `assess_reactivity_ablation` quantifier. Diagnostic-tier result on `classic_crossing_subset`
  (goal + orca, 4 seeds, horizon 150): disabling reactivity raised collisions (orca 0.125 → 0.50,
  goal 0.50 → 0.625) and cut separation — reactive (yielding) pedestrians flatter planners. Evidence +
  caveats (sign convention, horizon-sensitivity, small N):
  `docs/context/evidence/issue_3573_reactivity_ablation_2026-06-25/` (#3573).
* Classified the ORCA-residual progress-probe lane decision (#2445):
  [`scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py`](scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py)
  reads the existing v1 bounded smoke result (SLURM job 12913) and applies the #2408/PR #2420 stop
  rule deterministically. Decision: **`stop`** the current residual-BC lane shape — the v1 smoke
  reproduces the v0 failure pattern (`success_rate=0.0` + `timeout_low_progress`) and is itself
  `failed_closed` with all four required smoke-evidence fields null, either of which forbids
  continuation/nominal escalation. Reopening requires a named objective/dataset redesign. The
  classifier fails closed (missing/invalid artifact or absent required fields → `stop` naming the
  gap). Analysis-only over failed-closed evidence — not readiness evidence; decision doc
  [`docs/context/issue_2445_orca_residual_progress_probe_decision.md`](docs/context/issue_2445_orca_residual_progress_probe_decision.md)
  with a #1358/#1475 handoff note (#2445).
* Added the #3556 belief-mode safety campaign harness: [`scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py`](scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py) runs the `oracle` / `uncertain_retained` / `uncertain_dropped` contrast through the **real** `map_runner.run_map_batch` on a crossing scenario family + seed matrix and classifies the episode-level safety result (`revise` / `retention_dominates` / `inconclusive` / `inconclusive_oracle_unsafe`). Current result on `classic_crossing_subset` is **`inconclusive`**: the gate now acts (the dropped mode differs on worst-case clearance), but the oracle baseline collides every episode so there is no near-safe headroom to attribute a collision effect — the remaining #3556 gap is a near-safe crossing family. Evidence + reproduction: `docs/context/evidence/issue_3556_belief_mode_real_runner_2026-06-24/`.

### Added

* Audited the headline 7-planner benchmark comparison for the silent-blind-planner failure mode found in #3556 (`stream_gap` read nested observations while `map_runner` feeds flat ones, so it saw nothing yet emitted collision "results"). **Result: the headline comparison is clean** — its classical planners share `socnav_occupancy._socnav_fields`, which handles both nested and flat observations; `stream_gap` was a standalone exception (not in the headline) and is already fixed (#3567). Added a regression guard `tests/benchmark/test_planner_observation_contract.py` pinning that the shared extractor returns a non-degenerate view of the flat benchmark observation (sees the robot pose and real pedestrian count, not the `[0,0]`/empty blind default) and stays backward-compatible. Audit write-up: `docs/context/evidence/issue_3556_planner_obs_audit_2026-06-24/`.
### Changed

* Made the `pr-body-contracts` CI check **advisory** and fixed two false-positive sources in `scripts/dev/check_pr_followups.py`. (1) The domain-approval trigger is now **negation-aware**: a body that honestly *describes* an evidence boundary ("makes no paper-facing claim") no longer trips the gate meant for bodies that *make* such a claim. (2) The **Follow-Up Issues section is optional** — it is only required when a PR closes an issue while declaring residual scope, so self-contained PRs no longer need a boilerplate "Follow-Up Issues: none" section. (3) New `--advisory` flag (used by `.github/workflows/pr-body-contracts.yml`) reports violations as GitHub `::warning::` annotations and exits 0 instead of blocking the PR; re-promote to blocking by dropping the flag once body conventions settle. Covered by 5 new tests (48 total).

### Added

* Wired the ScenarioBelief drop-vs-retain safety contrast into the **real benchmark runner** (#3556, follows #3471/PR #3553). [`robot_sf/benchmark/scenario_belief_policy_hook.py`](robot_sf/benchmark/scenario_belief_policy_hook.py) builds a ScenarioBelief from each benchmark observation, degrades the existence confidence of out-of-field-of-view agents (keeping uncertainty rows 1:1 with the observation so the gate can act), projects through the production `project_scenario_belief_for_planner`, and merges the uncertainty sidecar into the planner observation; `BeliefModeStreamGapAdapter` wraps the real `StreamGapPlannerAdapter` (fail-closed: a projection failure omits the sidecar, so the planner keeps every agent). `map_runner`'s `stream_gap` branch now reads a `belief_mode` knob (`oracle`/`uncertain_retained`/`uncertain_dropped`) and sets the uncertainty gate (ON only for dropping). Mechanism proof (unit): retaining the uncertain agent matches oracle (both wait), while dropping it makes the planner commit — the #3471 result, now through the real adapter+gate. Ships [`configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml`](configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml) as the campaign launch packet. Integration + mechanism + map_runner wiring tested; the nominal-benchmark campaign run is the next step. The FOV uncertainty source is not a calibrated perception model — no paper-grade claim. Refs #3556.
* Added the episode-level ScenarioBelief uncertainty → planner-safety experiment (#3471): [`scripts/validation/run_scenario_belief_episode_safety_issue_3471.py`](scripts/validation/run_scenario_belief_episode_safety_issue_3471.py) is the bounded follow-up PR #3450 named — it rolls a controlled crossing scenario over time with the real `stream_gap` planner under three belief modes (`oracle`, `uncertain_retained` = gate-off conservative, `uncertain_dropped` = gate-on) sharing one ground truth. Finding (`revise`, diagnostic-tier, 12 seeds): retaining uncertain agents is *identical* to oracle (so representational uncertainty alone is harmless), while **dropping** them raises collisions 0.42→0.92 and unsafe-commit steps 45→246 — the uncertainty-dropping default should be revised/blocked for safety-relevant use. Controlled-scenario evidence, **not** the full benchmark env and **not** paper-grade (the oracle baseline itself collides ~42%; the result is the relative contrast). Config `configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml`; evidence `docs/context/evidence/issue_3471_scenario_belief_episode_safety_2026-06-24/`. Refs #3471.
* Made research leverage a first-class part of autonomous issue selection. `docs/project_prioritization.md` now defines a **Research-Leverage Interpretation** (claim-boundary/hypothesis → `Improvement`; headline-companion/unblocks-downstream → `Unlock Factor`; local-implementable vs SLURM/data-gated → `Success Probability`; deadline → `Time Criticality`) plus a **verify-before-scoring gate** (already-merged/superseded issues route to closure, not a high score). `gh-issue-priority-assessor` gains an **auto mode** that assesses **only empty `Priority Score` items** and never overwrites an existing priority, and `goal-autopilot` runs it as a new `prioritize` phase before `implement` so unassessed issues become leverage-aware scores without churning human-set ones. Mechanically enforced by a new `--only-empty` flag on [`scripts/tools/project_priority_score.py`](scripts/tools/project_priority_score.py) (skips any item that already has a score), with test coverage. `goal-issue-implementation` documents that its Project #5 queue order is now leverage-aware.
* Added a CPU-only pre-submit evidence-contract preflight (#1475): [`scripts/validation/preflight_evidence_contract.py`](scripts/validation/preflight_evidence_contract.py) checks that the current public commit will emit a named evidence contract's required fields BEFORE a SLURM job is submitted, so a job is never run only to fail closed on missing-field bookkeeping (the #1475 / job 12913 GPU-hour waste). It composes the canonical contract definition (a public alias `REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS` exported from `robot_sf/training/orca_residual_lineage_packet.py`, identity-asserted in a test so there is no second source of truth) and the production `_attach_orca_residual_smoke_evidence` builder, evaluated against a representative `GuardedPPOAdapter` row that mirrors the real on-main shape. Exit 0 = conforms (safe to submit), non-zero = block (names the missing fields). Seeded with the `orca_residual_smoke` contract; adding another is a one-entry registry change. The private-ops sbatch wrapper should call it before `sbatch` (doc: `docs/context/preflight_evidence_contract.md`). Refs #1475.
* Added the headline planner-comparison CI + rank-stability report harness (#3216): [`scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py`](scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py) reports per-cell confidence intervals (bootstrap/per-seed) and a Kendall-τ / rank-flip stability statistic across seed resamples for the 7×7 planner×scenario headline grid, REUSING the canonical owners (`seed_variance`, `fidelity_rank_stability.rank_planners`/`kendall_tau`, `canonical_table_export`) rather than reinventing the statistics. Fail-closed (degraded/fallback/not_available cells never count as success) and it **never self-certifies paper-grade**: insufficient seed budget returns `diagnostic`/`blocked_until_run` pending the S20/S30 SLURM run (via #1554) and claim-card review. Ships the SLURM run as a launch packet (real referenced configs + sha256). Coordinates with #1554 (per-seed bundle) and #3078 (package A) without duplication. The paper-grade run remains SLURM (Refs #3216).
* Added S20/S30 seed-budget archival tooling + SLURM launch packet (#1554): [`scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py`](scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py) builds a durable per-planner-by-seed comparison bundle (success/collision/near-miss/timeout/clearance-uncertainty/`time_to_goal_norm`) with bootstrap/per-seed uncertainty and a seed-resampling rank-flip analysis, REUSING the canonical stats (`seed_variance`, `snqi.bootstrap.bootstrap_stability`, `rank_metrics.kendall_tau`/`rank_order`) rather than reinventing them. Fail-closed (fallback/degraded/unavailable/failed/partial/not_available/diagnostic_only rows never count as success); when no real S20/S30 rows exist it emits `blocked_until_run` instead of fabricating a bundle. Ships the SLURM run as a launch packet (`configs/benchmarks/s20_s30_seed_budget_issue_1554_launch_packet.yaml`, real referenced configs + sha256) and a context note recording the claim-map gate and #1545 seed-budget methodology. The heavy S20/S30 h500 run remains SLURM (Refs #1554).

* Clarified token-efficient goal-thread startup rules for workflow-improvement turns: agents now
  explicitly park stale prior work after compaction or user pivots, scope meta-workflow requests to
  fresh docs-or-workflow worktrees, and classify high-priority SLURM work through the appropriate
  single-lane or capacity-aware submit workflow before burning parent-thread context.
* Documented how to obtain, place, and **share external datasets across git worktrees** in
  [`docs/external_data_setup.md`](docs/external_data_setup.md) (#1498): a per-asset obtain/place/
  validate quick-reference for `sdd`, `socnavbench-s3dis-eth`, and `socnavbench-control` (all
  license/agreement-gated — no scriptable download), plus a "Sharing external data across git
  worktrees" section. Staging paths resolve per-worktree, so the guide shows staging one physical
  copy at a machine-stable location and symlinking it into each worktree's git-ignored expected
  path, the `--source` validation option, and the planned `ROBOT_SF_EXTERNAL_DATA_ROOT` shared-root
  follow-up. Docs-only; actual licensed acquisition remains a manual user step (#1498).
* Added legacy PPO snapshot parity checks (#3469):
  [`scripts/validation/check_legacy_ppo_snapshot_parity.py`](scripts/validation/check_legacy_ppo_snapshot_parity.py)
  inventories supported BR-06 PPO registry checkpoints, verifies durable GitHub-release pointers,
  explicitly classifies root-local debug `.zip` snapshots as unsupported for durable compatibility,
  and provides an opt-in hydrated-checkpoint one-step Gymnasium smoke path. Focused tests cover the
  inventory, fail-closed durable-pointer checks, unsupported local snapshots, JSON CLI output, and a
  mocked model/factory step smoke (#3469).
* Added a lightweight PR body-contract workflow (#3472):
  [`.github/workflows/pr-body-contracts.yml`](.github/workflows/pr-body-contracts.yml)
  now validates live pull-request bodies with
  [`scripts/dev/check_pr_followups.py`](scripts/dev/check_pr_followups.py), requiring body input,
  open linked follow-up issues for declared residual work, domain-aware approval for
  evidence-validity-sensitive claims, and a human-written description for substantive
  source/configuration changes. The checker now accepts a changed-file list and rejects empty or
  bot-only substantive PR descriptions, with fixtures covering the #3414/#3415 empty-body pattern,
  the #3416 CodeRabbit-only pattern, and the #3449/#3450 evidence-sensitive missing-contract
  pattern (#3472).
* Added a bounded robot-influence-on-pedestrian-flow v0 slice (#3066):
  [`scripts/benchmark/run_robot_influence_flow_slice_issue_3066.py`](scripts/benchmark/run_robot_influence_flow_slice_issue_3066.py)
  runs a small same-seed campaign (reusing `robot_sf.benchmark.runner.run_batch`) comparing two
  rule-based robot policies (`social_force` vs `orca`, CPU-only) on a corridor/crossing subset of the
  issue-3059 suite (seeds 111/112/113, 12 episodes, 0 degraded) and measures whether the robot policy
  changes pedestrian-flow dynamics (near-vs-far pedestrian accel/turn-rate), reporting per-row
  success/safety + flow/comfort + denominator/uncertainty with fail-closed status and a robot-influence
  interpretation kept separate from nav performance. Result: `diagnostic` — 3 of 4 powered flow deltas
  exceed pooled seed variance, but tiny n, low density, an outlier-dominated corridor cell, and
  confounding with nav outcome keep it diagnostic-only. Smoke/diagnostic, not paper-grade; no
  real-world or sim-to-real claim. Tracked summary under
  `docs/context/evidence/issue_3066_robot_influence_flow_2026-06-23/`. Unblocked once #3062 merged
  (#3066).
* Added the PPO curriculum-learning launch packet (#3068): a pre-launch spec
  ([`configs/training/ppo_curriculum_issue_3068_launch_packet.yaml`](configs/training/ppo_curriculum_issue_3068_launch_packet.yaml),
  schema `ppo-curriculum-launch-packet.v1`) defining a 4-stage density/complexity curriculum over
  real env knobs, a matched fixed-difficulty PPO baseline comparator, seeds, run budget, stop rule,
  metrics, expected artifacts, validation command, and durable-artifact policy — with the three
  competing explanations (extra-budget vs curriculum; train-curve vs final benchmark; insufficient
  provenance) encoded as discriminating checks. All referenced configs exist with verified sha256
  checksums; the context doc
  [`docs/context/issue_3068_ppo_curriculum_launch_packet.md`](docs/context/issue_3068_ppo_curriculum_launch_packet.md)
  records the no-claim boundary and flags the per-stage scheduler wiring as a downstream
  prerequisite. Authoring-only and `evidence:proposal` — no training-result or benchmark claim; the
  long training run remains a separate SLURM issue (#3068).
* Automated stale state-label cleanup at issue closure (#3456): a new GitHub Action
  [`.github/workflows/strip-closed-state-labels.yml`](.github/workflows/strip-closed-state-labels.yml)
  triggers on `issues: closed` and strips any live `state:*` label (`state:ready`/`state:running`/
  `state:blocked`) from the just-closed issue — covering all close paths (manual, duplicate/wontfix,
  PR-merge) at one choke point, with least-privilege `issues: write`, a read-then-write CLOSED
  re-confirm, and no-op when no stale label is present. `scripts/dev/closed_state_label_hygiene.py`
  gains a `--fix` mode that reuses the same `LIVE_STATE_LABELS` (single source of truth), so the
  read-only detector remains the periodic CI/backstop while the event-driven Action removes the root
  cause of recurring manual scrubs (134 stale labels on 2026-06-18, ~66 more in 5 days). Documented
  in [`docs/context/issue_3098_closed_state_label_hygiene.md`](docs/context/issue_3098_closed_state_label_hygiene.md) (#3456).
* Added the Qwen-RobotNav feasibility assessment (#2952):
  [`docs/context/issue_2952_qwen_robotnav_assessment.md`](docs/context/issue_2952_qwen_robotnav_assessment.md)
  audits availability (weights/code/license/size/hardware) and compares Qwen-RobotNav's
  observation/action/tool interface against the real Robot SF local-navigation surfaces
  (`robot_sf/nav/`, `robot_sf/gym_env/`, differential-drive `step()` contract). Decision:
  **`blocked_asset_tracking`** — the report is real (arXiv 2606.18112) and on-target, but public
  weights, a concrete code repo, and a usable license are all **unverified** (with one source
  conflict explicitly flagged), and the interface gaps are large (camera-history obs vs LiDAR+ego;
  8-waypoint output vs single `(v, ω)`; agentic two-tier vs stateless numeric `step`). Every
  availability claim is tabulated as verified/unverified with its source; `evidence_tier: idea`,
  exploratory only — no integration, no weights downloaded. Indexed in
  [`docs/context/INDEX.md`](docs/context/INDEX.md) (#2952).
* Added a bounded sensor-noise / partial-observability robustness slice (#3067):
  [`scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py`](scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py)
  runs a same-seed clean / noisy / partial-observation comparison on a pedestrian-dominated fixture,
  reusing the observation-noise/perturbation wrappers, and reports clean-vs-perturbed deltas
  (min observed distance, observation continuity, near-field exposure, observation count) per row
  with fail-closed status — and the overall classification kept separate from nominal performance.
  Result: `diagnostic` (non-null — perturbations measurably change the observed state the policy
  would consume; partial observation drops actor observations/continuity/near-field), trace-derived,
  single fixture/seed. Diagnostic-only/smoke, not paper-grade; explicitly NOT a real-sensor
  certification or sim-to-real claim. Tracked summary under
  `docs/context/evidence/issue_3067_sensor_noise_robustness_2026-06-23/`. Unblocked once #3062 merged
  (#3067).

* Added a bounded, same-seed forecast-risk closed-loop coupling gate (#2916): a deterministic risk
  adapter [`robot_sf/benchmark/forecast_risk_adapter.py`](robot_sf/benchmark/forecast_risk_adapter.py)
  mapping a `ForecastBatch.v1` to a bounded `[0,1]` per-step risk signal (fail-closed on
  degraded/fallback/oracle batches), a config
  [`configs/research/forecast_risk_coupling_issue_2916.yaml`](configs/research/forecast_risk_coupling_issue_2916.yaml)
  with `no_forecast`/`cv_risk`/`semantic_risk`/`interaction_risk` rows pinned to identical
  seed/scenario, and a fixture-driven runner
  [`scripts/benchmark/run_forecast_risk_coupling_gate.py`](scripts/benchmark/run_forecast_risk_coupling_gate.py)
  that scores collision/near-miss, route progress, stop/yield timing, false-positive stopping,
  runtime, and SNQI per row and emits a `continue|revise|stop` verdict with the
  false-positive-stopping vs safety-benefit trade-off explicit. On the bundled single-pedestrian
  occluded-emergence fixture the gate returns `continue` (forecast risk removed the near-miss with 0
  false-positive stops at a throughput cost). Evidence is `diagnostic_only`/`stress`, not
  paper-grade, and promotes no learned predictor; the three forecast sources are indistinguishable on
  this single fixture (caveat recorded). Tracked summary under
  `docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23/` (#2916).
* Added a bounded ScenarioBelief uncertainty diagnostic (#2546):
  [`scripts/analysis/run_scenario_belief_uncertainty_diagnostic_issue_2546.py`](scripts/analysis/run_scenario_belief_uncertainty_diagnostic_issue_2546.py)
  compares a fixed scenario across five belief conditions (oracle/deterministic, visibility-limited,
  covariance-inflated, class-probability uncertainty, existence-confidence degradation), projecting
  each through `project_scenario_belief_for_planner` for both a consuming planner (`stream_gap`) and
  an unsupported one (`predictive_planner_v2`). The run finds a difference (not null): every
  uncertain condition flips the consuming planner's single-step decision from WAIT to COMMIT versus
  oracle by dropping the uncertain corridor agent (at the visibility-projection step or the opt-in
  `stream_gap` uncertainty gate), while the unsupported planner fails closed
  (`unsupported_uncertainty_planner`). The WAIT→COMMIT flip is explicitly flagged as the expected
  consequence of dropping uncertain agents, **not** a safety improvement. Follow-up decision:
  `continue` (a runtime uncertainty producer + end-to-end stress run is the next bounded step).
  Evidence is `diagnostic_only`/`stress`, not paper-grade; tracked summary under
  `docs/context/evidence/issue_2546_scenario_belief_uncertainty_2026-06-23/` (#2546).
* Added an AMMV mechanism-divergence classification diagnostic (#2444):
  [`scripts/analysis/run_ammv_divergence_classification_issue_2444.py`](scripts/analysis/run_ammv_divergence_classification_issue_2444.py)
  runs direct `SocialForcePlanner` mechanism probes (bypassing the differential-drive benchmark
  adapter that produced #2434's zero-delta result) with an **AMMV-isolated control** -- the same
  AMMV-aware config with `ammv_aware_enabled` toggled off, so the paired delta reflects only the
  AMMV term. Result: `nonzero_divergence_found` -- the AMMV term activates (~2.64 N) and changes the
  same-seed trajectory (robot-state delta 0.21-0.78 m) under isolation, so #2434's zero result is an
  adapter-mode artifact, not a globally inactive mechanism. Includes a zero-divergence guard so
  identical (#2434-style) pairs cannot be reused as behavioral evidence in #2159/#2227 panels.
  `scripts/tools/run_ammv_social_force_pair_diagnostic.py::_run_mechanism_probe` gained a
  backward-compatible `default_config` parameter to support the isolated control. Evidence is
  `diagnostic_only`/`stress`, not paper-grade; tracked under
  `docs/context/evidence/issue_2444_ammv_divergence_2026-06-23/` (#2444).
* Made SDD scenario-prior staging reproducible (#2657): added
  [`scripts/data/stage_sdd_dataset_issue_2657.py`](scripts/data/stage_sdd_dataset_issue_2657.py)
  (subcommands `plan`/`validate`/`status`/`mode`/`download`) and a provenance manifest
  [`configs/data/sdd_staging_manifest.yaml`](configs/data/sdd_staging_manifest.yaml) (source URL,
  CC BY-NC-SA 3.0 license + readme pointer, version tag, expected files/size, checksum placeholders,
  `local_availability`). The tool **never auto-downloads**: a bare run only plans/reports (disk check,
  availability) and exits without network access; `download` requires explicit `--confirm-download` +
  y/N (or `--yes`), checks free disk vs expected size and fails closed if insufficient, stages into a
  git-ignored subfolder (`output/external_data/sdd`), and validates a SHA-256 tree checksum before
  marking `staged`. Wired a `proxy_schema_smoke` vs `dataset_backed_prior` gate into the scenario-prior
  generator (#2726) so a missing/unvalidated SDD can never be presented as dataset-backed evidence.
  Audit doc: [`docs/context/issue_2657_sdd_staging.md`](docs/context/issue_2657_sdd_staging.md) (#2657).
* Added the AMMV contrastive mechanism panel (partial #2227, AMV sub-target):
  [`scripts/analysis/build_ammv_mechanism_panel_issue_2227.py`](scripts/analysis/build_ammv_mechanism_panel_issue_2227.py)
  runs `SocialForcePlanner` twice on one fixed scenario (seed 42) toggling only `ammv_aware_enabled`
  in `configs/baselines/social_force_ammv_aware.yaml`, exports both arms as schema-validated
  `simulation_trace_export.v1` traces, and renders contrastive trajectory panels via
  `generate_trajectory_panel_bundle`. Applies #2444's finding (PR #3451) that the AMMV term is a
  genuine same-seed divergent pair: AMMV-off max force 0.0 vs AMMV-on 2.64, final-position
  divergence 0.58 m. Evidence is `diagnostic_only`/`stress`, not paper-grade — a planner-level
  mechanism difference, not a navigation-success or benchmark claim. Tracked panels + captions +
  provenance under `docs/context/evidence/issue_2227_ammv_mechanism_panel_2026-06-23/`. The
  static-recentering and topology-guided-recovery panel sub-targets of #2227 remain follow-up
  (Refs #2227).
* Completed #2227 mechanism panels with the static-recentering and topology-guided-recovery
  contrastive sub-targets:
  [`scripts/analysis/build_recenter_topology_panels_issue_2227.py`](scripts/analysis/build_recenter_topology_panels_issue_2227.py)
  runs each mechanism's planner twice on one fixed activation-capable scenario, toggling ONLY the
  mechanism flag (`static_recenter_enabled`; `topology_command_enabled`), exports schema-valid
  `simulation_trace_export.v1` traces, and renders contrastive trajectory panels with per-step
  activation diagnostics and command-source. Findings (real, diagnostic-only/stress): static
  recentering activates (recenter term positive from step 7) and the on-arm reaches the goal where
  the off-arm fails; the topology command activates but the on-arm *degrades* the outcome versus the
  off-arm — reported honestly as a single-row degradation, not a benefit (consistent with the prior
  Issue #2752 "no useful topology alternative" diagnosis). Together with the AMV/AMMV panel sub-target this
  completes #2227. Tracked panels + captions + provenance under
  `docs/context/evidence/issue_2227_recenter_topology_panels_2026-06-23/` (#2227).
* Added a canonical [`docs/glossary.md`](docs/glossary.md) defining the project's acronyms and
  domain terms (VRU, AMV, AMMV, SNQI, occluder, the evidence ladder, and run modes) in plain
  language, and made "understandable" a first-class maintainer value. [`maintainer_values.md`](docs/maintainer_values.md#clarity)
  now carries a `## Clarity` rule, `AGENTS.md` scopes token-efficiency to agent-internal surfaces
  (clarity wins on human-facing surfaces), and the README, docs index, and CONTRIBUTING checklist
  link the glossary so jargon is defined on first use.
* Added a diagnostic-only multi-robot research smoke foothold (#3069): a small
  runnable scenario config (`configs/multi_robot/issue_3069_smoke.yaml`), a smoke
  runner (`scripts/validation/run_multi_robot_smoke_issue_3069.py`) that rolls out
  the multi-robot environment via `make_multi_robot_env` and reports per-agent
  inter-robot collision and goal-progress telemetry from the environment's native
  `info["agents"]` metadata, and a guard test
  (`tests/gym_env/test_multi_robot_smoke_issue_3069.py`). The report carries an
  explicit `claim_boundary: diagnostic_only` / `evidence_tier: smoke` and
  `multi_robot_benchmark_claim: false`; it proves only that the multi-robot path
  runs end-to-end and is not a fleet-scale, MAPPO, or paper-facing claim (#3069).
* Added durable issue-2904 forecast-risk eligibility fixtures
  (`tests/fixtures/benchmark/forecast_risk_eligibility/`) and the regenerated
  `ForecastCalibrationReport.v1` evidence bundle under
  `docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20/`. The pedestrian
  (`deployable_tracked`) and bicycle (`deployable_observation`) fixtures yield calibrated,
  risk-scoring-eligible reliability rows so the forecast-risk calibration filter's
  `calibration_filtered`, `actor_class_aware`, and `observation_tier_aware` modes are no longer
  blocked for lack of eligible rows: the calibration report recommendation flips from
  `wait`/blocked to `continue`/analysis-only and the filter's overall recommendation flips from
  `wait` to `diagnostic_only`. Fixtures and evidence are diagnostic/analysis-only and do not change
  `_risk_scoring_eligibility`, calibration thresholds, or planner risk coupling (#2904).

### Fixed

* Fixed the ORCA-residual BC smoke evidence contract (#1475): the per-decision `action_adaptation`
  payload from `GuardedPPOAdapter` now always carries a truthful `residual_clipped` signal — on the
  direct pass-through, guard-fallback, and default branches no bounded residual is applied, so
  `residual_clipped: False` is emitted (a real signal, not fabricated). This lets the smoke
  finalizer populate `residual_clipping_rate` (and, with the existing `shield_stats`,
  `guard_veto_rate`, `fallback_degraded_status`, `artifact_pointer_status`) instead of failing
  closed on `missing_required_smoke_evidence` — which is what blocked the #1475 SLURM smoke rerun
  (job 12913). Contract-completeness only: it does not make the BC lane succeed (the run may still
  be `timeout_low_progress`/`success_rate=0.0` per #2445), and no fail-closed gate is weakened
  (a genuinely degraded run still classifies degraded). (#1475)

* Fixed `scripts/tools/issue_template_audit.py` so it recognizes the repo's own YAML issue
  forms (`epic`/`execution-run`/`test-debt`/`blocked-external-artifact`) and agent-authored
  bodies instead of only the markdown templates. The audit previously matched a single strict
  12-section markdown contract and reported 8–12 false "missing section" gaps on nearly every
  issue filed from the leaner YAML forms. The auditor now (a) maps heading variants such as
  `Scope and non-goals`/`Out of scope`/`Scope / decision needed`, `Acceptance criteria`/
  `Acceptance / stop rule`/`Accept / revise / reject`, `Estimate`/`Estimate metadata`,
  `Validation commands`, and `Question / Goal`/`Background`/`Summary`/`What to build` to their
  canonical sections; (b) strips trailing parenthetical qualifiers (e.g.
  `Acceptance criteria (mirrors body)`); (c) credits `###` contract subheadings carried inside
  the appended `agent-exec-spec` block; and (d) audits leaner-template issues against a shared
  agent-ready core (`Goal / Problem`, `Scope`, `Definition of Done`, `Validation / Testing`)
  while bare/markdown-style bodies keep the strict full 12-section contract. A standalone effort
  estimate is intentionally not in the core: leaner/agent issues state effort under
  `Project Metadata` or defer it on blocked work, and the agent-exec-spec agents execute from
  carries no estimate. Across the 91 open issues this cut spurious "missing section" findings
  from 762 (≈87 issues, many with 8–12 false gaps) to ~30 (26 issues, genuinely-absent
  sections), with 65 issues now fully clean. The archetype-metadata audit and
  `audit_archetype_metadata` API are unchanged.
* Expanded the canonical issue-archetype taxonomy in
  `docs/context/issue_1512_issue_archetypes.md` and the validator
  (`scripts/tools/issue_template_audit.py`) to bless the work-type archetypes the issue
  automation already emits — `implementation`, `test`, `refactor`, `data` — and to accept the
  deprecated spellings `agent_task` (→ `implementation`) and the evidence tier `proposal`
  (→ `idea`) on read without rewriting issue bodies. This clears the validator's
  invalid-metadata findings on the four affected open issues (#2000, #3069, #3071, #3206)
  without per-issue edits, and wires `implementation`/`data` into
  `issue_archetype_sync.py`'s `SAFE_ARCHETYPE_LABEL_MAP` (`type:implementation`/`type:data`,
  both existing labels).
* Fixed `scripts/validation/validate_forecast_risk_calibration_filter.py` so its
  `_any_eligible_for_risk_scoring` gate recognizes the canonical `eligible_analysis_only` token
  emitted by `build_forecast_calibration_report`. The gate previously matched only the legacy
  `{"eligible", "calibrated"}` strings, which the report never emits, leaving the
  `calibration_filtered` mode permanently blocked for every real calibration report regardless of
  eligible rows. Legacy tokens are retained for backward compatibility (#2904).

### Changed

* Consolidated SDD staging into the canonical external-data subsystem (#3473): the SDD asset
  checksum policy, proxy-vs-dataset-backed availability gate, pinned-`expected_tree_sha256`
  validation, disk-space fail-closed check, and no-auto-download safety contract (originally added
  for #2657) now live in
  [`scripts/tools/manage_external_data.py`](scripts/tools/manage_external_data.py) (new
  `load_sdd_staging_spec`/`validate_sdd_staging`/`resolve_sdd_scenario_prior_mode`/`run_sdd_download`
  functions and `sdd-plan|sdd-status|sdd-validate|sdd-mode|sdd-download` CLI commands). The `sdd`
  `AssetSpec` now references the single staging manifest
  ([`configs/data/sdd_staging_manifest.yaml`](configs/data/sdd_staging_manifest.yaml)) and carries
  the availability/mode states, removing the previous two-sources-of-truth split.
  [`scripts/data/stage_sdd_dataset_issue_2657.py`](scripts/data/stage_sdd_dataset_issue_2657.py) is
  reduced to a thin compatibility wrapper that delegates to the canonical subsystem while preserving
  its CLI/exit surface, and scenario-prior gating
  ([`scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py`](scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py))
  now consumes the canonical gate. Behavior preserved: the proxy-vs-dataset-backed gate stays
  fail-closed and downloads still require explicit `--confirm-download` + y/N (or `--yes`). Migration
  tests in
  [`tests/tools/test_sdd_staging_consolidation_issue_3473.py`](tests/tools/test_sdd_staging_consolidation_issue_3473.py)
  prove the proxy and dataset-backed decisions are identical through the canonical and wrapper paths
  (#3473).
* Refined the `goal-pr-review` skill (`.agents/skills/goal-pr-review/SKILL.md`) for clarity and
  determinism: added a mapping table that bridges `scripts/dev/pr_loop_policy.py` classifications
  (`pending_ci`, `failed_ci`, `missing_artifacts`, `stale_worktree`, `ready_to_merge`, `no_action`)
  to the skill's own state machine, noting that `ready_to_merge` is a candidate signal still gated by
  the intended-design and proof bars; split `## Read First` into always-read versus
  read-when-applicable (`gh-pr-comment-fixer` only for unresolved threads, `review-benchmark-change`
  only for benchmark-facing PRs); and tidied the `## When to use` section formatting. Docs-only; the
  skill validator (`scripts/dev/check_skills.py`) passes.
* Continued the `_build_policy` decomposition (#3384) by migrating the `risk_surface_dwa` and
  `lidar_social_force` adapter families from the inline dispatcher into
  `robot_sf/benchmark/map_runner_policies/adapters.py`, registered in `_POLICY_BUILDERS`. The builders
  reuse the neutral `map_runner_policy_common.build_adapter_policy` (#3403) and import their adapter
  classes from `robot_sf.planner.*`, so no import cycle is introduced. Behavior-preserving (the
  `map_runner` regression suite passes; one test monkeypatch repoints to the new builder namespace).
  Adapter families that still depend on `map_runner`-local helpers (e.g. `trivial_reference` via
  `_build_socnav_config`) are deferred to a later slice.
* Consolidated the RFC6901 JSON-pointer renderer that was copy-pasted in eight modules into a single
  shared helper `robot_sf.common.json_pointer.json_pointer` (#3386). The eight schema-validation call
  sites (`benchmark/{scenario_schema,odd_contract,hazard_traceability,scenario_contract,odd_hazard_coverage_matrix,artifact_catalog}.py`
  and `analysis_workbench/{simulation_trace_export,trace_annotation}.py`) now import it, removing the
  latent hazard of an escaping fix diverging between copies. As part of unifying the two prior copies,
  the three `analysis_workbench`/`artifact_catalog` sites now render the *root* path (empty error path)
  as the RFC6901-correct empty string `""` instead of `"/"`; this only affects the rare root-level
  schema-error message string and is covered by a new unit test
  (`tests/common/test_json_pointer.py`). Also fixed `_copy_figures` in `research/imitation_report.py`
  to actually `return` its `dict[str, Path]` mapping (it previously fell through to `None` despite the
  annotation), with a regression test (#3386).
* Began decomposing the ~4967-line `robot_sf/benchmark/camera_ready_campaign.py` (the largest module in
  `robot_sf/`) by extracting its 6 config dataclasses (`AmvProfileConfig`, `SeedPolicy`,
  `ScenarioCandidateSelection`, `PlannerSpec`, `SnqiContractConfig`, `CampaignConfig`) plus the
  `_AMV_DIMENSIONS` / `DEFAULT_SEED_SETS_PATH` constants into a new
  `robot_sf/benchmark/camera_ready_campaign_config.py` (#3405, first slice of #3385). The names are
  re-exported from `camera_ready_campaign`, so existing imports (e.g. in `release_protocol.py`,
  `orca_preflight.py`, and tests) are unchanged. Behavior-preserving verbatim move; the
  camera-ready/release regression suites pass.
* Consolidated the duplicated durable atomic-JSON-write helper into a single shared
  `robot_sf.common.atomic_io.atomic_write_json` (#3386). The `mkstemp -> fsync -> os.replace` helper
  previously copy-pasted in `benchmark/imitation_manifest.py` and `benchmark/full_classic/io_utils.py`
  now imports the shared version; behavior is preserved (the helper resolves the path and creates the
  parent directory, a safe superset of the prior two). `benchmark/forecast_dataset_recorder.py` keeps
  its own writer for now because it has a different output contract (trailing newline, no `fsync`).
  Covered by `tests/common/test_atomic_io.py`.
* Began decomposing the ~1270-line `_build_policy` dispatcher in `robot_sf/benchmark/map_runner.py`
  into a `robot_sf/benchmark/map_runner_policies/` builder package backed by a registry (#3400, first
  slice of Issue #3384). The built-in goal/simple policy family now lives in `map_runner_policies/goal.py`
  (`build(...) -> (policy_fn, meta)`); `_build_policy` consults `_POLICY_BUILDERS` before its remaining
  inline branches, which are unchanged. Behavior-preserving (the regression net in
  `tests/benchmark/test_map_runner_utils.py` still passes); the only test change repoints one
  monkeypatch to the new builder namespace.
* Relocated the shared `_build_adapter_policy` helper out of `robot_sf/benchmark/map_runner.py` into a
  neutral `robot_sf/benchmark/map_runner_policy_common.py` (`build_adapter_policy`), re-exported under
  the old private name so all ~17 in-module call sites are unchanged (#3403, prerequisite for the
  Issue #3384 adapter-family decomposition). Behavior-preserving — the helper is a verbatim move and the
  `map_runner` regression suite passes. This lets future `map_runner_policies/` builder modules reuse
  the helper without an import cycle back into `map_runner`.
* Cut pull-request CI wall-clock by parallelizing the `fast-feedback` test phase. Pull requests now
  split the suite into 4 `pytest-split` shards (a matrix on the existing job, so the CI topology is
  unchanged) while `main`/`workflow_dispatch` keep a single full pass so the advisory coverage
  baseline stays accurate without cross-shard combining. The shared `run_tests_parallel.sh` wrapper
  gained `PYTEST_SHARD_COUNT`/`PYTEST_SHARD_INDEX` env hooks (`--splits`/`--group`); coverage is
  skipped while sharding. Also enabled the CPython 3.12+ `COVERAGE_CORE=sysmon` backend to lower
  coverage instrumentation overhead, and added a docs-only `paths-ignore` filter so Markdown/`docs/`
  changes no longer trigger the full CI workflow. Adds `pytest-split` to the `dev` dependency group.
* Removed redundant Black/autopep8 formatter configuration and the unused Black optional
  development dependency; Ruff format remains the canonical formatter.
* Extended the issue-1515 hybrid-evidence matrix validator with an opt-in git-history proof path:
  default validation still only checks `commit_artifact` shape plus provenance pointers, while
  `scripts/validation/validate_hybrid_evidence_matrix.py --check-git-history` now additionally fails
  closed on unknown local commit SHAs before issue-1489 synthesis consumes stress/full-matrix rows.
* Updated the issue-1519 predictive ego-feature contract so ego-conditioned schema metadata now
  records a machine-readable motion-channel producer key for same-seed/runtime vs standalone
  collection, and runtime inference rejects explicit standalone-producer checkpoints instead of
  silently treating mixed producer rows as equivalent.
* Tightened the issue-1519 mixed predictive dataset flow so ego-conditioned base/hardcase inputs now
  fail closed on missing or mismatched producer metadata before mixing/training, and successful
  mixed NPZ outputs preserve `feature_schema_json` producer provenance instead of silently dropping
  it.
* Updated the issue-template audit flow for issue-1513 so `scripts/tools/issue_template_audit.py`
  now parses the `## Archetype Metadata` YAML block, flags invalid canonical `archetype` and
  `evidence_tier` values plus malformed metadata, reports those findings alongside the existing
  section audit, and aligns the issue-audit skill docs with the bounded metadata triage behavior.
* Implemented the issue-1187 render-helper slice: `capture_frames()` now samples real RGB frame
  buffers or one direct render result, `generate_video_contact_sheet.py` writes deterministic PNG
  contact sheets from episode frame metadata, and the helper catalog documents the supported
  surfaces and failure modes.
* Updated the issue-1186 post-`#243` output-root cleanup slice so remaining active benchmark and
  baseline commands now point at `output/benchmarks/...`, the factory-performance baseline tooling
  resolves its default path through canonical artifact helpers, and the social-navigation benchmark
  runner no longer advertises a legacy `results/` output root.
* Updated the issue-1180 `goal-pr-review` workflow so autonomous PR review defaults to a
  fix-first repair loop on writable branches: proof failures are now classified as auto-fixable
  now, deferred follow-up, or handoff-only blockers; safe actionable gaps should be repaired and
  revalidated before withholding `merge-ready`; and the skill index plus shared goal-loop note now
  reflect the new validation-and-reassess contract.
* Removed deprecated `robot_sf_snqi recompute --method` aliases (`pareto_optimization`,
  `equal_weights`, and `safety_focused`). Use canonical method names
  `canonical`, `balanced`, or `optimized`; deprecated names now fail argument parsing.

### Fixed

* Filled the issue-1572 AMV actuation metadata gaps so the issue-1556 synthetic slice can carry
  slice-local scenario AMV taxonomy without editing global scenario files, local `social_force`
  rows now expose explicit command-space/projection-policy metadata, and compact actuation summary
  artifacts distinguish AMV coverage status from adapter projection metadata without weakening
  fail-closed unavailable/fallback semantics.
* Fixed the issue-1527 ORCA benchmark preflight gap by moving the `rvo2` fail-fast check into the
  shared camera-ready campaign layer, so direct `prepare_campaign_preflight(...)`,
  `run_campaign(...)`, and wrapper entrypoints such as `scripts/tools/run_benchmark_release.py`
  cannot bypass the actionable `uv sync --extra orca` / `uv sync --all-extras` guidance for
  enabled ORCA-dependent rows.
* Fixed the README Zenodo DOI header by replacing the fragile badge image with a plain-text DOI
  link while preserving the canonical `https://doi.org/10.5281/zenodo.19563812` target.
* Fixed the crowd-only Gymnasium environment contract so `CrowdSimEnv` keeps a stable
  observation-space shape across resets, `make_crowd_sim_env()` preserves preconfigured
  config values unless callers override them explicitly, reset-time map selection no longer
  mutates NumPy's global RNG state, and compact JSONL recordings keep static scene metadata
  on reset events instead of repeating it on every step.

### Added

* Added the `robot_sf/py.typed` marker and `Typing :: Typed` classifier so downstream type checkers
  can consume the package's inline annotations under PEP 561.
* Added the issue-1556 synthetic AMV actuation stress slice: the new
  `configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml` config stays
  non-paper-facing and differential-drive-only, camera-ready campaign manifests/reports now carry
  synthetic actuation provenance plus a diagnostic `actuation_envelope_summary.{json,md}` artifact,
  and the map runner reports auditable saturation metrics or fails closed when the synthetic
  profile cannot be applied.
* Added the issue-1550 predictive same-seed row-summary validation surface:
  `robot_sf/benchmark/predictive_same_seed_row_summary.py` plus
  `scripts/validation/validate_predictive_same_seed_row_summary.py` now validate small durable
  predictive comparison rows for variant/scenario/seed/planner-grid outcomes, provenance pointers,
  and explicit `unavailable`/`unknown` handling, with a checked-in template and focused regression
  coverage under `tests/benchmark/test_predictive_same_seed_row_summary.py`.
* Added the issue-1515 hybrid-evidence matrix validation surface: `robot_sf/benchmark/hybrid_evidence_matrix.py`
  and `scripts/validation/validate_hybrid_evidence_matrix.py` now validate the #1499 row contract
  for enums, nullability, repository-relative provenance, fail-closed fallback/degraded semantics,
  and synthesis-candidate guard-veto consistency, with checked-in fixtures and targeted regression
  coverage under `tests/benchmark/test_hybrid_evidence_matrix.py`.
* Added the issue-1214 docs/proof consistency checker surface: `scripts/validation/check_docs_proof_consistency.py`
  now scans changed docs/evidence files for high-confidence PR handoff drift, the
  `scripts/dev/check_docs_proof_consistency_diff.sh` wrapper exposes the branch-diff command, and
  the contributor workflow docs now point at the new check before proof-heavy PR handoff.
* Added the issue-1185 dummy-backend smoke surface: `robot_sf/sim/backends/dummy_backend.py`
  now exposes the map metadata and minimal simulator contract that `RobotEnv` expects, so
  `examples/advanced/01_backend_selection.py` can run headlessly in CI and stays covered by the
  manifest-driven examples smoke suite.
* Added the issue-1181 `ml-intern` bounded-assistant assessment note, including the local-only
  proof ladder, trace/privacy boundary, verified Robot SF prompt/context stack, and the explicit
  recommendation to keep `ml-intern` as a bounded experiment assistant rather than a replacement
  for the repository's local/HPC/SLURM proof-first workflow.
* Added the issue-1168 multi-AMV planner support classification surface: multi-AMV episode
  metadata now carries explicit planner-family support records, planner support preflight checks
  fail closed for unsupported or smoke-only planner families, and the docs index records the
  current boundary between smoke execution and real multi-robot planner support.
* Added the issue-1153 manual-control replay/export helper surface: BC samples now carry source
  provenance, replay/profile helpers reject directory inputs fail-closed, and replay/export JSON
  writers preserve NumPy-backed payloads by normalizing them into JSON-safe builtins before
  serialization.
* Added the issue-1151 manual-control MVP foundation surface: append-only JSONL recording,
  fail-closed mode/session helpers, baseline comparison primitives, and BC export utilities now
  reject invalid mode values, non-finite speed multipliers, negative tolerances, and malformed
  record payloads while keeping NumPy-backed observations serializable for recorder/export flows.
* Added the issue-1152 manual-control mode experiment surface: typed control/view mode specs,
  cruise and mouse-target differential-drive mapper variants, runtime mode selection config, and
  manifest/record metadata for `control_mode`, `view_mode`, and `input_mapping_version`, while
  keeping unsupported ego-up renderer hooks fail-closed until the camera transform exists.
* Added the issue-1162 manual-control rewind boundary: manual-control records now carry explicit
  rewind metadata, replay JSON preserves rewind events, and BC export skips samples invalidated by
  append-only rewind records while repeated-rewind planning fails closed until active-timeline
  derivation exists.
* Added the issue-1110 CARLA oracle replay parity adapter surface: `compare_oracle_replay_metrics(...)`
  and its CLI now emit conservative parity reports, reject degraded CARLA `mode` or `status`
  fail-closed, treat non-finite numeric values as unavailable instead of serializing invalid JSON,
  and document the adapter as a comparison boundary rather than live CARLA evidence.
* Added the issue-1128 multi-AMV episode extension surface: `multi_amv_episode_extension(...)`
  now emits an additive namespaced `multi_amv` block for smoke outputs, requires explicit
  planner status, omits optional planner notes when absent, and keeps invalid single-robot or
  empty-metric inputs fail-closed through targeted regression coverage and the documented
  multi-AMV smoke validation path.

* Added the issue-857 horizon-alignment experiment surfaces: manifest-level `scenario_overrides`
  support in `robot_sf/training/scenario_loader.py`, the new
  `configs/scenarios/sets/ppo_full_maintained_eval_v1_horizon100.yaml` eval/training surface,
  the seed-123 retrain clone
  `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_horizon100.yaml`,
  the diagnostic benchmark probe
  `configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml`, and focused
  regression coverage for the new horizon-matched workflow.
* Added the issue-856 broad-training evidence surfaces for the completed seed-123 control: the 12223 baseline adapter (`configs/baselines/ppo_issue_856_all_scenarios_12223.yaml`), the dedicated camera-ready comparison matrix (`configs/benchmarks/paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml`), the campaign and pre-PR verification context notes, and the linked experiment-memory write-back. Recorded the verdict that, at fixed 10M budget, broad-training underperforms the eval-aligned leader on the camera-ready matrix (success −0.035, collisions +0.007, SNQI −0.040), strengthening the alignment-vs-diversity claim for the manuscript.
* Added a file-based policy-search workflow under `docs/context/policy_search/` with a canonical candidate registry, stage-gated runner (`scripts/validation/run_policy_search_candidate.py`), comparison/failure/promotion tools, and SLURM handoff notes for training-heavy follow-up work. The first non-training candidate, `hybrid_orca_sampler_v1`, now runs end to end in the benchmark stack and emits markdown/json validation artifacts locally.
* Added the deterministic `hybrid_rule_local_planner` policy-search record and ablation surfaces. The current best non-learning candidate is `hybrid_rule_v3_static_margin0_waypoint2`, which preserved zero collision terminations on nominal sanity and stress slices, while rejected static-escape, route-commitment, route-lookahead, and speed variants remain documented as reproducible ablations rather than promoted planner defaults.
* Added DreamerV3 BR-08 close-out documentation (2026-04-30), including the program-level stop decision, updated full-run outcome note for Slurm 12159, and linked follow-up surfaces from the 2026-04-28 handoff: checked-in Auxme launchers (`SLURM/Auxme/dreamer_br08_gate.sl`, `SLURM/Auxme/dreamer_br08_full.sl`), eval-parity regression coverage for the BR-08 full config, a `#782` pretraining design note, and a fail-closed `#789` note documenting that Ray 2.53.0 DreamerV3 still lacks mixed-observation support without a larger catalog/module fork.
* Documented durable artifact, linked-worktree bootstrap, branch-sync, and
  worktree-output handling rules in `AGENTS.md` and `docs/dev_guide.md`,
  clarifying that `output/` is temporary/local by default; fresh linked
  worktrees should detect the shared main checkout before symlinking
  `local.machine.md`; a worktree counts as fresh only when both
  `local.machine.md` and `.venv` are absent; active feature branches should
  merge latest `origin/main` early and again before PR creation; and PR
  preparation now includes reviewing ignored `output/*` files before handoff.

* Promoted the issue-791 Wave-5 leader (job 11724, WandB `ll7/robot_sf/ibo3aqus`,
  best success 0.929 / collision 0.071 / SNQI 0.353 on
  `ppo_full_maintained_eval_v1`) into the canonical PPO baseline at
  [configs/baselines/ppo_15m_grid_socnav.yaml](configs/baselines/ppo_15m_grid_socnav.yaml).
  All ~17 benchmark configs that reference this file (camera-ready, paper
  matrix, planner sanity, seed variability) now resolve PPO to the eval-aligned
  large-capacity grid_socnav policy with predictive foresight enabled and
  `fallback_to_goal: false` per the fail-closed benchmark policy. The previous
  BR-06 v3 15M artifact remains on disk for reproducibility but is no longer the
  default. Reported performance is benchmark-set (in-distribution), not OOD.

* Restored the asymmetric-critic robot training contract so `RobotEnv(..., asymmetric_critic=True)` and `make_robot_env(..., asymmetric_critic=True)` once again add the critic-only `critic_privileged_state` observation for SocNav structured runs. This keeps the public factory API stable for issue-791 training jobs and closes the regression that broke fresh SLURM submissions after the main-branch merge.

* Corrected the issue-791 reward-curriculum `resume_best` overlay so it preserves the original `grid_socnav` / `MultiInputPolicy` checkpoint contract when warm-starting from 11566 best.zip. This avoids the observation-space mismatch that appeared when the retry was pointed at the wrong model family.

* SVG obstacle-repair warnings now include the source `*.svg` filename, so invalid map repairs point directly to the offending asset; the regression test also checks the captured warning text.
* Added issue-832 paper-matrix extended seed schedules (`paper_eval_s5`, `paper_eval_s10`,
  `paper_eval_s20`), S5/S10 paper-matrix sibling configs that preserve the frozen
  `paper-matrix-v1` contract, `run_camera_ready_benchmark.py --campaign-id` for resumable fixed
  campaign roots, and `scripts/tools/compare_seed_schedule_campaigns.py` for reproducible
  S3-vs-higher-seed CI-width, mean-drift, ranking, and scenario-winner comparisons.

* Added fail-fast SoNIC / GenSafeNav benchmark wrapper aliases for the model-only checkpoint path: base aliases `sonic_crowdnav`, `gensafenav_ours_gst`, and `gensafenav_gst_predictor_rand`, plus guarded variants `ours_gst_guarded` and `gst_predictor_rand_guarded` with short-horizon guard telemetry (`guard_stats`) and goal fallback. This exposes the adapter surface explicitly for users and benchmark configs while keeping remaining risks visible: the base wrappers are still prototype-level model-only integrations, guarded behavior exists only on the guarded aliases, and guarded holonomic `vx_vy` runs now fail closed until the guard can preserve upstream ActionXY without a lossy `(v, w)` round-trip.

* Recorded-step playback analyzer for issue #585: telemetry replay now records per-step reward
  terms plus selected step metrics with episode IDs, JSONL episode metadata can link back to the
  corresponding telemetry stream, and interactive playback can show a synchronized analyzer overlay
  with reward tables, visible-metric filtering, and a scrub-aligned metric timeline while remaining
  backward compatible with older recordings.

* TEB corridor-commitment planner improvements for issue #805 (second iteration): fixed corner-cutting by tuning embedded route-guide `waypoint_lookahead_cells` to 5 (1.0 m target at 0.2 m/cell, full 0.9 m/s speed without diagonal clipping), increased `obstacle_inflation_cells` to 3 for 0.6 m corner clearance, added `stop_distance=0.5` for earlier front-clearance stops, and added `_rescue_or_stop` so all-blocked commitment falls back to the route guide rather than driving into obstacles;  `plan()` refactored into `_try_route_command`,  `_commitment_step`, and `_rescue_or_stop` helpers to satisfy the complexity limit.
* TEB corridor-commitment planner improvements for issue #805 (first iteration): multi-step corridor occupancy scoring, escalated lateral commitment gains, blocked-side flip fallback, route-waypoint guidance integration, and reproducible topology-slice artifacts (`configs/scenarios/sets/issue_805_teb_topology_slice.yaml`,   `docs/context/issue_805_teb_corridor_commitment_iteration.md`).

* Added a repo-local agent memory layer with `CLAUDE.md` startup imports, a canonical
`memory/MEMORY.md` index, typed memory subdirectories ( `architecture` , `decisions` ,
`experiments` , `failures` , `benchmarks` ), example notes for each type, and linked guidance in
`AGENTS.md` , `docs/dev_guide.md` , `docs/README.md` , and the AI-facing overview/deferral docs.
* Added canonical `classic_cross_trap_*` scenario IDs and manifests for the former
`classic_crossing_*` cross-shaped local-minimum trap scenarios, with legacy
  compatibility aliases and migration notes for existing configs.
* Added a canonical Markdown context-note workflow under `docs/context/README.md`, explicit
`AGENTS.md` / Copilot / dev-guide guidance for linked handoff notes, a new
`context-note-maintainer` repo-local skill, and an issue-796 policy note so reusable agent
  knowledge is easier to preserve, discover, and update without adding external infrastructure.

* Added an SB3 SAC training workflow for issue 790 with a config-first entrypoint (`scripts/training/train_sac_sb3.py`), gate/full configs under `configs/training/sac/`, strict config validation, and targeted training tests covering config load plus dry-run checkpoint creation.

* Added an all-scenarios SAC training config with random scenario sampling, non-fixed per-episode seeds, and W&B telemetry enabled (`configs/training/sac/gate_socnav_struct_ego_all_random_wandb_v3.yaml`), plus an autoresearch-list entry for repeatable experiment orchestration.

* Added scenario-weighted SAC curriculum configs plus a small SAC experiment harness (`scripts/tools/sac_autoresearch.py`) so issue-790 training loops can compare benchmark-style evaluation results reproducibly.

* Added config-driven SAC multi-env training support (`num_envs`) in `scripts/training/train_sac_sb3.py`, including subprocess vectorization for `num_envs > 1` and an updated all-scenarios W&B config using `num_envs: 4`.

* Paper results handoff export (`robot_sf.benchmark.paper_results_handoff`) now emits interval-inclusive planner-summary rows from frozen publication bundles, with CLI wrapper (`scripts/tools/paper_results_handoff.py`), confidence-interval metadata, seed/repeat provenance, deterministic JSON/CSV outputs, and contract documentation in `docs/context/issue_750_paper_results_handoff.md`.

* Policy analysis issue-768 coverage now includes ORCA variant CLI entries (`socnav_orca_nonholonomic`,        `socnav_orca_dd`,        `socnav_orca_relaxed`,        `socnav_hrvo`), targeted regression tests for the new builder paths and sweep policy resolver, and an issue note comparing the branch results against `main` plus full-eval artifacts under `output/experiments/768_*`.

* GitHub workflow guidance now prefers MCP / app tools for interactive issue, PR, and project work, keeps `gh` as the documented scripted fallback, and hardens `project_priority_score.py` with clearer auth diagnostics plus a retry path for the `--owner ll7` / `unknown owner type` gh quirk.
* Camera-ready campaign analysis now emits `scenario_difficulty_analysis.{json,md}` plus richer `campaign_analysis.{json,md}` content that ranks scenario difficulty from existing artifacts, summarizes family-level hardness, flags planner residual mismatch on easier scenarios, and records whether the verified-simple subset needs a bounded pilot before it can be treated as a calibration aid.
* Added adversarial pedestrian policy demo (`examples/advanced/32_demo_adversarial_pedestrian.py`) with factory-based environment setup, CLI overrides for map/model paths, and per-episode collision/kinematics summary logging.
* Added pedestrian policy collision benchmark script (`scripts/benchmark_ped_policy_collisions.py`) to evaluate pedestrian checkpoints with episode metrics plus collision kinematics (robot speed, pedestrian speed, and impact angle at robot-pedestrian collision), front/back/side collision-zone percentages, explicit timeout/robot-goal counts, and a console outcome-breakdown log line.
* Added differential-drive pedestrian policy debug runner (`scripts/debug_ped_policy_differential_drive.py`) combining the pedestrian debug loop with run_023-compatible differential-drive robot profile and observation adapter behavior.
* Added differential-drive pedestrian PPO training entrypoint (`scripts/training_ped_ppo_differential_drive.py`) mirroring the existing pedestrian PPO workflow with `DifferentialDriveSettings`.
* Pedestrian collision telemetry now records `collision_impact_angle_rad/deg` in per-step metadata, with aggregated TensorBoard scalars and histograms for overall and per-collision-type impact kinematics.
* APF model comparison benchmark script (`scripts/benchmark_ped_apf_models.py`) to run `run_023` and `run_043` with APF off/on (100 episodes each by default) and report aggregated episode metrics (steps, collisions, success/timeout, rewards) with JSON export under `output/benchmarks/`.
* Added benchmark release protocol v0.1 surfaces: canonical release manifests for the paper-facing matrix, a `run_benchmark_release.py` entrypoint layered on the camera-ready workflow, release/reproducibility docs,          `CITATION.cff`, and a benchmark-focused `RELEASE.md` checklist.
* Shipped adversarial-pedestrian assets for the new demos and training workflow: SVG maps `maps/svg_maps/masterthesis/corner.svg`,         `maps/svg_maps/masterthesis/headon.svg`,         `maps/svg_maps/masterthesis/intersection.svg`, plus pedestrian PPO checkpoints `model/pedestrian/ppo_corner.zip`,         `model/pedestrian/ppo_headon.zip`, and `model/pedestrian/ppo_intersection.zip` used by `examples/advanced/32_demo_adversarial_pedestrian.py`, `scripts/benchmark_ped_policy_collisions.py`, `scripts/debug_ped_policy_differential_drive.py`, `scripts/training_ped_ppo_differential_drive.py`, and `scripts/benchmark_ped_apf_models.py`.
* Added benchmark release protocol v0.1 surfaces: canonical release manifests for the paper-facing matrix, a `run_benchmark_release.py` entrypoint layered on the camera-ready workflow, release/reproducibility docs,         `CITATION.cff`, and a benchmark-focused `docs/RELEASE.md` checklist.
* Added Project #5 task prioritization support: a documented benchmark-oriented scoring model, a `project_priority_score.py` sync helper, and a GitHub Actions workflow for manual/scheduled/issue-event score synchronization into a numeric `Priority Score` field.
* Default PR template now prompts for summary, validation/proof, risks/rollout, docs/provenance, and follow-up issues so reviews stay proof-first and easier to act on.
* Canonical benchmark fallback policy note (`docs/context/issue_691_benchmark_fallback_policy.md`) plus shared availability helpers so benchmark CLI and camera-ready campaigns fail closed on fallback, degraded, skipped, and partial-failure outcomes.
* Repository agent guidance now requires proof-first validation for new planners, metrics, skills, and tests, with task-appropriate executable or reproducible evidence before a change is considered complete.
* Added a Codex-native repository context stack: strengthened `AGENTS.md`, new `docs/code_review.md` benchmark/provenance checklist, `.agents/PLANS.md` planning convention, repo-local context skills under `.agents/skills/`, and AI-facing overview/decision docs under `docs/ai/`.
* Promoted predictive planner checkpoint `predictive_proxy_selected_v2_xl_ego` into the model registry with a W&B-backed download entry (`ll7/robot_sf/i17pmely`).
* Added explicit BR-06 v6, v7, and v9 auto-env PPO resubmit configs for 24-CPU Auxme jobs so queued reruns can resolve `num_envs` from the host allocation instead of staying pinned to 8 env workers.
* Added `scripts/dev/sbatch_use_max_time.sh` plus SLURM submission docs so new batch jobs can query partition/QoS wall-time limits and default to the effective maximum at submit time.
* Added a SLURM resource-audit runbook documenting how to retrieve W&B `system` metrics, compare them against Slurm allocations, and reason about PPO `num_envs` vs reserved CPU count.
* `ClassicGlobalPlanner.plan_random_path()` now supports `allow_inflation_fallback=False` to keep the configured inflated area instead of shrinking inflation during random path sampling.
* Unified configs now support `map_id` for deterministic map selection when building environments.
* Config-first RLlib DreamerV3 training workflow for `drive_state` + `rays`, including:
`scripts/training/train_dreamerv3_rllib.py` , `configs/training/rllib_dreamerv3/drive_state_rays.yaml` ,
  deterministic observation flattening/action normalization wrappers, and a dedicated runbook.
* Optional dependency split for training stacks: `imitation` moved to the `imitation` dependency group so BC pre-training can be installed separately from RLlib-oriented environments.
* Added an explicit uv dependency conflict declaration between `--extra rllib` and `--group imitation` to provide actionable install errors when mixed.
* Scenario split helper (`robot_sf.training.scenario_split`) and CLI (`scripts/tools/split_scenarios.py`) for train/holdout validation.
* Optuna sqlite inspection helper (`scripts/tools/inspect_optuna_db.py`) for quick study summaries.
* Optuna expert PPO sweep report with recommended hyperparameter ranges (`docs/training/optuna_expert_ppo_sweep_2026-02-11.md`).
* Shared occupancy collision helpers to de-duplicate dynamic and obstacle collision checks.
* New force flags `peds_have_static_obstacle_forces` and `peds_have_robot_repulsion` with legacy alias support.
* Benchmark CLI warn-only scenario preview (`robot_sf_bench preview-scenarios`) plus warnings/coverage summary in validate-config output.
* Expert PPO training now supports `ppo_hyperparams`/`best_checkpoint_metric` overrides and saves a best-checkpoint snapshot per run.
* Optuna sweep helper for expert PPO training configs (`scripts/training/optuna_expert_ppo.py`).
* Safety-gated Optuna objectives for expert PPO sweeps (`constraint_collision_rate_max`,
  optional `constraint_comfort_exposure_max` , and `constraint_handling=penalize|prune` )
  with per-trial feasibility metadata and feasible/infeasible study summaries.
* Policy analysis episodes now store `shortest_path_len` in metrics to enable diagnostics of path-efficiency saturation.
* Policy analysis sweep mode can run multiple policies in one invocation (`--policy-sweep`,          `--policies`).
* Policy analysis now records `jerk_mean_eps0p1` and `curvature_mean_eps0p1` with a low-speed filter.
* Policy analysis supports named seed sets (`--seed-set` via `configs/benchmarks/seed_sets_v1.yaml`) and writes combined sweep reports.
* Policy analysis can optionally extract failure frames from report.json (`--extract-frames`), and the frame extractor accepts report inputs.
* Example SocNav social-force algorithm config for map-based benchmarks (`configs/algos/social_force_example.yaml`).
* Fast-pysf ground-truth planner option for scenario video rendering (`--policy fast_pysf` in `scripts/tools/render_scenario_videos.py`).
* Policy analysis sweep script with metrics + optional videos (`scripts/tools/policy_analysis_run.py`).
* Benchmark outputs now include wall collision and clearing distance metrics by default.
* Vendored GA3C-CADRL (SA-CADRL) checkpoint under `model/ga3c_cadrl/` with provenance + license metadata.
* Added `sacadrl` optional dependency extra (TensorFlow) for GA3C-CADRL baseline support.
* ORCA planner uses the rvo2 binding when available; added `orca` optional dependency extra.
* Added SocNavBench subset metrics (`socnavbench_path_length`,             `socnavbench_path_length_ratio`,             `socnavbench_path_irregularity`).
* Map registry (`maps/registry.yaml`) with generator script and `map_id` support for scenario files.
* Occupancy grid polish: ego-frame transforms applied consistently, query aggregation returns per-channel means without scaling errors, new quickstart/advanced/reward-shaping examples (`examples/quickstart/04_occupancy_grid.py`,             `examples/advanced/20_occupancy_grid_workflow.py`,             `examples/occupancy_reward_shaping.py`), and an expanded guide (API/config/troubleshooting + docs index link).
* Telemetry visualization (feature 343): docked Pygame telemetry pane with live charts, JSONL telemetry stream under `output/telemetry/`, replay/export helpers, headless smoke script/test, and a demo (`examples/advanced/22_telemetry_pane.py`).
* Automated research reporting pipeline (feature 270-imitation-report): multi-seed aggregation, statistical hypothesis evaluation (paired t-tests, effect sizes, threshold comparisons), publication-quality figure suite (learning curves, sample efficiency, distributions, effect sizes, sensitivity), ablation matrix orchestration, telemetry section, and programmatic + CLI workflows (`scripts/research/generate_report.py`,             `scripts/research/compare_ablations.py`). Includes success criteria tests and demo (`examples/advanced/17_research_report_demo.py`).
* Research reporting polish: metadata manifest aligned with `report_metadata` schema, schema validation tests for metrics/hypotheses, and smoke/performance harnesses (`scripts/validation/test_research_report_smoke.sh`,             `scripts/validation/performance_research_report.py`,             `tests/research/test_performance_smoke.py`,             `tests/research/test_schemas.py`).
* Multi-extractor training now auto-collects convergence/sample-efficiency metrics, baseline comparisons, and learning-curve/reward-distribution figures, emitting schema-compliant summaries (`summary.json`/`summary.md`) plus legacy `complete_results.json`.
* New extractor report generator: `scripts/research/generate_extractor_report.py` converts multi-extractor `summary.json` into research-ready `report.md`/`report.tex` with figures, reproducibility metadata, and baseline comparisons.
* Inkscape SVG map template (`maps/templates/map_template.svg`) and quickstart updates in `docs/SVG_MAP_EDITOR.md`.
* Classic planner backend option in `RobotSimulationConfig` (`planner_backend="classic"` with `planner_classic_config`) plus ORCA demo consuming it (`examples/advanced/31_classic_planner_orca_demo.py`); planner attachment now supports grid-based planning without manual wiring.
* Scenario-level single-pedestrian overrides now support POI-based goals/trajectories, per-ped speeds, wait notes, and a preview helper (`scripts/tools/preview_scenario_trajectories.py`) documented in `docs/single_pedestrians.md`, including clean rendering and `--all` batch export.
* Single pedestrian runtime behavior controller with waypoint advancement, wait handling, and role tags (follow/lead/accompany/join/leave) for robot-relative interactions.
* Added Francis 2023 crowd/traffic scenario maps (crowd navigation, parallel/perpendicular traffic, circular crossing, robot crowding) and scenario entries in `configs/scenarios/francis2023.yaml`.
* SVG map inspection helper (`scripts/validation/svg_inspect.py`) plus reusable API (`robot_sf.maps.verification.svg_inspection`) for route-only mode detection, route/zone index checks, risky path-command warnings, and obstacle-interior route checks.
* Full classic benchmark now emits `run_meta.json` traceability metadata (repo/branch/commit, CLI invocation, matrix path, seed plan, environment), with a mirror at `artifacts/<run_id>/run_meta.json` for paper artifact workflows.
* Full classic benchmark supports freeze-manifest validation (`--freeze-manifest`) with runtime contract checks (matrix hash/path, baselines/planner config, seed plan, metric subset, bootstrap settings, software identifiers) and structured status/mismatch reporting in `run_meta.json`.
* Added a machine-readable freeze manifest template at `configs/benchmarks/classic_benchmark_freeze.example.yaml` and documented the pilot -> freeze -> final workflow.
* Benchmark episodes now include canonical threshold profile metadata (`metric_parameters.threshold_profile` + signature).
* Aggregation now validates threshold-profile consistency and rejects mixed-threshold reports.
* Added threshold sensitivity tooling (`scripts/benchmark_threshold_sensitivity.py`) to report near-miss and comfort-threshold impacts by scenario family, including speed-aware alternatives.
* Root `README.md` now includes an explicit attribution section documenting that this project builds on Caruso et al. (Machines 2023) and listing upstream reference repositories.
* Root `README.md` attribution section now also documents fast-pysf acknowledgements (svenkreiss/socialforce and pedsim_ros) and links the core Social Force references (Helbing & Molnar 1995; Moussaid et al. 2010).

### Fixed

* Fixed scenario-driven timeout semantics so `max_episode_steps` now expires on the configured
  discrete step count instead of one step late from a floating elapsed-time comparison.

* Legacy `scripts/training_ppo.py` invocations now fail closed and point to the canonical PPO
  training workflow docs.
* Restored the existing `socnav_sacadrl` CLI choice in `scripts/tools/policy_analysis_run.py` after the issue-768 ORCA variant additions, preventing a regression where the builder still supported SA-CADRL but the parser rejected it.
* Paper results handoff export now fails closed on malformed metadata, validates JSONL rows as objects while streaming them, aggregates planner rows per run to reduce peak memory pressure, and writes standards-compliant JSON/CSV artifacts without permissive `NaN` serialization.

* Restored the backward-compatible `avg_collision_impact_angle_rad` pedestrian metric alias and corrected the adversarial-force plot labels in `scripts/debug_ped_apf.py`.
* Issue-708 PPO SLURM launcher now runs the training command directly in the batch shell instead of using a nested `srun`, avoiding an immediate allocation confirmation failure on Auxme.
* Differential-drive kinematics now match standard straight-line and in-place rotation formulas.
* Auxme GPU SLURM submissions now default to `48G` host memory instead of the previous `120G`, matching the documented PPO host-memory sizing guidance more closely while preserving headroom for `num_envs` auto-resolution on 24-CPU jobs.
* Auxme GPU SLURM submissions now default to the current `a30` partition maximum wall time instead of the previous 8-hour fallback.
* Auxme GPU batch runs now bootstrap the environment modules shell function when available and default staging/results paths to repo-local `output/slurm/` locations instead of failing on unavailable `/scratch/${USER}` paths.
* Auxme GPU batch runs now support forwarding extra `uv run` flags, allowing shared max-time submissions for jobs such as `uv run --extra rllib python ...`.
* Expert PPO warm-start runs now emit direct W&B `rollout/*`,             `train/*`, and `time/*` metrics during training so resumed runs show progress before the next scheduled evaluation checkpoint and do not depend solely on TensorBoard discovery.
* `robot_sf/benchmark/perf_trend.py` now treats `--history-limit 0` as "load no history reports" instead of unintentionally loading all matched files.
* `robot_sf_bench plot-scenarios` now avoids thumbnail filename collisions for name-only matrices by applying deterministic `id -> name -> scenario_id -> hash` fallback resolution, sanitization, and collision suffixing.
* DreamerV3 RLlib launcher now hardens Ray runtime environment setup by disabling `uv run` worker propagation, pinning worker interpreter execution, and applying packaging excludes to reduce startup fragility and upload size.
* Dreamer observation contract now emits float32-consistent reset/step payloads (`drive_state`,  `rays`) so Gymnasium space checks pass without dtype warnings.
* Simulation throughput perf guard now supports cluster-aware calibration through env overrides and enforce mode instead of hard-failing heterogeneous nodes by default.
* SVG map parsing now honors indexed spawn/goal labels to keep route ordering stable.
* Pedestrian env now validates robot action-space compatibility and falls back to null actions on mismatch.
* SocNavBenchSamplingAdapter now shares the upstream loader logic with SamplingPlannerAdapter to avoid drift and keep the vendored planner path working.
* SocNavBench path irregularity now uses heading vectors instead of origin-dependent position vectors.
* SocNavBench upstream planner loader rejects untrusted roots unless explicitly allowed via `ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT`.
* Vendored SocNavBench helpers now guard missing data files, fix numpy API mismatches, and correct plotting/trajectory utilities.
* Renamed `thrid_party` to `third_party` to fix the vendor directory typo.
* Vendored the MIT-ACL Python-RVO2 fork to keep ORCA builds compatible with newer CMake versions.
* SocialForcePlannerAdapter now uses fast-pysf social-force interactions (goal, pedestrian, obstacle forces) instead of the heuristic placeholder.

### Added

* Map Verification Workflow (Feature 001-map-verification)
  + Single-command map validation tool (`scripts/validation/verify_maps.py`) for SVG map quality checks
  + Rule-based validation engine checking file readability, SVG syntax, file size, and layer organization
  + Scope filtering supporting 'all', 'ci', 'changed', specific filenames, or glob patterns
  + Structured JSON/JSONL manifest output for tooling and dashboard integration
  + CI mode with strict exit codes for automated quality gates
  + Loguru-based diagnostics with human-readable console output
  + Map inventory system with tag-based classification and filtering
  + Verification results include timing, rule violations, and remediation hints
  + Documentation in `docs/SVG_MAP_EDITOR.md` and `specs/001-map-verification/quickstart.md`
  + Sample manifest artifacts under `output/validation/`
* Run tracking & telemetry for the imitation pipeline (Feature 001-performance-tracking)
  + Progress tracker with deterministic step ordinals, ETA smoothing, and manifest-backed step history
  + JSONL manifests enriched with telemetry snapshots, rule-based recommendations, and perf-test results stored under `output/run-tracker/`
  + CLI tooling (`scripts/tools/run_tracker_cli.py`) for status/watch/list/show/export/summary plus optional TensorBoard mirroring
  + Performance smoke CLI (`scripts/telemetry/run_perf_tests.py`) that wraps the existing validation harness and records pass/soft-breach/fail statuses with remediation hints
  + Documentation updates spanning quickstart, dev guide, and docs/README.md so teams can enable the tracker and interpret telemetry in CI or local runs
  + CI guard step invoking `scripts/validation/run_examples_smoke.py --perf-tests-only` so the tracker smoke + telemetry perf wrapper fail fast before pytest
* PPO Imitation Learning Pipeline (Feature 001)
  + Expert PPO training workflow with convergence criteria and evaluation schedules
  + Trajectory dataset collection and validation utilities
  + Behavioral cloning (BC) pre-training from expert demonstrations
  + PPO fine-tuning with warm-start from pre-trained policies
  + Comparative metrics CLI for sample-efficiency analysis
  + Playback and inspection tool for trajectory datasets
  + Bootstrap confidence intervals for metric aggregation
  + Complete artifact lineage tracking (expert → dataset → pre-trained → fine-tuned)
  + Configuration dataclasses for all imitation workflows
  + Integration tests for end-to-end pipeline validation
  + Sample-efficiency target: ≤70% of baseline timesteps to convergence
  + Documentation in `docs/dev_guide.md` and `specs/001-ppo-imitation-pretrain/quickstart.md`
  + Default pretraining configs for behavioral cloning and PPO fine-tuning (`configs/training/ppo_imitation/bc_pretrain.yaml`,        `configs/training/ppo_imitation/ppo_finetune.yaml`)
* Canonical artifact root enforcement and tooling (Feature 243)
  + Introduced `output/` hierarchy as single destination for coverage, benchmark, recording, wandb, and tmp artifacts
  + Added migration helper (`scripts/tools/migrate_artifacts.py`) and guard (`scripts/tools/check_artifact_root.py`) with console entry point and regression tests
  + Wired guard + migration into CI workflow, publishing artifacts from canonical paths only
  + Refreshed core docs (`docs/dev_guide.md`,               `docs/coverage_guide.md`,   `docs/README.md`, root `README.md`) with policy overview, quickstart links, and updated coverage instructions
  + Extended quickstart guidance to cover guard execution, artifact overrides, and validation expectations
* Visibility-graph global planner (Feature 342)
  + Planner API with POI routing, caching, smoothing; new `use_planner` flag and clearance config in unified configs
  + Map POI parsing, planner demo (`examples/advanced/20_global_planner_demo.py`), validation script (`scripts/validation/verify_planner.sh`), and benchmark script (`scripts/benchmark_planner.py`)
  + Tests for path planning, POI sampling, caching, smoothing, and navigation integration under `tests/test_planner/`
* Comprehensive configuration architecture documentation (#244)
  + Created `docs/architecture/configuration.md` with configuration precedence hierarchy
  + Documented three-tier precedence system: Code Defaults < YAML < Runtime
  + Added migration guide from legacy config classes to unified config
  + Documented all configuration modules (canonical vs legacy)
  + Linked from `docs/README.md` and `docs/dev_guide.md`
* Automated example smoke harness (`scripts/validation/run_examples_smoke.py`,               `tests/examples/test_examples_run.py`) wired into validation workflow (#245)

### Fixed

* Pedestrian route sampling now fails fast when obstacle constraints cannot be satisfied instead of silently ignoring obstacles.
* Model registry resolution now fails fast for `local_only` entries with explicit replacement guidance instead of attempting ambiguous off-machine fallback, and predictive promotion metadata now preserves portable W&B entries when provenance is supplied as either a full run path or split run identifiers.

### Changed

* Retired the PAT-dependent `project-priority-score-sync.yml` GitHub Actions workflow and updated Project #5 prioritization docs to treat `scripts/tools/project_priority_score.py sync` as the supported local/manual score-sync path.
* Policy analysis timestamps now use CET/CEST (Europe/Berlin) instead of UTC for output folders and episode metadata.
* Policy analysis runs now force `use_planner=False` in the episode config to avoid attaching global planners during metrics/video sweeps.
* Classic global planner defaults now use 0.5m grid cells (`cells_per_meter=2`) with zero inflation for shortest-path planning, reducing invalid start/goal cell failures.
* Default global planner selection now prefers the classic Theta* (v2) grid planner, and benchmark shortest-path calculations use the same planner.
* `SimulationView` now starts with lidar-ray visualization disabled by default (`show_lidar=False`); use `show_lidar=True` or press `O` to toggle observation-space overlays (auto-selecting lidar/grid/image based on the active observation mode).
* Occupancy grid rasterization now logs out-of-bounds obstacle segments at DEBUG instead of the custom SPAM level.
* Benchmark CLI list-algorithms now reports only implemented baseline planners to avoid registry KeyErrors.
* Expert PPO training and trajectory collection now honor scenario YAML entries, including map files, simulation overrides, and scenario identifiers, while publishing `scenario_coverage` metadata consistent with dataset validators.
* **[BREAKING for internal imports]** Consolidated utility modules into single `robot_sf/common/` directory (#241)
  + Moved `robot_sf/util/types.py` → `robot_sf/common/types.py`
  + Moved `robot_sf/utils/seed_utils.py` → `robot_sf/common/seed.py` (renamed)
  + Moved `robot_sf/util/compatibility.py` → `robot_sf/common/compat.py` (renamed)
  + Removed empty `robot_sf/util/` and `robot_sf/utils/` directories
* Example catalog reorganization and automation improvements (#245)
  + Moved benchmark and plotting scripts into dedicated `examples/benchmarks/` and `examples/plotting/` tiers
  + Regenerated manifest-backed `examples/README.md` and refreshed docs (`README.md`,   `docs/README.md`,               `docs/benchmark*.md`,               `docs/distribution_plots.md`) to reference new paths
  + Updated `examples/examples_manifest.yaml` metadata (tags, CI flags, summaries) and added quick links from docs
  + Imitation pipeline example now auto-selects simulator backends and generates run-specific BC/PPO configs under `output/tmp/imitation_pipeline/` to keep CLI invocations aligned with script requirements
* Visualization stack ownership clarified: the Full Classic pipeline (`robot_sf.benchmark.full_classic.visuals.generate_visual_artifacts`) is now the canonical path that emits manifest-backed plot/video artifacts; the legacy helper API (`robot_sf.benchmark.visualization.*`) is deprecated for benchmark runs and retained only for ad-hoc JSONL plotting.

### Fixed

* Policy analysis reports now include video links for problem episodes when videos are recorded, by attaching metadata after video write.
* Model registry W&B downloads now handle file-like download responses, avoiding path resolution errors.
* Policy analysis video sweeps now close environments via `exit()` to flush recordings, preventing empty output folders.
* OSM map conversion now decomposes obstacle polygons with holes before building `MapDefinition` obstacles, preventing walkable areas from being treated as obstacles during spawning and grid generation.
* Resolved OSM dependency pinning by aligning `networkx` with current `osmnx` constraints to avoid unsatisfiable installs.
* OSM driveable-area fallback now checks obstacle containment safely, preventing `AttributeError` when `allowed_areas` is absent.
* OSM map extraction now subtracts explicit obstacle features from walkable areas and respects configured buffer widths.
* OSM background rendering now validates projected CRS usage and records the UTM CRS in metadata for correct meter-scale overlays.
* Scenario switching during training now tolerates observation-space bound differences across maps while still enforcing action-space compatibility, preventing crashes when sampling mixed-size scenarios.
* Scenario-level single-pedestrian overrides now clone map definitions to prevent cross-scenario contamination during randomized training.
* Goal-distance observations now use a fixed 50m cap with runtime clipping, keeping observation bounds consistent across mixed map sizes.
* SocNav structured observations now cap map-dependent position bounds at 50m with clipping to avoid scenario-switching bound mismatches.
* Expert PPO evaluation now applies the same environment overrides as training, preventing observation-space mismatches during evaluation.
* fast-pysf group gaze force now guards against zero-distance divisions to prevent training crashes.
* Expert PPO CLI now supports `--log-file` to tee stdout/stderr into a log file.
* Scenario loader map resolution now anchors relative paths to `base_dir` directories instead of CWD, preventing off-by-one directory lookups and unintended map matches.
* Pedestrian route sampling now raises a descriptive error instead of dropping obstacle checks after repeated anchor failures.

### Documentation

* Reorganized documentation index with categorized sections (#242)
  + Added clear navigation sections: Getting Started, Benchmarking & Metrics, Tooling, Architecture & Refactoring, Simulation & UI, Figures & Visualization, Performance & CI, Hardware & Environment
  + Added cross-links between core guides for improved discoverability
  + Normalized H1 headings and purpose statements across key documentation files
  + Collapsed legacy detailed index into expandable section for backward compatibility
* Added a benchmark spec doc covering scenario splits, seeds, baseline categories, reproducible commands, and metric caveats.
* Updated benchmark spec for seed-holdout evaluation and policy sweep commands; documented map_id usage in scenario README.


### Migration Guide (Version 2.1.0)

**For robot_sf developers and contributors:**

All utility imports must be updated to reference `robot_sf.common` :

```python
# Before (old paths - no longer valid)
from robot_sf.util.types import Vec2D, RobotPose
from robot_sf.utils.seed_utils import set_global_seed
from robot_sf.util.compatibility import validate_compatibility

# After (new paths - required)
from robot_sf.common.types import Vec2D, RobotPose
from robot_sf.common.seed import set_global_seed
from robot_sf.common.compat import validate_compatibility

# Convenience imports also available:
from robot_sf.common import Vec2D, RobotPose, set_global_seed
```

**Why this change?**
* Eliminates navigation confusion from fragmented utility locations
* Improves IDE autocomplete and discoverability
* Reduces cognitive load for new contributors
* Establishes single canonical location for all shared utilities

**Impact:**
* ~50 import statements updated across codebase
* All 923 tests passing after migration
* No functional changes - pure reorganization

**For external consumers (if any):**
If your project imports from `robot_sf.util` or `robot_sf.utils` , update your imports using the patterns above. The behavior of all utilities remains unchanged.

### Added

* **Architecture Decoupling (Feature 149)**: Introduced simulator facade and backend/sensor registries scaffolding behind the existing factory pattern. Default backend is "fast-pysf" with future backend selection via unified config.
* Backend registry integrated into environment initialization: `BaseEnv` now resolves the simulator via a backend key (`env_config.backend`, default "fast-pysf") using `robot_sf.sim.registry`, with a safe fallback to legacy `init_simulators()` for full backward compatibility.
* **fast-pysf Integration Improvements (Feature 148)**: Enhanced fast-pysf integration with comprehensive testing and quality tooling
  + **Map Verification Enhancements (001-map-verification)**
    - Added informational rule `R005` (layer statistics) emitted when Inkscape-labeled groups (`<g inkscape:label="…">`) are present
    - Enhanced `R004` message and remediation guidance (descriptive labeling for obstacles/spawns/waypoints)
    - Inserted Map Verification section into `docs/benchmark.md` (CI invocation, rule table, manifest structure, extension guidance)
    - Added labeled example SVG (`maps/svg_maps/labeled_example.svg`) for demonstrating layer labeling and future semantic tagging
    - Improved internal type hints (`_load_inventory` return type, expanded docstrings) for verification modules
    - Test coverage extended (`test_layer_stats_info`) validating scope resolution and future R005 visibility

  + **Unified Test Suite**: Single `uv run pytest` command now executes both robot_sf (881 tests) and fast-pysf (12 tests) for total 893 tests
  + **Quality Tooling Extension**: Extended ruff linting, ty type checking, and coverage reporting to include fast-pysf subtree
  + **Type Annotations**: Added comprehensive type hints to fast-pysf public APIs (map_loader, forces, simulator, scene modules)
  + **Configuration**: Per-file ignores in pyproject.toml for gradual quality adoption, ty configuration includes fast-pysf/pysocialforce
  + **Code Quality**: Fixed circular imports, removed dead code, alphabetized imports, replaced wildcard imports
  + **Documentation**: Created annotation_plan.md with numba compatibility guidelines and implementation strategy
* **fast-pysf Subtree (Feature 146)**: Integrated `fast-pysf` as a git subtree for easier management and updates
* **pytest-cov Integration (Feature 145)**: Comprehensive code coverage monitoring and CI/CD integration
  + **Automatic Collection**: Coverage data collected automatically during test runs via `pytest-cov` without additional commands
  + **Multi-Format Reports**: Terminal summary, interactive HTML (`htmlcov/index.html`), and machine-readable JSON (`coverage.json`)
  + **Baseline Comparison**: CI/CD pipeline compares coverage against baseline with non-blocking warnings on decreases
  + **VS Code Integration**: Tasks for "Run Tests with Coverage" and "Open Coverage Report"
  + **CI/CD Workflow**: GitHub Actions integration with caching, baseline updates on main branch, and artifact uploads
  + **Library Infrastructure**:
    - `robot_sf/coverage_tools/baseline_comparator.py`: CoverageSnapshot, CoverageBaseline, CoverageDelta entities with comparison logic
    - `robot_sf/coverage_tools/report_formatter.py`: Multi-format report generation (terminal/JSON/markdown)
    - `scripts/coverage/compare_coverage.py`: CLI tool for local and CI baseline comparison
  + **Comprehensive Testing**: 18 unit tests (5 smoke + 13 baseline comparator) with 91.51% coverage of comparison logic
  + **Documentation**:
  + `docs/coverage_guide.md`: 500+ line comprehensive guide with quickstart, CI integration, troubleshooting
  + `examples/plotting/coverage_example.py`: Programmatic usage examples
    - Updated `docs/dev_guide.md` with coverage workflow section
  + **Configuration**: pyproject.toml with [tool.coverage.*] sections, automatic pytest integration, parallel execution support
  + Coverage excludes: tests, examples, scripts, fast-pysf submodule per omit configuration
  + Non-blocking CI design: warnings only, no build failures on coverage decreases
* Paper Metrics Implementation (Feature 144): Comprehensive implementation of 22 social navigation metrics from paper 2306.16740v4 (Table 1):
  + **NHT (Navigation/Hard Task) Metrics (11)**: `success_rate`,               `collision_count`,               `wall_collisions`,               `agent_collisions`,               `human_collisions`,               `timeout`,               `failure_to_progress`,               `stalled_time`,               `time_to_goal`,               `path_length`,  `success_path_length` (SPL)
  + **SHT (Social/Human-aware Task) Metrics (14)**: velocity statistics (`velocity_min/avg/max`), acceleration statistics (`acceleration_min/avg/max`), jerk statistics (`jerk_min/avg/max`), clearing distance (`clearing_distance_min/avg`),               `space_compliance`,               `distance_to_human_min`,               `time_to_collision_min`,               `aggregated_time`
  + Extended `EpisodeData` dataclass with optional `obstacles` and `other_agents_pos` fields for enhanced collision detection
  + Internal helper functions: `_compute_ped_velocities`,               `_compute_jerk`,               `_compute_distance_matrix`
  + Comprehensive unit test coverage (30+ tests) with edge case validation
  + All metrics documented with formulas, units, ranges, and paper references
  + Backward compatible integration with existing benchmark infrastructure
* Multi-extractor training flow refactor (Feature 141): `scripts/multi_extractor_training.py` now uses shared helpers in `robot_sf.training`, emits schema-backed `summary.json`/`summary.md` plus legacy `complete_results.json`, captures hardware profiles, honors macOS spawn semantics, and ships default/GPU configs alongside updated SLURM automation and analyzer support.
* Helper Catalog Consolidation (Feature 140): Extracted reusable helper logic from examples and scripts into organized library modules:
  + `robot_sf.benchmark.helper_catalog`: Policy loading (`load_trained_policy`), environment preparation (`prepare_classic_env`), and episode execution (`run_episodes_with_recording`) helpers.
  + `robot_sf.render.helper_catalog`: Directory management (`ensure_output_dir`) and frame capture utilities (`capture_frames`).
* Automated Research Reporting: Multi-seed aggregation & completeness (Feature 270-imitation-report)
  + Seed orchestration (`orchestrate_multi_seed`) combining baseline/pretrained manifests
  + Per-seed manifest parsing (`extract_seed_metrics`) tolerant of JSON/JSONL tracker outputs
  + Completeness scoring (`compute_completeness_score`) with PASS/PARTIAL classification
  + Standardized seed failure logging (`log_seed_failure`) and graceful missing-manifest handling
  + Seed summary table & completeness JSON artifact (`completeness.json`) rendered into `report.md`
  + Tracker manifest parsing (`parse_tracker_manifest`) for run-level status integration
  + Extended aggregation exports (JSON + CSV) and hypothesis evaluation incorporated in multi-seed flow
  + All US2 tasks (T033–T043) implemented with unit & integration test coverage
  + `robot_sf.docs.helper_catalog`: Documentation index management (`register_helper`) for automated helper catalog updates.
  + Complete refactoring of all maintained examples (`examples/demo_*.py`) and scripts (`scripts/*.py`) to use helper catalog functions instead of duplicate logic.
  + Helper registry data structures with typed interfaces for discoverable, testable helper capabilities.
  + All helper functions include comprehensive docstrings, error handling, and Loguru logging compliance.
* Episode Video Artifacts (MVP):
  + New CLI flags for benchmark runner: `--no-video` and `--video-renderer=synthetic|sim-view|none`.
  + Synthetic lightweight encoder that renders a red-dot path from robot positions and writes per‑episode MP4s under `results/videos/`. <!-- active-docs-check: allow historical changelog path -->
  + Episode JSON schema extension to include optional `video` manifest `{path, format, filesize_bytes, frames, renderer}`.
  + End‑to‑end wiring through CLI → batch runner → worker → episode; deterministic file naming `video_<episode_id>.mp4`.
  + Tests: CLI integration (`tests/test_cli_run_video.py`) and programmatic API (`tests/unit/test_runner_video.py`), both skipped when MoviePy/ffmpeg unavailable.
* Environment Factory Ergonomics (Feature 130): Structured `RenderOptions` / `RecordingOptions`, legacy kw mapping layer (`fps`,               `video_output_path`,               `record_video`), precedence normalization and logging diagnostics; performance guard (<10% creation mean regression) and new migration guide (`docs/dev/issues/130-improve-environment-factory/migration.md`). New example: `examples/demo_factory_options.py`.
* Governance: Constitution version 1.2.0 introducing Principle XII (Preferred Logging & Observability) establishing Loguru as the canonical logging facade for library code and prohibiting unapproved `print()` usage outside sanctioned CLI/test contexts.
* Documentation: Development guide updated with new Logging & Observability section summarizing usage guidelines (levels, performance constraints, acceptable exceptions).
* SVG Map Validation (Feature 131): Manual bulk SVG validation script (`examples/svg_map_example.py`) supporting strict/lenient modes, summary reporting, environment override (`SVG_VALIDATE_STRICT`), and compliance spec (FR-001–FR-014). Added missing spawn/goal zones and minimal `robot_route_0_0` / `ped_route_0_0` paths to classic interaction SVG maps and large map asset now includes explicit width/height and minimal routes.

### Fixed

* Normalized obstacle vertices to tuples to prevent ambiguous NumPy truth-value error during SVG path obstacle conversion in large map (`map3_1350_buildings_inkscape.svg`).
* Preserved per-algorithm benchmark aggregation (Feature 142): classic orchestrator now mirrors `algo` into `scenario_params`, aggregation raises on missing metadata, and summary outputs emit `_meta` diagnostics plus warnings when expected baselines are absent.
* Full Classic visuals: auto video renderer now falls back to synthetic when all SimulationView encodes fail, recording the downgrade in `performance_visuals`.

### Migration Notes

* No code changes required; existing Loguru usage already compliant. Any remaining incidental `print()` in library modules should be migrated opportunistically (PATCH) unless tied to user-facing CLI UX.

### Added (Performance Budget Feature 124)

* Per-test performance budget enforcement (soft 20s, hard 60s) with slow test report (top 10) and guidance suggestions.
* Environment variables: `ROBOT_SF_PERF_RELAX` (suppress soft breach enforcement) and `ROBOT_SF_PERF_ENFORCE` (escalate soft breaches to failures).
* Shared performance utilities (`tests/perf_utils/`): policy, reporting, guidance, minimal scenario matrix helper.
* Refactored benchmark integration tests (resume, reproducibility) to use minimal matrix for faster deterministic runs.
* Synthetic slow test and guidance validation tests.

### Added

* Classic Interactions PPO Visualization (Feature 128) – deterministic PPO-driven classic interaction scenario visualization script (`examples/classic_interactions_pygame.py`) with:
  + Constants-based configuration (no CLI) and dry-run validation path
  + Deterministic seed ordering & structured episode summaries (scenario, seed, steps, outcome, success/collision/timeout booleans, recorded)
  + Graceful recording guard (moviepy/ffmpeg optional) with informative skip notes
  + Logging verbosity toggle (`LOGGING_ENABLED`) and performance-friendly frame sampling
  + Improved model load error guidance (actionable download/help message)
  + Headless safety via SDL_VIDEODRIVER=dummy detection
  + Reward fallback integration log (env already falls back to simple_reward)
  + Summary table printer helper for human-readable output
* Benchmark visual artifact integration (plots + videos manifests) for Full Classic Interaction Benchmark:
  + Post-run single-pass generation of placeholder plots and representative episode videos
  + SimulationView-first architecture with graceful synthetic fallback (current release uses synthetic until replay support added)
  + Deterministic selection (first N episodes) and machine-readable manifests: `plot_artifacts.json`,               `video_artifacts.json`,               `performance_visuals.json`
  + Renderer attribution field (`renderer`) and budget timing flags
  + Renderer toggle flag (`--renderer=auto|synthetic|sim-view`) with forced mode diagnostics
  + Replay capture adapter enabling SimulationView reconstruction (episode + step validation)
  + Extended skip-note taxonomy: `simulation-view-missing`,               `moviepy-missing`,               `insufficient-replay`,               `render-error:<Type>`,               `disabled`,               `smoke-mode`
  + Performance split metrics (`first_video_render_time_s`,               `first_video_encode_time_s`) plus memory sampling & over‑budget flags
  + Dependency matrix + lifecycle documentation (`docs/benchmark_visuals.md`) covering fallback ladder and required optional deps (pygame, moviepy/ffmpeg, jsonschema, psutil)
* **Social Navigation Benchmark Platform** - Complete benchmark infrastructure for reproducible social navigation research
* **Full Classic Interaction Benchmark (Initial Implementation)**
  + Synthetic placeholder execution pipeline (planning → execution → aggregation → effect sizes → precision loop)
  + Adaptive sampling with CI half-width early stop targets (collision_rate, success_rate, placeholder snqi target)
  + Plot artifacts (distribution, trajectory, KDE/Pareto/force heatmap placeholders) and annotated video generation (graceful fallback)
  + CLI `scripts/classic_benchmark_full.py` exposing comprehensive flags (episodes, precision thresholds, videos)
  + Manifest instrumentation for runtime, episodes_per_second, scaling_efficiency placeholder metrics
  + Resume idempotency and performance smoke tests (T042, T044) plus failure injection test for videos (T043)
  + Dedicated documentation page `docs/benchmark_full_classic.md`

  + **Episode Runner**: Parallel execution with manifest-based resume functionality
  + **CLI Interface**: 15 comprehensive subcommands covering full experiment workflow
    - `run` - Execute episodes with parallel workers
    - `baseline` - Compute baseline statistics
    - `aggregate` - Generate summaries with bootstrap confidence intervals
    - `validate-config` - Schema validation for scenarios
    - `list-scenarios` - Display scenario configurations
    - `figure-*` - Distribution plots, Pareto frontiers, force field heatmaps, thumbnails
    - `table` - Generate baseline tables (Markdown/LaTeX)
    - `snqi-*` - SNQI weight recomputation and ablation analysis
    - Additional utilities for trajectory extraction and episode validation
  + **SNQI Metrics Suite**: Composite Social Navigation Quality Index with component breakdown
  + **Statistical Analysis**: Bootstrap confidence intervals and robust aggregation
  + **Unified Baseline Interface**: PlannerProtocol for consistent algorithm comparison
  + **Figure Orchestrator**: Publication-quality visualization pipeline
  + **Comprehensive Testing**: 108 tests including 33 new tests for benchmark functionality

* Classic interactions refactor (Feature 139): Extracted small reusable visualization and formatting helpers into `robot_sf.benchmark` (`visualization.py`,               `utils.py`) and added contract tests and a dry-run smoke test. See docs/dev/issues/classic-interactions-refactor/design.md.
* **Complete Documentation**: Step-by-step quickstart guide with example workflows
* **Performance Validation**: 20-25 steps/second with linear parallel scaling

### Changed

* **fast-pysf Type Annotations (Feature 148)**: Enhanced type safety across fast-pysf public APIs
  + Added return type annotations to all Force classes (`__call__` methods return `np.ndarray`)
  + Added type hints to Simulator and Simulator_v2 classes (step methods, state management)
  + Improved PedState class with proper array type annotations (`np.ndarray | None` for optional attributes)
  + Added `str | Path` type support to map_loader.load_map() for flexible path handling
* Classic interactions pygame demo now respects per-scenario `map_file` entries in the scenario matrix: on each selected scenario it loads the referenced SVG (via converter) or JSON map definition and injects a single-map `MapDefinitionPool` into the environment config. Falls back gracefully (with a warning) to the default pool if loading fails.
* SVG map conversion now validates presence of at least one `robot_route_*_*` path; missing robot routes raises a clear `ValueError` (callers can fallback to default maps) preventing downstream division-by-zero in simulator initialization. Added richer conversion logging (route and zone counts).
* Enhanced baseline planner interface with unified PlannerProtocol
* Improved test coverage with comprehensive benchmark validation
* Updated main documentation with prominent benchmark platform section
* Simplified video encode invocation by replacing signature introspection with ordered fallback attempts (keyword → positional → minimal) for improved maintainability and clearer failure surface.

### Technical Details

* All baseline planners (SocialForce, PPO, Random) now implement PlannerProtocol interface
* Episode schema includes comprehensive metadata and provenance tracking
* Deterministic execution with seed-based reproducibility
* Manifest-based resume enables incremental experiment extension
* Bootstrap statistical analysis provides uncertainty quantification
* Publication-ready figure generation with LaTeX table support

### Migration Notes

* No breaking changes to existing functionality
* New benchmark CLI commands available via `python -m robot_sf.benchmark.cli`
* Legacy environment interfaces remain fully supported
* Comprehensive backward compatibility maintained

### Fixed

* **fast-pysf Code Quality Issues (Feature 148 / PR #236)**: Resolved 24 PR review comments
  + Fixed circular import in forces.py (changed `from pysocialforce import logger` to `from pysocialforce.logging import logger`)
  + Removed dead code from scene.py (commented desired_directions method)
  + Alphabetized imports in simulator.py for consistency
  + Removed duplicate simulator assignment in ex09_inkscape_svg_map.py
  + Replaced wildcard import with explicit import in TestObstacleForce.py
  + Fixed file path resolution in test_map_loader.py to work with dynamic paths
  + Added per-file ruff ignores for complexity and print statements in examples
* Video artifact manifest now emits a per-episode `skipped` entry with `moviepy-missing` note instead of silently omitting episodes when SimulationView encoding is unavailable for that episode only.
* Robot / multi-robot environments now gracefully fallback to `simple_reward` when `reward_func=None` is passed via factory functions, preventing a `TypeError: 'NoneType' object is not callable` during `env.step` (affects new classic interactions PPO visualization demo).

---

## Guidelines for Contributors

When adding entries to this changelog:

1. **Group changes** by `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
2. **Write for users** - focus on user-visible changes and their benefits
3. **Include migration notes** for breaking changes
4. **Reference related issues/PRs** where applicable
5. **Keep descriptions concise** but informative

## Version Numbering

This project uses semantic versioning:
* **MAJOR** version for incompatible API changes
* **MINOR** version for backwards-compatible functionality additions
* **PATCH** version for backwards-compatible bug fixes
