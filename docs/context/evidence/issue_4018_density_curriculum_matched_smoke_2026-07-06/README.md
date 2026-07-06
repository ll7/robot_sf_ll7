# Issue #4018 Closure Audit And Matched Smoke Evidence

Plain-language summary: issue #4018 asked for an opt-in pedestrian-density curriculum
training lane, a fixed-density comparator, diagnostic reporting, and an explicit
non-benchmark claim boundary. The merged implementation PRs provide the code path, and this
2026-07-06 CPU run exercises the completed matched smoke comparison through the merged readiness
and report builder.

## Closure Decision

Recommended disposition: **close #4018 once this note merges**.

The original implementation criteria are met for the repository-supported CPU diagnostic smoke
scope. The remaining stronger empirical step is a full benchmark or longer training campaign, which
is outside this issue's no-Slurm/no-campaign closure-audit lane and outside the claim boundary below.

Claim boundary: this is diagnostic integration evidence only. It is not benchmark evidence, not a
training-result quality claim, and not a paper or dissertation claim.

## Acceptance Criteria To Evidence

| Criterion | Evidence | Status |
| --- | --- | --- |
| Curriculum schedule parses and fails closed on invalid configs. | PR #4169 added `robot_sf/training/density_curriculum.py` plus focused parser/validator tests. The reproduced validation on 2026-07-06 included `tests/training/test_density_curriculum.py`. | Met |
| Disabled or missing curriculum preserves current behavior. | PR #4169 added disabled-schedule and existing-config compatibility paths. The reproduced validation included disabled no-op coverage in `tests/training/test_density_curriculum.py` and smoke config loading in `tests/training/test_train_ppo_density_curriculum.py`. | Met |
| Training environment applies active density/difficulty stage at reset or rollout-safe boundary. | PR #4169 wired `ScenarioSwitchingEnv.set_curriculum_timestep()` and the PPO callback. The reproduced validation included `tests/training/test_density_curriculum_env.py` and `tests/training/test_train_ppo_density_curriculum.py`. | Met |
| Fixed-density comparator config exists with matched seeds, scenario family, timesteps, network settings, and evaluation cadence. | PR #4169 added `configs/training/ppo/ablations/issue_4018_density_curriculum_smoke.yaml`, `configs/training/ppo/ablations/issue_4018_fixed_density_smoke.yaml`, and `scripts/training/run_density_curriculum_comparison.py`. The 2026-07-06 run used those configs and produced a non-dry-run `density_curriculum_comparison.v1` manifest with both arms at 96 timesteps. | Met |
| Diagnostic comparison is reproducible from configs and metadata. | PR #4478 added `issue_4018.density_curriculum_readiness.v1` readiness checks. PR #4580 added completed-checkpoint artifact writing and `scripts/training/compare_density_curriculum_results.py`. The 2026-07-06 run produced `readiness.json` with `status: ready_diagnostic_smoke`, no blockers, and `comparison_report.json`. | Met |
| Report includes success rate, collision rate, eval return, final stage reached, sample-efficiency proxy, and runtime. | `comparison_report.json` records success rate, collision rate, eval return, final stage reached, timesteps to convergence, total timesteps executed, and runtime for both arms. The Markdown report is summarized below. | Met |
| Claim boundary remains explicit and excludes benchmark-strength or paper-facing claims. | PRs #4169, #4478, and #4580 each state diagnostic/readiness boundaries. This note, `comparison_manifest.json`, `readiness.json`, and `comparison_report.json` all repeat that the evidence is not benchmark evidence and not a training-result claim. | Met |
| Full curriculum benchmark campaign or longer training run. | Live comments mentioned a full curriculum benchmark run as still open after #4478. The current closure-audit directive excludes campaigns/Slurm/GPU and treats a campaign-only remainder as complete for this lane. | Out of scope here |

## Diagnostic Smoke Result

| Metric | Curriculum (`issue_4018_density_curriculum_smoke`) | Fixed-density (`issue_4018_fixed_density_smoke`) |
| --- | --- | --- |
| Final stage reached | `dense` | `N/A (Disabled)` |
| Success rate mean | `0.0000` | `0.0000` |
| Collision rate mean | `0.0000` | `0.0000` |
| Evaluation episode return mean | `-0.8846` | `-1.6966` |
| Timesteps to convergence | `96.0` | `96.0` |
| Total steps executed | `96` | `96` |
| Runtime, wall clock | `6s` | `6s` |

Interpretation: the matched smoke run proves the training and comparison path completes and reaches
the readiness/reporting contract. It does not prove curriculum performance.

## Commands

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/training/test_density_curriculum.py \
  tests/training/test_density_curriculum_env.py \
  tests/training/test_train_ppo_density_curriculum.py \
  tests/training/test_density_curriculum_comparison.py \
  tests/training/test_density_curriculum_readiness.py \
  tests/training/test_compare_density_curriculum_results.py \
  -q
```

Result: `19 passed in 2.62s`.

```bash
CUDA_VISIBLE_DEVICES= scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/training/run_density_curriculum_comparison.py \
  --curriculum-config configs/training/ppo/ablations/issue_4018_density_curriculum_smoke.yaml \
  --baseline-config configs/training/ppo/ablations/issue_4018_fixed_density_smoke.yaml \
  --output-dir <ignored local validation directory>
```

Result: exit `0`; completed both 96-timestep CPU smoke arms and updated the local non-dry-run
comparison manifest with checkpoint artifact paths.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/training/check_density_curriculum_readiness.py \
  <ignored local validation directory>/comparison_manifest.json \
  --output <ignored local validation directory>/readiness.json
```

Result: exit `0`; `status: ready_diagnostic_smoke`, `blockers: []`.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/training/compare_density_curriculum_results.py \
  --manifest <ignored local validation directory>/comparison_manifest.json \
  --output-dir <ignored local validation directory>
```

Result: exit `0`; wrote `comparison_report.md` and `comparison_report.json`.

## Artifact Disposition

Tracked durable evidence:

- `docs/context/evidence/issue_4018_density_curriculum_matched_smoke_2026-07-06/README.md`
- `docs/context/evidence/issue_4018_density_curriculum_matched_smoke_2026-07-06/comparison_report.json`
- `docs/context/evidence/issue_4018_density_curriculum_matched_smoke_2026-07-06/readiness.json`

Local generated artifacts intentionally not tracked:

- `<ignored local validation directory>/`
- `<ignored local policy artifacts>`
- `<ignored local episode traces>`
- `<ignored local run summaries>`

Those local artifacts are diagnostic smoke byproducts and are not durable benchmark artifacts.
