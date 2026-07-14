> Note: all `output/benchmarks/splits/...` paths in this packet are generated at plan time
> by `run_split_camera_ready_campaign.py plan` (gitignored), not tracked configs.

# Release 0.0.3 (h600/s30 extended, 14 arms) — SLURM execution packet + local fallback

Date: 2026-07-13. Worktree: `rebase-0_0_3-campaign` at commit `00c5a2410`.

Provisioning-only note: nothing in this document runs a benchmark, submits Slurm, or
constitutes benchmark evidence. It plans, documents, and verifies tooling (`plan`/`aggregate`
are CPU-only per their own docstrings) so a human can execute the campaign safely.

## 0. Campaign under plan

- Campaign config: `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml`
  (untracked/new in this worktree) — 14 planner arms, `horizon: 600`, `workers: 8`,
  `seed_policy.seed_set: paper_eval_s30` (30 seeds), scenario matrix
  `configs/scenarios/classic_interactions_francis2023.yaml`.
- Release manifest: `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3.yaml`
  (untracked/new) — `release_id: paper_experiment_matrix_v2_h600_s30_v0_0_3`, pins
  `campaign_config_sha256` to the campaign config above and lists 10 `artifacts.required_paths`.
- Episode math: 47 scenarios × 30 seeds × 14 arms = 19,740 episodes (1,410 episodes/arm).
- Why not the single-process route: `scripts/tools/run_benchmark_release.py --mode run` always
  calls `run_campaign(cfg, ...)` in-process, arm-by-arm, in one Python process. The prior
  in-process h600/s30 6-arm run (issue #4826) accumulated a GPU-memory leak across arms and
  failed after **14 hours** of compute (`docs/context/issue_4826_camera_ready_gpu_lifecycle.md:22`:
  "A 2 MiB allocation failing on a 44 GiB card after 14 hours"). On this 24 GB local machine,
  with no discrete GPU (`Darwin`/Apple Silicon) and several arms loading PyTorch/TensorFlow
  checkpoints, the same in-process route is a proven OOM/leak risk, not merely a slow one.

## 1. Splitter manifest + execution packet (done, verified)

### Splitter-manifest format (read from `scripts/tools/run_split_camera_ready_campaign.py`)

`_validated_children()` (lines 79-149) requires:

```jsonc
{
  "source_config": "<path to the parent campaign YAML>",   // must exist, is_file()
  "source_sha256": "<sha256 hex of source_config, exactly 64 chars>",
  "children": [
    {
      "filename": "<config file relative to the manifest's own directory>",
      "sha256": "<sha256 hex of that child file, exactly 64 chars>",
      "planner_keys": ["<planner key>", "..."]   // non-empty, unique across all children
    }
  ]
}
```

Each child file must parse as YAML with a top-level `seed_policy` mapping. Every declared
digest is re-verified at `plan` time (and again at `aggregate`/packet-load time), so a stale
or hand-edited child fails closed with a specific `ValueError`.

### How the packet was generated

The repo already ships the generator for exactly this manifest shape:
`scripts/tools/split_campaign_config_by_planner.py` (issue #5251). It splits one campaign YAML
into one child per enabled planner arm, writes `split_manifest.json` next to the children, and
refuses to run if the campaign already has a `split_provenance` block (no double-splitting).

Commands actually run in this worktree:

```bash
uv run python scripts/tools/split_campaign_config_by_planner.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml \
  --out-dir output/benchmarks/splits/paper_experiment_matrix_v2_h600_s30_extended
# -> Wrote 14 split config(s): output/benchmarks/splits/.../split_manifest.json

uv run python scripts/tools/run_split_camera_ready_campaign.py plan \
  --split-manifest output/benchmarks/splits/paper_experiment_matrix_v2_h600_s30_extended/split_manifest.json \
  --output-root output/benchmarks/release_v0_0_3_arms \
  --campaign-prefix release_0_0_3_h600_s30 \
  --packet output/benchmarks/release_v0_0_3_split_plan/execution_packet.json
# -> exit 0, "status": "planned_not_executed", 14 arms
```

No manifest-validation errors occurred — the splitter tool produces an already-valid manifest,
so no fix-up loop was needed. `plan` was re-run a second time to confirm exit code `0`
deterministically.

**Artifacts (all under this worktree, not committed):**

| Artifact | Path |
|---|---|
| Per-arm child configs + manifest | `output/benchmarks/splits/paper_experiment_matrix_v2_h600_s30_extended/` (14 `*.yaml` + `split_manifest.json`) |
| Execution packet (the plan output) | `output/benchmarks/release_v0_0_3_split_plan/execution_packet.json` |
| Per-arm campaign roots (created when arms run) | `output/benchmarks/release_v0_0_3_arms/release_0_0_3_h600_s30__arm_<key>/` |

The 14 arm keys (in packet order) and their exact planned command (identical shape, `<key>`
substituted):

```text
prediction_planner, goal, social_force, orca, ppo, socnav_sampling, sacadrl,
scenario_adaptive_hybrid_orca_v1, scenario_adaptive_hybrid_orca_v2_collision_guard,
hybrid_rule_v3_fast_progress_static_escape, hybrid_rule_v3_fast_progress_static_escape_continuous,
guarded_ppo, predictive_mppi, risk_dwa
```

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config output/benchmarks/splits/paper_experiment_matrix_v2_h600_s30_extended/paper_experiment_matrix_v2_h600_s30_extended__arm_<key>.yaml \
  --output-root output/benchmarks/release_v0_0_3_arms \
  --campaign-id release_0_0_3_h600_s30__arm_<key> \
  --mode run --skip-publication-bundle
```

Because `--campaign-id` is fixed per arm and the campaign config has `resume: true`, re-running
the identical command after an interruption resumes the same campaign root instead of starting
over (`run_camera_ready_benchmark.py --campaign-id` docstring, line ~68).

### Important OOM-relevant finding: `workers: 8` survives the split unchanged

The splitter only overrides `name`, `planners`, and adds `split_provenance`
(`split_campaign_config_by_planner.py:148-153`); it does **not** reduce the inherited top-level
`workers: 8`. Each arm's campaign run uses a `ProcessPoolExecutor` (`robot_sf/benchmark/map_runner.py:6,2613,3199`)
with `workers` OS processes. For the 5 checkpoint-loading arms (see §4), **each of the 8 worker
processes independently imports PyTorch/TensorFlow and loads the arm's checkpoint** — this
process-level multiplication, not the checkpoint file size itself (94 MB combined in
`output/model_cache` today), is the dominant memory-tier driver in §3.

If a human wants to reduce this risk further, the lever is a per-planner `workers:` override
inside the single planner stanza of an arm's child YAML (`robot_sf/benchmark/camera_ready/_config.py:865`,
`workers_override`); e.g. adding `workers: 2` under the one `planners: - key: ...` entry in
`..._arm_ppo.yaml` before re-running `plan` (the child's sha256 in `split_manifest.json` must be
recomputed and updated first, or the digest check fails closed). This was **not** applied here —
it changes wall-clock/parallelism, so it is left as a documented option, not a silent edit.

## 2. SLURM reachability + cluster conventions

**SLURM is not reachable from this machine.** `which sbatch sinfo squeue` all report
"not found" — this is a local development machine (macOS/Darwin, 24 GB RAM), not a login node.

`~/.ssh/config` (read-only, no secrets shown) lists a cluster login host consistent with the
repo's own cluster docs:

```text
Host licca licca-li-01
  HostName licca-li-01.rz.uni-augsburg.de
```

This matches `SLURM/Licca/README.md`, the repository's public LiCCA cluster guide:

- Login node `licca-li-01`; AMD EPYC compute nodes, 128 cores each.
- Partitions: `epyc` (general CPU), `epyc-mem` (4 TiB RAM), `epyc-gpu` (3× A100 80GB),
  `epyc-gpu-sxm` (4× A100-SXM 80GB), `xeon-gpu` (H100-NVL), `test` (short runs).
- Filesystem: home `/hpc/gpfs2/home/u/$USER` (backed up), scratch
  `/hpc/gpfs2/scratch/u/$USER` (not backed up), node-local `/tmp` (800 GB, wiped at job exit),
  `/dev/shm` RAM disk counts toward `--mem`.
- The README states "the cluster does not support `uv`" and documents a conda/micromamba path
  (`setup_conda_environment.sh`), **but** `SLURM/Licca/setup_uv_environment.sh` exists and stages
  a working `uv` (via `curl -LsSf https://astral.sh/uv/install.sh`) into
  `/hpc/gpfs2/scratch/u/$USER/venvs/robot-sf-uv` after loading `miniforge`/`gcc` modules — so a
  `uv`-based flow (matching this worktree's `.venv` and the packet's `uv run python ...` commands)
  is available; confirm with the human whether the scratch uv env is already provisioned before
  assuming either path.
- The generic `SLURM/slurm_train.sl` / `SLURM/feature_extractor_comparison/*.slurm` templates add
  the shared conventions used repo-wide: `module purge` first, `export OMP_NUM_THREADS=1` when the
  job itself parallelizes (as this campaign's `ProcessPoolExecutor` does), headless rendering guards
  (`DISPLAY=""`, `MPLBACKEND=Agg`, `SDL_VIDEODRIVER=dummy`), and `#SBATCH --output=output/slurm/%j-<description>.out`
  (job ID first, gitignored `output/`).
- `docs/dev/slurm_submission.md` documents `scripts/dev/sbatch_use_max_time.sh` as the preferred
  wrapper over raw `sbatch --time=...` (it queries live partition/QoS `MaxTime` instead of a
  hardcoded value) — recommended for the arm submissions here.
- Auxme-specific scripts under `SLURM/Auxme/*.sl` are stubs in the public repo ("This Auxme batch
  script moved to the private operations overlay") — they require `ROBOT_SF_PRIVATE_OPS`, which is
  not configured here and not needed: LiCCA is the public, documented cluster path for this task.
- No `local.machine.md` exists in this worktree, so `allow_slurm_submission` is unset — per
  `docs/dev/slurm_submission.md`, autonomous `sbatch` is not permitted from this session even if
  SLURM were reachable, which it is not.

**GPU note for the `ppo` arm specifically:** `configs/baselines/ppo_15m_grid_socnav.yaml:39` sets
`predictive_foresight_device: cuda` (hardcoded, not `auto`). `robot_sf/planner/predictive_foresight.py:46`
passes this string straight through to the foresight model's device; there is no CPU fallback in
that path. This arm needs a GPU partition (`epyc-gpu`/`epyc-gpu-sxm`/`xeon-gpu` on LiCCA); the other
13 arms either force `predictive_device: cpu` explicitly (`prediction_planner`, `predictive_mppi`)
or have no device field (classical arms) or use `device: auto` (`guarded_ppo`), which degrades to
CPU cleanly.

## 3. Per-arm sbatch template + memory tiers

Template: `SLURM/submit_release_campaign_arm.sh` (new, executable). Design:

- Takes `ARM_KEY` (required) and `PACKET` (defaults to the packet path above) via
  `sbatch --export=...`; looks up that arm's exact `command` and `campaign_root` from the packet
  JSON with `mapfile` (not `read var1 var2`, which would corrupt on the command's internal spaces
  — verified in a local dry run with a synthetic packet before relying on it).
- Sets `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
  (macOS/torch segfault guard locally; correct thread-limiting on Linux compute nodes too, given
  the `ProcessPoolExecutor` fan-out described in §1).
- Refuses to run (`exit 2`) if `CHECKPOINT_STAGING_REPORT` is set but missing or
  `submit_safe != true` — a runtime backstop for the pre-sbatch staging contract in §4, not a
  replacement for it.
- Built-in resume: the looked-up command already carries the arm's fixed `--campaign-id`.
- Logs to both the sbatch `--output` (set at submission time, not hardcoded in the template) and
  a per-arm tee target `output/slurm/logs/<jobid>-<arm_key>.log`.
- Exit code passthrough matches `scripts/tools/run_camera_ready_benchmark.py`'s own contract
  (`0` success, `2` unexpected failure, `3` accepted-unavailable-only).

Verified locally (no real benchmark executed): syntax-checked with `bash -n` and `shellcheck`
(zero findings), then exercised against a synthetic packet with `exit 0`, `exit 3`, a missing
staging report, `submit_safe: false`, and `submit_safe: true` — all five cases behaved as
designed (correct log path, correct message, correct propagated exit code).

### Memory tiers

| Tier | Arms | `--mem` | Why |
|---|---|---|---|
| Heavy | `prediction_planner`, `predictive_mppi`, `ppo`, `guarded_ppo`, `sacadrl` | `16G` | All 5 are exactly the arms `scripts/benchmark/preflight_campaign_checkpoints.py --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml --json` reports as checkpoint-bound (verified locally, see §4 output). `ppo`/`guarded_ppo` load PyTorch (`stable_baselines3`, `robot_sf/baselines/ppo.py`) policy + feature-extractor checkpoints; `prediction_planner`/`predictive_mppi` load a PyTorch predictive proxy model (`robot_sf/planner/predictive_model.py`); `sacadrl` loads a TensorFlow GA3C-CADRL checkpoint (`robot_sf/planner/socnav.py:35`, `import tensorflow.compat.v1 as tf`). With `workers: 8` (ProcessPoolExecutor, unchanged by the split — §1), each arm's job runs up to 8 independent processes that each import a full deep-learning framework and load the checkpoint; framework import + simulation state easily reaches ~1.5-2 GB RSS per worker for these arms, i.e. ~12-16 GB total — 16G is a workable but not generously margined tier. Checkpoint bytes on disk are small (94 MB combined for the 2 currently cached models), so file size is not the driver — process multiplication is. |
| Light | `goal`, `social_force`, `orca`, `socnav_sampling`, `risk_dwa`, `scenario_adaptive_hybrid_orca_v1`, `scenario_adaptive_hybrid_orca_v2_collision_guard`, `hybrid_rule_v3_fast_progress_static_escape`, `hybrid_rule_v3_fast_progress_static_escape_continuous` | `8G` | Confirmed by grep: none of these 9 arms' `algo_config` files (or their absence, for arms with no `algo_config`) declare `model_id`/`model_path`/`sacadrl_model_id`/`predictive_model_id` — classical/rule-based planners (ORCA, social-force, DWA-style risk, hybrid-rule ORCA blends). 8 worker processes of NumPy-only simulation state is comfortably under 8G in the prior h100 7-arm run's non-`prediction_planner` arms (see §6 for the exact numbers this is based on). |

**Wall-clock is a separate axis from memory.** Within the heavy tier, only `prediction_planner`
and `predictive_mppi` do expensive **per-simulation-step** predictive rollout computation; `ppo`,
`guarded_ppo`, and `sacadrl` do one cheap forward pass per step. The h100/3-seed evidence in §6
shows `prediction_planner` alone was 53% of a 7-arm campaign's wall time while the other 6 arms
(including `ppo` and `sacadrl`) were comparatively fast — i.e., "heavy" here means memory-tier
grouping, not a claim that all 5 arms are equally slow.

### Submission loop (documented, not executed — SLURM unreachable + no submission authorization)

```bash
# 1. One-time: refresh the split + packet if the campaign config changed since §1.
# 2. Stage checkpoints once for the whole campaign (§4) BEFORE any sbatch call.
# 3. Submit all 14 arms, tier-appropriate --mem, arm-specific --job-name/--output:

declare -A HEAVY=( [prediction_planner]=1 [predictive_mppi]=1 [ppo]=1 [guarded_ppo]=1 [sacadrl]=1 )
PACKET=output/benchmarks/release_v0_0_3_split_plan/execution_packet.json
STAGING_REPORT=output/benchmarks/release_v0_0_3_split_plan/checkpoint_staging.json

for key in prediction_planner goal social_force orca ppo socnav_sampling sacadrl \
           scenario_adaptive_hybrid_orca_v1 scenario_adaptive_hybrid_orca_v2_collision_guard \
           hybrid_rule_v3_fast_progress_static_escape \
           hybrid_rule_v3_fast_progress_static_escape_continuous \
           guarded_ppo predictive_mppi risk_dwa; do
  MEM=8G; PARTITION=epyc
  if [[ -n "${HEAVY[$key]:-}" ]]; then MEM=16G; fi
  if [[ "$key" == "ppo" ]]; then PARTITION=epyc-gpu; fi   # hardcoded cuda foresight device, §2
  scripts/dev/sbatch_use_max_time.sh \
    --sbatch-arg --job-name=rsf-rel003-"$key" \
    --sbatch-arg --output=output/slurm/%j-rel003-"$key".out \
    --sbatch-arg --mem="$MEM" \
    --sbatch-arg --partition="$PARTITION" \
    --sbatch-arg --export=ALL,ARM_KEY="$key",PACKET="$PACKET",CHECKPOINT_STAGING_REPORT="$STAGING_REPORT" \
    SLURM/submit_release_campaign_arm.sh
done
```

Partition names above (`epyc`, `epyc-gpu`) come from `SLURM/Licca/README.md`; confirm live
availability with `sinfo` on the login node before submitting, since this session cannot.

## 4. Checkpoint staging (issue #4613 contract)

`docs/context/issue_4613_camera_ready_checkpoint_provisioning.md` is the runbook: the strict
all-rows campaign policy turns one un-loadable arm checkpoint into a whole-campaign failure
**after** compute has already run (the runbook cites jobs 13296/13301 failing ~14h in on a
missing PPO `model_cache` entry). The fix is provisioning at submit time, not inside the job.

Verified locally (network-free `metadata_only` mode, no download attempted):

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml \
  --json --report-path /tmp/staging_check.json
```

Result: `"checked": 5, "resolved": 5, "stage": false, "submit_safe": false`. Exactly 5 arms
declare a checkpoint — `prediction_planner`, `ppo`, `sacadrl`, `guarded_ppo`, `predictive_mppi`
(the same 5 as the heavy memory tier above, confirmed by running the tool, not assumed).
`sacadrl`'s default `ga3c_cadrl_iros18` checkpoint is already `present_local` in this worktree
(`model/ga3c_cadrl/IROS18/network_01900000.meta`); the other 4 are `stageable_remote`
(registry-declared GitHub-release/W&B sources, not yet downloaded into `output/model_cache`,
which today only holds 94 MB for 2 unrelated models) — hence `submit_safe: false`.

**Required pre-sbatch step** (not run in this session — it downloads model weights over the
network, which is out of scope for a planning-only pass):

```bash
scripts/benchmark/submit_camera_ready_checkpoint_gate.sh \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml \
  --report-path output/benchmarks/release_v0_0_3_split_plan/checkpoint_staging.json
```

This wraps `preflight_campaign_checkpoints.py --stage --json`, downloads + checksum-verifies each
registry checkpoint into the durable cache, and exits `3` (do-not-submit) on any unresolvable
checkpoint. Run it **once against the full parent campaign config** (not per-arm child config) —
all arms share the same `output/model_cache`, so one gate call stages everything the 5 heavy arms
need; then pass `--report-path` through to every `sbatch --export=...CHECKPOINT_STAGING_REPORT=...`
call in §3 so the per-arm template's runtime guard can confirm `submit_safe: true` before it runs.

Note that `run_camera_ready_benchmark.py --checkpoint-preflight-mode` only affects the
**preflight-only** CLI path; `--mode run` (what every arm in the packet uses) always keeps the
cheap `metadata_only` guard and expects checkpoints already staged
(`scripts/tools/run_camera_ready_benchmark.py:84-97`) — the gate above is not optional for the 4
`stageable_remote` arms; skipping it reproduces the exact issue #4613 failure mode.

## 5. Post-run path: aggregate, publication bundle, and the release runner

### `aggregate` (verified locally against the just-generated packet, before any arm has run)

```bash
uv run python scripts/tools/run_split_camera_ready_campaign.py aggregate \
  --packet output/benchmarks/release_v0_0_3_split_plan/execution_packet.json \
  --output-dir output/benchmarks/release_v0_0_3_split_plan/aggregate_dryrun
```

Result (expected, since no arm has executed yet): `exit 2`, `"status": "blocked"`, 14
`missing_campaign_artifacts` exclusions (one per arm, each missing `campaign_manifest.json` and
`campaign_summary.json` under its not-yet-created campaign root). This confirms the tool fails
closed correctly rather than silently reporting a false success — it is safe to re-run this exact
command as arms complete; it will report `native_aggregate_complete` only once every arm's root
has compatible native rows and no arm is missing/incompatible (`_row_exclusion_reason`,
`aggregate_execution_packet` in `run_split_camera_ready_campaign.py:254-460`).

The aggregate's own schema (`split-camera-ready-native-aggregate.v1`) carries an explicit
`claim_boundary`: *"Native-row aggregation of independently executed split campaigns. Adapter,
fallback, degraded, unavailable, failed, missing, and incompatible rows are excluded and cannot
support benchmark-success claims in this aggregate."* It writes only 3 files at the aggregate
output dir: `aggregate_manifest.json`, `reports/campaign_summary.json`,
`reports/campaign_report.md`.

### Can `benchmark_publication_bundle.py` consume the aggregate? Mechanically yes, substantively no.

`scripts/tools/benchmark_publication_bundle.py export --run-dir <dir>` calls
`export_publication_bundle()` → `list_publication_files()`
(`robot_sf/benchmark/artifact_publication.py:157-186`), which recursively globs **any** non-hidden
file under `run-dir` with no required filename — so pointing `--run-dir` at the aggregate output
dir would technically produce *a* bundle (the 3 files above). `_validate_publication_requirements()`
(`artifact_publication.py:430-438`) only enforces the stricter preflight-artifact completeness
check **when `campaign_manifest.json` exists** in the run dir — the aggregate dir never writes
that file, so the check silently no-ops. This means a thin, mechanically-valid bundle is possible,
but it is not a release-manifest-validated bundle (next point) and should be labeled with the
aggregate's own claim boundary if ever exported this way.

### Can `run_benchmark_release.py` consume the aggregate or resume/report over an existing root? No — verified by reading the whole file.

`scripts/tools/run_benchmark_release.py` (`parse_release_args` in `robot_sf/benchmark/release_protocol.py:652-674`)
exposes exactly `--manifest`, `--output-root`, `--label`, `--mode {run,preflight}`. There is:

- **No `--campaign-id` / `--campaign-root` flag** (unlike `run_camera_ready_benchmark.py`, which
  has one specifically for resuming a fixed root).
- **No report-only mode.** `--mode preflight` calls `prepare_campaign_preflight(cfg, ...)`, which
  generates fresh preflight artifacts, not a report over completed results.
  `--mode run` (the default) **always** calls `run_campaign(cfg, output_root=..., label=...,
  skip_publication_bundle=True, invoked_command=...)` (`run_benchmark_release.py:221-227`) — i.e.
  it always executes the full campaign itself, in-process, from scratch (or resumes via the
  config's own `resume: true` + whatever campaign_id `run_campaign` derives — but that id is not
  guaranteed to match `release_0_0_3_h600_s30__native_aggregate`, and there is no CLI surface to
  force it).

So there is **no way, as currently coded, to point `run_benchmark_release.py` at the split
aggregate root and have it validate/wrap that data.** The release manifest's own
`artifacts.required_paths` (10 entries: `campaign_manifest.json`, `manifest.json`, `run_meta.json`,
`preflight/validate_config.json`, `preflight/preview_scenarios.json`,
`reports/campaign_summary.json`, `reports/campaign_report.md`, `reports/matrix_summary.json`,
`reports/campaign_table.md`, `reports/snqi_diagnostics.json`) would also fail the
`_required_artifacts_missing()` check (`run_benchmark_release.py:103-110`) against the aggregate
root, since the `aggregate` command only ever writes 3 of those 10 files.

**Documented alternative** (no new tooling built here; this is the closest available path with
existing code):

1. Treat the split-aggregate `campaign_summary.json`/`campaign_report.md` as **diagnostic/internal
   evidence only**, under its own `split-camera-ready-native-aggregate.v1` claim boundary — this is
   already exactly what the tool is designed and labeled to produce, and is sufficient for
   per-planner comparison analysis.
2. To get an actual release-manifest-validated, publication-bundle-eligible artifact set from
   split execution, someone would need to write new tooling to backfill the missing 7 required
   paths at the aggregate root (a synthetic `campaign_manifest.json`/`manifest.json`/`run_meta.json`
   stitched from the 14 per-arm roots' own manifests, plus `preflight/*`, `reports/matrix_summary.json`,
   `reports/campaign_table.md`, `reports/snqi_diagnostics.json` recomputed over the merged rows).
   That tooling does not exist today — this is a real gap, not a configuration issue, and is
   flagged here as a blocker for anyone who later wants release-runner-validated evidence from a
   split campaign rather than a single in-process run.
3. Alternatively, once GPU-leak fix #4826 lands and the in-process route is judged safe on
   suitable hardware (a LiCCA GPU node, not this 24 GB local machine), running
   `run_benchmark_release.py --mode run` natively (unsplit) remains the only currently-coded path
   to a release-manifest-validated bundle.

## 6. Local fallback (sequential, single 24 GB machine)

### Timing anchor (verified from repo history, not assumed)

`docs/context/camera_ready_all_planners_slurm_2026-05-04.md` records a real prior h100/3-seed,
7-arm, 48-scenario campaign (`ppo`, `orca`, `socnav_sampling`, `prediction_planner`, `goal`,
`social_force`, `sacadrl`; `socnav_bench` failed preflight and is excluded from these numbers):

- Total campaign runtime: **1894.28 s** (~31.6 min) for 1008 episodes (144 episodes/arm × 7).
- `prediction_planner` alone: **1006.07 s** (~16.8 min), 53% of total wall time, for its 144
  episodes (**6.99 s/episode**).
- The other 6 arms combined: 888.21 s for 864 episodes (**1.03 s/episode**, blended).

(The task brief's own "~40 min total / ~35 min prediction_planner" figures are the same order of
magnitude as this measured anchor but roughly 2× higher — no document in this repo matches those
exact numbers; the analysis below uses the measured 1894.28 s / 1006.07 s anchor since it is
directly traceable to a specific campaign root and report.)

### Scaling to h600/s30 (1,410 episodes/arm vs 144; horizon ×6)

- Scenario/seed multiplier alone: 1410/144 ≈ **9.79×** episodes per arm.
- Horizon multiplier is sub-linear and uncertain: episodes that already terminate early
  (goal-reached/collision) cost about the same regardless of the horizon cap; only episodes that
  were hitting the old 100-step timeout run longer under a 600-step cap (up to 6× worst case for
  those episodes specifically). The campaign config's own authoring rationale
  (`paper_experiment_matrix_v2_h600_s30_extended.yaml`, top comment) states the horizon bump is
  specifically because "timeout no longer dominant, waiting strategies visible" at h100 — i.e., a
  non-trivial share of episodes were already near the h100 wall, so a blended per-episode
  multiplier of roughly **2-4×** (not the full 6×, since not all episodes were timing out) is used
  below as a central estimate, not a precise figure.
- `prediction_planner` and `predictive_mppi` (the two arms doing genuinely expensive per-step
  predictive-rollout computation) are treated at the high end of that range;
  `ppo`/`guarded_ppo`/`sacadrl` (cheap-forward-pass, checkpoint-loading but not compute-dominant —
  see §3) and the 9 classical/rule-based arms are treated at the blended 1.03 s/episode anchor.

| Bucket | Arms | Per-episode estimate (h600/s30) | Episodes/arm | Est. wall time/arm |
|---|---|---:|---:|---:|
| Dominant predictive compute | `prediction_planner`, `predictive_mppi` | ~14-28 s | 1,410 | ~5.5-11 h **each** |
| Cheap-inference + classical | remaining 12 arms | ~1-3 s | 1,410 | ~24-70 min **each** |

**Sequential total for all 14 arms on this machine: roughly 16-36 hours** (dominant pair:
~11-22 h combined; remaining 12 arms: ~4.7-14 h combined). This is a wide, genuinely uncertain
range — it is bounded on the low end by the measured h100 anchor and on the high end by the
independent evidence that the prior in-process 6-arm h600/s30 campaign ran at least 14 hours
before failing on the GPU leak (issue #4826), which is consistent with 1-2 of the dominant arms
alone consuming most of that window.

### Concrete blockers for local execution of specific arms (found by reading code, not assumed)

- **`ppo` arm will likely error, not just run slowly, on this machine.** Its algo config
  (`configs/baselines/ppo_15m_grid_socnav.yaml:39`) hardcodes `predictive_foresight_device: cuda`,
  and `robot_sf/planner/predictive_foresight.py:46` passes that string straight to the foresight
  model with no CPU fallback branch. This machine is Darwin/Apple Silicon with no CUDA device.
  **Before running the `ppo` arm locally, its child config
  (`output/benchmarks/splits/.../paper_experiment_matrix_v2_h600_s30_extended__arm_ppo.yaml`)
  needs `predictive_foresight_device` overridden to `cpu`** (and the child's sha256 in
  `split_manifest.json` recomputed, then `plan` re-run) — otherwise the arm fails at
  model-load time, not at OOM time.
- **Checkpoint staging still applies locally.** The same 5 arms in §4 need
  `submit_camera_ready_checkpoint_gate.sh` (or the plain `--stage` CLI) run once before any local
  sequential loop, for the same reason as the cluster path — a missing/`stageable_remote`
  checkpoint fails the arm, just without losing 14 hours of unrelated compute first.
- **Recommend a probe before committing to the full sequential run.** Run `prediction_planner`
  and `predictive_mppi` first (they dominate wall time and are the least certain estimate above)
  against a small scenario/seed subset — or accept the first native rows from those two arms as
  the actual timing calibration — before scheduling the remaining 12 arms overnight.

### Local fallback command shape (documented, not executed)

```bash
# 0. Stage checkpoints once (network required; not run in this session):
scripts/benchmark/submit_camera_ready_checkpoint_gate.sh \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml \
  --report-path output/benchmarks/release_v0_0_3_split_plan/checkpoint_staging.json

# 1. Fix the ppo child's predictive_foresight_device: cuda -> cpu (see blocker above),
#    recompute its sha256, and patch split_manifest.json + re-run `plan` (§1).

# 2. Run arms sequentially, one at a time, reusing the exact packet commands from §1:
for key in prediction_planner predictive_mppi ppo goal social_force orca socnav_sampling \
           sacadrl scenario_adaptive_hybrid_orca_v1 \
           scenario_adaptive_hybrid_orca_v2_collision_guard \
           hybrid_rule_v3_fast_progress_static_escape \
           hybrid_rule_v3_fast_progress_static_escape_continuous guarded_ppo risk_dwa; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python scripts/tools/run_camera_ready_benchmark.py \
    --config "output/benchmarks/splits/paper_experiment_matrix_v2_h600_s30_extended/paper_experiment_matrix_v2_h600_s30_extended__arm_${key}.yaml" \
    --output-root output/benchmarks/release_v0_0_3_arms \
    --campaign-id "release_0_0_3_h600_s30__arm_${key}" \
    --mode run --skip-publication-bundle \
    2>&1 | tee "output/slurm/logs/local-${key}.log"
done

# 3. Aggregate once all 14 arms have produced campaign_summary.json (§5):
uv run python scripts/tools/run_split_camera_ready_campaign.py aggregate \
  --packet output/benchmarks/release_v0_0_3_split_plan/execution_packet.json \
  --output-dir output/benchmarks/release_v0_0_3_split_plan/aggregate
```

`workers: 8` is left unchanged in this loop (matching the cluster path) because the same
process-multiplication risk in §1 applies locally too, arguably more acutely on a 24 GB machine —
if the human wants a safety margin without editing configs, reducing to `workers: 2`-`4` for the
5 heavy arms specifically (same mechanism as §1) is the lowest-risk lever before running anything
overnight unattended.

## Summary for the human operator

(a) **SLURM reachability:** not reachable from this machine (`sbatch`/`sinfo`/`squeue` all
absent). The cluster is LiCCA, reachable via `ssh licca` per `~/.ssh/config`
(`licca-li-01.rz.uni-augsburg.de`), documented in `SLURM/Licca/README.md`.

(b) **Exact next command(s):**

- On the LiCCA login node (cluster path): first
  `scripts/benchmark/submit_camera_ready_checkpoint_gate.sh --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml --report-path output/benchmarks/release_v0_0_3_split_plan/checkpoint_staging.json`,
  confirm `submit_safe: true`, then submit the 14-arm loop in §3 (tier-appropriate `--mem`,
  `epyc-gpu` for `ppo`, `epyc`/`epyc-mem` for the rest — confirm partition names with `sinfo`
  first, since this session could not).
- Locally (fallback path): first override `ppo`'s `predictive_foresight_device` to `cpu` in its
  split child config and re-`plan`, then run the §6 sequential loop, budgeting **16-36 hours**
  wall clock with `prediction_planner`/`predictive_mppi` as the dominant, least-certain cost.

(c) **Blockers found:**

1. 4 of 5 checkpoint-bound arms (`prediction_planner`, `ppo`, `guarded_ppo`, `predictive_mppi`)
   are only `stageable_remote` today — enforced staging (`--stage`) has not been run in this
   session (network download, out of scope for a planning pass) and must happen before any
   `sbatch` or local execution.
2. `ppo`'s algo config hardcodes a CUDA-only foresight device with no CPU fallback in code — it
   will fail outright on this local Darwin/no-GPU machine without a config edit, and needs a GPU
   partition on LiCCA.
3. `run_benchmark_release.py` has no supported way to consume the split-execution `aggregate`
   output or to resume/report over an existing campaign root — the split-aggregate path and the
   release-manifest-validated path are not currently connected in code; treating the aggregate as
   diagnostic-only evidence (not a release bundle) is the only code-supported option today.
4. `workers: 8` survives the per-arm split unchanged, so each arm's job still fans out to 8
   OS processes; for the 5 checkpoint-loading arms this multiplies PyTorch/TensorFlow import +
   checkpoint memory across all 8 workers, which is the actual justification for the 16G heavy
   tier (not checkpoint file size) — a `workers_override` reduction is available but was not
   applied, left as an operator decision.
5. Cluster partition names/QoS are documented in `SLURM/Licca/README.md` but could not be
   live-verified (`sinfo` unreachable from here); confirm before the first submission.
