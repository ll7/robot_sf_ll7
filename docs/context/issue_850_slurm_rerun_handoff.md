# Issue #850 — SLURM Rerun Handoff

**Related issue:** [#850](https://github.com/ll7/robot_sf_ll7/issues/850)  
**Predecessor note:** [Issue #850 PPO Collision Failures](issue_850_ppo_collision_failures.md)  
**Historical baseline note:** [Issue #193 Feature Extractor Optuna Study](issue_193_feature_extractor_optuna_study.md)

## Decision

Yes: if the deleted worktree also deleted the local issue-193 checkpoints and benchmark outputs, the
correct recovery path is step 3 from the canonical issue-850 note: re-run the committed training
surface instead of trying to recover artifacts from W&B.

Why this is the right path:

* the deleted artifacts were local-only, 
* the original issue-193 fixed-candidate batch was launched with `--disable-wandb`, 
* the W&B API is reachable but does not contain those candidate runs, 
* the repository already has the exact fixed-candidate launcher and configs needed to reproduce the
  lost checkpoints and then queue the issue-850 follow-up.

## Queue Plan

Use three phases.

### Phase 1 — reconstruct the deleted issue-193 12M candidate batch

This recreates the lost checkpoints and the comparison surface that issue 850 depends on.

* Candidate file: `configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml`
* Launcher: `SLURM/submit_feature_extractor_fixed_candidates.sh`
* Candidate count: `10` jobs (`0-9`)
* Important change versus the historical run: **do not pass `--disable-wandb` this time**

Suggested study/storage names:

* study: `feat_extractor_12m_recovery_issue850_20260429`
* storage: `sqlite:///output/optuna/feat_extractor/feat_extractor_12m_recovery_issue850_20260429.db`

Queue command:

```bash
bash SLURM/submit_feature_extractor_fixed_candidates.sh \
    --candidate-file configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml \
    --study-name feat_extractor_12m_recovery_issue850_20260429 \
    --storage sqlite:///output/optuna/feat_extractor/feat_extractor_12m_recovery_issue850_20260429.db \
    --partition <GPU_PARTITION> \
    --account <ACCOUNT> \
    --qos <GPU_QOS> \
    --time <MAX_VALID_TIME> \
    --array-concurrency 2 \
    --job-name feat_12m_i193_recovery
```

Notes:

* Historical successful partitions on the original cluster were `pro6000` / `pro6000-gpu` and
`a30` / `a30-gpu` .
* Use the live max valid wall time for the partition/QoS you actually have on the cluster.
* The launcher defaults to W&B enabled unless `--disable-wandb` is passed, so omitting that flag is
  enough to preserve remote run metadata this time.

### Phase 2 — queue the issue-850 reward-v3 fixed-candidate follow-up

This keeps the issue-193 candidate comparison surface but swaps the reward family to the stronger
`route_completion_v3` profile.

* Candidate file: `configs/training/ppo/feature_extractor_candidates_12m_issue850_reward_v3.yaml`
* Candidate count: `5` jobs (`0-4`)

Suggested study/storage names:

* study: `feat_extractor_issue850_reward_v3_20260429`
* storage: `sqlite:///output/optuna/feat_extractor/feat_extractor_issue850_reward_v3_20260429.db`

Queue command:

```bash
bash SLURM/submit_feature_extractor_fixed_candidates.sh \
    --candidate-file configs/training/ppo/feature_extractor_candidates_12m_issue850_reward_v3.yaml \
    --study-name feat_extractor_issue850_reward_v3_20260429 \
    --storage sqlite:///output/optuna/feat_extractor/feat_extractor_issue850_reward_v3_20260429.db \
    --partition <GPU_PARTITION> \
    --account <ACCOUNT> \
    --qos <GPU_QOS> \
    --time <MAX_VALID_TIME> \
    --array-concurrency 2 \
    --job-name feat_12m_i850_rcv3
```

Recommended order:

* If cluster capacity is limited: queue phase 1 first, then phase 2 after the recovery run is
  stable.
* If cluster capacity is fine: queue phase 2 immediately after phase 1 submission so the mitigation
  run is not blocked on wall-clock delay.

### Phase 3 — optional single-config hotspot-weighted mitigation

This is the narrower issue-850 mitigation config that keeps `dyn_large_med` fixed and also
upweights the known obstacle-collision hotspot scenarios.

* Config: `configs/training/ppo/feature_extractor_sweep_dyn_large_med_safety_v3.yaml`

Queue command:

```bash
sbatch \
    --job-name=feat_i850_safety_v3 \
    --time=<MAX_VALID_TIME> \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --output=output/slurm/feat_i850_safety_v3_%j.out \
    --error=output/slurm/feat_i850_safety_v3_%j.err \
    --partition=<GPU_PARTITION> \
    --account=<ACCOUNT> \
    --qos=<GPU_QOS> \
    --wrap 'uv run python scripts/training/train_ppo.py --config configs/training/ppo/feature_extractor_sweep_dyn_large_med_safety_v3.yaml --log-level WARNING'
```

If you only want the minimum necessary recovery for issue 850, phases 1 and 2 are the core runs.
Phase 3 is optional additional evidence.

## Preflight On The Cluster

Run these from the repo root after checking out this branch:

```bash
source .venv/bin/activate
uv sync --all-extras

uv run python scripts/training/fixed_feature_extractor_candidates.py \
  --candidate-file configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml \
  --candidate-index 0 \
  --study-name issue850_preflight \
  --storage sqlite:///output/optuna/feat_extractor/issue850_preflight.db \
  --print-count

bash SLURM/submit_feature_extractor_fixed_candidates.sh \
  --candidate-file configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml \
  --study-name issue850_preflight \
  --storage sqlite:///output/optuna/feat_extractor/issue850_preflight.db \
  --dry-run
```

Check the live partition/QoS limit before replacing `<MAX_VALID_TIME>` :

```bash
sinfo -p <GPU_PARTITION> -o '%P %l %D %T'
sacctmgr show qos format=Name,MaxWall | rg '<GPU_QOS>'
```

## Monitoring

Issue-193 recovery batch:

```bash
uv run python scripts/tools/inspect_optuna_db.py \
  --db output/optuna/feat_extractor/feat_extractor_12m_recovery_issue850_20260429.db \
  --study-name feat_extractor_12m_recovery_issue850_20260429 \
  --show-params \
  --top-n 10
```

Issue-850 reward-v3 matrix:

```bash
uv run python scripts/tools/inspect_optuna_db.py \
  --db output/optuna/feat_extractor/feat_extractor_issue850_reward_v3_20260429.db \
  --study-name feat_extractor_issue850_reward_v3_20260429 \
  --show-params \
  --top-n 5
```

SLURM queue / logs:

```bash
squeue --me
tail -f output/slurm/feat_12m_i193_recovery_<jobid>_<taskid>.out
tail -f output/slurm/feat_12m_i850_rcv3_<jobid>_<taskid>.out
```

## Reconstructing The Hold-Out Surface After Training

The fixed-candidate runner sets policy IDs as:

* `feat_extractor_sweep_base_12m_<candidate_id>`

That makes the local model lookup deterministic enough for a simple `find` -based evaluation loop.

Recreate the issue-193 hold-out outputs:

```bash
for candidate in \
  dyn_large_med_s123 \
  dyn_large_med_s231 \
  dyn_large_med_s1337 \
  lc_small_med_s231 \
  dyn_default_s1337
do
  model_path=$(find output -path "*feat_extractor_sweep_base_12m_${candidate}*model.zip" | head -n 1)
  SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python scripts/tools/policy_analysis_run.py \
    --training-config configs/training/ppo/feature_extractor_sweep_base.yaml \
    --policy ppo \
    --model-path "$model_path" \
    --seed-set eval \
    --max-seeds 3 \
    --output "output/benchmarks/issue193_policy_analysis_${candidate}" \
    --video-output "output/recordings/issue193_policy_analysis_${candidate}" \
    --all
done
```

Run the collision split report:

```bash
uv run python scripts/tools/analyze_policy_collision_failures.py \
  output/benchmarks/issue193_policy_analysis_dyn_large_med_s231 \
  output/benchmarks/issue193_policy_analysis_dyn_large_med_s1337 \
  output/benchmarks/issue193_policy_analysis_dyn_large_med_s123 \
  --output-md output/analysis/issue850_collision_failures.md \
  --output-json output/analysis/issue850_collision_failures.json
```

Evaluate the issue-850 reward-v3 matrix the same way, but target the `_rcv3` candidate IDs and
write outputs under `output/benchmarks/issue850_policy_analysis_<candidate>` .

## Preservation Checklist

Do not lose the reconstructed artifacts again. Preserve at least these paths before deleting any
worktree or temporary checkout:

* `output/optuna/feat_extractor/*.db`
* `output/slurm/feat_12m_i193_recovery_*.{out,err}`
* `output/slurm/feat_12m_i850_rcv3_*.{out,err}`
* `output/wandb/`
* `output/benchmarks/issue193_policy_analysis_*/`
* `output/benchmarks/issue850_policy_analysis_*/`

## Handoff Boundary

This note is only the rerun/queue handoff.  The interpretation and promotion decision still belong
in [issue_850_ppo_collision_failures.md](issue_850_ppo_collision_failures.md).
