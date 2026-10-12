#!/usr/bin/env bash
#SBATCH --job-name=ppo-ftplan
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=output/slurm/%j-ppo-ftplan.out
# PREPARED, NOT SUBMITTED. Site partition/QoS comes from the private overlay.
# One segment per submission. CPU selection runs between GPU segments.
set -euo pipefail
: "${FTPLAN_REPO_ROOT:?set owned checkout at reviewed source head}"
: "${FTPLAN_SOURCE_HEAD:?set exact reviewed commit}"
: "${FTPLAN_REPLICATION_MANIFEST:?set durable replication-report manifest}"
: "${FTPLAN_SELECTION_RECEIPT:?set previous CPU selection receipt}"
: "${FTPLAN_ACTION_CONTRACT_RECEIPT:?set reviewed action-interface receipt}"
: "${FTPLAN_HUB_ROOT:?set durable external hub root}"
: "${FTPLAN_SEGMENT:?set 1, 2, or 3 after previous selection}"
cd "$FTPLAN_REPO_ROOT"
[[ $(git rev-parse HEAD) == "$FTPLAN_SOURCE_HEAD" ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
[[ "$FTPLAN_SEGMENT" =~ ^[123]$ ]]
export FTPLAN_INIT_SEED=$((100001 + ${SLURM_ARRAY_TASK_ID:?requires Slurm array task}))
[[ "$FTPLAN_INIT_SEED" -ge 100001 && "$FTPLAN_INIT_SEED" -le 100003 ]]
export ROBOT_SF_ARTIFACT_ROOT="output/ftplan/init${FTPLAN_INIT_SEED}/segment${FTPLAN_SEGMENT}"
export OMP_NUM_THREADS=1
export WANDB_MODE=online
mkdir -p "$ROBOT_SF_ARTIFACT_ROOT"
# Custody runs for success, failure and catchable interruption, including preflight.
training_pid=''
training_completed=0
preserve_segment() {
scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- python - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
import wandb
root = Path(os.environ['ROBOT_SF_ARTIFACT_ROOT'])
(root / 'segment_status.json').write_text(json.dumps({'status': os.environ['FTPLAN_SEGMENT_STATUS'], 'original_exit_code': int(os.environ['FTPLAN_ORIGINAL_EXIT_CODE']), 'source_head': os.environ['FTPLAN_SOURCE_HEAD']}, indent=2) + '\n')
dest = Path(os.environ['FTPLAN_HUB_ROOT']) / 'ppo-ftplan-20261012' / root.relative_to('output/ftplan')
if dest.exists():
    raise ValueError('Hub segment already exists; do not overwrite custody')
shutil.copytree(root, dest)
for source in root.rglob('*'):
    if source.is_file():
        copied = dest / source.relative_to(root)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(copied.read_bytes()).digest():
            raise ValueError('Hub checkpoint preservation mismatch')
with wandb.init(project='robot_sf', group='ppo-ftplan-released-20261012', job_type='segment-preservation') as run:
    artifact = wandb.Artifact(f'ppo-ftplan-init{os.environ["FTPLAN_INIT_SEED"]}-segment{os.environ["FTPLAN_SEGMENT"]}', type='training-segment', metadata={'source_head': os.environ['FTPLAN_SOURCE_HEAD'], 'status': os.environ['FTPLAN_SEGMENT_STATUS'], 'original_exit_code': int(os.environ['FTPLAN_ORIGINAL_EXIT_CODE'])})
    artifact.add_dir(str(root))
    committed = run.log_artifact(artifact)
    committed.wait()
print('Segment copied to hub and W&B; independent readback and CPU selection required')
PY
}
finish_segment() {
    local original_rc=$?
    trap - EXIT
    trap '' INT TERM
    set +e
    export FTPLAN_ORIGINAL_EXIT_CODE="$original_rc"
    export FTPLAN_SEGMENT_STATUS=incomplete
    if [[ "$original_rc" -eq 0 && "$training_completed" -eq 1 ]]; then
        export FTPLAN_SEGMENT_STATUS=complete
    fi
    preserve_segment
    local custody_rc=$?
    if [[ "$custody_rc" -ne 0 ]]; then
        echo "Segment preservation failed; retain local outputs and block continuation" >&2
        if [[ "$original_rc" -eq 0 ]]; then original_rc=$custody_rc; fi
    fi
    exit "$original_rc"
}
interrupt_segment() {
    local signal_name=$1 exit_code=$2
    trap '' INT TERM
    if [[ -n "$training_pid" ]]; then
        kill -s "$signal_name" "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
        training_pid=''
    fi
    exit "$exit_code"
}
trap finish_segment EXIT
trap 'interrupt_segment INT 130' INT
trap 'interrupt_segment TERM 143' TERM
# Validate bound reports before creating a warm checkpoint or invoking training.
scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- python - <<'PY'
import hashlib, json, os
from pathlib import Path
import yaml
import numpy as np
from gymnasium import spaces
from robot_sf.training.released_warm_start import load_warm_start, released_checkpoint
from robot_sf.training.scenario_family_weights import with_family_weights
from scripts.training.train_ppo import load_expert_training_config

def verified(record):
    p = Path(record['artifact_path'])
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Receipt artifact paths must be repository relative')
    if hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Receipt artifact digest mismatch')
    if not record['hub_artifact'] or not record['wandb_artifact']:
        raise ValueError('Receipt requires external-hub and W&B readback references')
    return p

manifest = json.loads(Path(os.environ['FTPLAN_REPLICATION_MANIFEST']).read_text())
if manifest['status'] != 'reported' or manifest['worker_receipt_audit'] != 'passed':
    raise ValueError('Controlled seed replication has not reported with a passed receipt audit')
if sorted(r['master_seed'] for r in manifest['paired_master_reports']) != [100000, 100001, 100002]:
    raise ValueError('All three paired-master reports are required')
for r in manifest['paired_master_reports']:
    verified(r)
receipt = json.loads(Path(os.environ['FTPLAN_SELECTION_RECEIPT']).read_text())
verified(receipt['results_artifact'])
seed, segment = int(os.environ['FTPLAN_INIT_SEED']), int(os.environ['FTPLAN_SEGMENT'])
if receipt['evaluation_seeds'] != list(range(1031, 1061)) or receipt['scenario_count'] != 48:
    raise ValueError('Wrong selection pairs')
if receipt['fallback_used'] or receipt['execution_errors']:
    raise ValueError('Selection degraded or failed')
if receipt['completed_segment'] != segment - 1:
    raise ValueError('Selection receipt is from the wrong segment')
source, provenance = released_checkpoint('ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417')
action_contract = json.loads(Path(os.environ['FTPLAN_ACTION_CONTRACT_RECEIPT']).read_text())
verified(action_contract['review_artifact'])
if action_contract['source_head'] != os.environ['FTPLAN_SOURCE_HEAD'] or action_contract['source_checkpoint_sha256'] != provenance['sha256']:
    raise ValueError('Action contract is not bound to this source/checkpoint')
if action_contract['mapping'] != 'bounds_only' or action_contract['target_semantics'] != 'acceleration':
    raise ValueError('Other action-semantic migrations require the alignment lane integration')
target_space = spaces.Box(np.asarray(action_contract['target_action_low'], dtype=np.float32), np.asarray(action_contract['target_action_high'], dtype=np.float32))
if segment == 1:
    if receipt['checkpoint_sha256'] != provenance['sha256']:
        raise ValueError('First segment requires the released baseline selection')
    model, _ = load_warm_start(source, init_seed=seed, target_action_space=target_space)
    checkpoint = Path(os.environ['ROBOT_SF_ARTIFACT_ROOT']) / 'warm.zip'
    model.save(checkpoint)
else:
    if receipt['initialization_seed'] != seed:
        raise ValueError('Selection receipt is from the wrong initialization seed')
    if not 0 <= receipt['previous_success_mean'] < receipt['success_mean'] <= 1:
        raise ValueError('Stop: selection success did not strictly rise')
    checkpoint = verified(receipt['checkpoint_artifact'])
    if receipt['checkpoint_sha256'] != receipt['checkpoint_artifact']['sha256']:
        raise ValueError('Selection checkpoint identity mismatch')
base = Path('configs/training/ppo/ftplan/released_warm_weighted_3m_20261012.yaml.template').resolve()
runtime = Path(os.environ['ROBOT_SF_ARTIFACT_ROOT']) / 'runtime.yaml'
runtime.write_text(yaml.safe_dump({
    'base_config': os.path.relpath(base, runtime.parent),
    'policy_id': f'ppo_ftplan_init{seed}_segment{segment}',
    'training_master_seed': seed, 'ppo_initialization_seed': seed,
    'scenario_config': os.path.relpath(Path('configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml').resolve(), runtime.parent),
    'evaluation': {'scenario_config': os.path.relpath(Path('configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml').resolve(), runtime.parent)},
    'total_timesteps': segment * 1000000,
    'resume_from': os.path.relpath(checkpoint.resolve(), runtime.parent),
}))
# The main loader must understand the seed-policy successor fields; no fallback.
config = with_family_weights(load_expert_training_config(runtime))
if getattr(config, 'training_seed_range', None) != (100000, 199999):
    raise ValueError('Training-only seed-policy implementation is required')
print('Pilot prerequisites validated; bound runtime config prepared')
PY
scripts/dev/run_worktree_shared_venv.sh --profile all-extras -- \
  python -m scripts.training.train_ppo_with_family_weights \
  --config "$ROBOT_SF_ARTIFACT_ROOT/runtime.yaml" \
  --run-id "ftplan_init${FTPLAN_INIT_SEED}_segment${FTPLAN_SEGMENT}" &
training_pid=$!
if wait "$training_pid"; then
    training_completed=1
    training_pid=''
else
    training_rc=$?
    training_pid=''
    exit "$training_rc"
fi
