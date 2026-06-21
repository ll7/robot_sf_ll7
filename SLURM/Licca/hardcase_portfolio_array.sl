#!/usr/bin/env bash
# Hard-case breakthrough portfolio sweep (#3215): authority variants x checkpoints on hard seeds.
# Array index -> (checkpoint, authority variant). CPU-only predictive eval.
#SBATCH --job-name=hc_portfolio
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

REPO="${HC_REPO:-$HOME/git/robot_sf_ll7}"
SCRATCH="${HC_SCRATCH:-/hpc/gpfs2/scratch/u/luttkule/robot_sf/hardcase_portfolio}"
PYTHON="${HC_PYTHON:-$REPO/.venv/bin/python}"
SCOPE_TAG="${HC_SCOPE_TAG:-hard}"
SEED_MANIFEST="${HC_SEED_MANIFEST:-configs/benchmarks/predictive_hard_seeds_v1.yaml}"

CKPTS=(predictive_proxy_selected_v2_full predictive_proxy_selected_v1)
VARIANTS=(baseline high_angular dense_lattice deep_sequence nearfield_turn combined_max_authority)

i="${SLURM_ARRAY_TASK_ID:-0}"
ckpt="${CKPTS[$((i / ${#VARIANTS[@]}))]}"
variant="${VARIANTS[$((i % ${#VARIANTS[@]}))]}"
tag="${ckpt}__${variant}__${SCOPE_TAG}"

SEED_ARGS=()
if [ "$SEED_MANIFEST" != "none" ]; then
  SEED_ARGS=(--seed-manifest "$SEED_MANIFEST")
fi

echo "[hc] task=$i checkpoint=$ckpt variant=$variant scope=$SCOPE_TAG"
cd "$REPO"
PYTHONPATH="$REPO" "$PYTHON" scripts/validation/evaluate_predictive_planner.py \
  --checkpoint "$SCRATCH/ckpt/${ckpt}.pt" \
  --algo-config "configs/algos/hardcase_authority/prediction_planner_authority_${variant}.yaml" \
  --scenario-matrix configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml \
  "${SEED_ARGS[@]}" \
  --min-success-rate 0.0 \
  --output-dir "$SCRATCH/results/${tag}" \
  --tag "$tag" --workers 4
echo "[hc] done task=$i tag=$tag"
