#!/usr/bin/env bash
# Run every (arm, scenario) job of the design for one seed list.
# usage (repo root): run_campaign.sh <python> <out_dir> <seeds_csv> <parallel>
set -u
PY=$1; OUT=$2; SEEDS=$3; PAR=$4
export TMPDIR=${TMPDIR:-/tmp} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/logs"
for arm in none G1E1 G0E1 orca; do
  for cell in classic_doorway_medium classic_doorway_high classic_bottleneck_medium classic_bottleneck_high; do
    for cond in traverse outside; do echo "$arm ${cell}__${cond}"; done
  done
done | xargs -P "$PAR" -L 1 bash -c '
  "'"$PY"'" docs/context/guard_crossing_pedestrians/tools/xped_run.py --arm "$0" --scenario "$1" \
    --seeds "'"$SEEDS"'" --out "'"$OUT"'" > "'"$OUT"'/logs/$0__$1.log" 2>&1 || echo "FAILED $0 $1"'
echo CAMPAIGN_DONE
