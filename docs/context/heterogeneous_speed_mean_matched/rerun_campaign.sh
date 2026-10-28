#!/usr/bin/env bash
# Rerun the whole main campaign (1,600 episodes) from the pinned source, then extract and analyze.
# Prerequisites: checkout of 373dbfde4f39667cf9e8732dabe7df5118bdeab1 with this packet copied in,
# `uv sync --frozen --extra benchmark --extra training`, and the PPO checkpoint
# (model/registry.yaml model_id ppo_release_robot_b1002_last_20261001, sha256 764a7d88...24bc)
# at output/model_cache/ppo_release_robot_b1002_last_20261001/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
P=docs/context/heterogeneous_speed_mean_matched
R=${XHET_RUNS:-output/xhet_runs}
S=(1121 1122 1123 1124 1125 1126 1127 1128 1129 1130 $(seq 1161 1200))
for p in hybrid_v4_continuous orca social_force ppo_release; do
  for i in 0 10 20 30 40; do bash $P/run_chunk.sh main "$p" "c$i" "${S[@]:$i:10}" & done
done
wait
LOGURU_LEVEL=ERROR .venv/bin/python $P/analyze.py extract --runs "$R/main" --out "$R/evidence"
.venv/bin/python $P/analyze.py analyze --episodes "$R/evidence/episodes.csv" --contract $P/contract.yaml --out "$R/evidence/primary"
.venv/bin/python $P/analyze.py analyze --episodes "$R/evidence/episodes.csv" --contract $P/contract.yaml --out "$R/evidence/sensitivity_excl_relocation" --exclude-relocation-pairs
