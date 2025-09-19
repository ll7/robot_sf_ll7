uv run python -m robot_sf.benchmark.cli run \
  --scenarios configs/scenarios/debug_senario.yaml \
  --output results/debug_scenario.jsonl \              
  --workers 4 \
  --resume