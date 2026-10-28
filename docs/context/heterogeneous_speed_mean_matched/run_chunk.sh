#!/usr/bin/env bash
# usage: launch.sh <phase-dir> <planner> <chunk-name> <seeds...>
set -u
cd "$(git rev-parse --show-toplevel)"
phase=$1; planner=$2; chunk=$3; shift 3
out="${XHET_RUNS:-output/xhet_runs}/$phase/$planner/$chunk"; mkdir -p "$out"
run_rc=0
LOGURU_LEVEL=ERROR .venv/bin/python docs/context/heterogeneous_speed_mean_matched/driver.py --planner "$planner" --seeds "$@" --out "$out" > "$out/run.log" 2>&1 || run_rc=$?
printf "EXIT %s\n" "$run_rc" >> "$out/run.log"
exit "$run_rc"
