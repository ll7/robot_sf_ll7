#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/check_carla_runtime.sh [--smoke] [--pull] [--startup-timeout-s SECONDS]

Check the reproducible CARLA client/runtime path for this checkout.

By default, this script:
  1. runs the host-side CARLA Python API check through the locked uv `carla` dependency group;
  2. runs the pinned CARLA Docker runtime preflight without starting the simulator container.

Use --smoke when the current host may start the CARLA Docker server for a bounded connectivity
proof. Use --pull when the pinned Docker image may be downloaded if it is missing.

Examples:
  scripts/dev/check_carla_runtime.sh
  scripts/dev/check_carla_runtime.sh --pull
  scripts/dev/check_carla_runtime.sh --smoke
  scripts/dev/check_carla_runtime.sh --smoke --pull --startup-timeout-s 90
EOF
}

mode="preflight"
pull=false
startup_timeout_s=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      mode="smoke"
      shift
      ;;
    --preflight)
      mode="preflight"
      shift
      ;;
    --pull)
      pull=true
      shift
      ;;
    --startup-timeout-s)
      if [[ $# -lt 2 ]]; then
        echo "--startup-timeout-s requires a value" >&2
        exit 2
      fi
      if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--startup-timeout-s must be a positive integer number of seconds" >&2
        exit 2
      fi
      startup_timeout_s="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

runtime_args=("${mode}" "--json")
if [[ "${pull}" == true ]]; then
  runtime_args+=("--pull")
fi
if [[ "${mode}" == "smoke" && -n "${startup_timeout_s}" ]]; then
  runtime_args+=("--startup-timeout-s" "${startup_timeout_s}")
elif [[ "${mode}" == "preflight" && -n "${startup_timeout_s}" ]]; then
  echo "--startup-timeout-s is only valid with --smoke" >&2
  exit 2
fi

echo "Checking CARLA Python API through uv dependency group: carla" >&2
uv run --frozen --group carla robot-sf-check-carla --json --require

echo "Checking pinned CARLA Docker runtime: ${mode}" >&2
uv run --frozen --group carla robot-sf-carla-docker-runtime "${runtime_args[@]}"
