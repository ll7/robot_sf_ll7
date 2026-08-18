#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_ci_local.sh [--no-setup] [--scratch-dir PATH] [<phase> ...]

Run the canonical CI validation phases locally through the shared driver.
When no phases are provided, the local wrapper runs every phase advertised by:
  scripts/dev/ci_driver.sh --list-phases

Wrapper options:
  --no-setup     Skip dependency sync and artifact migration for repeat local runs
  --scratch-dir PATH
                 Use PATH for temporary files and default uv/XDG caches for this run
  --list-phases  List canonical CI phases without running setup
  -h, --help     Show this help message

Environment:
  ROBOT_SF_CI_MIN_FREE_BYTES
                 Minimum free bytes required for the effective temporary directory
                 (default: 1073741824)

Examples:
  scripts/dev/run_ci_local.sh
  scripts/dev/run_ci_local.sh --no-setup lint test
  scripts/dev/run_ci_local.sh lint test
  CI_DRIVER_EVENT_NAME=workflow_dispatch scripts/dev/run_ci_local.sh smoke
EOF
}

if [[ "$#" -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

check_scratch_capacity() {
  local space_path="$1"
  local minimum_bytes="${ROBOT_SF_CI_MIN_FREE_BYTES:-1073741824}"
  local minimum_kib
  local df_output
  local available_kib

  if ! [[ "$minimum_bytes" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ROBOT_SF_CI_MIN_FREE_BYTES must be a non-negative integer (got '$minimum_bytes')." >&2
    return 2
  fi
  minimum_kib=$(( (10#$minimum_bytes + 1023) / 1024 ))

  if [[ ! -d "$space_path" || ! -w "$space_path" ]]; then
    echo "ERROR: local CI scratch path is missing or not writable: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk or set TMPDIR to a writable disk-backed directory." >&2
    return 2
  fi
  if ! command -v df >/dev/null 2>&1 || ! command -v awk >/dev/null 2>&1; then
    echo "ERROR: local CI scratch preflight requires both 'df' and 'awk'." >&2
    return 2
  fi
  if ! df_output="$(df -Pk "$space_path" 2>/dev/null)"; then
    echo "ERROR: local CI scratch preflight could not inspect filesystem capacity: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk and retry." >&2
    return 2
  fi
  available_kib="$(awk 'NR > 1 && $4 ~ /^[0-9]+$/ { print $4; exit }' <<<"$df_output")"
  if ! [[ "$available_kib" =~ ^[0-9]+$ ]]; then
    echo "ERROR: local CI scratch preflight could not parse available space for: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk and retry." >&2
    return 2
  fi
  if (( 10#$available_kib < minimum_kib )); then
    echo "ERROR: local CI scratch preflight failed: ${available_kib} KiB available at $space_path; ${minimum_kib} KiB required." >&2
    echo "No CI phase was started. Free space or retry with --scratch-dir /path/on/disk." >&2
    echo "For a deliberately bounded run only, lower ROBOT_SF_CI_MIN_FREE_BYTES explicitly." >&2
    return 2
  fi
  echo "Local CI scratch preflight passed: path=$space_path available=${available_kib}KiB required=${minimum_kib}KiB" >&2
}

configure_scratch_dir() {
  local requested_path="$1"
  local scratch_root

  if ! mkdir -p "$requested_path"; then
    echo "ERROR: could not create local CI scratch directory: $requested_path" >&2
    echo "Choose a writable disk-backed path and retry." >&2
    return 2
  fi
  if ! scratch_root="$(cd "$requested_path" 2>/dev/null && pwd -P)"; then
    echo "ERROR: could not resolve local CI scratch directory: $requested_path" >&2
    return 2
  fi
  if ! mkdir -p "$scratch_root/tmp" "$scratch_root/uv-cache" "$scratch_root/xdg-cache" "$scratch_root/mplconfig"; then
    echo "ERROR: could not create local CI scratch subdirectories under: $scratch_root" >&2
    echo "Choose a writable disk-backed path and retry." >&2
    return 2
  fi

  export ROBOT_SF_CI_SCRATCH_DIR="$scratch_root"
  export TMPDIR="$scratch_root/tmp"
  export UV_CACHE_DIR="$scratch_root/uv-cache"
  export XDG_CACHE_HOME="$scratch_root/xdg-cache"
  export MPLCONFIGDIR="$scratch_root/mplconfig"
  echo "Using local CI scratch directory: $scratch_root" >&2
}

load_default_phases() {
  mapfile -t default_phases < <("$SCRIPT_DIR/ci_driver.sh" --list-phases)
  if [[ ${#default_phases[@]} -eq 0 ]]; then
    echo "Failed to load default CI phases from scripts/dev/ci_driver.sh --list-phases" >&2
    exit 2
  fi
  printf "%s\n" "${default_phases[@]}"
}

run_setup="1"
phases=()
scratch_dir="${ROBOT_SF_CI_SCRATCH_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-setup)
      run_setup="0"
      shift
      ;;
    --scratch-dir)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--scratch-dir requires a path." >&2
        exit 2
      fi
      scratch_dir="$2"
      shift 2
      ;;
    --list-phases)
      "$SCRIPT_DIR/ci_driver.sh" --list-phases
      exit 0
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      phases+=("$@")
      break
      ;;
    *)
      phases+=("$1")
      shift
      ;;
  esac
done

if [[ ${#phases[@]} -eq 0 ]]; then
  mapfile -t phases < <(load_default_phases)
fi

if [[ -n "$scratch_dir" ]]; then
  configure_scratch_dir "$scratch_dir"
fi
check_scratch_capacity "${TMPDIR:-/tmp}"

if [[ "$run_setup" == "1" ]]; then
  bash "$SCRIPT_DIR/ci_step_timer.sh" "Sync dependencies (locked)" uv sync --all-extras --frozen
  bash "$SCRIPT_DIR/ci_step_timer.sh" "Migrate legacy artifacts into canonical root" \
    uv run python scripts/tools/migrate_artifacts.py
else
  echo "Skipping run_ci_local setup (--no-setup)."
fi

"$SCRIPT_DIR/ci_driver.sh" "${phases[@]}"
