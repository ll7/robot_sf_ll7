#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

show_help() {
  cat <<'EOF'
Usage: scripts/dev/sbatch_use_max_time.sh [wrapper-options] <slurm-script> [sbatch-args...]

Submits a Slurm batch script with --time set to the effective maximum wall time
for the selected partition and QoS, unless an explicit --time override is given.

Wrapper options:
  --partition <name>  Override partition discovery
  --qos <name>        Override QoS discovery
  --time <value>      Explicit wall time; disables auto-max discovery
  --sbatch-arg <arg>  Additional sbatch option to forward before the script path
  --dry-run           Print the resolved sbatch command without submitting
  -h, --help          Show this help message

Discovery order:
  1. Wrapper options
  2. #SBATCH directives in the target script

Examples:
  scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/auxme_gpu.sl
  scripts/dev/sbatch_use_max_time.sh --dry-run SLURM/Auxme/auxme_gpu.sl
  scripts/dev/sbatch_use_max_time.sh --partition a30 --qos a30-gpu SLURM/Auxme/auxme_gpu.sl
  scripts/dev/sbatch_use_max_time.sh --sbatch-arg=--dependency=afterok:12345 SLURM/Auxme/auxme_gpu.sl
EOF
}

extract_sbatch_value() {
  local script_path="$1"
  local key="$2"
  python - "$script_path" "$key" <<'PY'
import pathlib
import re
import sys

script_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
patterns = [
    re.compile(rf"^#SBATCH\s+--{re.escape(key)}=(\S+)\s*$"),
    re.compile(rf"^#SBATCH\s+--{re.escape(key)}\s+(\S+)\s*$"),
]

value = ""
for line in script_path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    for pattern in patterns:
        match = pattern.match(stripped)
        if match:
            value = match.group(1)

print(value)
PY
}

query_partition_max_time() {
  local partition="$1"
  scontrol show partition "$partition" | sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' | head -n 1
}

query_qos_max_time() {
  local qos="$1"
  if ! command -v sacctmgr >/dev/null 2>&1; then
    return 0
  fi
  sacctmgr -nP show qos "$qos" format=Name,MaxWall 2>/dev/null | awk -F'|' 'NR==1 {print $2}'
}

compute_effective_time() {
  python - "$@" <<'PY'
import sys

values = sys.argv[1:]

def to_seconds(value: str | None):
    if not value:
        return None
    normalized = value.strip()
    if not normalized or normalized.upper() in {"NONE", "UNLIMITED", "INFINITE", "NOTSET"}:
        return None
    days = 0
    if "-" in normalized:
        day_part, normalized = normalized.split("-", 1)
        days = int(day_part)
    parts = normalized.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = map(int, parts)
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = int(parts[0])
    else:
        raise SystemExit(f"Unsupported Slurm time format: {value}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def from_seconds(total_seconds: int):
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

finite = [to_seconds(value) for value in values]
finite = [value for value in finite if value is not None]
if not finite:
    print("")
else:
    print(from_seconds(min(finite)))
PY
}

partition=""
qos=""
explicit_time=""
dry_run=0
extra_sbatch_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      partition="$2"
      shift 2
      ;;
    --qos)
      qos="$2"
      shift 2
      ;;
    --time)
      explicit_time="$2"
      shift 2
      ;;
    --sbatch-arg)
      extra_sbatch_args+=("$2")
      shift 2
      ;;
    --sbatch-arg=*)
      extra_sbatch_args+=("${1#*=}")
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown wrapper option: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  show_help >&2
  exit 2
fi

script_path="$1"
shift

if [[ ! -f "$script_path" ]]; then
  echo "Slurm script not found: $script_path" >&2
  exit 2
fi

if [[ -z "$partition" ]]; then
  partition="$(extract_sbatch_value "$script_path" "partition")"
fi

if [[ -z "$qos" ]]; then
  qos="$(extract_sbatch_value "$script_path" "qos")"
fi

resolved_time="$explicit_time"
partition_max=""
qos_max=""

if [[ -z "$resolved_time" ]]; then
  if [[ -z "$partition" ]]; then
    echo "Could not determine partition. Pass --partition or add #SBATCH --partition to $script_path." >&2
    exit 2
  fi
  if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is required for auto-max wall time discovery." >&2
    exit 2
  fi
  partition_max="$(query_partition_max_time "$partition")"
  if [[ -n "$qos" ]]; then
    qos_max="$(query_qos_max_time "$qos")"
  fi
  resolved_time="$(compute_effective_time "$partition_max" "$qos_max")"
fi

cmd=(sbatch)
if [[ -n "$resolved_time" ]]; then
  cmd+=("--time=$resolved_time")
fi
if [[ ${#extra_sbatch_args[@]} -gt 0 ]]; then
  cmd+=("${extra_sbatch_args[@]}")
fi
cmd+=("$script_path")
if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

if [[ -n "$resolved_time" ]]; then
  echo "[slurm] partition=${partition:-unknown} partition_max=${partition_max:-unknown} qos=${qos:-none} qos_max=${qos_max:-none} effective_time=$resolved_time" >&2
else
  echo "[slurm] no finite max wall time discovered; submitting without a --time override" >&2
fi

printf '[slurm] command:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2

if [[ "$dry_run" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"