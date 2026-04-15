#!/usr/bin/env bash
set -euo pipefail

# Snapshot Auxme partition pressure and suggest where to submit the next GPU job.
# Intended partitions: a30 and l40s.

PARTITIONS_CSV="a30,l40s"
SHOW_RECOMMEND=0
USER_NAME="${USER:-}"

usage() {
  cat <<'EOF'
Usage: scripts/dev/auxme_partition_status.sh [options]

Options:
  --partitions a30,l40s  Comma-separated partition list (default: a30,l40s)
  --user <name>          User for per-user running-job counts (default: $USER)
  --recommend            Print a machine-readable recommendation line
  -h, --help             Show help

Output columns:
  partition qos total_gpu alloc_gpu free_gpu running pending user_running slots_left score

Recommendation format:
  partition=<name> qos=<name>-gpu free_gpu=<n> pending=<n> slots_left=<n> score=<n>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partitions)
      PARTITIONS_CSV="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --recommend)
      SHOW_RECOMMEND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${USER_NAME}" ]]; then
  USER_NAME="$(whoami)"
fi

if ! command -v squeue >/dev/null 2>&1 || ! command -v scontrol >/dev/null 2>&1; then
  echo "squeue and scontrol are required." >&2
  exit 2
fi

parse_gpu_tres() {
  local tres="$1"
  if [[ -z "${tres}" ]]; then
    echo 0
    return
  fi
  echo "${tres}" | tr ',' '\n' | awk -F= '$1=="gres/gpu" {sum+=$2} END {print sum+0}'
}

partition_has_node() {
  local partitions_field="$1"
  local target="$2"
  [[ ",${partitions_field}," == *",${target},"* ]]
}

collect_partition_row() {
  local partition="$1"
  local qos="${partition}-gpu"
  local total_gpu=0
  local alloc_gpu=0

  while IFS= read -r line; do
    local part_field cfg_tres alloc_tres
    part_field="$(echo "${line}" | sed -n 's/.* Partitions=\([^ ]*\).*/\1/p')"
    if ! partition_has_node "${part_field}" "${partition}"; then
      continue
    fi
    cfg_tres="$(echo "${line}" | sed -n 's/.* CfgTRES=\([^ ]*\).*/\1/p')"
    alloc_tres="$(echo "${line}" | sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p')"
    total_gpu=$((total_gpu + $(parse_gpu_tres "${cfg_tres}")))
    alloc_gpu=$((alloc_gpu + $(parse_gpu_tres "${alloc_tres}")))
  done < <(scontrol show node -o 2>/dev/null || true)

  local free_gpu running pending user_running slots_left score
  free_gpu=$((total_gpu - alloc_gpu))
  if (( free_gpu < 0 )); then
    free_gpu=0
  fi

  running="$(squeue -h -p "${partition}" -t RUNNING 2>/dev/null | wc -l | tr -d ' ')"
  pending="$(squeue -h -p "${partition}" -t PENDING 2>/dev/null | wc -l | tr -d ' ')"
  user_running="$(squeue -h -u "${USER_NAME}" -p "${partition}" -t RUNNING 2>/dev/null | wc -l | tr -d ' ')"

  # Auxme gpu qos profiles currently limit to 2 jobs/user/qos.
  slots_left=$((2 - user_running))
  if (( slots_left < 0 )); then
    slots_left=0
  fi

  # Heuristic score: prefer free GPUs and available user slots, penalize long pending queues.
  score=$((free_gpu * 100 + slots_left * 25 - pending * 7))
  if (( slots_left == 0 )); then
    score=-9999
  fi

  echo "${partition}|${qos}|${total_gpu}|${alloc_gpu}|${free_gpu}|${running}|${pending}|${user_running}|${slots_left}|${score}"
}

IFS=',' read -r -a PARTITIONS <<< "${PARTITIONS_CSV}"
rows=()
for partition in "${PARTITIONS[@]}"; do
  rows+=("$(collect_partition_row "${partition}")")
done

if [[ "${SHOW_RECOMMEND}" == "0" ]]; then
  printf '%-10s %-10s %-9s %-9s %-8s %-8s %-8s %-12s %-10s %-7s\n' \
    "partition" "qos" "total_gpu" "alloc_gpu" "free_gpu" "running" "pending" "user_running" "slots_left" "score"
  for row in "${rows[@]}"; do
    IFS='|' read -r partition qos total_gpu alloc_gpu free_gpu running pending user_running slots_left score <<< "${row}"
    printf '%-10s %-10s %-9s %-9s %-8s %-8s %-8s %-12s %-10s %-7s\n' \
      "${partition}" "${qos}" "${total_gpu}" "${alloc_gpu}" "${free_gpu}" "${running}" "${pending}" "${user_running}" "${slots_left}" "${score}"
  done
fi

best_row="$(printf '%s\n' "${rows[@]}" | sort -t'|' -k10,10nr | head -n 1)"
IFS='|' read -r best_partition best_qos _best_total _best_alloc best_free best_running best_pending best_user_running best_slots best_score <<< "${best_row}"

if [[ "${SHOW_RECOMMEND}" == "1" ]]; then
  echo "partition=${best_partition} qos=${best_qos} free_gpu=${best_free} pending=${best_pending} slots_left=${best_slots} score=${best_score}"
fi
