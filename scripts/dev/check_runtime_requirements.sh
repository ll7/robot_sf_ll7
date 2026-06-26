#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/check_runtime_requirements.sh [--strict]

Report non-Python runtime tools used by Robot SF development, CI parity, and optional
machine-specific workflows. The default mode is advisory: required core tools fail the check,
while optional capabilities are reported as present or missing.

Options:
  --strict  Also fail when recommended CI-parity tools are missing.
  -h, --help             Show help
EOF
}

strict=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      strict=true
      shift
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

failures=0

print_result() {
  local status="$1"
  local name="$2"
  local detail="$3"
  printf "%-8s %-24s %s\n" "$status" "$name" "$detail"
}

require_command() {
  local name="$1"
  local why="$2"
  if command -v "$name" >/dev/null 2>&1; then
    print_result "ok" "$name" "$(command -v "$name") - $why"
  else
    print_result "missing" "$name" "$why"
    failures=$((failures + 1))
  fi
}

recommend_command() {
  local name="$1"
  local why="$2"
  if command -v "$name" >/dev/null 2>&1; then
    print_result "ok" "$name" "$(command -v "$name") - $why"
  else
    print_result "missing" "$name" "$why"
    if [[ "$strict" == true ]]; then
      failures=$((failures + 1))
    fi
  fi
}

optional_command() {
  local name="$1"
  local why="$2"
  if command -v "$name" >/dev/null 2>&1; then
    print_result "ok" "$name" "$(command -v "$name") - $why"
  else
    print_result "optional" "$name" "$why"
  fi
}

echo "Robot SF runtime requirements"
echo
echo "Core repository tools"
require_command git "source checkout and worktree management"
require_command uv "Python environment, dependency sync, and command runner"
require_command python3 "bootstrap interpreter; project commands should still run through uv"

echo
echo "GitHub and CI-parity tools"
recommend_command gh "GitHub issues, PRs, checks, and Actions log inspection"
recommend_command jq "JSON validation in CI smoke paths and local diagnostics"
recommend_command ffmpeg "video/rendering workflows and GitHub CI parity"

gh_act_pattern='(^|[[:space:]])((nektos/)?gh-act|gh[[:space:]]+act|act)([[:space:]]|$)'
if command -v gh >/dev/null 2>&1 && gh extension list 2>/dev/null | grep -qE "$gh_act_pattern"; then
  version="$(gh act --version 2>/dev/null || true)"
  print_result "ok" "gh-act" "${version:-GitHub CLI extension installed}"
else
  print_result "optional" "gh-act" "install with: gh extension install https://github.com/nektos/gh-act"
fi
if command -v gh >/dev/null 2>&1 && gh extension list 2>/dev/null | grep -qE '(^|[[:space:]])basecamp/gh-signoff([[:space:]]|$)'; then
  version="$(gh signoff version 2>/dev/null || true)"
  print_result "ok" "gh-signoff" "${version:-GitHub CLI extension installed}"
else
  print_result "optional" "gh-signoff" "advisory local CI statuses; auto-installed by scripts/dev/local_signoff.sh"
fi

echo
echo "Container and accelerator capabilities"
if command -v docker >/dev/null 2>&1; then
  if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    docker_version="$(docker version --format '{{.Server.Version}}')"
    print_result "ok" "docker" "daemon available, server ${docker_version}"
  else
    print_result "optional" "docker" "CLI present but daemon unavailable"
  fi
else
  print_result "optional" "docker" "needed for gh-act execution, CARLA Docker, and benchmark repro"
fi
optional_command nvidia-smi "GPU diagnostics for CUDA/GPU benchmark hosts"
optional_command sbatch "SLURM/Auxme submission hosts"

echo
echo "Headless rendering environment"
print_result "info" "MPLBACKEND" "${MPLBACKEND:-recommended local value: Agg}"
print_result "info" "SDL_VIDEODRIVER" "${SDL_VIDEODRIVER:-recommended local value: dummy}"
print_result "info" "DISPLAY" "${DISPLAY:-empty is acceptable for headless tests}"

if [[ "$failures" -gt 0 ]]; then
  echo
  if [[ "$strict" == true ]]; then
    echo "Runtime requirement check failed with ${failures} missing required/recommended tool(s)." >&2
  else
    echo "Runtime requirement check failed with ${failures} missing required tool(s)." >&2
  fi
  exit 1
fi
