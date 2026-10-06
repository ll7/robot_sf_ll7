#!/usr/bin/env bash
# Reject untrusted jobs before the runner executes any workflow step.
set -euo pipefail

deny() {
  echo "Runner job rejected by trust hook" >&2
  exit 1
}

[[ "${GITHUB_REPOSITORY:-}" == ll7/robot_sf_ll7 ]] || deny
[[ "${GITHUB_ACTOR:-}" == ll7 ]] || deny
[[ "${GITHUB_TRIGGERING_ACTOR:-}" == ll7 ]] || deny
[[ "${GITHUB_EVENT_NAME:-}" == push || "${GITHUB_EVENT_NAME:-}" == pull_request ]] || deny
[[ -f "${GITHUB_EVENT_PATH:-}" ]] || deny

jq -e --arg event "$GITHUB_EVENT_NAME" '
  .repository.full_name == "ll7/robot_sf_ll7" and
  (if $event == "pull_request" then
    .pull_request.head.repo.full_name == "ll7/robot_sf_ll7" and
    .pull_request.user.login == "ll7"
  else true end)
' "$GITHUB_EVENT_PATH" >/dev/null 2>&1 || deny
