#!/usr/bin/env bash
# Shared hosted routing decision and installed pre-job execution boundary.
# Public API reads require no credentials; errors/rate limits never grant access.
set -euo pipefail

reject() {
  echo "Self-hosted admission denied: $1" >&2
  return 1
}

api() {
  local response
  response="$(curl --disable --fail --silent --show-error --proto '=https' \
    --connect-timeout 5 --max-time 20 \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2022-11-28' \
    "https://api.github.com/repos/ll7/robot_sf_ll7/$1")" || return 1
  # jq -e accepts an empty input stream: explicitly require one JSON document.
  jq -e -s 'length == 1 and (.[0] | type == "object" or type == "array")' \
    <<<"$response" >/dev/null 2>&1 || return 1
  printf '%s\n' "$response"
}

owner_approval() {
  local number="$1" head="$2" label="ci-owner:$2" pr events latest='' page
  pr="$(api "pulls/$number")" || { reject 'approval API unavailable'; return 1; }
  jq -e --arg head "$head" --arg label "$label" --argjson number "$number" '
    .number == $number and .state == "open" and
    .head.sha == $head and .head.repo.full_name == "ll7/robot_sf_ll7" and
    (.labels | type == "array") and any(.labels[]; .name == $label)
  ' <<<"$pr" >/dev/null 2>&1 || { reject 'no live exact-head approval'; return 1; }
  # A bounded, complete event history is required. Do not trust label presence
  # alone: collaborators can apply labels too. Re-applying revokes old authority.
  for ((page=1; page<=100; page++)); do
    events="$(api "issues/$number/events?per_page=100&page=$page")" || {
      reject 'approval history unavailable'; return 1;
    }
    jq -e '
      type == "array" and length <= 100 and
      all(.[]; (.event | type == "string") and
        (if .event == "labeled" or .event == "unlabeled" then
           (.label.name | type == "string") and
           (.actor.login | type == "string") and
           (.actor.type == "User" or .actor.type == "Bot")
         else true end))
    ' <<<"$events" >/dev/null 2>&1 || { reject 'invalid approval history'; return 1; }
    local matching
    matching="$(jq -c --arg label "$label" '
      [.[] | select((.event == "labeled" or .event == "unlabeled") and
                    .label.name == $label)] | last // empty
    ' <<<"$events")" || return 1
    [[ -z "$matching" ]] || latest="$matching"
    if [[ "$(jq length <<<"$events")" -lt 100 ]]; then
      [[ -n "$latest" ]] || { reject 'missing approval event'; return 1; }
      jq -e '.event == "labeled" and .actor.login == "ll7" and .actor.type == "User"' \
        <<<"$latest" >/dev/null 2>&1 || { reject 'approval not applied by owner'; return 1; }
      # Re-read live state after pagination to catch changed head/removed label.
      pr="$(api "pulls/$number")" || { reject 'approval recheck unavailable'; return 1; }
      jq -e --arg head "$head" --arg label "$label" '
        .state == "open" and .head.sha == $head and
        .head.repo.full_name == "ll7/robot_sf_ll7" and
        (.labels | type == "array") and any(.labels[]; .name == $label)
      ' <<<"$pr" >/dev/null 2>&1 || { reject 'approval changed'; return 1; }
      return 0
    fi
  done
  reject 'approval history exceeds safety bound'
}

admit() {
  [[ "${GITHUB_REPOSITORY:-}" == ll7/robot_sf_ll7 ]] || { reject 'repository'; return 1; }
  [[ "${GITHUB_ACTOR:-}" == ll7 && "${GITHUB_TRIGGERING_ACTOR:-}" == ll7 ]] || {
    reject 'actor'; return 1;
  }
  [[ "${GITHUB_EVENT_NAME:-}" == push || "${GITHUB_EVENT_NAME:-}" == pull_request ]] || {
    reject 'event'; return 1;
  }
  [[ -f "${GITHUB_EVENT_PATH:-}" ]] || { reject 'missing event'; return 1; }
  jq -e -s 'length == 1 and (.[0] | type == "object")' \
    "$GITHUB_EVENT_PATH" >/dev/null 2>&1 || { reject 'invalid event JSON'; return 1; }
  local range base head number comparison total='' count=0 page seen='[]' owner_only=true
  range="$(jq -er --arg event "$GITHUB_EVENT_NAME" '
    if .repository.full_name != "ll7/robot_sf_ll7" then error("repository")
    elif $event == "pull_request" then
      if .pull_request.head.repo.full_name != "ll7/robot_sf_ll7" or
         .pull_request.user.login != "ll7" or
         (.pull_request.number | type != "number") or
         .pull_request.number < 1 or
         (.pull_request.number | floor) != .pull_request.number
      then error("PR identity") else
        [.pull_request.base.sha, .pull_request.head.sha, (.pull_request.number | tostring)]
      end
    else [.before, .after, "0"] end |
    if all(.[0:2][]; type == "string" and test("^[0-9a-f]{40}$") and
           . != "0000000000000000000000000000000000000000") and .[0] != .[1]
    then join(" ") else error("range") end
  ' "$GITHUB_EVENT_PATH" 2>/dev/null)" || { reject 'unknown range or PR identity'; return 1; }
  read -r base head number <<<"$range" || { reject 'unknown range'; return 1; }
  if [[ "$GITHUB_EVENT_NAME" == push && "${GITHUB_SHA:-}" != "$head" ]]; then
    reject 'push head mismatch'; return 1;
  fi
  # Compare immutable endpoints, not the webhook's potentially truncated commits.
  # Validate totals, every page, uniqueness and final head before granting access.
  for ((page=1; page<=10; page++)); do
    comparison="$(api "compare/$base...$head?per_page=100&page=$page")" || {
      reject 'commit API unavailable'; return 1;
    }
    jq -e --arg base "$base" '
      .base_commit.sha == $base and (.status == "ahead" or .status == "diverged") and
      (.total_commits | type == "number") and .total_commits > 0 and
      .total_commits <= 1000 and (.total_commits | floor) == .total_commits and
      (.commits | type == "array") and (.commits | length) > 0 and
      (.commits | length) <= 100 and
      all(.commits[];
        (.sha | type == "string" and test("^[0-9a-f]{40}$")) and
        (.author.login | type == "string" and length > 0) and
        (.committer.login | type == "string" and length > 0) and
        (.author.type == "User" or .author.type == "Bot") and
        (.committer.type == "User" or .committer.type == "Bot"))
    ' <<<"$comparison" >/dev/null 2>&1 || { reject 'incomplete commit data'; return 1; }
    local page_total page_count
    page_total="$(jq -r .total_commits <<<"$comparison")" || return 1
    [[ -n "$total" ]] || total="$page_total"
    [[ "$total" == "$page_total" ]] || { reject 'changed commit total'; return 1; }
    page_count="$(jq '.commits | length' <<<"$comparison")" || return 1
    count=$((count + page_count))
    seen="$(jq -c --argjson seen "$seen" '$seen + [.commits[].sha]' <<<"$comparison")" || return 1
    jq -e 'length == (unique | length)' <<<"$seen" >/dev/null || {
      reject 'duplicate commit data'; return 1;
    }
    if ! jq -e 'all(.commits[];
      .author.login == "ll7" and .author.type == "User" and
      .committer.login == "ll7" and .committer.type == "User")' \
      <<<"$comparison" >/dev/null; then owner_only=false; fi
    if ((count == total)); then
      [[ "$(jq -r '.commits[-1].sha' <<<"$comparison")" == "$head" ]] || {
        reject 'missing range head'; return 1;
      }
      if [[ "$owner_only" == true ]]; then return 0; fi
      if [[ "$GITHUB_EVENT_NAME" == pull_request ]]; then owner_approval "$number" "$head"; return $?; fi
      reject 'non-owner author or committer'; return 1
    fi
    ((count < total && page_count == 100)) || { reject 'truncated commit range'; return 1; }
  done
  reject 'commit range exceeds safety bound'
}

if [[ "$#" == 1 && "$1" == --route ]]; then
  if (admit); then echo 'self_hosted=true'; else echo 'self_hosted=false'; fi
elif [[ "$#" == 0 ]]; then
  admit || exit 1
else
  reject 'invalid invocation'
  exit 1
fi
