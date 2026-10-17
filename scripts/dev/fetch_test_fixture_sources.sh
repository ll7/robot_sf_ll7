#!/usr/bin/env bash
# Retrieve historical source objects required by fail-closed fixture validation.
# Remote preservation branches keep the pinned objects reachable independently
# of release refs; they must not be deleted with PR branches.
set -euo pipefail

fetch_fixture_source() {
  local fixture="$1" source_commit="$2" command_script="$3" source_path="$4"
  local preservation_ref="refs/heads/fixtures/${fixture}-historical-source"
  local local_ref="refs/fixture-sources/${fixture}"

  if ! git cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
    git fetch --no-tags origin "${preservation_ref}:${local_ref}"
    if [[ "$(git rev-parse "${local_ref}^{commit}")" != "$source_commit" ]]; then
      echo "Historical fixture source ref does not match the pinned commit: ${fixture}" >&2
      return 1
    fi
  fi

  git cat-file -e "${source_commit}:${command_script}"
  git cat-file -e "${source_commit}:${source_path}"
  echo "Historical fixture source verified: ${fixture} ${source_commit}"
}

fetch_fixture_source issue-6474 2fc4498cc5499bd3569eb1ac941a3029e0f51040 \
  scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py \
  docs/context/evidence/issue_6474_social_compliance_preregistration.json
fetch_fixture_source issue-6944 8f9438632e794f084db72bb016a14b539bbca648 \
  scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  docs/context/evidence/issue_6944_brne_candidate_transition_summary.json
