#!/usr/bin/env bash
# Retrieve historical source objects required by fail-closed fixture validation.
# The remote preservation branch keeps the exact issue-6474 object reachable;
# it is independent of release refs and must not be deleted with PR branches.
set -euo pipefail

source_commit=2fc4498cc5499bd3569eb1ac941a3029e0f51040
preservation_ref=refs/heads/fixtures/issue-6474-historical-source
local_ref=refs/fixture-sources/issue-6474

if ! git cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
  git fetch --no-tags origin "${preservation_ref}:${local_ref}"
  actual_commit="$(git rev-parse "${local_ref}^{commit}")"
  if [[ "$actual_commit" != "$source_commit" ]]; then
    echo "Historical fixture source ref does not match the pinned commit." >&2
    exit 1
  fi
fi

git cat-file -e "${source_commit}:scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py"
git cat-file -e "${source_commit}:docs/context/evidence/issue_6474_social_compliance_preregistration.json"
echo "Historical fixture source verified: ${source_commit}"
