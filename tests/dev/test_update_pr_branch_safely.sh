#!/usr/bin/env bash
# Smoke tests for scripts/dev/update_pr_branch_safely.sh (issue #5775,
# deleted-source-ref restore coverage for issue #6689).
#
# The wrapper shells out to `gh` for metadata and to `git` for the local
# fallback.  We mock both so the tests stay fully offline and do not depend on
# GitHub availability, credentials, or a real remote.  The mock records which
# mutating path (REST update-branch vs. local rebase/push vs. plain restore
# push) the wrapper selected.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${REPO_ROOT}/scripts/dev/update_pr_branch_safely.sh"
PASS=0
FAIL=0

MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$MOCK_DIR"' EXIT
export MOCK_DIR

assert_ok() {
  local desc="$1" rc="$2"
  if [[ $rc -eq 0 ]]; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc (expected exit 0, got $rc)"
    FAIL=$((FAIL + 1))
  fi
}

assert_fail() {
  local desc="$1" rc="$2"
  if [[ $rc -ne 0 ]]; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc (expected nonzero exit)"
    FAIL=$((FAIL + 1))
  fi
}

assert_json() {
  local desc="$1" payload="$2"
  if python3 -c 'import json, sys; json.load(sys.stdin)' <<<"$payload"; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc (invalid JSON)"
    FAIL=$((FAIL + 1))
  fi
}

# Mock gh: returns PR metadata for the pulls endpoint and makes the REST
# update-branch endpoint fail (404-style), so the tests can exercise both the
# guarded local fallback and the no-fallback branch. The mock honors the
# wrapper's compact TSV metadata selector (head SHA, head ref, base ref, head
# repository, base repository) and scalar selectors used by older callers.
# PR 3 is a cross-fork PR whose head lives in another repository.
make_gh() {
  cat > "${MOCK_DIR}/gh" <<'EOF'
#!/usr/bin/env bash
# Minimal gh mock honoring `--jq '.head.sha'`, `.head.ref`, `.base.ref`,
# and `repo view --json nameWithOwner --jq .nameWithOwner`.
printf '%s\n' "$*" >> "${MOCK_DIR}/gh_calls"
jq=""
url=""
prev=""
for a in "$@"; do
  if [[ "$prev" == "--jq" ]]; then jq="$a"; fi
  case "$a" in repos/*/pulls/*) url="$a";; esac
  prev="$a"
done
if [[ "$1 $2" == "repo view" ]]; then
  printf 'owner/repo\n'; exit 0
fi
case "$url" in
  *"/pulls/1/update-branch")
    if [[ "${MOCK_UPDATE_BRANCH_SUCCESS:-0}" -eq 1 ]]; then
      printf 'Branch update scheduled\n'
      exit 0
    fi
    echo "gh: 'update-branch' is not a gh command" >&2
    exit 1
    ;;
  *"/pulls/1")
    if [[ "$jq" == *"@tsv"* ]]; then
      printf 'headsha\tfeature\tmain\towner/repo\towner/repo'
    else
      case "$jq" in
        ".head.sha") printf 'headsha';;
        ".head.ref") printf 'feature';;
        ".base.ref") printf 'main';;
        *) printf '{"head":{"sha":"headsha"},"head_ref":"feature","base_ref":"main"}';;
      esac
    fi
    exit 0
    ;;
  *"/pulls/2")
    if [[ "$jq" == *"@tsv"* ]]; then
      printf 'othersha\tfeature2\tmain\towner/repo\towner/repo'
    else
      case "$jq" in
        ".head.sha") printf 'othersha';;
        ".head.ref") printf 'feature2';;
        ".base.ref") printf 'main';;
      esac
    fi
    exit 0
    ;;
  *"/pulls/3")
    if [[ "$jq" == *"@tsv"* ]]; then
      printf 'forksha\tfeature3\tmain\tfork/repo\towner/repo'
    else
      case "$jq" in
        ".head.sha") printf 'forksha';;
        ".head.ref") printf 'feature3';;
        ".base.ref") printf 'main';;
      esac
    fi
    exit 0
    ;;
  *)
    echo "gh mock: unhandled $*" >&2
    exit 1
    ;;
esac
EOF
  chmod +x "${MOCK_DIR}/gh"

  # Default git mock: answers the deleted-source-ref probe as "ref present" so
  # tests that do not exercise restore keep the historical update path, and
  # refuses every other (mutating) git operation. Tests that exercise restore
  # overwrite this stub with a scenario-specific one.
  cat > "${MOCK_DIR}/git" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${MOCK_DIR}/git_calls"
if [[ "$1" == "ls-remote" ]]; then
  ref=""
  for a in "$@"; do ref="$a"; done
  printf 'headsha\trefs/heads/%s\n' "$ref"
  exit 0
fi
echo "git mock: refusing $*" >&2
exit 1
EOF
  chmod +x "${MOCK_DIR}/git"
}

# 1. Missing --expected-head-sha must fail closed before any network call.
make_gh
RC=0
PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" 1 --repo owner/repo 2>/dev/null >/dev/null || RC=$?
assert_fail "missing --expected-head-sha rejected" "$RC"

# 2. Head mismatch must fail closed (expected != live) without mutating.
make_gh
RC=0
PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" 1 --repo owner/repo \
  --expected-head-sha wrongsha >/dev/null 2>&1 || RC=$?
assert_fail "head mismatch rejected" "$RC"

# 3. When gh update-branch is unavailable and local fallback is explicitly
#    disabled, the wrapper must report a machine-readable error without
#    invoking any local git mutation.
make_gh
RC=0
OUT="$(PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha --no-local-fallback 2>/dev/null)" || RC=$?
assert_json "machine-readable output is valid JSON" "$OUT"
if echo "$OUT" | grep -q '"status":"head_mismatch"' || echo "$OUT" | grep -q '"status":"error"'; then
  echo "PASS: machine-readable result emitted on unavailable update-branch"
  PASS=$((PASS + 1))
else
  echo "FAIL: no machine-readable result on unavailable update-branch"
  FAIL=$((FAIL + 1))
fi
assert_fail "fails closed when update-branch unavailable and no fallback" "$RC"

QUOTED_OUT="$(PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo 'owner/"quoted' \
  --expected-head-sha headsha --no-local-fallback 2>/dev/null)" || true
assert_json "quoted repository values remain valid JSON" "$QUOTED_OUT"
if python3 -c 'import json, sys; assert json.load(sys.stdin)["repo"] == "owner/\"quoted"' <<<"$QUOTED_OUT"; then
  echo "PASS: quoted repository value is preserved"
  PASS=$((PASS + 1))
else
  echo "FAIL: quoted repository value was not preserved"
  FAIL=$((FAIL + 1))
fi

# 4. --dry-run must stop after read-only metadata validation. In particular,
#    it must not invoke the update-branch endpoint even when that mocked
#    endpoint would succeed (issue #6439).
make_gh
: > "${MOCK_DIR}/gh_calls"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/lease_calls"
# A dry-run should not need any local Git operation after the remote metadata
# guard. Fail if the helper reaches this stub.
cat > "${MOCK_DIR}/git" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${MOCK_DIR}/git_calls"
echo "git stub: refusing dry-run op $*" >&2
exit 1
EOF
chmod +x "${MOCK_DIR}/git"
DRY_RUN_LEASE_HELPER="${MOCK_DIR}/dry_run_lease.py"
cat > "$DRY_RUN_LEASE_HELPER" <<'EOF'
#!/usr/bin/env python3
import os
import sys

with open(os.path.join(os.environ["MOCK_DIR"], "lease_calls"), "a") as call_log:
    call_log.write(" ".join(sys.argv[1:]) + "\n")
EOF
chmod +x "$DRY_RUN_LEASE_HELPER"
# The helper resolves its lease helper relative to itself. Use a copied wrapper
# so this regression can fail if dry-run reaches either lease action.
DRY_RUN_SCRIPT="${MOCK_DIR}/update_pr_branch_safely_dry_run.sh"
cp "$SCRIPT" "$DRY_RUN_SCRIPT"
python3 - "$DRY_RUN_SCRIPT" "$DRY_RUN_LEASE_HELPER" <<'PY'
import sys

script_path, lease_helper = sys.argv[1:]
source = open(script_path).read()
source = source.replace(
    'LEASE_HELPER="${SCRIPT_DIR}/pr_gate_lease.py"',
    f'LEASE_HELPER="{lease_helper}"',
)
open(script_path, "w").write(source)
PY
chmod +x "$DRY_RUN_SCRIPT"
RC=0
OUT="$(MOCK_UPDATE_BRANCH_SUCCESS=1 PATH="${MOCK_DIR}:$PATH" bash "$DRY_RUN_SCRIPT" 1 --repo owner/repo \
  --expected-head-sha headsha --remote custom --dry-run 2>/dev/null)" || RC=$?
assert_json "dry-run output is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "dry_run" and d["updated"] is False' <<<"$OUT"; then
  echo "PASS: dry-run reports a non-mutating plan"
  PASS=$((PASS + 1))
else
  echo "FAIL: dry-run did not report a non-mutating plan"
  FAIL=$((FAIL + 1))
fi
if python3 -c 'import json, sys; assert json.load(sys.stdin)["remote"] == "custom"' <<<"$OUT"; then
  echo "PASS: configured remote is preserved in the result"
  PASS=$((PASS + 1))
else
  echo "FAIL: configured remote was not preserved in the result"
  FAIL=$((FAIL + 1))
fi
if grep -q '/pulls/1/update-branch' "${MOCK_DIR}/gh_calls"; then
  echo "FAIL: dry-run invoked the remote update-branch endpoint"
  FAIL=$((FAIL + 1))
else
  echo "PASS: dry-run skipped the remote update-branch endpoint"
  PASS=$((PASS + 1))
fi
if [[ -s "${MOCK_DIR}/git_calls" ]]; then
  echo "FAIL: dry-run invoked local Git"
  FAIL=$((FAIL + 1))
else
  echo "PASS: dry-run skipped local Git"
  PASS=$((PASS + 1))
fi
if [[ -s "${MOCK_DIR}/lease_calls" ]]; then
  echo "FAIL: dry-run invoked the PR-gate lease helper"
  FAIL=$((FAIL + 1))
else
  echo "PASS: dry-run skipped the PR-gate lease helper"
  PASS=$((PASS + 1))
fi
assert_ok "dry-run exits 0" "$RC"

# The same mocked successful endpoint must remain available to a non-dry-run
# invocation, proving the safety guard did not disable the primary update path.
# Reinstall the default mocks so the deleted-source-ref probe reports the ref
# as present and every mutating git operation stays refused.
make_gh
: > "${MOCK_DIR}/gh_calls"
: > "${MOCK_DIR}/git_calls"
RC=0
OUT="$(MOCK_UPDATE_BRANCH_SUCCESS=1 PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" 1 --repo owner/repo \
  --expected-head-sha headsha 2>/dev/null)" || RC=$?
assert_ok "non-dry-run REST update exits 0" "$RC"
assert_json "non-dry-run REST output is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "update_requested" and d["updated"] is True and d["source_ref_restored"] is False' <<<"$OUT"; then
  echo "PASS: non-dry-run retains the REST update path"
  PASS=$((PASS + 1))
else
  echo "FAIL: non-dry-run did not report the REST update path"
  FAIL=$((FAIL + 1))
fi
if grep -q '/pulls/1/update-branch' "${MOCK_DIR}/gh_calls"; then
  echo "PASS: non-dry-run invoked the remote update-branch endpoint"
  PASS=$((PASS + 1))
else
  echo "FAIL: non-dry-run skipped the remote update-branch endpoint"
  FAIL=$((FAIL + 1))
fi
if grep -Eq '^(fetch|rebase|push) ' "${MOCK_DIR}/git_calls"; then
  echo "FAIL: successful REST update invoked mutating local Git"
  FAIL=$((FAIL + 1))
else
  echo "PASS: successful REST update skipped mutating local Git"
  PASS=$((PASS + 1))
fi

# 5. The local fallback must exercise fetch, rebase, lease-protected push, and
#    post-push verification with the configured remote.
make_gh
cat > "${MOCK_DIR}/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${MOCK_DIR}/git_calls"
case "$*" in
  "rev-parse --git-common-dir")
    mkdir -p "${MOCK_DIR}/git-common"
    printf '%s\n' "${MOCK_DIR}/git-common"
    ;;
  "rev-parse --show-toplevel") printf '%s\n' "${REPO_ROOT}";;
  "rev-parse --abbrev-ref HEAD") printf 'feature';;
  "rev-parse HEAD")
    if [[ -f "${MOCK_DIR}/rebased" ]]; then printf 'newhead'; else printf 'headsha'; fi
    ;;
  "fetch custom main feature") :;;
  "rebase custom/main") : > "${MOCK_DIR}/rebased";;
  "push --force-with-lease=custom/feature:headsha custom HEAD:refs/heads/feature") :;;
  "ls-remote --heads custom feature")
    # Deleted-source-ref pre-check: report the head ref as present so this
    # fallback scenario runs unchanged.
    printf 'headsha\trefs/heads/feature\n'
    ;;
  "ls-remote --heads custom refs/heads/feature")
    if [[ "${EMPTY_VERIFY:-0}" -eq 1 ]]; then exit 0; fi
    printf 'newhead\trefs/heads/feature\n'
    ;;
  *) echo "git mock: unhandled $*" >&2; exit 1;;
esac
EOF
chmod +x "${MOCK_DIR}/git"
RC=0
OUT="$(PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha --remote custom 2>/dev/null)" || RC=$?
assert_ok "successful local fallback exits 0" "$RC"
assert_json "successful fallback output is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "fallback_local_rebase" and d["remote"] == "custom" and d["updated"] is True' <<<"$OUT"; then
  echo "PASS: successful fallback reports configured remote and update"
  PASS=$((PASS + 1))
else
  echo "FAIL: successful fallback result fields are incorrect"
  FAIL=$((FAIL + 1))
fi
if grep -q '^rebase custom/main$' "${MOCK_DIR}/git_calls" && ! grep -q 'origin/main' "${MOCK_DIR}/git_calls"; then
  echo "PASS: fallback rebases onto configured remote"
  PASS=$((PASS + 1))
else
  echo "FAIL: fallback did not use configured remote for rebase"
  FAIL=$((FAIL + 1))
fi

# An empty post-push lookup must fail closed rather than report success.
rm -f "${MOCK_DIR}/rebased"
RC=0
OUT="$(EMPTY_VERIFY=1 PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha --remote custom 2>/dev/null)" || RC=$?
assert_fail "empty post-push verification rejected" "$RC"
assert_json "empty-verification failure output is valid JSON" "$OUT"
if echo "$OUT" | grep -q 'post-push verification failed: remote SHA was empty'; then
  echo "PASS: empty post-push SHA is reported as an error"
  PASS=$((PASS + 1))
else
  echo "FAIL: empty post-push SHA was not reported"
  FAIL=$((FAIL + 1))
fi

# 6. -h / --help print usage and exit 0 without touching GitHub; unknown flags
# are still rejected. A deliberately failing stub keeps these checks offline and
# proves the help branch exits before repository resolution.
cat > "${MOCK_DIR}/gh" <<'EOF'
#!/usr/bin/env bash
echo "gh mock: help must not invoke gh ($*)" >&2
exit 99
EOF
chmod +x "${MOCK_DIR}/gh"

RC=0
HELP_STDOUT="${MOCK_DIR}/help.stdout"
HELP_STDERR="${MOCK_DIR}/help.stderr"
PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --help >"$HELP_STDOUT" 2>"$HELP_STDERR" || RC=$?
assert_ok "--help exits 0" "$RC"
OUT="$(<"$HELP_STDOUT")"
if echo "$OUT" | grep -q 'Usage:' && echo "$OUT" | grep -q 'Options:'; then
  echo "PASS: --help prints usage and option text"
  PASS=$((PASS + 1))
else
  echo "FAIL: --help did not print usage and option text"
  FAIL=$((FAIL + 1))
fi
if [[ ! -s "$HELP_STDERR" ]]; then
  echo "PASS: --help writes help text to stdout"
  PASS=$((PASS + 1))
else
  echo "FAIL: --help unexpectedly writes to stderr"
  FAIL=$((FAIL + 1))
fi

RC=0
HELP_STDOUT="${MOCK_DIR}/short-help.stdout"
HELP_STDERR="${MOCK_DIR}/short-help.stderr"
PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" -h >"$HELP_STDOUT" 2>"$HELP_STDERR" || RC=$?
assert_ok "-h exits 0" "$RC"
OUT="$(<"$HELP_STDOUT")"
if echo "$OUT" | grep -q 'Usage:' && echo "$OUT" | grep -q 'Options:'; then
  echo "PASS: -h prints usage and option text"
  PASS=$((PASS + 1))
else
  echo "FAIL: -h did not print usage and option text"
  FAIL=$((FAIL + 1))
fi
if [[ ! -s "$HELP_STDERR" ]]; then
  echo "PASS: -h writes help text to stdout"
  PASS=$((PASS + 1))
else
  echo "FAIL: -h unexpectedly writes to stderr"
  FAIL=$((FAIL + 1))
fi

RC=0
bash "$SCRIPT" --bogus 2>/dev/null || RC=$?
assert_fail "unknown flag rejected" "$RC"

# 7. A registered gate worktree that has vanished must fail closed before the
#    local branch-switch/conflict-resolution path, reporting the lease cleanup
#    owner (issue #5967). The guard helper is mocked to report exists=false with
#    a cleanup owner; the local fallback must NOT run any git mutation.
make_gh
GONE_WT="${MOCK_DIR}/gone-gate-wt"
mkdir -p "$GONE_WT"
GUARD_DIR="${MOCK_DIR}/guard"
mkdir -p "$GUARD_DIR"
cat > "${GUARD_DIR}/gate_worktree_guard.py" <<'EOF'
#!/usr/bin/env python3
import json, sys
# Mock of the gate worktree guard's `verify` subcommand.
assert sys.argv[1] == "verify", sys.argv
path = sys.argv[sys.argv.index("--path") + 1]
print(json.dumps({
    "schema": "gate_worktree_guard.v1",
    "path": path,
    "exists": False,
    "classification": "missing",
    "cleanup_owner": "owner=auto-smart-routing; pr=#5819; gate=gate-5819",
}))
sys.exit(1)
EOF
chmod +x "${GUARD_DIR}/gate_worktree_guard.py"
# Point the script's helper resolution at the mock by shadowing via PATH is not
# possible (it uses SCRIPT_DIR). Instead, write a tiny wrapper that injects the
# mock path through PYTHONPATH is not honored; use a sed-free approach: copy the
# real script but rewrite GUARD_HELPER to the mock.
cp "$SCRIPT" "${MOCK_DIR}/update_pr_branch_safely_wg.sh"
python3 - "$GONE_WT" "${GUARD_DIR}" "${MOCK_DIR}" <<'PY'
import sys
gone, guard_dir, mock_dir = sys.argv[1:]
src = open(f"{mock_dir}/update_pr_branch_safely_wg.sh").read()
src = src.replace(
    'GUARD_HELPER="${SCRIPT_DIR}/gate_worktree_guard.py"',
    f'GUARD_HELPER="{guard_dir}/gate_worktree_guard.py"',
)
open(f"{mock_dir}/update_pr_branch_safely_wg.sh", "w").write(src)
PY
chmod +x "${MOCK_DIR}/update_pr_branch_safely_wg.sh"
RC=0
OUT="$(PATH="${MOCK_DIR}:$PATH" bash "${MOCK_DIR}/update_pr_branch_safely_wg.sh" \
  --pr 1 --repo owner/repo --expected-head-sha headsha --gate-worktree-path "$GONE_WT" 2>/dev/null)" || RC=$?
assert_json "vanished-gate-worktree output is valid JSON" "$OUT"
assert_fail "vanished gate worktree fails closed" "$RC"
if echo "$OUT" | grep -q '"status":"gate_worktree_missing"'; then
  echo "PASS: vanished gate worktree reports gate_worktree_missing"
  PASS=$((PASS + 1))
else
  echo "FAIL: vanished gate worktree did not report gate_worktree_missing"
  FAIL=$((FAIL + 1))
fi
if echo "$OUT" | grep -q 'auto-smart-routing'; then
  echo "PASS: vanished gate worktree reports the lease cleanup owner"
  PASS=$((PASS + 1))
else
  echo "FAIL: vanished gate worktree did not report the cleanup owner"
  FAIL=$((FAIL + 1))
fi
# The local fallback must not have run git rebase/push; there is no git call that
# would have mutated. The mock git (if any) is absent, so command-not-found would
# have surfaced; ensure no git rebase reached the stub by checking exit closed.

# 8. Malformed guard output must produce a deterministic JSON error rather than
#    a second parser traceback or an unstructured shell failure.
BAD_GUARD_DIR="${MOCK_DIR}/bad-guard"
mkdir -p "$BAD_GUARD_DIR"
cat > "${BAD_GUARD_DIR}/gate_worktree_guard.py" <<'EOF'
#!/usr/bin/env python3
print("not-json")
raise SystemExit(1)
EOF
chmod +x "${BAD_GUARD_DIR}/gate_worktree_guard.py"
cp "$SCRIPT" "${MOCK_DIR}/update_pr_branch_safely_bad_guard.sh"
python3 - "$BAD_GUARD_DIR" "$MOCK_DIR" <<'PY'
import sys

bad_guard_dir, mock_dir = sys.argv[1:]
script_path = f"{mock_dir}/update_pr_branch_safely_bad_guard.sh"
source = open(script_path).read()
source = source.replace(
    'GUARD_HELPER="${SCRIPT_DIR}/gate_worktree_guard.py"',
    f'GUARD_HELPER="{bad_guard_dir}/gate_worktree_guard.py"',
)
open(script_path, "w").write(source)
PY
chmod +x "${MOCK_DIR}/update_pr_branch_safely_bad_guard.sh"
RC=0
OUT="$(PATH="${MOCK_DIR}:$PATH" bash "${MOCK_DIR}/update_pr_branch_safely_bad_guard.sh" \
  --pr 1 --repo owner/repo --expected-head-sha headsha --gate-worktree-path "$GONE_WT" 2>/dev/null)" || RC=$?
assert_json "malformed guard output remains valid JSON" "$OUT"
assert_fail "malformed guard output fails closed" "$RC"
if echo "$OUT" | grep -q '"status":"error"'; then
  echo "PASS: malformed guard output reports a deterministic error"
  PASS=$((PASS + 1))
else
  echo "FAIL: malformed guard output did not report a deterministic error"
  FAIL=$((FAIL + 1))
fi

# 9. Deleted PR source ref restore (issue #6689): when the head branch is
#    missing on the remote, the wrapper restores refs/heads/<head-ref> with a
#    plain (non-force) push of the immutable PR head SHA before running the
#    update path. Restore failures fail closed, dry-run never restores,
#    cross-fork PRs fail closed, and a concurrently reappeared ref is
#    re-detected so the normal update path continues without overwriting.
make_gh
cat > "${MOCK_DIR}/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${MOCK_DIR}/git_calls"
case "$*" in
  "ls-remote --heads origin feature")
    if [[ -f "${MOCK_DIR}/ref_present" ]]; then printf 'headsha\trefs/heads/feature\n'; fi
    ;;
  "ls-remote --heads origin feature3")
    if [[ -f "${MOCK_DIR}/ref_present_f3" ]]; then printf 'forksha\trefs/heads/feature3\n'; fi
    ;;
  "fetch origin headsha")
    if [[ "${MOCK_FETCH_FAIL:-0}" -eq 1 ]]; then exit 1; fi
    : > "${MOCK_DIR}/fetched"
    ;;
  "rev-parse FETCH_HEAD")
    if [[ -f "${MOCK_DIR}/fetched" ]]; then printf 'headsha'; else exit 1; fi
    ;;
  "push origin headsha:refs/heads/feature")
    if [[ "${MOCK_RESTORE_PUSH_REJECT:-0}" -eq 1 ]]; then
      if [[ "${MOCK_RESTORE_REAPPEAR:-0}" -eq 1 ]]; then : > "${MOCK_DIR}/ref_present"; fi
      exit 1
    fi
    : > "${MOCK_DIR}/ref_present"
    ;;
  "ls-remote --heads origin refs/heads/feature")
    if [[ -f "${MOCK_DIR}/ref_present" ]]; then printf 'headsha\trefs/heads/feature\n'; fi
    ;;
  *) echo "git mock: unhandled $*" >&2; exit 1;;
esac
EOF
chmod +x "${MOCK_DIR}/git"

# 9a. A deleted head ref is restored at the expected head SHA before the
#     update path runs; the restore is reported in the JSON result.
rm -f "${MOCK_DIR}/ref_present" "${MOCK_DIR}/fetched"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/gh_calls"
RC=0
OUT="$(MOCK_UPDATE_BRANCH_SUCCESS=1 PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha 2>/dev/null)" || RC=$?
assert_ok "deleted head ref restore followed by REST update exits 0" "$RC"
assert_json "restore result is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "update_requested" and d["updated"] is True and d["source_ref_restored"] is True' <<<"$OUT"; then
  echo "PASS: restore is reported in the JSON result before the update path"
  PASS=$((PASS + 1))
else
  echo "FAIL: restore was not reported in the JSON result"
  FAIL=$((FAIL + 1))
fi
if grep -q '^fetch origin headsha$' "${MOCK_DIR}/git_calls" \
  && grep -q '^push origin headsha:refs/heads/feature$' "${MOCK_DIR}/git_calls" \
  && grep -q '^ls-remote --heads origin refs/heads/feature$' "${MOCK_DIR}/git_calls"; then
  echo "PASS: restore fetched, pushed, and verified the immutable head SHA"
  PASS=$((PASS + 1))
else
  echo "FAIL: restore did not fetch/push/verify the immutable head SHA"
  FAIL=$((FAIL + 1))
fi
if grep -q -- '--force' "${MOCK_DIR}/git_calls"; then
  echo "FAIL: restore used a force push"
  FAIL=$((FAIL + 1))
else
  echo "PASS: restore used a plain (non-force) push"
  PASS=$((PASS + 1))
fi
FETCH_LN="$(grep -n '^fetch origin headsha$' "${MOCK_DIR}/git_calls" | head -n1 | cut -d: -f1 || true)"
PUSH_LN="$(grep -n '^push origin headsha:refs/heads/feature$' "${MOCK_DIR}/git_calls" | head -n1 | cut -d: -f1 || true)"
VERIFY_LN="$(grep -n '^ls-remote --heads origin refs/heads/feature$' "${MOCK_DIR}/git_calls" | head -n1 | cut -d: -f1 || true)"
if [[ -n "$FETCH_LN" && -n "$PUSH_LN" && -n "$VERIFY_LN" \
  && "$FETCH_LN" -lt "$PUSH_LN" && "$PUSH_LN" -lt "$VERIFY_LN" ]]; then
  echo "PASS: restore order is fetch, push, then verification"
  PASS=$((PASS + 1))
else
  echo "FAIL: restore order was not fetch, push, then verification"
  FAIL=$((FAIL + 1))
fi
if grep -q '/pulls/1/update-branch' "${MOCK_DIR}/gh_calls"; then
  echo "PASS: update path ran after the restore"
  PASS=$((PASS + 1))
else
  echo "FAIL: update path did not run after the restore"
  FAIL=$((FAIL + 1))
fi

# 9b. A restore failure (unreachable immutable head SHA) fails closed with a
#     machine-readable error before any update path runs.
rm -f "${MOCK_DIR}/ref_present" "${MOCK_DIR}/fetched"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/gh_calls"
RC=0
OUT="$(MOCK_FETCH_FAIL=1 MOCK_UPDATE_BRANCH_SUCCESS=1 PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha 2>/dev/null)" || RC=$?
assert_fail "unreachable immutable head SHA fails closed" "$RC"
assert_json "restore failure result is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "source_ref_restore_failed" and d["method"] == "source_ref_restore" and "could not fetch immutable PR head SHA" in (d["error"] or "")' <<<"$OUT"; then
  echo "PASS: unreachable SHA reports source_ref_restore_failed"
  PASS=$((PASS + 1))
else
  echo "FAIL: unreachable SHA did not report source_ref_restore_failed"
  FAIL=$((FAIL + 1))
fi
if grep -q '^push ' "${MOCK_DIR}/git_calls" || grep -q '/pulls/1/update-branch' "${MOCK_DIR}/gh_calls"; then
  echo "FAIL: failed restore still attempted an update path"
  FAIL=$((FAIL + 1))
else
  echo "PASS: failed restore attempted no push and no update-branch"
  PASS=$((PASS + 1))
fi

# 9c. Dry-run performs no restore even when the head ref is deleted.
rm -f "${MOCK_DIR}/ref_present" "${MOCK_DIR}/fetched"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/gh_calls"
RC=0
OUT="$(MOCK_UPDATE_BRANCH_SUCCESS=1 PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha --dry-run 2>/dev/null)" || RC=$?
assert_ok "dry-run with deleted head ref exits 0" "$RC"
assert_json "dry-run restore result is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "dry_run" and d["updated"] is False and d["source_ref_restored"] is False' <<<"$OUT"; then
  echo "PASS: dry-run reports a non-mutating plan without restore"
  PASS=$((PASS + 1))
else
  echo "FAIL: dry-run did not report a non-mutating plan"
  FAIL=$((FAIL + 1))
fi
if [[ -s "${MOCK_DIR}/git_calls" ]] || [[ -f "${MOCK_DIR}/ref_present" ]]; then
  echo "FAIL: dry-run performed a restore or git operation"
  FAIL=$((FAIL + 1))
else
  echo "PASS: dry-run performed no restore and no git operation"
  PASS=$((PASS + 1))
fi

# 9d. A cross-fork PR with a deleted head branch fails closed instead of
#     attempting a restore through the base repository remote.
rm -f "${MOCK_DIR}/ref_present_f3" "${MOCK_DIR}/fetched"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/gh_calls"
RC=0
OUT="$(PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 3 --repo owner/repo \
  --expected-head-sha forksha 2>/dev/null)" || RC=$?
assert_fail "cross-fork PR with deleted head ref fails closed" "$RC"
assert_json "cross-fork result is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "source_ref_restore_failed" and d["method"] == "source_ref_restore" and "cross-fork" in (d["error"] or "")' <<<"$OUT"; then
  echo "PASS: cross-fork PR reports a machine-readable restore refusal"
  PASS=$((PASS + 1))
else
  echo "FAIL: cross-fork PR did not report a machine-readable restore refusal"
  FAIL=$((FAIL + 1))
fi
if grep -Eq '^(fetch|push) ' "${MOCK_DIR}/git_calls"; then
  echo "FAIL: cross-fork PR attempted a fetch or push"
  FAIL=$((FAIL + 1))
else
  echo "PASS: cross-fork PR attempted no fetch or push"
  PASS=$((PASS + 1))
fi

# 9e. A concurrently reappeared ref is re-detected after a rejected restore
#     push; the normal update path continues without overwriting the ref.
rm -f "${MOCK_DIR}/ref_present" "${MOCK_DIR}/fetched"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/gh_calls"
RC=0
OUT="$(MOCK_RESTORE_PUSH_REJECT=1 MOCK_RESTORE_REAPPEAR=1 MOCK_UPDATE_BRANCH_SUCCESS=1 \
  PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha 2>/dev/null)" || RC=$?
assert_ok "reappeared ref continues with the normal update path" "$RC"
assert_json "reappeared ref result is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "update_requested" and d["updated"] is True and d["source_ref_restored"] is False' <<<"$OUT"; then
  echo "PASS: reappeared ref is not claimed as a restore"
  PASS=$((PASS + 1))
else
  echo "FAIL: reappeared ref handling was incorrect"
  FAIL=$((FAIL + 1))
fi
if grep -q '^push origin headsha:refs/heads/feature$' "${MOCK_DIR}/git_calls" \
  && grep -q '/pulls/1/update-branch' "${MOCK_DIR}/gh_calls"; then
  echo "PASS: rejected restore push re-detected the ref and ran the update path"
  PASS=$((PASS + 1))
else
  echo "FAIL: rejected restore push did not continue with the update path"
  FAIL=$((FAIL + 1))
fi

# 9f. A rejected restore push with the ref still missing fails closed.
rm -f "${MOCK_DIR}/ref_present" "${MOCK_DIR}/fetched"
: > "${MOCK_DIR}/git_calls"
: > "${MOCK_DIR}/gh_calls"
RC=0
OUT="$(MOCK_RESTORE_PUSH_REJECT=1 MOCK_UPDATE_BRANCH_SUCCESS=1 \
  PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" --pr 1 --repo owner/repo \
  --expected-head-sha headsha 2>/dev/null)" || RC=$?
assert_fail "rejected restore push with missing ref fails closed" "$RC"
assert_json "rejected restore result is valid JSON" "$OUT"
if python3 -c 'import json, sys; d=json.load(sys.stdin); assert d["status"] == "source_ref_restore_failed" and "still missing" in (d["error"] or "")' <<<"$OUT"; then
  echo "PASS: rejected restore push reports a machine-readable error"
  PASS=$((PASS + 1))
else
  echo "FAIL: rejected restore push did not report a machine-readable error"
  FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] || exit 1
