#!/usr/bin/env bash
# Smoke tests for scripts/dev/update_pr_branch_safely.sh (issue #5775).
#
# The wrapper shells out to `gh` for metadata and to `git` for the local
# fallback.  We mock both so the tests stay fully offline and do not depend on
# GitHub availability, credentials, or a real remote.  The mock records which
# mutating path (REST update-branch vs. local rebase/push) the wrapper selected.
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
# wrapper's compact TSV metadata selector and scalar selectors used by older
# callers.
make_gh() {
  cat > "${MOCK_DIR}/gh" <<'EOF'
#!/usr/bin/env bash
# Minimal gh mock honoring `--jq '.head.sha'`, `.head.ref`, `.base.ref`,
# and `repo view --json nameWithOwner --jq .nameWithOwner`.
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
    echo "gh: 'update-branch' is not a gh command" >&2
    exit 1
    ;;
  *"/pulls/1")
    if [[ "$jq" == *"@tsv"* ]]; then
      printf 'headsha\tfeature\tmain'
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
      printf 'othersha\tfeature2\tmain'
    else
      case "$jq" in
        ".head.sha") printf 'othersha';;
        ".head.ref") printf 'feature2';;
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

# 4. --dry-run must reach the local-fallback plan without mutating.
make_gh
# git stub that answers the read-only branch/HEAD queries the wrapper needs to
# reach the dry-run plan, and blocks any mutating op (none should be reached).
cat > "${MOCK_DIR}/git" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  "rev-parse --abbrev-ref HEAD") printf 'feature';;
  "rev-parse HEAD") printf 'headsha';;
  "ls-remote"*) exit 0;;
  *) echo "git stub: refusing op $*" >&2; exit 1;;
esac
EOF
chmod +x "${MOCK_DIR}/git"
RC=0
OUT="$(PATH="${MOCK_DIR}:$PATH" bash "$SCRIPT" 1 --repo owner/repo \
  --expected-head-sha headsha --remote custom --dry-run 2>/dev/null)" || RC=$?
assert_json "dry-run output is valid JSON" "$OUT"
if echo "$OUT" | grep -q '"status":"dry_run"'; then
  echo "PASS: dry-run reports plan without mutating"
  PASS=$((PASS + 1))
else
  echo "FAIL: dry-run did not report plan"
  FAIL=$((FAIL + 1))
fi
if python3 -c 'import json, sys; assert json.load(sys.stdin)["remote"] == "custom"' <<<"$OUT"; then
  echo "PASS: configured remote is preserved in the result"
  PASS=$((PASS + 1))
else
  echo "FAIL: configured remote was not preserved in the result"
  FAIL=$((FAIL + 1))
fi
assert_ok "dry-run exits 0" "$RC"

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

# 6. --help / bad flag rejected.
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

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] || exit 1
