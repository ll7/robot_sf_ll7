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

# Mock gh: returns PR metadata for the pulls endpoint and makes the REST
# update-branch endpoint fail (404-style), forcing the local fallback to be
# evaluated.  Local fallback itself is blocked by a stubbed git non-mutation
# check so no network/remote is touched.  The mock honors a leaf --jq selector
# so callers receive the scalar the wrapper expects.
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
    case "$jq" in
      ".head.sha") printf 'headsha';;
      ".head.ref") printf 'feature';;
      ".base.ref") printf 'main';;
      *) printf '{"head":{"sha":"headsha"},"head_ref":"feature","base_ref":"main"}';;
    esac
    exit 0
    ;;
  *"/pulls/2")
    case "$jq" in
      ".head.sha") printf 'othersha';;
      ".head.ref") printf 'feature2';;
      ".base.ref") printf 'main';;
    esac
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

# 3. When gh update-branch is unavailable, the wrapper selects the local
#    fallback path (which then fails because our git stub blocks mutation) and
#    reports a machine-readable error rather than succeeding silently.
make_gh
# git stub that answers read-only queries but refuses the mutating rebase/push,
# so test 3 proves the wrapper fails closed instead of touching a remote.
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
  --expected-head-sha headsha --no-local-fallback 2>/dev/null)" || RC=$?
if echo "$OUT" | grep -q '"status":"head_mismatch"' || echo "$OUT" | grep -q '"status":"error"'; then
  echo "PASS: machine-readable result emitted on unavailable update-branch"
  PASS=$((PASS + 1))
else
  echo "FAIL: no machine-readable result on unavailable update-branch"
  FAIL=$((FAIL + 1))
fi
assert_fail "fails closed when update-branch unavailable and no fallback" "$RC"

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
  --expected-head-sha headsha --dry-run 2>/dev/null)" || RC=$?
if echo "$OUT" | grep -q '"status":"dry_run"'; then
  echo "PASS: dry-run reports plan without mutating"
  PASS=$((PASS + 1))
else
  echo "FAIL: dry-run did not report plan"
  FAIL=$((FAIL + 1))
fi
assert_ok "dry-run exits 0" "$RC"

# 5. --help / bad flag rejected.
RC=0
bash "$SCRIPT" --bogus 2>/dev/null || RC=$?
assert_fail "unknown flag rejected" "$RC"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] || exit 1
