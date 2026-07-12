#!/usr/bin/env bash
# Smoke tests for scripts/dev/gh_issue_view.sh (issue #5188).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PASS=0
FAIL=0
MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$MOCK_DIR"' EXIT
MOCK_ARGS_FILE="${MOCK_DIR}/args"
export MOCK_ARGS_FILE

# Keep this test offline: the wrapper's Python helper invokes ``gh`` to read an
# issue, so a local successful native response exercises the full delegation
# path without depending on GitHub availability or credentials.
cat > "${MOCK_DIR}/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${MOCK_ARGS_FILE}"
printf 'title:\trestore live issue reads\nstate:\tOPEN\nurl:\thttps://example.test/issues/5188\n--\nmock body\n'
EOF
chmod +x "${MOCK_DIR}/gh"

assert_ok() {
  local desc="$1" output="$2"
  if [[ -n "$output" ]]; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc (empty output)"
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

# 1. Basic usage prints issue content
OUT=$(PATH="${MOCK_DIR}:$PATH" bash "${REPO_ROOT}/scripts/dev/gh_issue_view.sh" 5188 \
  --repo 'll7/robot sf' 2>/dev/null)
assert_ok "reads issue body" "$OUT"

if grep -Fxq 'll7/robot sf' "${MOCK_ARGS_FILE}"; then
  echo "PASS: repo argument remains one value"
  PASS=$((PASS + 1))
else
  echo "FAIL: repo argument was split or missing"
  FAIL=$((FAIL + 1))
fi

# 2. Output contains expected title fragment
if echo "$OUT" | grep -q "restore live issue reads"; then
  echo "PASS: output contains issue title"
  PASS=$((PASS + 1))
else
  echo "FAIL: output missing issue title"
  FAIL=$((FAIL + 1))
fi

# 3. Unknown flag should fail with nonzero exit
RC=0
bash "${REPO_ROOT}/scripts/dev/gh_issue_view.sh" 5188 --bad-flag 2>/dev/null || RC=$?
assert_fail "unknown flag rejected" "$RC"

# 4. Missing number should fail
RC=0
bash "${REPO_ROOT}/scripts/dev/gh_issue_view.sh" 2>/dev/null || RC=$?
assert_fail "missing number rejected" "$RC"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] || exit 1
